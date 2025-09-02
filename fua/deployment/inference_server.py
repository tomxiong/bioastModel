"""
FUA 推理服务器

基于 FastAPI 的高性能模型推理服务器
支持批处理、模型热加载、性能监控等高级功能
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Union, Tuple
import uvicorn
import logging
import time
import asyncio
import threading
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict, deque
import statistics

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 请求和响应模型
class InferenceRequest(BaseModel):
    """推理请求模型"""
    model_name: str = Field(..., description="模型名称")
    input_data: Union[List[List[float]], List[float]] = Field(..., description="输入数据")
    batch_size: Optional[int] = Field(1, description="批次大小")
    threshold: Optional[float] = Field(0.5, description="分类阈值")


class BatchInferenceRequest(BaseModel):
    """批量推理请求"""
    model_name: str = Field(..., description="模型名称")
    inputs: List[List[float]] = Field(..., description="多个输入数据")
    threshold: Optional[float] = Field(0.5, description="分类阈值")


class InferenceResponse(BaseModel):
    """推理响应模型"""
    predictions: List[float]
    confidence: float
    processing_time: float
    model_version: Optional[str] = None
    request_id: Optional[str] = None


class BatchInferenceResponse(BaseModel):
    """批量推理响应"""
    results: List[Dict[str, Any]]
    total_time: float
    average_time: float
    throughput: float


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    input_shape: List[int]
    output_shape: List[int]
    input_type: str
    output_type: str
    providers: List[str]
    load_time: Optional[str] = None
    file_size_mb: Optional[float] = None
    inference_count: int = 0


class LoadModelRequest(BaseModel):
    """加载模型请求"""
    model_name: str
    model_path: str
    provider: Optional[str] = None
    opt_level: Optional[int] = None


class PerformanceMetrics(BaseModel):
    """性能指标"""
    total_requests: int
    average_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    error_rate: float
    model_stats: Dict[str, Dict[str, Any]]


class FUAInferenceServer:
    """FUA 推理服务器"""
    
    def __init__(self, max_models: int = 10, metrics_window: int = 1000):
        self.app = FastAPI(
            title="FUA Inference Server",
            version="1.0.0",
            description="高性能模型推理服务器，支持批处理、热加载和监控"
        )
        
        # 模型存储
        self.models = {}  # model_name -> ort_session
        self.model_info = {}  # model_name -> ModelInfo
        self.max_models = max_models
        
        # 性能监控
        self.metrics_window = metrics_window
        self.request_times = deque(maxlen=metrics_window)
        self.error_count = 0
        self.model_stats = defaultdict(lambda: {
            'request_count': 0,
            'total_time': 0,
            'error_count': 0
        })
        
        # 配置 CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 注册路由
        self._register_routes()
        
        # 启动监控线程
        self._start_monitoring()
    
    def _register_routes(self):
        """注册 API 路由"""
        
        @self.app.get("/")
        async def root():
            """服务器信息"""
            return {
                "service": "FUA Inference Server",
                "version": "1.0.0",
                "status": "running",
                "models_loaded": len(self.models),
                "uptime": self._get_uptime()
            }
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "models": len(self.models),
                "memory_usage": self._get_memory_usage(),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/predict", response_model=InferenceResponse)
        async def predict(request: InferenceRequest):
            """单个推理请求"""
            start_time = time.time()
            request_id = f"req_{int(time.time() * 1000000)}"
            
            try:
                # 验证模型
                if request.model_name not in self.models:
                    raise HTTPException(404, f"Model '{request.model_name}' not found")
                
                # 准备输入数据
                input_array = self._prepare_input(request.input_data, request.model_name)
                
                # 执行推理
                outputs = await self._run_inference(request.model_name, input_array)
                
                # 处理输出
                predictions, confidence = self._process_outputs(outputs, request.threshold)
                
                processing_time = time.time() - start_time
                
                # 更新统计
                self._update_metrics(request.model_name, processing_time, False)
                
                return InferenceResponse(
                    predictions=predictions,
                    confidence=confidence,
                    processing_time=processing_time,
                    model_version=self.model_info[request.model_name].name,
                    request_id=request_id
                )
                
            except Exception as e:
                self._update_metrics(request.model_name, time.time() - start_time, True)
                logger.error(f"Prediction error [{request_id}]: {e}")
                raise HTTPException(500, detail=str(e))
        
        @self.app.post("/predict/batch", response_model=BatchInferenceResponse)
        async def predict_batch(request: BatchInferenceRequest):
            """批量推理"""
            start_time = time.time()
            
            try:
                if request.model_name not in self.models:
                    raise HTTPException(404, f"Model '{request.model_name}' not found")
                
                # 批量推理
                results = []
                for i, input_data in enumerate(request.inputs):
                    try:
                        input_array = self._prepare_input(input_data, request.model_name)
                        outputs = await self._run_inference(request.model_name, input_array)
                        predictions, confidence = self._process_outputs(outputs, request.threshold)
                        
                        results.append({
                            "index": i,
                            "predictions": predictions,
                            "confidence": confidence,
                            "success": True
                        })
                    except Exception as e:
                        results.append({
                            "index": i,
                            "error": str(e),
                            "success": False
                        })
                
                total_time = time.time() - start_time
                average_time = total_time / len(request.inputs)
                throughput = len(request.inputs) / total_time
                
                # 更新统计
                for _ in request.inputs:
                    self._update_metrics(request.model_name, average_time, False)
                
                return BatchInferenceResponse(
                    results=results,
                    total_time=total_time,
                    average_time=average_time,
                    throughput=throughput
                )
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(500, detail=str(e))
        
        @self.app.post("/load_model")
        async def load_model(request: LoadModelRequest):
            """加载模型"""
            try:
                # 检查模型数量限制
                if len(self.models) >= self.max_models:
                    # 移除最少使用的模型
                    self._unload_lru_model()
                
                model_path = Path(request.model_path)
                if not model_path.exists():
                    raise HTTPException(404, f"Model file not found: {request.model_path}")
                
                # 创建 ONNX Runtime 会话
                session_options = ort.SessionOptions()
                
                if request.provider:
                    session_options.intra_op_num_threads = 1
                    if request.provider == 'cuda':
                        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                
                if request.opt_level is not None:
                    session_options.graph_optimization_level = request.opt_level
                
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options,
                    providers=['CPUExecutionProvider']  # 可以根据需要添加 CUDA
                )
                
                # 获取模型信息
                input_info = session.get_inputs()[0]
                output_info = session.get_outputs()[0]
                
                # 存储模型和信息
                self.models[request.model_name] = session
                self.model_info[request.model_name] = ModelInfo(
                    name=request.model_name,
                    input_shape=list(input_info.shape),
                    output_shape=list(output_info.shape),
                    input_type=input_info.type,
                    output_type=output_info.type,
                    providers=session.get_providers(),
                    load_time=datetime.now().isoformat(),
                    file_size_mb=model_path.stat().st_size / (1024 * 1024)
                )
                
                logger.info(f"Model loaded: {request.model_name}")
                return {"message": f"Model '{request.model_name}' loaded successfully"}
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise HTTPException(500, detail=str(e))
        
        @self.app.delete("/models/{model_name}")
        async def unload_model(model_name: str):
            """卸载模型"""
            if model_name in self.models:
                del self.models[model_name]
                del self.model_info[model_name]
                logger.info(f"Model unloaded: {model_name}")
                return {"message": f"Model '{model_name}' unloaded successfully"}
            else:
                raise HTTPException(404, f"Model '{model_name}' not found")
        
        @self.app.get("/models")
        async def list_models():
            """列出所有模型"""
            return {
                "models": [info.dict() for info in self.model_info.values()],
                "count": len(self.models),
                "max_capacity": self.max_models
            }
        
        @self.app.get("/models/{model_name}")
        async def get_model_info(model_name: str):
            """获取特定模型信息"""
            if model_name not in self.model_info:
                raise HTTPException(404, f"Model '{model_name}' not found")
            
            info = self.model_info[model_name].dict()
            info.update(self.model_stats[model_name])
            return info
        
        @self.app.post("/models/{model_name}/warmup")
        async def warmup_model(model_name: str, iterations: int = 10):
            """模型预热"""
            if model_name not in self.models:
                raise HTTPException(404, f"Model '{model_name}' not found")
            
            session = self.models[model_name]
            input_shape = self.model_info[model_name].input_shape
            
            # 创建随机输入
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # 预热运行
            for _ in range(iterations):
                session.run(None, {'input': dummy_input})
            
            return {"message": f"Model '{model_name}' warmed up with {iterations} iterations"}
        
        @self.app.get("/metrics")
        async def get_metrics():
            """获取性能指标"""
            if not self.request_times:
                return PerformanceMetrics(
                    total_requests=0,
                    average_latency=0,
                    p95_latency=0,
                    p99_latency=0,
                    throughput=0,
                    error_rate=0,
                    model_stats=dict(self.model_stats)
                )
            
            # 计算延迟百分位
            sorted_times = sorted(self.request_times)
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
            
            # 计算吞吐量（最近60秒）
            recent_requests = [t for t in self.request_times if time.time() - t < 60]
            throughput = len(recent_requests) / 60 if recent_requests else 0
            
            return PerformanceMetrics(
                total_requests=len(self.request_times),
                average_latency=statistics.mean(self.request_times),
                p95_latency=p95,
                p99_latency=p99,
                throughput=throughput,
                error_rate=self.error_count / len(self.request_times) if self.request_times else 0,
                model_stats=dict(self.model_stats)
            )
        
        @self.app.post("/upload_model")
        async def upload_model(
            model_name: str,
            file: UploadFile = File(...),
            provider: Optional[str] = None
        ):
            """上传并加载模型"""
            try:
                # 保存上传的文件
                model_path = Path(f"./models/{model_name}.onnx")
                model_path.parent.mkdir(exist_ok=True)
                
                with open(model_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # 加载模型
                load_request = LoadModelRequest(
                    model_name=model_name,
                    model_path=str(model_path),
                    provider=provider
                )
                
                return await load_model(load_request)
                
            except Exception as e:
                logger.error(f"Failed to upload model: {e}")
                raise HTTPException(500, detail=str(e))
    
    def _prepare_input(self, input_data: Union[List[List[float]], List[float]], model_name: str) -> np.ndarray:
        """准备输入数据"""
        model_info = self.model_info[model_name]
        
        # 转换为 numpy 数组
        if isinstance(input_data[0], (int, float)):
            # 单个样本
            input_array = np.array(input_data, dtype=np.float32)
        else:
            # 批次
            input_array = np.array(input_data, dtype=np.float32)
        
        # 调整形状
        expected_shape = model_info.input_shape
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(expected_shape)
        elif len(input_array.shape) == 2:
            input_array = input_array.reshape((-1, *expected_shape[1:]))
        
        return input_array
    
    async def _run_inference(self, model_name: str, input_array: np.ndarray) -> List[np.ndarray]:
        """执行推理"""
        session = self.models[model_name]
        
        # 在线程池中运行以避免阻塞事件循环
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: session.run(None, {'input': input_array})
        )
        
        return outputs
    
    def _process_outputs(self, outputs: List[np.ndarray], threshold: float) -> Tuple[List[float], float]:
        """处理模型输出"""
        # 假设是二元分类
        logits = outputs[0]
        if len(logits.shape) > 1:
            logits = logits.flatten()
        
        # 应用 softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
        # 获取预测和置信度
        predictions = probabilities.tolist()
        confidence = float(np.max(probabilities))
        
        return predictions, confidence
    
    def _update_metrics(self, model_name: str, processing_time: float, is_error: bool):
        """更新性能指标"""
        self.request_times.append(processing_time)
        
        if is_error:
            self.error_count += 1
            self.model_stats[model_name]['error_count'] += 1
        
        self.model_stats[model_name]['request_count'] += 1
        self.model_stats[model_name]['total_time'] += processing_time
    
    def _unload_lru_model(self):
        """卸载最近最少使用的模型"""
        if not self.models:
            return
        
        # 找到最少使用的模型
        lru_model = min(
            self.model_stats.items(),
            key=lambda x: x[1]['request_count']
        )[0]
        
        del self.models[lru_model]
        del self.model_info[lru_model]
        del self.model_stats[lru_model]
        
        logger.info(f"Unloaded LRU model: {lru_model}")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024
            }
        except ImportError:
            return {}
    
    def _get_uptime(self) -> str:
        """获取运行时间"""
        if hasattr(self, 'start_time'):
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            return f"{hours}h {minutes}m {seconds}s"
        return "N/A"
    
    def _start_monitoring(self):
        """启动监控线程"""
        self.start_time = time.time()
        
        def monitor():
            while True:
                time.sleep(60)  # 每分钟记录一次
                if self.models:
                    logger.info(f"Server stats - Models: {len(self.models)}, "
                              f"Requests: {len(self.request_times)}, "
                              f"Errors: {self.error_count}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def run(self, 
            host: str = "0.0.0.0", 
            port: int = 8000,
            workers: int = 1,
            log_level: str = "info"):
        """运行服务器"""
        logger.info(f"Starting FUA Inference Server on {host}:{port}")
        logger.info(f"Max models: {self.max_models}")
        logger.info(f"Workers: {workers}")
        
        # 配置
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level=log_level,
            access_log=True
        )
        
        # 运行服务器
        server = uvicorn.Server(config)
        server.run()


# 工厂函数
def create_inference_server(max_models: int = 10, metrics_window: int = 1000) -> FUAInferenceServer:
    """创建推理服务器实例"""
    return FUAInferenceServer(max_models, metrics_window)


if __name__ == "__main__":
    # 创建并运行服务器
    server = create_inference_server()
    server.run()
