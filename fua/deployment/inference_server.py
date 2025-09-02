"""
FUA 推理服务器

基于 FastAPI 的高性能模型推理服务器
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any
import uvicorn
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """推理请求模型"""
    model_name: str
    input_data: List[List[float]]  # 简化的输入格式
    batch_size: int = 1


class InferenceResponse(BaseModel):
    """推理响应模型"""
    predictions: List[float]
    confidence: float
    processing_time: float


class FUAInferenceServer:
    """FUA 推理服务器"""
    
    def __init__(self):
        self.app = FastAPI(title="FUA Inference Server", version="1.0.0")
        self.models = {}  # model_name -> ort_session
        self.model_metadata = {}  # model_name -> metadata
        
        # 注册路由
        self._register_routes()
    
    def _register_routes(self):
        """注册 API 路由"""
        
        @self.app.get("/")
        async def root():
            return {"message": "FUA Inference Server", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "models_loaded": len(self.models)}
        
        @self.app.post("/predict", response_model=InferenceResponse)
        async def predict(request: InferenceRequest):
            """执行推理"""
            import time
            start_time = time.time()
            
            try:
                # 检查模型是否已加载
                if request.model_name not in self.models:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Model '{request.model_name}' not found"
                    )
                
                # 准备输入数据
                input_array = np.array(request.input_data, dtype=np.float32)
                
                # 执行推理
                session = self.models[request.model_name]
                outputs = session.run(None, {'input': input_array})
                
                # 处理输出
                predictions = outputs[0].flatten().tolist()
                confidence = float(np.max(predictions))
                
                processing_time = time.time() - start_time
                
                return InferenceResponse(
                    predictions=predictions,
                    confidence=confidence,
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/load_model")
        async def load_model(model_name: str, model_path: str):
            """加载模型"""
            try:
                if not Path(model_path).exists():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model file not found: {model_path}"
                    )
                
                # 创建 ONNX Runtime 会话
                session = ort.InferenceSession(model_path)
                self.models[model_name] = session
                
                # 获取模型信息
                input_info = session.get_inputs()[0]
                output_info = session.get_outputs()[0]
                
                self.model_metadata[model_name] = {
                    'input_shape': input_info.shape,
                    'input_type': input_info.type,
                    'output_shape': output_info.shape,
                    'output_type': output_info.type
                }
                
                logger.info(f"Model loaded: {model_name}")
                return {"message": f"Model '{model_name}' loaded successfully"}
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def list_models():
            """列出已加载的模型"""
            return {
                "models": list(self.models.keys()),
                "metadata": self.model_metadata
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行服务器"""
        logger.info(f"Starting inference server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# 工厂函数
def create_inference_server() -> FUAInferenceServer:
    """创建推理服务器实例"""
    return FUAInferenceServer()


if __name__ == "__main__":
    server = create_inference_server()
    server.run()
