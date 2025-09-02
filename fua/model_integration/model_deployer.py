"""
模型部署器

提供模型部署功能，包括格式转换、优化、容器化和
多种部署场景支持（边缘设备、云端、嵌入式等）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import json
import logging
import shutil
import subprocess
import tempfile
import platform
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import onnx
import onnxruntime as ort
from torch.quantization import quantize_dynamic, prepare_qat, convert
try:
    import tensorrt as trt
except ImportError:
    trt = None
try:
    import docker
except ImportError:
    docker = None
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from .model_integrator import ModelIntegrator, ModelMetadata, ModelFormat
from .model_evaluator import EvaluationResult

logger = logging.getLogger(__name__)


class DeploymentPlatform(Enum):
    """部署平台枚举"""
    LOCAL = "local"
    CLOUD = "cloud"
    EDGE = "edge"
    EMBEDDED = "embedded"
    MOBILE = "mobile"
    BROWSER = "browser"
    SERVERLESS = "serverless"


class DeploymentFormat(Enum):
    """部署格式枚举"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TFLITE = "tflite"
    COREML = "coreml"
    OPENVINO = "openvino"
    TVM = "tvm"
    JIT = "jit"


class OptimizationLevel(Enum):
    """优化级别枚举"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    AGGRESSIVE = "aggressive"


class DeploymentStatus(Enum):
    """部署状态枚举"""
    PENDING = "pending"
    PREPARING = "preparing"
    CONVERTING = "converting"
    OPTIMIZING = "optimizing"
    PACKAGING = "packaging"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETRACTED = "retracted"


@dataclass
class DeploymentConfig:
    """部署配置"""
    platform: DeploymentPlatform
    format: DeploymentFormat
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    target_device: str = "cpu"  # cpu, gpu, tpu, etc.
    quantization: bool = False
    pruning: bool = False
    batch_size: int = 1
    input_shape: Tuple[int, ...] = None
    output_names: List[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['platform'] = self.platform.value
        data['format'] = self.format.value
        data['optimization_level'] = self.optimization_level.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentConfig':
        """从字典创建"""
        data['platform'] = DeploymentPlatform(data['platform'])
        data['format'] = DeploymentFormat(data['format'])
        data['optimization_level'] = OptimizationLevel(data['optimization_level'])
        return cls(**data)


@dataclass
class DeploymentMetrics:
    """部署指标"""
    model_size_mb: float = 0.0
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    throughput_qps: float = 0.0
    error_rate: float = 0.0
    availability_percent: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentMetrics':
        """从字典创建"""
        return cls(**data)


@dataclass
class DeploymentResult:
    """部署结果"""
    deployment_id: str
    model_id: str
    version_id: str
    config: DeploymentConfig
    status: DeploymentStatus
    output_path: str
    deployment_url: Optional[str] = None
    metrics: DeploymentMetrics = field(default_factory=DeploymentMetrics)
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['config'] = self.config.to_dict()
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeploymentResult':
        """从字典创建"""
        data['config'] = DeploymentConfig.from_dict(data['config'])
        data['status'] = DeploymentStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class ModelConverter:
    """模型转换器"""
    
    @staticmethod
    def convert_to_onnx(model: nn.Module,
                       input_shape: Tuple[int, ...],
                       output_path: str,
                       output_names: List[str] = None) -> bool:
        """转换为ONNX格式"""
        try:
            model.eval()
            dummy_input = torch.randn(input_shape)
            
            if output_names is None:
                output_names = ['output']
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=output_names,
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
            
            # 验证ONNX模型
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"Model converted to ONNX: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert to ONNX: {e}")
            return False
    
    @staticmethod
    def convert_to_tensorrt(onnx_path: str,
                           output_path: str,
                           max_batch_size: int = 1,
                           max_workspace_size: int = 1 << 30) -> bool:
        """转换为TensorRT格式"""
        try:
            import tensorrt as trt
            
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network()
            parser = trt.OnnxParser(network, logger)
            
            # 解析ONNX模型
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False
            
            # 构建配置
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            # 创建优化配置
            profile = builder.create_optimization_profile()
            profile.set_shape("input", (1, *input_shape[1:]), 
                            (max_batch_size, *input_shape[1:]))
            config.add_optimization_profile(profile)
            
            # 构建引擎
            engine = builder.build_engine(network, config)
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # 保存引擎
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"Model converted to TensorRT: {output_path}")
            return True
            
        except ImportError:
            logger.error("TensorRT not available")
            return False
        except Exception as e:
            logger.error(f"Failed to convert to TensorRT: {e}")
            return False
    
    @staticmethod
    def quantize_model(model: nn.Module,
                      calibration_data: torch.Tensor = None) -> nn.Module:
        """量化模型"""
        try:
            # 动态量化
            model_quantized = quantize_dynamic(
                model,
                {nn.Conv2d, nn.Linear},
                dtype=torch.qint8
            )
            
            logger.info("Model quantized successfully")
            return model_quantized
            
        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            return model


class DockerPackager:
    """Docker打包器"""
    
    @staticmethod
    def create_dockerfile(model_path: str,
                         deployment_format: DeploymentFormat,
                         requirements: List[str] = None,
                         base_image: str = "python:3.8-slim") -> str:
        """创建Dockerfile"""
        dockerfile = f"""FROM {base_image}

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir pip && \\
    pip install --no-cache-dir torch torchvision
"""
        
        if requirements:
            dockerfile += f"RUN pip install --no-cache-dir {' '.join(requirements)}\n"
        
        if deployment_format == DeploymentFormat.ONNX:
            dockerfile += "RUN pip install --no-cache-dir onnxruntime\n"
        elif deployment_format == DeploymentFormat.TENSORRT:
            dockerfile += "RUN pip install --no-cache-dir tensorrt\n"
        
        dockerfile += f"""
# Copy model
COPY {Path(model_path).name} /app/model.{deployment_format.value}

# Copy inference script
COPY inference.py /app/inference.py

EXPOSE 8080

CMD ["python", "inference.py"]
"""
        
        return dockerfile
    
    @staticmethod
    def build_image(dockerfile_path: str,
                   context_path: str,
                   image_name: str,
                   image_tag: str = "latest") -> bool:
        """构建Docker镜像"""
        try:
            client = docker.from_env()
            
            # 构建镜像
            image, logs = client.images.build(
                path=context_path,
                dockerfile=dockerfile_path,
                tag=f"{image_name}:{image_tag}"
            )
            
            logger.info(f"Docker image built: {image_name}:{image_tag}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build Docker image: {e}")
            return False
    
    @staticmethod
    def create_inference_script(model_path: str,
                              deployment_format: DeploymentFormat,
                              input_shape: Tuple[int, ...]) -> str:
        """创建推理脚本"""
        script = f"""import torch
import numpy as np
from flask import Flask, request, jsonify
import os
from pathlib import Path

app = Flask(__name__)

# Load model
model_path = f"/app/model.{deployment_format.value}"
input_shape = {input_shape}

"""
        
        if deployment_format == DeploymentFormat.PYTORCH:
            script += """
model = torch.load(model_path, map_location='cpu')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input'])
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.numpy().tolist()
        
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
"""
        elif deployment_format == DeploymentFormat.ONNX:
            script += """
import onnxruntime as ort

sess = ort.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data['input'])
        
        outputs = sess.run(None, {input_name: input_data})
        prediction = outputs[0].tolist()
        
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
"""
        
        script += """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
"""
        
        return script


class CloudDeployer:
    """云端部署器"""
    
    def __init__(self, provider: str = "aws"):
        """
        初始化云端部署器
        
        Args:
            provider: 云服务提供商 (aws, gcp, azure)
        """
        self.provider = provider
        
        # 这里应该初始化相应的SDK
        # 例如：boto3 for AWS, google.cloud for GCP, azure.identity for Azure
        self.client = None
        
        logger.info(f"CloudDeployer initialized for provider: {provider}")
    
    def deploy_to_lambda(self,
                        deployment_package: str,
                        function_name: str,
                        handler: str = "inference.handler") -> bool:
        """部署到AWS Lambda"""
        try:
            import boto3
            
            lambda_client = boto3.client('lambda')
            
            # 创建或更新函数
            with open(deployment_package, 'rb') as f:
                zipped_code = f.read()
            
            try:
                # 更新现有函数
                response = lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zipped_code
                )
                logger.info(f"Updated Lambda function: {function_name}")
            except lambda_client.exceptions.ResourceNotFoundException:
                # 创建新函数
                response = lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime='python3.8',
                    Role='arn:aws:iam::123456789012:role/lambda-role',
                    Handler=handler,
                    Code={'ZipFile': zipped_code},
                    Timeout=30,
                    MemorySize=512
                )
                logger.info(f"Created Lambda function: {function_name}")
            
            return True
            
        except ImportError:
            logger.error("boto3 not available")
            return False
        except Exception as e:
            logger.error(f"Failed to deploy to Lambda: {e}")
            return False
    
    def deploy_to_sagemaker(self,
                          model_artifact: str,
                          model_name: str,
                          instance_type: str = "ml.m5.large") -> bool:
        """部署到Amazon SageMaker"""
        try:
            import boto3
            
            sagemaker = boto3.client('sagemaker')
            
            # 创建模型
            response = sagemaker.create_model(
                ModelName=model_name,
                ExecutionRoleArn='arn:aws:iam::123456789012:role/SageMakerRole',
                Containers=[{
                    'Image': '763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.5.0-cpu',
                    'ModelDataUrl': f's3://your-bucket/{model_artifact}'
                }]
            )
            
            # 创建端点配置
            endpoint_config_name = f"{model_name}-config"
            sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': 1
                }]
            )
            
            # 创建端点
            endpoint_name = f"{model_name}-endpoint"
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            logger.info(f"Deployed to SageMaker endpoint: {endpoint_name}")
            return True
            
        except ImportError:
            logger.error("boto3 not available")
            return False
        except Exception as e:
            logger.error(f"Failed to deploy to SageMaker: {e}")
            return False


class EdgeDeployer:
    """边缘设备部署器"""
    
    @staticmethod
    def create_tflite_model(model: nn.Module,
                           input_shape: Tuple[int, ...],
                           output_path: str) -> bool:
        """创建TensorFlow Lite模型"""
        try:
            # 转换为TensorFlow模型（需要先转换为TF格式）
            # 这里简化处理，实际需要更复杂的转换流程
            logger.warning("TFLite conversion requires TensorFlow model")
            return False
            
        except Exception as e:
            logger.error(f"Failed to create TFLite model: {e}")
            return False
    
    @staticmethod
    def create_coreml_model(model: nn.Module,
                           input_shape: Tuple[int, ...],
                           output_path: str) -> bool:
        """创建CoreML模型"""
        try:
            import coremltools as ct
            
            # 转换为CoreML模型
            traced_model = torch.jit.trace(model, torch.randn(input_shape))
            
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_shape, name="input")]
            )
            
            mlmodel.save(output_path)
            
            logger.info(f"CoreML model created: {output_path}")
            return True
            
        except ImportError:
            logger.error("coremltools not available")
            return False
        except Exception as e:
            logger.error(f"Failed to create CoreML model: {e}")
            return False


class ModelDeployer:
    """模型部署器主类"""
    
    def __init__(self,
                 output_dir: str = "./deployments",
                 model_integrator: ModelIntegrator = None):
        """
        初始化模型部署器
        
        Args:
            output_dir: 输出目录
            model_integrator: 模型集成器实例
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model_integrator = model_integrator
        
        # 部署历史
        self.deployments: Dict[str, DeploymentResult] = {}
        
        # 组件初始化
        self.converter = ModelConverter()
        self.docker_packager = DockerPackager()
        self.cloud_deployer = CloudDeployer()
        self.edge_deployer = EdgeDeployer()
        
        # 部署监控
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info("ModelDeployer initialized")
    
    def deploy_model(self,
                    model_id: str,
                    version_id: str,
                    config: DeploymentConfig,
                    deployment_id: str = None) -> DeploymentResult:
        """
        部署模型
        
        Args:
            model_id: 模型ID
            version_id: 版本ID
            config: 部署配置
            deployment_id: 部署ID（可选）
            
        Returns:
            部署结果
        """
        if deployment_id is None:
            deployment_id = f"deploy_{model_id}_{version_id}_{int(time.time())}"
        
        # 创建部署目录
        deploy_dir = self.output_dir / deployment_id
        deploy_dir.mkdir(exist_ok=True)
        
        # 初始化部署结果
        result = DeploymentResult(
            deployment_id=deployment_id,
            model_id=model_id,
            version_id=version_id,
            config=config,
            status=DeploymentStatus.PENDING,
            output_path=str(deploy_dir)
        )
        
        try:
            # 1. 准备阶段
            result.status = DeploymentStatus.PREPARING
            result.logs.append("Starting deployment preparation...")
            
            # 加载模型
            if self.model_integrator:
                model = self.model_integrator.load_model(model_id, version_id)
            else:
                # 从文件加载模型
                model_path = self._find_model_path(model_id, version_id)
                if model_path:
                    model = torch.load(model_path, map_location='cpu')
                else:
                    raise FileNotFoundError(f"Model not found: {model_id}:{version_id}")
            
            result.logs.append("Model loaded successfully")
            
            # 2. 转换阶段
            result.status = DeploymentStatus.CONVERTING
            result.logs.append(f"Converting to {config.format.value} format...")
            
            converted_model_path = self._convert_model(
                model, config, deploy_dir
            )
            
            if not converted_model_path:
                raise Exception("Model conversion failed")
            
            result.logs.append(f"Model converted: {converted_model_path}")
            
            # 3. 优化阶段
            if config.optimization_level != OptimizationLevel.NONE:
                result.status = DeploymentStatus.OPTIMIZING
                result.logs.append("Optimizing model...")
                
                optimized_model_path = self._optimize_model(
                    converted_model_path, config
                )
                
                if optimized_model_path:
                    converted_model_path = optimized_model_path
                    result.logs.append("Model optimized")
            
            # 4. 打包阶段
            result.status = DeploymentStatus.PACKAGING
            result.logs.append("Packaging for deployment...")
            
            package_path = self._package_model(
                converted_model_path, config, deploy_dir
            )
            
            if not package_path:
                raise Exception("Model packaging failed")
            
            result.logs.append(f"Model packaged: {package_path}")
            
            # 5. 部署阶段
            result.status = DeploymentStatus.DEPLOYING
            result.logs.append(f"Deploying to {config.platform.value}...")
            
            deployment_url = self._deploy_to_platform(
                package_path, config, deployment_id
            )
            
            if deployment_url:
                result.deployment_url = deployment_url
                result.status = DeploymentStatus.DEPLOYED
                result.logs.append(f"Deployment successful: {deployment_url}")
            else:
                result.status = DeploymentStatus.FAILED
                result.logs.append("Deployment failed")
            
            # 6. 收集指标
            if result.status == DeploymentStatus.DEPLOYED:
                result.metrics = self._collect_metrics(
                    converted_model_path, config
                )
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.logs.append(f"Deployment failed: {str(e)}")
            logger.error(f"Deployment failed for {deployment_id}: {e}")
        
        result.updated_at = datetime.now()
        
        # 保存结果
        self.deployments[deployment_id] = result
        self._save_deployment_result(result)
        
        logger.info(f"Deployment completed: {deployment_id} - {result.status.value}")
        return result
    
    def _convert_model(self,
                       model: nn.Module,
                       config: DeploymentConfig,
                       deploy_dir: Path) -> Optional[str]:
        """转换模型格式"""
        output_path = None
        
        if config.format == DeploymentFormat.ONNX:
            output_path = deploy_dir / f"model.onnx"
            success = self.converter.convert_to_onnx(
                model,
                config.input_shape or (1, 3, 224, 224),
                str(output_path),
                config.output_names
            )
        
        elif config.format == DeploymentFormat.TENSORRT:
            # 先转换为ONNX
            onnx_path = deploy_dir / "temp.onnx"
            if self.converter.convert_to_onnx(
                model,
                config.input_shape or (1, 3, 224, 224),
                str(onnx_path)
            ):
                output_path = deploy_dir / "model.trt"
                success = self.converter.convert_to_tensorrt(
                    str(onnx_path),
                    str(output_path),
                    config.batch_size
                )
                # 清理临时文件
                onnx_path.unlink()
        
        elif config.format == DeploymentFormat.PYTORCH:
            output_path = deploy_dir / "model.pth"
            torch.save(model, output_path)
            success = True
        
        elif config.format == DeploymentFormat.JIT:
            output_path = deploy_dir / "model_jit.pt"
            model.eval()
            dummy_input = torch.randn(config.input_shape or (1, 3, 224, 224))
            traced_model = torch.jit.trace(model, dummy_input)
            torch.jit.save(traced_model, output_path)
            success = True
        
        else:
            logger.warning(f"Unsupported format: {config.format}")
            return None
        
        return str(output_path) if success else None
    
    def _optimize_model(self,
                       model_path: str,
                       config: DeploymentConfig) -> Optional[str]:
        """优化模型"""
        if config.optimization_level == OptimizationLevel.NONE:
            return None
        
        optimized_path = model_path
        
        try:
            if config.quantization:
                # 量化优化
                model = torch.load(model_path, map_location='cpu')
                quantized_model = self.converter.quantize_model(model)
                
                optimized_path = str(Path(model_path).parent / "model_quantized.pth")
                torch.save(quantized_model, optimized_path)
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None
        
        return optimized_path
    
    def _package_model(self,
                      model_path: str,
                      config: DeploymentConfig,
                      deploy_dir: Path) -> Optional[str]:
        """打包模型"""
        if config.platform in [DeploymentPlatform.CLOUD, DeploymentPlatform.LOCAL]:
            # Docker打包
            dockerfile = self.docker_packager.create_dockerfile(
                model_path,
                config.format,
                base_image="python:3.8-slim"
            )
            
            dockerfile_path = deploy_dir / "Dockerfile"
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile)
            
            # 创建推理脚本
            inference_script = self.docker_packager.create_inference_script(
                model_path,
                config.format,
                config.input_shape or (1, 3, 224, 224)
            )
            
            inference_path = deploy_dir / "inference.py"
            with open(inference_path, 'w') as f:
                f.write(inference_script)
            
            # 构建Docker镜像
            image_name = f"model-{model_id}-{version_id}"
            success = self.docker_packager.build_image(
                str(dockerfile_path),
                str(deploy_dir),
                image_name
            )
            
            return str(deploy_dir) if success else None
        
        elif config.platform == DeploymentPlatform.EDGE:
            # 边缘设备打包
            if config.format == DeploymentFormat.TFLITE:
                return self.edge_deployer.create_tflite_model(
                    torch.load(model_path),
                    config.input_shape or (1, 3, 224, 224),
                    str(deploy_dir / "model.tflite")
                )
            elif config.format == DeploymentFormat.COREML:
                return self.edge_deployer.create_coreml_model(
                    torch.load(model_path),
                    config.input_shape or (1, 3, 224, 224),
                    str(deploy_dir / "model.mlmodel")
                )
        
        return model_path
    
    def _deploy_to_platform(self,
                           package_path: str,
                           config: DeploymentConfig,
                           deployment_id: str) -> Optional[str]:
        """部署到指定平台"""
        if config.platform == DeploymentPlatform.CLOUD:
            # 云端部署
            if self.cloud_deployer.provider == "aws":
                # 部署到Lambda
                zip_path = self._create_lambda_package(package_path)
                if zip_path:
                    success = self.cloud_deployer.deploy_to_lambda(
                        zip_path,
                        f"model-{deployment_id}"
                    )
                    return f"https://{deployment_id}.execute-api.us-west-2.amazonaws.com/prod" if success else None
        
        elif config.platform == DeploymentPlatform.LOCAL:
            # 本地部署（返回本地路径）
            return f"file://{package_path}"
        
        return None
    
    def _create_lambda_package(self, package_path: str) -> Optional[str]:
        """创建Lambda部署包"""
        try:
            import zipfile
            
            zip_path = Path(package_path).parent / "lambda_package.zip"
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # 添加模型文件
                model_file = Path(package_path)
                if model_file.exists():
                    zipf.write(model_file, model_file.name)
                
                # 添加依赖
                requirements_path = Path(package_path).parent / "requirements.txt"
                if requirements_path.exists():
                    zipf.write(requirements_path, "requirements.txt")
            
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Failed to create Lambda package: {e}")
            return None
    
    def _find_model_path(self, model_id: str, version_id: str) -> Optional[str]:
        """查找模型文件路径"""
        # 在模型集成器中查找
        if self.model_integrator:
            version = self.model_integrator.registry.get_version(version_id)
            if version:
                return version.model_path
        
        # 在输出目录中查找
        for pattern in [f"{model_id}.pth", f"{model_id}.pt", f"{model_id}.onnx"]:
            path = self.output_dir / pattern
            if path.exists():
                return str(path)
        
        return None
    
    def _collect_metrics(self,
                        model_path: str,
                        config: DeploymentConfig) -> DeploymentMetrics:
        """收集部署指标"""
        metrics = DeploymentMetrics()
        
        # 模型大小
        model_file = Path(model_path)
        if model_file.exists():
            metrics.model_size_mb = model_file.stat().st_size / 1024 / 1024
        
        # 推理时间
        try:
            if config.format == DeploymentFormat.ONNX:
                sess = ort.InferenceSession(model_path)
                input_name = sess.get_inputs()[0].name
                dummy_input = np.random.randn(
                    *config.input_shape
                ).astype(np.float32)
                
                # 预热
                _ = sess.run(None, {input_name: dummy_input})
                
                # 测量时间
                times = []
                for _ in range(100):
                    start = time.time()
                    _ = sess.run(None, {input_name: dummy_input})
                    times.append(time.time() - start)
                
                metrics.inference_time_ms = np.mean(times) * 1000
                metrics.throughput_qps = 1 / np.mean(times)
            
            elif config.format == DeploymentFormat.PYTORCH:
                model = torch.load(model_path, map_location='cpu')
                model.eval()
                dummy_input = torch.randn(config.input_shape)
                
                # 预热
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # 测量时间
                times = []
                for _ in range(100):
                    start = time.time()
                    with torch.no_grad():
                        _ = model(dummy_input)
                    times.append(time.time() - start)
                
                metrics.inference_time_ms = np.mean(times) * 1000
                metrics.throughput_qps = 1 / np.mean(times)
        
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    def list_deployments(self,
                        status: Optional[DeploymentStatus] = None) -> List[DeploymentResult]:
        """列出部署"""
        deployments = list(self.deployments.values())
        
        if status:
            deployments = [d for d in deployments if d.status == status]
        
        return sorted(deployments, key=lambda x: x.created_at, reverse=True)
    
    def get_deployment(self, deployment_id: str) -> Optional[DeploymentResult]:
        """获取部署信息"""
        return self.deployments.get(deployment_id)
    
    def update_deployment(self,
                         deployment_id: str,
                         status: DeploymentStatus,
                         metrics: DeploymentMetrics = None) -> bool:
        """更新部署状态"""
        deployment = self.deployments.get(deployment_id)
        if deployment:
            deployment.status = status
            deployment.updated_at = datetime.now()
            
            if metrics:
                deployment.metrics = metrics
            
            self._save_deployment_result(deployment)
            return True
        
        return False
    
    def retract_deployment(self, deployment_id: str) -> bool:
        """撤回部署"""
        deployment = self.deployments.get(deployment_id)
        if deployment:
            # 这里应该实现实际的撤回逻辑
            # 例如：删除云端资源、停止服务等
            
            deployment.status = DeploymentStatus.RETRACTED
            deployment.updated_at = datetime.now()
            
            self._save_deployment_result(deployment)
            logger.info(f"Deployment retracted: {deployment_id}")
            return True
        
        return False
    
    def start_monitoring(self, interval: int = 60):
        """开始监控部署"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitor_deployments,
                args=(interval,),
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Deployment monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Deployment monitoring stopped")
    
    def _monitor_deployments(self, interval: int):
        """监控部署状态"""
        while self.monitoring_active:
            try:
                # 检查活跃部署
                active_deployments = [
                    d for d in self.deployments.values()
                    if d.status == DeploymentStatus.DEPLOYED
                ]
                
                for deployment in active_deployments:
                    # 检查健康状态
                    if deployment.deployment_url and deployment.deployment_url.startswith("http"):
                        try:
                            response = requests.get(
                                f"{deployment.deployment_url}/health",
                                timeout=5
                            )
                            if response.status_code != 200:
                                deployment.metrics.error_rate += 0.01
                            else:
                                deployment.metrics.error_rate *= 0.99  # 衰减
                            
                            deployment.metrics.availability_percent = 100 - deployment.metrics.error_rate * 100
                            
                        except:
                            deployment.metrics.error_rate += 0.05
                            deployment.metrics.availability_percent = 100 - deployment.metrics.error_rate * 100
                    
                    # 保存更新
                    self._save_deployment_result(deployment)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def generate_deployment_report(self,
                                  output_path: str = None) -> str:
        """生成部署报告"""
        if output_path is None:
            output_path = self.output_dir / f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report = f"# Model Deployment Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 统计信息
        total_deployments = len(self.deployments)
        active_deployments = len([d for d in self.deployments.values() if d.status == DeploymentStatus.DEPLOYED])
        failed_deployments = len([d for d in self.deployments.values() if d.status == DeploymentStatus.FAILED])
        
        report += "## Summary\n\n"
        report += f"- Total deployments: {total_deployments}\n"
        report += f"- Active deployments: {active_deployments}\n"
        report += f"- Failed deployments: {failed_deployments}\n"
        report += f"- Success rate: {active_deployments/total_deployments*100:.1f}%\n\n"
        
        # 按平台统计
        platform_stats = {}
        for deployment in self.deployments.values():
            platform = deployment.config.platform.value
            if platform not in platform_stats:
                platform_stats[platform] = {'total': 0, 'active': 0}
            
            platform_stats[platform]['total'] += 1
            if deployment.status == DeploymentStatus.DEPLOYED:
                platform_stats[platform]['active'] += 1
        
        report += "## Platform Statistics\n\n"
        report += "| Platform | Total | Active | Success Rate |\n"
        report += "|----------|-------|--------|--------------|\n"
        
        for platform, stats in platform_stats.items():
            success_rate = stats['active'] / stats['total'] * 100 if stats['total'] > 0 else 0
            report += f"| {platform} | {stats['total']} | {stats['active']} | {success_rate:.1f}% |\n"
        
        # 活跃部署详情
        report += "\n## Active Deployments\n\n"
        active_deployments_list = [
            d for d in self.deployments.values()
            if d.status == DeploymentStatus.DEPLOYED
        ]
        
        if active_deployments_list:
            report += "| Deployment ID | Model | Platform | Format | URL | Metrics |\n"
            report += "|---------------|-------|----------|--------|-----|---------|\n"
            
            for deployment in sorted(active_deployments_list, key=lambda x: x.created_at, reverse=True):
                report += f"| {deployment.deployment_id} | {deployment.model_id} | "
                report += f"{deployment.config.platform.value} | {deployment.config.format.value} | "
                report += f"{deployment.deployment_url or 'N/A'} | "
                report += f"Size: {deployment.metrics.model_size_mb:.2f}MB, "
                report += f"Time: {deployment.metrics.inference_time_ms:.2f}ms |\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Deployment report saved to: {output_path}")
        return str(output_path)
    
    def _save_deployment_result(self, result: DeploymentResult):
        """保存部署结果"""
        result_path = self.output_dir / f"deployment_{result.deployment_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_deployment_result(self, deployment_id: str) -> Optional[DeploymentResult]:
        """加载部署结果"""
        result_path = self.output_dir / f"deployment_{deployment_id}.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
                return DeploymentResult.from_dict(data)
        return None
    
    def get_deployment_summary(self) -> Dict[str, Any]:
        """获取部署摘要"""
        if not self.deployments:
            return {}
        
        total = len(self.deployments)
        active = len([d for d in self.deployments.values() if d.status == DeploymentStatus.DEPLOYED])
        failed = len([d for d in self.deployments.values() if d.status == DeploymentStatus.FAILED])
        
        # 平台分布
        platforms = {}
        formats = {}
        
        for deployment in self.deployments.values():
            platform = deployment.config.platform.value
            format_ = deployment.config.format.value
            
            platforms[platform] = platforms.get(platform, 0) + 1
            formats[format_] = formats.get(format_, 0) + 1
        
        # 平均指标
        active_deployments = [d for d in self.deployments.values() if d.status == DeploymentStatus.DEPLOYED]
        if active_deployments:
            avg_size = np.mean([d.metrics.model_size_mb for d in active_deployments])
            avg_time = np.mean([d.metrics.inference_time_ms for d in active_deployments])
        else:
            avg_size = 0
            avg_time = 0
        
        return {
            'total_deployments': total,
            'active_deployments': active,
            'failed_deployments': failed,
            'success_rate': active / total * 100 if total > 0 else 0,
            'platform_distribution': platforms,
            'format_distribution': formats,
            'average_model_size_mb': avg_size,
            'average_inference_time_ms': avg_time
        }


def create_model_deployer(output_dir: str = "./deployments",
                          model_integrator: ModelIntegrator = None) -> ModelDeployer:
    """创建模型部署器实例"""
    return ModelDeployer(output_dir, model_integrator)