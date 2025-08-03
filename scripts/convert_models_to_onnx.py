#!/usr/bin/env python3
"""
模型ONNX转换脚本
将所有训练好的PyTorch模型转换为ONNX格式，用于C#项目部署

更新：实现了针对每个模型的单独转换函数，以解决不同模型架构的特殊需求
"""

import os
import sys
import torch
import torch.onnx
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型类
import torch.nn as nn

# 定义SimplifiedAirBubbleDetector类（从训练脚本复制）
class SimplifiedAirBubbleDetector(nn.Module):
    """简化版气孔检测器 - 解决过拟合问题"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        # 大幅简化的特征提取器 (目标: <100k参数)
        self.features = nn.Sequential(
            # 第一层: 保持分辨率
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 第二层: 轻微下采样
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 35x35
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # 第三层: 特征提取
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # 第四层: 进一步下采样
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 18x18
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

# 导入其他模型类
try:
    from models.efficientnet_b0 import EfficientNetB0
except ImportError:
    EfficientNetB0 = None

try:
    from models.resnet18_improved import ResNet18Improved
except ImportError:
    ResNet18Improved = None

try:
    from models.coatnet import CoAtNet
except ImportError:
    CoAtNet = None

try:
    from models.convnext_tiny import ConvNextTiny
except ImportError:
    ConvNextTiny = None

try:
    from models.vit_tiny import ViTTiny
except ImportError:
    ViTTiny = None

try:
    from models.airbubble_hybrid_net import AirBubbleHybridNet
except ImportError:
    AirBubbleHybridNet = None

try:
    from models.mic_mobilenetv3 import MICMobileNetV3
except ImportError:
    MICMobileNetV3 = None

try:
    from models.micro_vit import MicroViT
except ImportError:
    MicroViT = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('onnx_conversion.log'),
        logging.StreamHandler(sys.stdout)  # 确保输出到标准输出
    ]
)
logger = logging.getLogger(__name__)

class ONNXConverter:
    """ONNX模型转换器"""
    
    def __init__(self):
        self.input_size = (1, 3, 70, 70)  # MIC测试图像尺寸
        self.output_dir = Path("deployment/onnx_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型配置映射
        self.model_configs = {
            "simplified_airbubble_detector": {
                "class": SimplifiedAirBubbleDetector,
                "checkpoint": "experiments/simplified_airbubble_detector/simplified_airbubble_best.pth",
                "priority": 1,  # 最高优先级
                "description": "Simplified AirBubble Detector - Champion Model (100% accuracy)",
                "converter_function": self.convert_simplified_airbubble_detector
            },
            "efficientnet_b0": {
                "class": EfficientNetB0,
                "checkpoint": "experiments/experiment_20250802_140818/efficientnet_b0/best_model.pth",
                "priority": 2,
                "description": "EfficientNet-B0 - Historical Baseline (98.14% accuracy)",
                "converter_function": self.convert_efficientnet_b0
            },
            "resnet18_improved": {
                "class": ResNet18Improved,
                "checkpoint": "experiments/experiment_20250802_164948/resnet18_improved/best_model.pth",
                "priority": 3,
                "description": "ResNet18-Improved - High Performance (97.83% accuracy)",
                "converter_function": self.convert_resnet18_improved
            },
            "coatnet": {
                "class": CoAtNet,
                "checkpoint": "experiments/experiment_20250803_032628/coatnet/best_model.pth",
                "priority": 4,
                "description": "CoAtNet - Hybrid Architecture (91.30% accuracy)",
                "converter_function": self.convert_coatnet
            },
            "convnext_tiny": {
                "class": ConvNextTiny,
                "checkpoint": "experiments/experiment_20250802_231639/convnext_tiny/best_model.pth",
                "priority": 5,
                "description": "ConvNeXt Tiny - Modern CNN (89.70% accuracy)",
                "converter_function": self.convert_convnext_tiny
            },
            "vit_tiny": {
                "class": ViTTiny,
                "checkpoint": "experiments/experiment_20250803_020217/vit_tiny/best_model.pth",
                "priority": 6,
                "description": "Vision Transformer Tiny - Attention-based (88.50% accuracy)",
                "converter_function": self.convert_vit_tiny
            },
            "airbubble_hybrid_net": {
                "class": AirBubbleHybridNet,
                "checkpoint": "experiments/experiment_20250803_115344/airbubble_hybrid_net/best_model.pth",
                "priority": 7,
                "description": "AirBubble Hybrid Net - Domain-specific (87.40% accuracy)",
                "converter_function": self.convert_airbubble_hybrid_net
            },
            "mic_mobilenetv3": {
                "class": MICMobileNetV3,
                "checkpoint": "experiments/experiment_20250803_101438/mic_mobilenetv3/best_model.pth",
                "priority": 8,
                "description": "MIC MobileNetV3 - Mobile Optimized (85.20% accuracy)",
                "converter_function": self.convert_mic_mobilenetv3
            },
            "micro_vit": {
                "class": MicroViT,
                "checkpoint": "experiments/experiment_20250803_102845/micro_vit/best_model.pth",
                "priority": 9,
                "description": "Micro ViT - Lightweight Transformer (83.60% accuracy)",
                "converter_function": self.convert_micro_vit
            },
            "enhanced_airbubble_detector": {
                "class": SimplifiedAirBubbleDetector,  # 使用相同的类，但不同的权重
                "checkpoint": "experiments/enhanced_airbubble_detector/best_model.pth",
                "priority": 10,
                "description": "Enhanced AirBubble Detector (52.00% accuracy - overfit)",
                "converter_function": self.convert_enhanced_airbubble_detector
            }
        }
        
    def load_model(self, model_name: str):
        """加载PyTorch模型"""
        try:
            config = self.model_configs[model_name]
            model_class = config["class"]
            checkpoint_path = config["checkpoint"]
            
            # 创建模型实例
            if model_name == "simplified_airbubble_detector" or model_name == "enhanced_airbubble_detector":
                model = model_class(input_channels=3, num_classes=2)
            else:
                model = model_class(num_classes=2)
            
            # 加载权重
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"✅ 成功加载模型权重: {checkpoint_path}")
            else:
                logger.warning(f"⚠️ 权重文件不存在: {checkpoint_path}")
                return None
                
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败 {model_name}: {str(e)}")
            # 尝试使用备用方法加载模型
            if model_name == "simplified_airbubble_detector":
                try:
                    logger.info(f"🔄 尝试使用备用方法加载 {model_name}...")
                    # 检查模型结构是否匹配
                    logger.info(f"检查模型结构...")
                    
                    # 加载检查点以检查结构
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                        
                        # 打印模型结构信息
                        logger.info(f"检查点中的键: {state_dict.keys()}")
                        
                        # 创建一个新的模型实例
                        from scripts.train_simplified_airbubble_detector import SimplifiedAirBubbleDetector as TrainSimplifiedAirBubbleDetector
                        model = TrainSimplifiedAirBubbleDetector(input_channels=3, num_classes=2)
                        model.load_state_dict(state_dict)
                        model.eval()
                        logger.info(f"✅ 使用备用方法成功加载模型: {model_name}")
                        return model
                except Exception as e2:
                    logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
            return None
    
    # 通用转换函数 - 作为基础实现
    def convert_to_onnx(self, model_name: str, model: torch.nn.Module, opset_version=11):
        """将PyTorch模型转换为ONNX格式 - 通用方法"""
        try:
            # 创建示例输入
            dummy_input = torch.randn(self.input_size)
            
            # ONNX输出路径
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 转换为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],  # [batch_size, num_classes]
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
            
    def convert_resnet18_improved(self, model_name: str, model: torch.nn.Module):
        """转换ResNet18Improved模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ResNet是标准架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_coatnet(self, model_name: str, model: torch.nn.Module):
        """转换CoAtNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # CoAtNet包含注意力机制，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_convnext_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ConvNextTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ConvNeXt包含特殊的LayerNorm实现，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_vit_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ViTTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # Vision Transformer需要特殊处理注意力机制
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.script创建脚本模型
                scripted_model = torch.jit.script(model)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                dummy_input = torch.randn(self.input_size)
                
                # 导出脚本模型
                torch.onnx.export(
                    scripted_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_airbubble_hybrid_net(self, model_name: str, model: torch.nn.Module):
        """转换AirBubbleHybridNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 混合网络架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=12)
    
    def convert_mic_mobilenetv3(self, model_name: str, model: torch.nn.Module):
        """转换MICMobileNetV3模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # MobileNetV3可能包含特殊的激活函数和SE模块
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_micro_vit(self, model_name: str, model: torch.nn.Module):
        """转换MicroViT模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 微型ViT，使用与ViT类似的方法
        return self.convert_vit_tiny(model_name, model)
    
    def convert_enhanced_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换EnhancedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 与SimplifiedAirBubbleDetector类似，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    # 针对每个模型的专用转换函数
    def convert_simplified_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换SimplifiedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 简单模型，使用标准转换方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_efficientnet_b0(self, model_name: str, model: torch.nn.Module):
        """转换EfficientNetB0模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # EfficientNet可能需要特殊处理激活函数
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_resnet18_improved(self, model_name: str, model: torch.nn.Module):
        """转换ResNet18Improved模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ResNet是标准架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_coatnet(self, model_name: str, model: torch.nn.Module):
        """转换CoAtNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # CoAtNet包含注意力机制，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_convnext_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ConvNextTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ConvNeXt包含特殊的LayerNorm实现，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_vit_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ViTTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # Vision Transformer需要特殊处理注意力机制
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.script创建脚本模型
                scripted_model = torch.jit.script(model)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                dummy_input = torch.randn(self.input_size)
                
                # 导出脚本模型
                torch.onnx.export(
                    scripted_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_airbubble_hybrid_net(self, model_name: str, model: torch.nn.Module):
        """转换AirBubbleHybridNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 混合网络架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=12)
    
    def convert_mic_mobilenetv3(self, model_name: str, model: torch.nn.Module):
        """转换MICMobileNetV3模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # MobileNetV3可能包含特殊的激活函数和SE模块
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_micro_vit(self, model_name: str, model: torch.nn.Module):
        """转换MicroViT模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 微型ViT，使用与ViT类似的方法
        return self.convert_vit_tiny(model_name, model)
    
    def convert_enhanced_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换EnhancedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 与SimplifiedAirBubbleDetector类似，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
        
    # 删除重复的load_model方法
#!/usr/bin/env python3
"""
模型ONNX转换脚本
将所有训练好的PyTorch模型转换为ONNX格式，用于C#项目部署

更新：实现了针对每个模型的单独转换函数，以解决不同模型架构的特殊需求
"""

import os
import sys
import torch
import torch.onnx
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型类
import torch.nn as nn

# 定义SimplifiedAirBubbleDetector类（从训练脚本复制）
class SimplifiedAirBubbleDetector(nn.Module):
    """简化版气孔检测器 - 解决过拟合问题"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        # 大幅简化的特征提取器 (目标: <100k参数)
        self.features = nn.Sequential(
            # 第一层: 保持分辨率
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 第二层: 轻微下采样
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 35x35
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # 第三层: 特征提取
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # 第四层: 进一步下采样
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 18x18
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

# 导入其他模型类
try:
    from models.efficientnet_b0 import EfficientNetB0
except ImportError:
    EfficientNetB0 = None

try:
    from models.resnet18_improved import ResNet18Improved
except ImportError:
    ResNet18Improved = None

try:
    from models.coatnet import CoAtNet
except ImportError:
    CoAtNet = None

try:
    from models.convnext_tiny import ConvNextTiny
except ImportError:
    ConvNextTiny = None

try:
    from models.vit_tiny import ViTTiny
except ImportError:
    ViTTiny = None

try:
    from models.airbubble_hybrid_net import AirBubbleHybridNet
except ImportError:
    AirBubbleHybridNet = None

try:
    from models.mic_mobilenetv3 import MICMobileNetV3
except ImportError:
    MICMobileNetV3 = None

try:
    from models.micro_vit import MicroViT
except ImportError:
    MicroViT = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('onnx_conversion.log'),
        logging.StreamHandler(sys.stdout)  # 确保输出到标准输出
    ]
)
logger = logging.getLogger(__name__)

class ONNXConverter:
    """ONNX模型转换器"""
    
    def __init__(self):
        self.input_size = (1, 3, 70, 70)  # MIC测试图像尺寸
        self.output_dir = Path("deployment/onnx_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型配置映射
        self.model_configs = {
            "simplified_airbubble_detector": {
                "class": SimplifiedAirBubbleDetector,
                "checkpoint": "experiments/simplified_airbubble_detector/simplified_airbubble_best.pth",
                "priority": 1,  # 最高优先级
                "description": "Simplified AirBubble Detector - Champion Model (100% accuracy)",
                "converter_function": self.convert_simplified_airbubble_detector
            },
            "efficientnet_b0": {
                "class": EfficientNetB0,
                "checkpoint": "experiments/experiment_20250802_140818/efficientnet_b0/best_model.pth",
                "priority": 2,
                "description": "EfficientNet-B0 - Historical Baseline (98.14% accuracy)",
                "converter_function": self.convert_efficientnet_b0
            },
            "resnet18_improved": {
                "class": ResNet18Improved,
                "checkpoint": "experiments/experiment_20250802_164948/resnet18_improved/best_model.pth",
                "priority": 3,
                "description": "ResNet18-Improved - High Performance (97.83% accuracy)",
                "converter_function": self.convert_resnet18_improved
            },
            "coatnet": {
                "class": CoAtNet,
                "checkpoint": "experiments/experiment_20250803_032628/coatnet/best_model.pth",
                "priority": 4,
                "description": "CoAtNet - Hybrid Architecture (91.30% accuracy)",
                "converter_function": self.convert_coatnet
            },
            "convnext_tiny": {
                "class": ConvNextTiny,
                "checkpoint": "experiments/experiment_20250802_231639/convnext_tiny/best_model.pth",
                "priority": 5,
                "description": "ConvNeXt Tiny - Modern CNN (89.70% accuracy)",
                "converter_function": self.convert_convnext_tiny
            },
            "vit_tiny": {
                "class": ViTTiny,
                "checkpoint": "experiments/experiment_20250803_020217/vit_tiny/best_model.pth",
                "priority": 6,
                "description": "Vision Transformer Tiny - Attention-based (88.50% accuracy)",
                "converter_function": self.convert_vit_tiny
            },
            "airbubble_hybrid_net": {
                "class": AirBubbleHybridNet,
                "checkpoint": "experiments/experiment_20250803_115344/airbubble_hybrid_net/best_model.pth",
                "priority": 7,
                "description": "AirBubble Hybrid Net - Domain-specific (87.40% accuracy)",
                "converter_function": self.convert_airbubble_hybrid_net
            },
            "mic_mobilenetv3": {
                "class": MICMobileNetV3,
                "checkpoint": "experiments/experiment_20250803_101438/mic_mobilenetv3/best_model.pth",
                "priority": 8,
                "description": "MIC MobileNetV3 - Mobile Optimized (85.20% accuracy)",
                "converter_function": self.convert_mic_mobilenetv3
            },
            "micro_vit": {
                "class": MicroViT,
                "checkpoint": "experiments/experiment_20250803_102845/micro_vit/best_model.pth",
                "priority": 9,
                "description": "Micro ViT - Lightweight Transformer (83.60% accuracy)",
                "converter_function": self.convert_micro_vit
            },
            "enhanced_airbubble_detector": {
                "class": SimplifiedAirBubbleDetector,  # 使用相同的类，但不同的权重
                "checkpoint": "experiments/enhanced_airbubble_detector/best_model.pth",
                "priority": 10,
                "description": "Enhanced AirBubble Detector (52.00% accuracy - overfit)",
                "converter_function": self.convert_enhanced_airbubble_detector
            }
        }
        
    def load_model(self, model_name: str):
        """加载PyTorch模型"""
        try:
            config = self.model_configs[model_name]
            model_class = config["class"]
            checkpoint_path = config["checkpoint"]
            
            # 创建模型实例
            if model_name == "simplified_airbubble_detector" or model_name == "enhanced_airbubble_detector":
                model = model_class(input_channels=3, num_classes=2)
            else:
                model = model_class(num_classes=2)
            
            # 加载权重
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"✅ 成功加载模型权重: {checkpoint_path}")
            else:
                logger.warning(f"⚠️ 权重文件不存在: {checkpoint_path}")
                return None
                
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败 {model_name}: {str(e)}")
            # 尝试使用备用方法加载模型
            if model_name == "simplified_airbubble_detector":
                try:
                    logger.info(f"🔄 尝试使用备用方法加载 {model_name}...")
                    # 检查模型结构是否匹配
                    logger.info(f"检查模型结构...")
                    
                    # 加载检查点以检查结构
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                        
                        # 打印模型结构信息
                        logger.info(f"检查点中的键: {state_dict.keys()}")
                        
                        # 创建一个新的模型实例
                        from scripts.train_simplified_airbubble_detector import SimplifiedAirBubbleDetector as TrainSimplifiedAirBubbleDetector
                        model = TrainSimplifiedAirBubbleDetector(input_channels=3, num_classes=2)
                        model.load_state_dict(state_dict)
                        model.eval()
                        logger.info(f"✅ 使用备用方法成功加载模型: {model_name}")
                        return model
                except Exception as e2:
                    logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
            return None
    
    # 通用转换函数 - 作为基础实现
    def convert_to_onnx(self, model_name: str, model: torch.nn.Module, opset_version=11):
        """将PyTorch模型转换为ONNX格式 - 通用方法"""
        try:
            # 创建示例输入
            dummy_input = torch.randn(self.input_size)
            
            # ONNX输出路径
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 转换为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],  # [batch_size, num_classes]
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
            
    def convert_resnet18_improved(self, model_name: str, model: torch.nn.Module):
        """转换ResNet18Improved模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ResNet是标准架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_coatnet(self, model_name: str, model: torch.nn.Module):
        """转换CoAtNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # CoAtNet包含注意力机制，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_convnext_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ConvNextTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ConvNeXt包含特殊的LayerNorm实现，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_vit_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ViTTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # Vision Transformer需要特殊处理注意力机制
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.script创建脚本模型
                scripted_model = torch.jit.script(model)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                dummy_input = torch.randn(self.input_size)
                
                # 导出脚本模型
                torch.onnx.export(
                    scripted_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_airbubble_hybrid_net(self, model_name: str, model: torch.nn.Module):
        """转换AirBubbleHybridNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 混合网络架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=12)
    
    def convert_mic_mobilenetv3(self, model_name: str, model: torch.nn.Module):
        """转换MICMobileNetV3模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # MobileNetV3可能包含特殊的激活函数和SE模块
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_micro_vit(self, model_name: str, model: torch.nn.Module):
        """转换MicroViT模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 微型ViT，使用与ViT类似的方法
        return self.convert_vit_tiny(model_name, model)
    
    def convert_enhanced_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换EnhancedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 与SimplifiedAirBubbleDetector类似，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    # 针对每个模型的专用转换函数
    def convert_simplified_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换SimplifiedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 简单模型，使用标准转换方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_efficientnet_b0(self, model_name: str, model: torch.nn.Module):
        """转换EfficientNetB0模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # EfficientNet可能需要特殊处理激活函数
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_resnet18_improved(self, model_name: str, model: torch.nn.Module):
        """转换ResNet18Improved模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ResNet是标准架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_coatnet(self, model_name: str, model: torch.nn.Module):
        """转换CoAtNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # CoAtNet包含注意力机制，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_convnext_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ConvNextTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ConvNeXt包含特殊的LayerNorm实现，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_vit_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ViTTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # Vision Transformer需要特殊处理注意力机制
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.script创建脚本模型
                scripted_model = torch.jit.script(model)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                dummy_input = torch.randn(self.input_size)
                
                # 导出脚本模型
                torch.onnx.export(
                    scripted_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_airbubble_hybrid_net(self, model_name: str, model: torch.nn.Module):
        """转换AirBubbleHybridNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 混合网络架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=12)
    
    def convert_mic_mobilenetv3(self, model_name: str, model: torch.nn.Module):
        """转换MICMobileNetV3模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # MobileNetV3可能包含特殊的激活函数和SE模块
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_micro_vit(self, model_name: str, model: torch.nn.Module):
        """转换MicroViT模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 微型ViT，使用与ViT类似的方法
        return self.convert_vit_tiny(model_name, model)
    
    def convert_enhanced_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换EnhancedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 与SimplifiedAirBubbleDetector类似，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
        
#!/usr/bin/env python3
"""
模型ONNX转换脚本
将所有训练好的PyTorch模型转换为ONNX格式，用于C#项目部署

更新：实现了针对每个模型的单独转换函数，以解决不同模型架构的特殊需求
"""

import os
import sys
import torch
import torch.onnx
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型类
import torch.nn as nn

# 定义SimplifiedAirBubbleDetector类（从训练脚本复制）
class SimplifiedAirBubbleDetector(nn.Module):
    """简化版气孔检测器 - 解决过拟合问题"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        # 大幅简化的特征提取器 (目标: <100k参数)
        self.features = nn.Sequential(
            # 第一层: 保持分辨率
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 第二层: 轻微下采样
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 35x35
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # 第三层: 特征提取
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # 第四层: 进一步下采样
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 18x18
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

# 导入其他模型类
try:
    from models.efficientnet_b0 import EfficientNetB0
except ImportError:
    EfficientNetB0 = None

try:
    from models.resnet18_improved import ResNet18Improved
except ImportError:
    ResNet18Improved = None

try:
    from models.coatnet import CoAtNet
except ImportError:
    CoAtNet = None

try:
    from models.convnext_tiny import ConvNextTiny
except ImportError:
    ConvNextTiny = None

try:
    from models.vit_tiny import ViTTiny
except ImportError:
    ViTTiny = None

try:
    from models.airbubble_hybrid_net import AirBubbleHybridNet
except ImportError:
    AirBubbleHybridNet = None

try:
    from models.mic_mobilenetv3 import MICMobileNetV3
except ImportError:
    MICMobileNetV3 = None

try:
    from models.micro_vit import MicroViT
except ImportError:
    MicroViT = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('onnx_conversion.log'),
        logging.StreamHandler(sys.stdout)  # 确保输出到标准输出
    ]
)
logger = logging.getLogger(__name__)

class ONNXConverter:
    """ONNX模型转换器"""
    
    def __init__(self):
        self.input_size = (1, 3, 70, 70)  # MIC测试图像尺寸
        self.output_dir = Path("deployment/onnx_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型配置映射
        self.model_configs = {
            "simplified_airbubble_detector": {
                "class": SimplifiedAirBubbleDetector,
                "checkpoint": "experiments/simplified_airbubble_detector/simplified_airbubble_best.pth",
                "priority": 1,  # 最高优先级
                "description": "Simplified AirBubble Detector - Champion Model (100% accuracy)",
                "converter_function": self.convert_simplified_airbubble_detector
            },
            "efficientnet_b0": {
                "class": EfficientNetB0,
                "checkpoint": "experiments/experiment_20250802_140818/efficientnet_b0/best_model.pth",
                "priority": 2,
                "description": "EfficientNet-B0 - Historical Baseline (98.14% accuracy)",
                "converter_function": self.convert_efficientnet_b0
            },
            "resnet18_improved": {
                "class": ResNet18Improved,
                "checkpoint": "experiments/experiment_20250802_164948/resnet18_improved/best_model.pth",
                "priority": 3,
                "description": "ResNet18-Improved - High Performance (97.83% accuracy)",
                "converter_function": self.convert_resnet18_improved
            },
            "coatnet": {
                "class": CoAtNet,
                "checkpoint": "experiments/experiment_20250803_032628/coatnet/best_model.pth",
                "priority": 4,
                "description": "CoAtNet - Hybrid Architecture (91.30% accuracy)",
                "converter_function": self.convert_coatnet
            },
            "convnext_tiny": {
                "class": ConvNextTiny,
                "checkpoint": "experiments/experiment_20250802_231639/convnext_tiny/best_model.pth",
                "priority": 5,
                "description": "ConvNeXt Tiny - Modern CNN (89.70% accuracy)",
                "converter_function": self.convert_convnext_tiny
            },
            "vit_tiny": {
                "class": ViTTiny,
                "checkpoint": "experiments/experiment_20250803_020217/vit_tiny/best_model.pth",
                "priority": 6,
                "description": "Vision Transformer Tiny - Attention-based (88.50% accuracy)",
                "converter_function": self.convert_vit_tiny
            },
            "airbubble_hybrid_net": {
                "class": AirBubbleHybridNet,
                "checkpoint": "experiments/experiment_20250803_115344/airbubble_hybrid_net/best_model.pth",
                "priority": 7,
                "description": "AirBubble Hybrid Net - Domain-specific (87.40% accuracy)",
                "converter_function": self.convert_airbubble_hybrid_net
            },
            "mic_mobilenetv3": {
                "class": MICMobileNetV3,
                "checkpoint": "experiments/experiment_20250803_101438/mic_mobilenetv3/best_model.pth",
                "priority": 8,
                "description": "MIC MobileNetV3 - Mobile Optimized (85.20% accuracy)",
                "converter_function": self.convert_mic_mobilenetv3
            },
            "micro_vit": {
                "class": MicroViT,
                "checkpoint": "experiments/experiment_20250803_102845/micro_vit/best_model.pth",
                "priority": 9,
                "description": "Micro ViT - Lightweight Transformer (83.60% accuracy)",
                "converter_function": self.convert_micro_vit
            },
            "enhanced_airbubble_detector": {
                "class": SimplifiedAirBubbleDetector,  # 使用相同的类，但不同的权重
                "checkpoint": "experiments/enhanced_airbubble_detector/best_model.pth",
                "priority": 10,
                "description": "Enhanced AirBubble Detector (52.00% accuracy - overfit)",
                "converter_function": self.convert_enhanced_airbubble_detector
            }
        }
        
    def load_model(self, model_name: str):
        """加载PyTorch模型"""
        try:
            config = self.model_configs[model_name]
            model_class = config["class"]
            checkpoint_path = config["checkpoint"]
            
            # 创建模型实例
            if model_name == "simplified_airbubble_detector" or model_name == "enhanced_airbubble_detector":
                model = model_class(input_channels=3, num_classes=2)
            else:
                model = model_class(num_classes=2)
            
            # 加载权重
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"✅ 成功加载模型权重: {checkpoint_path}")
            else:
                logger.warning(f"⚠️ 权重文件不存在: {checkpoint_path}")
                return None
                
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败 {model_name}: {str(e)}")
            # 尝试使用备用方法加载模型
            if model_name == "simplified_airbubble_detector":
                try:
                    logger.info(f"🔄 尝试使用备用方法加载 {model_name}...")
                    # 检查模型结构是否匹配
                    logger.info(f"检查模型结构...")
                    
                    # 加载检查点以检查结构
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                        else:
                            state_dict = checkpoint
                        
                        # 打印模型结构信息
                        logger.info(f"检查点中的键: {state_dict.keys()}")
                        
                        # 创建一个新的模型实例
                        from scripts.train_simplified_airbubble_detector import SimplifiedAirBubbleDetector as TrainSimplifiedAirBubbleDetector
                        model = TrainSimplifiedAirBubbleDetector(input_channels=3, num_classes=2)
                        model.load_state_dict(state_dict)
                        model.eval()
                        logger.info(f"✅ 使用备用方法成功加载模型: {model_name}")
                        return model
                except Exception as e2:
                    logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
            return None
    
    # 通用转换函数 - 作为基础实现
    def convert_to_onnx(self, model_name: str, model: torch.nn.Module, opset_version=11):
        """将PyTorch模型转换为ONNX格式 - 通用方法"""
        try:
            # 创建示例输入
            dummy_input = torch.randn(self.input_size)
            
            # ONNX输出路径
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 转换为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],  # [batch_size, num_classes]
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
            
    def convert_resnet18_improved(self, model_name: str, model: torch.nn.Module):
        """转换ResNet18Improved模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ResNet是标准架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_coatnet(self, model_name: str, model: torch.nn.Module):
        """转换CoAtNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # CoAtNet包含注意力机制，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_convnext_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ConvNextTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ConvNeXt包含特殊的LayerNorm实现，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_vit_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ViTTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # Vision Transformer需要特殊处理注意力机制
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.script创建脚本模型
                scripted_model = torch.jit.script(model)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                dummy_input = torch.randn(self.input_size)
                
                # 导出脚本模型
                torch.onnx.export(
                    scripted_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_airbubble_hybrid_net(self, model_name: str, model: torch.nn.Module):
        """转换AirBubbleHybridNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 混合网络架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=12)
    
    def convert_mic_mobilenetv3(self, model_name: str, model: torch.nn.Module):
        """转换MICMobileNetV3模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # MobileNetV3可能包含特殊的激活函数和SE模块
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_micro_vit(self, model_name: str, model: torch.nn.Module):
        """转换MicroViT模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 微型ViT，使用与ViT类似的方法
        return self.convert_vit_tiny(model_name, model)
    
    def convert_enhanced_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换EnhancedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 与SimplifiedAirBubbleDetector类似，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    # 针对每个模型的专用转换函数
    def convert_simplified_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换SimplifiedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 简单模型，使用标准转换方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_efficientnet_b0(self, model_name: str, model: torch.nn.Module):
        """转换EfficientNetB0模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # EfficientNet可能需要特殊处理激活函数
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_resnet18_improved(self, model_name: str, model: torch.nn.Module):
        """转换ResNet18Improved模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ResNet是标准架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_coatnet(self, model_name: str, model: torch.nn.Module):
        """转换CoAtNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # CoAtNet包含注意力机制，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.trace创建可追踪模型
                dummy_input = torch.randn(self.input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                
                # 导出追踪模型
                torch.onnx.export(
                    traced_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_convnext_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ConvNextTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # ConvNeXt包含特殊的LayerNorm实现，需要特殊处理
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_vit_tiny(self, model_name: str, model: torch.nn.Module):
        """转换ViTTiny模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # Vision Transformer需要特殊处理注意力机制
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持注意力机制
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本支持注意力机制
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            # 尝试使用备用方法
            logger.info(f"🔄 尝试使用备用方法转换 {model_name}...")
            try:
                # 使用torch.jit.script创建脚本模型
                scripted_model = torch.jit.script(model)
                
                onnx_path = self.output_dir / f"{model_name}.onnx"
                dummy_input = torch.randn(self.input_size)
                
                # 导出脚本模型
                torch.onnx.export(
                    scripted_model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                # 验证ONNX模型
                import onnx
                onnx_model = onnx.load(str(onnx_path))
                onnx.checker.check_model(onnx_model)
                
                # 获取文件大小
                file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
                
                logger.info(f"✅ 使用备用方法成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
                
                return {
                    "model_name": model_name,
                    "onnx_path": str(onnx_path),
                    "file_size_mb": round(file_size, 2),
                    "input_shape": list(self.input_size),
                    "output_shape": [1, 2],
                    "success": True
                }
            except Exception as e2:
                logger.error(f"❌ 备用方法也失败 {model_name}: {str(e2)}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"主要错误: {str(e)}; 备用方法错误: {str(e2)}"
                }
    
    def convert_airbubble_hybrid_net(self, model_name: str, model: torch.nn.Module):
        """转换AirBubbleHybridNet模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 混合网络架构，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=12)
    
    def convert_mic_mobilenetv3(self, model_name: str, model: torch.nn.Module):
        """转换MICMobileNetV3模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # MobileNetV3可能包含特殊的激活函数和SE模块
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def convert_micro_vit(self, model_name: str, model: torch.nn.Module):
        """转换MicroViT模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 微型ViT，使用与ViT类似的方法
        return self.convert_vit_tiny(model_name, model)
    
    def convert_enhanced_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换EnhancedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 与SimplifiedAirBubbleDetector类似，使用通用方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
        
    def load_model(self, model_name: str):
        """加载PyTorch模型"""
        try:
            config = self.model_configs[model_name]
            model_class = config["class"]
            checkpoint_path = config["checkpoint"]
            
            # 创建模型实例
            if model_name == "simplified_airbubble_detector" or model_name == "enhanced_airbubble_detector":
                model = model_class(input_channels=3, num_classes=2)
            else:
                model = model_class(num_classes=2)
            
            # 加载权重
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"✅ 成功加载模型权重: {checkpoint_path}")
            else:
                logger.warning(f"⚠️ 权重文件不存在: {checkpoint_path}")
                return None
                
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败 {model_name}: {str(e)}")
            return None
    
    # 通用转换函数 - 作为基础实现
    def convert_to_onnx(self, model_name: str, model: torch.nn.Module, opset_version=11):
        """将PyTorch模型转换为ONNX格式 - 通用方法"""
        try:
            # 创建示例输入
            dummy_input = torch.randn(self.input_size)
            
            # ONNX输出路径
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 转换为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],  # [batch_size, num_classes]
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    # 针对每个模型的专用转换函数
    def convert_simplified_airbubble_detector(self, model_name: str, model: torch.nn.Module):
        """转换SimplifiedAirBubbleDetector模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # 简单模型，使用标准转换方法
        return self.convert_to_onnx(model_name, model, opset_version=11)
    
    def convert_efficientnet_b0(self, model_name: str, model: torch.nn.Module):
        """转换EfficientNetB0模型"""
        logger.info(f"🔄 使用专用转换函数处理 {model_name}...")
        # EfficientNet可能需要特殊处理激活函数
        try:
            dummy_input = torch.randn(self.input_size)
            onnx_path = self.output_dir / f"{model_name}.onnx"
            
            # 使用更高的opset版本以支持更多操作
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=12,  # 使用更高版本
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 验证ONNX模型
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # 获取文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": str(onnx_path),
                "file_size_mb": round(file_size, 2),
                "input_shape": list(self.input_size),
                "output_shape": [1, 2],
                "success": True
            }
        except Exception as e:
            logger.error(f"❌ ONNX转换失败 {model_name}: {str(e)}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def test_onnx_model(self, onnx_path: str):
        """测试ONNX模型推理"""
        try:
            import onnxruntime as ort
            
            # 创建推理会话
            session = ort.InferenceSession(onnx_path)
            
            # 创建测试输入
            test_input = torch.randn(self.input_size).numpy()
            
            # 运行推理
            outputs = session.run(None, {'input': test_input})
            
            logger.info(f"✅ ONNX模型测试成功: {onnx_path}")
            logger.info(f"   输出形状: {outputs[0].shape}")
            logger.info(f"   输出范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX模型测试失败: {str(e)}")
            return False
            
    def convert_single_model(self, model_name: str):
        """转换单个指定的模型"""
        if model_name not in self.model_configs:
            logger.error(f"❌ 未知模型: {model_name}")
            logger.info(f"可用模型: {', '.join(self.model_configs.keys())}")
            return False
            
        logger.info(f"\n📦 正在处理模型: {model_name}")
        logger.info(f"   描述: {self.model_configs[model_name]['description']}")
        
        # 加载模型
        model = self.load_model(model_name)
        if model is None:
            return False
        
        # 获取专用转换函数
        converter_function = self.model_configs[model_name].get('converter_function', self.convert_to_onnx)
        
        # 转换为ONNX
        result = converter_function(model_name, model)
        
        # 测试ONNX模型
        if result["success"]:
            if self.test_onnx_model(result["onnx_path"]):
                logger.info(f"✅ 模型 {model_name} 转换并测试成功!")
                return True
            else:
                logger.error(f"❌ 模型 {model_name} 转换成功但测试失败")
                return False
        else:
            logger.error(f"❌ 模型 {model_name} 转换失败: {result.get('error', '未知错误')}")
            return False
    
    def generate_model_info(self, conversion_results: list):
        """生成模型信息文件"""
        model_info = {
            "conversion_date": datetime.now().isoformat(),
            "input_shape": list(self.input_size),
            "output_shape": [1, 2],
            "models": []
        }
        
        for result in conversion_results:
            if result["success"]:
                model_name = result["model_name"]
                config = self.model_configs[model_name]
                
                model_info["models"].append({
                    "name": model_name,
                    "description": config["description"],
                    "priority": config["priority"],
                    "onnx_file": os.path.basename(result["onnx_path"]),
                    "file_size_mb": result["file_size_mb"],
                    "input_shape": result["input_shape"],
                    "output_shape": result["output_shape"]
                })
        
        # 按优先级排序
        model_info["models"].sort(key=lambda x: x["priority"])
        
        # 保存模型信息
        info_path = self.output_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 模型信息已保存: {info_path}")
        return info_path
    
    def convert_all_models(self):
        """转换所有模型"""
        logger.info("🚀 开始批量转换模型为ONNX格式...")
        
        conversion_results = []
        successful_conversions = 0
        
        # 按优先级排序转换
        sorted_models = sorted(
            self.model_configs.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for model_name, config in sorted_models:
            logger.info(f"\n📦 正在处理模型: {model_name}")
            logger.info(f"   描述: {config['description']}")
            
            # 加载模型
            model = self.load_model(model_name)
            if model is None:
                conversion_results.append({
                    "model_name": model_name,
                    "success": False,
                    "error": "Failed to load model"
                })
                continue
            
            # 获取专用转换函数
            converter_function = config.get('converter_function', self.convert_to_onnx)
            
            # 转换为ONNX
            result = converter_function(model_name, model)
            conversion_results.append(result)
            
            # 测试ONNX模型
            if result["success"]:
                if self.test_onnx_model(result["onnx_path"]):
                    successful_conversions += 1
                else:
                    result["success"] = False
                    result["error"] = "ONNX model test failed"
        
        # 生成模型信息文件
        info_path = self.generate_model_info(conversion_results)
        
        # 输出转换总结
        logger.info(f"\n🎉 ONNX转换完成!")
        logger.info(f"✅ 成功转换: {successful_conversions}/{len(self.model_configs)} 个模型")
        logger.info(f"📁 输出目录: {self.output_dir}")
        logger.info(f"📋 模型信息: {info_path}")
        
        return conversion_results

def main():
    """主函数"""
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='将PyTorch模型转换为ONNX格式')
        parser.add_argument('--model', type=str, help='要转换的特定模型名称，不指定则转换所有模型')
        parser.add_argument('--list', action='store_true', help='列出所有可用的模型')
        parser.add_argument('--output-dir', type=str, default='deployment/onnx_models', help='ONNX模型输出目录')
        parser.add_argument('--opset', type=int, default=12, help='ONNX opset版本')
        args = parser.parse_args()
        
        # 检查依赖
        try:
            import onnx
            import onnxruntime
        except ImportError:
            logger.error("❌ 缺少依赖包，请安装: pip install onnx onnxruntime")
            return
        
        # 创建转换器
        converter = ONNXConverter()
        
        # 如果指定了--list参数，列出所有可用模型
        if args.list:
            logger.info("\n📋 可用模型列表:")
            for model_name, config in sorted(converter.model_configs.items(), key=lambda x: x[1]["priority"]):
                logger.info(f"   {model_name}: {config['description']}")
            return
        
        # 如果指定了特定模型，只转换该模型
        if args.model:
            if args.model not in converter.model_configs:
                logger.error(f"❌ 未知模型: {args.model}")
                logger.info(f"可用模型: {', '.join(converter.model_configs.keys())}")
                return
            
            logger.info(f"🚀 开始转换单个模型: {args.model}")
            success = converter.convert_single_model(args.model)
            
            if success:
                logger.info(f"\n🎉 模型 {args.model} 转换成功!")
                logger.info(f"📁 输出目录: {converter.output_dir}")
            else:
                logger.error(f"\n❌ 模型 {args.model} 转换失败!")
            
            return
        
        # 否则转换所有模型
        results = converter.convert_all_models()
        
        # 输出成功的模型列表
        successful_models = [r for r in results if r["success"]]
        if successful_models:
            logger.info("\n🏆 成功转换的模型:")
            for model in successful_models:
                logger.info(f"   ✅ {model['model_name']} ({model['file_size_mb']} MB)")
        
        # 输出失败的模型列表
        failed_models = [r for r in results if not r["success"]]
        if failed_models:
            logger.info("\n❌ 转换失败的模型:")
            for model in failed_models:
                logger.info(f"   ❌ {model['model_name']}: {model.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()