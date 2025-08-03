"""
将ResNet18-Improved模型转换为ONNX格式
"""

import os
import sys
import torch
import logging
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ResNet18ONNXConverter:
    def __init__(self, output_dir="deployment/onnx_models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_checkpoint_path(self, model_name):
        """获取模型检查点路径"""
        # 根据model_configs.py中的配置，resnet18_improved的实验模式是experiment_20250802_164948
        checkpoint_path = "experiments/experiment_20250802_164948/resnet18_improved/best_model.pth"
        
        if os.path.exists(checkpoint_path):
            return checkpoint_path
            
        # 尝试其他可能的路径
        alt_paths = [
            f"experiments/{model_name}/{model_name}_best.pth",
            f"experiments/{model_name}/best_model.pth",
            f"experiments/experiment_20250802_164948/{model_name}_best.pth"
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                return path
                
        # 如果找不到，返回默认路径
        return checkpoint_path
    
    def load_model(self):
        """加载ResNet18-Improved模型"""
        try:
            # 导入模型定义
            from models.resnet_improved import create_resnet18_improved
            
            # 创建模型实例
            model = create_resnet18_improved(num_classes=2)
            
            # 加载模型权重
            checkpoint_path = self._get_checkpoint_path("resnet18_improved")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 尝试加载模型权重
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                # 如果直接加载失败，尝试处理权重键
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            
            logger.info(f"✅ 成功加载模型权重: {checkpoint_path}")
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {str(e)}")
            return None
    
    def convert_to_onnx(self, model, opset_version=11):
        """将模型转换为ONNX格式"""
        try:
            model_name = "resnet18_improved"
            onnx_path = os.path.join(self.output_dir, f"{model_name}.onnx")
            
            # 准备输入张量
            dummy_input = torch.randn(1, 3, 70, 70, device=self.device)
            
            # 导出为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
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
            
            # 检查文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": onnx_path,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"❌ 转换失败: {str(e)}")
            return None
    
    def test_onnx_model(self, onnx_path):
        """测试ONNX模型"""
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # 创建推理会话
            session = ort.InferenceSession(onnx_path)
            
            # 准备输入数据
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            dummy_input = np.random.randn(1, 3, 70, 70).astype(np.float32)
            
            # 运行推理
            outputs = session.run(None, {input_name: dummy_input})
            output = outputs[0]
            
            logger.info(f"✅ ONNX模型测试成功: {onnx_path}")
            logger.info(f"   输出形状: {output.shape}")
            logger.info(f"   输出范围: [{output.min()} {output.max()}]")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX模型测试失败: {str(e)}")
            return False
    
    def convert(self):
        """执行转换流程"""
        # 加载模型
        model = self.load_model()
        if model is None:
            logger.error("\n❌ 模型加载失败!")
            return False
        
        # 转换为ONNX
        result = self.convert_to_onnx(model)
        if result is None:
            logger.error("\n❌ 模型转换失败!")
            return False
        
        # 测试ONNX模型
        if self.test_onnx_model(result["onnx_path"]):
            logger.info(f"✅ 模型 resnet18_improved 转换并测试成功!")
            return True
        else:
            logger.error("\n❌ 模型测试失败!")
            return False

def main():
    parser = argparse.ArgumentParser(description="将ResNet18-Improved模型转换为ONNX格式")
    parser.add_argument("--output_dir", type=str, default="deployment/onnx_models", 
                        help="ONNX模型输出目录")
    args = parser.parse_args()
    
    logger.info("🚀 开始转换ResNet18-Improved模型")
    
    converter = ResNet18ONNXConverter(output_dir=args.output_dir)
    success = converter.convert()
    
    if success:
        logger.info("\n🎉 模型 resnet18_improved 转换成功!")
        logger.info(f"📁 输出目录: {converter.output_dir}")
    else:
        logger.error("\n❌ 模型 resnet18_improved 转换失败!")

if __name__ == "__main__":
    main()