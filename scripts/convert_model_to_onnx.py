"""
通用模型转换脚本 - 将任意模型转换为ONNX格式
支持自定义模型加载和转换逻辑
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
import importlib

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ModelConverter:
    def __init__(self, output_dir="deployment/onnx_models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self, model_path, model_class, model_args=None):
        """
        加载模型
        
        Args:
            model_path: 模型权重文件路径
            model_class: 模型类或创建函数
            model_args: 模型初始化参数
        
        Returns:
            加载的模型
        """
        try:
            # 创建模型实例
            if model_args is None:
                model_args = {"num_classes": 2}
            
            model = model_class(**model_args)
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 尝试加载模型权重
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                # 如果直接加载失败，尝试处理权重键
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            
            logger.info(f"✅ 成功加载模型权重: {model_path}")
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {str(e)}")
            return None
    
    def convert_to_onnx(self, model, output_path, input_shape=(1, 3, 70, 70), opset_version=11):
        """
        将模型转换为ONNX格式
        
        Args:
            model: PyTorch模型
            output_path: ONNX模型输出路径
            input_shape: 输入张量形状
            opset_version: ONNX操作集版本
        
        Returns:
            转换结果
        """
        try:
            # 准备输入张量
            dummy_input = torch.randn(*input_shape, device=self.device)
            
            # 导出为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
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
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换模型 -> {output_path} ({file_size:.2f} MB)")
            
            return {
                "onnx_path": output_path,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"❌ 转换失败: {str(e)}")
            return None
    
    def test_onnx_model(self, onnx_path, input_shape=(1, 3, 70, 70)):
        """
        测试ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            input_shape: 输入张量形状
        
        Returns:
            测试结果
        """
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # 创建推理会话
            session = ort.InferenceSession(onnx_path)
            
            # 准备输入数据
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
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
    
    def convert_model(self, model_name, model_class, model_path, model_args=None, input_shape=(1, 3, 70, 70)):
        """
        执行完整的模型转换流程
        
        Args:
            model_name: 模型名称
            model_class: 模型类或创建函数
            model_path: 模型权重文件路径
            model_args: 模型初始化参数
            input_shape: 输入张量形状
        
        Returns:
            转换结果
        """
        logger.info(f"🚀 开始转换模型: {model_name}")
        
        # 加载模型
        model = self.load_model(model_path, model_class, model_args)
        if model is None:
            logger.error(f"❌ 模型 {model_name} 加载失败!")
            return False
        
        # 转换为ONNX
        onnx_path = os.path.join(self.output_dir, f"{model_name}.onnx")
        result = self.convert_to_onnx(model, onnx_path, input_shape)
        if result is None:
            logger.error(f"❌ 模型 {model_name} 转换失败!")
            return False
        
        # 测试ONNX模型
        if self.test_onnx_model(onnx_path, input_shape):
            logger.info(f"✅ 模型 {model_name} 转换并测试成功!")
            return True
        else:
            logger.error(f"❌ 模型 {model_name} 测试失败!")
            return False

def main():
    parser = argparse.ArgumentParser(description="将模型转换为ONNX格式")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="模型名称")
    parser.add_argument("--model_module", type=str, required=True, 
                        help="模型模块路径 (例如: models.coatnet)")
    parser.add_argument("--model_class", type=str, required=True, 
                        help="模型类名或创建函数名 (例如: CoAtNet 或 create_coatnet)")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="模型权重文件路径")
    parser.add_argument("--output_dir", type=str, default="deployment/onnx_models", 
                        help="ONNX模型输出目录")
    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 3, 70, 70], 
                        help="输入张量形状 (例如: 1 3 70 70)")
    parser.add_argument("--num_classes", type=int, default=2, 
                        help="类别数量")
    args = parser.parse_args()
    
    # 导入模型模块
    try:
        module = importlib.import_module(args.model_module)
        
        # 获取模型类或创建函数
        if hasattr(module, args.model_class):
            model_class = getattr(module, args.model_class)
        else:
            logger.error(f"❌ 未找到模型类或创建函数: {args.model_class}")
            return
        
        # 创建转换器
        converter = ModelConverter(output_dir=args.output_dir)
        
        # 执行转换
        success = converter.convert_model(
            model_name=args.model_name,
            model_class=model_class,
            model_path=args.model_path,
            model_args={"num_classes": args.num_classes},
            input_shape=tuple(args.input_shape)
        )
        
        if success:
            logger.info(f"\n🎉 模型 {args.model_name} 转换成功!")
            logger.info(f"📁 输出目录: {converter.output_dir}")
        else:
            logger.error(f"\n❌ 模型 {args.model_name} 转换失败!")
        
    except Exception as e:
        logger.error(f"❌ 转换过程出错: {str(e)}")

if __name__ == "__main__":
    main()