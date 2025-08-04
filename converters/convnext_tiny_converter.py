"""
ConvNextTiny模型转换器
"""

import os
import sys
import logging
import torch
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.onnx_converter_base import ONNXConverterBase
from models.convnext_tiny import ConvNextTiny, create_convnext_tiny

class ConvNextTinyConverter(ONNXConverterBase):
    """ConvNextTiny模型转换器"""
    
    def __init__(self):
        """初始化转换器"""
        super().__init__("convnext_tiny")
        self.input_shape = (3, 224, 224)  # 模型特定的输入形状
    
    def convert(self):
        """转换模型为ONNX格式
        
        Returns:
            是否成功
        """
        logging.info(f"开始将{self.model_name}转换为ONNX格式...")
        
        # 查找最新的检查点文件
        checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is None:
            logging.error(f"未找到{self.model_name}的检查点文件")
            # 尝试使用固定路径
            checkpoint_path = Path("experiments/experiment_20250802_224600/convnext_tiny/best_model.pth")
            if not checkpoint_path.exists():
                logging.error(f"固定路径也未找到检查点文件")
                return False
            logging.info(f"使用固定路径找到检查点文件: {checkpoint_path}")
        else:
            logging.info(f"找到最新的检查点文件: {checkpoint_path}")
        
        # 创建模型实例
        try:
            model = create_convnext_tiny(pretrained=False)
            model.eval()
            
            # 加载模型权重
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            
            # 检查权重键名是否匹配
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # 尝试直接加载
                model.load_state_dict(checkpoint)
            
            logging.info("模型权重加载成功")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            return False
        
        # 转换为ONNX格式
        # ConvNeXt可能需要特殊处理，使用较高的opset版本
        success = self.convert_to_onnx(
            model, 
            self.input_shape, 
            opset_version=12,  # ConvNeXt可能需要更高的opset版本
            dynamic_axes=True
        )
        
        if not success:
            return False
        
        # 验证ONNX模型
        success = self.validate_onnx_model(self.input_shape)
        
        if not success:
            return False
        
        logging.info(f"{self.model_name}已成功转换为ONNX格式并通过验证")
        return True

def main():
    """主函数"""
    converter = ConvNextTinyConverter()
    success = converter.convert()
    
    if success:
        logging.info(f"ConvNextTiny模型已成功转换为ONNX格式")
    else:
        logging.error(f"ConvNextTiny模型转换失败")

if __name__ == "__main__":
    main()
