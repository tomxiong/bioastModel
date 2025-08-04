"""
ResNet18Improved模型转换器
"""

import os
import sys
import logging
import torch
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.onnx_converter_base import ONNXConverterBase
from models.resnet_improved import ResNetImproved, create_resnet18_improved

class ResNet18ImprovedConverter(ONNXConverterBase):
    """ResNet18Improved模型转换器"""
    
    def __init__(self):
        """初始化转换器"""
        super().__init__("resnet18_improved")
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
            return False
        
        logging.info(f"找到最新的检查点文件: {checkpoint_path}")
        
        # 加载模型配置
        config = self.load_model_config()
        
        if config is None:
            return False
        
        # 创建模型实例
        try:
            model = create_resnet18_improved()
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
        # ResNet18是标准模型，通常不需要特殊处理
        success = self.convert_to_onnx(
            model, 
            self.input_shape, 
            opset_version=11,
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
    converter = ResNet18ImprovedConverter()
    success = converter.convert()
    
    if success:
        logging.info(f"ResNet18Improved模型已成功转换为ONNX格式")
    else:
        logging.error(f"ResNet18Improved模型转换失败")

if __name__ == "__main__":
    main()
