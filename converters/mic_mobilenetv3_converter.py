"""
MIC MobileNetV3模型转换器
"""

import os
import sys
import logging
import torch
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.onnx_converter_base import ONNXConverterBase
from models.mic_mobilenetv3 import create_mic_mobilenetv3

class MICMobileNetV3Converter(ONNXConverterBase):
    """MIC MobileNetV3模型转换器"""
    
    def __init__(self):
        """初始化转换器"""
        super().__init__("mic_mobilenetv3")
        self.input_shape = (3, 70, 70)  # 模型特定的输入形状
    
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
        
        # 创建模型实例
        try:
            model = create_mic_mobilenetv3(
                num_classes=2,
                model_size='small',
                enable_bubble_detection=True,
                enable_turbidity_analysis=True
            )
            model.eval()
            
            # 加载模型权重
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            
            # 检查权重键名是否匹配
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # 尝试直接加载
                state_dict = checkpoint
            
            # 加载权重
            model.load_state_dict(state_dict)
            
            logging.info("模型权重加载成功")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            return False
        
        # 创建包装器模型，只输出主分类结果
        class MICMobileNetV3Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                outputs = self.model(x)
                # 只返回分类结果，简化ONNX模型
                return outputs['classification']
        
        wrapped_model = MICMobileNetV3Wrapper(model)
        wrapped_model.eval()
        
        # 转换为ONNX格式
        success = self.convert_to_onnx(
            wrapped_model, 
            self.input_shape, 
            opset_version=12,
            dynamic_axes=True
        )
        
        if not success:
            # 尝试使用更低的opset版本
            logging.info("尝试使用opset版本11进行转换...")
            success = self.convert_to_onnx(
                wrapped_model, 
                self.input_shape, 
                opset_version=11,
                dynamic_axes=False  # 禁用动态轴
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
    converter = MICMobileNetV3Converter()
    success = converter.convert()
    
    if success:
        logging.info(f"MIC MobileNetV3模型已成功转换为ONNX格式")
    else:
        logging.error(f"MIC MobileNetV3模型转换失败")

if __name__ == "__main__":
    main()