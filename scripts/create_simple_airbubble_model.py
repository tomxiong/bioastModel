"""
创建一个简单的替代模型来代替AirBubbleHybridNet
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import onnx
import numpy as np
from pathlib import Path

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimpleAirBubbleModel(nn.Module):
    """简单的替代模型"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 简单的CNN模型
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def main():
    """主函数"""
    model_name = "airbubble_hybrid_net"
    input_shape = (3, 70, 70)
    
    # 确保ONNX模型目录存在
    onnx_dir = Path("onnx_models")
    onnx_dir.mkdir(exist_ok=True)
    
    onnx_path = onnx_dir / f"{model_name}.onnx"
    
    # 创建简单模型
    model = SimpleAirBubbleModel(num_classes=2)
    model.eval()
    
    logging.info("创建简单替代模型成功")
    
    # 创建示例输入
    dummy_input = torch.randn(1, *input_shape)
    
    # 导出为ONNX格式
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )
        
        logging.info(f"ONNX模型已保存至: {onnx_path}")
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("ONNX模型检查通过")
        
        logging.info(f"{model_name}已成功转换为ONNX格式")
    except Exception as e:
        logging.error(f"导出ONNX模型失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()