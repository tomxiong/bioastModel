#!/usr/bin/env python3
"""
调试模型前向传播问题
检查模型接收到的输入类型和格式
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_multilevel_mobilenetv3 import EnhancedMultiLevelMobileNetV3
from utils.data_loader import create_synthetic_data_loaders

def debug_model_forward():
    """调试模型前向传播"""
    print("🔍 调试模型前向传播...")
    
    # 创建模型
    config = {
        'num_classes': {
            'growth_level': 2,
            'growth_pattern': 12,
            'interference_factors': 4
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedMultiLevelMobileNetV3(
        model_size='small',
        input_channels=1,
        dropout_rate=0.3
    ).to(device)
    
    # 创建数据加载器
    train_loader, _ = create_synthetic_data_loaders(batch_size=4)
    
    # 获取一个批次
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"\n📊 批次 {batch_idx}:")
        print(f"  原始图像类型: {type(images)}")
        print(f"  原始图像形状: {images.shape}")
        print(f"  原始标签类型: {type(labels)}")
        
        # 移动到设备
        images = images.to(device)
        print(f"  移动后图像类型: {type(images)}")
        print(f"  移动后图像形状: {images.shape}")
        
        # 尝试前向传播
        try:
            print("  🚀 尝试前向传播...")
            with torch.no_grad():
                outputs = model(images)
            print(f"  ✅ 前向传播成功!")
            print(f"  输出类型: {type(outputs)}")
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    print(f"    {key}: 形状={value.shape}, 类型={value.dtype}")
        except Exception as e:
            print(f"  ❌ 前向传播失败: {e}")
            print(f"  错误类型: {type(e)}")
            import traceback
            traceback.print_exc()
        
        break  # 只测试第一个批次
    
    print("\n✅ 模型前向传播调试完成")

if __name__ == "__main__":
    debug_model_forward()