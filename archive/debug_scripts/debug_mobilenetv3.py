#!/usr/bin/env python3
"""
调试MobileNetV3的特征维度
"""

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

try:
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    MOBILENET_AVAILABLE = True
except ImportError:
    MOBILENET_AVAILABLE = False
    print("警告: MobileNetV3 不可用")

def debug_mobilenetv3():
    """调试MobileNetV3的特征输出"""
    if not MOBILENET_AVAILABLE:
        print("MobileNetV3不可用")
        return
    
    # 创建原始MobileNetV3
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    model = mobilenet_v3_large(weights=weights)
    
    # 修改第一层以适应灰度图输入
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )
    
    # 移除分类头
    model.classifier = nn.Identity()
    
    # 测试不同输入尺寸
    test_inputs = [
        (1, 1, 70, 70),   # 单张70x70灰度图
        (2, 1, 70, 70),   # 批次为2
        (1, 1, 224, 224), # 标准ImageNet尺寸
    ]
    
    model.eval()
    with torch.no_grad():
        for input_shape in test_inputs:
            print(f"\n输入形状: {input_shape}")
            x = torch.randn(input_shape)
            
            try:
                # 逐层检查特征
                features = x
                for i, layer in enumerate(model.features):
                    features = layer(features)
                    if i < 5 or i % 3 == 0:  # 打印前几层和每3层
                        print(f"  Layer {i}: {features.shape}")
                
                # 最终特征
                print(f"  最终特征形状: {features.shape}")
                
                # 全局平均池化后
                pooled = nn.AdaptiveAvgPool2d(1)(features)
                flattened = torch.flatten(pooled, 1)
                print(f"  池化后形状: {flattened.shape}")
                
            except Exception as e:
                print(f"  错误: {e}")

if __name__ == "__main__":
    debug_mobilenetv3()