#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import warnings
warnings.filterwarnings('ignore')

def test_efficientnet_output_shape():
    # 创建模型
    model = efficientnet_b0(weights='IMAGENET1K_V1')
    
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
    
    # 测试前向传播
    x = torch.randn(1, 1, 70, 70)  # 批大小1，灰度图，70x70
    
    print("输入形状:", x.shape)
    
    with torch.no_grad():
        output = model(x)
        print("输出形状:", output.shape)
        print("输出特征维度:", output.shape[-1])
        
        # 测试特征处理层
        feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        # 测试features输出
        features = model.features(x)
        print("Features形状:", features.shape)
        
        processed = feature_processor(features)
        print("处理后形状:", processed.shape)

if __name__ == "__main__":
    test_efficientnet_output_shape()