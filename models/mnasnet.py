#!/usr/bin/env python3
"""
MNASNet 模型实现
基于 MnasNet: Platform-Aware Neural Architecture Search for Mobile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation 模块"""
    
    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale

class InvertedResidual(nn.Module):
    """Inverted Residual Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expansion_factor: int, bn_momentum: float = 0.1):
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_channels = in_channels * expansion_factor
        self.apply_residual = (in_channels == out_channels and stride == 1)
        
        layers = []
        activation = nn.ReLU
        
        # Expansion
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
                activation(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride,
                     kernel_size // 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
            activation(inplace=True)
        ])
        
        # Squeeze and Excitation
        se_channels = max(1, in_channels // 4)
        layers.append(SqueezeExcitation(mid_channels, se_channels))
        
        # Projection
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        ])
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.apply_residual:
            return self.layers(x) + x
        else:
            return self.layers(x)

class MNASNet(nn.Module):
    """MNASNet 模型"""
    
    def __init__(self, alpha: float, num_classes: int = 1000, dropout: float = 0.2):
        super().__init__()
        assert alpha > 0.0
        self.alpha = alpha
        self.num_classes = num_classes
        
        depths = [1, 2, 3, 4, 2, 3, 1]
        layers = [
            # First layer: regular conv
            nn.Conv2d(3, int(32 * alpha), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * alpha)),
            nn.ReLU(inplace=True),
        ]
        
        # Separable conv
        layers.extend([
            nn.Conv2d(int(32 * alpha), int(32 * alpha), 3, stride=1, padding=1,
                     groups=int(32 * alpha), bias=False),
            nn.BatchNorm2d(int(32 * alpha)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(32 * alpha), int(16 * alpha), 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(16 * alpha)),
        ])
        
        # MBConv blocks
        in_channels_group = int(16 * alpha)
        
        # Stage 1: 16 -> 24, stride 2, expansion 6
        for i in range(depths[0]):
            stride = 2 if i == 0 else 1
            out_channels = int(24 * alpha)
            layers.append(InvertedResidual(in_channels_group, out_channels, 3, stride, 6))
            in_channels_group = out_channels
        
        # Stage 2: 24 -> 40, stride 2, expansion 3
        for i in range(depths[1]):
            stride = 2 if i == 0 else 1
            out_channels = int(40 * alpha)
            layers.append(InvertedResidual(in_channels_group, out_channels, 5, stride, 3))
            in_channels_group = out_channels
        
        # Stage 3: 40 -> 80, stride 2, expansion 6
        for i in range(depths[2]):
            stride = 2 if i == 0 else 1
            out_channels = int(80 * alpha)
            layers.append(InvertedResidual(in_channels_group, out_channels, 5, stride, 6))
            in_channels_group = out_channels
        
        # Stage 4: 80 -> 96, stride 1, expansion 6
        for i in range(depths[3]):
            stride = 1
            out_channels = int(96 * alpha)
            layers.append(InvertedResidual(in_channels_group, out_channels, 3, stride, 6))
            in_channels_group = out_channels
        
        # Stage 5: 96 -> 192, stride 2, expansion 6
        for i in range(depths[4]):
            stride = 2 if i == 0 else 1
            out_channels = int(192 * alpha)
            layers.append(InvertedResidual(in_channels_group, out_channels, 5, stride, 6))
            in_channels_group = out_channels
        
        # Stage 6: 192 -> 320, stride 1, expansion 6
        for i in range(depths[5]):
            stride = 1
            out_channels = int(320 * alpha)
            layers.append(InvertedResidual(in_channels_group, out_channels, 3, stride, 6))
            in_channels_group = out_channels
        
        # Final layers
        last_channel = int(1280 * alpha) if alpha > 1.0 else 1280
        layers.extend([
            nn.Conv2d(in_channels_group, last_channel, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(inplace=True),
        ])
        
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

def create_mnasnet_0_5(num_classes: int = 1000, **kwargs) -> MNASNet:
    """创建 MNASNet 0.5 模型"""
    return MNASNet(alpha=0.5, num_classes=num_classes)

def create_mnasnet_0_75(num_classes: int = 1000, **kwargs) -> MNASNet:
    """创建 MNASNet 0.75 模型"""
    return MNASNet(alpha=0.75, num_classes=num_classes)

def create_mnasnet_1_0(num_classes: int = 1000, **kwargs) -> MNASNet:
    """创建 MNASNet 1.0 模型"""
    return MNASNet(alpha=1.0, num_classes=num_classes)

def create_mnasnet_1_3(num_classes: int = 1000, **kwargs) -> MNASNet:
    """创建 MNASNet 1.3 模型"""
    return MNASNet(alpha=1.3, num_classes=num_classes)

if __name__ == "__main__":
    # 测试模型
    model_0_5 = create_mnasnet_0_5(num_classes=2)
    model_1_0 = create_mnasnet_1_0(num_classes=2)
    
    x = torch.randn(1, 3, 224, 224)
    
    print("MNASNet 0.5:")
    print(f"参数量: {sum(p.numel() for p in model_0_5.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_0_5(x).shape}")
    
    print("\nMNASNet 1.0:")
    print(f"参数量: {sum(p.numel() for p in model_1_0.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_1_0(x).shape}")