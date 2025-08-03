#!/usr/bin/env python3
"""
RegNet 模型实现
基于 Designing Network Design Spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
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

class Bottleneck(nn.Module):
    """RegNet Bottleneck Block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 group_width: int = 1, bottleneck_ratio: float = 1.0,
                 se_ratio: float = 0.0):
        super().__init__()
        
        # 计算中间通道数
        mid_channels = int(out_channels * bottleneck_ratio)
        groups = mid_channels // group_width
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1,
                              groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(mid_channels, se_channels)
        else:
            self.se = None
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.se is not None:
            out = self.se(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class RegNet(nn.Module):
    """RegNet 模型"""
    
    def __init__(self, depths: List[int], widths: List[int], group_widths: List[int],
                 bottleneck_ratio: float = 1.0, se_ratio: float = 0.0,
                 num_classes: int = 1000, stem_width: int = 32):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True)
        )
        
        # Stages
        self.stages = nn.ModuleList()
        in_channels = stem_width
        
        for i, (depth, width, group_width) in enumerate(zip(depths, widths, group_widths)):
            stage = []
            
            for j in range(depth):
                stride = 2 if j == 0 and i > 0 else 1
                
                stage.append(
                    Bottleneck(
                        in_channels, width, stride, group_width,
                        bottleneck_ratio, se_ratio
                    )
                )
                in_channels = width
            
            self.stages.append(nn.Sequential(*stage))
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        
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
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def create_regnet_x_400mf(num_classes: int = 1000, **kwargs) -> RegNet:
    """创建 RegNet X-400MF 模型"""
    return RegNet(
        depths=[1, 2, 7, 12],
        widths=[32, 64, 160, 384],
        group_widths=[8, 16, 16, 16],
        bottleneck_ratio=1.0,
        se_ratio=0.0,
        num_classes=num_classes,
        stem_width=32
    )

def create_regnet_y_400mf(num_classes: int = 1000, **kwargs) -> RegNet:
    """创建 RegNet Y-400MF 模型"""
    return RegNet(
        depths=[1, 3, 6, 6],
        widths=[48, 104, 208, 440],
        group_widths=[8, 8, 8, 8],
        bottleneck_ratio=1.0,
        se_ratio=0.25,  # RegNet Y 使用 SE
        num_classes=num_classes,
        stem_width=32
    )

def create_regnet_x_800mf(num_classes: int = 1000, **kwargs) -> RegNet:
    """创建 RegNet X-800MF 模型"""
    return RegNet(
        depths=[1, 3, 7, 5],
        widths=[64, 128, 288, 672],
        group_widths=[16, 16, 16, 16],
        bottleneck_ratio=1.0,
        se_ratio=0.0,
        num_classes=num_classes,
        stem_width=32
    )

def create_regnet_y_800mf(num_classes: int = 1000, **kwargs) -> RegNet:
    """创建 RegNet Y-800MF 模型"""
    return RegNet(
        depths=[1, 3, 8, 2],
        widths=[64, 128, 320, 768],
        group_widths=[16, 16, 16, 16],
        bottleneck_ratio=1.0,
        se_ratio=0.25,
        num_classes=num_classes,
        stem_width=32
    )

if __name__ == "__main__":
    # 测试模型
    model_x400 = create_regnet_x_400mf(num_classes=2)
    model_y400 = create_regnet_y_400mf(num_classes=2)
    
    x = torch.randn(1, 3, 224, 224)
    
    print("RegNet X-400MF:")
    print(f"参数量: {sum(p.numel() for p in model_x400.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_x400(x).shape}")
    
    print("\nRegNet Y-400MF:")
    print(f"参数量: {sum(p.numel() for p in model_y400.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_y400(x).shape}")