#!/usr/bin/env python3
"""
EfficientNet V2 模型实现
基于 EfficientNetV2: Smaller Models and Faster Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, expand_ratio: int, se_ratio: float = 0.25,
                 drop_rate: float = 0.0):
        super().__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Point-wise convolution
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout
        if drop_rate > 0:
            self.dropout = nn.Dropout2d(drop_rate)
        else:
            self.dropout = None
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        
        # Squeeze and Excitation
        if self.se is not None:
            x = x * self.se(x)
        
        # Project
        x = self.project_conv(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x

class FusedMBConvBlock(nn.Module):
    """Fused Mobile Inverted Bottleneck Convolution Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expand_ratio: int, se_ratio: float = 0.25,
                 drop_rate: float = 0.0):
        super().__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = in_channels * expand_ratio
        
        # Fused expansion and depthwise
        if expand_ratio != 1:
            self.fused_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size, 
                         stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.fused_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, kernel_size, 
                         stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        
        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Point-wise convolution (if needed)
        if expand_ratio != 1 or in_channels != out_channels:
            self.project_conv = nn.Sequential(
                nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.project_conv = nn.Identity()
        
        # Dropout
        if drop_rate > 0:
            self.dropout = nn.Dropout2d(drop_rate)
        else:
            self.dropout = None
    
    def forward(self, x):
        identity = x
        
        # Fused convolution
        x = self.fused_conv(x)
        
        # Squeeze and Excitation
        if self.se is not None:
            x = x * self.se(x)
        
        # Project
        x = self.project_conv(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x

class EfficientNetV2(nn.Module):
    """EfficientNet V2 模型"""
    
    def __init__(self, block_configs: List, num_classes: int = 1000, 
                 width_mult: float = 1.0, depth_mult: float = 1.0,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        # Stem
        stem_channels = int(24 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True)
        )
        
        # Blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        
        for block_type, kernel_size, stride, expand_ratio, out_channels, num_layers, se_ratio in block_configs:
            out_channels = int(out_channels * width_mult)
            num_layers = int(math.ceil(num_layers * depth_mult))
            
            for i in range(num_layers):
                stride_i = stride if i == 0 else 1
                
                if block_type == 'fused':
                    block = FusedMBConvBlock(
                        in_channels, out_channels, kernel_size, stride_i,
                        expand_ratio, se_ratio
                    )
                else:  # mbconv
                    block = MBConvBlock(
                        in_channels, out_channels, kernel_size, stride_i,
                        expand_ratio, se_ratio
                    )
                
                self.blocks.append(block)
                in_channels = out_channels
        
        # Head
        head_channels = int(1280 * width_mult)
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(head_channels, num_classes)
        
        # Initialize weights
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
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

def create_efficientnetv2_s(num_classes: int = 1000, **kwargs) -> EfficientNetV2:
    """创建 EfficientNet V2-S 模型"""
    # EfficientNet V2-S 配置
    # (block_type, kernel_size, stride, expand_ratio, out_channels, num_layers, se_ratio)
    block_configs = [
        ('fused', 3, 1, 1, 24, 2, 0),
        ('fused', 3, 2, 4, 48, 4, 0),
        ('fused', 3, 2, 4, 64, 4, 0),
        ('mbconv', 3, 2, 4, 128, 6, 0.25),
        ('mbconv', 3, 1, 6, 160, 9, 0.25),
        ('mbconv', 3, 2, 6, 256, 15, 0.25),
    ]
    
    return EfficientNetV2(
        block_configs=block_configs,
        num_classes=num_classes,
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.2
    )

def create_efficientnetv2_m(num_classes: int = 1000, **kwargs) -> EfficientNetV2:
    """创建 EfficientNet V2-M 模型"""
    # EfficientNet V2-M 配置
    # (block_type, kernel_size, stride, expand_ratio, out_channels, num_layers, se_ratio)
    block_configs = [
        ('fused', 3, 1, 1, 24, 3, 0),
        ('fused', 3, 2, 4, 48, 5, 0),
        ('fused', 3, 2, 4, 80, 5, 0),
        ('mbconv', 3, 2, 4, 160, 7, 0.25),
        ('mbconv', 3, 1, 6, 176, 14, 0.25),
        ('mbconv', 3, 2, 6, 304, 18, 0.25),
        ('mbconv', 3, 1, 6, 512, 5, 0.25),
    ]
    
    return EfficientNetV2(
        block_configs=block_configs,
        num_classes=num_classes,
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.3
    )

if __name__ == "__main__":
    # 测试模型
    model_s = create_efficientnetv2_s(num_classes=2)
    model_m = create_efficientnetv2_m(num_classes=2)
    
    x = torch.randn(1, 3, 224, 224)
    
    print("EfficientNet V2-S:")
    print(f"参数量: {sum(p.numel() for p in model_s.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_s(x).shape}")
    
    print("\nEfficientNet V2-M:")
    print(f"参数量: {sum(p.numel() for p in model_m.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_m(x).shape}")