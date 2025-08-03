#!/usr/bin/env python3
"""
MobileNet V3 模型实现
基于 Searching for MobileNetV3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math

class HardSwish(nn.Module):
    """Hard Swish 激活函数"""
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class HardSigmoid(nn.Module):
    """Hard Sigmoid 激活函数"""
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6

class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation 模块"""
    
    def __init__(self, in_channels: int, se_channels: int, activation=nn.ReLU):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.activation1 = activation(inplace=True)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)
        self.activation2 = HardSigmoid()
    
    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation1(scale)
        scale = self.fc2(scale)
        scale = self.activation2(scale)
        return x * scale

class InvertedResidual(nn.Module):
    """Inverted Residual Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expand_ratio: int, use_se: bool = False,
                 activation=nn.ReLU):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_channels = in_channels * expand_ratio
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                activation(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride,
                     kernel_size // 2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
        ])
        
        # SE
        if use_se:
            se_channels = max(1, hidden_channels // 4)
            layers.append(SqueezeExcitation(hidden_channels, se_channels, activation))
        
        layers.append(activation(inplace=True))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    """MobileNet V3 模型"""
    
    def __init__(self, block_configs: List, last_channel: int, num_classes: int = 1000,
                 width_mult: float = 1.0, dropout_rate: float = 0.2):
        super().__init__()
        
        # 第一层
        input_channel = int(16 * width_mult)
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channel),
                HardSwish(inplace=True)
            )
        ])
        
        # Inverted residual blocks
        for kernel_size, exp_size, out_channels, use_se, activation, stride in block_configs:
            output_channel = int(out_channels * width_mult)
            exp_channel = int(exp_size * width_mult)
            
            if activation == 'HS':
                act_layer = HardSwish
            else:
                act_layer = nn.ReLU
            
            self.features.append(
                InvertedResidual(
                    input_channel, output_channel, kernel_size, stride,
                    exp_channel // input_channel, use_se, act_layer
                )
            )
            input_channel = output_channel
        
        # 最后的卷积层
        last_conv_input = input_channel
        last_conv_output = int(exp_size * width_mult)
        self.features.append(
            nn.Sequential(
                nn.Conv2d(last_conv_input, last_conv_output, 1, bias=False),
                nn.BatchNorm2d(last_conv_output),
                HardSwish(inplace=True)
            )
        )
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_output, last_channel),
            HardSwish(inplace=True),
            nn.Dropout(dropout_rate),
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
        for layer in self.features:
            x = layer(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

def create_mobilenetv3_large(num_classes: int = 1000, width_mult: float = 1.0, **kwargs) -> MobileNetV3:
    """创建 MobileNet V3 Large 模型"""
    # MobileNet V3 Large 配置
    # (kernel_size, exp_size, out_channels, use_se, activation, stride)
    block_configs = [
        [3, 16, 16, False, 'RE', 1],
        [3, 64, 24, False, 'RE', 2],
        [3, 72, 24, False, 'RE', 1],
        [5, 72, 40, True, 'RE', 2],
        [5, 120, 40, True, 'RE', 1],
        [5, 120, 40, True, 'RE', 1],
        [3, 240, 80, False, 'HS', 2],
        [3, 200, 80, False, 'HS', 1],
        [3, 184, 80, False, 'HS', 1],
        [3, 184, 80, False, 'HS', 1],
        [3, 480, 112, True, 'HS', 1],
        [3, 672, 112, True, 'HS', 1],
        [5, 672, 160, True, 'HS', 2],
        [5, 960, 160, True, 'HS', 1],
        [5, 960, 160, True, 'HS', 1],
    ]
    
    return MobileNetV3(
        block_configs=block_configs,
        last_channel=1280,
        num_classes=num_classes,
        width_mult=width_mult
    )

def create_mobilenetv3_small(num_classes: int = 1000, width_mult: float = 1.0, **kwargs) -> MobileNetV3:
    """创建 MobileNet V3 Small 模型"""
    # MobileNet V3 Small 配置
    # (kernel_size, exp_size, out_channels, use_se, activation, stride)
    block_configs = [
        [3, 16, 16, True, 'RE', 2],
        [3, 72, 24, False, 'RE', 2],
        [3, 88, 24, False, 'RE', 1],
        [5, 96, 40, True, 'HS', 2],
        [5, 240, 40, True, 'HS', 1],
        [5, 240, 40, True, 'HS', 1],
        [5, 120, 48, True, 'HS', 1],
        [5, 144, 48, True, 'HS', 1],
        [5, 288, 96, True, 'HS', 2],
        [5, 576, 96, True, 'HS', 1],
        [5, 576, 96, True, 'HS', 1],
    ]
    
    return MobileNetV3(
        block_configs=block_configs,
        last_channel=1024,
        num_classes=num_classes,
        width_mult=width_mult
    )

if __name__ == "__main__":
    # 测试模型
    model_large = create_mobilenetv3_large(num_classes=2)
    model_small = create_mobilenetv3_small(num_classes=2)
    
    x = torch.randn(1, 3, 224, 224)
    
    print("MobileNet V3 Large:")
    print(f"参数量: {sum(p.numel() for p in model_large.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_large(x).shape}")
    
    print("\nMobileNet V3 Small:")
    print(f"参数量: {sum(p.numel() for p in model_small.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_small(x).shape}")