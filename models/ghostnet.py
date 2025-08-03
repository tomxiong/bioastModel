#!/usr/bin/env python3
"""
GhostNet 模型实现
基于 GhostNet: More Features from Cheap Operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math

class HardSwish(nn.Module):
    """Hard Swish 激活函数"""
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation 模块"""
    
    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)
    
    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return x * scale

class GhostModule(nn.Module):
    """Ghost Module"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 ratio: int = 2, dw_size: int = 3, stride: int = 1, relu: bool = True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                     kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                     groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class GhostBottleneck(nn.Module):
    """Ghost Bottleneck"""
    
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int,
                 kernel_size: int, stride: int, use_se: bool = False):
        super().__init__()
        assert stride in [1, 2]
        
        self.conv = nn.Sequential(
            # pw
            GhostModule(in_channels, hidden_dim, kernel_size=1, relu=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                     (kernel_size - 1) // 2, groups=hidden_dim, bias=False) if stride == 2 else nn.Sequential(),
            nn.BatchNorm2d(hidden_dim) if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            SqueezeExcitation(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, out_channels, kernel_size=1, relu=False),
        )
        
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                         (kernel_size - 1) // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class GhostNet(nn.Module):
    """GhostNet 模型"""
    
    def __init__(self, cfgs: List, num_classes: int = 1000, width: float = 1.0,
                 dropout: float = 0.2):
        super().__init__()
        
        # setting of inverted residual blocks
        self.cfgs = cfgs
        
        # building first layer
        output_channel = int(16 * width)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel
        
        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = int(c * width)
            hidden_channel = int(exp_size * width)
            stages.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.blocks = nn.Sequential(*stages)
        
        # building last several layers
        output_channel = int(exp_size * width)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)
        
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
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_ghostnet(num_classes: int = 1000, width: float = 1.0, **kwargs) -> GhostNet:
    """创建 GhostNet 模型"""
    # GhostNet 配置
    # k, t, c, SE, s
    cfgs = [
        # stage1
        [3,  16,  16, 0, 1],
        # stage2
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        # stage3
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        # stage4
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        # stage5
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]
    
    return GhostNet(cfgs, num_classes=num_classes, width=width)

def create_ghostnet_0_5x(num_classes: int = 1000, **kwargs) -> GhostNet:
    """创建 GhostNet 0.5x 模型"""
    return create_ghostnet(num_classes=num_classes, width=0.5)

def create_ghostnet_1_3x(num_classes: int = 1000, **kwargs) -> GhostNet:
    """创建 GhostNet 1.3x 模型"""
    return create_ghostnet(num_classes=num_classes, width=1.3)

if __name__ == "__main__":
    # 测试模型
    model = create_ghostnet(num_classes=2)
    
    x = torch.randn(1, 3, 224, 224)
    
    print("GhostNet:")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model(x).shape}")