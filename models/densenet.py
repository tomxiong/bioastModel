#!/usr/bin/env python3
"""
DenseNet 模型实现
基于 Densely Connected Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List

class DenseLayer(nn.Module):
    """Dense Layer (BN-ReLU-Conv)"""
    
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4,
                 drop_rate: float = 0.0):
        super().__init__()
        self.drop_rate = drop_rate
        
        # 1x1 conv
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, 1, bias=False)
        
        # 3x3 conv
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, 1, bias=False)
        
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        else:
            self.dropout = None
    
    def forward(self, x):
        # 1x1 conv
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        # 3x3 conv
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        # Concatenate input and output
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """Dense Block"""
    
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int,
                 bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Transition(nn.Module):
    """Transition Layer (BN-ReLU-Conv-Pool)"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    """DenseNet 模型"""
    
    def __init__(self, growth_rate: int = 32, block_config: List[int] = [6, 12, 24, 16],
                 num_init_features: int = 64, bn_size: int = 4, drop_rate: float = 0.0,
                 num_classes: int = 1000, compression: float = 0.5):
        super().__init__()
        
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, 7, 2, 3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(3, 2, 1)),
        ]))
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = Transition(
                    in_channels=num_features,
                    out_channels=int(num_features * compression)
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)
        
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_densenet121(num_classes: int = 1000, **kwargs) -> DenseNet:
    """创建 DenseNet-121 模型"""
    return DenseNet(
        growth_rate=32,
        block_config=[6, 12, 24, 16],
        num_init_features=64,
        num_classes=num_classes
    )

def create_densenet169(num_classes: int = 1000, **kwargs) -> DenseNet:
    """创建 DenseNet-169 模型"""
    return DenseNet(
        growth_rate=32,
        block_config=[6, 12, 32, 32],
        num_init_features=64,
        num_classes=num_classes
    )

def create_densenet201(num_classes: int = 1000, **kwargs) -> DenseNet:
    """创建 DenseNet-201 模型"""
    return DenseNet(
        growth_rate=32,
        block_config=[6, 12, 48, 32],
        num_init_features=64,
        num_classes=num_classes
    )

def create_densenet161(num_classes: int = 1000, **kwargs) -> DenseNet:
    """创建 DenseNet-161 模型"""
    return DenseNet(
        growth_rate=48,
        block_config=[6, 12, 36, 24],
        num_init_features=96,
        num_classes=num_classes
    )

if __name__ == "__main__":
    # 测试模型
    model_121 = create_densenet121(num_classes=2)
    model_169 = create_densenet169(num_classes=2)
    
    x = torch.randn(1, 3, 224, 224)
    
    print("DenseNet-121:")
    print(f"参数量: {sum(p.numel() for p in model_121.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_121(x).shape}")
    
    print("\nDenseNet-169:")
    print(f"参数量: {sum(p.numel() for p in model_169.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_169(x).shape}")