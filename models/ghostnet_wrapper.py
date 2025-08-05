#!/usr/bin/env python3
"""
GhostNet 包装器，适配70x70输入尺寸的菌落检测任务
"""

import torch
import torch.nn as nn
from .ghostnet import create_ghostnet

class GhostNetWrapper(nn.Module):
    """GhostNet 适配70x70输入的包装器"""
    
    def __init__(self, num_classes=2, width=1.0):
        super().__init__()
        
        # 创建基础GhostNet模型
        self.backbone = create_ghostnet(num_classes=1000, width=width)
        
        # 修改stem层以适配70x70输入
        # 原始stem: Conv2d(3, 16, 3, 2, 1) -> 输出35x35
        # 修改为: Conv2d(3, 16, 3, 1, 1) -> 输出70x70，然后添加池化
        output_channel = int(16 * width)
        self.backbone.conv_stem = nn.Conv2d(3, output_channel, 3, 1, 1, bias=False)
        
        # 添加额外的池化层来降采样
        self.downsample = nn.MaxPool2d(2, 2)
        
        # 获取原始分类器的输入特征数
        original_classifier = self.backbone.classifier
        in_features = original_classifier.in_features
        
        # 创建新的分类器
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 初始化新添加的层
        self._initialize_new_layers()
    
    def _initialize_new_layers(self):
        """初始化新添加的层"""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Stem
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        
        # 添加降采样
        x = self.downsample(x)
        
        # Blocks
        x = self.backbone.blocks(x)
        
        # Head
        x = self.backbone.global_pool(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.act2(x)
        x = x.view(x.size(0), -1)
        x = self.backbone.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """获取特征图用于可视化"""
        features = []
        
        # Stem
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        features.append(x)
        
        # 降采样
        x = self.downsample(x)
        features.append(x)
        
        # Blocks
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            # 保存关键层的特征图
            if i in [1, 3, 6, 10, 14]:  # 选择性保存
                features.append(x)
        
        # Head
        x = self.backbone.global_pool(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.act2(x)
        features.append(x)
        
        return features

class GhostNet05x(GhostNetWrapper):
    """GhostNet 0.5x 版本（轻量级）"""
    
    def __init__(self, num_classes=2):
        super().__init__(num_classes=num_classes, width=0.5)

class GhostNet13x(GhostNetWrapper):
    """GhostNet 1.3x 版本（高性能）"""
    
    def __init__(self, num_classes=2):
        super().__init__(num_classes=num_classes, width=1.3)

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    models = {
        'GhostNet 1.0x': GhostNetWrapper(num_classes=2),
        'GhostNet 0.5x': GhostNet05x(num_classes=2),
        'GhostNet 1.3x': GhostNet13x(num_classes=2)
    }
    
    # 测试70x70输入
    x = torch.randn(1, 3, 70, 70)
    
    for name, model in models.items():
        print(f"\n{name} (70x70适配版):")
        print(f"参数量: {count_parameters(model) / 1e6:.2f}M")
        
        # 前向传播测试
        with torch.no_grad():
            output = model(x)
            print(f"输入形状: {x.shape}")
            print(f"输出形状: {output.shape}")
            
            # 特征图测试
            features = model.get_feature_maps(x)
            print(f"特征图数量: {len(features)}")
            for i, feat in enumerate(features):
                print(f"特征图 {i}: {feat.shape}")