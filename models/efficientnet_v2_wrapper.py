#!/usr/bin/env python3
"""
EfficientNet V2 包装器，适配70x70输入尺寸的菌落检测任务
"""

import torch
import torch.nn as nn
from .efficientnet_v2 import create_efficientnetv2_s

class EfficientNetV2S(nn.Module):
    """EfficientNet V2-S 适配70x70输入的包装器"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 创建基础EfficientNet V2-S模型
        self.backbone = create_efficientnetv2_s(num_classes=1000)
        
        # 修改stem层以适配70x70输入
        # 原始stem: Conv2d(3, 24, 3, 2, 1) -> 输出35x35
        # 修改为: Conv2d(3, 24, 3, 1, 1) -> 输出70x70，然后添加池化
        self.backbone.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1, bias=False),  # 保持70x70
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 降采样到35x35
        )
        
        # 替换分类器
        # 获取原始分类器的输入特征数
        original_classifier = self.backbone.classifier
        in_features = original_classifier.in_features
        
        # 创建新的分类器
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
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
        return self.backbone(x)
    
    def get_feature_maps(self, x):
        """获取特征图用于可视化"""
        features = []
        
        # Stem
        x = self.backbone.stem(x)
        features.append(x)
        
        # Blocks
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            # 保存关键层的特征图
            if i in [1, 3, 5, 7]:  # 选择性保存
                features.append(x)
        
        # Head
        x = self.backbone.head_conv(x)
        features.append(x)
        
        return features

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    model = EfficientNetV2S(num_classes=2)
    
    # 测试70x70输入
    x = torch.randn(1, 3, 70, 70)
    
    print("EfficientNet V2-S (70x70适配版):")
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