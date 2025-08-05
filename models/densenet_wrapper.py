#!/usr/bin/env python3
"""
DenseNet 包装器，适配70x70输入尺寸的菌落检测任务
"""

import torch
import torch.nn as nn
from .densenet import create_densenet121, create_densenet169

class DenseNetWrapper(nn.Module):
    """DenseNet 适配70x70输入的包装器"""
    
    def __init__(self, num_classes=2, variant='121'):
        super().__init__()
        
        # 根据变体选择模型
        if variant == '121':
            self.backbone = create_densenet121(num_classes=1000)
        elif variant == '169':
            self.backbone = create_densenet169(num_classes=1000)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        # 修改第一个卷积层以适配70x70输入
        # 原始: Conv2d(3, 64, 7, 2, 3) -> 输出35x35 -> MaxPool -> 17x17
        # 修改: Conv2d(3, 64, 5, 1, 2) -> 输出70x70 -> MaxPool -> 35x35
        original_conv0 = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            3, original_conv0.out_channels, 
            kernel_size=5, stride=1, padding=2, bias=False
        )
        
        # 修改第一个池化层
        self.backbone.features.pool0 = nn.MaxPool2d(2, 2, 0)
        
        # 获取原始分类器的输入特征数
        original_classifier = self.backbone.classifier
        in_features = original_classifier.in_features
        
        # 创建新的分类器
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 初始化新添加的层
        self._initialize_new_layers()
    
    def _initialize_new_layers(self):
        """初始化新添加的层"""
        # 初始化修改的conv0
        nn.init.kaiming_normal_(self.backbone.features.conv0.weight, 
                               mode='fan_out', nonlinearity='relu')
        
        # 初始化新的分类器
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
        
        # 通过特征提取器的各个阶段
        x = self.backbone.features.conv0(x)
        x = self.backbone.features.norm0(x)
        x = self.backbone.features.relu0(x)
        features.append(x)  # 初始特征
        
        x = self.backbone.features.pool0(x)
        features.append(x)  # 池化后特征
        
        # Dense blocks
        for name, module in self.backbone.features.named_children():
            if name.startswith('denseblock'):
                x = module(x)
                features.append(x)  # Dense block输出
            elif name.startswith('transition'):
                x = module(x)
                features.append(x)  # Transition输出
            elif name in ['norm5', 'relu5']:
                x = module(x)
        
        features.append(x)  # 最终特征
        
        return features

class DenseNet121(DenseNetWrapper):
    """DenseNet-121 版本"""
    
    def __init__(self, num_classes=2):
        super().__init__(num_classes=num_classes, variant='121')

class DenseNet169(DenseNetWrapper):
    """DenseNet-169 版本（更深）"""
    
    def __init__(self, num_classes=2):
        super().__init__(num_classes=num_classes, variant='169')

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    models = {
        'DenseNet-121': DenseNet121(num_classes=2),
        'DenseNet-169': DenseNet169(num_classes=2)
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