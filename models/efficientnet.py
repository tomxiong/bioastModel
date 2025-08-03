"""
轻量级EfficientNet实现
针对70x70小图像优化的EfficientNet变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, expand_ratio: int = 6, se_ratio: float = 0.25):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False) if expand_ratio != 1 else None
        self.expand_bn = nn.BatchNorm2d(expanded_channels) if expand_ratio != 1 else None
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze and Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se_reduce = nn.Conv2d(expanded_channels, se_channels, 1)
        self.se_expand = nn.Conv2d(se_channels, expanded_channels, 1)
        
        # Output projection
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = F.silu(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise
        x = F.silu(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze and Excitation
        se_weight = F.adaptive_avg_pool2d(x, 1)
        se_weight = F.silu(self.se_reduce(se_weight))
        se_weight = torch.sigmoid(self.se_expand(se_weight))
        x = x * se_weight
        
        # Output projection
        x = self.project_bn(self.project_conv(x))
        
        # Residual connection
        if self.use_residual:
            x = self.dropout(x) + identity
            
        return x

class EfficientNetCustom(nn.Module):
    """
    针对70x70图像优化的轻量级EfficientNet
    适用于菌落检测的二分类任务
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        
        # Stem
        self.stem_conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        
        # MBConv blocks - 针对小图像优化的配置
        self.blocks = nn.ModuleList([
            # Stage 1: 35x35 -> 35x35
            MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            
            # Stage 2: 35x35 -> 18x18
            MBConvBlock(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            
            # Stage 3: 18x18 -> 9x9
            MBConvBlock(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            
            # Stage 4: 9x9 -> 5x5
            MBConvBlock(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            
            # Stage 5: 5x5 -> 3x3
            MBConvBlock(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            
            # Stage 6: 3x3 -> 2x2
            MBConvBlock(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expand_ratio=6),
        ])
        
        # Head
        self.head_conv = nn.Conv2d(192, 320, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(320)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(320, num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Stem
        x = F.silu(self.stem_bn(self.stem_conv(x)))
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = F.silu(self.head_bn(self.head_conv(x)))
        
        # Classifier
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """获取特征图用于可视化分析"""
        features = []
        
        # Stem
        x = F.silu(self.stem_bn(self.stem_conv(x)))
        features.append(('stem', x.clone()))
        
        # MBConv blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [1, 3, 5, 7, 9, 11]:  # 关键层的特征图
                features.append((f'block_{i}', x.clone()))
        
        # Head
        x = F.silu(self.head_bn(self.head_conv(x)))
        features.append(('head', x.clone()))
        
        return features

def create_efficientnet_b0(num_classes: int = 2, pretrained: bool = False) -> EfficientNetCustom:
    """创建EfficientNet-B0变体"""
    model = EfficientNetCustom(num_classes=num_classes, dropout_rate=0.2)
    
    if pretrained:
        # 这里可以加载预训练权重（如果有的话）
        print("Warning: 预训练权重暂未实现")
    
    return model

def create_efficientnet_b1(num_classes: int = 2, pretrained: bool = False) -> EfficientNetCustom:
    """创建EfficientNet-B1变体（稍大一些）"""
    model = EfficientNetCustom(num_classes=num_classes, dropout_rate=0.3)
    
    if pretrained:
        print("Warning: 预训练权重暂未实现")
    
    return model

if __name__ == "__main__":
    # 测试模型
    model = create_efficientnet_b0(num_classes=2)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"EfficientNet Custom:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 测试前向传播
    x = torch.randn(1, 3, 70, 70)
    with torch.no_grad():
        output = model(x)
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        
        # 测试特征图提取
        features = model.get_feature_maps(x)
        print(f"  特征图数量: {len(features)}")
        for name, feat in features:
            print(f"    {name}: {feat.shape}")