#!/usr/bin/env python3
"""
MobileNetV3 多任务学习模型
同时预测生长级别（3分类）和生长模式（4分类）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import math

class MobileNetV3MultiTask(nn.Module):
    """MobileNetV3 多任务学习模型"""
    
    def __init__(self, growth_level_classes: int = 3, growth_pattern_classes: int = 4, 
                 width_mult: float = 1.0, dropout_rate: float = 0.2):
        super().__init__()
        
        self.growth_level_classes = growth_level_classes
        self.growth_pattern_classes = growth_pattern_classes
        
        # 共享的特征提取器（基于MobileNetV3-Small）
        self.backbone = self._create_backbone(width_mult)
        
        # 获取 backbone 的输出通道数
        backbone_output_channels = self._get_backbone_output_channels(width_mult)
        
        # 任务特定的分类头
        self.growth_level_head = self._create_classification_head(
            backbone_output_channels, growth_level_classes, dropout_rate
        )
        
        self.growth_pattern_head = self._create_classification_head(
            backbone_output_channels, growth_pattern_classes, dropout_rate
        )
        
        # 注意力机制（可选）
        self.attention = SEAttention(backbone_output_channels)
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_backbone(self, width_mult: float):
        """创建共享的backbone特征提取器"""
        # 简化的MobileNetV3-Small backbone
        layers = []
        
        # 第一层
        input_channels = int(16 * width_mult)
        layers.extend([
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.Hardswish(inplace=True)
        ])
        
        # MobileNetV3-Small 的配置
        block_configs = [
            # [kernel, exp_channels, out_channels, SE, activation, stride]
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
        
        current_channels = input_channels
        for kernel, exp_channels, out_channels, use_se, activation, stride in block_configs:
            exp_channels = int(exp_channels * width_mult)
            out_channels = int(out_channels * width_mult)
            
            layers.append(
                InvertedResidual(
                    current_channels, out_channels, kernel, stride,
                    exp_channels // current_channels, use_se, activation
                )
            )
            current_channels = out_channels
        
        # 最后的卷积层
        final_exp_channels = int(576 * width_mult)
        final_out_channels = int(96 * width_mult)
        
        layers.extend([
            nn.Conv2d(current_channels, final_exp_channels, 1, bias=False),
            nn.BatchNorm2d(final_exp_channels),
            nn.Hardswish(inplace=True)
        ])
        
        self.backbone_output_channels = final_exp_channels
        
        return nn.Sequential(*layers)
    
    def _get_backbone_output_channels(self, width_mult: float):
        """获取backbone输出通道数"""
        return int(576 * width_mult)
    
    def _create_classification_head(self, input_channels: int, num_classes: int, 
                                   dropout_rate: float):
        """创建分类头"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, input_channels // 2),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_channels // 2, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = self.backbone(x)
        
        # 应用注意力机制
        features = self.attention(features)
        
        # 多任务预测
        growth_level_logits = self.growth_level_head(features)
        growth_pattern_logits = self.growth_pattern_head(features)
        
        return {
            'growth_level': growth_level_logits,
            'growth_pattern': growth_pattern_logits
        }
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class InvertedResidual(nn.Module):
    """Inverted Residual Block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expand_ratio: int, use_se: bool = False,
                 activation='RE'):
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
                self._get_activation(activation)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride,
                     kernel_size // 2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            self._get_activation(activation)
        ])
        
        # SE
        if use_se:
            layers.append(SEAttention(hidden_channels))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def _get_activation(self, activation_type):
        """获取激活函数"""
        if activation_type == 'HS':
            return nn.Hardswish(inplace=True)
        elif activation_type == 'RE':
            return nn.ReLU(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Hardsigmoid(inplace=True)
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def create_multitask_mobilenetv3(growth_level_classes: int = 3, 
                               growth_pattern_classes: int = 4,
                               width_mult: float = 1.0,
                               **kwargs) -> MobileNetV3MultiTask:
    """创建多任务MobileNetV3模型"""
    model = MobileNetV3MultiTask(
        growth_level_classes=growth_level_classes,
        growth_pattern_classes=growth_pattern_classes,
        width_mult=width_mult,
        **kwargs
    )
    return model

def test_multitask_model():
    """测试多任务模型"""
    print("=== 测试多任务MobileNetV3模型 ===")
    
    # 创建模型
    model = create_multitask_mobilenetv3(
        growth_level_classes=3,
        growth_pattern_classes=4,
        width_mult=1.0
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, 3, 70, 70)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"生长级别输出形状: {outputs['growth_level'].shape}")
    print(f"生长模式输出形状: {outputs['growth_pattern'].shape}")
    
    # 测试损失函数
    growth_level_criterion = nn.CrossEntropyLoss()
    growth_pattern_criterion = nn.CrossEntropyLoss()
    
    # 模拟标签
    growth_level_labels = torch.randint(0, 3, (batch_size,))
    growth_pattern_labels = torch.randint(0, 4, (batch_size,))
    
    # 计算损失
    growth_level_loss = growth_level_criterion(outputs['growth_level'], growth_level_labels)
    growth_pattern_loss = growth_pattern_criterion(outputs['growth_pattern'], growth_pattern_labels)
    
    # 组合损失
    total_loss = growth_level_loss + growth_pattern_loss
    
    print(f"生长级别损失: {growth_level_loss:.4f}")
    print(f"生长模式损失: {growth_pattern_loss:.4f}")
    print(f"总损失: {total_loss:.4f}")
    
    # 测试预测
    growth_level_pred = torch.argmax(outputs['growth_level'], dim=1)
    growth_pattern_pred = torch.argmax(outputs['growth_pattern'], dim=1)
    
    print(f"生长级别预测: {growth_level_pred}")
    print(f"生长模式预测: {growth_pattern_pred}")
    
    print("[OK] 多任务模型测试成功")

if __name__ == "__main__":
    test_multitask_model()