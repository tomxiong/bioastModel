#!/usr/bin/env python3
"""
Multi-level MobileNetV4 for Bacterial Image Classification
基于MobileNetV4的多层分类模型，用于细菌图像的多级分类任务

参考: https://github.com/lars-uav/LARS-MobileNet-V4
改进: 结合 multilevel_mobilenetv3 的架构和训练经验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


# ============================================================================
# Attention Mechanisms (from LARS-MobileNet-V4)
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3))
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class ECABlock(nn.Module):
    """Efficient Channel Attention Block"""
    def __init__(self, in_channels: int, gamma: int = 2, b: int = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log2(in_channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


# ============================================================================
# MobileNetV4 Building Blocks
# ============================================================================

class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, groups: int = 1,
                 activation: str = 'relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'relu6':
            self.act = nn.ReLU6(inplace=True)
        elif activation == 'hardswish':
            self.act = nn.Hardswish(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UniversalInvertedBottleneck(nn.Module):
    """
    Universal Inverted Bottleneck (UIB) - MobileNetV4 的核心模块
    结合了 Inverted Residual 和多种注意力机制
    """
    def __init__(self, in_channels: int, out_channels: int,
                 expand_ratio: int = 4, stride: int = 1,
                 use_se: bool = True, use_eca: bool = False,
                 activation: str = 'relu'):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion phase
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim,
                                   kernel_size=1, padding=0,
                                   activation=activation))

        # Depthwise convolution
        layers.append(ConvBNAct(hidden_dim, hidden_dim,
                               kernel_size=3, stride=stride,
                               groups=hidden_dim, activation=activation))

        # Attention mechanisms
        if use_se:
            layers.append(SEBlock(hidden_dim))
        if use_eca:
            layers.append(ECABlock(hidden_dim))

        # Projection phase
        layers.append(ConvBNAct(hidden_dim, out_channels,
                               kernel_size=1, padding=0,
                               activation='none'))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)


# ============================================================================
# MobileNetV4 Backbone
# ============================================================================

class MobileNetV4Backbone(nn.Module):
    """
    MobileNetV4 Backbone
    针对 70x70 小图像优化的轻量级架构
    """
    def __init__(self, input_channels: int = 1, width_mult: float = 1.0):
        super().__init__()

        # 为70x70图像设计的配置
        # [expand_ratio, out_channels, num_blocks, stride, use_se, use_eca]
        self.cfgs = [
            # Stage 1
            [1, 32, 1, 1, False, False],   # 70x70 -> 70x70
            # Stage 2
            [4, 48, 2, 2, True, False],    # 70x70 -> 35x35
            # Stage 3
            [4, 64, 3, 2, True, False],    # 35x35 -> 18x18
            # Stage 4
            [4, 96, 3, 2, True, True],     # 18x18 -> 9x9
            # Stage 5
            [6, 128, 2, 1, True, True],    # 9x9 -> 9x9
        ]

        # 应用宽度倍数
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # First convolution layer
        input_channel = _make_divisible(32 * width_mult)
        self.stem = ConvBNAct(input_channels, input_channel,
                             kernel_size=3, stride=1,
                             activation='hardswish')

        # Build inverted residual blocks
        layers = []
        for expand_ratio, out_ch, num_blocks, stride, use_se, use_eca in self.cfgs:
            output_channel = _make_divisible(out_ch * width_mult)
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(
                    UniversalInvertedBottleneck(
                        input_channel, output_channel,
                        expand_ratio, s, use_se, use_eca,
                        activation='hardswish'
                    )
                )
                input_channel = output_channel

        self.features = nn.Sequential(*layers)
        self.feature_dim = input_channel

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return x


# ============================================================================
# Multi-level Classification Model
# ============================================================================

class MultiLevelMobileNetV4(nn.Module):
    """
    Multi-level classification model based on MobileNetV4

    Three-level classification hierarchy (基于改进版经验):
    1. growth_level (positive/negative - 2 classes)
    2. growth_pattern (10 classes based on growth characteristics)
    3. interference_factors (multi-label - 4 classes)
    """

    def __init__(self,
                 model_size: str = 'small',  # 'small' or 'medium' or 'large'
                 input_channels: int = 1,
                 dropout_rate: float = 0.3,
                 use_hierarchical_loss: bool = True):
        super().__init__()

        self.model_size = model_size
        self.input_channels = input_channels
        self.use_hierarchical_loss = use_hierarchical_loss

        # Width multipliers for different model sizes
        width_mults = {
            'small': 0.75,
            'medium': 1.0,
            'large': 1.25
        }
        width_mult = width_mults.get(model_size, 1.0)

        # Create MobileNetV4 backbone
        self.backbone = MobileNetV4Backbone(
            input_channels=input_channels,
            width_mult=width_mult
        )

        feature_dim = self.backbone.feature_dim

        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Shared feature processor (基于改进版的成功配置)
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        # Task definitions (基于改进版的最佳配置)
        self.num_classes = {
            'growth_level': 2,      # positive/negative
            'growth_pattern': 10,   # 10种生长模式
            'interference_factors': 4  # 4种干扰因子（多标签）
        }

        # Task-specific classifiers
        self.classifiers = nn.ModuleDict({
            'growth_level': self._create_classifier(
                512, self.num_classes['growth_level'], dropout_rate
            ),
            'growth_pattern': self._create_classifier(
                512, self.num_classes['growth_pattern'], dropout_rate
            ),
            'interference_factors': self._create_classifier(
                512, self.num_classes['interference_factors'], dropout_rate
            )
        })

        # Task weights (基于改进版的成功经验: 统一权重效果最好)
        self.task_weights = {
            'growth_level': 1.0,
            'growth_pattern': 1.0,
            'interference_factors': 1.0
        }

        # Initialize weights
        self._initialize_weights()

    def _create_classifier(self, input_dim: int, num_classes: int,
                          dropout_rate: float) -> nn.Module:
        """创建任务特定的分类器"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [batch_size, 1, 70, 70]

        Returns:
            Dict containing logits for each classification task
        """
        # Feature extraction
        features = self.backbone(x)

        # Global average pooling
        features = self.global_avgpool(features)
        features = torch.flatten(features, 1)

        # Shared feature processing
        shared_features = self.feature_processor(features)

        # Multi-task classification
        outputs = {}
        for task_name, classifier in self.classifiers.items():
            outputs[task_name] = classifier(shared_features)

        return outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        Args:
            outputs: 模型输出的logits字典
            targets: 目标标签字典

        Returns:
            包含各任务损失和总损失的字典
        """
        losses = {}
        total_loss = 0.0

        # Growth level classification loss (binary)
        if 'growth_level' in outputs and 'growth_level' in targets:
            loss_fn = nn.CrossEntropyLoss()
            losses['growth_level'] = loss_fn(
                outputs['growth_level'],
                targets['growth_level']
            )
            total_loss += self.task_weights['growth_level'] * losses['growth_level']

        # Growth pattern classification loss (multi-class)
        if 'growth_pattern' in outputs and 'growth_pattern' in targets:
            loss_fn = nn.CrossEntropyLoss()
            losses['growth_pattern'] = loss_fn(
                outputs['growth_pattern'],
                targets['growth_pattern']
            )
            total_loss += self.task_weights['growth_pattern'] * losses['growth_pattern']

        # Interference factors loss (multi-label)
        if 'interference_factors' in outputs and 'interference_factors' in targets:
            loss_fn = nn.BCEWithLogitsLoss()
            losses['interference_factors'] = loss_fn(
                outputs['interference_factors'],
                targets['interference_factors'].float()
            )
            total_loss += self.task_weights['interference_factors'] * losses['interference_factors']

        losses['total'] = total_loss
        return losses

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测函数

        Args:
            x: 输入图像

        Returns:
            Dict containing predictions for each task
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

            predictions = {}

            # Growth level prediction
            predictions['growth_level'] = torch.softmax(outputs['growth_level'], dim=1)

            # Growth pattern prediction
            predictions['growth_pattern'] = torch.softmax(outputs['growth_pattern'], dim=1)

            # Interference factors prediction (multi-label)
            predictions['interference_factors'] = torch.sigmoid(outputs['interference_factors'])

            return predictions

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': f'MultiLevelMobileNetV4-{self.model_size}',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'task_weights': self.task_weights
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_multilevel_mobilenetv4_small(input_channels: int = 1,
                                       dropout_rate: float = 0.3,
                                       **kwargs) -> MultiLevelMobileNetV4:
    """创建小型 MobileNetV4 多级分类模型"""
    return MultiLevelMobileNetV4(
        model_size='small',
        input_channels=input_channels,
        dropout_rate=dropout_rate,
        **kwargs
    )


def create_multilevel_mobilenetv4_medium(input_channels: int = 1,
                                        dropout_rate: float = 0.3,
                                        **kwargs) -> MultiLevelMobileNetV4:
    """创建中型 MobileNetV4 多级分类模型"""
    return MultiLevelMobileNetV4(
        model_size='medium',
        input_channels=input_channels,
        dropout_rate=dropout_rate,
        **kwargs
    )


def create_multilevel_mobilenetv4_large(input_channels: int = 1,
                                       dropout_rate: float = 0.3,
                                       **kwargs) -> MultiLevelMobileNetV4:
    """创建大型 MobileNetV4 多级分类模型"""
    return MultiLevelMobileNetV4(
        model_size='large',
        input_channels=input_channels,
        dropout_rate=dropout_rate,
        **kwargs
    )


# ============================================================================
# Model Configuration
# ============================================================================

MODEL_CONFIG = {
    'model_name': 'MultiLevelMobileNetV4',
    'input_size': (1, 70, 70),  # 灰度图像, 70x70
    'num_classes': {
        'growth_level': 2,
        'growth_pattern': 10,
        'interference_factors': 4
    },
    'architecture': 'MobileNetV4-based Multi-task Learning',
    'features': [
        'Universal Inverted Bottleneck (UIB)',
        'SE and ECA attention mechanisms',
        'Multi-task learning architecture',
        'Optimized for 70x70 images',
        'Lightweight and efficient'
    ],
    'recommended_config': {
        'batch_size': 64,
        'learning_rate': 0.002,
        'weight_decay': 0.01,
        'num_epochs': 20,
        'patience': 10,
        'warmup_epochs': 5,
        'dropout_rate': 0.3,
        'task_weights': {
            'growth_level': 1.0,
            'growth_pattern': 1.0,
            'interference_factors': 1.0
        }
    }
}


if __name__ == '__main__':
    # 测试模型
    print("Testing MultiLevelMobileNetV4...")

    # 创建模型
    model_small = create_multilevel_mobilenetv4_small()
    model_medium = create_multilevel_mobilenetv4_medium()
    model_large = create_multilevel_mobilenetv4_large()

    # 测试输入
    x = torch.randn(2, 1, 70, 70)

    # 前向传播
    for name, model in [('Small', model_small),
                        ('Medium', model_medium),
                        ('Large', model_large)]:
        outputs = model(x)
        info = model.get_model_info()

        print(f"\n{name} Model:")
        print(f"  Total Parameters: {info['total_parameters']:,}")
        print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
        print("  Output shapes:")
        for task, output in outputs.items():
            print(f"    {task}: {output.shape}")

    print("\n✓ Model test completed successfully!")
