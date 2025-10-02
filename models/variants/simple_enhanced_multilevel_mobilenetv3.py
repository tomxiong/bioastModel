#!/usr/bin/env python3
"""
Simple Enhanced Multi-level MobileNetV3 Model
简单增强版多层分类MobileNetV3模型

基于原始multilevel_mobilenetv3.py的简单优化版本，专注于：
1. 增强growth_pattern和interference_factors的权重
2. 添加简单的特征增强
3. 保持原有的损失函数和训练稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multilevel_mobilenetv3 import MultiLevelMobileNetV3


class SimpleTaskWeighting(nn.Module):
    """简单的任务权重调整模块"""
    
    def __init__(self, task_names: list):
        super().__init__()
        self.task_names = task_names
        
        # 为growth_pattern和interference_factors设置更高的基础权重
        base_weights = {}
        for task in task_names:
            if task == 'growth_pattern':
                base_weights[task] = 1.5  # 增加50%权重
            elif task == 'interference_factors':
                base_weights[task] = 1.3  # 增加30%权重
            else:
                base_weights[task] = 1.0
        
        self.task_weights = base_weights
    
    def get_weights(self) -> Dict[str, float]:
        """获取任务权重"""
        return self.task_weights


class FeatureEnhancer(nn.Module):
    """简单的特征增强模块"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 简单的特征增强层
        self.enhancer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 残差连接
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        enhanced = self.enhancer(x)
        # 残差连接，权重较小以保持稳定性
        return x + self.residual_weight * enhanced


class SimpleEnhancedMultiLevelMobileNetV3(MultiLevelMobileNetV3):
    """简单增强版多层分类MobileNetV3模型"""
    
    def __init__(self, 
                 model_size: str = 'small',
                 input_channels: int = 1,  # 默认为灰度图像
                 dropout_rate: float = 0.2,
                 freeze_backbone: bool = False,
                 **kwargs):
        
        # 保存参数
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels
        
        # 初始化父类（父类已经处理了输入通道修改）
        super().__init__(
            model_size=model_size,
            input_channels=input_channels,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            **kwargs
        )
        
        # 获取特征维度（从父类的分类器获取）
        if hasattr(self, 'growth_level_classifier') and hasattr(self.growth_level_classifier[1], 'in_features'):
            feature_dim = self.growth_level_classifier[1].in_features
        else:
            # 默认特征维度
            feature_dim = 576 if model_size == 'small' else 960
        
        # 添加特征增强模块
        self.feature_enhancer = FeatureEnhancer(feature_dim)
        
        # 任务权重模块
        task_names = ['microbe_type', 'growth_level', 'growth_pattern', 'interference_factors']
        self.task_weighting = SimpleTaskWeighting(task_names)
        
        # 重新初始化分类器以使用增强特征
        self._reinit_classifiers(feature_dim)
    
    def _reinit_classifiers(self, feature_dim: int):
        """重新初始化分类器"""
        # 移除原始backbone的分类器
        self.backbone.classifier = nn.Identity()
        
        # 重新定义分类器
        self.microbe_type_classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_dim, 1)  # 只有bacteria一类
        )
        
        self.growth_level_classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_dim, 2)  # positive/negative
        )
        
        self.growth_pattern_classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_dim, 12)  # 12种模式
        )
        
        self.interference_factors_classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_dim, 4)  # 4种干扰因子（多标签）
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 特征提取 - 正确处理ModuleList
        features = x
        for layer in self.backbone.features:
            features = layer(features)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        # 特征增强
        enhanced_features = self.feature_enhancer(features)
        
        # 分类预测
        outputs = {
            'microbe_type': self.microbe_type_classifier(enhanced_features),
            'growth_level': self.growth_level_classifier(enhanced_features),
            'growth_pattern': self.growth_pattern_classifier(enhanced_features),
            'interference_factors': self.interference_factors_classifier(enhanced_features)
        }
        
        return outputs
    
    def compute_loss(self, 
                    outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor],
                    epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算损失（使用简单的权重调整）"""
        
        losses = {}
        task_weights = self.task_weighting.get_weights()
        
        # 计算各任务损失（使用原始的损失函数）
        if 'microbe_type' in outputs and 'microbe_type' in targets:
            losses['microbe_type'] = F.binary_cross_entropy_with_logits(
                outputs['microbe_type'].squeeze(), 
                targets['microbe_type'].float()
            )
        
        if 'growth_level' in outputs and 'growth_level' in targets:
            losses['growth_level'] = F.cross_entropy(
                outputs['growth_level'], 
                targets['growth_level']
            )
        
        if 'growth_pattern' in outputs and 'growth_pattern' in targets:
            losses['growth_pattern'] = F.cross_entropy(
                outputs['growth_pattern'], 
                targets['growth_pattern']
            )
        
        if 'interference_factors' in outputs and 'interference_factors' in targets:
            losses['interference_factors'] = F.binary_cross_entropy_with_logits(
                outputs['interference_factors'], 
                targets['interference_factors'].float()
            )
        
        # 应用任务权重
        weighted_losses = {}
        total_loss = 0
        
        for task_name, loss in losses.items():
            weight = task_weights.get(task_name, 1.0)
            weighted_loss = weight * loss
            weighted_losses[task_name] = weighted_loss
            total_loss += weighted_loss
        
        return total_loss, weighted_losses


def create_simple_enhanced_multilevel_mobilenetv3(
    model_size: str = 'small',
    input_channels: int = 1,  # 默认为灰度图像
    dropout_rate: float = 0.2,
    freeze_backbone: bool = False,
    **kwargs
) -> SimpleEnhancedMultiLevelMobileNetV3:
    """创建简单增强版多层分类MobileNetV3模型"""
    
    model = SimpleEnhancedMultiLevelMobileNetV3(
        model_size=model_size,
        input_channels=input_channels,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_simple_enhanced_multilevel_mobilenetv3()
    
    # 创建测试输入
    x = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    outputs = model(x)
    
    print("模型输出形状:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    # 测试损失计算
    targets = {
        'microbe_type': torch.ones(2),
        'growth_level': torch.randint(0, 2, (2,)),
        'growth_pattern': torch.randint(0, 12, (2,)),
        'interference_factors': torch.randint(0, 2, (2, 4)).float()
    }
    
    total_loss, task_losses = model.compute_loss(outputs, targets)
    print(f"\n总损失: {total_loss.item():.4f}")
    print("任务损失:")
    for task, loss in task_losses.items():
        print(f"  {task}: {loss.item():.4f}")