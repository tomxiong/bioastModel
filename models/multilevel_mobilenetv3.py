#!/usr/bin/env python3
"""
Multi-level MobileNetV3 for Bacterial Image Classification
基于MobileNetV3的多层分类模型，用于细菌图像的四层分类任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
from .mobilenet_v3 import create_mobilenetv3_small, create_mobilenetv3_large

class MultiLevelMobileNetV3(nn.Module):
    """
    Multi-level classification model based on MobileNetV3
    
    Four-level classification hierarchy:
    1. microbe_type (currently only bacteria - can be skipped)
    2. growth_level (positive/negative - 2 classes)
    3. growth_pattern (12 classes based on growth_level)
    4. interference_factors (multi-label - 5 classes)
    """
    
    def __init__(self, 
                 model_size: str = 'small',
                 input_channels: int = 1,  # 灰度图像
                 dropout_rate: float = 0.3,
                 use_hierarchical_loss: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        
        self.model_size = model_size
        self.input_channels = input_channels
        self.use_hierarchical_loss = use_hierarchical_loss
        
        # 创建MobileNetV3 backbone
        if model_size == 'large':
            self.backbone = create_mobilenetv3_large(num_classes=1000)  # 临时创建
            feature_dim = 960
        else:
            self.backbone = create_mobilenetv3_small(num_classes=1000)   # 临时创建
            feature_dim = 576
            
        # 修改第一层以适配灰度图像
        if input_channels != 3:
            first_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels, first_conv.out_channels, 
                first_conv.kernel_size, first_conv.stride, 
                first_conv.padding, bias=False
            )
        
        # 移除原始分类器
        self.backbone.classifier = nn.Identity()
        
        # 冻结backbone（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 全局平均池化
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 共享特征处理器
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 分类头定义
        self.num_classes = {
            'growth_level': 2,      # positive/negative
            'growth_pattern': 12,   # 12种生长模式
            'interference_factors': 4  # 4种干扰因子（多标签）- 修正为实际数量
        }
        
        # 任务特定的分类头
        self.classifiers = nn.ModuleDict({
            'growth_level': self._create_classifier(512, self.num_classes['growth_level'], dropout_rate),
            'growth_pattern': self._create_classifier(512, self.num_classes['growth_pattern'], dropout_rate),
            'interference_factors': self._create_classifier(512, self.num_classes['interference_factors'], dropout_rate)
        })
        
        # 层次化权重（用于损失函数）- 优化版权重配置
        self.task_weights = {
            'growth_level': 1.0,
            'growth_pattern': 1.0,  # 提高权重以改善分类性能
            'interference_factors': 0.8  # 适度提高权重
        }
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_classifier(self, input_dim: int, num_classes: int, dropout_rate: float) -> nn.Module:
        """创建分类器"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        # 特征提取 - 正确处理ModuleList
        features = x
        for layer in self.backbone.features:
            features = layer(features)
        
        # 全局平均池化
        features = self.global_avgpool(features)
        features = torch.flatten(features, 1)
        
        # 共享特征处理
        shared_features = self.feature_processor(features)
        
        # 多任务分类
        outputs = {}
        for task_name, classifier in self.classifiers.items():
            outputs[task_name] = classifier(shared_features)
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    custom_criterions: Optional[Dict[str, torch.nn.Module]] = None) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        Args:
            outputs: 模型输出
            targets: 真实标签
            custom_criterions: 自定义损失函数字典 {task_name: criterion}

        Returns:
            Dict containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Growth level loss (二分类)
        if 'growth_level' in outputs and 'growth_level' in targets:
            if custom_criterions and 'growth_level' in custom_criterions:
                loss_fn = custom_criterions['growth_level']
            else:
                loss_fn = nn.CrossEntropyLoss()
            losses['growth_level'] = loss_fn(outputs['growth_level'], targets['growth_level'])
            total_loss += self.task_weights['growth_level'] * losses['growth_level']

        # Growth pattern loss (多分类)
        if 'growth_pattern' in outputs and 'growth_pattern' in targets:
            if custom_criterions and 'growth_pattern' in custom_criterions:
                loss_fn = custom_criterions['growth_pattern']
            else:
                loss_fn = nn.CrossEntropyLoss()
            losses['growth_pattern'] = loss_fn(outputs['growth_pattern'], targets['growth_pattern'])
            total_loss += self.task_weights['growth_pattern'] * losses['growth_pattern']

        # Interference factors loss (多标签)
        if 'interference_factors' in outputs and 'interference_factors' in targets:
            if custom_criterions and 'interference_factors' in custom_criterions:
                loss_fn = custom_criterions['interference_factors']
            else:
                loss_fn = nn.BCEWithLogitsLoss()
            losses['interference_factors'] = loss_fn(outputs['interference_factors'], targets['interference_factors'])
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
            'model_name': f'MultiLevel-MobileNetV3-{self.model_size}',
            'input_size': (self.input_channels, 70, 70),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'task_weights': self.task_weights
        }

def create_multilevel_mobilenetv3(model_size: str = 'small', 
                                 input_channels: int = 1,
                                 **kwargs) -> MultiLevelMobileNetV3:
    """
    创建多层分类MobileNetV3模型
    
    Args:
        model_size: 模型大小 ('small' or 'large')
        input_channels: 输入通道数 (1 for grayscale, 3 for RGB)
        **kwargs: 其他参数
        
    Returns:
        MultiLevelMobileNetV3 model
    """
    return MultiLevelMobileNetV3(
        model_size=model_size,
        input_channels=input_channels,
        **kwargs
    )

if __name__ == "__main__":
    # 测试模型
    model = create_multilevel_mobilenetv3(model_size='small', input_channels=1)
    
    # 打印模型信息
    info = model.get_model_info()
    print("=== Model Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 70, 70)  # batch_size=2, grayscale, 70x70
    outputs = model(x)
    
    print("\n=== Output Shapes ===")
    for task, output in outputs.items():
        print(f"{task}: {output.shape}")
    
    # 测试预测
    predictions = model.predict(x)
    print("\n=== Prediction Shapes ===")
    for task, pred in predictions.items():
        print(f"{task}: {pred.shape}")