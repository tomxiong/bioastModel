#!/usr/bin/env python3
"""
Optimized Multi-level MobileNetV3 for Bacterial Image Classification
优化版基于MobileNetV3的多层分类模型，专门针对growth_pattern和interference_factors进行优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
from .mobilenet_v3 import create_mobilenetv3_small, create_mobilenetv3_large

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class TaskSpecificAttention(nn.Module):
    """Task-specific attention mechanism"""
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Multi-head attention
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim)
        
        return self.out_proj(attn_output)

class AdaptiveTaskWeighting(nn.Module):
    """Adaptive task weighting based on training progress"""
    def __init__(self, num_tasks: int, initial_weights: Dict[str, float]):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = list(initial_weights.keys())
        
        # Initialize learnable weights
        initial_values = torch.tensor([initial_weights[name] for name in self.task_names])
        self.task_weights = nn.Parameter(initial_values)
        
        # Temperature parameter for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, task_losses: Dict[str, torch.Tensor], epoch: int = 0):
        """Compute adaptive weights based on task losses"""
        # Convert losses to tensor (create new tensors to avoid in-place operations)
        loss_values = torch.stack([task_losses[name].clone() for name in self.task_names])
        
        # Compute adaptive weights using softmax with temperature
        adaptive_weights = F.softmax(self.task_weights / self.temperature, dim=0)
        
        # Apply epoch-based scaling for growth_pattern and interference_factors
        if epoch > 0:
            # Create a new tensor for weights to avoid in-place operations
            scaled_weights = adaptive_weights.clone()
            # Increase weights for growth_pattern and interference_factors over time
            for i, task_name in enumerate(self.task_names):
                if task_name in ['growth_pattern', 'interference_factors']:
                    # Gradually increase weight up to 1.5x original
                    scale_factor = 1.0 + 0.5 * min(epoch / 50.0, 1.0)
                    scaled_weights[i] = scaled_weights[i] * scale_factor
            adaptive_weights = scaled_weights
        
        # Normalize weights
        adaptive_weights = adaptive_weights / adaptive_weights.sum()
        
        return {name: weight for name, weight in zip(self.task_names, adaptive_weights)}

class OptimizedMultiLevelMobileNetV3(nn.Module):
    """
    Optimized Multi-level classification model based on MobileNetV3
    专门针对growth_pattern和interference_factors进行优化
    """
    
    def __init__(self, 
                 model_size: str = 'small',
                 input_channels: int = 1,
                 dropout_rate: float = 0.3,
                 use_hierarchical_loss: bool = True,
                 freeze_backbone: bool = False,
                 use_focal_loss: bool = True,
                 use_asymmetric_loss: bool = True,
                 use_task_attention: bool = True,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.model_size = model_size
        self.input_channels = input_channels
        self.use_hierarchical_loss = use_hierarchical_loss
        self.use_focal_loss = use_focal_loss
        self.use_asymmetric_loss = use_asymmetric_loss
        self.use_task_attention = use_task_attention
        
        # 创建MobileNetV3 backbone
        if model_size == 'large':
            self.backbone = create_mobilenetv3_large(num_classes=1000)
            feature_dim = 960
        else:
            self.backbone = create_mobilenetv3_small(num_classes=1000)
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
        
        # 增强的特征处理器 - 双层设计
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 分类头定义
        self.num_classes = {
            'growth_level': 2,      # positive/negative
            'growth_pattern': 10,   # 修正为10种生长模式
            'interference_factors': 4  # 4种干扰因子（多标签）
        }
        
        # 任务特定的注意力机制
        if use_task_attention:
            self.task_attentions = nn.ModuleDict({
                'growth_pattern': TaskSpecificAttention(512, num_heads=8),
                'interference_factors': TaskSpecificAttention(512, num_heads=8)
            })
        
        # 任务特定的分类头 - 增强版
        self.classifiers = nn.ModuleDict({
            'growth_level': self._create_enhanced_classifier(512, self.num_classes['growth_level'], dropout_rate),
            'growth_pattern': self._create_enhanced_classifier(512, self.num_classes['growth_pattern'], dropout_rate),
            'interference_factors': self._create_enhanced_classifier(512, self.num_classes['interference_factors'], dropout_rate)
        })
        
        # 自适应任务权重
        initial_weights = {
            'growth_level': 1.0,
            'growth_pattern': 1.2,  # 提高growth_pattern权重
            'interference_factors': 1.1  # 提高interference_factors权重
        }
        self.adaptive_weighting = AdaptiveTaskWeighting(len(initial_weights), initial_weights)
        
        # 损失函数
        self.loss_functions = self._create_loss_functions(focal_alpha, focal_gamma)
        
        # 类别权重（将在训练时设置）
        self.class_weights = {}
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_enhanced_classifier(self, input_dim: int, num_classes: int, dropout_rate: float) -> nn.Module:
        """创建增强的分类器"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
    
    def _create_loss_functions(self, focal_alpha: float, focal_gamma: float) -> Dict[str, nn.Module]:
        """创建损失函数"""
        loss_functions = {}
        
        # Growth level - 标准交叉熵
        loss_functions['growth_level'] = nn.CrossEntropyLoss()
        
        # Growth pattern - Focal Loss
        if self.use_focal_loss:
            loss_functions['growth_pattern'] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            loss_functions['growth_pattern'] = nn.CrossEntropyLoss()
        
        # Interference factors - Asymmetric Loss
        if self.use_asymmetric_loss:
            loss_functions['interference_factors'] = AsymmetricLoss(gamma_neg=4, gamma_pos=1)
        else:
            loss_functions['interference_factors'] = nn.BCEWithLogitsLoss()
        
        return loss_functions
    
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
    
    def set_class_weights(self, class_weights: Dict[str, torch.Tensor]):
        """设置类别权重"""
        self.class_weights = class_weights
        
        # 更新损失函数的权重
        if 'growth_pattern' in class_weights and hasattr(self.loss_functions['growth_pattern'], 'weight'):
            self.loss_functions['growth_pattern'].weight = class_weights['growth_pattern']
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 特征提取
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
            # 应用任务特定注意力
            if self.use_task_attention and task_name in self.task_attentions:
                # 为注意力机制添加序列维度
                task_features = shared_features.unsqueeze(1)  # [batch, 1, feature_dim]
                attended_features = self.task_attentions[task_name](task_features)
                attended_features = attended_features.squeeze(1)  # [batch, feature_dim]
                
                # 残差连接
                task_features = shared_features + attended_features
            else:
                task_features = shared_features
            
            outputs[task_name] = classifier(task_features)
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor],
                    epoch: int = 0) -> Dict[str, torch.Tensor]:
        """计算优化的多任务损失"""
        losses = {}
        task_losses = {}
        
        # 计算各任务损失
        for task_name in outputs.keys():
            if task_name in targets:
                loss_fn = self.loss_functions[task_name]
                
                if task_name == 'growth_pattern' and 'growth_pattern' in self.class_weights:
                    # 使用类别权重
                    if isinstance(loss_fn, FocalLoss):
                        loss_fn.weight = self.class_weights['growth_pattern']
                    losses[task_name] = loss_fn(outputs[task_name], targets[task_name])
                else:
                    losses[task_name] = loss_fn(outputs[task_name], targets[task_name])
                
                task_losses[task_name] = losses[task_name].detach()
        
        # 计算自适应权重
        adaptive_weights = self.adaptive_weighting(task_losses, epoch)
        
        # 计算加权总损失
        total_loss = 0.0
        # 创建损失字典的副本以避免在迭代时修改
        loss_items = list(losses.items())
        for task_name, loss in loss_items:
            weighted_loss = adaptive_weights[task_name] * loss
            losses[f'{task_name}_weighted'] = weighted_loss
            total_loss += weighted_loss
        
        losses['total'] = total_loss
        losses['weights'] = adaptive_weights
        
        return losses
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """预测函数"""
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
            'model_name': f'Optimized-MultiLevel-MobileNetV3-{self.model_size}',
            'input_size': (self.input_channels, 70, 70),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'optimizations': {
                'focal_loss': self.use_focal_loss,
                'asymmetric_loss': self.use_asymmetric_loss,
                'task_attention': self.use_task_attention,
                'adaptive_weighting': True
            }
        }

def create_optimized_multilevel_mobilenetv3(model_size: str = 'small', 
                                          input_channels: int = 1,
                                          **kwargs) -> OptimizedMultiLevelMobileNetV3:
    """
    创建优化版多层分类MobileNetV3模型
    
    Args:
        model_size: 模型大小 ('small' or 'large')
        input_channels: 输入通道数 (1 for grayscale, 3 for RGB)
        **kwargs: 其他参数
        
    Returns:
        OptimizedMultiLevelMobileNetV3 model
    """
    return OptimizedMultiLevelMobileNetV3(
        model_size=model_size,
        input_channels=input_channels,
        **kwargs
    )

if __name__ == "__main__":
    # 测试模型
    model = create_optimized_multilevel_mobilenetv3(model_size='small', input_channels=1)
    
    # 打印模型信息
    info = model.get_model_info()
    print("=== Optimized Model Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 测试前向传播
    x = torch.randn(2, 1, 70, 70)
    outputs = model(x)
    
    print("\n=== Output Shapes ===")
    for task, output in outputs.items():
        print(f"{task}: {output.shape}")
    
    # 测试损失计算
    targets = {
        'growth_level': torch.randint(0, 2, (2,)),
        'growth_pattern': torch.randint(0, 10, (2,)),
        'interference_factors': torch.rand(2, 4)
    }
    
    losses = model.compute_loss(outputs, targets, epoch=10)
    print("\n=== Loss Information ===")
    for key, value in losses.items():
        if key != 'weights':
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value}")