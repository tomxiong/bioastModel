"""
损失函数工厂

提供各种损失函数的统一接口和自定义损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss 用于处理类别不平衡"""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数
            reduction: 归约方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    
    def __init__(self, 
                 num_classes: int,
                 smoothing: float = 0.1,
                 reduction: str = 'mean'):
        """
        Args:
            num_classes: 类别数
            smoothing: 平滑系数
            reduction: 归约方式
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 创建平滑标签
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1)
        smooth_label = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (self.num_classes - 1)
        
        # 计算KL散度
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -torch.sum(smooth_label * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """类别平衡损失"""
    
    def __init__(self,
                 num_classes: int,
                 samples_per_class: List[int],
                 beta: float = 0.9999,
                 loss_type: str = 'focal'):
        """
        Args:
            num_classes: 类别数
            samples_per_class: 每个类别的样本数
            beta: 平衡参数
            loss_type: 基础损失类型
        """
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        
        # 计算有效样本数
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * self.num_classes
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        # 创建基础损失
        if loss_type == 'focal':
            self.base_loss = FocalLoss()
        elif loss_type == 'cross_entropy':
            self.base_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        weights = self.weights.to(device)
        
        # 计算基础损失
        loss = self.base_loss(inputs, targets)
        
        # 应用权重
        weights = weights[targets]
        weighted_loss = loss * weights
        
        return weighted_loss.mean()


class TripletLoss(nn.Module):
    """三元组损失"""
    
    def __init__(self, margin: float = 1.0, distance_metric: str = 'euclidean'):
        """
        Args:
            margin: 边界值
            distance_metric: 距离度量方式
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")
        
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()


class ContrastiveLoss(nn.Module):
    """对比损失"""
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: 边界值
        """
        super().__init__()
        self.margin = margin
        
    def forward(self, 
                output1: torch.Tensor, 
                output2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # 相同类别的损失
        positive_loss = (1 - label) * torch.pow(euclidean_distance, 2)
        
        # 不同类别的损失
        negative_loss = label * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        losses = positive_loss + negative_loss
        return losses.mean()


class ArcFaceLoss(nn.Module):
    """ArcFace 损失"""
    
    def __init__(self, 
                 num_classes: int,
                 embedding_size: int,
                 margin: float = 0.5,
                 scale: float = 64.0):
        """
        Args:
            num_classes: 类别数
            embedding_size: 特征维度
            margin: 角度边界
            scale: 缩放因子
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        
        # 初始化权重
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 归一化
        features = F.normalize(features, dim=-1)
        weights = F.normalize(self.weight, dim=-1)
        
        # 计算余弦相似度
        cosine = F.linear(features, weights)
        
        # 转换为角度
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        
        # 添加角度边界
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        theta = theta + one_hot * self.margin
        
        # 转换回余弦
        cosine = torch.cos(theta)
        
        # 缩放
        logits = cosine * self.scale
        
        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss


class DiceLoss(nn.Module):
    """Dice 损失，常用于分割任务"""
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth: 平滑因子，避免除零
        """
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 转换为概率
        inputs = torch.sigmoid(inputs)
        
        # 展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算交集和并集
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky 损失，Dice损失的泛化"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
        """
        Args:
            alpha: 假阴性的权重
            beta: 假阳性的权重
            smooth: 平滑因子
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        
        # 展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 计算真阳性、假阳性、假阴性
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, 
                 losses: List[Dict[str, Any]],
                 weights: Optional[List[float]] = None):
        """
        Args:
            losses: 损失函数配置列表
            weights: 各损失函数的权重
        """
        super().__init__()
        self.losses = nn.ModuleList()
        self.weights = weights or [1.0 / len(losses)] * len(losses)
        
        for loss_config in losses:
            loss_fn = create_loss(loss_config)
            self.losses.append(loss_fn)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            loss = loss_fn(inputs, targets)
            total_loss += weight * loss
        
        return total_loss


class LossFunctionFactory:
    """损失函数工厂"""
    
    _registry = {
        'cross_entropy': {
            'class': nn.CrossEntropyLoss,
            'default_params': {}
        },
        'bce': {
            'class': nn.BCEWithLogitsLoss,
            'default_params': {}
        },
        'mse': {
            'class': nn.MSELoss,
            'default_params': {}
        },
        'mae': {
            'class': nn.L1Loss,
            'default_params': {}
        },
        'smooth_l1': {
            'class': nn.SmoothL1Loss,
            'default_params': {}
        },
        'huber': {
            'class': nn.HuberLoss,
            'default_params': {}
        },
        'kldiv': {
            'class': nn.KLDivLoss,
            'default_params': {'reduction': 'batchmean'}
        },
        'focal': {
            'class': FocalLoss,
            'default_params': {'alpha': 1.0, 'gamma': 2.0}
        },
        'label_smoothing': {
            'class': LabelSmoothingLoss,
            'default_params': {'smoothing': 0.1}
        },
        'class_balanced': {
            'class': ClassBalancedLoss,
            'default_params': {'beta': 0.9999, 'loss_type': 'cross_entropy'}
        },
        'triplet': {
            'class': TripletLoss,
            'default_params': {'margin': 1.0}
        },
        'contrastive': {
            'class': ContrastiveLoss,
            'default_params': {'margin': 1.0}
        },
        'arcface': {
            'class': ArcFaceLoss,
            'default_params': {'margin': 0.5, 'scale': 64.0}
        },
        'dice': {
            'class': DiceLoss,
            'default_params': {}
        },
        'tversky': {
            'class': TverskyLoss,
            'default_params': {'alpha': 0.5, 'beta': 0.5}
        }
    }
    
    @classmethod
    def register_loss(cls, name: str, loss_class: type, default_params: Dict[str, Any]):
        """注册自定义损失函数"""
        cls._registry[name] = {
            'class': loss_class,
            'default_params': default_params
        }
        logger.info(f"注册损失函数: {name}")
    
    @classmethod
    def create_loss(cls, config: Union[str, Dict[str, Any]]) -> nn.Module:
        """创建损失函数"""
        if isinstance(config, str):
            config = {'type': config}
        
        loss_type = config['type']
        
        if loss_type not in cls._registry:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
        
        # 获取损失函数类和默认参数
        loss_info = cls._registry[loss_type]
        loss_class = loss_info['class']
        default_params = loss_info['default_params'].copy()
        
        # 合并参数
        params = default_params
        params.update(config.get('params', {}))
        
        # 特殊处理需要额外参数的损失函数
        if loss_type == 'label_smoothing' and 'num_classes' not in params:
            raise ValueError("LabelSmoothingLoss 需要 num_classes 参数")
        
        if loss_type == 'class_balanced':
            if 'num_classes' not in params:
                raise ValueError("ClassBalancedLoss 需要 num_classes 参数")
            if 'samples_per_class' not in params:
                raise ValueError("ClassBalancedLoss 需要 samples_per_class 参数")
        
        if loss_type == 'arcface':
            if 'num_classes' not in params:
                raise ValueError("ArcFaceLoss 需要 num_classes 参数")
            if 'embedding_size' not in params:
                raise ValueError("ArcFaceLoss 需要 embedding_size 参数")
        
        if loss_type == 'combined':
            if 'losses' not in params:
                raise ValueError("CombinedLoss 需要 losses 参数")
            # 递归创建子损失函数
            sub_losses = []
            for loss_config in params['losses']:
                sub_losses.append(cls.create_loss(loss_config))
            params['losses'] = sub_losses
        
        try:
            loss_fn = loss_class(**params)
            logger.info(f"创建损失函数: {loss_type} with params: {params}")
            return loss_fn
        except Exception as e:
            logger.error(f"创建损失函数失败: {e}")
            raise
    
    @classmethod
    def list_available_losses(cls) -> List[str]:
        """列出可用的损失函数"""
        return list(cls._registry.keys())


# 便捷函数
def create_loss(config: Union[str, Dict[str, Any]]) -> nn.Module:
    """创建损失函数的便捷函数"""
    return LossFunctionFactory.create_loss(config)


def register_custom_loss(name: str, loss_class: type, default_params: Dict[str, Any]):
    """注册自定义损失函数的便捷函数"""
    LossFunctionFactory.register_loss(name, loss_class, default_params)


# 预定义的损失函数配置
def get_classification_loss_configs() -> Dict[str, Dict[str, Any]]:
    """分类任务预定义损失配置"""
    return {
        'standard': {'type': 'cross_entropy'},
        'with_smoothing': {
            'type': 'label_smoothing',
            'params': {'smoothing': 0.1}
        },
        'focal': {
            'type': 'focal',
            'params': {'alpha': 1.0, 'gamma': 2.0}
        },
        'combined': {
            'type': 'combined',
            'params': {
                'losses': [
                    {'type': 'cross_entropy'},
                    {'type': 'focal', 'params': {'gamma': 1.0}}
                ],
                'weights': [0.7, 0.3]
            }
        }
    }


def get_imbalanced_loss_configs() -> Dict[str, Dict[str, Any]]:
    """类别不平衡任务预定义损失配置"""
    return {
        'weighted_ce': {
            'type': 'cross_entropy',
            'params': {'weight': None}  # 需要用户提供权重
        },
        'focal_balanced': {
            'type': 'focal',
            'params': {'alpha': 0.25, 'gamma': 2.0}
        },
        'class_balanced': {
            'type': 'class_balanced',
            'params': {
                'beta': 0.9999,
                'loss_type': 'focal'
            }
        }
    }


def get_metric_learning_loss_configs() -> Dict[str, Dict[str, Any]]:
    """度量学习预定义损失配置"""
    return {
        'triplet': {
            'type': 'triplet',
            'params': {'margin': 1.0}
        },
        'contrastive': {
            'type': 'contrastive',
            'params': {'margin': 1.0}
        },
        'arcface': {
            'type': 'arcface',
            'params': {
                'margin': 0.5,
                'scale': 64.0
            }
        }
    }


def get_segmentation_loss_configs() -> Dict[str, Dict[str, Any]]:
    """分割任务预定义损失配置"""
    return {
        'dice': {'type': 'dice'},
        'tversky': {
            'type': 'tversky',
            'params': {'alpha': 0.7, 'beta': 0.3}
        },
        'combined_seg': {
            'type': 'combined',
            'params': {
                'losses': [
                    {'type': 'bce'},
                    {'type': 'dice'}
                ],
                'weights': [0.5, 0.5]
            }
        }
    }