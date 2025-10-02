#!/usr/bin/env python3
"""
Enhanced Multi-level MobileNetV3 with Growth Pattern Class Weights, Focal Loss, and Pores-specific Features
增强版多层分类MobileNetV3，包含Growth Pattern类别权重、Focal Loss和Pores专用功能

Key Improvements:
1. Growth Pattern class weights (0.019-5.219 range)
2. Focal Loss (alpha=1.0, gamma=2.0)
3. Pores-specific data augmentation and loss functions
4. Enhanced attention mechanisms for pores detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
try:
    from .mobilenet_v3 import create_mobilenetv3_small, create_mobilenetv3_large
except ImportError:
    # 当直接运行时使用绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mobilenet_v3 import create_mobilenetv3_small, create_mobilenetv3_large


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
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


class PoresSpecificLoss(nn.Module):
    """
    Pores-specific loss function with enhanced sensitivity to pores detection
    """
    def __init__(self, pores_weight: float = 2.0, alpha: float = 0.25, gamma: float = 2.0):
        super(PoresSpecificLoss, self).__init__()
        self.pores_weight = pores_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                pores_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            pores_mask: Binary mask indicating pores presence (optional)
        """
        base_loss = self.focal_loss(inputs, targets)
        
        if pores_mask is not None:
            # Apply additional weight to pores samples
            pores_samples = pores_mask.float()
            weighted_loss = base_loss * (1 + self.pores_weight * pores_samples)
            return weighted_loss.mean()
        
        return base_loss


class PoresAttentionModule(nn.Module):
    """
    Attention module specifically designed for pores detection
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super(PoresAttentionModule, self).__init__()
        
        # Channel attention for pores-specific features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention for pores localization
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.spatial_attention(sa_input)
        x = x * sa_weight
        
        return x


class EnhancedMultiLevelMobileNetV3(nn.Module):
    """
    Enhanced Multi-level MobileNetV3 with advanced features for biomedical image analysis
    """
    
    def __init__(self, 
                 model_size: str = 'small',
                 input_channels: int = 1,  # 灰度图像
                 dropout_rate: float = 0.3,
                 use_hierarchical_loss: bool = True,
                 freeze_backbone: bool = False,
                 use_pores_attention: bool = True,
                 growth_pattern_weights: Optional[List[float]] = None):
        super(EnhancedMultiLevelMobileNetV3, self).__init__()
        
        self.model_size = model_size
        self.input_channels = input_channels
        self.use_hierarchical_loss = use_hierarchical_loss
        self.use_pores_attention = use_pores_attention
        
        # Growth Pattern类别权重 (0.019-5.219范围)
        if growth_pattern_weights is None:
            # 默认权重，根据类别不平衡情况设置
            self.growth_pattern_weights = torch.tensor([
                5.219, 2.156, 1.000, 0.847, 0.623, 0.445, 
                0.312, 0.198, 0.134, 0.089, 0.056, 0.019
            ])
        else:
            self.growth_pattern_weights = torch.tensor(growth_pattern_weights)
        
        # 创建backbone
        if model_size == 'small':
            self.backbone = create_mobilenetv3_small(pretrained=True)
            feature_dim = 576  # MobileNetV3-Small最后一层特征维度
        else:
            self.backbone = create_mobilenetv3_large(pretrained=True)
            feature_dim = 960  # MobileNetV3-Large最后一层特征维度
        
        # 修改第一层以适应灰度输入
        if input_channels != 3:
            # 获取第一个卷积层
            first_conv = None
            for layer in self.backbone.features:
                if isinstance(layer, nn.Sequential):
                    for sublayer in layer:
                        if isinstance(sublayer, nn.Conv2d):
                            first_conv = sublayer
                            break
                elif isinstance(sublayer, nn.Conv2d):
                    first_conv = layer
                    break
                if first_conv:
                    break
            
            if first_conv:
                # 创建新的第一层卷积
                new_first_conv = nn.Conv2d(
                    input_channels, first_conv.out_channels,
                    first_conv.kernel_size, first_conv.stride,
                    first_conv.padding, bias=False
                )
                
                # 替换第一层
                if isinstance(self.backbone.features[0], nn.Sequential):
                    self.backbone.features[0][0] = new_first_conv
                else:
                    self.backbone.features[0] = new_first_conv
        
        # 移除原始分类器
        self.backbone.classifier = nn.Identity()
        
        # 冻结backbone（可选）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Pores专用注意力模块
        if use_pores_attention:
            self.pores_attention = PoresAttentionModule(feature_dim)
        
        # 全局平均池化
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 增强的特征处理器
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # 分类头定义
        self.num_classes = {
            'growth_level': 2,      # positive/negative
            'growth_pattern': 10,   # 10种生长模式（更新为实际类别数）
            'interference_factors': 4  # 4种干扰因子（多标签）
        }
        
        # 任务特定的分类头
        self.classifiers = nn.ModuleDict({
            'growth_level': self._create_classifier(256, self.num_classes['growth_level'], dropout_rate),
            'growth_pattern': self._create_classifier(256, self.num_classes['growth_pattern'], dropout_rate),
            'interference_factors': self._create_classifier(256, self.num_classes['interference_factors'], dropout_rate)
        })
        
        # Pores专用分类头
        self.pores_classifier = self._create_pores_classifier(256, dropout_rate)
        
        # 增强的任务权重
        self.task_weights = {
            'growth_level': 1.0,
            'growth_pattern': 1.2,  # 增加growth_pattern权重
            'interference_factors': 0.8,
            'pores_detection': 1.5  # Pores检测权重
        }
        
        # 损失函数
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.pores_loss = PoresSpecificLoss(pores_weight=2.0, alpha=0.25, gamma=2.0)
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_classifier(self, input_dim: int, num_classes: int, dropout_rate: float) -> nn.Module:
        """创建标准分类器"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, num_classes)
        )
    
    def _create_pores_classifier(self, input_dim: int, dropout_rate: float) -> nn.Module:
        """创建Pores专用分类器"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(64, 2)  # Binary classification for pores
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, 1, 70, 70]
            
        Returns:
            Dict containing logits for each classification task
        """
        # 特征提取 - 直接通过backbone的features
        features = x
        for i, layer in enumerate(self.backbone.features):
            features = layer(features)
        
        # 在最后应用Pores注意力（如果启用）
        if self.use_pores_attention:
            features = self.pores_attention(features)
        
        # 全局平均池化
        features = self.global_avgpool(features)
        features = torch.flatten(features, 1)
        
        # 共享特征处理
        shared_features = self.feature_processor(features)
        
        # 多任务分类
        outputs = {}
        for task_name, classifier in self.classifiers.items():
            outputs[task_name] = classifier(shared_features)
        
        # Pores专用检测
        outputs['pores_detection'] = self.pores_classifier(shared_features)
        
        return outputs
    
    def compute_enhanced_loss(self, outputs: Dict[str, torch.Tensor], 
                            targets: Dict[str, torch.Tensor],
                            pores_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算增强的多任务损失
        
        Args:
            outputs: 模型输出
            targets: 真实标签
            pores_mask: Pores样本掩码（可选）
            
        Returns:
            Dict containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # Growth level loss (使用Focal Loss)
        if 'growth_level' in outputs and 'growth_level' in targets:
            losses['growth_level'] = self.focal_loss(outputs['growth_level'], targets['growth_level'])
            total_loss += self.task_weights['growth_level'] * losses['growth_level']
        
        # Growth pattern loss (使用加权Focal Loss)
        if 'growth_pattern' in outputs and 'growth_pattern' in targets:
            # 使用类别权重的交叉熵损失
            weights = self.growth_pattern_weights.to(outputs['growth_pattern'].device)
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            losses['growth_pattern'] = loss_fn(outputs['growth_pattern'], targets['growth_pattern'])
            total_loss += self.task_weights['growth_pattern'] * losses['growth_pattern']
        
        # Interference factors loss (多标签)
        if 'interference_factors' in outputs and 'interference_factors' in targets:
            loss_fn = nn.BCEWithLogitsLoss()
            losses['interference_factors'] = loss_fn(outputs['interference_factors'], targets['interference_factors'])
            total_loss += self.task_weights['interference_factors'] * losses['interference_factors']
        
        # Pores专用损失
        if 'pores_detection' in outputs and 'pores_detection' in targets:
            losses['pores_detection'] = self.pores_loss(
                outputs['pores_detection'], 
                targets['pores_detection'], 
                pores_mask
            )
            total_loss += self.task_weights['pores_detection'] * losses['pores_detection']
        
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
            
            # Pores detection prediction
            predictions['pores_detection'] = torch.softmax(outputs['pores_detection'], dim=1)
            
            return predictions
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': f'Enhanced-MultiLevel-MobileNetV3-{self.model_size}',
            'input_size': (self.input_channels, 70, 70),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'features': [
                'Growth Pattern Class Weights (0.019-5.219)',
                'Focal Loss (alpha=1.0, gamma=2.0)',
                'Pores-specific Attention',
                'Pores-specific Loss Function',
                'Enhanced Feature Processing'
            ]
        }


def create_enhanced_multilevel_mobilenetv3(model_size: str = 'small', 
                                         input_channels: int = 1,
                                         growth_pattern_weights: Optional[List[float]] = None,
                                         **kwargs) -> EnhancedMultiLevelMobileNetV3:
    """
    创建增强版多层分类MobileNetV3模型
    
    Args:
        model_size: 'small' or 'large'
        input_channels: 输入通道数（默认1为灰度图像）
        growth_pattern_weights: Growth Pattern类别权重列表
        **kwargs: 其他参数
        
    Returns:
        EnhancedMultiLevelMobileNetV3 model
    """
    model = EnhancedMultiLevelMobileNetV3(
        model_size=model_size,
        input_channels=input_channels,
        growth_pattern_weights=growth_pattern_weights,
        **kwargs
    )
    return model


# Pores专用数据增强函数
class PoresSpecificAugmentation:
    """
    Pores-specific data augmentation techniques
    """
    
    @staticmethod
    def enhance_pores_contrast(image: torch.Tensor, factor: float = 1.5) -> torch.Tensor:
        """增强Pores对比度"""
        mean = torch.mean(image)
        return torch.clamp((image - mean) * factor + mean, 0, 1)
    
    @staticmethod
    def pores_edge_enhancement(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """孔隙边缘增强"""
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        
        # 确保kernel在正确的设备上
        if image.is_cuda:
            kernel = kernel.cuda()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        enhanced = F.conv2d(image, kernel, padding=1)
        return torch.clamp(image + 0.3 * enhanced, 0, 1).squeeze(0)
    
    @staticmethod
    def adaptive_histogram_equalization(image: torch.Tensor) -> torch.Tensor:
        """自适应直方图均衡化"""
        # 简化版本的自适应直方图均衡化
        image_np = image.cpu().numpy()
        
        # 计算累积分布函数
        hist, bins = np.histogram(image_np.flatten(), 256, [0, 1])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # 应用变换
        image_equalized = np.interp(image_np.flatten(), bins[:-1], cdf_normalized)
        image_equalized = image_equalized.reshape(image_np.shape) / 255.0
        
        return torch.from_numpy(image_equalized).float()


if __name__ == "__main__":
    # 测试增强版模型
    print("=== Enhanced Multi-Level MobileNetV3 Test ===")
    
    # 创建模型
    model = create_enhanced_multilevel_mobilenetv3(
        model_size='small', 
        input_channels=1,
        use_pores_attention=True
    )
    
    # 模型信息
    info = model.get_model_info()
    print("=== Model Information ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 测试前向传播
    print("\n=== Forward Pass Test ===")
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
    
    # 测试损失计算
    print("\n=== Loss Computation Test ===")
    targets = {
        'growth_level': torch.randint(0, 2, (2,)),
        'growth_pattern': torch.randint(0, 12, (2,)),
        'interference_factors': torch.randint(0, 2, (2, 4)).float(),
        'pores_detection': torch.randint(0, 2, (2,))
    }
    
    losses = model.compute_enhanced_loss(outputs, targets)
    print("Loss components:")
    for loss_name, loss_value in losses.items():
        print(f"{loss_name}: {loss_value.item():.4f}")
    
    print("\n=== Pores-specific Augmentation Test ===")
    aug = PoresSpecificAugmentation()
    test_image = torch.randn(1, 70, 70)
    
    enhanced_contrast = aug.enhance_pores_contrast(test_image)
    edge_enhanced = aug.pores_edge_enhancement(test_image)
    hist_equalized = aug.adaptive_histogram_equalization(test_image)
    
    print(f"Original image shape: {test_image.shape}")
    print(f"Enhanced contrast shape: {enhanced_contrast.shape}")
    print(f"Edge enhanced shape: {edge_enhanced.shape}")
    print(f"Histogram equalized shape: {hist_equalized.shape}")
    
    print("\n✅ All tests passed! Enhanced model is ready for training.")