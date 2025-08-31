#!/usr/bin/env python3
"""
增强的多层级生物图像分类系统
支持基础分类、生长模式、干扰因素和精细分类的多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

class EnhancedMobileNetV3MultiTask(nn.Module):
    """增强的MobileNetV3多任务学习模型"""
    
    def __init__(self, 
                 growth_level_classes: int = 3,
                 growth_pattern_classes: int = 9,
                 interference_classes: int = 3,
                 fine_grained_classes: int = 40,
                 width_mult: float = 1.0, 
                 dropout_rate: float = 0.2):
        super().__init__()
        
        # 任务定义
        self.growth_level_classes = growth_level_classes
        self.growth_pattern_classes = growth_pattern_classes
        self.interference_classes = interference_classes
        self.fine_grained_classes = fine_grained_classes
        
        # 共享的特征提取器
        self.backbone = self._create_backbone(width_mult)
        backbone_output_channels = self._get_backbone_output_channels(width_mult)
        
        # 任务特定的分类头
        self.growth_level_head = self._create_classification_head(
            backbone_output_channels, growth_level_classes, dropout_rate
        )
        
        self.growth_pattern_head = self._create_classification_head(
            backbone_output_channels, growth_pattern_classes, dropout_rate
        )
        
        # 干扰因素头（多标签分类）
        self.interference_head = self._create_multilabel_head(
            backbone_output_channels, interference_classes
        )
        
        # 精细分类头
        self.fine_grained_head = self._create_classification_head(
            backbone_output_channels, fine_grained_classes, dropout_rate
        )
        
        # 注意力机制
        self.attention = SEAttention(backbone_output_channels)
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_backbone(self, width_mult: float):
        """创建共享的backbone特征提取器"""
        # 基于MobileNetV3-Small的架构
        layers = []
        
        # 第一层
        input_channels = int(16 * width_mult)
        layers.extend([
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.Hardswish(inplace=True)
        ])
        
        # MobileNetV3-Small 配置
        block_configs = [
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
    
    def _create_multilabel_head(self, input_channels: int, num_classes: int):
        """创建多标签分类头"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, input_channels // 2),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_channels // 2, num_classes),
            nn.Sigmoid()  # 多标签分类使用sigmoid
        )
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = self.backbone(x)
        
        # 应用注意力机制
        features = self.attention(features)
        
        # 多任务预测
        outputs = {
            'growth_level': self.growth_level_head(features),
            'growth_pattern': self.growth_pattern_head(features),
            'interference_factors': self.interference_head(features),
            'fine_grained': self.fine_grained_head(features)
        }
        
        return outputs
    
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

def create_enhanced_multitask_mobilenetv3(
    growth_level_classes: int = 3,
    growth_pattern_classes: int = 7, 
    interference_classes: int = 8,
    fine_grained_classes: int = 15,
    width_mult: float = 1.0,
    **kwargs
) -> EnhancedMobileNetV3MultiTask:
    """创建增强的多任务MobileNetV3模型"""
    model = EnhancedMobileNetV3MultiTask(
        growth_level_classes=growth_level_classes,
        growth_pattern_classes=growth_pattern_classes,
        interference_classes=interference_classes,
        fine_grained_classes=fine_grained_classes,
        width_mult=width_mult,
        **kwargs
    )
    return model

def get_class_definitions():
    """获取分类定义"""
    return {
        'growth_level': {
            'classes': ['negative', 'positive', 'weak_growth'],
            'descriptions': {
                'negative': '无菌落生长，干净的培养基或仅有气孔',
                'positive': '明确的菌落生长，可见明显的微生物菌落',
                'weak_growth': '微弱生长，菌落不明显或生长稀少'
            }
        },
        'growth_pattern': {
            'classes': ['clean', 'clustered', 'scattered', 'heavy_growth', 'small_dots', 'irregular_areas', 'light_gray'],
            'descriptions': {
                'clean': '无菌落生长，培养基表面清洁',
                'clustered': '菌落成聚集状生长，最常见的阳性生长模式',
                'scattered': '分散型生长，菌落分布稀疏',
                'heavy_growth': '重度生长，菌落密集覆盖',
                'small_dots': '小点状生长，通常与弱生长相关',
                'irregular_areas': '不规则区域生长，形态不典型',
                'light_gray': '浅灰色菌落，颜色特征明显'
            }
        },
        'interference_factors': {
            'classes': ['pores', '气孔', '气孔重叠', 'debris', '杂质', 'noise', 'light_gray', 'pollution'],
            'descriptions': {
                'pores': '培养基中的气孔干扰',
                '气孔': '中文标注的气孔',
                '气孔重叠': '多个气孔重叠的干扰',
                'debris': '碎片或残渣',
                '杂质': '培养基中的杂质',
                'noise': '图像噪声',
                'light_gray': '浅灰色干扰',
                'pollution': '污染因素'
            }
        },
        'fine_grained': {
            'classes': [
                'negative_clean', 'negative_with_pores', 'negative_with_debris',
                'positive_clustered_clean', 'positive_clustered_with_pores', 'positive_clustered_with_debris',
                'positive_scattered_clean', 'positive_scattered_with_pores', 'positive_scattered_with_debris',
                'positive_heavy_growth_clean', 'positive_heavy_growth_with_pores', 'positive_heavy_growth_with_debris',
                'weak_growth_small_dots_clean', 'weak_growth_small_dots_with_pores', 'weak_growth_small_dots_with_debris'
            ],
            'descriptions': {
                'negative_clean': '阴性清洁样本',
                'negative_with_pores': '阴性带气孔样本',
                'negative_with_debris': '阴性带碎片样本',
                'positive_clustered_clean': '阳性聚集型清洁样本',
                'positive_clustered_with_pores': '阳性聚集型带气孔样本',
                'positive_clustered_with_debris': '阳性聚集型带碎片样本',
                'positive_scattered_clean': '阳性分散型清洁样本',
                'positive_scattered_with_pores': '阳性分散型带气孔样本',
                'positive_scattered_with_debris': '阳性分散型带碎片样本',
                'positive_heavy_growth_clean': '阳性重度生长清洁样本',
                'positive_heavy_growth_with_pores': '阳性重度生长带气孔样本',
                'positive_heavy_growth_with_debris': '阳性重度生长带碎片样本',
                'weak_growth_small_dots_clean': '弱生长小点型清洁样本',
                'weak_growth_small_dots_with_pores': '弱生长小点型带气孔样本',
                'weak_growth_small_dots_with_debris': '弱生长小点型带碎片样本'
            }
        }
    }

def test_enhanced_multitask_model():
    """测试增强的多任务模型"""
    print("=== 测试增强的多任务MobileNetV3模型 ===")
    
    # 创建模型
    model = create_enhanced_multitask_mobilenetv3(
        growth_level_classes=3,
        growth_pattern_classes=7,
        interference_classes=8,
        fine_grained_classes=15,
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
    print(f"干扰因素输出形状: {outputs['interference_factors'].shape}")
    print(f"精细分类输出形状: {outputs['fine_grained'].shape}")
    
    # 测试损失函数
    growth_level_criterion = nn.CrossEntropyLoss()
    growth_pattern_criterion = nn.CrossEntropyLoss()
    fine_grained_criterion = nn.CrossEntropyLoss()
    interference_criterion = nn.BCELoss()  # 多标签损失
    
    # 模拟标签
    growth_level_labels = torch.randint(0, 3, (batch_size,))
    growth_pattern_labels = torch.randint(0, 7, (batch_size,))
    fine_grained_labels = torch.randint(0, 15, (batch_size,))
    interference_labels = torch.rand(batch_size, 8) > 0.7
    interference_labels = interference_labels.float()
    
    # 计算损失
    growth_level_loss = growth_level_criterion(outputs['growth_level'], growth_level_labels)
    growth_pattern_loss = growth_pattern_criterion(outputs['growth_pattern'], growth_pattern_labels)
    fine_grained_loss = fine_grained_criterion(outputs['fine_grained'], fine_grained_labels)
    interference_loss = interference_criterion(outputs['interference_factors'], interference_labels)
    
    # 组合损失
    total_loss = (
        growth_level_loss + 
        growth_pattern_loss + 
        fine_grained_loss + 
        interference_loss
    )
    
    print(f"生长级别损失: {growth_level_loss:.4f}")
    print(f"生长模式损失: {growth_pattern_loss:.4f}")
    print(f"精细分类损失: {fine_grained_loss:.4f}")
    print(f"干扰因素损失: {interference_loss:.4f}")
    print(f"总损失: {total_loss:.4f}")
    
    # 测试预测
    growth_level_pred = torch.argmax(outputs['growth_level'], dim=1)
    growth_pattern_pred = torch.argmax(outputs['growth_pattern'], dim=1)
    fine_grained_pred = torch.argmax(outputs['fine_grained'], dim=1)
    interference_pred = (outputs['interference_factors'] > 0.5).float()
    
    print(f"生长级别预测: {growth_level_pred}")
    print(f"生长模式预测: {growth_pattern_pred}")
    print(f"精细分类预测: {fine_grained_pred}")
    print(f"干扰因素预测: {interference_pred}")
    
    print("[OK] 增强的多任务模型测试成功")

if __name__ == "__main__":
    test_enhanced_multitask_model()