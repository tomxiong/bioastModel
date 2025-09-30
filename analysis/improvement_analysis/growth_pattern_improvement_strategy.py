#!/usr/bin/env python3
"""
Growth Pattern 分类性能改进策略实施脚本
针对当前76.67%的准确率进行优化
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
import numpy as np
from pathlib import Path
from collections import Counter

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AttentionMobileNetV3(nn.Module):
    """Enhanced MobileNetV3 with attention mechanism for Growth Pattern classification"""
    
    def __init__(self, base_model, num_classes=12):
        super(AttentionMobileNetV3, self).__init__()
        self.backbone = base_model.features
        self.avgpool = base_model.avgpool
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(960, 480, kernel_size=1),  # MobileNetV3-small last feature dim
            nn.ReLU(inplace=True),
            nn.Conv2d(480, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Global average pooling
        x = self.avgpool(attended_features)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        return x

def calculate_class_weights():
    """计算类别权重以处理不平衡数据"""
    
    # 从混淆矩阵计算实际的类别分布
    class_counts = np.array([96, 820, 790, 3, 249, 243, 5, 140, 6, 108, 503, 37])
    
    # 计算权重 (inversely proportional to frequency)
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # 归一化权重
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print("类别样本分布:")
    labels = [
        'center_dots', 'clean', 'clustered', 'default_positive', 'focal',
        'heavy_growth', 'irregular', 'litter_center_dots', 'scattered',
        'strong_scattered', 'weak_scattered', 'weak_scattered_pos'
    ]
    
    for i, (label, count, weight) in enumerate(zip(labels, class_counts, class_weights)):
        print(f"{label:20}: {count:4d} 样本, 权重: {weight:.3f}")
    
    return torch.FloatTensor(class_weights)

def create_weighted_sampler(dataset_labels):
    """创建加权采样器"""
    
    class_counts = Counter(dataset_labels)
    total_samples = len(dataset_labels)
    
    # 计算每个样本的权重
    sample_weights = []
    for label in dataset_labels:
        weight = total_samples / (len(class_counts) * class_counts[label])
        sample_weights.append(weight)
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def enhanced_data_augmentation_config():
    """增强的数据增强配置"""
    
    augmentation_config = {
        'rotation_range': 30,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'brightness_range': [0.8, 1.2],
        'contrast_range': [0.8, 1.2],
        'saturation_range': [0.8, 1.2],
        'hue_range': [-0.1, 0.1],
        'gaussian_noise_std': 0.01,
        'gaussian_blur_sigma': [0.1, 2.0],
        'elastic_transform': True,
        'grid_distortion': True
    }
    
    return augmentation_config

def progressive_training_strategy():
    """渐进式训练策略"""
    
    strategy = {
        'phase_1': {
            'description': '简单类别预训练',
            'classes': ['clean', 'clustered', 'weak_scattered'],  # 样本数多且相对容易区分
            'epochs': 20,
            'learning_rate': 0.001,
            'batch_size': 64
        },
        'phase_2': {
            'description': '中等难度类别',
            'classes': ['center_dots', 'focal', 'heavy_growth', 'litter_center_dots', 'strong_scattered'],
            'epochs': 30,
            'learning_rate': 0.0005,
            'batch_size': 64
        },
        'phase_3': {
            'description': '困难类别微调',
            'classes': ['default_positive', 'irregular', 'scattered', 'weak_scattered_pos'],
            'epochs': 40,
            'learning_rate': 0.0002,
            'batch_size': 32  # 减小batch size以更好处理困难样本
        },
        'phase_4': {
            'description': '全类别联合训练',
            'classes': 'all',
            'epochs': 50,
            'learning_rate': 0.0001,
            'batch_size': 64
        }
    }
    
    return strategy

def mixup_augmentation(x, y, alpha=0.2):
    """Mixup数据增强"""
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def cutmix_augmentation(x, y, alpha=1.0):
    """CutMix数据增强"""
    
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def create_improved_training_config():
    """创建改进的训练配置"""
    
    config = {
        'model_architecture': {
            'base_model': 'mobilenetv3_small',
            'use_attention': True,
            'dropout_rate': 0.3,
            'num_classes': 12
        },
        
        'loss_function': {
            'type': 'focal_loss',
            'alpha': 1.0,
            'gamma': 2.0,
            'use_class_weights': True
        },
        
        'optimizer': {
            'type': 'AdamW',
            'learning_rate': 0.0008,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999]
        },
        
        'scheduler': {
            'type': 'CosineAnnealingWarmRestarts',
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 1e-6
        },
        
        'training': {
            'epochs': 100,
            'batch_size': 64,
            'use_progressive_training': True,
            'use_mixup': True,
            'mixup_alpha': 0.2,
            'use_cutmix': True,
            'cutmix_alpha': 1.0,
            'gradient_clip_norm': 1.0
        },
        
        'data_augmentation': enhanced_data_augmentation_config(),
        
        'validation': {
            'validation_split': 0.2,
            'early_stopping_patience': 15,
            'monitor_metric': 'f1_score_macro'
        }
    }
    
    return config

def generate_improvement_implementation_plan():
    """生成具体的改进实施计划"""
    
    plan = {
        'immediate_actions': [
            {
                'action': '实施类别权重',
                'description': '使用计算出的类别权重来处理数据不平衡',
                'expected_improvement': '+3-5% accuracy',
                'implementation_time': '1 day'
            },
            {
                'action': '增强数据增强',
                'description': '添加更多样化的数据增强技术',
                'expected_improvement': '+2-4% accuracy',
                'implementation_time': '1 day'
            },
            {
                'action': '使用Focal Loss',
                'description': '替换交叉熵损失为Focal Loss',
                'expected_improvement': '+2-3% accuracy',
                'implementation_time': '0.5 day'
            }
        ],
        
        'medium_term_actions': [
            {
                'action': '添加注意力机制',
                'description': '在MobileNetV3基础上添加注意力模块',
                'expected_improvement': '+4-6% accuracy',
                'implementation_time': '2-3 days'
            },
            {
                'action': '渐进式训练',
                'description': '实施多阶段训练策略',
                'expected_improvement': '+3-5% accuracy',
                'implementation_time': '2 days'
            },
            {
                'action': 'Mixup/CutMix增强',
                'description': '实施高级数据增强技术',
                'expected_improvement': '+2-4% accuracy',
                'implementation_time': '1 day'
            }
        ],
        
        'long_term_actions': [
            {
                'action': '数据集扩充',
                'description': '收集更多困难类别的样本',
                'expected_improvement': '+5-8% accuracy',
                'implementation_time': '1-2 weeks'
            },
            {
                'action': '集成学习',
                'description': '训练多个模型并进行集成',
                'expected_improvement': '+3-5% accuracy',
                'implementation_time': '3-5 days'
            },
            {
                'action': '架构搜索',
                'description': '寻找更适合的网络架构',
                'expected_improvement': '+5-10% accuracy',
                'implementation_time': '1-2 weeks'
            }
        ]
    }
    
    return plan

def save_improvement_config():
    """保存改进配置到文件"""
    
    config = create_improved_training_config()
    plan = generate_improvement_implementation_plan()
    class_weights = calculate_class_weights().tolist()
    
    output_data = {
        'training_config': config,
        'implementation_plan': plan,
        'class_weights': class_weights,
        'progressive_strategy': progressive_training_strategy()
    }
    
    output_dir = Path('/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'growth_pattern_improvement_config.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"改进配置已保存到: {output_dir / 'growth_pattern_improvement_config.json'}")

def main():
    """主函数"""
    
    print("=== Growth Pattern 分类性能改进策略 ===\n")
    
    print("1. 分析当前问题:")
    print("   - 类别不平衡严重 (default_positive仅3个样本)")
    print("   - 某些类别混淆严重 (focal -> clustered: 94.6%)")
    print("   - 小样本类别性能极差 (weak_scattered_pos: 43.2%)")
    
    print("\n2. 计算类别权重:")
    class_weights = calculate_class_weights()
    
    print("\n3. 生成改进配置:")
    config = create_improved_training_config()
    
    print("\n4. 渐进式训练策略:")
    strategy = progressive_training_strategy()
    for phase, details in strategy.items():
        print(f"   {phase}: {details['description']}")
        print(f"      轮数: {details['epochs']}, 学习率: {details['learning_rate']}")
    
    print("\n5. 预期改进效果:")
    plan = generate_improvement_implementation_plan()
    
    total_immediate = sum([5, 4, 3])  # 最大预期改进
    total_medium = sum([6, 5, 4])
    
    print(f"   立即实施: +{total_immediate}% (目标: 76.67% -> 88-90%)")
    print(f"   中期实施: +{total_medium}% (目标: 90% -> 95%+)")
    
    print("\n6. 保存配置文件...")
    save_improvement_config()
    
    print("\n改进策略制定完成!")
    print("建议优先实施立即行动项目，预期可将准确率提升至88-90%")

if __name__ == "__main__":
    main()