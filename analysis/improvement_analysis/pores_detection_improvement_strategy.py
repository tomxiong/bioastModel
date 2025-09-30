#!/usr/bin/env python3
"""
Pores 检测性能改进策略实施脚本
针对当前86.20%的准确率进行优化
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
from sklearn.metrics import classification_report

class PoresSpecificFeatureExtractor(nn.Module):
    """专门针对Pores检测的特征提取器"""
    
    def __init__(self, input_channels=3):
        super(PoresSpecificFeatureExtractor, self).__init__()
        
        # 多尺度特征提取
        self.scale1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        
        # 特征拼接
        combined = torch.cat([feat1, feat2, feat3], dim=1)
        
        # 特征融合
        fused = self.fusion(combined)
        
        # 注意力加权
        attention_weights = self.attention(fused)
        attended_features = fused * attention_weights
        
        return attended_features

class EnhancedPoresDetector(nn.Module):
    """增强的Pores检测器"""
    
    def __init__(self, base_model, num_interference_classes=4):
        super(EnhancedPoresDetector, self).__init__()
        
        # 基础特征提取器
        self.backbone = base_model.features
        
        # Pores专用特征提取器
        self.pores_extractor = PoresSpecificFeatureExtractor(960)  # MobileNetV3-small输出通道数
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(960 + 32, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_interference_classes)
        )
    
    def forward(self, x):
        # 基础特征
        base_features = self.backbone(x)
        
        # Pores专用特征
        pores_features = self.pores_extractor(x)
        
        # 调整尺寸匹配
        pores_features = F.adaptive_avg_pool2d(pores_features, base_features.shape[-2:])
        
        # 特征融合
        combined_features = torch.cat([base_features, pores_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 分类
        x = torch.flatten(fused_features, 1)
        x = self.classifier(x)
        
        return x

def analyze_pores_detection_challenges():
    """分析Pores检测面临的挑战"""
    
    challenges = {
        'visual_similarity': {
            'description': 'Pores与其他干扰因子在视觉上相似',
            'specific_issues': [
                'Pores与Debris的纹理相似性',
                'Pores与Artifacts的形状相似性',
                'Pores的大小变化范围大'
            ],
            'impact': 'high'
        },
        
        'scale_variation': {
            'description': 'Pores的尺度变化很大',
            'specific_issues': [
                '微小pores难以检测',
                '大型pores容易与其他因子混淆',
                '多尺度特征提取不足'
            ],
            'impact': 'high'
        },
        
        'context_dependency': {
            'description': 'Pores的检测依赖于上下文信息',
            'specific_issues': [
                '需要考虑周围的生长模式',
                '背景复杂度影响检测',
                '光照条件变化'
            ],
            'impact': 'medium'
        },
        
        'data_quality': {
            'description': '数据质量和标注一致性问题',
            'specific_issues': [
                '标注主观性强',
                '边界模糊的pores标注困难',
                '样本多样性不足'
            ],
            'impact': 'medium'
        }
    }
    
    return challenges

def create_pores_specific_augmentation():
    """创建针对Pores检测的专用数据增强策略"""
    
    augmentation_strategies = {
        'geometric_augmentation': {
            'rotation': {
                'range': [-15, 15],
                'probability': 0.7,
                'reason': 'Pores可能以任意角度出现'
            },
            'scaling': {
                'range': [0.8, 1.2],
                'probability': 0.6,
                'reason': '处理不同尺度的pores'
            },
            'elastic_deformation': {
                'alpha': 50,
                'sigma': 5,
                'probability': 0.3,
                'reason': '模拟生物样本的自然变形'
            }
        },
        
        'photometric_augmentation': {
            'contrast_adjustment': {
                'range': [0.8, 1.3],
                'probability': 0.8,
                'reason': '增强pores与背景的对比度'
            },
            'brightness_adjustment': {
                'range': [0.9, 1.1],
                'probability': 0.6,
                'reason': '适应不同光照条件'
            },
            'gamma_correction': {
                'range': [0.8, 1.2],
                'probability': 0.5,
                'reason': '调整图像的动态范围'
            },
            'histogram_equalization': {
                'probability': 0.3,
                'reason': '增强局部对比度'
            }
        },
        
        'noise_augmentation': {
            'gaussian_noise': {
                'std': 0.01,
                'probability': 0.4,
                'reason': '提高模型对噪声的鲁棒性'
            },
            'salt_pepper_noise': {
                'amount': 0.005,
                'probability': 0.2,
                'reason': '模拟图像采集过程中的噪声'
            }
        },
        
        'morphological_augmentation': {
            'erosion_dilation': {
                'kernel_size': [3, 5],
                'probability': 0.3,
                'reason': '模拟pores形状的细微变化'
            },
            'opening_closing': {
                'kernel_size': [3, 5],
                'probability': 0.2,
                'reason': '处理pores的连通性变化'
            }
        }
    }
    
    return augmentation_strategies

def design_pores_specific_loss():
    """设计针对Pores检测的专用损失函数"""
    
    class PoresFocusedLoss(nn.Module):
        def __init__(self, pores_class_idx=3, alpha=2.0, gamma=2.0):
            super(PoresFocusedLoss, self).__init__()
            self.pores_class_idx = pores_class_idx
            self.alpha = alpha
            self.gamma = gamma
            
        def forward(self, inputs, targets):
            # 标准交叉熵损失
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            
            # 对Pores类别应用更强的Focal Loss
            pt = torch.exp(-ce_loss)
            
            # 为Pores类别设置更高的权重
            alpha_weights = torch.ones_like(targets, dtype=torch.float)
            alpha_weights[targets == self.pores_class_idx] = self.alpha
            
            # Focal Loss
            focal_loss = alpha_weights * (1 - pt) ** self.gamma * ce_loss
            
            return focal_loss.mean()
    
    return PoresFocusedLoss

def create_pores_improvement_config():
    """创建Pores检测改进配置"""
    
    config = {
        'model_architecture': {
            'use_enhanced_detector': True,
            'multi_scale_features': True,
            'attention_mechanism': True,
            'feature_fusion': True
        },
        
        'loss_function': {
            'type': 'pores_focused_loss',
            'pores_class_weight': 2.0,
            'focal_gamma': 2.0,
            'use_class_balancing': True
        },
        
        'data_augmentation': create_pores_specific_augmentation(),
        
        'training_strategy': {
            'two_stage_training': {
                'stage_1': {
                    'description': '通用干扰因子检测预训练',
                    'epochs': 30,
                    'learning_rate': 0.001,
                    'focus': 'all_interference_factors'
                },
                'stage_2': {
                    'description': 'Pores专用微调',
                    'epochs': 50,
                    'learning_rate': 0.0005,
                    'focus': 'pores_specific',
                    'use_hard_negative_mining': True
                }
            }
        },
        
        'hard_negative_mining': {
            'enabled': True,
            'ratio': 0.3,
            'update_frequency': 5,
            'description': '重点训练容易混淆的样本'
        },
        
        'ensemble_strategy': {
            'enabled': True,
            'models': [
                'enhanced_mobilenetv3',
                'resnet50_pores_specific',
                'efficientnet_b0_pores_tuned'
            ],
            'voting_strategy': 'weighted_average'
        }
    }
    
    return config

def generate_pores_improvement_plan():
    """生成Pores检测改进实施计划"""
    
    plan = {
        'immediate_improvements': [
            {
                'action': '实施Pores专用数据增强',
                'description': '添加针对pores特征的增强策略',
                'expected_improvement': '+2-3% accuracy',
                'implementation_time': '1 day',
                'priority': 'high'
            },
            {
                'action': '调整类别权重',
                'description': '为Pores类别设置更高的损失权重',
                'expected_improvement': '+1-2% accuracy',
                'implementation_time': '0.5 day',
                'priority': 'high'
            },
            {
                'action': '实施Focal Loss',
                'description': '使用Focal Loss处理困难样本',
                'expected_improvement': '+2-3% accuracy',
                'implementation_time': '0.5 day',
                'priority': 'high'
            }
        ],
        
        'medium_term_improvements': [
            {
                'action': '多尺度特征提取',
                'description': '添加专门的多尺度特征提取模块',
                'expected_improvement': '+3-5% accuracy',
                'implementation_time': '2-3 days',
                'priority': 'medium'
            },
            {
                'action': '困难负样本挖掘',
                'description': '重点训练容易混淆的样本',
                'expected_improvement': '+2-4% accuracy',
                'implementation_time': '2 days',
                'priority': 'medium'
            },
            {
                'action': '两阶段训练',
                'description': '先通用训练，再Pores专用微调',
                'expected_improvement': '+3-4% accuracy',
                'implementation_time': '2 days',
                'priority': 'medium'
            }
        ],
        
        'advanced_improvements': [
            {
                'action': '集成学习',
                'description': '训练多个专门的Pores检测模型',
                'expected_improvement': '+4-6% accuracy',
                'implementation_time': '3-5 days',
                'priority': 'low'
            },
            {
                'action': '数据集扩充',
                'description': '收集更多高质量的Pores样本',
                'expected_improvement': '+5-8% accuracy',
                'implementation_time': '1-2 weeks',
                'priority': 'low'
            },
            {
                'action': '半监督学习',
                'description': '利用未标注数据提升性能',
                'expected_improvement': '+3-5% accuracy',
                'implementation_time': '1 week',
                'priority': 'low'
            }
        ]
    }
    
    return plan

def calculate_expected_improvement():
    """计算预期的性能提升"""
    
    current_accuracy = 86.20
    
    improvements = {
        'immediate': {
            'min_improvement': 5,  # 5%
            'max_improvement': 8,  # 8%
            'target_accuracy_range': [91.2, 94.2]
        },
        'medium_term': {
            'additional_min': 8,   # 额外8%
            'additional_max': 13,  # 额外13%
            'target_accuracy_range': [99.2, 107.2]  # 但实际上限为100%
        },
        'realistic_target': {
            'short_term': 92.0,   # 3个月内
            'medium_term': 95.0,  # 6个月内
            'long_term': 97.0     # 1年内
        }
    }
    
    return improvements

def save_pores_improvement_config():
    """保存Pores改进配置到文件"""
    
    config = create_pores_improvement_config()
    plan = generate_pores_improvement_plan()
    challenges = analyze_pores_detection_challenges()
    improvements = calculate_expected_improvement()
    
    output_data = {
        'current_performance': {
            'accuracy': 86.20,
            'challenges': challenges
        },
        'improvement_config': config,
        'implementation_plan': plan,
        'expected_improvements': improvements,
        'technical_details': {
            'multi_scale_kernels': [3, 5, 7],
            'attention_mechanism': 'channel_spatial_attention',
            'feature_fusion_strategy': 'concatenation_with_1x1_conv',
            'loss_function_weights': {
                'artifacts': 1.0,
                'contamination': 1.0,
                'debris': 1.0,
                'pores': 2.0
            }
        }
    }
    
    output_dir = Path('/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'pores_detection_improvement_config.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Pores检测改进配置已保存到: {output_dir / 'pores_detection_improvement_config.json'}")

def main():
    """主函数"""
    
    print("=== Pores 检测性能改进策略 ===\n")
    
    print("1. 当前性能分析:")
    print("   - Pores检测准确率: 86.20%")
    print("   - 在干扰因子检测中排名第3 (仅高于Artifacts 92.53%)")
    print("   - 主要挑战: 视觉相似性、尺度变化、上下文依赖")
    
    print("\n2. 问题分析:")
    challenges = analyze_pores_detection_challenges()
    for challenge, details in challenges.items():
        print(f"   {challenge}: {details['description']} (影响: {details['impact']})")
    
    print("\n3. 改进策略:")
    config = create_pores_improvement_config()
    print("   - 多尺度特征提取")
    print("   - 注意力机制")
    print("   - Pores专用损失函数")
    print("   - 困难负样本挖掘")
    print("   - 两阶段训练策略")
    
    print("\n4. 实施计划:")
    plan = generate_pores_improvement_plan()
    
    immediate_total = sum([3, 2, 3])  # 立即改进的最大预期
    medium_total = sum([5, 4, 4])     # 中期改进的最大预期
    
    print(f"   立即实施 (1-2天): +{immediate_total}% (目标: 86.20% -> 94%+)")
    print(f"   中期实施 (1周内): +{medium_total}% (目标: 94% -> 97%+)")
    
    print("\n5. 预期改进效果:")
    improvements = calculate_expected_improvement()
    print(f"   短期目标 (3个月): {improvements['realistic_target']['short_term']}%")
    print(f"   中期目标 (6个月): {improvements['realistic_target']['medium_term']}%")
    print(f"   长期目标 (1年): {improvements['realistic_target']['long_term']}%")
    
    print("\n6. 保存配置文件...")
    save_pores_improvement_config()
    
    print("\n改进策略制定完成!")
    print("建议优先实施立即改进项目，预期可将Pores检测准确率提升至94%以上")

if __name__ == "__main__":
    main()