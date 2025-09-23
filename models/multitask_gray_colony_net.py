"""
多任务版本的灰度菌落检测网络
基于GrayColonyNet扩展，支持4层标注任务

标注层级：
1. growth_level: 生长级别 (negative, positive, weak_growth) - 3类
2. growth_pattern: 生长模式 (clean, clustered, scattered, small_dots等) - 9类
3. interference_mapping: 干扰因素 (pores, debris, artifacts, 污染) - 4类，多标签
4. fine_grained: 精细分类 - 15类组合

精细分类包括：
- 阳性聚集型（带气孔、无气孔、气孔重叠）
- 阴性清亮型（带气孔、无气孔）
- 中心小点弱生长（带气孔、无气孔、气孔重叠）
- 弱生长分散区域（带气孔、不带气孔、气孔重叠）
- 含杂物的样本
- 污染的样本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np

# 导入GrayColonyNet的组件
from models.gray_colony_net import (
    GrayScaleStem, 
    TextureAwareConv, 
    HollowStructureDetector,
    BackgroundFilter,
    MicroTransformerBlock
)


class MultitaskHeads(nn.Module):
    """多任务分类头部"""
    
    def __init__(self, feature_dim: int = 128, dropout_rate: float = 0.2):
        super().__init__()
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        
        # 生长级别头部（3类：negative, positive, weak_growth）
        self.growth_level_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 3)
        )
        
        # 生长模式头部（9类）
        self.growth_pattern_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 9)
        )
        
        # 干扰因素头部（5类，多标签）
        self.interference_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 5)
        )
        
        # 精细分类头部（15类）
        self.fine_grained_head = nn.Sequential(
            nn.Dropout(dropout_rate * 1.5),  # 更高的dropout
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 15)
        )
        
        # 辅助输出：气孔置信度（用于精细分类）
        self.pore_confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 辅助输出：背景置信度
        self.bg_confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, 
                hollow_info: Dict[str, torch.Tensor] = None,
                bg_attention: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            features: 全局特征 [B, feature_dim]
            hollow_info: 中空结构检测信息
            bg_attention: 背景注意力图
            
        Returns:
            包含所有任务输出的字典
        """
        outputs = {}
        
        # 基础任务预测
        outputs['growth_level'] = self.growth_level_head(features)
        outputs['growth_pattern'] = self.growth_pattern_head(features)
        outputs['interference_mapping'] = self.interference_head(features)
        outputs['fine_grained'] = self.fine_grained_head(features)
        
        # 辅助输出
        outputs['pore_confidence'] = self.pore_confidence_head(features)
        outputs['bg_confidence'] = self.bg_confidence_head(features)
        
        # 如果有中空结构信息，添加到输出
        if hollow_info is not None:
            outputs['hollow_score'] = hollow_info.get('hollow_score', torch.zeros(features.size(0), 1, device=features.device))
            outputs['edge_irregularity'] = hollow_info.get('edge_irregularity', torch.zeros(features.size(0), 1, device=features.device))
        
        if bg_attention is not None:
            # 将背景注意力图全局池化作为背景强度指标
            bg_strength = F.adaptive_avg_pool2d(bg_attention, 1).flatten(1)
            outputs['bg_strength'] = bg_strength
        
        return outputs


class MultitaskGrayColonyNet(nn.Module):
    """
    多任务灰度菌落检测网络
    
    基于GrayColonyNet架构扩展，支持4层标注任务：
    1. 生长级别分类
    2. 生长模式分类
    3. 干扰因素检测（多标签）
    4. 精细分类
    
    保留原有的专有能力：
    - 中空结构检测
    - 背景过滤
    - 纹理感知
    """
    
    def __init__(self, 
                 feature_dim: int = 128,
                 enable_background_filter: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.enable_background_filter = enable_background_filter
        
        # 1. 灰度输入处理
        self.gray_stem = GrayScaleStem(in_channels=1)
        
        # 2. CNN特征提取
        self.cnn_backbone = nn.ModuleList([
            # Stage 1: 35x35 -> 18x18
            nn.Sequential(
                TextureAwareConv(32, 64),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # Stage 2: 18x18 -> 9x9
            nn.Sequential(
                TextureAwareConv(64, 96),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True)
            ),
            # Stage 3: 9x9 -> 7x7
            nn.Sequential(
                TextureAwareConv(96, feature_dim),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 3. 底纹过滤器
        if self.enable_background_filter:
            self.bg_filter = BackgroundFilter(feature_dim)
        
        # 4. 中空结构检测器
        self.hollow_detector = HollowStructureDetector(feature_dim)
        
        # 5. Transformer模块（调整为7x7特征图）
        self.transformer_blocks = nn.ModuleList([
            MicroTransformerBlock(feature_dim, num_heads=4, dropout=dropout_rate, feature_size=7)
            for _ in range(2)
        ])
        
        # 6. 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 7. 多任务头部
        self.multitask_heads = MultitaskHeads(
            feature_dim=feature_dim,
            dropout_rate=dropout_rate
        )
        
        # 8. 特征融合模块（用于精细分类）
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim + 3 + 9 + 5, feature_dim),  # 特征 + 各任务输出
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 9. 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward_features(self, x):
        """提取特征"""
        # 输入处理
        x = self.gray_stem(x)  # (B, 32, 35, 35)
        
        # CNN特征提取
        features = []
        for i, stage in enumerate(self.cnn_backbone):
            x = stage(x)
            features.append(x)
        
        # 底纹过滤
        if self.enable_background_filter:
            x, bg_attention = self.bg_filter(x)
        else:
            bg_attention = None
        
        # 中空结构检测
        hollow_info = self.hollow_detector(x)
        
        # Transformer处理
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        return x, hollow_info, bg_attention, features
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, 1, 70, 70) 灰度图像
            
        Returns:
            Dict: 包含所有任务输出的字典
        """
        # 确保输入是灰度的
        if x.shape[1] == 3:
            # 如果是RGB，转换为灰度
            x = torch.mean(x, dim=1, keepdim=True)
        
        # 特征提取
        features, hollow_info, bg_attention, cnn_features = self.forward_features(x)
        
        # 全局池化
        global_feat = self.global_pool(features).flatten(1)  # (B, feature_dim)
        
        # 多任务预测
        task_outputs = self.multitask_heads(global_feat, hollow_info, bg_attention)
        
        # 获取各任务的概率分布
        growth_level_probs = F.softmax(task_outputs['growth_level'], dim=1)
        growth_pattern_probs = F.softmax(task_outputs['growth_pattern'], dim=1)
        interference_probs = torch.sigmoid(task_outputs['interference_mapping'])
        
        # 特征融合用于精细分类
        fused_features = torch.cat([
            global_feat,
            growth_level_probs,
            growth_pattern_probs,
            interference_probs
        ], dim=1)
        
        fused_features = self.fusion_layer(fused_features)
        
        # 使用融合特征重新预测精细分类
        fine_grained_refined = self.multitask_heads.fine_grained_head(fused_features)
        task_outputs['fine_grained_refined'] = fine_grained_refined
        
        # 整合输出
        outputs = {
            # 原始任务输出
            'growth_level': task_outputs['growth_level'],
            'growth_pattern': task_outputs['growth_pattern'],
            'interference_mapping': task_outputs['interference_mapping'],
            'fine_grained': task_outputs['fine_grained'],
            'fine_grained_refined': fine_grained_refined,
            
            # 辅助信息
            'pore_confidence': task_outputs['pore_confidence'],
            'bg_confidence': task_outputs['bg_confidence'],
            
            # 特征信息
            'features': features,
            'hollow_detection': hollow_info,
            'background_attention': bg_attention,
            'cnn_features': cnn_features,
            
            # 概率分布
            'growth_level_probs': growth_level_probs,
            'growth_pattern_probs': growth_pattern_probs,
            'interference_probs': interference_probs
        }
        
        # 添加背景强度（如果有）
        if 'bg_strength' in task_outputs:
            outputs['bg_strength'] = task_outputs['bg_strength']
        
        return outputs
    
    def get_task_predictions(self, outputs: Dict[str, torch.Tensor], 
                           thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        获取任务预测结果（包含类别名称）
        
        Args:
            outputs: 模型输出
            thresholds: 各任务的阈值（主要用于多标签任务）
            
        Returns:
            包含预测类别和置信度的字典
        """
        if thresholds is None:
            thresholds = {
                'interference': 0.5,
                'pore_confidence': 0.5,
                'bg_confidence': 0.5
            }
        
        predictions = {}
        
        # 生长级别预测
        growth_level_idx = outputs['growth_level_probs'].argmax(dim=1)
        growth_level_names = ['negative', 'positive', 'weak_growth']
        predictions['growth_level'] = {
            'class': [growth_level_names[i] for i in growth_level_idx],
            'confidence': outputs['growth_level_probs'].max(dim=1)[0],
            'probabilities': outputs['growth_level_probs']
        }
        
        # 生长模式预测
        growth_pattern_idx = outputs['growth_pattern_probs'].argmax(dim=1)
        growth_pattern_names = [
            'clean', 'clustered', 'scattered', 'small_dots',
            'ring_shaped', 'irregular', 'mixed', 'sparse', 'dense'
        ]
        predictions['growth_pattern'] = {
            'class': [growth_pattern_names[i] for i in growth_pattern_idx],
            'confidence': outputs['growth_pattern_probs'].max(dim=1)[0],
            'probabilities': outputs['growth_pattern_probs']
        }
        
        # 干扰因素预测（多标签）
        interference_names = ['none', 'pores', 'debris', 'artifacts', 'contamination']
        interference_pred = (outputs['interference_probs'] > thresholds['interference']).cpu().numpy()
        predictions['interference_mapping'] = {
            'labels': [[interference_names[j] for j, present in enumerate(sample) if present] 
                      for sample in interference_pred],
            'probabilities': outputs['interference_probs']
        }
        
        # 精细分类预测
        fine_grained_idx = outputs['fine_grained_refined'].argmax(dim=1)
        fine_grained_names = [
            'positive_cluster_no_pores',
            'positive_cluster_with_pores',
            'positive_cluster_overlapping_pores',
            'negative_clean_no_pores',
            'negative_clean_with_pores',
            'weak_growth_center_no_pores',
            'weak_growth_center_with_pores',
            'weak_growth_center_overlapping_pores',
            'weak_growth_scattered_no_pores',
            'weak_growth_scattered_with_pores',
            'weak_growth_scattered_overlapping_pores',
            'with_debris',
            'with_artifacts',
            'contaminated',
            'other'
        ]
        predictions['fine_grained'] = {
            'class': [fine_grained_names[i] for i in fine_grained_idx],
            'confidence': F.softmax(outputs['fine_grained_refined'], dim=1).max(dim=1)[0],
            'probabilities': F.softmax(outputs['fine_grained_refined'], dim=1)
        }
        
        # 辅助信息
        predictions['auxiliary'] = {
            'pore_confidence': outputs['pore_confidence'],
            'bg_confidence': outputs['bg_confidence'],
            'has_pores': (outputs['pore_confidence'] > thresholds['pore_confidence']).squeeze().cpu().numpy().tolist()
        }
        
        if 'bg_strength' in outputs:
            predictions['auxiliary']['bg_strength'] = outputs['bg_strength']
        
        if 'hollow_score' in outputs['hollow_detection']:
            predictions['auxiliary']['hollow_score'] = outputs['hollow_detection']['hollow_score']
        
        return predictions
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MultitaskGrayColonyNet',
            'input_size': (1, 70, 70),
            'tasks': {
                'growth_level': {'classes': 3, 'type': 'single_label'},
                'growth_pattern': {'classes': 9, 'type': 'single_label'},
                'interference_mapping': {'classes': 5, 'type': 'multilabel'},
                'fine_grained': {'classes': 15, 'type': 'single_label'}
            },
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'architecture': 'CNN_Transformer_with_multitask_heads'
        }


def create_multitask_gray_colony_net(feature_dim: int = 128,
                                   enable_background_filter: bool = True,
                                   dropout_rate: float = 0.2) -> MultitaskGrayColonyNet:
    """创建多任务灰度菌落检测网络"""
    model = MultitaskGrayColonyNet(
        feature_dim=feature_dim,
        enable_background_filter=enable_background_filter,
        dropout_rate=dropout_rate
    )
    return model


# 测试代码
def test_model():
    """测试模型功能"""
    print("=== 测试多任务灰度菌落检测网络 ===")
    
    # 创建模型
    model = create_multitask_gray_colony_net()
    model_info = model.get_model_info()
    
    print(f"模型名称: {model_info['model_name']}")
    print(f"参数量: {model_info['total_parameters']:,}")
    print(f"输入尺寸: {model_info['input_size']}")
    print(f"任务配置: {model_info['tasks']}")
    
    # 测试前向传播
    dummy_input = torch.randn(2, 1, 70, 70)  # 灰度输入
    model.eval()
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"\n任务输出:")
    for task_name, output in outputs.items():
        if task_name in ['growth_level', 'growth_pattern', 'interference_mapping', 'fine_grained', 'fine_grained_refined']:
            print(f"  {task_name}: {output.shape}")
    
    print(f"\n辅助输出:")
    print(f"  pore_confidence: {outputs['pore_confidence'].shape}")
    print(f"  bg_confidence: {outputs['bg_confidence'].shape}")
    
    # 测试预测解析
    predictions = model.get_task_predictions(outputs)
    print(f"\n预测结果示例:")
    print(f"  生长级别: {predictions['growth_level']['class'][0]} (置信度: {predictions['growth_level']['confidence'][0]:.3f})")
    print(f"  生长模式: {predictions['growth_pattern']['class'][0]} (置信度: {predictions['growth_pattern']['confidence'][0]:.3f})")
    print(f"  干扰因素: {predictions['interference_mapping']['labels'][0]}")
    print(f"  精细分类: {predictions['fine_grained']['class'][0]}")
    print(f"  气孔置信度: {predictions['auxiliary']['pore_confidence'][0].item():.3f}")
    
    print(f"\n✓ 多任务灰度菌落检测网络测试通过")


if __name__ == "__main__":
    test_model()