"""
多任务增强MIC MobileNetV3 - 专注于生长模式和干扰因素的精确分类
针对灰度图像和细粒度分类优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import math

class CBAM(nn.Module):
    """Convolutional Block Attention Module for enhanced attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        
        channel_att = self.channel_attention(avg_pool) + self.channel_attention(max_pool)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        
        x = x * spatial_att
        return x

class EnhancedInvertedResidual(nn.Module):
    """Enhanced Inverted Residual block with attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        use_se: bool = True,
        activation: str = 'relu'
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Pointwise expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # SE module
        if use_se:
            layers.append(self._make_se_layer(hidden_dim))
        
        # Pointwise linear
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(0.1) if self.use_residual else None
    
    def _make_se_layer(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if self.use_residual:
            out = self.conv(x)
            if self.dropout is not None:
                out = self.dropout(out)
            return x + out
        else:
            return self.conv(x)

class GrowthPatternClassifier(nn.Module):
    """专门的生长模式分类器，针对细粒度边界情况优化"""
    
    def __init__(self, in_features: int, num_patterns: int):
        super().__init__()
        
        # 特征增强层，专注于边界特征
        self.feature_enhancer = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 边界敏感分类器 - 专门处理相似模式
        self.boundary_classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features // 2, num_patterns)
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征增强
        enhanced_features = self.feature_enhancer(x)
        # 残差连接
        enhanced_features = enhanced_features + x
        # 分类
        pattern_logits = self.boundary_classifier(enhanced_features)
        return pattern_logits

class InterferenceFactorClassifier(nn.Module):
    """专门的干扰因素多标签分类器，针对气孔相关干扰优化"""
    
    def __init__(self, in_features: int, num_factors: int = 4):
        super().__init__()
        
        # 气孔特征专门提取器
        self.pore_feature_extractor = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # 每个干扰因素的专门分类器
        self.factor_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(in_features // 4, 1)
            ) for _ in range(num_factors)
        ])
        
        # 因素间相关性建模
        self.correlation_layer = nn.Sequential(
            nn.Linear(num_factors, num_factors),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 气孔特征增强
        pore_features = self.pore_feature_extractor(x)
        enhanced_features = pore_features + x
        
        # 各因素独立预测
        factor_logits = []
        for classifier in self.factor_classifiers:
            logit = classifier(enhanced_features)
            factor_logits.append(logit)
        
        factor_logits = torch.cat(factor_logits, dim=1)
        
        # 相关性调整
        correlation_weights = self.correlation_layer(factor_logits)
        adjusted_logits = factor_logits + 0.1 * correlation_weights
        
        return adjusted_logits

class MultiTaskMICMobileNetV3(nn.Module):
    """
    多任务MIC MobileNetV3
    专注于生长模式和干扰因素的精确分类
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        num_growth_patterns: int = 12,
        num_interference_factors: int = 4,
        width_mult: float = 1.0,
        input_channels: int = 1  # 灰度图单通道
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_growth_patterns = num_growth_patterns
        self.num_interference_factors = num_interference_factors
        
        # 计算通道数
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # 灰度图输入的Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, _make_divisible(16 * width_mult), 3, 2, 1, bias=False),
            nn.BatchNorm2d(_make_divisible(16 * width_mult)),
            nn.ReLU6(inplace=True)
        )
        
        # MobileNetV3 配置 (适配70x70输入)
        # [input_channels, output_channels, kernel_size, stride, expand_ratio, use_se]
        self.configs = [
            [16, 16, 3, 1, 1, True],    # 70x70 -> 70x70
            [16, 24, 3, 2, 6, False],   # 70x70 -> 35x35
            [24, 24, 3, 1, 3, False],   # 35x35 -> 35x35
            [24, 40, 5, 2, 6, True],    # 35x35 -> 18x18
            [40, 40, 5, 1, 6, True],    # 18x18 -> 18x18
            [40, 40, 5, 1, 6, True],    # 18x18 -> 18x18
            [40, 80, 3, 2, 6, False],   # 18x18 -> 9x9
            [80, 80, 3, 1, 3, False],   # 9x9 -> 9x9 (修正expand_ratio)
            [80, 80, 3, 1, 2, False],   # 9x9 -> 9x9 (修正expand_ratio)
            [80, 80, 3, 1, 2, False],   # 9x9 -> 9x9 (修正expand_ratio)
            [80, 112, 3, 1, 6, True],   # 9x9 -> 9x9
            [112, 112, 3, 1, 6, True],  # 9x9 -> 9x9
            [112, 160, 5, 2, 6, True],  # 9x9 -> 5x5
            [160, 160, 5, 1, 6, True],  # 5x5 -> 5x5
            [160, 160, 5, 1, 6, True],  # 5x5 -> 5x5
        ]
        
        # 构建特征提取层
        layers = []
        input_channels = _make_divisible(16 * width_mult)
        
        for i, (in_c, out_c, k, s, e, se) in enumerate(self.configs):
            output_channels = _make_divisible(out_c * width_mult)
            layers.append(
                EnhancedInvertedResidual(
                    input_channels, output_channels, k, s, e, se
                )
            )
            input_channels = output_channels
        
        self.features = nn.Sequential(*layers)
        
        # 最终卷积层
        conv_out = _make_divisible(960 * width_mult)
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channels, conv_out, 1, bias=False),
            nn.BatchNorm2d(conv_out),
            nn.ReLU6(inplace=True)
        )
        
        # CBAM注意力
        self.cbam = CBAM(conv_out)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 主干特征维度
        feature_dim = conv_out
        
        # 任务特定的分类器
        # 1. 主分类任务 (阴性/阳性)
        self.main_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )
        
        # 2. 辅助分类任务
        self.aux_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_classes)
        )
        
        # 3. 生长模式精细分类器
        self.growth_pattern_classifier = GrowthPatternClassifier(
            feature_dim, num_growth_patterns
        )
        
        # 4. 干扰因素多标签分类器  
        self.interference_classifier = InterferenceFactorClassifier(
            feature_dim, num_interference_factors
        )
        
        # 5. 质量评估
        self.quality_regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        """提取特征"""
        x = self.stem(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.cbam(x)  # 应用注意力
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量 [B, 1, 70, 70] (灰度图)
        Returns:
            包含所有任务输出的字典
        """
        # 特征提取
        features = self.forward_features(x)
        pooled_features = self.global_pool(features).flatten(1)
        
        outputs = {}
        
        # 主分类任务
        outputs['classification'] = self.main_classifier(pooled_features)
        
        # 辅助分类任务
        outputs['aux_classification'] = self.aux_classifier(pooled_features)
        
        # 生长模式分类 (重点优化)
        outputs['growth_pattern'] = self.growth_pattern_classifier(pooled_features)
        
        # 干扰因素分类 (重点优化)
        outputs['interference_factors'] = self.interference_classifier(pooled_features)
        
        # 质量评估
        outputs['quality'] = self.quality_regressor(pooled_features)
        
        return outputs

def create_multitask_mic_mobilenetv3(
    num_classes: int = 2,
    num_growth_patterns: int = 12,
    num_interference_factors: int = 4,
    width_mult: float = 1.0,
    **kwargs
) -> MultiTaskMICMobileNetV3:
    """
    创建多任务MIC MobileNetV3模型
    
    Args:
        num_classes: 主分类类别数 (默认2: 阴性/阳性)
        num_growth_patterns: 生长模式类别数 (默认12)
        num_interference_factors: 干扰因素数量 (默认4)
        width_mult: 宽度倍增因子
    """
    model = MultiTaskMICMobileNetV3(
        num_classes=num_classes,
        num_growth_patterns=num_growth_patterns,
        num_interference_factors=num_interference_factors,
        width_mult=width_mult,
        input_channels=1  # 灰度图单通道
    )
    return model

if __name__ == "__main__":
    # 测试模型
    model = create_multitask_mic_mobilenetv3()
    
    # 测试输入 (灰度图)
    x = torch.randn(4, 1, 70, 70)
    outputs = model(x)
    
    print("模型输出:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\\n参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")