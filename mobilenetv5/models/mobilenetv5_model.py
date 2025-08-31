"""
Improved MobileNetV5 Implementation
Optimized for 70x70 colony detection images with brightness invariance and pore detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
import random
import math


class BrightnessNormalization(nn.Module):
    """亮度归一化模块，解决不同采样亮度和菌液透光问题"""
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # 可学习的归一化参数
        self.norm_weight = nn.Parameter(torch.ones(1))
        self.norm_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # 计算全局亮度统计
        global_mean = self.adaptive_pool(x).mean(dim=[2, 3], keepdim=True)
        global_std = x.std(dim=[2, 3], keepdim=True)
        
        # 自适应亮度归一化
        normalized = (x - global_mean) / (global_std + self.epsilon)
        
        # 应用可学习的缩放和偏移
        normalized = normalized * self.norm_weight + self.norm_bias
        
        return normalized


class AirPoreDetectionModule(nn.Module):
    """气孔检测模块 - 检测中间有空心的不规则边缘特征"""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # 中心空心检测
        self.center_hollow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 边缘不规则性检测
        self.edge_irregularity = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 1, 1),
            nn.Sigmoid()
        )
        
        # 纹理复杂度检测
        self.texture_complexity = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 检测不同的气孔特征
        center_response = self.center_hollow_detector(x)  # 中心空心特征
        edge_response = self.edge_irregularity(x)         # 边缘不规则特征
        texture_response = self.texture_complexity(x)    # 纹理复杂度特征
        
        # 融合所有响应
        combined = torch.cat([center_response, edge_response, texture_response], dim=1)
        pore_mask = self.fusion(combined)
        
        return {
            'pore_mask': pore_mask,
            'center_hollow': center_response,
            'edge_irregularity': edge_response,
            'texture_complexity': texture_response
        }


class ColonyFeatureEnhancement(nn.Module):
    """菌落特征增强模块 - 增强规则边缘和圆形/椭圆形聚集特征"""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # 规则边缘检测
        self.regular_edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # 圆形/椭圆形形状检测
        self.shape_detector = nn.Sequential(
            nn.Conv2d(in_channels, 24, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 1, 1),
            nn.Sigmoid()
        )
        
        # 聚集密度检测
        self.density_detector = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # 特征增强
        self.enhancement = nn.Sequential(
            nn.Conv2d(in_channels + 3, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 检测菌落特征
        edge_response = self.regular_edge_detector(x)   # 规则边缘
        shape_response = self.shape_detector(x)         # 圆形/椭圆形
        density_response = self.density_detector(x)     # 聚集密度
        
        # 拼接特征用于增强
        enhanced_features = torch.cat([x, edge_response, shape_response, density_response], dim=1)
        enhanced_features = self.enhancement(enhanced_features)
        
        return {
            'enhanced_features': enhanced_features,
            'regular_edges': edge_response,
            'circular_shapes': shape_response,
            'density': density_response
        }


class MultiTaskAttentionFusion(nn.Module):
    """多任务注意力融合模块"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # 注意力权重生成
        self.attention_weights = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, 3, padding=1),  # +2 for pore and colony masks
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1),  # pore and colony attention
            nn.Softmax(dim=1)
        )
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, pore_mask, colony_mask):
        # 生成注意力权重
        attention_input = torch.cat([features, pore_mask, colony_mask], dim=1)
        attention = self.attention_weights(attention_input)
        
        # 应用注意力
        pore_attention = attention[:, 0:1, :, :]  # [B, 1, H, W]
        colony_attention = attention[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 注意力加权特征
        pore_weighted = features * pore_attention
        colony_weighted = features * colony_attention
        
        # 特征融合
        fused_features = torch.cat([pore_weighted, colony_weighted], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        return fused_features


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class MBV5Block(nn.Module):
    """MobileNetV5 Block with improved architecture"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, expand_ratio: int = 4, use_se: bool = True,
                 use_hs: bool = True):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False) if expand_ratio != 1 else None
        self.expand_bn = nn.BatchNorm2d(expanded_channels) if expand_ratio != 1 else None
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze and Excitation
        self.se = SEBlock(expanded_channels) if use_se else None
        
        # Output projection
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Activation function
        self.use_hs = use_hs  # Hard Swish vs ReLU
        
        # Dropout
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.expand_bn(self.expand_conv(x))
            if self.use_hs:
                x = x * (x + 3) / 6  # Hard Swish
            else:
                x = F.relu(x)
        
        # Depthwise
        x = self.depthwise_bn(self.depthwise_conv(x))
        if self.use_hs:
            x = x * (x + 3) / 6
        else:
            x = F.relu(x)
        
        # Squeeze and Excitation
        if self.se is not None:
            x = self.se(x)
        
        # Project
        x = self.project_bn(self.project_conv(x))
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        # Dropout
        x = self.dropout(x)
        
        return x


class MobileNetV5(nn.Module):
    """Improved MobileNetV5 optimized for 70x70 colony detection with brightness invariance"""
    
    def __init__(self, num_classes: int = 2, input_size: int = 70, 
                 width_multiplier: float = 1.0, dropout_rate: float = 0.2,
                 enable_brightness_norm: bool = True,
                 enable_pore_detection: bool = True,
                 enable_colony_enhancement: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Feature flags
        self.enable_brightness_norm = enable_brightness_norm
        self.enable_pore_detection = enable_pore_detection
        self.enable_colony_enhancement = enable_colony_enhancement
        
        # Brightness normalization module
        if self.enable_brightness_norm:
            self.brightness_norm = BrightnessNormalization()
        
        # Adjust channel widths based on width multiplier
        def adjust_channels(channels):
            return int(channels * width_multiplier)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, adjust_channels(24), 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(adjust_channels(24))
        
        # MobileNetV5 blocks
        self.blocks = nn.ModuleList([
            # Stage 1
            MBV5Block(adjust_channels(24), adjust_channels(24), stride=1, expand_ratio=2, use_se=False),
            MBV5Block(adjust_channels(24), adjust_channels(24), stride=1, expand_ratio=2, use_se=True),
            
            # Stage 2
            MBV5Block(adjust_channels(24), adjust_channels(48), stride=2, expand_ratio=4, use_se=True),
            MBV5Block(adjust_channels(48), adjust_channels(48), stride=1, expand_ratio=4, use_se=True),
            MBV5Block(adjust_channels(48), adjust_channels(48), stride=1, expand_ratio=4, use_se=True),
            
            # Stage 3
            MBV5Block(adjust_channels(48), adjust_channels(96), stride=2, expand_ratio=4, use_se=True),
            MBV5Block(adjust_channels(96), adjust_channels(96), stride=1, expand_ratio=4, use_se=True),
            MBV5Block(adjust_channels(96), adjust_channels(96), stride=1, expand_ratio=4, use_se=True),
            
            # Stage 4
            MBV5Block(adjust_channels(96), adjust_channels(192), stride=1, expand_ratio=6, use_se=True),
            MBV5Block(adjust_channels(192), adjust_channels(192), stride=1, expand_ratio=6, use_se=True),
        ])
        
        # Feature dimension after backbone
        feature_dim = adjust_channels(192)
        
        # Specialized detection modules
        if self.enable_pore_detection:
            self.pore_detector = AirPoreDetectionModule(feature_dim)
        
        if self.enable_colony_enhancement:
            self.colony_enhancer = ColonyFeatureEnhancement(feature_dim)
        
        # Multi-task attention fusion
        if self.enable_pore_detection and self.enable_colony_enhancement:
            self.attention_fusion = MultiTaskAttentionFusion(feature_dim)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Main classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Auxiliary classifiers for multi-task learning
        if self.enable_pore_detection:
            self.pore_classifier = nn.Linear(feature_dim, 2)  # pore vs non-pore
            
        if self.enable_colony_enhancement:
            self.colony_classifier = nn.Linear(feature_dim, 2)  # colony vs non-colony
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
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
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        results = {}
        
        # Brightness normalization
        if self.enable_brightness_norm:
            x = self.brightness_norm(x)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # MobileNetV5 blocks
        for block in self.blocks:
            x = block(x)
        
        # Extract features for specialized processing
        features = x
        
        # Pore detection
        if self.enable_pore_detection:
            pore_results = self.pore_detector(features)
            results.update({f'pore_{k}': v for k, v in pore_results.items()})
            pore_mask = pore_results['pore_mask']
        else:
            pore_mask = torch.zeros(features.size(0), 1, features.size(2), features.size(3)).to(features.device)
        
        # Colony enhancement
        if self.enable_colony_enhancement:
            colony_results = self.colony_enhancer(features)
            results.update({f'colony_{k}': v for k, v in colony_results.items()})
            colony_mask = colony_results['density']  # Use density as colony mask
            features = colony_results['enhanced_features']
        else:
            colony_mask = torch.zeros(features.size(0), 1, features.size(2), features.size(3)).to(features.device)
        
        # Multi-task attention fusion
        if self.enable_pore_detection and self.enable_colony_enhancement:
            features = self.attention_fusion(features, pore_mask, colony_mask)
        
        # Global average pooling
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Main classifier
        main_output = self.dropout(pooled_features)
        main_output = self.classifier(main_output)
        results['classification'] = main_output
        
        # Auxiliary classifiers
        if self.enable_pore_detection:
            pore_output = self.pore_classifier(pooled_features)
            results['pore_classification'] = pore_output
        
        if self.enable_colony_enhancement:
            colony_output = self.colony_classifier(pooled_features)
            results['colony_classification'] = colony_output
        
        return results
    
    def get_feature_maps(self, x):
        """获取特征图用于可视化"""
        feature_maps = []
        
        # Brightness normalization
        if self.enable_brightness_norm:
            x = self.brightness_norm(x)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        feature_maps.append(x)
        
        # MobileNetV5 blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [2, 5, 8]:  # Save intermediate feature maps
                feature_maps.append(x)
        
        return feature_maps
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'mobilenetv5_improved',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, self.input_size, self.input_size),
            'output_classes': self.num_classes,
            'features': {
                'brightness_normalization': self.enable_brightness_norm,
                'pore_detection': self.enable_pore_detection,
                'colony_enhancement': self.enable_colony_enhancement,
                'multi_task': self.enable_pore_detection or self.enable_colony_enhancement
            }
        }


class MobileNetV5Small(MobileNetV5):
    """Smaller version of MobileNetV5 for faster inference"""
    
    def __init__(self, num_classes: int = 2, input_size: int = 70, 
                 width_multiplier: float = 0.75, dropout_rate: float = 0.1):
        super().__init__(num_classes, input_size, width_multiplier, dropout_rate)


def create_mobilenetv5(model_name: str = 'mobilenetv5', num_classes: int = 2, 
                      input_size: int = 70, **kwargs):
    """Factory function to create MobileNetV5 variants"""
    
    if model_name == 'mobilenetv5':
        return MobileNetV5(num_classes=num_classes, input_size=input_size, **kwargs)
    elif model_name == 'mobilenetv5_small':
        return MobileNetV5Small(num_classes=num_classes, input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unknown MobileNetV5 variant: {model_name}")


def test_mobilenetv5():
    """Test improved MobileNetV5 implementation"""
    model = MobileNetV5(num_classes=2, input_size=70)
    x = torch.randn(1, 3, 70, 70)
    
    # Forward pass
    outputs = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output keys: {list(outputs.keys())}")
    
    # Main classification output
    main_output = outputs['classification']
    print(f"Main classification shape: {main_output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test feature maps
    feature_maps = model.get_feature_maps(x)
    print(f"Number of feature maps: {len(feature_maps)}")
    for i, fmap in enumerate(feature_maps):
        print(f"Feature map {i}: {fmap.shape}")
    
    # Model info
    model_info = model.get_model_info()
    print(f"Model info: {model_info}")
    
    return model


if __name__ == "__main__":
    test_mobilenetv5()