"""
Enhanced MobileNetV5 Implementation with MIC-inspired improvements
Key improvements:
1. Optical Interference Suppressor (from MIC_MobileNetV3)
2. Turbidity Analysis Module
3. Simplified Multi-task Fusion
4. Optimized backbone for 70x70 images
5. Quality Assessment Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


class BrightnessNormalization(nn.Module):
    """Enhanced brightness normalization for varying lighting conditions"""
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        # Learnable normalization parameters
        self.norm_weight = nn.Parameter(torch.ones(1))
        self.norm_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Calculate global brightness statistics
        global_mean = self.adaptive_pool(x).mean(dim=[2, 3], keepdim=True)
        global_std = x.std(dim=[2, 3], keepdim=True)
        
        # Adaptive brightness normalization
        normalized = (x - global_mean) / (global_std + self.epsilon)
        
        # Apply learnable scaling and bias
        normalized = normalized * self.norm_weight + self.norm_bias
        
        return normalized


class OpticalInterferenceSuppressor(nn.Module):
    """
    Optical interference suppressor adapted from MIC_MobileNetV3
    Reduces bubble interference in colony detection
    """
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        self.suppression_weights = nn.Sequential(
            nn.Conv2d(in_channels + 1, 32, 3, padding=1),  # +1 for pore/bubble mask
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, pore_mask):
        # Combine features with pore mask
        combined = torch.cat([features, pore_mask], dim=1)
        
        # Generate suppression weights
        suppression = self.suppression_weights(combined)
        
        # Apply suppression (reduce features in bubble regions)
        suppressed_features = features * (1.0 - 0.7 * suppression)
        
        return suppressed_features


class TurbidityAnalysisModule(nn.Module):
    """
    Turbidity analysis module for MIC testing
    Provides additional information for colony detection
    """
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        self.turbidity_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.turbidity_extractor(x)


class AirPoreDetectionModule(nn.Module):
    """Simplified air pore detection module"""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # Center hollow detection
        self.center_hollow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 24, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 1, 1),
            nn.Sigmoid()
        )
        
        # Edge irregularity detection
        self.edge_irregularity = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # Simplified fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Detect pore features
        center_response = self.center_hollow_detector(x)
        edge_response = self.edge_irregularity(x)
        
        # Fuse responses
        combined = torch.cat([center_response, edge_response], dim=1)
        pore_mask = self.fusion(combined)
        
        return {
            'pore_mask': pore_mask,
            'center_hollow': center_response,
            'edge_irregularity': edge_response
        }


class SimpleFeatureFusion(nn.Module):
    """
    Simplified feature fusion mechanism
    Replaces the complex multi-task attention fusion
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features, pore_mask):
        # Simple mask-weighted fusion
        weighted_features = features * (1.0 - 0.5 * pore_mask)
        return self.fusion_conv(weighted_features)


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
    """Optimized MobileNetV5 Block"""
    
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
        
        return x


class EnhancedMobileNetV5(nn.Module):
    """
    Enhanced MobileNetV5 with MIC-inspired improvements
    Optimized for 70x70 colony detection
    """
    
    def __init__(self, num_classes: int = 2, input_size: int = 70, 
                 width_multiplier: float = 0.75, dropout_rate: float = 0.2,
                 enable_brightness_norm: bool = True,
                 enable_pore_detection: bool = True,
                 enable_turbidity_analysis: bool = True,
                 enable_quality_assessment: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Feature flags
        self.enable_brightness_norm = enable_brightness_norm
        self.enable_pore_detection = enable_pore_detection
        self.enable_turbidity_analysis = enable_turbidity_analysis
        self.enable_quality_assessment = enable_quality_assessment
        
        # Brightness normalization module
        if self.enable_brightness_norm:
            self.brightness_norm = BrightnessNormalization()
        
        # Adjust channel widths based on width multiplier (reduced for efficiency)
        def adjust_channels(channels):
            return max(8, int(channels * width_multiplier))
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, adjust_channels(16), 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(adjust_channels(16))
        
        # Optimized MobileNetV5 blocks (simplified architecture)
        self.blocks = nn.ModuleList([
            # Stage 1
            MBV5Block(adjust_channels(16), adjust_channels(16), stride=1, expand_ratio=2, use_se=False),
            MBV5Block(adjust_channels(16), adjust_channels(16), stride=1, expand_ratio=2, use_se=True),
            
            # Stage 2
            MBV5Block(adjust_channels(16), adjust_channels(24), stride=2, expand_ratio=3, use_se=True),
            MBV5Block(adjust_channels(24), adjust_channels(24), stride=1, expand_ratio=3, use_se=True),
            MBV5Block(adjust_channels(24), adjust_channels(24), stride=1, expand_ratio=3, use_se=True),
            
            # Stage 3
            MBV5Block(adjust_channels(24), adjust_channels(48), stride=2, expand_ratio=4, use_se=True),
            MBV5Block(adjust_channels(48), adjust_channels(48), stride=1, expand_ratio=4, use_se=True),
            MBV5Block(adjust_channels(48), adjust_channels(48), stride=1, expand_ratio=4, use_se=True),
        ])
        
        # Feature dimension after backbone (reduced from 192 to 48)
        feature_dim = adjust_channels(48)
        
        # Specialized detection modules
        if self.enable_pore_detection:
            self.pore_detector = AirPoreDetectionModule(feature_dim)
            self.optical_suppressor = OpticalInterferenceSuppressor(feature_dim)
        
        if self.enable_turbidity_analysis:
            self.turbidity_analyzer = TurbidityAnalysisModule(feature_dim)
        
        # Simplified feature fusion
        if self.enable_pore_detection:
            self.feature_fusion = SimpleFeatureFusion(feature_dim)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Main classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Quality assessment head
        if self.enable_quality_assessment:
            self.quality_head = nn.Sequential(
                nn.Linear(feature_dim, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 4)  # A, B, C, D quality grades
            )
        
        # Auxiliary classifier for pore detection
        if self.enable_pore_detection:
            self.pore_classifier = nn.Linear(feature_dim, 2)  # pore vs non-pore
        
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
        
        # Pore detection and optical interference suppression
        if self.enable_pore_detection:
            pore_results = self.pore_detector(features)
            results.update({f'pore_{k}': v for k, v in pore_results.items()})
            pore_mask = pore_results['pore_mask']
            
            # Apply optical interference suppression
            features = self.optical_suppressor(features, pore_mask)
        else:
            pore_mask = torch.zeros(features.size(0), 1, features.size(2), features.size(3)).to(features.device)
        
        # Turbidity analysis
        if self.enable_turbidity_analysis:
            turbidity_score = self.turbidity_analyzer(features)
            results['turbidity'] = turbidity_score
        
        # Simplified feature fusion
        if self.enable_pore_detection:
            features = self.feature_fusion(features, pore_mask)
        
        # Global average pooling
        pooled_features = self.global_pool(features).flatten(1)
        
        # Main classification
        classification_logits = self.classifier(pooled_features)
        results['classification'] = classification_logits
        
        # Quality assessment
        if self.enable_quality_assessment:
            quality_scores = self.quality_head(pooled_features)
            results['quality'] = quality_scores
        
        # Auxiliary classifier for pore detection
        if self.enable_pore_detection:
            pore_output = self.pore_classifier(pooled_features)
            results['pore_classification'] = pore_output
        
        return results
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'enhanced_mobilenetv5',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, self.input_size, self.input_size),
            'output_classes': self.num_classes,
            'features': {
                'brightness_normalization': self.enable_brightness_norm,
                'pore_detection': self.enable_pore_detection,
                'optical_suppression': self.enable_pore_detection,
                'turbidity_analysis': self.enable_turbidity_analysis,
                'quality_assessment': self.enable_quality_assessment,
                'simplified_fusion': self.enable_pore_detection
            }
        }


def create_enhanced_mobilenetv5(num_classes: int = 2, input_size: int = 70, **kwargs):
    """Factory function to create Enhanced MobileNetV5"""
    return EnhancedMobileNetV5(num_classes=num_classes, input_size=input_size, **kwargs)


def test_enhanced_mobilenetv5():
    """Test Enhanced MobileNetV5 implementation"""
    model = EnhancedMobileNetV5(num_classes=2, input_size=70)
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
    
    # Model info
    model_info = model.get_model_info()
    print(f"Model info: {model_info}")
    
    return model


if __name__ == "__main__":
    test_enhanced_mobilenetv5()