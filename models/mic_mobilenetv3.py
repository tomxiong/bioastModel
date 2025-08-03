"""
MIC-specific MobileNetV3 with SE attention for colony detection.

This model is specifically designed for MIC testing scenarios with:
- 70x70 small image optimization
- Air bubble detection and suppression
- Turbidity analysis capabilities
- Optical interference handling

Based on ideas.md specifications for MIC testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import math

class SEModule(nn.Module):
    """Squeeze-and-Excitation module optimized for small images."""
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        scale = self.global_pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale

class InvertedResidual(nn.Module):
    """Inverted Residual block with SE attention."""
    
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
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True) if activation == 'relu' else nn.Hardswish(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True) if activation == 'relu' else nn.Hardswish(inplace=True)
        ])
        
        # SE
        if use_se:
            layers.append(SEModule(hidden_dim))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class AirBubbleDetectionModule(nn.Module):
    """Specialized air bubble detection module for MIC testing."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # Ring structure detector
        self.ring_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Center darkness detector
        self.center_detector = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Edge irregularity detector
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Detect different bubble characteristics
        ring_response = self.ring_detector(x)
        center_response = self.center_detector(x)
        edge_response = self.edge_detector(x)
        
        # Fuse all responses
        combined = torch.cat([ring_response, center_response, edge_response], dim=1)
        bubble_mask = self.fusion(combined)
        
        return {
            'bubble_mask': bubble_mask,
            'ring_strength': ring_response,
            'center_darkness': center_response,
            'edge_irregularity': edge_response
        }

class TurbidityAnalysisModule(nn.Module):
    """Turbidity analysis module for MIC testing."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        self.turbidity_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
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

class OpticalInterferenceSuppressor(nn.Module):
    """Optical interference suppression module."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        self.suppression_weights = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 3, padding=1),  # +1 for bubble mask
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, bubble_mask):
        # Combine features with bubble mask
        combined = torch.cat([features, bubble_mask], dim=1)
        
        # Generate suppression weights
        suppression = self.suppression_weights(combined)
        
        # Apply suppression (reduce features in bubble regions)
        suppressed_features = features * (1.0 - 0.7 * suppression)
        
        return suppressed_features

class MIC_MobileNetV3(nn.Module):
    """
    MIC-specific MobileNetV3 for colony detection in 70x70 images.
    
    Features:
    - Optimized for small images (70x70)
    - Air bubble detection and suppression
    - Turbidity analysis
    - Optical interference handling
    - Multi-task learning capability
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        width_mult: float = 1.0,
        dropout_rate: float = 0.2,
        enable_bubble_detection: bool = True,
        enable_turbidity_analysis: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.enable_bubble_detection = enable_bubble_detection
        self.enable_turbidity_analysis = enable_turbidity_analysis
        
        # Calculate channel dimensions
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # Stem
        input_channel = make_divisible(16 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),  # 70x70 -> 35x35
            nn.BatchNorm2d(input_channel),
            nn.Hardswish(inplace=True)
        )
        
        # MobileNetV3-Small configuration adapted for 70x70
        # [kernel, exp_size, out_channels, use_se, activation, stride]
        mobile_setting = [
            [3, 16, 16, True, 'relu', 2],      # 35x35 -> 18x18
            [3, 72, 24, False, 'relu', 2],     # 18x18 -> 9x9
            [3, 88, 24, False, 'relu', 1],     # 9x9 -> 9x9
            [5, 96, 40, True, 'hardswish', 2], # 9x9 -> 5x5 (modified for small input)
            [5, 240, 40, True, 'hardswish', 1], # 5x5 -> 5x5
            [5, 240, 40, True, 'hardswish', 1], # 5x5 -> 5x5
            [5, 120, 48, True, 'hardswish', 1], # 5x5 -> 5x5
            [5, 144, 48, True, 'hardswish', 1], # 5x5 -> 5x5
            [5, 288, 96, True, 'hardswish', 2], # 5x5 -> 3x3 (modified)
            [5, 576, 96, True, 'hardswish', 1], # 3x3 -> 3x3
            [5, 576, 96, True, 'hardswish', 1], # 3x3 -> 3x3
        ]
        
        # Build inverted residual blocks
        features = []
        for k, exp_size, c, use_se, act, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp_size * width_mult)
            features.append(InvertedResidual(
                input_channel, output_channel, k, s, 
                exp_channel // input_channel, use_se, act
            ))
            input_channel = output_channel
        
        self.features = nn.Sequential(*features)
        
        # Feature dimension after backbone
        feature_dim = make_divisible(96 * width_mult)
        
        # MIC-specific modules
        if self.enable_bubble_detection:
            self.bubble_detector = AirBubbleDetectionModule(feature_dim)
            self.optical_suppressor = OpticalInterferenceSuppressor(feature_dim)
        
        if self.enable_turbidity_analysis:
            self.turbidity_analyzer = TurbidityAnalysisModule(feature_dim)
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)  # A, B, C, D quality grades
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
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
    
    def forward_features(self, x):
        """Extract features from input."""
        x = self.stem(x)
        x = self.features(x)
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 70, 70)
            
        Returns:
            Dictionary containing:
            - classification: Main classification logits
            - turbidity: Turbidity score (if enabled)
            - bubble_analysis: Bubble detection results (if enabled)
            - quality: Quality assessment scores
        """
        # Extract features
        features = self.forward_features(x)
        
        results = {}
        
        # Bubble detection and suppression
        if self.enable_bubble_detection:
            bubble_analysis = self.bubble_detector(features)
            results['bubble_analysis'] = bubble_analysis
            
            # Apply optical interference suppression
            features = self.optical_suppressor(features, bubble_analysis['bubble_mask'])
        
        # Turbidity analysis
        if self.enable_turbidity_analysis:
            turbidity_score = self.turbidity_analyzer(features)
            results['turbidity'] = turbidity_score
        
        # Global pooling for classification
        pooled_features = self.global_pool(features).flatten(1)
        
        # Main classification
        classification_logits = self.classifier(pooled_features)
        results['classification'] = classification_logits
        
        # Quality assessment
        quality_scores = self.quality_head(pooled_features)
        results['quality'] = quality_scores
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'mic_mobilenetv3',
            'architecture': 'mobilenetv3_mic',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes,
            'features': {
                'bubble_detection': self.enable_bubble_detection,
                'turbidity_analysis': self.enable_turbidity_analysis,
                'multi_task': True
            }
        }

def create_mic_mobilenetv3(
    num_classes: int = 2,
    model_size: str = 'small',
    **kwargs
) -> MIC_MobileNetV3:
    """
    Create MIC-specific MobileNetV3 model.
    
    Args:
        num_classes: Number of output classes
        model_size: Model size ('small', 'large')
        **kwargs: Additional arguments
        
    Returns:
        MIC_MobileNetV3: Initialized model
    """
    
    configs = {
        'small': {
            'width_mult': 1.0,
            'dropout_rate': 0.2
        },
        'large': {
            'width_mult': 1.25,
            'dropout_rate': 0.3
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    model = MIC_MobileNetV3(num_classes=num_classes, **config)
    return model

# Model configuration for integration
MODEL_CONFIG = {
    'name': 'mic_mobilenetv3',
    'architecture': 'mobilenetv3_mic',
    'create_function': create_mic_mobilenetv3,
    'default_params': {
        'num_classes': 2,
        'model_size': 'small',
        'dropout_rate': 0.2,
        'enable_bubble_detection': True,
        'enable_turbidity_analysis': True
    },
    'training_params': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    },
    'estimated_parameters': 2.5,
    'description': 'MIC-specific MobileNetV3 with air bubble detection and turbidity analysis'
}

if __name__ == "__main__":
    # Test model creation
    print("🔍 Testing MIC MobileNetV3 model creation...")
    
    model = create_mic_mobilenetv3()
    model_info = model.get_model_info()
    
    print(f"✅ Created {model_info['name']} with {model_info['total_parameters']:,} parameters")
    print(f"   Features: {model_info['features']}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 70, 70)
    model.eval()
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Classification output: {outputs['classification'].shape}")
    
    if 'turbidity' in outputs:
        print(f"   - Turbidity output: {outputs['turbidity'].shape}")
    
    if 'bubble_analysis' in outputs:
        print(f"   - Bubble mask: {outputs['bubble_analysis']['bubble_mask'].shape}")
    
    print(f"🎯 MIC MobileNetV3 ready for training!")