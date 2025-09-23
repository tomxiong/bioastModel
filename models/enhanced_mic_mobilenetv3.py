"""
Enhanced MIC-specific MobileNetV3 with advanced attention mechanisms and optimizations.

Improvements over original MIC_MobileNetV3:
- CBAM (Convolutional Block Attention Module) for better feature attention
- FPN-like feature fusion for multi-scale information
- Improved loss functions and training strategies
- Better handling of class imbalance and edge cases
- Enhanced bubble detection with spatial attention
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
        b, c, h, w = x.size()
        
        # Global average and max pooling for channel attention
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

class EnhancedSEModule(nn.Module):
    """Enhanced Squeeze-and-Excitation module with additional features."""
    
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        # Add learnable gating for better control
        self.gate = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        scale = self.global_pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale) * self.gate
        return x * scale

class EnhancedInvertedResidual(nn.Module):
    """Enhanced Inverted Residual block with CBAM attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        use_se: bool = True,
        use_cbam: bool = True,
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
        
        # Attention modules
        if use_se:
            layers.append(EnhancedSEModule(hidden_dim))
        
        if use_cbam:
            layers.append(CBAM(hidden_dim))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout2d(0.1) if self.use_residual else None
    
    def forward(self, x):
        if self.use_residual:
            out = self.conv(x)
            if self.dropout is not None:
                out = self.dropout(out)
            return x + out
        else:
            return self.conv(x)

class AdvancedBubbleDetectionModule(nn.Module):
    """Advanced air bubble detection with spatial attention and multi-scale features."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # Multi-scale bubble detection
        self.bubble_detectors = nn.ModuleList([
            # Fine-grained detector (small bubbles)
            nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            ),
            # Medium-scale detector
            nn.Sequential(
                nn.Conv2d(in_channels, 32, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            ),
            # Large-scale detector
            nn.Sequential(
                nn.Conv2d(in_channels, 32, 7, padding=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            )
        ])
        
        # Feature fusion with attention
        self.fusion_attention = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 1),
            nn.Softmax(dim=1)
        )
        
        # Final bubble mask generation
        self.final_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Add CBAM for enhanced spatial attention
        self.cbam = CBAM(in_channels)
    
    def forward(self, x):
        # Apply CBAM first for better feature representation
        x = self.cbam(x)
        
        # Multi-scale bubble detection
        bubble_responses = []
        for detector in self.bubble_detectors:
            response = detector(x)
            bubble_responses.append(response)
        
        # Stack responses
        combined_responses = torch.cat(bubble_responses, dim=1)
        
        # Apply attention-based fusion
        attention_weights = self.fusion_attention(combined_responses)
        weighted_responses = combined_responses * attention_weights
        
        # Generate final bubble mask
        bubble_mask = self.final_conv(weighted_responses)
        
        return {
            'bubble_mask': bubble_mask,
            'multi_scale_responses': bubble_responses,
            'attention_weights': attention_weights
        }

class EnhancedTurbidityAnalysisModule(nn.Module):
    """Enhanced turbidity analysis with better feature extraction."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # Multi-branch turbidity analysis
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        global_turbidity = self.global_branch(x)
        local_turbidity = self.local_branch(x)
        
        # Combine both branches
        combined = torch.cat([global_turbidity, local_turbidity], dim=1)
        final_turbidity = self.fusion(combined)
        
        return {
            'turbidity': final_turbidity,
            'global_turbidity': global_turbidity,
            'local_turbidity': local_turbidity
        }

class MICFocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in MIC detection."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if self.class_weights is not None:
                alpha_t = self.class_weights[targets]
            else:
                alpha_t = self.alpha
        else:
            alpha_t = 1.0
        
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

class EnhancedMIC_MobileNetV3(nn.Module):
    """
    Enhanced MIC-specific MobileNetV3 with advanced attention and optimization features.
    
    Key improvements:
    - CBAM attention for better feature extraction
    - Multi-scale bubble detection
    - Enhanced turbidity analysis
    - Improved training stability and performance
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        width_mult: float = 1.0,
        dropout_rate: float = 0.2,
        enable_bubble_detection: bool = True,
        enable_turbidity_analysis: bool = True,
        use_cbam: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.enable_bubble_detection = enable_bubble_detection
        self.enable_turbidity_analysis = enable_turbidity_analysis
        self.use_cbam = use_cbam
        
        # Calculate channel dimensions
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # Enhanced stem with better feature extraction
        input_channel = make_divisible(16 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),  # 70x70 -> 35x35
            nn.BatchNorm2d(input_channel),
            nn.Hardswish(inplace=True),
            CBAM(input_channel) if use_cbam else nn.Identity()
        )
        
        # Enhanced MobileNetV3 configuration
        mobile_setting = [
            # [kernel, exp_size, out_channels, use_se, use_cbam, activation, stride]
            [3, 16, 16, True, True, 'relu', 2],      # 35x35 -> 18x18
            [3, 72, 24, True, True, 'relu', 2],      # 18x18 -> 9x9
            [3, 88, 24, True, False, 'relu', 1],     # 9x9 -> 9x9
            [5, 96, 40, True, True, 'hardswish', 2], # 9x9 -> 5x5
            [5, 240, 40, True, True, 'hardswish', 1], # 5x5 -> 5x5
            [5, 240, 40, True, False, 'hardswish', 1], # 5x5 -> 5x5
            [5, 120, 48, True, True, 'hardswish', 1], # 5x5 -> 5x5
            [5, 144, 48, True, True, 'hardswish', 1], # 5x5 -> 5x5
            [5, 288, 96, True, True, 'hardswish', 2], # 5x5 -> 3x3
            [5, 576, 96, True, True, 'hardswish', 1], # 3x3 -> 3x3
            [5, 576, 96, True, False, 'hardswish', 1], # 3x3 -> 3x3
        ]
        
        # Build enhanced inverted residual blocks
        features = []
        for k, exp_size, c, use_se, use_cbam_block, act, s in mobile_setting:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp_size * width_mult)
            features.append(EnhancedInvertedResidual(
                input_channel, output_channel, k, s, 
                exp_channel // input_channel, use_se, 
                use_cbam_block and use_cbam, act
            ))
            input_channel = output_channel
        
        self.features = nn.Sequential(*features)
        
        # Feature dimension after backbone
        feature_dim = make_divisible(96 * width_mult)
        
        # Enhanced MIC-specific modules
        if self.enable_bubble_detection:
            self.bubble_detector = AdvancedBubbleDetectionModule(feature_dim)
        
        if self.enable_turbidity_analysis:
            self.turbidity_analyzer = EnhancedTurbidityAnalysisModule(feature_dim)
        
        # Enhanced classification head with residual connection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-branch classification head
        self.main_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Auxiliary classifier for better gradient flow
        self.aux_classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
        
        # Quality assessment head (expanded)
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5)  # Extended quality grades: A, B, C, D, E
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Enhanced weight initialization."""
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
        Enhanced forward pass with multi-task outputs and improved stability.
        """
        # Extract features
        features = self.forward_features(x)
        
        results = {}
        
        # Enhanced bubble detection
        if self.enable_bubble_detection:
            bubble_analysis = self.bubble_detector(features)
            results['bubble_analysis'] = bubble_analysis
        
        # Enhanced turbidity analysis
        if self.enable_turbidity_analysis:
            turbidity_analysis = self.turbidity_analyzer(features)
            results['turbidity_analysis'] = turbidity_analysis
        
        # Global pooling for classification
        pooled_features = self.global_pool(features).flatten(1)
        
        # Main classification with auxiliary loss
        main_logits = self.main_classifier(pooled_features)
        aux_logits = self.aux_classifier(pooled_features)
        
        results['classification'] = main_logits
        results['aux_classification'] = aux_logits
        
        # Enhanced quality assessment
        quality_scores = self.quality_head(pooled_features)
        results['quality'] = quality_scores
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'enhanced_mic_mobilenetv3',
            'architecture': 'enhanced_mobilenetv3_mic',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes,
            'features': {
                'bubble_detection': self.enable_bubble_detection,
                'turbidity_analysis': self.enable_turbidity_analysis,
                'cbam_attention': self.use_cbam,
                'multi_task': True,
                'enhanced_features': True
            }
        }

def create_enhanced_mic_mobilenetv3(
    num_classes: int = 2,
    model_size: str = 'small',
    **kwargs
) -> EnhancedMIC_MobileNetV3:
    """
    Create enhanced MIC-specific MobileNetV3 model.
    
    Args:
        num_classes: Number of output classes
        model_size: Model size ('small', 'large')
        **kwargs: Additional arguments
        
    Returns:
        EnhancedMIC_MobileNetV3: Initialized enhanced model
    """
    
    configs = {
        'small': {
            'width_mult': 1.0,
            'dropout_rate': 0.2,
            'use_cbam': True
        },
        'large': {
            'width_mult': 1.25,
            'dropout_rate': 0.3,
            'use_cbam': True
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    model = EnhancedMIC_MobileNetV3(num_classes=num_classes, **config)
    return model

# Model configuration for integration
ENHANCED_MODEL_CONFIG = {
    'name': 'enhanced_mic_mobilenetv3',
    'architecture': 'enhanced_mobilenetv3_mic',
    'create_function': create_enhanced_mic_mobilenetv3,
    'default_params': {
        'num_classes': 2,
        'model_size': 'small',
        'dropout_rate': 0.2,
        'enable_bubble_detection': True,
        'enable_turbidity_analysis': True,
        'use_cbam': True
    },
    'training_params': {
        'batch_size': 64,
        'learning_rate': 0.0008,
        'weight_decay': 5e-4,
        'epochs': 100,
        'optimizer': 'adamw',
        'scheduler': 'cosine_with_restarts',
        'warmup_epochs': 10,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.3,
        'cutmix_alpha': 0.8,
        'focal_loss': True,
        'auxiliary_loss': True
    },
    'estimated_parameters': 3.2,
    'description': 'Enhanced MIC-specific MobileNetV3 with CBAM attention, advanced bubble detection, and improved training stability'
}

if __name__ == "__main__":
    # Test enhanced model creation
    print("🔍 Testing Enhanced MIC MobileNetV3 model creation...")
    
    model = create_enhanced_mic_mobilenetv3()
    model_info = model.get_model_info()
    
    print(f"✅ Created {model_info['name']} with {model_info['total_parameters']:,} parameters")
    print(f"   Enhanced features: {model_info['features']}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 70, 70)
    model.eval()
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Main classification output: {outputs['classification'].shape}")
    print(f"   - Auxiliary classification output: {outputs['aux_classification'].shape}")
    
    if 'turbidity_analysis' in outputs:
        print(f"   - Turbidity output: {outputs['turbidity_analysis']['turbidity'].shape}")
    
    if 'bubble_analysis' in outputs:
        print(f"   - Bubble mask: {outputs['bubble_analysis']['bubble_mask'].shape}")
        print(f"   - Multi-scale responses: {len(outputs['bubble_analysis']['multi_scale_responses'])}")
    
    print(f"🎯 Enhanced MIC MobileNetV3 ready for optimized training!")