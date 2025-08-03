"""
Air Bubble Aware Hybrid Network for MIC Testing.

This model specifically addresses the membrane air bubble optical magnification
effects in MIC testing scenarios. It combines:
- Lightweight CNN for local texture features
- Micro-Transformer for global context
- Specialized air bubble detection and suppression
- Optical distortion correction
- Multi-task learning for comprehensive analysis

Based on ideas.md specifications for handling 70x70 MIC images with
membrane air bubble interference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math

class InvertedResidual(nn.Module):
    """Inverted Residual block for efficient feature extraction."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4
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
                nn.ReLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        ])
        
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

class RingStructureDetector(nn.Module):
    """Specialized detector for ring-like structures (air bubble edges)."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # Multi-scale ring detection
        self.ring_detectors = nn.ModuleList([
            self._create_ring_detector(in_channels, kernel_size=k) 
            for k in [3, 5, 7]
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(len(self.ring_detectors), 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def _create_ring_detector(self, in_channels: int, kernel_size: int) -> nn.Module:
        """Create a ring detection convolution."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply multi-scale ring detection
        ring_responses = []
        for detector in self.ring_detectors:
            response = detector(x)
            ring_responses.append(response)
        
        # Fuse responses
        combined = torch.cat(ring_responses, dim=1)
        ring_strength = self.fusion(combined)
        
        return ring_strength

class CenterDarkSpotDetector(nn.Module):
    """Detector for dark spots at bubble centers."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # Center bias - emphasize center regions
        self.register_buffer('center_bias', self._create_center_bias())
    
    def _create_center_bias(self) -> torch.Tensor:
        """Create center-biased weight map."""
        size = 9  # Assuming 9x9 feature maps at this stage
        center = size // 2
        bias = torch.zeros(1, 1, size, size)
        
        for i in range(size):
            for j in range(size):
                dist = math.sqrt((i - center)**2 + (j - center)**2)
                bias[0, 0, i, j] = math.exp(-dist / 2.0)  # Gaussian center bias
        
        return bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        center_response = self.detector(x)
        
        # Apply center bias
        if center_response.shape[-2:] == self.center_bias.shape[-2:]:
            center_response = center_response * self.center_bias
        
        return center_response

class EdgeIrregularityDetector(nn.Module):
    """Detector for irregular edges (characteristic of air bubbles)."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # Edge detection
        self.edge_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Irregularity analysis
        self.irregularity_analyzer = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detect edges
        edge_features = self.edge_detector(x)
        
        # Analyze irregularity
        irregularity = self.irregularity_analyzer(edge_features)
        
        return irregularity

class AirBubbleDetectionModule(nn.Module):
    """Comprehensive air bubble detection module."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # Individual detectors
        self.ring_detector = RingStructureDetector(in_channels)
        self.center_detector = CenterDarkSpotDetector(in_channels)
        self.edge_detector = EdgeIrregularityDetector(in_channels)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Individual detections
        ring_strength = self.ring_detector(x)
        center_darkness = self.center_detector(x)
        edge_irregularity = self.edge_detector(x)
        
        # Combine features
        combined_features = torch.cat([ring_strength, center_darkness, edge_irregularity], dim=1)
        
        # Final bubble mask
        bubble_mask = self.fusion(combined_features)
        
        # Estimate confidence
        confidence = self.confidence_estimator(combined_features)
        
        return {
            'bubble_mask': bubble_mask,
            'ring_strength': ring_strength,
            'center_darkness': center_darkness,
            'edge_irregularity': edge_irregularity,
            'confidence': confidence
        }

class OpticalDistortionCorrector(nn.Module):
    """Optical distortion correction module."""
    
    def __init__(self, in_channels: int = 96):
        super().__init__()
        
        # Distortion field estimator
        self.distortion_estimator = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 3, padding=1),  # +1 for bubble mask
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)  # Output (dx, dy) distortion field
        )
        
        # Correction strength controller
        self.correction_controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels + 1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, bubble_mask: torch.Tensor) -> torch.Tensor:
        # Combine features with bubble mask
        combined = torch.cat([features, bubble_mask], dim=1)
        
        # Estimate distortion field
        distortion_field = self.distortion_estimator(combined)
        
        # Estimate correction strength
        correction_strength = self.correction_controller(combined)
        
        # Apply distortion correction (simplified version)
        corrected_features = self._apply_distortion_correction(
            features, distortion_field, correction_strength
        )
        
        return corrected_features
    
    def _apply_distortion_correction(
        self, 
        features: torch.Tensor, 
        distortion_field: torch.Tensor, 
        strength: torch.Tensor
    ) -> torch.Tensor:
        """Apply distortion correction to features."""
        # Simplified correction: weighted combination
        correction = F.conv2d(features, self._get_correction_kernel(features.shape[1]), padding=1, stride=1, groups=features.shape[1])
        corrected = features + strength.unsqueeze(-1).unsqueeze(-1) * correction
        
        return corrected
    
    def _get_correction_kernel(self, num_channels: int) -> torch.Tensor:
        """Get correction convolution kernel."""
        # Simple edge-preserving kernel
        kernel = torch.tensor([
            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]]
        ], dtype=torch.float32) / 8.0
        
        # Expand for all channels - kernel is already (1, 1, 3, 3)
        kernel = kernel.repeat(num_channels, 1, 1, 1)
        return kernel

class MicroTransformerBlock(nn.Module):
    """Lightweight transformer block for global context."""
    
    def __init__(
        self,
        dim: int = 96,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        drop: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class AirBubbleHybridNet(nn.Module):
    """
    Air Bubble Aware Hybrid Network for MIC Testing.
    
    This model specifically addresses membrane air bubble optical magnification
    effects in 70x70 MIC testing images.
    
    Architecture:
    1. Lightweight CNN backbone for local features
    2. Air bubble detection module
    3. Optical distortion correction
    4. Micro-transformer for global context
    5. Multi-task output heads
    """
    
    def __init__(
        self,
        num_classes: int = 4,  # Including air bubble interference class
        dropout_rate: float = 0.2,
        enable_distortion_correction: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.enable_distortion_correction = enable_distortion_correction
        
        # CNN Backbone - Lightweight design for 70x70 images
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # 70x70 -> 35x35
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 35x35 -> 18x18
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 64, stride=2),
            InvertedResidual(64, 64, stride=1)
        )
        
        # Stage 2: 18x18 -> 9x9
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 96, stride=2),
            InvertedResidual(96, 96, stride=1)
        )
        
        # Air bubble detection module
        self.bubble_detector = AirBubbleDetectionModule(96)
        
        # Optical distortion correction
        if self.enable_distortion_correction:
            self.distortion_corrector = OpticalDistortionCorrector(96)
        
        # Micro-transformer for global context (9x9 = 81 tokens)
        self.transformer_prep = nn.Sequential(
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        self.transformer_blocks = nn.ModuleList([
            MicroTransformerBlock(dim=96, num_heads=6, mlp_ratio=2.0)
            for _ in range(3)
        ])
        
        self.transformer_norm = nn.LayerNorm(96)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-task output heads
        
        # Main classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Turbidity regression head
        self.turbidity_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Air bubble parameter regression head
        self.bubble_param_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)  # center_x, center_y, radius, irregularity
        )
        
        # Quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)  # A, B, C, D grades
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def forward_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features."""
        x = self.stem(x)      # 70x70 -> 35x35
        x = self.stage1(x)    # 35x35 -> 18x18
        x = self.stage2(x)    # 18x18 -> 9x9
        return x
    
    def forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer blocks."""
        B, C, H, W = x.shape
        
        # Prepare for transformer: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)  # (B, 81, 96)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.transformer_norm(x)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with comprehensive air bubble handling.
        
        Args:
            x: Input tensor of shape (B, 3, 70, 70)
            
        Returns:
            Dictionary containing:
            - classification: Main classification logits
            - turbidity: Turbidity score
            - bubble_analysis: Comprehensive bubble analysis
            - bubble_params: Bubble parameter estimates
            - quality: Quality assessment scores
        """
        # Extract CNN features
        cnn_features = self.forward_cnn_features(x)  # (B, 96, 9, 9)
        
        # Air bubble detection
        bubble_analysis = self.bubble_detector(cnn_features)
        
        # Optical distortion correction
        corrected_features = cnn_features
        if self.enable_distortion_correction:
            corrected_features = self.distortion_corrector(
                cnn_features, bubble_analysis['bubble_mask']
            )
        
        # Apply transformer for global context
        transformer_features = self.forward_transformer(corrected_features)
        
        # Global pooling
        pooled_features = self.global_pool(transformer_features).flatten(1)  # (B, 96)
        
        # Multi-task outputs
        results = {
            'classification': self.classification_head(pooled_features),
            'turbidity': self.turbidity_head(pooled_features),
            'bubble_analysis': bubble_analysis,
            'bubble_params': self.bubble_param_head(pooled_features),
            'quality': self.quality_head(pooled_features)
        }
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'airbubble_hybrid_net',
            'architecture': 'cnn_transformer_hybrid',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes,
            'features': {
                'air_bubble_detection': True,
                'optical_distortion_correction': self.enable_distortion_correction,
                'turbidity_analysis': True,
                'quality_assessment': True,
                'multi_task': True,
                'transformer_blocks': len(self.transformer_blocks)
            }
        }

def create_airbubble_hybrid_net(
    num_classes: int = 4,
    model_size: str = 'base',
    **kwargs
) -> AirBubbleHybridNet:
    """
    Create Air Bubble Aware Hybrid Network.
    
    Args:
        num_classes: Number of output classes (including bubble interference)
        model_size: Model size ('base', 'large')
        **kwargs: Additional arguments
        
    Returns:
        AirBubbleHybridNet: Initialized model
    """
    
    configs = {
        'base': {
            'dropout_rate': 0.2,
            'enable_distortion_correction': True
        },
        'large': {
            'dropout_rate': 0.3,
            'enable_distortion_correction': True
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    model = AirBubbleHybridNet(num_classes=num_classes, **config)
    return model

# Model configuration for integration
MODEL_CONFIG = {
    'name': 'airbubble_hybrid_net',
    'architecture': 'cnn_transformer_hybrid',
    'create_function': create_airbubble_hybrid_net,
    'default_params': {
        'num_classes': 4,  # Including air bubble interference class
        'model_size': 'base',
        'dropout_rate': 0.2,
        'enable_distortion_correction': True
    },
    'training_params': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    },
    'estimated_parameters': 3.2,
    'description': 'Hybrid CNN-Transformer network with specialized air bubble detection and optical distortion correction for MIC testing'
}

if __name__ == "__main__":
    # Test model creation
    print("🔍 Testing Air Bubble Hybrid Network creation...")
    
    model = create_airbubble_hybrid_net()
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
    print(f"   - Turbidity output: {outputs['turbidity'].shape}")
    print(f"   - Bubble analysis keys: {list(outputs['bubble_analysis'].keys())}")
    print(f"   - Bubble params: {outputs['bubble_params'].shape}")
    print(f"   - Quality assessment: {outputs['quality'].shape}")
    
    print(f"🎯 Air Bubble Hybrid Network ready for training!")