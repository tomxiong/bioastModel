#!/usr/bin/env python3
"""
EfficientNet V2 Micro - Optimized for 70x70 biomedical images
Based on EfficientNetV2 but scaled down for smaller input sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MBConvBlockMicro(nn.Module):
    """Micro Mobile Inverted Bottleneck Convolution Block for 70x70 images"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, expand_ratio: int, se_ratio: float = 0.25,
                 drop_rate: float = 0.0):
        super().__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride, kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Point-wise convolution
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout
        if drop_rate > 0:
            self.dropout = nn.Dropout2d(drop_rate)
        else:
            self.dropout = None
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        
        # Squeeze and Excitation
        if self.se is not None:
            x = x * self.se(x)
        
        # Project
        x = self.project_conv(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x

class FusedMBConvBlockMicro(nn.Module):
    """Fused Mobile Inverted Bottleneck Convolution Block for 70x70 images"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, expand_ratio: int, se_ratio: float = 0.0,
                 drop_rate: float = 0.0):
        super().__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        
        expanded_channels = in_channels * expand_ratio
        
        # Fused expansion and depthwise
        self.fused_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size, 
                     stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Point-wise convolution (if needed)
        if expanded_channels != out_channels:
            self.project_conv = nn.Sequential(
                nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.project_conv = nn.Identity()
        
        # Dropout
        if drop_rate > 0:
            self.dropout = nn.Dropout2d(drop_rate)
        else:
            self.dropout = None
    
    def forward(self, x):
        identity = x
        
        # Fused convolution
        x = self.fused_conv(x)
        
        # Squeeze and Excitation
        if self.se is not None:
            x = x * self.se(x)
        
        # Project
        x = self.project_conv(x)
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x

class EfficientNetV2Micro(nn.Module):
    """EfficientNet V2 Micro - Optimized for 70x70 biomedical images"""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Stem - optimized for 70x70 input
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1, bias=False),  # Keep 70x70
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True)
        )
        
        # Blocks - micro configuration for 70x70
        self.blocks = nn.ModuleList([
            # Stage 1: Fused blocks
            FusedMBConvBlockMicro(24, 24, 3, 1, 1, 0.0),  # 70x70
            FusedMBConvBlockMicro(24, 48, 3, 2, 4, 0.0),  # 35x35
            
            # Stage 2: Fused blocks
            FusedMBConvBlockMicro(48, 48, 3, 1, 4, 0.0),  # 35x35
            FusedMBConvBlockMicro(48, 64, 3, 2, 4, 0.0),  # 17x17
            
            # Stage 3: MBConv blocks with SE
            MBConvBlockMicro(64, 64, 3, 1, 4, 0.25),      # 17x17
            MBConvBlockMicro(64, 96, 3, 2, 6, 0.25),      # 8x8
            
            # Stage 4: MBConv blocks with SE
            MBConvBlockMicro(96, 96, 3, 1, 6, 0.25),      # 8x8
            MBConvBlockMicro(96, 160, 3, 2, 6, 0.25),     # 4x4
        ])
        
        # Head
        self.head_conv = nn.Sequential(
            nn.Conv2d(160, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.SiLU(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(320, num_classes)
        
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
        # Validate input size
        if x.shape[-2:] != (70, 70):
            raise ValueError(f"Expected input size (70, 70), got {x.shape[-2:]}")
        
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'efficientnet_v2_micro',
            'architecture': 'efficientnet_v2',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes
        }

def create_efficientnet_v2_micro(num_classes: int = 2, **kwargs) -> EfficientNetV2Micro:
    """Create EfficientNet V2 Micro model optimized for 70x70 images"""
    return EfficientNetV2Micro(num_classes=num_classes, **kwargs)

# Alias for compatibility
EfficientnetV2 = EfficientNetV2Micro

if __name__ == "__main__":
    # Test model creation
    print("🔍 Testing EfficientNet V2 Micro model creation...")
    model = create_efficientnet_v2_micro()
    
    model_info = model.get_model_info()
    print(f"✅ Created {model_info['name']} with {model_info['total_parameters']:,} parameters")
    print(f"📊 Architecture: {model_info['architecture']}")
    print(f"   - Input size: {model_info['input_size']}")
    print(f"   - Output classes: {model_info['output_size']}")
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    dummy_input = torch.randn(2, 3, 70, 70)  # Batch of 2 images
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Forward pass successful!")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print(f"\n🎯 Model ready for training!")