"""
MIC MobileNetV3 - Mobile-optimized CNN with medical features
Optimized for 70x70 biomedical image analysis
Parameters: ~2.5M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels: int, reduction: int = 4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HardSwish(nn.Module):
    """Hard Swish activation function"""
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0

class MobileNetV3Block(nn.Module):
    """MobileNetV3 building block with medical imaging optimizations"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, use_se=False, use_hs=False):
        super(MobileNetV3Block, self).__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        hidden_dim = int(in_channels * expand_ratio)
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            HardSwish() if use_hs else nn.ReLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        if use_se:
            self.se = SEBlock(hidden_dim)
        else:
            self.se = nn.Identity()
        
        # Pointwise convolution
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # Expansion
        out = self.expand_conv(x)
        
        # Depthwise
        out = self.depthwise_conv(out)
        
        # SE block
        out = self.se(out)
        
        # Pointwise
        out = self.pointwise_conv(out)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
            
        return out

class MedicalFeatureExtractor(nn.Module):
    """Medical-specific feature extraction module"""
    def __init__(self, in_channels, out_channels):
        super(MedicalFeatureExtractor, self).__init__()
        
        # Multi-scale feature extraction for medical patterns
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1, padding=0),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 5, padding=2),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # Global context for medical image understanding
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//4, 1),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Get input size for upsampling
        _, _, h, w = x.size()
        
        # Multi-scale features
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        
        # Global context
        gc = self.global_context(x)
        gc = F.interpolate(gc, size=(h, w), mode='nearest')
        
        # Concatenate and fuse
        features = torch.cat([s1, s2, s3, gc], dim=1)
        out = self.fusion(features)
        
        return out

class MICMobileNetV3(nn.Module):
    """MIC MobileNetV3 for 70x70 biomedical image classification"""
    
    def __init__(self, num_classes=2, width_mult=1.0):
        super(MICMobileNetV3, self).__init__()
        
        # Initial convolution optimized for 70x70 input
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),  # 70x70 -> 35x35
            nn.BatchNorm2d(16),
            HardSwish()
        )
        
        # MobileNetV3 blocks configuration
        # [in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, use_hs]
        self.block_configs = [
            [16, 16, 3, 1, 1, False, False],    # 35x35
            [16, 24, 3, 2, 4, False, False],    # 35x35 -> 18x18
            [24, 24, 3, 1, 3, False, False],    # 18x18
            [24, 40, 5, 2, 3, True, False],     # 18x18 -> 9x9
            [40, 40, 5, 1, 3, True, False],     # 9x9
            [40, 40, 5, 1, 3, True, False],     # 9x9
            [40, 80, 3, 2, 6, False, True],     # 9x9 -> 5x5
            [80, 80, 3, 1, 2.5, False, True],   # 5x5
            [80, 80, 3, 1, 2.3, False, True],   # 5x5
            [80, 80, 3, 1, 2.3, False, True],   # 5x5
            [80, 112, 3, 1, 6, True, True],     # 5x5
            [112, 112, 3, 1, 6, True, True],    # 5x5
            [112, 160, 5, 2, 6, True, True],    # 5x5 -> 3x3
            [160, 160, 5, 1, 6, True, True],    # 3x3
            [160, 160, 5, 1, 6, True, True],    # 3x3
        ]
        
        # Build MobileNetV3 blocks
        self.blocks = nn.ModuleList()
        for i, config in enumerate(self.block_configs):
            in_ch, out_ch, k, s, e, se, hs = config
            # Apply width multiplier
            in_ch = int(in_ch * width_mult)
            out_ch = int(out_ch * width_mult)
            expand_ratio = e
            
            self.blocks.append(
                MobileNetV3Block(in_ch, out_ch, k, s, expand_ratio, se, hs)
            )
        
        # Medical feature extraction
        self.medical_features = MedicalFeatureExtractor(
            int(160 * width_mult), int(160 * width_mult)
        )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(int(160 * width_mult), int(960 * width_mult), 1, bias=False),
            nn.BatchNorm2d(int(960 * width_mult)),
            HardSwish()
        )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with dropout for medical imaging
        self.classifier = nn.Sequential(
            nn.Linear(int(960 * width_mult), int(1280 * width_mult)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(int(1280 * width_mult), num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Validate input size
        if x.shape[-2:] != (70, 70):
            raise ValueError(f"Expected input size (70, 70), got {x.shape[-2:]}")
        
        # Stem
        x = self.stem(x)  # 70x70 -> 35x35
        
        # MobileNetV3 blocks
        for block in self.blocks:
            x = block(x)
        
        # Medical feature extraction
        x = self.medical_features(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Extract feature maps for visualization"""
        features = []
        
        # Stem
        x = self.stem(x)
        features.append(('stem', x.clone()))
        
        # Blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 0:  # Save every 3rd block
                features.append((f'block_{i}', x.clone()))
        
        # Medical features
        x = self.medical_features(x)
        features.append(('medical_features', x.clone()))
        
        return features

def create_mic_mobilenetv3(num_classes=2, width_mult=1.0):
    """Create MIC MobileNetV3 model"""
    return MICMobileNetV3(num_classes=num_classes, width_mult=width_mult)

# Model variants
def mic_mobilenetv3_small(num_classes=2):
    """Small variant with reduced width"""
    return create_mic_mobilenetv3(num_classes=num_classes, width_mult=0.75)

def mic_mobilenetv3_large(num_classes=2):
    """Large variant with increased width"""
    return create_mic_mobilenetv3(num_classes=num_classes, width_mult=1.25)

if __name__ == "__main__":
    # Test the model
    model = MICMobileNetV3(num_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"MIC MobileNetV3 Model:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 70, 70)
    try:
        output = model(test_input)
        print(f"✓ Forward pass successful")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Test feature extraction
        features = model.get_feature_maps(test_input)
        print(f"✓ Feature extraction successful")
        print(f"Number of feature maps: {len(features)}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")