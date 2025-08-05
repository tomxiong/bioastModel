"""
RegNet wrapper for 70x70 input images
Adapted for bioast colony detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation module"""
    
    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale

class Bottleneck(nn.Module):
    """RegNet Bottleneck Block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 group_width: int = 1, bottleneck_ratio: float = 1.0,
                 se_ratio: float = 0.0):
        super().__init__()
        
        # Calculate intermediate channels
        mid_channels = int(out_channels * bottleneck_ratio)
        groups = mid_channels // group_width
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1,
                              groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Squeeze and Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(mid_channels, se_channels)
        else:
            self.se = None
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        if self.se is not None:
            out = self.se(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class RegNetY400MF(nn.Module):
    """
    RegNet Y-400MF adapted for 70x70 input images
    
    Features:
    - Optimized for small input size (70x70)
    - Includes Squeeze-and-Excitation modules
    - Designed for binary classification (colony detection)
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Configuration for RegNet Y-400MF adapted for 70x70
        depths = [1, 3, 6, 6]
        widths = [48, 104, 208, 440]
        group_widths = [8, 8, 8, 8]
        bottleneck_ratio = 1.0
        se_ratio = 0.25  # RegNet Y uses SE
        stem_width = 32
        
        # Stem - adapted for 70x70 input
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, 1, 1, bias=False),  # stride=1 for small input
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True)
        )
        
        # Stages
        self.stages = nn.ModuleList()
        in_channels = stem_width
        
        for i, (depth, width, group_width) in enumerate(zip(depths, widths, group_widths)):
            stage = []
            
            for j in range(depth):
                # Adjust stride for small input size
                if i == 0:
                    stride = 1  # No downsampling in first stage
                elif i == 1 and j == 0:
                    stride = 2  # First downsampling
                elif i >= 2 and j == 0:
                    stride = 2  # Subsequent downsamplings
                else:
                    stride = 1
                
                stage.append(
                    Bottleneck(
                        in_channels, width, stride, group_width,
                        bottleneck_ratio, se_ratio
                    )
                )
                in_channels = width
            
            self.stages.append(nn.Sequential(*stage))
        
        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_channels, num_classes)
        
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
        # Input: (batch_size, 3, 70, 70)
        x = self.stem(x)  # (batch_size, 32, 70, 70)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # After stage 0: (batch_size, 48, 70, 70)
            # After stage 1: (batch_size, 104, 35, 35)
            # After stage 2: (batch_size, 208, 18, 18)
            # After stage 3: (batch_size, 440, 9, 9)
        
        x = self.avgpool(x)  # (batch_size, 440, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, 440)
        x = self.dropout(x)
        x = self.fc(x)  # (batch_size, num_classes)
        
        return x
    
    def get_feature_maps(self, x):
        """Extract feature maps from different stages"""
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features

# Alias for compatibility
RegNetY400MF_70x70 = RegNetY400MF

def create_regnet_y400mf_70x70(num_classes: int = 2, **kwargs) -> RegNetY400MF:
    """
    Create RegNet Y-400MF model adapted for 70x70 input
    
    Args:
        num_classes: Number of output classes (default: 2 for binary classification)
        **kwargs: Additional arguments (for compatibility)
    
    Returns:
        RegNetY400MF model instance
    """
    return RegNetY400MF(num_classes=num_classes)

if __name__ == "__main__":
    # Test the model
    model = create_regnet_y400mf_70x70(num_classes=2)
    
    # Test with 70x70 input
    x = torch.randn(1, 3, 70, 70)
    
    print("RegNet Y-400MF for 70x70 input:")
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Test feature extraction
    features = model.get_feature_maps(x)
    print("\nFeature map shapes:")
    for i, feat in enumerate(features):
        print(f"Stage {i}: {feat.shape}")