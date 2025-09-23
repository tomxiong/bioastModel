#!/usr/bin/env python3
"""
EfficientNetV2 models for multitask learning
Extended versions with proper factory functions for the multitask system
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from models.efficientnet_v2 import EfficientNetV2, create_efficientnetv2_s, create_efficientnetv2_m


class MultitaskEfficientNetV2(EfficientNetV2):
    """EfficientNetV2 adapted for multitask learning"""
    
    def __init__(self, 
                 num_classes: int = 2,
                 model_name: str = 'efficientnet_v2_s',
                 **kwargs):
        # Initialize with binary classification
        super().__init__(num_classes=num_classes, model_name=model_name, **kwargs)
        
        # Store model info for multitask system
        self.model_type = 'efficientnet_v2'
        self.feature_dim = self._get_feature_dim()
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension before classification head"""
        # This would typically be the output dimension of the last convolutional layer
        # For EfficientNetV2-S, it's 1280
        if 's' in self.model_name.lower():
            return 1280
        elif 'm' in self.model_name.lower():
            return 1280
        elif 'b0' in self.model_name.lower():
            return 1280
        else:
            return 1280  # Default
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head"""
        # This is a simplified version - in practice, you'd need to modify
        # the forward method to stop before the classification head
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for multitask system"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': self.model_name,
            'architecture': 'efficientnet_v2',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': self.feature_dim,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes
        }


def create_efficientnet_v2_s(num_classes: int = 2, **kwargs) -> MultitaskEfficientNetV2:
    """Create EfficientNetV2-S model for multitask learning"""
    model = MultitaskEfficientNetV2(
        num_classes=num_classes,
        model_name='efficientnet_v2_s',
        **kwargs
    )
    return model


def create_efficientnet_v2_b0(num_classes: int = 2, **kwargs) -> MultitaskEfficientNetV2:
    """Create EfficientNetV2-B0 model for multitask learning"""
    # Note: EfficientNetV2-B0 is not in the original paper, but we'll create a variant
    model = MultitaskEfficientNetV2(
        num_classes=num_classes,
        model_name='efficientnet_v2_b0',
        # Use S configuration but with B0-like scaling
        width_mult=0.8,  # Scale down from S
        **kwargs
    )
    return model


def create_efficientnet_v2_m(num_classes: int = 2, **kwargs) -> MultitaskEfficientNetV2:
    """Create EfficientNetV2-M model for multitask learning"""
    model = MultitaskEfficientNetV2(
        num_classes=num_classes,
        model_name='efficientnet_v2_m',
        **kwargs
    )
    return model


# Test function
def test_multitask_efficientnet_v2():
    """Test the multitask EfficientNetV2 models"""
    print("=== Testing Multitask EfficientNetV2 Models ===")
    
    models_to_test = [
        ('EfficientNetV2-S', create_efficientnet_v2_s),
        ('EfficientNetV2-B0', create_efficientnet_v2_b0),
        ('EfficientNetV2-M', create_efficientnet_v2_m),
    ]
    
    for model_name, create_func in models_to_test:
        print(f"\nTesting {model_name}:")
        
        # Create model
        model = create_func(num_classes=2)
        model_info = model.get_model_info()
        
        print(f"  Parameters: {model_info['total_parameters']:,}")
        print(f"  Feature dim: {model_info['feature_dim']}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 70, 70)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ {model_name} works correctly")


if __name__ == "__main__":
    test_multitask_efficientnet_v2()