"""
ConvNext-Tiny model definition for colony detection.

This model is based on the ConvNext architecture and is designed for
binary classification of 70x70 images (colony vs no-colony).

ConvNext: A ConvNet for the 2020s
Paper: https://arxiv.org/abs/2201.03545

Created on: 2025-08-02 20:13:49
Updated: 2025-08-02 22:15:00 - Implemented full ConvNext architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    """ConvNext Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class ConvNextTiny(nn.Module):
    """ConvNext-Tiny model for colony detection.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        depths (list): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (list): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        pretrained (bool): Whether to use pretrained weights (default: True)
    """
    
    def __init__(
        self, 
        num_classes: int = 2,
        depths: list = [3, 3, 9, 3],
        dims: list = [96, 192, 384, 768],
        drop_path_rate: float = 0.,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.,
        dropout_rate: float = 0.2,
        pretrained: bool = True,
        **kwargs
    ):
        super(ConvNextTiny, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Stem layer
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # 3 intermediate downsampling conv layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final norm layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dims[-1], num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.head[-1].weight.data.mul_(head_init_scale)
        self.head[-1].bias.data.mul_(head_init_scale)
        
        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights()

    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_weights(self):
        """Load pretrained weights from torchvision if available."""
        try:
            import torchvision.models as models
            # Try to load ConvNext-Tiny pretrained weights
            # Note: This is a placeholder - actual implementation would need
            # to handle weight loading and adaptation for our specific input size
            print("⚠️  Pretrained weights loading not implemented yet - using random initialization")
        except ImportError:
            print("⚠️  torchvision not available - using random initialization")

    def forward_features(self, x):
        """Forward pass through feature extraction layers."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 70, 70)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'convnext_tiny',
            'architecture': 'convnext',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes,
            'depths': [3, 3, 9, 3],
            'dims': [96, 192, 384, 768]
        }

def create_convnext_tiny(
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> ConvNextTiny:
    """
    Create ConvNext-Tiny model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        **kwargs: Additional arguments for model initialization
        
    Returns:
        ConvNextTiny: Initialized model
    """
    model = ConvNextTiny(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
    
    return model

# Alias for backward compatibility
Convnexttiny = ConvNextTiny

# Model configuration for integration with training system
MODEL_CONFIG = {
    'name': 'convnext_tiny',
    'architecture': 'convnext',
    'create_function': create_convnext_tiny,
    'default_params': {
        'num_classes': 2,
        'pretrained': True,
        'dropout_rate': 0.2,
        'drop_path_rate': 0.1,
        'layer_scale_init_value': 1e-6
    },
    'training_params': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    },
    'estimated_parameters': 28.6,
    'description': 'ConvNext-Tiny model for efficient colony detection with modern convolution design'
}

if __name__ == "__main__":
    # Test model creation
    print("🔍 Testing ConvNext-Tiny model creation...")
    model = create_convnext_tiny()
    
    model_info = model.get_model_info()
    print(f"✅ Created {model_info['name']} with {model_info['total_parameters']:,} parameters")
    print(f"📊 Architecture details:")
    print(f"   - Depths: {model_info['depths']}")
    print(f"   - Dimensions: {model_info['dims']}")
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
    
    # Test with different input sizes to verify adaptability
    print("\n🔧 Testing different input sizes...")
    test_sizes = [(1, 3, 64, 64), (1, 3, 224, 224)]
    
    for size in test_sizes:
        try:
            test_input = torch.randn(*size)
            test_output = model(test_input)
            print(f"   ✅ {size} -> {test_output.shape}")
        except Exception as e:
            print(f"   ❌ {size} -> Error: {str(e)}")
    
    print(f"\n🎯 Model ready for training!")