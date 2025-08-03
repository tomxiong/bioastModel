"""
CoAtNet model definition for colony detection.

CoAtNet: Marrying Convolution and Attention for All Data Sizes
Paper: https://arxiv.org/abs/2106.04803

This model combines the efficiency of convolution with the expressiveness of attention,
designed for binary classification of 70x70 images (colony vs no-colony).

Created on: 2025-08-02 23:45:00
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import math

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False) if expand_ratio != 1 else nn.Identity()
        self.expand_bn = nn.BatchNorm2d(expanded_channels) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(expanded_channels, se_channels)
        
        # Output projection
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
        self.activation = nn.SiLU()
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if not isinstance(self.expand_conv, nn.Identity):
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.activation(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.activation(x)
        
        # Squeeze-and-Excitation
        x = self.se(x)
        
        # Output projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Residual connection with drop path
        if self.use_residual:
            x = identity + self.drop_path(x)
        
        return x

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation module."""
    
    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, se_channels, 1)
        self.conv2 = nn.Conv2d(se_channels, in_channels, 1)
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        scale = self.global_pool(x)
        scale = self.conv1(scale)
        scale = self.activation(scale)
        scale = self.conv2(scale)
        scale = self.sigmoid(scale)
        return x * scale

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention module."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class CoAtNet(nn.Module):
    """CoAtNet model for colony detection.
    
    Combines convolutional layers for local feature extraction with
    transformer layers for global context modeling.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        depths (list): Number of blocks at each stage
        dims (list): Feature dimensions at each stage
        num_heads (list): Number of attention heads for transformer stages
        drop_path_rate (float): Stochastic depth rate
        dropout_rate (float): Dropout rate for classifier
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        depths: List[int] = [2, 2, 3, 5, 2],  # 5 stages
        dims: List[int] = [64, 96, 192, 384, 768],
        num_heads: List[int] = [0, 0, 0, 12, 24],  # 0 means conv, >0 means transformer
        drop_path_rate: float = 0.1,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_stages = len(depths)
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(self.num_stages):
            # Downsample layer (except for first stage)
            if i > 0:
                downsample = nn.Sequential(
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2, bias=False),
                    nn.BatchNorm2d(dims[i])
                )
            else:
                downsample = nn.Identity()
            
            # Build blocks for this stage
            blocks = nn.ModuleList()
            for j in range(depths[i]):
                if num_heads[i] == 0:  # Convolutional stage
                    block = MBConvBlock(
                        in_channels=dims[i],
                        out_channels=dims[i],
                        stride=1,
                        drop_path_rate=dp_rates[cur + j]
                    )
                else:  # Transformer stage
                    block = TransformerBlock(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        drop_path=dp_rates[cur + j]
                    )
                blocks.append(block)
            
            stage = nn.ModuleDict({
                'downsample': downsample,
                'blocks': blocks
            })
            self.stages.append(stage)
            cur += depths[i]
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(dims[-1])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(dims[-1], num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        """Forward pass through feature extraction layers."""
        x = self.stem(x)
        
        for i, stage in enumerate(self.stages):
            # Apply downsampling
            x = stage['downsample'](x)
            
            # Apply blocks
            for block in stage['blocks']:
                if isinstance(block, TransformerBlock):
                    # Reshape for transformer: (B, C, H, W) -> (B, H*W, C)
                    B, C, H, W = x.shape
                    x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
                    x = block(x)
                    x = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
                else:
                    x = block(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 70, 70)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.forward_features(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.flatten(1)
        
        # Layer normalization and classification
        x = self.norm(x)
        x = self.classifier(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'coatnet',
            'architecture': 'coatnet',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes,
            'num_stages': self.num_stages
        }

def create_coatnet(
    num_classes: int = 2,
    model_size: str = 'small',
    **kwargs
) -> CoAtNet:
    """
    Create CoAtNet model with predefined configurations.
    
    Args:
        num_classes (int): Number of output classes
        model_size (str): Model size ('tiny', 'small', 'base')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        CoAtNet: Initialized model
    """
    
    configs = {
        'tiny': {
            'depths': [2, 2, 2, 2, 2],
            'dims': [48, 64, 128, 256, 512],
            'num_heads': [0, 0, 0, 8, 16],
            'drop_path_rate': 0.05
        },
        'small': {
            'depths': [2, 2, 3, 5, 2],
            'dims': [64, 96, 192, 384, 768],
            'num_heads': [0, 0, 0, 12, 24],
            'drop_path_rate': 0.1
        },
        'base': {
            'depths': [2, 2, 6, 14, 2],
            'dims': [96, 128, 256, 512, 1024],
            'num_heads': [0, 0, 0, 16, 32],
            'drop_path_rate': 0.2
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    model = CoAtNet(num_classes=num_classes, **config)
    return model

# Model configuration for integration with training system
MODEL_CONFIG = {
    'name': 'coatnet',
    'architecture': 'coatnet',
    'create_function': create_coatnet,
    'default_params': {
        'num_classes': 2,
        'model_size': 'small',
        'dropout_rate': 0.2,
        'drop_path_rate': 0.1
    },
    'training_params': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    },
    'estimated_parameters': 25.0,
    'description': 'CoAtNet model combining convolution and attention for efficient colony detection'
}

if __name__ == "__main__":
    # Test model creation
    print("🔍 Testing CoAtNet model creation...")
    
    for size in ['tiny', 'small']:
        print(f"\n📊 Testing {size} model...")
        model = create_coatnet(model_size=size)
        
        model_info = model.get_model_info()
        print(f"✅ Created {model_info['name']} with {model_info['total_parameters']:,} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 70, 70)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print(f"\n🎯 CoAtNet models ready for training!")