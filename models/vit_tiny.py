"""
Vision Transformer (ViT) Tiny implementation for colony detection.

A lightweight Vision Transformer designed for 70x70 images with efficient
patch embedding and reduced transformer layers for colony detection tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""
    
    def __init__(self, img_size: int = 70, patch_size: int = 7, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        x = self.projection(x)
        # Flatten patches: (B, embed_dim, num_patches_h, num_patches_w) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int = 192, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    def __init__(self, embed_dim: int = 192, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int = 192, num_heads: int = 6, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerTiny(nn.Module):
    """
    Tiny Vision Transformer for colony detection.
    
    Optimized for 70x70 images with reduced parameters while maintaining
    the core transformer architecture benefits.
    """
    
    def __init__(
        self,
        img_size: int = 70,
        patch_size: int = 7,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token for classification
        x = self.head(cls_token_final)
        
        return x
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Get attention maps for visualization."""
        B = x.shape[0]
        
        # Forward pass until specified layer
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply blocks until target layer
        target_layer = len(self.blocks) + layer_idx if layer_idx < 0 else layer_idx
        
        for i, block in enumerate(self.blocks):
            if i == target_layer:
                # Extract attention from this layer
                x_norm = block.norm1(x)
                B, N, C = x_norm.shape
                qkv = block.attn.qkv(x_norm).reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = attn.softmax(dim=-1)
                return attn
            x = block(x)
        
        return None


def create_vit_tiny(num_classes: int = 2, dropout_rate: float = 0.1) -> VisionTransformerTiny:
    """
    Create a tiny Vision Transformer model.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        VisionTransformerTiny model
    """
    model = VisionTransformerTiny(
        img_size=70,
        patch_size=7,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=dropout_rate
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("🔍 Testing Vision Transformer Tiny model creation...")
    
    # Test model creation
    model = create_vit_tiny(num_classes=2, dropout_rate=0.1)
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"✅ Created ViT-Tiny with {num_params:,} parameters")
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 70, 70)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test attention visualization
    with torch.no_grad():
        attn_maps = model.get_attention_maps(test_input, layer_idx=-1)
        if attn_maps is not None:
            print(f"   - Attention maps shape: {attn_maps.shape}")
    
    print("🎯 ViT-Tiny model ready for training!")