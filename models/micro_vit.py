"""
Micro Vision Transformer (Micro-ViT) for MIC testing.

This model is specifically designed for 70x70 small images with:
- Ultra-small patch size (5x5) for fine-grained analysis
- Lightweight transformer architecture
- MIC-specific positional encoding
- Air bubble awareness
- Turbidity-focused attention mechanisms

Based on ideas.md specifications for MIC testing scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math

class PatchEmbedding(nn.Module):
    """Patch embedding optimized for 70x70 images."""
    
    def __init__(
        self,
        img_size: int = 70,
        patch_size: int = 5,
        in_channels: int = 3,
        embed_dim: int = 192
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 14x14 = 196 patches
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Layer norm for patch embeddings
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"
        
        # Project to patches: (B, embed_dim, 14, 14)
        x = self.projection(x)
        
        # Flatten patches: (B, embed_dim, 196) -> (B, 196, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Apply layer norm
        x = self.norm(x)
        
        return x

class TurbidityPositionalEncoding(nn.Module):
    """Turbidity-aware positional encoding for MIC testing."""
    
    def __init__(
        self,
        embed_dim: int = 192,
        num_patches: int = 196,
        img_size: int = 70,
        patch_size: int = 5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.grid_size = img_size // patch_size  # 14
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Center-aware encoding (important for MIC testing)
        self.center_encoding = self._create_center_encoding()
        
        # Distance-based encoding for radial patterns
        self.distance_encoding = self._create_distance_encoding()
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def _create_center_encoding(self) -> torch.Tensor:
        """Create center-aware positional encoding."""
        center = self.grid_size // 2
        encoding = torch.zeros(self.num_patches, self.embed_dim)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch_idx = i * self.grid_size + j
                
                # Distance from center
                dist_from_center = math.sqrt((i - center)**2 + (j - center)**2)
                
                # Encode center proximity in first few dimensions
                for d in range(min(8, self.embed_dim)):
                    encoding[patch_idx, d] = math.sin(dist_from_center / (10000 ** (d / 8)))
        
        return nn.Parameter(encoding.unsqueeze(0), requires_grad=False)
    
    def _create_distance_encoding(self) -> torch.Tensor:
        """Create distance-based encoding for radial patterns."""
        encoding = torch.zeros(self.num_patches, self.embed_dim)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch_idx = i * self.grid_size + j
                
                # Radial distance encoding
                for d in range(8, min(16, self.embed_dim)):
                    freq = 2 ** ((d - 8) / 8)
                    encoding[patch_idx, d] = math.cos(freq * i) * math.sin(freq * j)
        
        return nn.Parameter(encoding.unsqueeze(0), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to patch embeddings."""
        B, N, D = x.shape
        
        # Combine all encodings
        pos_encoding = (self.pos_embed + 
                       self.center_encoding + 
                       self.distance_encoding)
        
        return x + pos_encoding

class MultiHeadAttention(nn.Module):
    """Multi-head attention with MIC-specific modifications."""
    
    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Turbidity-aware attention bias
        self.turbidity_bias = nn.Parameter(torch.zeros(num_heads, 196, 196))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add turbidity-aware bias
        attn = attn + self.turbidity_bias.unsqueeze(0)
        
        # Apply softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    """MLP block for transformer."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block optimized for MIC analysis."""
    
    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        # Layer normalization
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class BubbleAwareAttentionPool(nn.Module):
    """Bubble-aware attention pooling for final classification."""
    
    def __init__(self, embed_dim: int = 192):
        super().__init__()
        
        # Attention weights for pooling
        self.attention_weights = nn.Linear(embed_dim, 1)
        
        # Bubble suppression weights
        self.bubble_suppression = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, bubble_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings (B, N, D)
            bubble_mask: Optional bubble mask (B, N, 1)
        Returns:
            Pooled features (B, D)
        """
        B, N, D = x.shape
        
        # Compute attention weights
        attn_weights = self.attention_weights(x)  # (B, N, 1)
        
        # Apply bubble suppression if mask is provided
        if bubble_mask is not None:
            suppression = self.bubble_suppression(x)  # (B, N, 1)
            attn_weights = attn_weights * (1.0 - bubble_mask * suppression)
        
        # Apply softmax to attention weights
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum of patch embeddings
        pooled = (x * attn_weights).sum(dim=1)  # (B, D)
        
        return pooled

class MicroViT(nn.Module):
    """
    Micro Vision Transformer for MIC testing.
    
    Optimized for 70x70 images with ultra-small patches (5x5) for fine-grained analysis.
    Includes MIC-specific features like turbidity analysis and bubble detection.
    """
    
    def __init__(
        self,
        img_size: int = 70,
        patch_size: int = 5,
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        enable_bubble_detection: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.enable_bubble_detection = enable_bubble_detection
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Positional encoding
        self.pos_embed = TurbidityPositionalEncoding(
            embed_dim=embed_dim,
            num_patches=num_patches,
            img_size=img_size,
            patch_size=patch_size
        )
        
        # Dropout
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Bubble detection head (if enabled)
        if self.enable_bubble_detection:
            self.bubble_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Bubble-aware attention pooling
        self.attention_pool = BubbleAwareAttentionPool(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Turbidity regression head
        self.turbidity_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 4)  # A, B, C, D grades
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch features."""
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add positional encoding
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs.
        
        Args:
            x: Input tensor of shape (B, 3, 70, 70)
            
        Returns:
            Dictionary containing:
            - classification: Main classification logits
            - turbidity: Turbidity score
            - bubble_detection: Bubble detection scores (if enabled)
            - quality: Quality assessment scores
        """
        # Extract patch features
        patch_features = self.forward_features(x)  # (B, N, D)
        
        results = {}
        
        # Bubble detection (per patch)
        bubble_mask = None
        if self.enable_bubble_detection:
            bubble_scores = self.bubble_head(patch_features)  # (B, N, 1)
            results['bubble_detection'] = bubble_scores
            bubble_mask = bubble_scores
        
        # Bubble-aware attention pooling
        pooled_features = self.attention_pool(patch_features, bubble_mask)  # (B, D)
        
        # Main classification
        classification_logits = self.head(pooled_features)
        results['classification'] = classification_logits
        
        # Turbidity analysis
        turbidity_score = self.turbidity_head(pooled_features)
        results['turbidity'] = turbidity_score
        
        # Quality assessment
        quality_scores = self.quality_head(pooled_features)
        results['quality'] = quality_scores
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'micro_vit',
            'architecture': 'vision_transformer_micro',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes,
            'patch_size': self.patch_embed.patch_size,
            'num_patches': self.patch_embed.num_patches,
            'embed_dim': self.embed_dim,
            'features': {
                'bubble_detection': self.enable_bubble_detection,
                'turbidity_analysis': True,
                'quality_assessment': True,
                'multi_task': True
            }
        }

def create_micro_vit(
    num_classes: int = 2,
    model_size: str = 'tiny',
    **kwargs
) -> MicroViT:
    """
    Create Micro Vision Transformer model.
    
    Args:
        num_classes: Number of output classes
        model_size: Model size ('tiny', 'small', 'base')
        **kwargs: Additional arguments
        
    Returns:
        MicroViT: Initialized model
    """
    
    configs = {
        'tiny': {
            'embed_dim': 192,
            'depth': 6,
            'num_heads': 6,
            'mlp_ratio': 2.0,
            'drop_path_rate': 0.05
        },
        'small': {
            'embed_dim': 256,
            'depth': 8,
            'num_heads': 8,
            'mlp_ratio': 2.5,
            'drop_path_rate': 0.1
        },
        'base': {
            'embed_dim': 384,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 3.0,
            'drop_path_rate': 0.15
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    model = MicroViT(num_classes=num_classes, **config)
    return model

# Model configuration for integration
MODEL_CONFIG = {
    'name': 'micro_vit',
    'architecture': 'vision_transformer_micro',
    'create_function': create_micro_vit,
    'default_params': {
        'num_classes': 2,
        'model_size': 'tiny',
        'drop_rate': 0.1,
        'enable_bubble_detection': True
    },
    'training_params': {
        'batch_size': 32,
        'learning_rate': 0.0005,
        'weight_decay': 0.05,
        'epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    },
    'estimated_parameters': 1.8,
    'description': 'Micro Vision Transformer optimized for 70x70 MIC testing images'
}

if __name__ == "__main__":
    # Test model creation
    print("🔍 Testing Micro-ViT model creation...")
    
    for size in ['tiny', 'small']:
        print(f"\n📊 Testing {size} model...")
        model = create_micro_vit(model_size=size)
        
        model_info = model.get_model_info()
        print(f"✅ Created {model_info['name']} with {model_info['total_parameters']:,} parameters")
        print(f"   - Patch size: {model_info['patch_size']}x{model_info['patch_size']}")
        print(f"   - Num patches: {model_info['num_patches']}")
        print(f"   - Embed dim: {model_info['embed_dim']}")
        print(f"   - Features: {model_info['features']}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 70, 70)
        model.eval()
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Classification output: {outputs['classification'].shape}")
        print(f"   - Turbidity output: {outputs['turbidity'].shape}")
        
        if 'bubble_detection' in outputs:
            print(f"   - Bubble detection: {outputs['bubble_detection'].shape}")
    
    print(f"\n🎯 Micro-ViT models ready for training!")