"""
Simplified Air Bubble Detector model definition.

This model is a lightweight CNN designed for efficient air bubble detection
with minimal parameters to prevent overfitting.

Created on: 2025-08-03 18:35:29
Updated: 2025-08-03 22:59:54 - Extracted from training script for ONNX conversion
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple

class SimplifiedAirBubbleDetector(nn.Module):
    """
    Simplified Air Bubble Detector - A lightweight CNN model designed to solve overfitting issues
    while maintaining high detection accuracy for air bubbles in medical images.
    """
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2, dropout_rate: float = 0.5):
        super().__init__()
        
        # Simplified feature extractor (target: <100k parameters)
        self.features = nn.Sequential(
            # Layer 1: Maintain resolution
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # Layer 2: Light downsampling
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 35x35
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # Layer 3: Feature extraction
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # Layer 4: Further downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 18x18
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Calculate parameter count
        self.param_count = sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 70, 70)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'simplified_airbubble_detector',
            'architecture': 'lightweight_cnn',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': 2
        }

def create_simplified_airbubble_detector(
    input_channels: int = 3,
    num_classes: int = 2,
    dropout_rate: float = 0.5,
    **kwargs
) -> SimplifiedAirBubbleDetector:
    """
    Create a Simplified Air Bubble Detector model.
    
    Args:
        input_channels (int): Number of input channels (default: 3)
        num_classes (int): Number of output classes (default: 2)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        SimplifiedAirBubbleDetector: Initialized model
    """
    model = SimplifiedAirBubbleDetector(
        input_channels=input_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    return model

# Model configuration for integration with training system
MODEL_CONFIG = {
    'name': 'simplified_airbubble_detector',
    'architecture': 'lightweight_cnn',
    'create_function': create_simplified_airbubble_detector,
    'default_params': {
        'input_channels': 3,
        'num_classes': 2,
        'dropout_rate': 0.5
    },
    'training_params': {
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 30,
        'optimizer': 'adam',
        'scheduler': 'step'
    },
    'estimated_parameters': 0.1,  # ~100k parameters
    'description': 'Lightweight CNN model designed for efficient air bubble detection with minimal parameters to prevent overfitting'
}

def generate_synthetic_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing the model.
    
    Args:
        num_samples (int): Number of samples to generate
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features) and y (labels)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize arrays
    X = np.zeros((num_samples, 3, 70, 70), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)
    
    # Generate balanced dataset
    for i in range(num_samples):
        # Determine class (0: no bubble, 1: bubble)
        label = i % 2
        y[i] = label
        
        # Generate base image (random noise)
        base_image = np.random.normal(0, 0.1, (3, 70, 70))
        
        if label == 1:  # Air bubble class
            # Add circular bubble pattern
            center_x = np.random.randint(20, 50)
            center_y = np.random.randint(20, 50)
            radius = np.random.randint(5, 15)
            
            for c in range(3):  # For each channel
                for x in range(70):
                    for y in range(70):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < radius:
                            # Bright center of bubble
                            base_image[c, x, y] += 0.5 * (1 - dist/radius)
        else:  # No bubble class
            # Add random texture
            texture = np.random.normal(0, 0.2, (3, 70, 70))
            base_image += texture * 0.3
        
        # Normalize to [0, 1] range
        base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min() + 1e-8)
        
        # Store in array
        X[i] = base_image
    
    return X, y

if __name__ == "__main__":
    # Test model creation
    print("🔍 Testing Simplified Air Bubble Detector model creation...")
    model = create_simplified_airbubble_detector()
    
    model_info = model.get_model_info()
    print(f"✅ Created {model_info['name']} with {model_info['total_parameters']:,} parameters")
    print(f"📊 Architecture details:")
    print(f"   - Architecture: {model_info['architecture']}")
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
    
    # Test synthetic data generation
    print("\n🧪 Testing synthetic data generation...")
    X, y = generate_synthetic_data(10)
    print(f"✅ Generated {len(X)} synthetic samples")
    print(f"   - X shape: {X.shape}")
    print(f"   - y shape: {y.shape}")
    print(f"   - Class distribution: {np.bincount(y)}")
    
    print(f"\n🎯 Model ready for training!")
