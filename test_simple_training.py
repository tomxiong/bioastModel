#!/usr/bin/env python3
"""
Simple training test script without OpenCV dependency.
This script creates dummy data to test the basic training functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.mic_mobilenetv3 import create_mic_mobilenetv3, MODEL_CONFIG as MIC_MOBILENET_CONFIG

class DummyDataset(Dataset):
    """Dummy dataset for testing"""
    
    def __init__(self, num_samples=100, image_size=70):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create random image data
        image = torch.randn(3, self.image_size, self.image_size)
        # Random binary label
        label = torch.randint(0, 2, (1,)).item()
        return image, label

def test_training():
    """Test basic training functionality"""
    print("🚀 Testing basic training functionality...")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create model
    print("🏗️ Creating MIC MobileNetV3 model...")
    model = create_mic_mobilenetv3(**MIC_MOBILENET_CONFIG['default_params'])
    model = model.to(device)
    
    model_info = model.get_model_info()
    print(f"📊 Model parameters: {model_info['total_parameters']}")
    
    # Create dummy data
    print("📂 Creating dummy dataset...")
    train_dataset = DummyDataset(num_samples=64)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Test one epoch
    print("🔄 Testing one training epoch...")
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs['classification']
        else:
            logits = outputs
            
        loss = criterion(logits, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    print(f"✅ Training test completed!")
    print(f"📊 Average Loss: {avg_loss:.4f}")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"🎯 Training functionality is working correctly!")

if __name__ == "__main__":
    test_training()