#!/usr/bin/env python3
"""
Biomedical Model Training Pipeline
Optimized training script for 70x70 biomedical images with positive/negative classification
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import json
import time
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import logging

# Import available models
from models.simplified_airbubble_detector import create_simplified_airbubble_detector
from models.micro_vit import create_micro_vit
from models.mic_mobilenetv3 import create_mic_mobilenetv3
from models.efficientnet import create_efficientnet_b0
from models.ghostnet import create_ghostnet

class BiomedicalDataset(Dataset):
    """Dataset class for biomedical positive/negative classification"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.samples = []
        self.class_to_idx = {'negative': 0, 'positive': 1}
        
        for class_name in ['negative', 'positive']:
            class_dir = self.data_dir / class_name / split
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Print class distribution
        labels = [sample[1] for sample in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique, counts):
            class_name = list(self.class_to_idx.keys())[cls]
            print(f"  {class_name}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (70, 70), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(split: str = 'train'):
    """Get data transforms for training/validation"""
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """Create model by name"""
    
    models = {
        'simplified_airbubble_detector': lambda: create_simplified_airbubble_detector(num_classes=num_classes),
        'micro_vit_tiny': lambda: create_micro_vit(num_classes=num_classes, model_size='tiny'),
        'micro_vit_small': lambda: create_micro_vit(num_classes=num_classes, model_size='small'),
        'mic_mobilenetv3': lambda: create_mic_mobilenetv3(num_classes=num_classes, model_size='small'),
        'efficientnet_b0': lambda: create_efficientnet_b0(num_classes=num_classes),
        'ghostnet': lambda: create_ghostnet(num_classes=num_classes, width=1.0),
        'ghostnet_0_5x': lambda: create_ghostnet(num_classes=num_classes, width=0.5),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if hasattr(model, 'forward') and 'micro_vit' in str(type(model)).lower():
            # Handle multi-task models
            outputs = model(data)
            if isinstance(outputs, dict):
                output = outputs['classification']
            else:
                output = outputs
        elif hasattr(model, 'forward') and 'mic_mobilenetv3' in str(type(model)).lower():
            # Handle MIC MobileNetV3 multi-task output
            outputs = model(data)
            if isinstance(outputs, dict):
                output = outputs['classification']
            else:
                output = outputs
        else:
            output = model(data)
        
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and 'micro_vit' in str(type(model)).lower():
                outputs = model(data)
                if isinstance(outputs, dict):
                    output = outputs['classification']
                else:
                    output = outputs
            elif hasattr(model, 'forward') and 'mic_mobilenetv3' in str(type(model)).lower():
                outputs = model(data)
                if isinstance(outputs, dict):
                    output = outputs['classification']
                else:
                    output = outputs
            else:
                output = model(data)
            
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description='Biomedical Model Training')
    parser.add_argument('--model', type=str, default='simplified_airbubble_detector',
                       help='Model to train')
    parser.add_argument('--data_dir', type=str, default='bioast_dataset',
                       help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='trained_models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(save_dir / f'{args.model}_training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Dataset directory {args.data_dir} not found!")
        print("Please upload your dataset first using the upload guide.")
        return
    
    # Create datasets
    print("📊 Loading datasets...")
    train_dataset = BiomedicalDataset(args.data_dir, 'train', get_transforms('train'))
    val_dataset = BiomedicalDataset(args.data_dir, 'val', get_transforms('val'))
    test_dataset = BiomedicalDataset(args.data_dir, 'test', get_transforms('test'))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    print(f"🏗️ Creating model: {args.model}")
    model = create_model(args.model, num_classes=2)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print(f"🚀 Starting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n📈 Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_name': args.model
            }, save_dir / f'{args.model}_best.pth')
            print(f"💾 Saved best model with validation accuracy: {val_acc:.2f}%")
    
    # Final test evaluation
    print("\n🧪 Final test evaluation...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Save final model and training history
    torch.save(model.state_dict(), save_dir / f'{args.model}_final.pth')
    
    with open(save_dir / f'{args.model}_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Training summary
    training_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"⏱️ Total training time: {training_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"🧪 Final test accuracy: {test_acc:.2f}%")
    print(f"💾 Models saved in: {save_dir}")

if __name__ == "__main__":
    main()