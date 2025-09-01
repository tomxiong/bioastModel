#!/usr/bin/env python3
"""
Simple Training Script for regnet_wrapper
Direct approach with manual configuration
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.regnet_wrapper import RegNetWrapper
from core.real_data_loader import BiomedicalDataset

def train_model():
    """Train the regnet_wrapper model"""
    print(f"🚀 Starting training for regnet_wrapper")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: " + str(device))
    
    # Data loading
    try:
        train_dataset = BiomedicalDataset('bioast_dataset', split='train')
        val_dataset = BiomedicalDataset('bioast_dataset', split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        print(f"✅ Data loaded - Train: " + str(len(train_dataset)) + ", Val: " + str(len(val_dataset)))
    except Exception as e:
        print(f"❌ Data loading failed: " + str(e))
        return False
    
    # Model initialization
    try:
        model = RegNetWrapper(num_classes=2)
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model initialized: " + f"{param_count:.1f}M parameters")
    except Exception as e:
        print(f"❌ Model initialization failed: " + str(e))
        return False
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    
    # Training loop
    for epoch in range(30):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Handle multi-output models
            if isinstance(output, dict):
                output = output.get('classification', output.get('logits', list(output.values())[0]))
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                acc = 100.*train_correct/train_total
                print(f'Epoch {epoch+1}/30, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Handle multi-output models
                if isinstance(output, dict):
                    output = output.get('classification', output.get('logits', list(output.values())[0]))
                
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/30: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f'checkpoints/regnet_wrapper_' + timestamp + '_best.pth'
            os.makedirs('checkpoints', exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_name': 'regnet_wrapper',
                'class_name': 'RegNetWrapper'
            }, checkpoint_path)
            
            print(f'✅ New best model saved: ' + checkpoint_path + f' ({best_val_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        scheduler.step()
    
    print(f"🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return True

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
