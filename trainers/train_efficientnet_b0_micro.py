#!/usr/bin/env python3
"""
Training script for EfficientnetB0Micro model
Priority: 8
Description: Micro EfficientNet-B0 variant (~1.9M params)
Parameters: ~1.9M
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.real_data_loader import create_real_data_loaders
from models.efficientnet_b0_micro import EfficientnetB0Micro

def create_model():
    """EfficientnetB0Micro model"""
    model = EfficientnetB0Micro(num_classes=2)
    
    # Validate input size
    test_input = torch.randn(1, 3, 70, 70)
    try:
        output = model(test_input)
        print(f"✓ Model accepts 70x70 input, output shape: {output.shape}")
        assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"
    except Exception as e:
        print(f"✗ Model input validation failed: {e}")
        raise
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def main():
    print("=" * 60)
    print("Training EfficientnetB0Micro Model")
    print("Priority: 8")
    print("Description: Micro EfficientNet-B0 variant (~1.9M params)")
    print("Parameters: ~1.9M")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "efficientnet_b0_micro"
    
    # Create model
    print("\nCreating model...")
    model = create_model()
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Get data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_real_data_loaders(
        batch_size=32,
        num_workers=4
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Training configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training parameters
    num_epochs = 50
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"\nTraining configuration:")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: 32")
    print(f"Learning rate: 0.001")
    print(f"Weight decay: 0.01")
    print(f"Patience: {patience}")
    print(f"Scheduler: CosineAnnealingLR")
    
    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f"checkpoints/{model_name}_{timestamp}_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc,
                'timestamp': timestamp
            }, checkpoint_path)
            
            print(f"✓ New best model saved: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    
    print(f"Test Accuracy: {test_acc:.4f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save training results
    training_results = {
        'model_name': model_name,
        'priority': 8,
        'description': 'Micro EfficientNet-B0 variant (~1.9M params)',
        'timestamp': timestamp,
        'training_time_seconds': float(training_time),
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'final_results': {
            'best_val_accuracy': float(best_val_acc),
            'final_test_accuracy': float(test_acc),
            'final_test_loss': float(test_loss),
            'epochs_trained': len(train_losses)
        },
        'training_history': {
            'train_loss': [float(x) for x in train_losses],
            'train_accuracy': [float(x) for x in train_accuracies],
            'val_loss': [float(x) for x in val_losses],
            'val_accuracy': [float(x) for x in val_accuracies],
            'learning_rates': [float(x) for x in learning_rates]
        },
        'model_files': {
            'checkpoint': checkpoint_path,
            'training_report': f"reports/{model_name}_{timestamp}_training.json"
        },
        'status': 'completed'
    }
    
    # Save training results
    os.makedirs('reports', exist_ok=True)
    results_path = f"reports/{model_name}_{timestamp}_training.json"
    
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\nTraining results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("EfficientnetB0Micro Training Summary")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Priority: 8")
    print(f"Parameters: {total_params:,}")
    print(f"Training time: {training_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training report: {results_path}")
    print("=" * 60)
    
    return training_results

if __name__ == "__main__":
    main()
