#!/usr/bin/env python3
"""
Training script for EfficientNet-V2-S model using bioast_dataset
Based on the successful Inception_Micro training pattern
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model():
    """Train the EfficientNet-V2-S model using bioast_dataset"""
    print("🚀 Starting EfficientNet-V2-S training with bioast_dataset")
    
    try:
        # Import model and data loader
        from models.efficientnet_v2 import create_efficientnetv2_s
        from core.real_data_loader import create_real_data_loaders
        
        # Create model
        print("📦 Creating EfficientNet-V2-S model...")
        model = create_efficientnetv2_s(num_classes=2)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"🔧 Using device: {device}")
        
        # Create data loaders
        print("📊 Loading bioast_dataset...")
        train_loader, val_loader, test_loader = create_real_data_loaders(
            batch_size=32,
            num_workers=4
        )
        
        print(f"   - Training samples: {len(train_loader.dataset)}")
        print(f"   - Validation samples: {len(val_loader.dataset)}")
        print(f"   - Test samples: {len(test_loader.dataset)}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        # Training parameters
        num_epochs = 50
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"checkpoints/efficientnet_v2_s_{timestamp}_best.pth"
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        print(f"🎯 Training for {num_epochs} epochs with early stopping (patience={patience})")
        
        # Training loop
        start_time = time.time()
        training_history = []
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_acc = 100.0 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            
            # Record training history
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'lr': current_lr,
                'epoch_time': epoch_time
            }
            training_history.append(epoch_info)
            
            print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
                  f"Train: {train_acc:6.2f}% ({avg_train_loss:.4f}) | "
                  f"Val: {val_acc:6.2f}% ({avg_val_loss:.4f}) | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'model_name': 'efficientnet_v2_s',
                    'timestamp': timestamp
                }, checkpoint_path)
                print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                    break
        
        total_training_time = time.time() - start_time
        
        # Test evaluation
        print("\n🧪 Evaluating on test set...")
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        # Create training report
        report = {
            'model_name': 'efficientnet_v2_s',
            'timestamp': timestamp,
            'dataset': 'bioast_dataset',
            'total_epochs': len(training_history),
            'best_val_acc': best_val_acc,
            'final_test_acc': test_acc,
            'total_training_time': total_training_time,
            'training_history': training_history,
            'model_checkpoint': checkpoint_path,
            'hyperparameters': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'optimizer': 'Adam',
                'scheduler': 'ReduceLROnPlateau',
                'patience': patience
            }
        }
        
        # Save training report
        report_path = f"reports/efficientnet_v2_s_{timestamp}_training.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print final results
        print(f"\n🎉 Training completed!")
        print(f"   - Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   - Final test accuracy: {test_acc:.2f}%")
        print(f"   - Total training time: {total_training_time:.2f} seconds")
        print(f"   - Model saved: {checkpoint_path}")
        print(f"   - Report saved: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model()
    if success:
        print("✅ EfficientNet-V2-S training completed successfully!")
    else:
        print("❌ EfficientNet-V2-S training failed!")