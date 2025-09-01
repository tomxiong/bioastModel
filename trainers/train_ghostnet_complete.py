#!/usr/bin/env python3
"""
Complete Training Script for ghostnet
Proper initialization with all required parameters
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.ghostnet import GhostNet
    from core.real_data_loader import create_real_data_loaders
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def train_model():
    """Train the ghostnet model with complete configuration"""
    print(f"🚀 Starting complete training for ghostnet")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading using working pattern
    try:
        train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32, num_workers=4)
        print(f"✅ Data loaded successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        traceback.print_exc()
        return False
    
    # Model initialization with complete configuration
    model = None
    try:
        # Complete initialization with all parameters
        model = GhostNet(num_classes=2, width=1.0)
        print(f"✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        traceback.print_exc()
        return False
    
    if model is None:
        print(f"❌ Model is None after initialization")
        return False
    
    try:
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model moved to device: {param_count:.1f}M parameters")
    except Exception as e:
        print(f"❌ Failed to move model to device: {e}")
        return False
    
    # Training setup
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        print(f"✅ Training setup completed")
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        return False
    
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    
    # Training loop
    for epoch in range(30):
        try:
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    
                    # Handle multi-output models
                    if isinstance(output, dict):
                        output = output.get('classification', output.get('logits', list(output.values())[0]))
                    
                    # Handle dimension mismatches
                    if output.dim() > 2:
                        output = output.view(output.size(0), -1)
                    if output.size(1) != 2:
                        # Add a linear layer to map to correct output size
                        if not hasattr(model, 'final_classifier'):
                            model.final_classifier = nn.Linear(output.size(1), 2).to(device)
                        output = model.final_classifier(output)
                    
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
                        
                except Exception as e:
                    print(f"❌ Training batch {batch_idx} failed: {e}")
                    continue
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    try:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        
                        # Handle multi-output models
                        if isinstance(output, dict):
                            output = output.get('classification', output.get('logits', list(output.values())[0]))
                        
                        if output.dim() > 2:
                            output = output.view(output.size(0), -1)
                        if output.size(1) != 2 and hasattr(model, 'final_classifier'):
                            output = model.final_classifier(output)
                        
                        loss = criterion(output, target)
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                    except Exception as e:
                        print(f"❌ Validation batch failed: {e}")
                        continue
            
            train_acc = 100. * train_correct / train_total if train_total > 0 else 0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            
            print(f'Epoch {epoch+1}/30: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save checkpoint
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f'checkpoints/ghostnet_{timestamp}_best.pth'
                os.makedirs('checkpoints', exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'model_name': 'ghostnet',
                    'class_name': 'GhostNet'
                }, checkpoint_path)
                
                print(f'✅ New best model saved: {checkpoint_path} ({best_val_acc:.2f}%)')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            scheduler.step()
            
        except Exception as e:
            print(f"❌ Epoch {epoch+1} failed: {e}")
            traceback.print_exc()
            continue
    
    print(f"🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc > 0

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
