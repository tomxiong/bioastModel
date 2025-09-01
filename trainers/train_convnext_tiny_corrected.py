#!/usr/bin/env python3
"""
Fixed Training Script for convnext_tiny
Auto-generated with correct class name: ConvNextTiny
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime

# Import the model
try:
    from models.convnext_tiny import ConvNextTiny
    print(f"Successfully imported ConvNextTiny from models.convnext_tiny")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Trying alternative import methods...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("convnext_tiny", f"models/convnext_tiny.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ConvNextTiny = getattr(module, "ConvNextTiny")
        print(f"Successfully imported ConvNextTiny using importlib")
    except Exception as e2:
        print(f"Alternative import failed: {e2}")
        sys.exit(1)

# Import data loader
from core.real_data_loader import create_real_data_loaders

def train_model():
    """Train the convnext_tiny model"""
    print(f"Starting training for convnext_tiny (ConvNextTiny)")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_real_data_loaders(
            batch_size=32,
            num_workers=4
        )
        print(f"Data loaders created successfully")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Create model
        print(f"Creating model: ConvNextTiny")
        model = ConvNextTiny()
        
        # Test with sample input to determine input size
        sample_input = torch.randn(1, 3, 70, 70)
        try:
            with torch.no_grad():
                output = model(sample_input)
            print(f"Model accepts 70x70 input, output shape: {output.shape}")
            input_size = 70
        except Exception as e:
            print(f"70x70 input failed: {e}")
            print("Trying 224x224 input...")
            try:
                sample_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    output = model(sample_input)
                print(f"Model accepts 224x224 input, output shape: {output.shape}")
                input_size = 224
                
                # Recreate data loaders with 224x224
                train_loader, val_loader, test_loader = create_real_data_loaders(
                    batch_size=32,
                    num_workers=4,
                    img_size=224
                )
                print("Recreated data loaders for 224x224 input")
            except Exception as e2:
                print(f"224x224 input also failed: {e2}")
                raise e2
        
        model = model.to(device)
        print(f"Model moved to {device}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        # Training parameters
        num_epochs = 30
        best_val_acc = 0.0
        patience = 8
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Handle multi-output models (like AirBubbleHybridNet)
                if isinstance(output, dict):
                    output = output['classification']
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
            
            train_acc = 100. * train_correct / train_total
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
                    
                    # Handle multi-output models
                    if isinstance(output, dict):
                        output = output['classification']
                    
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f'checkpoints/convnext_tiny_{timestamp}_best.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'model_class': 'ConvNextTiny',
                    'input_size': input_size
                }, checkpoint_path)
                print(f'  New best model saved: {checkpoint_path} (Val Acc: {val_acc:.2f}%)')
            else:
                patience_counter += 1
                print(f'  No improvement (patience: {patience_counter}/{patience})')
            
            scheduler.step()
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        print(f'Training completed for convnext_tiny')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        return True
        
    except Exception as e:
        print(f"Training failed for convnext_tiny: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model()
    if success:
        print(f"✅ convnext_tiny training completed successfully")
    else:
        print(f"❌ convnext_tiny training failed")
        sys.exit(1)
