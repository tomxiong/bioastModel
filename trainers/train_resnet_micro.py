import sys
import os
sys.path.append('/home/aaa/ws/bioastModel')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime
from core.real_data_loader import create_real_data_loaders
from models.resnet_micro import ResnetMicro

def train_resnet_micro():
    print("Starting ResnetMicro training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_real_data_loaders(
        batch_size=32,
        num_workers=4
    )
    
    # Initialize model
    model = ResnetMicro().to(device)
    
    # Validate input size
    test_input = torch.randn(1, 3, 70, 70).to(device)
    try:
        test_output = model(test_input)
        print(f"✅ Model accepts 70x70 input, output shape: {test_output.shape}")
    except Exception as e:
        print(f"❌ Model failed 70x70 input validation: {e}")
        return
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    # Training variables
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    num_epochs = 50
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
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
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f'checkpoints/resnet_micro_{timestamp}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, checkpoint_path)
            print(f'  ✅ New best model saved: {checkpoint_path}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Test evaluation
    print("\nEvaluating on test set...")
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
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Save training report
    report = {
        'model_name': 'resnet_micro',
        'timestamp': datetime.now().isoformat(),
        'training_completed': True,
        'epochs_trained': epoch + 1,
        'best_val_accuracy': best_val_acc,
        'test_accuracy': test_acc,
        'final_checkpoint': checkpoint_path,
        'history': history,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'device_used': str(device)
    }
    
    report_path = f'reports/resnet_micro_training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    os.makedirs('reports', exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ ResnetMicro training completed!")
    print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📊 Test accuracy: {test_acc:.2f}%")
    print(f"💾 Model saved: {checkpoint_path}")
    print(f"📄 Report saved: {report_path}")
    
    return report

if __name__ == "__main__":
    train_resnet_micro()