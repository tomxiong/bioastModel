#!/usr/bin/env python3
"""
Training script for MIC MobileNetV3 model
Priority: 3
Description: Mobile-optimized CNN with medical features
Parameters: ~2.5M
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
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.real_data_loader import create_real_data_loaders
from models.mic_mobilenetv3 import MICMobileNetV3
from core.training_utils import EarlyStopping, ModelCheckpoint, calculate_metrics

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
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    # Generate classification report
    report = classification_report(all_targets, all_preds, output_dict=True)
    
    return test_acc, test_loss, report

def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'accuracy': accuracy,
    }, path)

def create_model():
    """Create MIC MobileNetV3 model"""
    model = MICMobileNetV3(num_classes=2)
    
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

def main():
    print("=" * 60)
    print("Training MIC MobileNetV3 Model")
    print("Priority: 3")
    print("Description: Mobile-optimized CNN with medical features")
    print("Parameters: ~2.5M")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "mic_mobilenetv3"
    
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
    patience = 10
    
    print(f"\nTraining configuration:")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: 32")
    print(f"Learning rate: 0.001")
    print(f"Weight decay: 0.01")
    print(f"Patience: {patience}")
    print(f"Scheduler: CosineAnnealingLR")
    
    # Initialize tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=patience)
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = f"checkpoints/{model_name}_{timestamp}_best.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, checkpoint_path)
            print(f"✓ New best model saved: {val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping.update(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Load best model for evaluation
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Best model loaded for evaluation")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_accuracy, test_loss, test_report = evaluate_model(model, test_loader, criterion, device)
    
    print(f"Test Accuracy: {test_accuracy:.4f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Prepare training results
    training_results = {
        'model_name': model_name,
        'priority': 3,
        'description': 'Mobile-optimized CNN with medical features',
        'timestamp': timestamp,
        'training_time_seconds': float(training_time),
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'training_config': {
            'epochs': num_epochs,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'criterion': 'CrossEntropyLoss',
            'patience': patience
        },
        'dataset_info': {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'input_size': '70x70',
            'num_classes': 2
        },
        'training_history': {
            'train_loss': [float(x) for x in train_losses],
            'train_accuracy': [float(x) for x in train_accuracies],
            'val_loss': [float(x) for x in val_losses],
            'val_accuracy': [float(x) for x in val_accuracies],
            'learning_rates': [float(x) for x in learning_rates]
        },
        'final_results': {
            'best_val_accuracy': float(best_val_acc),
            'best_val_loss': float(min(val_losses)) if val_losses else 0.0,
            'final_test_accuracy': float(test_accuracy),
            'final_test_loss': float(test_loss),
            'epochs_trained': len(train_losses)
        },
        'model_files': {
            'checkpoint': checkpoint_path,
            'training_report': f"reports/{model_name}_{timestamp}_training.json",
            'performance_report': f"reports/{model_name}_{timestamp}_performance.html"
        },
        'classification_report': test_report,
        'status': 'completed'
    }
    
    # Save training results
    os.makedirs('reports', exist_ok=True)
    results_path = f"reports/{model_name}_{timestamp}_training.json"
    
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\nTraining results saved to: {results_path}")
    
    # Generate HTML performance report (simplified)
    html_path = f"reports/{model_name}_{timestamp}_performance.html"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MIC MobileNetV3 Training Report</title>
    </head>
    <body>
        <h1>MIC MobileNetV3 Training Report</h1>
        <h2>Model Information</h2>
        <p>Model: {model_name}</p>
        <p>Priority: 3</p>
        <p>Parameters: {total_params:,}</p>
        <p>Training Time: {training_time:.2f}s</p>
        
        <h2>Results</h2>
        <p>Best Validation Accuracy: {best_val_acc:.2f}%</p>
        <p>Test Accuracy: {test_accuracy:.2f}%</p>
        <p>Test Loss: {test_loss:.4f}</p>
        
        <h2>Training Configuration</h2>
        <p>Epochs: {num_epochs}</p>
        <p>Batch Size: 32</p>
        <p>Learning Rate: 0.001</p>
        <p>Optimizer: AdamW</p>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Performance report saved to: {html_path}")
    
    # Update model registry
    print("\nUpdating model registry...")
    try:
        with open('model_registry.json', 'r') as f:
            registry = json.load(f)
        
        # Update the specific model entry
        if 'mic_mobilenetv3' in registry['models']:
            model_entry = registry['models']['mic_mobilenetv3']
            model_entry['training_history'].append({
                'timestamp': timestamp,
                'status': 'completed',
                'best_val_accuracy': float(best_val_acc),
                'test_accuracy': float(test_accuracy),
                'training_time_seconds': float(training_time),
                'checkpoint_path': checkpoint_path,
                'training_report': results_path,
                'performance_report': html_path
            })
            model_entry['latest_training'] = model_entry['training_history'][-1]
            model_entry['onnx_status'] = 'not_converted'
        
        # Update registry metadata
        registry['last_updated'] = datetime.now().isoformat()
        
        with open('model_registry.json', 'w') as f:
            json.dump(registry, f, indent=2)
        
        print("✓ Model registry updated successfully")
        
    except Exception as e:
        print(f"✗ Failed to update model registry: {e}")
    
    print("\n" + "=" * 60)
    print("MIC MobileNetV3 Training Summary")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Priority: 3")
    print(f"Parameters: {total_params:,}")
    print(f"Training time: {training_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training report: {results_path}")
    print(f"Performance report: {html_path}")
    print("=" * 60)
    
    return training_results

if __name__ == "__main__":
    main()