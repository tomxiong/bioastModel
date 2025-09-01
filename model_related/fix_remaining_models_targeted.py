#!/usr/bin/env python3
"""
Targeted Fix for Remaining Models
Fix the remaining failed models with correct class names and improved training
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def get_correct_class_names():
    """Manual mapping of correct class names for problematic models"""
    return {
        'densenet': 'DenseNet',
        'efficient_cnn': 'EfficientCNN', 
        'efficientnet': 'EfficientNet',
        'efficientnet_v2': 'EfficientNetV2',
        'enhanced_airbubble_detector': 'EnhancedAirBubbleDetector',
        'ghostnet': 'GhostNet',
        'mic_mobilenetv3': 'MIC_MobileNetV3',
        'micro_vit': 'MicroViT',
        'mnasnet': 'MNASNet',
        'mobilenet_v3': 'MobileNetV3',
        'regnet': 'RegNet',
        'regnet_wrapper': 'RegNetWrapper',
        'resnet_improved': 'ResNetImproved',
        'shufflenet_v2': 'ShuffleNetV2',
        'simplified_airbubble_detector': 'SimplifiedAirBubbleDetector',
        'vit_tiny': 'ViTTiny'
    }

def create_targeted_trainer(model_name, class_name):
    """Create a targeted training script for the specific model"""
    trainer_content = f'''#!/usr/bin/env python3
"""
Targeted Training Script for {model_name}
Generated automatically with correct class name: {class_name}
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.{model_name} import {class_name}
from core.real_data_loader import BiomedicalDataset

def train_model():
    """Train the {model_name} model"""
    print(f"🚀 Starting training for {{model_name}} ({{class_name}})")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {{device}}")
    
    # Data loading
    try:
        train_dataset = BiomedicalDataset('bioast_dataset', split='train')
        val_dataset = BiomedicalDataset('bioast_dataset', split='val')
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        print(f"✅ Data loaded - Train: {{len(train_dataset)}}, Val: {{len(val_dataset)}}")
    except Exception as e:
        print(f"❌ Data loading failed: {{e}}")
        return False
    
    # Model initialization
    try:
        model = {class_name}(num_classes=2)
        model = model.to(device)
        print(f"✅ Model initialized: {{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    except Exception as e:
        print(f"❌ Model initialization failed: {{e}}")
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
                print(f'Epoch {{epoch+1}}/30, Batch {{batch_idx}}/{{len(train_loader)}}, '
                      f'Loss: {{loss.item():.4f}}, Acc: {{100.*train_correct/train_total:.2f}}%')
        
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
        
        print(f'Epoch {{epoch+1}}/30: Train Acc: {{train_acc:.2f}}%, Val Acc: {{val_acc:.2f}}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f'checkpoints/{model_name}_{{timestamp}}_best.pth'
            os.makedirs('checkpoints', exist_ok=True)
            
            torch.save({{
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_name': '{model_name}',
                'class_name': '{class_name}'
            }}, checkpoint_path)
            
            print(f'✅ New best model saved: {{checkpoint_path}} ({{best_val_acc:.2f}}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {{epoch+1}} epochs')
                break
        
        scheduler.step()
    
    print(f"🎉 Training completed! Best validation accuracy: {{best_val_acc:.2f}}%")
    return True

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
'''
    
    trainer_path = f"trainers/train_{model_name}_targeted.py"
    os.makedirs("trainers", exist_ok=True)
    
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(trainer_content)
    
    return trainer_path

def check_training_status():
    """Check which models still need training"""
    failed_models = [
        'densenet', 'efficient_cnn', 'efficientnet', 'efficientnet_v2',
        'enhanced_airbubble_detector', 'ghostnet', 'mic_mobilenetv3',
        'micro_vit', 'mnasnet', 'mobilenet_v3', 'regnet', 'regnet_wrapper',
        'resnet_improved', 'shufflenet_v2', 'simplified_airbubble_detector',
        'vit_tiny'
    ]
    
    still_needed = []
    for model_name in failed_models:
        # Check for recent checkpoints
        checkpoint_pattern = f"checkpoints/{model_name}_*_best.pth"
        import glob
        recent_checkpoints = glob.glob(checkpoint_pattern)
        
        if not recent_checkpoints:
            still_needed.append(model_name)
        else:
            # Check if checkpoint is recent (within last hour)
            latest_checkpoint = max(recent_checkpoints, key=os.path.getctime)
            checkpoint_time = os.path.getctime(latest_checkpoint)
            current_time = time.time()
            
            if current_time - checkpoint_time > 3600:  # 1 hour
                still_needed.append(model_name)
    
    return still_needed

def main():
    """Main execution function"""
    print("🎯 Targeted Model Training Fix")
    print("=" * 50)
    
    # Get correct class names
    class_names = get_correct_class_names()
    
    # Check which models still need training
    models_to_train = check_training_status()
    
    print(f"📋 Models requiring training: {len(models_to_train)}")
    for model in models_to_train:
        print(f"  - {model} -> {class_names.get(model, 'UNKNOWN')}")
    
    if not models_to_train:
        print("✅ All models appear to be trained!")
        return
    
    # Train each model
    python_cmd = sys.executable
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n🚀 Training Model {i}/{len(models_to_train)}: {model_name}")
        print("-" * 40)
        
        class_name = class_names.get(model_name)
        if not class_name:
            print(f"❌ No class name mapping for {model_name}")
            continue
        
        # Create targeted trainer
        try:
            trainer_path = create_targeted_trainer(model_name, class_name)
            print(f"✅ Created trainer: {trainer_path}")
        except Exception as e:
            print(f"❌ Failed to create trainer for {model_name}: {e}")
            continue
        
        # Execute training
        try:
            print(f"🏃 Starting training for {model_name}...")
            result = subprocess.run(
                [python_cmd, trainer_path],
                timeout=1800,  # 30 minutes timeout
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {model_name} training completed successfully!")
            else:
                print(f"❌ {model_name} training failed:")
                print(f"STDOUT: {result.stdout[-500:]}")  # Last 500 chars
                print(f"STDERR: {result.stderr[-500:]}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_name} training timed out (30 minutes)")
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
        
        # Small delay between models
        time.sleep(2)
    
    print("\n🎉 Targeted training process completed!")
    
    # Final status check
    remaining = check_training_status()
    if remaining:
        print(f"⚠️  Still need training: {remaining}")
    else:
        print("✅ All models successfully trained!")

if __name__ == "__main__":
    main()