#!/usr/bin/env python3
"""
Simple Direct Fix for Remaining Models
Direct approach with manual class mappings and proper initialization
"""

import os
import sys
import subprocess
import time

def get_model_configs():
    """Get correct model configurations"""
    return {
        'densenet': {
            'class_name': 'DenseNet',
            'import_line': 'from models.densenet import DenseNet',
            'init_code': 'model = DenseNet(num_classes=2)'
        },
        'mobilenet_v3': {
            'class_name': 'MobileNetV3',
            'import_line': 'from models.mobilenet_v3 import MobileNetV3',
            'init_code': 'model = MobileNetV3(num_classes=2)'
        },
        'regnet': {
            'class_name': 'RegNet',
            'import_line': 'from models.regnet import RegNet',
            'init_code': 'model = RegNet(num_classes=2)'
        },
        'regnet_wrapper': {
            'class_name': 'RegNetWrapper',
            'import_line': 'from models.regnet_wrapper import RegNetWrapper',
            'init_code': 'model = RegNetWrapper(num_classes=2)'
        },
        'resnet_improved': {
            'class_name': 'ResNetImproved',
            'import_line': 'from models.resnet_improved import ResNetImproved',
            'init_code': 'model = ResNetImproved(num_classes=2)'
        },
        'shufflenet_v2': {
            'class_name': 'ShuffleNetV2',
            'import_line': 'from models.shufflenet_v2 import ShuffleNetV2',
            'init_code': 'model = ShuffleNetV2(num_classes=2)'
        },
        'simplified_airbubble_detector': {
            'class_name': 'SimplifiedAirBubbleDetector',
            'import_line': 'from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector',
            'init_code': 'model = SimplifiedAirBubbleDetector(num_classes=2)'
        },
        'vit_tiny': {
            'class_name': 'ViTTiny',
            'import_line': 'from models.vit_tiny import ViTTiny',
            'init_code': 'model = ViTTiny(num_classes=2)'
        },
        'micro_vit': {
            'class_name': 'MicroViT',
            'import_line': 'from models.micro_vit import MicroViT',
            'init_code': 'model = MicroViT(num_classes=2)'
        },
        'enhanced_airbubble_detector': {
            'class_name': 'EnhancedAirBubbleDetector',
            'import_line': 'from models.enhanced_airbubble_detector import EnhancedAirBubbleDetector',
            'init_code': 'model = EnhancedAirBubbleDetector(num_classes=2)'
        }
    }

def create_simple_trainer(model_name, config):
    """Create a simple, direct training script"""
    trainer_content = f'''#!/usr/bin/env python3
"""
Simple Training Script for {model_name}
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

{config['import_line']}
from core.real_data_loader import BiomedicalDataset

def train_model():
    """Train the {model_name} model"""
    print(f"🚀 Starting training for {model_name}")
    
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
        {config['init_code']}
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
                'class_name': '{config["class_name"]}'
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
    
    trainer_path = f"trainers/train_{model_name}_simple.py"
    os.makedirs("trainers", exist_ok=True)
    
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(trainer_content)
    
    return trainer_path

def main():
    """Main execution function"""
    print("🎯 Simple Direct Model Training Fix")
    print("=" * 50)
    
    # Get model configurations
    model_configs = get_model_configs()
    
    # Models that need fixing based on the terminal output
    failed_models = [
        'mobilenet_v3', 'regnet', 'regnet_wrapper', 'resnet_improved',
        'shufflenet_v2', 'simplified_airbubble_detector', 'vit_tiny'
    ]
    
    print(f"📋 Models to fix: {len(failed_models)}")
    for model in failed_models:
        print(f"  - {model}")
    
    # Train each model
    python_cmd = sys.executable
    
    for i, model_name in enumerate(failed_models, 1):
        print(f"\n🚀 Training Model {i}/{len(failed_models)}: {model_name}")
        print("-" * 40)
        
        config = model_configs.get(model_name)
        if not config:
            print(f"❌ No configuration for {model_name}")
            continue
        
        # Create simple trainer
        try:
            trainer_path = create_simple_trainer(model_name, config)
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
    
    print("\n🎉 Simple training process completed!")

if __name__ == "__main__":
    main()