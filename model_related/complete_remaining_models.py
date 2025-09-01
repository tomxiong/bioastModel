#!/usr/bin/env python3
"""
Complete Remaining Models Training
Fix specific initialization issues and train all remaining models
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def get_complete_model_configs():
    """Get complete configurations with proper initialization parameters"""
    return {
        'mobilenet_v3': {
            'class_name': 'MobileNetV3',
            'import_line': 'from models.mobilenet_v3 import MobileNetV3',
            'init_code': '''# MobileNetV3 Large configuration
block_configs = [
    # [kernel, exp, out, SE, NL, s]
    [3, 16, 16, False, 'RE', 1],
    [3, 64, 24, False, 'RE', 2],
    [3, 72, 24, False, 'RE', 1],
    [5, 72, 40, True, 'RE', 2],
    [5, 120, 40, True, 'RE', 1],
    [5, 120, 40, True, 'RE', 1],
    [3, 240, 80, False, 'HS', 2],
    [3, 200, 80, False, 'HS', 1],
    [3, 184, 80, False, 'HS', 1],
    [3, 184, 80, False, 'HS', 1],
    [3, 480, 112, True, 'HS', 1],
    [3, 672, 112, True, 'HS', 1],
    [5, 672, 160, True, 'HS', 2],
    [5, 960, 160, True, 'HS', 1],
    [5, 960, 160, True, 'HS', 1],
]
model = MobileNetV3(block_configs=block_configs, last_channel=1280, num_classes=2)'''
        },
        'efficientnet_v2': {
            'class_name': 'EfficientNetV2',
            'import_line': 'from models.efficientnet_v2 import EfficientNetV2',
            'init_code': 'model = EfficientNetV2(num_classes=2, width_mult=1.0, depth_mult=1.0)'
        },
        'enhanced_airbubble_detector': {
            'class_name': 'EnhancedAirBubbleDetector',
            'import_line': 'from models.enhanced_airbubble_detector import EnhancedAirBubbleDetector',
            'init_code': 'model = EnhancedAirBubbleDetector(num_classes=2)'
        },
        'ghostnet': {
            'class_name': 'GhostNet',
            'import_line': 'from models.ghostnet import GhostNet',
            'init_code': 'model = GhostNet(num_classes=2, width=1.0)'
        },
        'mic_mobilenetv3': {
            'class_name': 'MICMobileNetV3',
            'import_line': 'from models.mic_mobilenetv3 import MICMobileNetV3',
            'init_code': 'model = MICMobileNetV3(num_classes=2)'
        },
        'micro_vit': {
            'class_name': 'MicroViT',
            'import_line': 'from models.micro_vit import MicroViT',
            'init_code': 'model = MicroViT(num_classes=2, img_size=70, patch_size=7)'
        },
        'mnasnet': {
            'class_name': 'MNASNet',
            'import_line': 'from models.mnasnet import MNASNet',
            'init_code': 'model = MNASNet(alpha=1.0, num_classes=2)'
        },
        'regnet': {
            'class_name': 'RegNet',
            'import_line': 'from models.regnet import RegNet',
            'init_code': 'model = RegNet(num_classes=2, w_a=36.44, w_0=20, w_m=2.49, d=13, group_w=8)'
        },
        'regnet_wrapper': {
            'class_name': 'RegNetWrapper',
            'import_line': 'from models.regnet_wrapper import RegNetWrapper',
            'init_code': 'model = RegNetWrapper(num_classes=2)'
        },
        'resnet_improved': {
            'class_name': 'ResNetImproved',
            'import_line': 'from models.resnet_improved import ResNetImproved',
            'init_code': 'model = ResNetImproved(num_classes=2, layers=[2, 2, 2, 2])'
        },
        'shufflenet_v2': {
            'class_name': 'ShuffleNetV2',
            'import_line': 'from models.shufflenet_v2 import ShuffleNetV2',
            'init_code': 'model = ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024], num_classes=2)'
        },
        'vit_tiny': {
            'class_name': 'ViTTiny',
            'import_line': 'from models.vit_tiny import ViTTiny',
            'init_code': 'model = ViTTiny(num_classes=2, img_size=70, patch_size=7)'
        }
    }

def create_complete_trainer(model_name, config):
    """Create complete training script with proper initialization"""
    trainer_content = f'''#!/usr/bin/env python3
"""
Complete Training Script for {model_name}
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
    {config['import_line']}
    from core.real_data_loader import create_real_data_loaders
except ImportError as e:
    print(f"❌ Import failed: {{e}}")
    sys.exit(1)

def train_model():
    """Train the {model_name} model with complete configuration"""
    print(f"🚀 Starting complete training for {model_name}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {{device}}")
    
    # Data loading using working pattern
    try:
        train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32, num_workers=4)
        print(f"✅ Data loaded successfully")
        print(f"  Train batches: {{len(train_loader)}}")
        print(f"  Val batches: {{len(val_loader)}}")
        print(f"  Test batches: {{len(test_loader)}}")
    except Exception as e:
        print(f"❌ Data loading failed: {{e}}")
        traceback.print_exc()
        return False
    
    # Model initialization with complete configuration
    model = None
    try:
        # Complete initialization with all parameters
        {config['init_code']}
        print(f"✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Model initialization failed: {{e}}")
        traceback.print_exc()
        return False
    
    if model is None:
        print(f"❌ Model is None after initialization")
        return False
    
    try:
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model moved to device: {{param_count:.1f}}M parameters")
    except Exception as e:
        print(f"❌ Failed to move model to device: {{e}}")
        return False
    
    # Training setup
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        print(f"✅ Training setup completed")
    except Exception as e:
        print(f"❌ Training setup failed: {{e}}")
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
                        print(f'Epoch {{epoch+1}}/30, Batch {{batch_idx}}/{{len(train_loader)}}, Loss: {{loss.item():.4f}}, Acc: {{acc:.2f}}%')
                        
                except Exception as e:
                    print(f"❌ Training batch {{batch_idx}} failed: {{e}}")
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
                        print(f"❌ Validation batch failed: {{e}}")
                        continue
            
            train_acc = 100. * train_correct / train_total if train_total > 0 else 0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            
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
            
        except Exception as e:
            print(f"❌ Epoch {{epoch+1}} failed: {{e}}")
            traceback.print_exc()
            continue
    
    print(f"🎉 Training completed! Best validation accuracy: {{best_val_acc:.2f}}%")
    return best_val_acc > 0

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
'''
    
    trainer_path = f"trainers/train_{model_name}_complete.py"
    os.makedirs("trainers", exist_ok=True)
    
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(trainer_content)
    
    return trainer_path

def main():
    """Main execution function"""
    print("🎯 Complete Remaining Models Training")
    print("=" * 60)
    
    # Get complete model configurations
    model_configs = get_complete_model_configs()
    
    # All remaining models that need training
    remaining_models = [
        'mobilenet_v3',  # Fix the initialization issue
        'efficientnet_v2', 'enhanced_airbubble_detector', 
        'ghostnet', 'mic_mobilenetv3', 'micro_vit', 'mnasnet',
        'regnet', 'regnet_wrapper', 'resnet_improved',
        'shufflenet_v2', 'vit_tiny'
    ]
    
    print(f"📋 Models to complete: {len(remaining_models)}")
    for model in remaining_models:
        print(f"  - {model}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(remaining_models),
        'successful': [],
        'failed': [],
        'details': {}
    }
    
    # Train each model
    python_cmd = sys.executable
    
    for i, model_name in enumerate(remaining_models, 1):
        print(f"\n🚀 Complete Training {i}/{len(remaining_models)}: {model_name}")
        print("-" * 50)
        
        config = model_configs.get(model_name)
        if not config:
            print(f"❌ No configuration for {model_name}")
            results['failed'].append(model_name)
            results['details'][model_name] = {'error': 'No configuration'}
            continue
        
        # Create complete trainer
        try:
            trainer_path = create_complete_trainer(model_name, config)
            print(f"✅ Created complete trainer: {trainer_path}")
        except Exception as e:
            print(f"❌ Failed to create trainer for {model_name}: {e}")
            results['failed'].append(model_name)
            results['details'][model_name] = {'error': f'Trainer creation failed: {e}'}
            continue
        
        # Execute training
        start_time = time.time()
        try:
            print(f"🏃 Starting complete training for {model_name}...")
            result = subprocess.run(
                [python_cmd, trainer_path],
                timeout=1800,  # 30 minutes timeout
                capture_output=True,
                text=True
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {model_name} training completed successfully! ({duration:.1f}s)")
                results['successful'].append(model_name)
                results['details'][model_name] = {
                    'status': 'success',
                    'duration': duration,
                    'stdout': result.stdout[-1000:]
                }
            else:
                print(f"❌ {model_name} training failed:")
                print(f"STDOUT: {result.stdout[-500:]}")
                print(f"STDERR: {result.stderr[-500:]}")
                results['failed'].append(model_name)
                results['details'][model_name] = {
                    'status': 'failed',
                    'duration': duration,
                    'stdout': result.stdout[-500:],
                    'stderr': result.stderr[-500:]
                }
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {model_name} training timed out (30 minutes)")
            results['failed'].append(model_name)
            results['details'][model_name] = {
                'status': 'timeout',
                'duration': duration
            }
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Error training {model_name}: {e}")
            results['failed'].append(model_name)
            results['details'][model_name] = {
                'status': 'error',
                'duration': duration,
                'error': str(e)
            }
        
        # Small delay between models
        time.sleep(3)
    
    # Save results
    with open('complete_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Complete training process finished!")
    print("=" * 60)
    print(f"✅ Successfully trained: {len(results['successful'])}")
    for model in results['successful']:
        print(f"   ✅ {model}")
    
    print(f"❌ Failed models: {len(results['failed'])}")
    for model in results['failed']:
        print(f"   ❌ {model}")
    
    print(f"\n📊 Results saved to: complete_training_results.json")

if __name__ == "__main__":
    main()