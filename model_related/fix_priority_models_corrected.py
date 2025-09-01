#!/usr/bin/env python3
"""
Corrected Fix Priority Models Script
Uses the correct main class names extracted from model files
"""

import os
import subprocess
import time
import json
from pathlib import Path

def create_fixed_training_script(model_name, class_name):
    """Create a fixed training script for a model with correct class name"""
    
    training_script_content = f'''#!/usr/bin/env python3
"""
Fixed Training Script for {model_name}
Auto-generated with correct class name: {class_name}
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
    from models.{model_name} import {class_name}
    print(f"Successfully imported {class_name} from models.{model_name}")
except ImportError as e:
    print(f"Import error: {{e}}")
    print(f"Trying alternative import methods...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("{model_name}", f"models/{model_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        {class_name} = getattr(module, "{class_name}")
        print(f"Successfully imported {class_name} using importlib")
    except Exception as e2:
        print(f"Alternative import failed: {{e2}}")
        sys.exit(1)

# Import data loader
from core.real_data_loader import create_real_data_loaders

def train_model():
    """Train the {model_name} model"""
    print(f"Starting training for {model_name} ({class_name})")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {{device}}")
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = create_real_data_loaders(
            batch_size=32,
            num_workers=4
        )
        print(f"Data loaders created successfully")
        print(f"Train samples: {{len(train_loader.dataset)}}")
        print(f"Val samples: {{len(val_loader.dataset)}}")
        print(f"Test samples: {{len(test_loader.dataset)}}")
        
        # Create model
        print(f"Creating model: {class_name}")
        model = {class_name}()
        
        # Test with sample input to determine input size
        sample_input = torch.randn(1, 3, 70, 70)
        try:
            with torch.no_grad():
                output = model(sample_input)
            print(f"Model accepts 70x70 input, output shape: {{output.shape}}")
            input_size = 70
        except Exception as e:
            print(f"70x70 input failed: {{e}}")
            print("Trying 224x224 input...")
            try:
                sample_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    output = model(sample_input)
                print(f"Model accepts 224x224 input, output shape: {{output.shape}}")
                input_size = 224
                
                # Recreate data loaders with 224x224
                train_loader, val_loader, test_loader = create_real_data_loaders(
                    batch_size=32,
                    num_workers=4,
                    img_size=224
                )
                print("Recreated data loaders for 224x224 input")
            except Exception as e2:
                print(f"224x224 input also failed: {{e2}}")
                raise e2
        
        model = model.to(device)
        print(f"Model moved to {{device}}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        # Training parameters
        num_epochs = 30
        best_val_acc = 0.0
        patience = 8
        patience_counter = 0
        
        print(f"Starting training for {{num_epochs}} epochs...")
        
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
                    print(f'Epoch {{epoch+1}}/{{num_epochs}}, Batch {{batch_idx}}/{{len(train_loader)}}, '
                          f'Loss: {{loss.item():.4f}}')
            
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
            
            print(f'Epoch {{epoch+1}}/{{num_epochs}}:')
            print(f'  Train Loss: {{avg_train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%')
            print(f'  Val Loss: {{avg_val_loss:.4f}}, Val Acc: {{val_acc:.2f}}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f'checkpoints/{model_name}_{{timestamp}}_best.pth'
                torch.save({{
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'model_class': '{class_name}',
                    'input_size': input_size
                }}, checkpoint_path)
                print(f'  New best model saved: {{checkpoint_path}} (Val Acc: {{val_acc:.2f}}%)')
            else:
                patience_counter += 1
                print(f'  No improvement (patience: {{patience_counter}}/{{patience}})')
            
            scheduler.step()
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {{epoch+1}} epochs')
                break
        
        print(f'Training completed for {model_name}')
        print(f'Best validation accuracy: {{best_val_acc:.2f}}%')
        
        return True
        
    except Exception as e:
        print(f"Training failed for {model_name}: {{e}}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model()
    if success:
        print(f"✅ {model_name} training completed successfully")
    else:
        print(f"❌ {model_name} training failed")
        sys.exit(1)
'''
    
    # Write the training script
    trainer_path = f"trainers/train_{model_name}_corrected.py"
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(training_script_content)
    
    # Make it executable
    os.chmod(trainer_path, 0o755)
    
    return trainer_path

def main():
    """Main function to fix priority models with correct class names"""
    print("🔧 Starting Corrected Priority Models Fix Script")
    print("=" * 50)
    
    # Priority models with correct class names
    priority_models = [
        ('airbubble_hybrid_net', 'AirBubbleHybridNet'),
        ('coatnet', 'CoAtNet'), 
        ('convnext_tiny', 'ConvNextTiny')
    ]
    
    results = {}
    
    for model_name, class_name in priority_models:
        print(f"\n📋 Processing {model_name} with class {class_name}...")
        
        # Check if model file exists
        model_file_path = f"models/{model_name}.py"
        if not os.path.exists(model_file_path):
            print(f"  ❌ Model file not found: {model_file_path}")
            results[model_name] = {'status': 'failed', 'reason': 'model_file_not_found'}
            continue
        
        # Create fixed training script
        try:
            trainer_path = create_fixed_training_script(model_name, class_name)
            print(f"  ✅ Created fixed training script: {trainer_path}")
            
            # Execute the training script
            print(f"  🚀 Starting training for {model_name}...")
            
            cmd = [
                "/home/aaa/ws/bioastModel/.venv/bin/python",
                trainer_path
            ]
            
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/home/aaa/ws/bioastModel"
            )
            
            # Monitor the process with timeout
            timeout = 1800  # 30 minutes
            try:
                stdout, _ = process.communicate(timeout=timeout)
                end_time = time.time()
                duration = end_time - start_time
                
                if process.returncode == 0:
                    print(f"  ✅ {model_name} training completed successfully in {duration:.1f}s")
                    results[model_name] = {
                        'status': 'success',
                        'class_name': class_name,
                        'trainer_path': trainer_path,
                        'duration': duration
                    }
                else:
                    print(f"  ❌ {model_name} training failed with return code {process.returncode}")
                    print(f"  Output: {stdout[-500:]}")  # Last 500 chars
                    results[model_name] = {
                        'status': 'failed',
                        'reason': 'training_failed',
                        'return_code': process.returncode,
                        'output': stdout[-500:]
                    }
                    
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"  ⏰ {model_name} training timed out after {timeout}s")
                results[model_name] = {
                    'status': 'failed',
                    'reason': 'timeout',
                    'timeout': timeout
                }
                
        except Exception as e:
            print(f"  ❌ Error creating/running training script: {e}")
            results[model_name] = {
                'status': 'failed',
                'reason': 'script_creation_failed',
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Corrected Priority Models Fix Summary")
    print("=" * 50)
    
    successful = [name for name, result in results.items() if result['status'] == 'success']
    failed = [name for name, result in results.items() if result['status'] == 'failed']
    
    print(f"✅ Successful: {len(successful)}/{len(priority_models)}")
    for name in successful:
        duration = results[name].get('duration', 0)
        print(f"  - {name} ({duration:.1f}s)")
    
    print(f"❌ Failed: {len(failed)}/{len(priority_models)}")
    for name in failed:
        reason = results[name].get('reason', 'unknown')
        print(f"  - {name} (reason: {reason})")
    
    # Save results
    with open('corrected_priority_models_fix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: corrected_priority_models_fix_results.json")
    
    if len(successful) == len(priority_models):
        print("🎉 All priority models fixed and trained successfully!")
        return True
    else:
        print(f"⚠️  {len(failed)} models still need attention")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)