#!/usr/bin/env python3
"""
Fix Remaining Models Script
Systematically fix and train the 16 remaining models
"""

import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

# List of models that need fixing (from the analysis)
FAILED_MODELS = [
    'densenet',
    'efficient_cnn', 
    'efficientnet',
    'efficientnet_v2',
    'enhanced_airbubble_detector',
    'ghostnet',
    'mic_mobilenetv3',
    'micro_vit',
    'mnasnet',
    'mobilenet_v3',
    'regnet',
    'regnet_wrapper',
    'resnet_improved',
    'shufflenet_v2',
    'simplified_airbubble_detector',
    'vit_tiny'
]

def extract_main_class_name(model_file_path):
    """Extract the main model class name from Python file"""
    try:
        with open(model_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for class definitions that inherit from nn.Module
        import re
        class_pattern = r'class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\):'
        matches = re.findall(class_pattern, content)
        
        if matches:
            # Filter out utility classes
            utility_classes = {
                'InvertedResidual', 'MBConvBlock', 'LayerNorm', 'Block', 
                'Attention', 'MLP', 'SqueezeExcitation', 'SEBlock',
                'DepthwiseSeparableConv', 'ConvBNReLU', 'BasicBlock'
            }
            main_classes = [cls for cls in matches if cls not in utility_classes]
            
            if main_classes:
                return main_classes[0]
            
            # If only utility classes, try to infer from filename
            filename = os.path.basename(model_file_path).replace('.py', '')
            parts = filename.split('_')
            camel_case = ''.join(word.capitalize() for word in parts)
            
            # Check if this exists in content
            if camel_case in content:
                return camel_case
            
            # Return first match as fallback
            return matches[0]
        
        return None
        
    except Exception as e:
        print(f"Error extracting class name from {model_file_path}: {e}")
        return None

def create_fixed_training_script(model_name, class_name):
    """Create a robust training script for the model"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Fixed Training Script for {model_name}
Class: {class_name}
Auto-generated with robust error handling
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
import traceback

# Import the model
try:
    from models.{model_name} import {class_name}
    print(f"✅ Successfully imported {class_name} from models.{model_name}")
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    print(f"🔄 Trying alternative import methods...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("{model_name}", f"models/{model_name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        {class_name} = getattr(module, "{class_name}")
        print(f"✅ Successfully imported {class_name} using importlib")
    except Exception as e2:
        print(f"❌ Alternative import failed: {{e2}}")
        sys.exit(1)

# Import data loader
from core.real_data_loader import create_real_data_loaders

def train_model():
    """Train the {model_name} model"""
    print(f"🚀 Starting training for {model_name} ({class_name})")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {{device}}")
    
    try:
        # Create data loaders
        print("📊 Creating data loaders...")
        train_loader, val_loader, test_loader = create_real_data_loaders(
            batch_size=32,
            num_workers=4
        )
        print(f"✅ Data loaders created successfully")
        print(f"   📈 Train samples: {{len(train_loader.dataset)}}")
        print(f"   📊 Val samples: {{len(val_loader.dataset)}}")
        print(f"   🧪 Test samples: {{len(test_loader.dataset)}}")
        
        # Create model with error handling
        print(f"🏗️  Creating model: {class_name}")
        
        # Try different initialization approaches
        model = None
        input_size = 70
        
        # Approach 1: Default initialization
        try:
            model = {class_name}()
            print(f"✅ Model created with default parameters")
        except Exception as e1:
            print(f"⚠️  Default initialization failed: {{e1}}")
            
            # Approach 2: Try with num_classes parameter
            try:
                model = {class_name}(num_classes=2)
                print(f"✅ Model created with num_classes=2")
            except Exception as e2:
                print(f"⚠️  num_classes initialization failed: {{e2}}")
                
                # Approach 3: Try with common parameters
                try:
                    model = {class_name}(num_classes=2, pretrained=False)
                    print(f"✅ Model created with num_classes=2, pretrained=False")
                except Exception as e3:
                    print(f"❌ All initialization attempts failed: {{e3}}")
                    raise e3
        
        if model is None:
            raise ValueError("Failed to create model")
        
        # Test input size compatibility
        sample_input = torch.randn(1, 3, 70, 70)
        try:
            with torch.no_grad():
                output = model(sample_input)
            
            # Handle different output types
            if isinstance(output, dict):
                if 'classification' in output:
                    output_tensor = output['classification']
                    print(f"✅ Model accepts 70x70 input, multi-output model, classification shape: {{output_tensor.shape}}")
                else:
                    output_tensor = list(output.values())[0]
                    print(f"✅ Model accepts 70x70 input, dict output, first output shape: {{output_tensor.shape}}")
            else:
                output_tensor = output
                print(f"✅ Model accepts 70x70 input, output shape: {{output_tensor.shape}}")
            
            input_size = 70
            
        except Exception as e:
            print(f"⚠️  70x70 input failed: {{e}}")
            print("🔄 Trying 224x224 input...")
            try:
                sample_input = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    output = model(sample_input)
                
                if isinstance(output, dict):
                    if 'classification' in output:
                        output_tensor = output['classification']
                        print(f"✅ Model accepts 224x224 input, multi-output model, classification shape: {{output_tensor.shape}}")
                    else:
                        output_tensor = list(output.values())[0]
                        print(f"✅ Model accepts 224x224 input, dict output, first output shape: {{output_tensor.shape}}")
                else:
                    output_tensor = output
                    print(f"✅ Model accepts 224x224 input, output shape: {{output_tensor.shape}}")
                
                input_size = 224
                
                # Recreate data loaders with 224x224
                train_loader, val_loader, test_loader = create_real_data_loaders(
                    batch_size=32,
                    num_workers=4,
                    img_size=224
                )
                print("✅ Recreated data loaders for 224x224 input")
                
            except Exception as e2:
                print(f"❌ 224x224 input also failed: {{e2}}")
                raise e2
        
        model = model.to(device)
        print(f"✅ Model moved to {{device}}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        # Training parameters
        num_epochs = 30
        best_val_acc = 0.0
        patience = 8
        patience_counter = 0
        
        print(f"🎯 Starting training for {{num_epochs}} epochs...")
        
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
                
                # Handle different output types
                if isinstance(output, dict):
                    if 'classification' in output:
                        classification_output = output['classification']
                    else:
                        classification_output = list(output.values())[0]
                else:
                    classification_output = output
                
                loss = criterion(classification_output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(classification_output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'📊 Epoch {{epoch+1}}/{{num_epochs}}, Batch {{batch_idx}}/{{len(train_loader)}}, '
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
                    
                    # Handle different output types
                    if isinstance(output, dict):
                        if 'classification' in output:
                            classification_output = output['classification']
                        else:
                            classification_output = list(output.values())[0]
                    else:
                        classification_output = output
                    
                    loss = criterion(classification_output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(classification_output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            print(f'📈 Epoch {{epoch+1}}/{{num_epochs}}:')
            print(f'   🚂 Train Loss: {{avg_train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%')
            print(f'   ✅ Val Loss: {{avg_val_loss:.4f}}, Val Acc: {{val_acc:.2f}}%')
            
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
                print(f'💾 New best model saved: {{checkpoint_path}} (Val Acc: {{val_acc:.2f}}%)')
            else:
                patience_counter += 1
                print(f'⏳ No improvement (patience: {{patience_counter}}/{{patience}})')
            
            scheduler.step()
            
            # Early stopping
            if patience_counter >= patience:
                print(f'🛑 Early stopping triggered after {{epoch+1}} epochs')
                break
        
        print(f'🎉 Training completed for {model_name}')
        print(f'🏆 Best validation accuracy: {{best_val_acc:.2f}}%')
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed for {model_name}: {{e}}")
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
    trainer_path = f"trainers/train_{model_name}_fixed.py"
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(trainer_path, 0o755)
    
    return trainer_path

def train_single_model(model_name):
    """Train a single model with comprehensive error handling"""
    print(f"\n{'='*60}")
    print(f"🔧 Processing {model_name}")
    print(f"{'='*60}")
    
    # Check if model file exists
    model_file_path = f"models/{model_name}.py"
    if not os.path.exists(model_file_path):
        print(f"❌ Model file not found: {model_file_path}")
        return {'status': 'failed', 'reason': 'model_file_not_found'}
    
    # Extract class name
    class_name = extract_main_class_name(model_file_path)
    if not class_name:
        print(f"❌ Could not extract class name from {model_file_path}")
        return {'status': 'failed', 'reason': 'class_name_extraction_failed'}
    
    print(f"✅ Extracted class name: {class_name}")
    
    # Create fixed training script
    try:
        trainer_path = create_fixed_training_script(model_name, class_name)
        print(f"✅ Created fixed training script: {trainer_path}")
        
        # Execute the training script
        print(f"🚀 Starting training for {model_name}...")
        
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
                print(f"✅ {model_name} training completed successfully in {duration:.1f}s")
                return {
                    'status': 'success',
                    'class_name': class_name,
                    'trainer_path': trainer_path,
                    'duration': duration,
                    'output': stdout[-1000:]  # Last 1000 chars
                }
            else:
                print(f"❌ {model_name} training failed with return code {process.returncode}")
                print(f"📝 Output: {stdout[-500:]}")  # Last 500 chars
                return {
                    'status': 'failed',
                    'reason': 'training_failed',
                    'return_code': process.returncode,
                    'output': stdout[-500:]
                }
                
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"⏰ {model_name} training timed out after {timeout}s")
            return {
                'status': 'failed',
                'reason': 'timeout',
                'timeout': timeout
            }
            
    except Exception as e:
        print(f"❌ Error creating/running training script: {e}")
        return {
            'status': 'failed',
            'reason': 'script_creation_failed',
            'error': str(e)
        }

def main():
    """Main function to fix and train all remaining models"""
    print("🔧 Starting Systematic Model Fixing and Training")
    print("=" * 80)
    print(f"📊 Total models to fix: {len(FAILED_MODELS)}")
    print("=" * 80)
    
    results = {}
    successful_models = []
    failed_models = []
    
    for i, model_name in enumerate(FAILED_MODELS, 1):
        print(f"\n🎯 Progress: {i}/{len(FAILED_MODELS)} - {model_name}")
        
        result = train_single_model(model_name)
        results[model_name] = result
        
        if result['status'] == 'success':
            successful_models.append(model_name)
            duration = result.get('duration', 0)
            print(f"✅ {model_name} completed in {duration:.1f}s")
        else:
            failed_models.append(model_name)
            reason = result.get('reason', 'unknown')
            print(f"❌ {model_name} failed: {reason}")
        
        # Small delay between models
        time.sleep(2)
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL TRAINING SUMMARY")
    print("=" * 80)
    
    print(f"✅ Successfully trained: {len(successful_models)}/{len(FAILED_MODELS)}")
    for model in successful_models:
        duration = results[model].get('duration', 0)
        print(f"   ✅ {model} ({duration:.1f}s)")
    
    print(f"\n❌ Failed models: {len(failed_models)}/{len(FAILED_MODELS)}")
    for model in failed_models:
        reason = results[model].get('reason', 'unknown')
        print(f"   ❌ {model} (reason: {reason})")
    
    # Save detailed results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(FAILED_MODELS),
        'successful_count': len(successful_models),
        'failed_count': len(failed_models),
        'successful_models': successful_models,
        'failed_models': failed_models,
        'detailed_results': results
    }
    
    with open('fix_remaining_models_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: fix_remaining_models_results.json")
    
    if len(successful_models) == len(FAILED_MODELS):
        print("🎉 ALL MODELS SUCCESSFULLY TRAINED!")
        return True
    else:
        print(f"⚠️  {len(failed_models)} models still need attention")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)