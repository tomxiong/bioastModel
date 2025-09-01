#!/usr/bin/env python3
"""
Final training script for remaining untrained models
Continues the biomedical image analysis model training pipeline
"""

import os
import sys
import json
import glob
import importlib
import subprocess
from datetime import datetime
import traceback

def get_model_status():
    """Get current training status of all models"""
    # Get all model files
    model_files = []
    for f in glob.glob('models/*.py'):
        if '__init__' not in f and '.pkl' not in f:
            model_name = os.path.basename(f).replace('.py', '')
            model_files.append(model_name)
    
    # Get trained models from checkpoints
    checkpoint_files = glob.glob('checkpoints/*.pth')
    trained_models = set()
    for checkpoint in checkpoint_files:
        filename = os.path.basename(checkpoint)
        parts = filename.split('_')
        if len(parts) >= 2:
            model_name = '_'.join(parts[:-2]) if parts[-1] == 'best.pth' else '_'.join(parts[:-1])
            trained_models.add(model_name)
    
    # Find untrained models
    untrained = [model for model in model_files if model not in trained_models]
    
    return {
        'total_models': len(model_files),
        'trained_models': list(trained_models),
        'untrained_models': untrained,
        'trained_count': len(trained_models),
        'untrained_count': len(untrained)
    }

def create_trainer_script(model_name):
    """Create a training script for the given model"""
    trainer_content = f'''#!/usr/bin/env python3
"""
Training script for {model_name}
Auto-generated for biomedical image analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime
import traceback

# Import the model
try:
    from models.{model_name} import *
except ImportError as e:
    print(f"Error importing model {{model_name}}: {{e}}")
    sys.exit(1)

# Import data loader
from core.real_data_loader import create_real_data_loaders

def get_model_class():
    """Get the model class from the module"""
    import models.{model_name} as model_module
    
    # Common model class name patterns
    possible_names = [
        '{model_name.title().replace("_", "")}',
        '{model_name.upper()}',
        '{model_name}',
        'Model',
        'Net',
        'Network'
    ]
    
    for name in possible_names:
        if hasattr(model_module, name):
            return getattr(model_module, name)
    
    # If no standard name found, get the first class that's a nn.Module
    for attr_name in dir(model_module):
        attr = getattr(model_module, attr_name)
        if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
            return attr
    
    raise ValueError(f"No suitable model class found in models.{model_name}")

def train_model():
    """Train the model"""
    print(f"Starting training for {{model_name}}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {{device}}")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32, num_workers=4)
        print(f"Data loaders created successfully")
        print(f"Train samples: {{len(train_loader.dataset)}}")
        print(f"Val samples: {{len(val_loader.dataset)}}")
        print(f"Test samples: {{len(test_loader.dataset)}}")
        
        # Get model class and create instance
        ModelClass = get_model_class()
        
        # Try different initialization approaches
        try:
            model = ModelClass(num_classes=len(train_loader.dataset.classes))
        except:
            try:
                model = ModelClass()
            except Exception as e:
                print(f"Error creating model: {{e}}")
                return False
        
        model = model.to(device)
        print(f"Model created and moved to device")
        
        # Test model with sample input
        sample_input = torch.randn(1, 3, 70, 70).to(device)
        try:
            with torch.no_grad():
                output = model(sample_input)
                if isinstance(output, dict):
                    # Handle multi-output models
                    if 'classification' in output:
                        output = output['classification']
                    else:
                        output = list(output.values())[0]
                print(f"Model output shape: {{output.shape}}")
        except Exception as e:
            print(f"Error testing model: {{e}}")
            return False
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=30)
        
        # Training loop
        best_val_acc = 0.0
        patience = 8
        patience_counter = 0
        
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
                    if 'classification' in output:
                        output = output['classification']
                    else:
                        output = list(output.values())[0]
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {{epoch+1}}, Batch {{batch_idx}}, Loss: {{loss.item():.4f}}')
            
            train_acc = 100. * train_correct / train_total
            
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
                        if 'classification' in output:
                            output = output['classification']
                        else:
                            output = list(output.values())[0]
                    
                    val_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100. * val_correct / val_total
            scheduler.step()
            
            print(f'Epoch {{epoch+1}}/30:')
            print(f'  Train Loss: {{train_loss/len(train_loader):.4f}}, Train Acc: {{train_acc:.2f}}%')
            print(f'  Val Loss: {{val_loss/len(val_loader):.4f}}, Val Acc: {{val_acc:.2f}}%')
            
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
                    'train_acc': train_acc
                }}, checkpoint_path)
                print(f'  New best model saved: {{checkpoint_path}}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'  Early stopping triggered after {{epoch+1}} epochs')
                    break
        
        # Test phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Handle multi-output models
                if isinstance(output, dict):
                    if 'classification' in output:
                        output = output['classification']
                    else:
                        output = list(output.values())[0]
                
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        # Save results
        results = {{
            'model_name': '{model_name}',
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }}
        
        with open(f'results/{model_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training completed for {model_name}")
        print(f"Best validation accuracy: {{best_val_acc:.2f}}%")
        print(f"Test accuracy: {{test_acc:.2f}}%")
        
        return True
        
    except Exception as e:
        print(f"Error during training: {{e}}")
        traceback.print_exc()
        
        # Save error results
        results = {{
            'model_name': '{model_name}',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'status': 'failed'
        }}
        
        with open(f'results/{model_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return False

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
'''
    
    # Create trainer file
    trainer_path = f'trainers/train_{model_name}.py'
    os.makedirs('trainers', exist_ok=True)
    with open(trainer_path, 'w') as f:
        f.write(trainer_content)
    
    return trainer_path

def main():
    """Main training pipeline"""
    print("=== Final Remaining Models Training Pipeline ===")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get current status
    status = get_model_status()
    
    print(f"Total models: {status['total_models']}")
    print(f"Trained models: {status['trained_count']}")
    print(f"Untrained models: {status['untrained_count']}")
    
    if status['untrained_count'] == 0:
        print("All models are already trained!")
        return
    
    print(f"\nUntrained models to process:")
    for model in status['untrained_models']:
        print(f"  - {model}")
    
    # Train each untrained model
    results = []
    
    for i, model_name in enumerate(status['untrained_models'], 1):
        print(f"\n{'='*60}")
        print(f"Training model {i}/{len(status['untrained_models'])}: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Create trainer script
            trainer_path = create_trainer_script(model_name)
            print(f"Created trainer: {trainer_path}")
            
            # Run training
            result = subprocess.run([
                sys.executable, trainer_path
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                print(f"✅ Successfully trained {model_name}")
                results.append({
                    'model': model_name,
                    'status': 'success',
                    'output': result.stdout
                })
            else:
                print(f"❌ Failed to train {model_name}")
                print(f"Error: {result.stderr}")
                results.append({
                    'model': model_name,
                    'status': 'failed',
                    'error': result.stderr,
                    'output': result.stdout
                })
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Training timeout for {model_name}")
            results.append({
                'model': model_name,
                'status': 'timeout',
                'error': 'Training exceeded 30 minute timeout'
            })
        except Exception as e:
            print(f"💥 Exception training {model_name}: {e}")
            results.append({
                'model': model_name,
                'status': 'exception',
                'error': str(e)
            })
    
    # Generate final report
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"\n{'='*60}")
    print("FINAL TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total models processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✅ Successfully trained models:")
        for result in successful:
            print(f"  - {result['model']}")
    
    if failed:
        print(f"\n❌ Failed models:")
        for result in failed:
            print(f"  - {result['model']}: {result['status']}")
    
    # Save detailed results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(results),
        'successful_count': len(successful),
        'failed_count': len(failed),
        'success_rate': len(successful) / len(results) * 100 if results else 0,
        'results': results
    }
    
    with open('final_training_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to: final_training_results.json")
    print(f"Overall success rate: {final_results['success_rate']:.1f}%")

if __name__ == "__main__":
    main()