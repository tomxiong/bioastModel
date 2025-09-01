#!/usr/bin/env python3
"""
Complete ONNX conversion for all trained models
Ensures all models are converted to ONNX format with performance validation
"""

import os
import sys
import json
import glob
import torch
import torch.onnx
import numpy as np
from datetime import datetime
import importlib
import traceback

def get_trained_models():
    """Get list of all trained models from checkpoints"""
    checkpoint_files = glob.glob('checkpoints/*.pth')
    trained_models = {}
    
    for checkpoint in checkpoint_files:
        filename = os.path.basename(checkpoint)
        parts = filename.replace('_best.pth', '').split('_')
        if len(parts) >= 3:
            model_name = '_'.join(parts[:-2])
            if model_name not in trained_models or filename > trained_models[model_name]['filename']:
                trained_models[model_name] = {
                    'checkpoint_path': checkpoint,
                    'filename': filename
                }
    
    return trained_models

def get_existing_onnx_models():
    """Get list of existing ONNX models"""
    onnx_files = glob.glob('onnx_models/*.onnx')
    existing_onnx = set()
    
    for onnx_file in onnx_files:
        filename = os.path.basename(onnx_file)
        # Extract model name from ONNX filename
        parts = filename.split('_')
        if len(parts) >= 2:
            model_name = '_'.join(parts[:2])
            existing_onnx.add(model_name)
    
    return existing_onnx

def load_model_class(model_name):
    """Load model class from models directory"""
    try:
        module = importlib.import_module(f'models.{model_name}')
        
        # Common model class name patterns
        possible_names = [
            model_name.title().replace('_', ''),
            model_name.upper(),
            model_name,
            'Model',
            'Net',
            'Network'
        ]
        
        for name in possible_names:
            if hasattr(module, name):
                return getattr(module, name)
        
        # Find first nn.Module subclass
        import torch.nn as nn
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                return attr
        
        raise ValueError(f"No suitable model class found in models.{model_name}")
        
    except Exception as e:
        print(f"Error loading model class for {model_name}: {e}")
        return None

def convert_model_to_onnx(model_name, checkpoint_path):
    """Convert a single model to ONNX format"""
    print(f"Converting {model_name} to ONNX...")
    
    try:
        # Load model class
        ModelClass = load_model_class(model_name)
        if ModelClass is None:
            return False, "Failed to load model class"
        
        # Create model instance
        try:
            model = ModelClass(num_classes=4)  # bioast_dataset has 4 classes
        except:
            try:
                model = ModelClass()
            except Exception as e:
                return False, f"Failed to create model instance: {e}"
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 70, 70)
        
        # Test model forward pass
        with torch.no_grad():
            output = model(dummy_input)
            if isinstance(output, dict):
                if 'classification' in output:
                    output = output['classification']
                else:
                    output = list(output.values())[0]
        
        # Generate ONNX filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        onnx_path = f"onnx_models/{model_name}_{timestamp}.onnx"
        
        # Convert to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            import onnxruntime as ort
            ort_session = ort.InferenceSession(onnx_path)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare outputs
            pytorch_output = output.detach().numpy()
            onnx_output = ort_outputs[0]
            
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            result_info = {
                'model_name': model_name,
                'onnx_path': onnx_path,
                'file_size_mb': file_size,
                'max_difference': float(max_diff),
                'validation_accuracy': float(checkpoint.get('val_acc', 0)),
                'conversion_timestamp': timestamp,
                'status': 'success'
            }
            
            return True, result_info
            
        except ImportError:
            print("Warning: onnx or onnxruntime not available for verification")
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)
            result_info = {
                'model_name': model_name,
                'onnx_path': onnx_path,
                'file_size_mb': file_size,
                'validation_accuracy': float(checkpoint.get('val_acc', 0)),
                'conversion_timestamp': timestamp,
                'status': 'success_unverified'
            }
            return True, result_info
            
    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        print(f"Error converting {model_name}: {error_msg}")
        traceback.print_exc()
        return False, error_msg

def main():
    """Main ONNX conversion pipeline"""
    print("🔄 Starting comprehensive ONNX conversion...")
    
    # Create onnx_models directory
    os.makedirs('onnx_models', exist_ok=True)
    
    # Get trained models and existing ONNX models
    trained_models = get_trained_models()
    existing_onnx = get_existing_onnx_models()
    
    print(f"Found {len(trained_models)} trained models")
    print(f"Found {len(existing_onnx)} existing ONNX models")
    
    # Find models that need ONNX conversion
    models_to_convert = []
    for model_name in trained_models:
        if model_name not in existing_onnx:
            models_to_convert.append(model_name)
    
    print(f"Models needing ONNX conversion: {len(models_to_convert)}")
    
    if not models_to_convert:
        print("✅ All trained models already have ONNX versions!")
        return
    
    # Convert models
    conversion_results = []
    successful_conversions = 0
    
    for i, model_name in enumerate(models_to_convert, 1):
        print(f"\n{'='*60}")
        print(f"Converting {i}/{len(models_to_convert)}: {model_name}")
        print(f"{'='*60}")
        
        checkpoint_path = trained_models[model_name]['checkpoint_path']
        success, result = convert_model_to_onnx(model_name, checkpoint_path)
        
        if success:
            print(f"✅ Successfully converted {model_name}")
            if isinstance(result, dict):
                print(f"   File size: {result['file_size_mb']:.2f} MB")
                if 'max_difference' in result:
                    print(f"   Max output difference: {result['max_difference']:.2e}")
            successful_conversions += 1
        else:
            print(f"❌ Failed to convert {model_name}: {result}")
        
        conversion_results.append({
            'model_name': model_name,
            'success': success,
            'result': result
        })
    
    # Generate summary
    print(f"\n{'='*60}")
    print("ONNX CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models processed: {len(models_to_convert)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {len(models_to_convert) - successful_conversions}")
    print(f"Success rate: {(successful_conversions / len(models_to_convert) * 100):.1f}%")
    
    # Save detailed results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'total_trained_models': len(trained_models),
        'existing_onnx_models': len(existing_onnx),
        'models_converted': len(models_to_convert),
        'successful_conversions': successful_conversions,
        'conversion_rate': (successful_conversions / len(models_to_convert) * 100) if models_to_convert else 100,
        'conversion_results': conversion_results
    }
    
    with open('onnx_conversion_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to: onnx_conversion_results.json")
    
    # List successful conversions
    if successful_conversions > 0:
        print(f"\n✅ Successfully converted models:")
        for result in conversion_results:
            if result['success']:
                print(f"  - {result['model_name']}")
    
    # List failed conversions
    failed_conversions = [r for r in conversion_results if not r['success']]
    if failed_conversions:
        print(f"\n❌ Failed conversions:")
        for result in failed_conversions:
            print(f"  - {result['model_name']}: {result['result']}")

if __name__ == "__main__":
    main()