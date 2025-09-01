import torch
import torch.onnx
import os
import json
from datetime import datetime
import onnxruntime as ort
import numpy as np

# Import model classes
from models.mic_mobilenetv3 import MICMobileNetV3
from models.efficientnet_v2 import create_efficientnetv2_s
from models.resnet_micro import ResnetMicro
from models.inception_micro import InceptionMicro

def load_model_with_checkpoint(model, checkpoint_path):
    """Load model with checkpoint weights"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {checkpoint_path}")
        return True
    else:
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return False

def verify_onnx_model(onnx_path, pytorch_model, dummy_input):
    """Verify ONNX model produces same output as PyTorch model"""
    try:
        # Run PyTorch model
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input)
        
        # Run ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        onnx_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]
        
        # Compare outputs
        pytorch_np = pytorch_output.numpy()
        max_diff = np.max(np.abs(pytorch_np - onnx_output))
        
        print(f"Max difference between PyTorch and ONNX: {max_diff:.8f}")
        
        if max_diff < 1e-5:
            print("✅ ONNX conversion verified - outputs match!")
            return True
        else:
            print("❌ ONNX conversion failed - outputs don't match!")
            return False
            
    except Exception as e:
        print(f"❌ ONNX verification failed: {str(e)}")
        return False

def convert_top4_models_to_onnx():
    """Convert the top 4 high-performance models to ONNX format"""
    
    # Create output directory
    os.makedirs('onnx_models', exist_ok=True)
    
    # Define the top 4 models with their checkpoints
    models_config = [
        {
            'name': 'MIC_MobileNetV3',
            'model_class': MICMobileNetV3,
            'model_args': {'num_classes': 2},
            'checkpoint': 'checkpoints/mic_mobilenetv3_20250809_105643_best.pth',
            'input_size': (1, 3, 70, 70),
            'performance': {'val_acc': 97.15, 'test_acc': 95.05}
        },
        {
            'name': 'EfficientNet_V2_S',
            'model_class': create_efficientnetv2_s,
            'model_args': {'num_classes': 2},
            'checkpoint': 'checkpoints/efficientnet_v2_s_20250809_104714_best.pth',
            'input_size': (1, 3, 70, 70),
            'performance': {'val_acc': 95.85, 'test_acc': 92.18}
        },
        {
            'name': 'ResNet_Micro',
            'model_class': ResnetMicro,
            'model_args': {'num_classes': 2},
            'checkpoint': 'checkpoints/resnet_micro_20250809_105509_best.pth',
            'input_size': (1, 3, 70, 70),
            'performance': {'val_acc': 94.04, 'test_acc': 84.75}
        },
        {
            'name': 'Inception_Micro',
            'model_class': InceptionMicro,
            'model_args': {'num_classes': 2},
            'checkpoint': 'checkpoints/inception_micro_20250809_105443_best.pth',
            'input_size': (1, 3, 70, 70),
            'performance': {'val_acc': 93.01, 'test_acc': 86.83}
        }
    ]
    
    conversion_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("Converting Top 4 High-Performance Models to ONNX")
    print("=" * 60)
    
    for i, config in enumerate(models_config, 1):
        print(f"\n[{i}/4] Converting {config['name']}...")
        print(f"Performance: {config['performance']['val_acc']:.2f}% val, {config['performance']['test_acc']:.2f}% test")
        
        try:
            # Initialize model
            if config['name'] == 'EfficientNet_V2_S':
                model = config['model_class'](**config['model_args'])
            else:
                model = config['model_class'](**config['model_args'])
            
            model.eval()
            
            # Load checkpoint
            checkpoint_loaded = load_model_with_checkpoint(model, config['checkpoint'])
            
            # Create dummy input
            dummy_input = torch.randn(config['input_size'])
            
            # Define output path
            onnx_filename = f"{config['name'].lower()}_{timestamp}.onnx"
            onnx_path = os.path.join('onnx_models', onnx_filename)
            
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
            
            # Verify conversion
            verification_passed = verify_onnx_model(onnx_path, model, dummy_input)
            
            # Record results
            result = {
                'model_name': config['name'],
                'onnx_path': onnx_path,
                'checkpoint_loaded': checkpoint_loaded,
                'conversion_success': True,
                'verification_passed': verification_passed,
                'input_shape': config['input_size'],
                'performance': config['performance'],
                'file_size_mb': round(os.path.getsize(onnx_path) / (1024*1024), 2)
            }
            
            conversion_results.append(result)
            
            print(f"✅ Successfully converted {config['name']}")
            print(f"   Output: {onnx_path}")
            print(f"   File size: {result['file_size_mb']} MB")
            print(f"   Verification: {'✅ Passed' if verification_passed else '❌ Failed'}")
            
        except Exception as e:
            print(f"❌ Failed to convert {config['name']}: {str(e)}")
            result = {
                'model_name': config['name'],
                'onnx_path': None,
                'checkpoint_loaded': False,
                'conversion_success': False,
                'verification_passed': False,
                'error': str(e),
                'performance': config['performance']
            }
            conversion_results.append(result)
    
    # Save conversion results
    results_file = f'onnx_conversion_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_models': len(models_config),
            'successful_conversions': sum(1 for r in conversion_results if r['conversion_success']),
            'verified_conversions': sum(1 for r in conversion_results if r.get('verification_passed', False)),
            'results': conversion_results
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in conversion_results if r['conversion_success'])
    verified = sum(1 for r in conversion_results if r.get('verification_passed', False))
    
    print(f"Total models: {len(models_config)}")
    print(f"Successful conversions: {successful}/{len(models_config)}")
    print(f"Verified conversions: {verified}/{len(models_config)}")
    print(f"Results saved to: {results_file}")
    
    if successful == len(models_config) and verified == len(models_config):
        print("\n🎉 All models successfully converted and verified!")
    elif successful == len(models_config):
        print("\n⚠️  All models converted but some verification failed")
    else:
        print("\n❌ Some conversions failed")
    
    return conversion_results

if __name__ == "__main__":
    convert_top4_models_to_onnx()