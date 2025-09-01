#!/usr/bin/env python3
"""
ONNX Converter for ResnetMicro Model
Converts trained PyTorch ResnetMicro model to ONNX format with performance validation
"""

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import json
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_micro import ResnetMicro
from core.real_data_loader import create_real_data_loaders

def load_trained_model(checkpoint_path):
    """Load trained ResnetMicro model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Initialize model
    model = ResnetMicro(num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Best validation accuracy from checkpoint: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    return model

def convert_to_onnx(model, onnx_path, input_shape=(1, 3, 70, 70)):
    """Convert PyTorch model to ONNX format"""
    print(f"\n🔄 Converting model to ONNX...")
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
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
    
    print(f"✅ ONNX model saved to: {onnx_path}")
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        return True
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")
        return False

def validate_onnx_performance(pytorch_model, onnx_path, test_loader, num_samples=100):
    """Validate ONNX model performance against PyTorch model"""
    print(f"\n🧪 Validating ONNX model performance...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Test on sample data
    pytorch_predictions = []
    onnx_predictions = []
    pytorch_times = []
    onnx_times = []
    
    pytorch_model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if sample_count >= num_samples:
                break
                
            for i in range(data.size(0)):
                if sample_count >= num_samples:
                    break
                    
                single_input = data[i:i+1]  # Shape: (1, 3, 70, 70)
                
                # PyTorch inference
                start_time = time.time()
                pytorch_output = pytorch_model(single_input)
                pytorch_time = time.time() - start_time
                pytorch_times.append(pytorch_time)
                pytorch_pred = torch.softmax(pytorch_output, dim=1).numpy()
                pytorch_predictions.append(pytorch_pred)
                
                # ONNX inference
                onnx_input = {ort_session.get_inputs()[0].name: single_input.numpy()}
                start_time = time.time()
                onnx_output = ort_session.run(None, onnx_input)[0]
                onnx_time = time.time() - start_time
                onnx_times.append(onnx_time)
                onnx_pred = torch.softmax(torch.from_numpy(onnx_output), dim=1).numpy()
                onnx_predictions.append(onnx_pred)
                
                sample_count += 1
    
    # Calculate metrics
    pytorch_predictions = np.vstack(pytorch_predictions)
    onnx_predictions = np.vstack(onnx_predictions)
    
    # Accuracy comparison
    pytorch_classes = np.argmax(pytorch_predictions, axis=1)
    onnx_classes = np.argmax(onnx_predictions, axis=1)
    accuracy_match = np.mean(pytorch_classes == onnx_classes)
    
    # Prediction difference
    pred_diff = np.mean(np.abs(pytorch_predictions - onnx_predictions))
    max_pred_diff = np.max(np.abs(pytorch_predictions - onnx_predictions))
    
    # Speed comparison
    avg_pytorch_time = np.mean(pytorch_times) * 1000  # ms
    avg_onnx_time = np.mean(onnx_times) * 1000  # ms
    speedup = avg_pytorch_time / avg_onnx_time if avg_onnx_time > 0 else 1.0
    
    # Results
    results = {
        'samples_tested': sample_count,
        'accuracy_match': float(accuracy_match),
        'prediction_consistency': float(1.0 - pred_diff),
        'mean_prediction_difference': float(pred_diff),
        'max_prediction_difference': float(max_pred_diff),
        'pytorch_avg_inference_time_ms': float(avg_pytorch_time),
        'onnx_avg_inference_time_ms': float(avg_onnx_time),
        'performance_speedup': float(speedup),
        'validation_passed': accuracy_match > 0.99 and pred_diff < 0.01
    }
    
    print(f"📊 Performance Validation Results:")
    print(f"   Samples tested: {sample_count}")
    print(f"   Accuracy match: {accuracy_match:.4f} ({accuracy_match*100:.2f}%)")
    print(f"   Prediction consistency: {results['prediction_consistency']:.4f}")
    print(f"   Mean prediction difference: {pred_diff:.6f}")
    print(f"   Max prediction difference: {max_pred_diff:.6f}")
    print(f"   PyTorch avg inference time: {avg_pytorch_time:.2f}ms")
    print(f"   ONNX avg inference time: {avg_onnx_time:.2f}ms")
    print(f"   Performance speedup: {speedup:.2f}x")
    
    if results['validation_passed']:
        print("✅ ONNX conversion validation PASSED")
    else:
        print("❌ ONNX conversion validation FAILED")
    
    return results

def get_model_size(file_path):
    """Get model file size in MB"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def main():
    print("=" * 60)
    print("ResnetMicro ONNX Conversion")
    print("=" * 60)
    
    # Find latest checkpoint
    checkpoint_dir = "checkpoints"
    resnet_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("resnet_micro_") and f.endswith("_best.pth")]
    
    if not resnet_checkpoints:
        print("❌ No ResnetMicro checkpoints found!")
        return
    
    # Use the latest checkpoint
    latest_checkpoint = sorted(resnet_checkpoints)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load trained model
    model = load_trained_model(checkpoint_path)
    
    # Create output directory
    os.makedirs("onnx_models", exist_ok=True)
    
    # Generate timestamp and output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_filename = f"resnet_micro_{timestamp}.onnx"
    onnx_path = os.path.join("onnx_models", onnx_filename)
    
    # Convert to ONNX
    conversion_success = convert_to_onnx(model, onnx_path)
    
    if not conversion_success:
        print("❌ ONNX conversion failed!")
        return
    
    # Get model size
    model_size_mb = get_model_size(onnx_path)
    print(f"📦 ONNX model size: {model_size_mb:.2f} MB")
    
    # Load test data for validation
    print("\n📊 Loading test data for validation...")
    _, _, test_loader = create_real_data_loaders(batch_size=32, num_workers=4)
    
    # Validate performance
    validation_results = validate_onnx_performance(model, onnx_path, test_loader)
    
    # Create conversion report
    conversion_report = {
        'model_name': 'resnet_micro',
        'conversion_timestamp': timestamp,
        'pytorch_checkpoint': checkpoint_path,
        'onnx_model_path': onnx_path,
        'model_size_mb': model_size_mb,
        'conversion_successful': conversion_success,
        'validation_results': validation_results,
        'conversion_settings': {
            'opset_version': 11,
            'input_shape': [1, 3, 70, 70],
            'dynamic_axes': True
        }
    }
    
    # Save conversion report
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/resnet_micro_{timestamp}_onnx_conversion.json"
    
    with open(report_path, 'w') as f:
        json.dump(conversion_report, f, indent=2)
    
    print(f"\n📄 Conversion report saved: {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ResnetMicro ONNX Conversion Summary")
    print("=" * 60)
    print(f"✅ Conversion: {'SUCCESS' if conversion_success else 'FAILED'}")
    print(f"📦 Model size: {model_size_mb:.2f} MB")
    print(f"🚀 Performance speedup: {validation_results['performance_speedup']:.2f}x")
    print(f"🎯 Accuracy match: {validation_results['accuracy_match']*100:.2f}%")
    print(f"✅ Validation: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
    print(f"💾 ONNX model: {onnx_path}")
    print(f"📄 Report: {report_path}")
    print("=" * 60)
    
    return conversion_report

if __name__ == "__main__":
    main()