#!/usr/bin/env python3
"""
InceptionMicro ONNX Converter
Converts trained InceptionMicro model to ONNX format with performance validation
"""

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import time
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.inception_micro import InceptionMicro
from core.real_data_loader import create_real_data_loaders

def load_latest_checkpoint():
    """Load the latest InceptionMicro checkpoint"""
    checkpoint_dir = "checkpoints"
    inception_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("inception_micro_") and f.endswith("_best.pth")]
    
    if not inception_files:
        raise FileNotFoundError("No InceptionMicro checkpoint found")
    
    # Sort by timestamp and get the latest
    inception_files.sort(reverse=True)
    latest_checkpoint = os.path.join(checkpoint_dir, inception_files[0])
    
    print(f"Using checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def validate_onnx_performance(pytorch_model, onnx_path, test_loader, num_samples=100):
    """Validate ONNX model performance against PyTorch model"""
    print(f"\n🔍 Validating ONNX performance with {num_samples} samples...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    pytorch_model.eval()
    correct_pytorch = 0
    correct_onnx = 0
    total_samples = 0
    prediction_diffs = []
    
    pytorch_times = []
    onnx_times = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if total_samples >= num_samples:
                break
                
            data = data.cuda()
            batch_size = data.size(0)
            
            # PyTorch inference
            start_time = time.time()
            pytorch_output = pytorch_model(data)
            pytorch_time = time.time() - start_time
            pytorch_times.append(pytorch_time)
            
            pytorch_probs = torch.softmax(pytorch_output, dim=1)
            pytorch_pred = pytorch_output.argmax(dim=1)
            
            # ONNX inference
            data_np = data.cpu().numpy()
            start_time = time.time()
            onnx_output = ort_session.run(None, {'input': data_np})[0]
            onnx_time = time.time() - start_time
            onnx_times.append(onnx_time)
            
            onnx_probs = torch.softmax(torch.from_numpy(onnx_output), dim=1)
            onnx_pred = torch.from_numpy(onnx_output).argmax(dim=1)
            
            # Calculate accuracy
            correct_pytorch += (pytorch_pred.cpu() == target).sum().item()
            correct_onnx += (onnx_pred == target).sum().item()
            
            # Calculate prediction differences
            prob_diff = torch.abs(pytorch_probs.cpu() - onnx_probs).max(dim=1)[0]
            prediction_diffs.extend(prob_diff.tolist())
            
            total_samples += batch_size
    
    # Calculate metrics
    pytorch_accuracy = correct_pytorch / total_samples
    onnx_accuracy = correct_onnx / total_samples
    accuracy_match = min(pytorch_accuracy, onnx_accuracy) / max(pytorch_accuracy, onnx_accuracy)
    
    avg_pytorch_time = np.mean(pytorch_times)
    avg_onnx_time = np.mean(onnx_times)
    speedup = avg_pytorch_time / avg_onnx_time
    
    pred_diff = np.mean(prediction_diffs)
    
    results = {
        'pytorch_accuracy': float(pytorch_accuracy),
        'onnx_accuracy': float(onnx_accuracy),
        'accuracy_match': float(accuracy_match),
        'prediction_consistency': float(1.0 - pred_diff),
        'avg_pytorch_time_ms': float(avg_pytorch_time * 1000),
        'avg_onnx_time_ms': float(avg_onnx_time * 1000),
        'performance_speedup': float(speedup),
        'max_prediction_diff': float(np.max(prediction_diffs)),
        'validation_passed': accuracy_match > 0.99 and pred_diff < 0.01
    }
    
    print(f"📊 PyTorch Accuracy: {pytorch_accuracy:.4f}")
    print(f"📊 ONNX Accuracy: {onnx_accuracy:.4f}")
    print(f"📊 Accuracy Match: {accuracy_match:.4f}")
    print(f"📊 Prediction Consistency: {1.0 - pred_diff:.4f}")
    print(f"⚡ PyTorch Time: {avg_pytorch_time * 1000:.2f}ms")
    print(f"⚡ ONNX Time: {avg_onnx_time * 1000:.2f}ms")
    print(f"⚡ Speedup: {speedup:.2f}x")
    print(f"✅ Validation {'PASSED' if results['validation_passed'] else 'FAILED'}")
    
    return results

def main():
    print("=" * 60)
    print("InceptionMicro ONNX Conversion")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_path = load_latest_checkpoint()
    
    # Create model
    model = InceptionMicro()
    
    # Load checkpoint
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    print("✅ Model loaded successfully")
    
    # Print checkpoint info
    if 'best_val_acc' in checkpoint:
        print(f"📊 Best validation accuracy from checkpoint: {checkpoint['best_val_acc']:.2f}%")
    
    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 70, 70).cuda()
    
    # Test model with dummy input
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Model test successful, output shape: {output.shape}")
    
    # Generate ONNX filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_filename = f"inception_micro_{timestamp}.onnx"
    onnx_path = os.path.join("onnx_models", onnx_filename)
    
    # Create onnx_models directory if it doesn't exist
    os.makedirs("onnx_models", exist_ok=True)
    
    print(f"\n🔄 Converting model to ONNX...")
    
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
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")
        return
    
    # Get model size
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"📦 ONNX model size: {model_size_mb:.2f} MB")
    
    # Load test data for validation
    print(f"\n📊 Loading test data for validation...")
    _, _, test_loader = create_real_data_loaders(batch_size=32)
    
    # Validate ONNX performance
    validation_results = validate_onnx_performance(model, onnx_path, test_loader)
    
    # Save conversion report
    report = {
        'model_name': 'inception_micro',
        'conversion_timestamp': timestamp,
        'checkpoint_path': checkpoint_path,
        'onnx_path': onnx_path,
        'model_size_mb': model_size_mb,
        'validation_results': validation_results,
        'conversion_successful': validation_results['validation_passed']
    }
    
    report_filename = f"inception_micro_onnx_conversion_{timestamp}.json"
    report_path = os.path.join("reports", report_filename)
    os.makedirs("reports", exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Conversion report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ InceptionMicro ONNX conversion completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()