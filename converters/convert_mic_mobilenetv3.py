#!/usr/bin/env python3
"""
ONNX Converter for MIC MobileNetV3 model
Converts trained PyTorch model to ONNX format with performance validation
"""

import os
import sys
import json
import time
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mic_mobilenetv3 import MICMobileNetV3
from core.data_loader import MICDataLoader, create_data_loaders

def load_trained_model(checkpoint_path: str, device: torch.device) -> MICMobileNetV3:
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model
    model = MICMobileNetV3(num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    val_acc = checkpoint.get('val_accuracy', 'N/A')
    if isinstance(val_acc, (int, float)):
        print(f"  Validation accuracy: {val_acc:.4f}")
    else:
        print(f"  Validation accuracy: {val_acc}")
    
    return model

def convert_to_onnx(model: torch.nn.Module, onnx_path: str, device: torch.device) -> bool:
    """Convert PyTorch model to ONNX format"""
    print(f"\nConverting model to ONNX format...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 70, 70).to(device)
    
    # Validate model with dummy input
    try:
        with torch.no_grad():
            pytorch_output = model(dummy_input)
        print(f"✓ PyTorch model validation successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {pytorch_output.shape}")
    except Exception as e:
        print(f"✗ PyTorch model validation failed: {e}")
        return False
    
    # Export to ONNX
    try:
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
            },
            verbose=False
        )
        print(f"✓ ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification successful")
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return False
    
    return True

def compare_outputs(pytorch_model: torch.nn.Module, onnx_path: str, 
                   test_loader, device: torch.device, num_samples: int = 100):
    """Compare outputs between PyTorch and ONNX models"""
    print(f"\nComparing PyTorch vs ONNX outputs...")
    
    # Load ONNX model
    try:
        ort_session = ort.InferenceSession(onnx_path)
        print(f"✓ ONNX Runtime session created")
    except Exception as e:
        print(f"✗ Failed to create ONNX Runtime session: {e}")
        return None
    
    pytorch_model.eval()
    
    # Collect samples for comparison
    pytorch_outputs = []
    onnx_outputs = []
    labels = []
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if sample_count >= num_samples:
                break
                
            data = data.to(device)
            batch_size = data.size(0)
            
            # PyTorch inference
            pytorch_output = pytorch_model(data)
            
            # ONNX inference
            onnx_input = {ort_session.get_inputs()[0].name: data.cpu().numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]
            
            # Store results
            pytorch_outputs.append(pytorch_output.cpu().numpy())
            onnx_outputs.append(onnx_output)
            labels.extend(target.cpu().numpy())
            
            sample_count += batch_size
    
    # Concatenate all outputs
    pytorch_outputs = np.concatenate(pytorch_outputs, axis=0)
    onnx_outputs = np.concatenate(onnx_outputs, axis=0)
    labels = np.array(labels)
    
    # Calculate metrics
    pytorch_preds = np.argmax(pytorch_outputs, axis=1)
    onnx_preds = np.argmax(onnx_outputs, axis=1)
    
    pytorch_accuracy = np.mean(pytorch_preds == labels)
    onnx_accuracy = np.mean(onnx_preds == labels)
    
    # Calculate output differences
    max_diff = np.max(np.abs(pytorch_outputs - onnx_outputs))
    mean_diff = np.mean(np.abs(pytorch_outputs - onnx_outputs))
    
    # Prediction consistency
    prediction_consistency = np.mean(pytorch_preds == onnx_preds)
    
    comparison_results = {
        'samples_compared': len(labels),
        'pytorch_accuracy': float(pytorch_accuracy),
        'onnx_accuracy': float(onnx_accuracy),
        'accuracy_difference': float(abs(pytorch_accuracy - onnx_accuracy)),
        'max_output_difference': float(max_diff),
        'mean_output_difference': float(mean_diff),
        'prediction_consistency': float(prediction_consistency),
        'outputs_identical': bool(max_diff < 1e-5),
        'predictions_identical': bool(prediction_consistency == 1.0)
    }
    
    print(f"✓ Output comparison completed")
    print(f"  Samples compared: {comparison_results['samples_compared']}")
    print(f"  PyTorch accuracy: {comparison_results['pytorch_accuracy']:.4f}")
    print(f"  ONNX accuracy: {comparison_results['onnx_accuracy']:.4f}")
    print(f"  Accuracy difference: {comparison_results['accuracy_difference']:.6f}")
    print(f"  Max output difference: {comparison_results['max_output_difference']:.8f}")
    print(f"  Mean output difference: {comparison_results['mean_output_difference']:.8f}")
    print(f"  Prediction consistency: {comparison_results['prediction_consistency']:.4f}")
    
    return comparison_results

def benchmark_performance(pytorch_model: torch.nn.Module, onnx_path: str, 
                         device: torch.device, num_runs: int = 100):
    """Benchmark inference performance"""
    print(f"\nBenchmarking inference performance...")
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create test input
    test_input = torch.randn(1, 3, 70, 70).to(device)
    test_input_np = test_input.cpu().numpy()
    
    pytorch_model.eval()
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = pytorch_model(test_input)
            _ = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input_np})
    
    # Benchmark PyTorch
    pytorch_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = pytorch_model(test_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            pytorch_times.append(time.time() - start_time)
    
    # Benchmark ONNX
    onnx_times = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input_np})
        onnx_times.append(time.time() - start_time)
    
    pytorch_avg_time = np.mean(pytorch_times) * 1000  # Convert to ms
    onnx_avg_time = np.mean(onnx_times) * 1000
    
    performance_results = {
        'pytorch_avg_time_ms': float(pytorch_avg_time),
        'onnx_avg_time_ms': float(onnx_avg_time),
        'speedup_ratio': float(pytorch_avg_time / onnx_avg_time),
        'pytorch_std_ms': float(np.std(pytorch_times) * 1000),
        'onnx_std_ms': float(np.std(onnx_times) * 1000),
        'num_runs': num_runs
    }
    
    print(f"✓ Performance benchmark completed")
    print(f"  PyTorch avg time: {performance_results['pytorch_avg_time_ms']:.2f} ms")
    print(f"  ONNX avg time: {performance_results['onnx_avg_time_ms']:.2f} ms")
    print(f"  Speedup ratio: {performance_results['speedup_ratio']:.2f}x")
    
    return performance_results

def get_model_info(pytorch_model: torch.nn.Module, onnx_path: str):
    """Get model information"""
    # PyTorch model info
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
    
    # ONNX model info
    onnx_model = onnx.load(onnx_path)
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    model_info = {
        'pytorch_total_params': int(total_params),
        'pytorch_trainable_params': int(trainable_params),
        'onnx_file_size_mb': float(onnx_size_mb),
        'onnx_opset_version': int(onnx_model.opset_import[0].version),
        'input_shape': [1, 3, 70, 70],
        'output_shape': [1, 2]
    }
    
    return model_info

def main():
    print("=" * 60)
    print("MIC MobileNetV3 ONNX Conversion")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "mic_mobilenetv3"
    
    # Find latest checkpoint
    checkpoint_dir = "checkpoints"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(model_name) and f.endswith('.pth')]
    
    if not checkpoint_files:
        print(f"✗ No checkpoint files found for {model_name}")
        return
    
    # Use the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Load trained model
    model = load_trained_model(checkpoint_path, device)
    
    # Create ONNX output path
    os.makedirs('onnx_models', exist_ok=True)
    onnx_path = f"onnx_models/{model_name}_{timestamp}.onnx"
    
    # Convert to ONNX
    if not convert_to_onnx(model, onnx_path, device):
        print("✗ ONNX conversion failed")
        return
    
    # Load test data for comparison
    print(f"\nLoading test data...")
    data_loader = MICDataLoader()
    _, _, test_loader = create_data_loaders(data_loader, batch_size=32, num_workers=4)
    
    # Compare outputs
    comparison_results = compare_outputs(model, onnx_path, test_loader, device, num_samples=500)
    if comparison_results is None:
        print("✗ Output comparison failed")
        return
    
    # Benchmark performance
    performance_results = benchmark_performance(model, onnx_path, device, num_runs=100)
    
    # Get model information
    model_info = get_model_info(model, onnx_path)
    
    # Compile final results
    conversion_results = {
        'model_name': model_name,
        'timestamp': timestamp,
        'checkpoint_path': checkpoint_path,
        'onnx_path': onnx_path,
        'device': str(device),
        'model_info': model_info,
        'comparison_results': comparison_results,
        'performance_results': performance_results,
        'conversion_status': 'success',
        'validation_passed': (
            comparison_results['accuracy_difference'] < 0.01 and
            comparison_results['prediction_consistency'] > 0.99
        )
    }
    
    # Save results
    os.makedirs('reports', exist_ok=True)
    results_path = f"reports/{model_name}_{timestamp}_onnx_conversion.json"
    
    with open(results_path, 'w') as f:
        json.dump(conversion_results, f, indent=2)
    
    print(f"\n" + "=" * 60)
    print("MIC MobileNetV3 ONNX Conversion Summary")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"PyTorch parameters: {model_info['pytorch_total_params']:,}")
    print(f"ONNX file size: {model_info['onnx_file_size_mb']:.2f} MB")
    print(f"Accuracy difference: {comparison_results['accuracy_difference']:.6f}")
    print(f"Prediction consistency: {comparison_results['prediction_consistency']:.4f}")
    print(f"Performance speedup: {performance_results['speedup_ratio']:.2f}x")
    print(f"Validation passed: {'✓' if conversion_results['validation_passed'] else '✗'}")
    print(f"ONNX model: {onnx_path}")
    print(f"Conversion report: {results_path}")
    print("=" * 60)
    
    return conversion_results

if __name__ == "__main__":
    main()