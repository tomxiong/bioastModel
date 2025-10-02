"""
Test script for Enhanced MIC MobileNetV3 model.
This script validates that the enhanced model can be created and run inference correctly.
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.enhanced_mic_mobilenetv3 import create_enhanced_mic_mobilenetv3, MICFocalLoss

def test_model_creation():
    """Test enhanced model creation and basic functionality."""
    print("🔍 Testing Enhanced MIC MobileNetV3 Model...")
    
    # Test model creation
    try:
        model = create_enhanced_mic_mobilenetv3(
            num_classes=2,
            model_size='small',
            use_cbam=True
        )
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Get model info
    model_info = model.get_model_info()
    print(f"📊 Model Information:")
    print(f"   Name: {model_info['name']}")
    print(f"   Architecture: {model_info['architecture']}")
    print(f"   Total Parameters: {model_info['total_parameters']:,}")
    print(f"   Input Size: {model_info['input_size']}")
    print(f"   Features: {model_info['features']}")
    
    return model

def test_forward_pass(model):
    """Test forward pass with dummy data."""
    print("\n🔄 Testing Forward Pass...")
    
    model.eval()
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 70, 70)
    
    try:
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("✅ Forward pass successful")
        print(f"📤 Output Components:")
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"   {key}: (dict)")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        print(f"     {sub_key}: {sub_value.shape}")
                    else:
                        print(f"     {sub_key}: {type(sub_value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_focal_loss():
    """Test MIC Focal Loss function."""
    print("\n🎯 Testing MIC Focal Loss...")
    
    try:
        # Create loss function
        class_weights = torch.tensor([0.58, 0.42])  # For class imbalance
        focal_loss = MICFocalLoss(
            alpha=0.75,
            gamma=2.0,
            class_weights=class_weights
        )
        
        # Test with dummy data
        logits = torch.randn(4, 2)  # 4 samples, 2 classes
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = focal_loss(logits, targets)
        print(f"✅ Focal Loss computation successful")
        print(f"   Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Focal Loss test failed: {e}")
        return False

def test_training_compatibility():
    """Test training mode compatibility."""
    print("\n🏋️ Testing Training Mode Compatibility...")
    
    try:
        model = create_enhanced_mic_mobilenetv3()
        model.train()
        
        # Test with gradient computation
        dummy_input = torch.randn(2, 3, 70, 70, requires_grad=True)
        outputs = model(dummy_input)
        
        # Test backward pass
        loss = outputs['classification'].sum()
        loss.backward()
        
        print("✅ Training mode compatibility verified")
        print("✅ Gradient computation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Training compatibility test failed: {e}")
        return False

def test_gpu_compatibility():
    """Test GPU compatibility if available."""
    if not torch.cuda.is_available():
        print("\n🚫 GPU not available, skipping GPU test")
        return True
    
    print("\n🖥️ Testing GPU Compatibility...")
    
    try:
        device = torch.device('cuda')
        model = create_enhanced_mic_mobilenetv3()
        model.to(device)
        
        dummy_input = torch.randn(2, 3, 70, 70).to(device)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("✅ GPU compatibility verified")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Input device: {dummy_input.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU compatibility test failed: {e}")
        return False

def performance_benchmark(model):
    """Basic performance benchmark."""
    print("\n⚡ Performance Benchmark...")
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Warmup
    dummy_input = torch.randn(1, 3, 70, 70).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    num_runs = 100
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_time.record()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    else:
        import time
        start = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        elapsed_time = time.time() - start
    
    avg_inference_time = elapsed_time / num_runs * 1000  # Convert to ms
    fps = 1000 / avg_inference_time
    
    print(f"📈 Performance Results:")
    print(f"   Average inference time: {avg_inference_time:.2f} ms")
    print(f"   Throughput: {fps:.1f} FPS")
    print(f"   Device: {device}")

def main():
    """Run all tests."""
    print("🧪 Enhanced MIC MobileNetV3 Comprehensive Test Suite")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Model Creation
    model = test_model_creation()
    if not model:
        all_tests_passed = False
        return
    
    # Test 2: Forward Pass
    if not test_forward_pass(model):
        all_tests_passed = False
    
    # Test 3: Focal Loss
    if not test_focal_loss():
        all_tests_passed = False
    
    # Test 4: Training Compatibility
    if not test_training_compatibility():
        all_tests_passed = False
    
    # Test 5: GPU Compatibility
    if not test_gpu_compatibility():
        all_tests_passed = False
    
    # Test 6: Performance Benchmark
    try:
        performance_benchmark(model)
    except Exception as e:
        print(f"⚠️ Performance benchmark failed: {e}")
    
    # Final Report
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All tests passed! Enhanced MIC MobileNetV3 is ready for training.")
        print("\n📋 Next Steps:")
        print("   1. Run training with: python scripts/train_enhanced_mic_mobilenetv3.py")
        print("   2. Use enhanced configuration: --config enhanced_mic_mobilenetv3_optimized")
        print("   3. Monitor training progress for improved accuracy")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()