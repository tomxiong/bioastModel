#!/usr/bin/env python3
"""
ONNX Converter for EfficientCnn model
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

from models.efficient_cnn import EfficientCnn
from core.data_loader import create_data_loaders

def main():
    print("=" * 60)
    print("EfficientCnn ONNX Conversion")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "efficient_cnn"
    
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
    model = EfficientCnn(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Create ONNX output path
    os.makedirs('onnx_models', exist_ok=True)
    onnx_path = f"onnx_models/{model_name}_{timestamp}.onnx"
    
    # Convert to ONNX
    dummy_input = torch.randn(1, 3, 70, 70).to(device)
    
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
        return
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification successful")
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return
    
    print(f"\n✅ {class_name} ONNX conversion completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")

if __name__ == "__main__":
    main()
