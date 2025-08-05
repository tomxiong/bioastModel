"""
Individual Model Analysis - Clean Version
Focused analysis for single models
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.individual_model_analyzer import analyze_single_model

def main():
    """Analyze ResNet18-Improved model first"""
    
    model_config = {
        'module': 'models.resnet_improved',
        'factory_function': 'create_resnet18_improved',
        'input_shape': (3, 70, 70),
        'architecture_type': 'CNN',
        'checkpoint_path': 'experiments/experiment_20250802_164948/resnet18_improved/best_model.pth'
    }
    
    print("Analyzing ResNet18-Improved model...")
    result = analyze_single_model('resnet18_improved', model_config)
    
    if result['success']:
        summary = result['summary']
        print(f"\nRESNET18_IMPROVED ANALYSIS RESULTS:")
        print(f"  PyTorch Accuracy: {summary['pytorch_accuracy']:.1f}%")
        print(f"  ONNX Accuracy: {summary['onnx_accuracy']:.1f}%")
        print(f"  Speedup: {summary['speedup']:.2f}x")
        print(f"  Disagreements: {summary['disagreements']}")
        print(f"  Assessment: {summary['overall_assessment']}")
        print(f"  HTML Report: {result['html_report']}")
        print(f"  JSON Report: {result['json_report']}")
    else:
        print(f"Analysis failed: {result['error']}")

if __name__ == "__main__":
    main()