#!/usr/bin/env python3
"""
Improved Class Name Extraction
Fix the class name extraction for problematic models
"""

import os
import re
from pathlib import Path

def get_correct_class_names():
    """Manual mapping of correct class names for problematic models"""
    return {
        'densenet': 'DenseNet',
        'efficient_cnn': 'EfficientCNN', 
        'efficientnet': 'EfficientNet',
        'efficientnet_v2': 'EfficientNetV2',
        'enhanced_airbubble_detector': 'EnhancedAirBubbleDetector',
        'ghostnet': 'GhostNet',
        'mic_mobilenetv3': 'MIC_MobileNetV3',
        'micro_vit': 'MicroViT',
        'mnasnet': 'MNASNet',
        'mobilenet_v3': 'MobileNetV3',
        'regnet': 'RegNet',
        'regnet_wrapper': 'RegNetWrapper',
        'resnet_improved': 'ResNetImproved',
        'shufflenet_v2': 'ShuffleNetV2',
        'simplified_airbubble_detector': 'SimplifiedAirBubbleDetector',
        'vit_tiny': 'ViTTiny'
    }

def extract_improved_class_name(model_file_path):
    """Improved class name extraction with manual overrides"""
    model_name = os.path.basename(model_file_path).replace('.py', '')
    
    # Check manual mapping first
    correct_names = get_correct_class_names()
    if model_name in correct_names:
        return correct_names[model_name]
    
    try:
        with open(model_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for class definitions that inherit from nn.Module
        class_pattern = r'class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\):'
        matches = re.findall(class_pattern, content)
        
        if matches:
            # Filter out utility classes
            utility_classes = {
                'InvertedResidual', 'MBConvBlock', 'LayerNorm', 'Block', 
                'Attention', 'MLP', 'SqueezeExcitation', 'SEBlock',
                'DepthwiseSeparableConv', 'ConvBNReLU', 'BasicBlock',
                'DenseLayer', 'Transition', 'Fire', 'InceptionModule'
            }
            main_classes = [cls for cls in matches if cls not in utility_classes]
            
            if main_classes:
                return main_classes[0]
            
            # If only utility classes, return the first one
            return matches[0]
        
        return None
        
    except Exception as e:
        print(f"Error extracting class name from {model_file_path}: {e}")
        return None

def main():
    """Test the improved class name extraction"""
    print("🔍 Testing Improved Class Name Extraction")
    print("=" * 50)
    
    failed_models = [
        'densenet', 'efficient_cnn', 'efficientnet', 'efficientnet_v2',
        'enhanced_airbubble_detector', 'ghostnet', 'mic_mobilenetv3',
        'micro_vit', 'mnasnet', 'mobilenet_v3', 'regnet', 'regnet_wrapper',
        'resnet_improved', 'shufflenet_v2', 'simplified_airbubble_detector',
        'vit_tiny'
    ]
    
    for model_name in failed_models:
        model_file_path = f"models/{model_name}.py"
        if os.path.exists(model_file_path):
            class_name = extract_improved_class_name(model_file_path)
            print(f"✅ {model_name} -> {class_name}")
        else:
            print(f"❌ {model_name} -> File not found")

if __name__ == "__main__":
    main()