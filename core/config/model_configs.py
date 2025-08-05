"""
Model configuration definitions for all supported models.
"""

MODEL_CONFIGS = {
    'simplified_airbubble_detector': {
        'name': 'simplified_airbubble_detector',
        'class_name': 'SimplifiedAirBubbleDetector',
        'module_path': 'models.simplified_airbubble_detector',
        'params_millions': 0.14,
        'parameters': 139266,  # Actual parameter count
        'input_size': 70,
        'num_classes': 2,
        'description': 'Lightweight CNN model designed for efficient air bubble detection with minimal parameters to prevent overfitting',
        'color': 'cyan',  # For visualization
        'experiment_pattern': 'experiment_20250803_180308'  # Current experiment pattern
    },
    'efficientnet_b0': {
        'name': 'efficientnet_b0',
        'class_name': 'EfficientNetB0',
        'module_path': 'models.efficientnet',
        'params_millions': 1.56,
        'parameters': 1560000,  # For compatibility
        'input_size': 70,
        'num_classes': 2,
        'description': 'EfficientNet-B0 lightweight model optimized for efficiency',
        'color': 'blue',  # For visualization
        'experiment_pattern': 'experiment_20250802_140818'  # Current experiment pattern
    },
    'resnet18_improved': {
        'name': 'resnet18_improved',
        'class_name': 'ResNet18Improved', 
        'module_path': 'models.resnet_improved',
        'params_millions': 11.26,
        'parameters': 11260000,  # For compatibility
        'input_size': 70,
        'num_classes': 2,
        'description': 'ResNet-18 improved version with enhanced architecture',
        'color': 'red',  # For visualization
        'experiment_pattern': 'experiment_20250802_164948'  # Current experiment pattern
    },
    'convnext_tiny': {
        'name': 'convnext_tiny',
        'class_name': 'ConvNextTiny',
        'module_path': 'models.convnext_tiny',
        'params_millions': 28.6,
        'parameters': 28600000,  # For compatibility
        'input_size': 70,
        'num_classes': 2,
        'description': 'ConvNext-Tiny model for efficient colony detection with modern convolution design',
        'color': 'green',  # For visualization
        'architecture': 'convnext',
        'created_date': '2025-08-02T20:13:49.363452',
        'base_config': 'efficientnet_b0'
    },
    'coatnet': {
        'name': 'coatnet',
        'class_name': 'CoAtNet',
        'module_path': 'models.coatnet',
        'params_millions': 9.07,
        'parameters': 9070000,  # Updated based on actual test
        'input_size': 70,
        'num_classes': 2,
        'description': 'CoAtNet model combining convolution and attention for efficient colony detection',
        'color': 'purple',  # For visualization
        'architecture': 'coatnet',
        'created_date': '2025-08-02T23:45:00',
        'base_config': 'efficientnet_b0'
    },
    'vit_tiny': {
        'name': 'vit_tiny',
        'class_name': 'VisionTransformerTiny',
        'module_path': 'models.vit_tiny',
        'params_millions': 2.72,
        'parameters': 2720000,  # Based on actual test: 2,717,954
        'input_size': 70,
        'num_classes': 2,
        'description': 'Vision Transformer Tiny model with pure attention mechanism for colony detection',
        'color': 'orange',  # For visualization
        'architecture': 'transformer',
        'created_date': '2025-08-03T01:58:00',
        'base_config': 'efficientnet_b0'
    },
    'mic_mobilenetv3': {
        'name': 'mic_mobilenetv3',
        'class_name': 'MIC_MobileNetV3',
        'module_path': 'models.mic_mobilenetv3',
        'params_millions': 2.5,
        'parameters': 2500000,  # Estimated
        'input_size': 70,
        'num_classes': 2,
        'description': 'MIC-specific MobileNetV3 with air bubble detection and turbidity analysis',
        'color': 'teal',  # For visualization
        'architecture': 'mobilenetv3_mic',
        'created_date': '2025-08-04T11:00:00',
        'base_config': 'efficientnet_b0',
        'features': ['bubble_detection', 'turbidity_analysis', 'multi_task']
    },
    'micro_vit': {
        'name': 'micro_vit',
        'class_name': 'MicroViT',
        'module_path': 'models.micro_vit',
        'params_millions': 1.8,
        'parameters': 1800000,  # Estimated
        'input_size': 70,
        'num_classes': 2,
        'description': 'Micro Vision Transformer optimized for 70x70 MIC testing images',
        'color': 'pink',  # For visualization
        'architecture': 'vision_transformer_micro',
        'created_date': '2025-08-04T11:00:00',
        'base_config': 'vit_tiny',
        'features': ['bubble_detection', 'turbidity_analysis', 'quality_assessment']
    },
    'airbubble_hybrid_net': {
        'name': 'airbubble_hybrid_net',
        'class_name': 'AirBubbleHybridNet',
        'module_path': 'models.airbubble_hybrid_net',
        'params_millions': 3.2,
        'parameters': 3200000,  # Estimated
        'input_size': 70,
        'num_classes': 4,  # Including air bubble interference class
        'description': 'Hybrid CNN-Transformer network with specialized air bubble detection and optical distortion correction',
        'color': 'brown',  # For visualization
        'architecture': 'cnn_transformer_hybrid',
        'created_date': '2025-08-04T11:00:00',
        'base_config': 'coatnet',
        'features': ['air_bubble_detection', 'optical_distortion_correction', 'turbidity_analysis', 'quality_assessment']
    },
    'efficientnet_v2_s': {
        'name': 'efficientnet_v2_s',
        'class_name': 'EfficientNetV2S',
        'module_path': 'models.efficientnet_v2_wrapper',
        'params_millions': 20.83,
        'parameters': 20834386,  # Actual parameter count
        'input_size': 70,
        'num_classes': 2,
        'description': 'EfficientNet V2-S with improved training efficiency and smaller model size',
        'color': 'lightblue',  # For visualization
        'architecture': 'efficientnet_v2',
        'created_date': '2025-01-02T15:30:00',
        'base_config': 'efficientnet_b0',
        'features': ['fused_mbconv', 'progressive_resizing', 'improved_training'],
        'experiment_pattern': 'experiment_20250804_123239'  # Current experiment pattern
    },
    'ghostnet': {
        'name': 'ghostnet',
        'class_name': 'GhostNetWrapper',
        'module_path': 'models.ghostnet_wrapper',
        'params_millions': 2.92,
        'parameters': 2916226,  # Actual parameter count
        'input_size': 70,
        'num_classes': 2,
        'description': 'GhostNet with efficient Ghost modules for lightweight feature extraction',
        'color': 'darkgreen',  # For visualization
        'architecture': 'ghostnet',
        'created_date': '2025-01-02T16:00:00',
        'base_config': 'efficientnet_b0',
        'features': ['ghost_modules', 'cheap_operations', 'lightweight_design'],
        'experiment_pattern': 'experiment_20250804_130938'  # Current experiment pattern
    },
    'densenet121': {
        'name': 'densenet121',
        'class_name': 'DenseNet121',
        'module_path': 'models.densenet_wrapper',
        'params_millions': 7.54,
        'parameters': 7539970,  # Actual parameter count
        'input_size': 70,
        'num_classes': 2,
        'description': 'DenseNet-121 with dense connections for efficient feature reuse (synthetic data)',
        'color': 'purple',  # For visualization
        'architecture': 'densenet',
        'created_date': '2025-01-02T16:30:00',
        'base_config': 'efficientnet_b0',
        'features': ['dense_connections', 'feature_reuse', 'gradient_flow'],
        'experiment_pattern': 'experiment_20250804_153316'  # Synthetic data experiment
    },
    'densenet121_real': {
        'name': 'densenet121_real',
        'class_name': 'DenseNet121',
        'module_path': 'models.densenet_wrapper',
        'params_millions': 7.54,
        'parameters': 7539970,  # Actual parameter count
        'input_size': 70,
        'num_classes': 2,
        'description': 'DenseNet-121 trained on real data - BEST PERFORMANCE (98.88% accuracy)',
        'color': 'gold',  # For visualization - gold for champion
        'architecture': 'densenet',
        'created_date': '2025-01-04T18:41:00',
        'base_config': 'efficientnet_b0',
        'features': ['dense_connections', 'feature_reuse', 'gradient_flow', 'real_data_trained'],
        'experiment_pattern': 'experiment_20250804_184102',  # Real data experiment
        'data_type': 'real_data',
        'train_samples': 3714,
        'val_samples': 538,
        'test_samples': 1059,
        'best_val_accuracy': 98.88,
        'test_accuracy': 98.02,
        'onnx_path': 'onnx_models/densenet121_real.onnx'
    },
    'regnet_y400mf': {
        'name': 'regnet_y400mf',
        'class_name': 'RegNetY400MF',
        'module_path': 'models.regnet_wrapper',
        'params_millions': 4.3,
        'parameters': 4300000,  # Estimated for RegNet Y-400MF
        'input_size': 70,
        'num_classes': 2,
        'description': 'RegNet Y-400MF with Squeeze-and-Excitation modules for efficient feature extraction',
        'color': 'navy',  # For visualization
        'architecture': 'regnet',
        'created_date': '2025-01-04T21:30:00',
        'base_config': 'efficientnet_b0',
        'features': ['squeeze_excitation', 'group_convolution', 'bottleneck_design', 'design_space_optimization']
    }
}

def get_model_config(model_name):
    """
    Get configuration for a specific model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        dict: Model configuration
        
    Raises:
        KeyError: If model_name is not found
    """
    if model_name not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return MODEL_CONFIGS[model_name].copy()

def get_all_model_names():
    """
    Get list of all available model names.
    
    Returns:
        list: List of model names
    """
    return list(MODEL_CONFIGS.keys())

def add_model_config(model_name, config):
    """
    Add a new model configuration.
    
    Args:
        model_name (str): Name of the new model
        config (dict): Model configuration dictionary
    """
    required_keys = ['class_name', 'module_path', 'params_millions', 'input_size', 'num_classes']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    MODEL_CONFIGS[model_name] = config

def get_model_comparison_data():
    """
    Get data formatted for model comparison.
    
    Returns:
        dict: Comparison data with model names as keys
    """
    comparison_data = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        comparison_data[model_name] = {
            'name': model_name,
            'params': config['params_millions'],
            'description': config['description'],
            'color': config.get('color', 'gray')
        }
    
    return comparison_data