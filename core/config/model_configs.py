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