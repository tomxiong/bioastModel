"""
Training configuration definitions for all supported models.
"""

TRAINING_CONFIGS = {
    'default': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 30,
        'optimizer': 'adam',
        'scheduler': 'step',
        'step_size': 10,
        'gamma': 0.1,
        'description': 'Default training configuration'
    },
    
    'efficientnet_optimized': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'description': 'Optimized configuration for EfficientNet training'
    },
    
    'resnet_optimized': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'description': 'Optimized configuration for ResNet training'
    },
    
    'convnext_optimized': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'description': 'Optimized configuration for ConvNext-Tiny training'
    },
    
    'coatnet_optimized': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'description': 'Optimized configuration for CoAtNet training'
    },
    
    'vit_optimized': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'dropout_rate': 0.1,
        'description': 'Optimized configuration for Vision Transformer training'
    },
    
    'quick_test': {
        'batch_size': 16,
        'learning_rate': 0.01,
        'weight_decay': 1e-4,
        'num_epochs': 5,
        'optimizer': 'adam',
        'scheduler': 'step',
        'step_size': 2,
        'gamma': 0.5,
        'description': 'Quick test configuration for debugging'
    }
}

# Model-specific configuration mapping
MODEL_CONFIG_MAPPING = {
    'efficientnet_b0': 'efficientnet_optimized',
    'resnet18_improved': 'resnet_optimized',
    'convnext_tiny': 'convnext_optimized',
    'coatnet': 'coatnet_optimized',
    'vit_tiny': 'vit_optimized'
}

def get_training_config(config_name):
    """
    Get training configuration by name.
    
    Args:
        config_name (str): Name of the training configuration
        
    Returns:
        dict: Training configuration
        
    Raises:
        KeyError: If config_name is not found
    """
    if config_name not in TRAINING_CONFIGS:
        available_configs = list(TRAINING_CONFIGS.keys())
        raise KeyError(f"Training config '{config_name}' not found. Available configs: {available_configs}")
    
    return TRAINING_CONFIGS[config_name].copy()

def get_model_specific_config(model_name):
    """
    Get the recommended training configuration for a specific model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        dict: Training configuration
        
    Raises:
        KeyError: If model_name is not found
    """
    if model_name not in MODEL_CONFIG_MAPPING:
        available_models = list(MODEL_CONFIG_MAPPING.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    config_name = MODEL_CONFIG_MAPPING[model_name]
    return get_training_config(config_name)

def get_all_config_names():
    """
    Get list of all available training configuration names.
    
    Returns:
        list: List of configuration names
    """
    return list(TRAINING_CONFIGS.keys())

def add_training_config(config_name, config):
    """
    Add a new training configuration.
    
    Args:
        config_name (str): Name of the new configuration
        config (dict): Training configuration dictionary
    """
    required_keys = ['batch_size', 'learning_rate', 'num_epochs', 'optimizer']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    TRAINING_CONFIGS[config_name] = config