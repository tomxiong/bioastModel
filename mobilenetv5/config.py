"""
MobileNetV5 Configuration
"""

import os
from pathlib import Path


class MobileNetV5Config:
    """Configuration for MobileNetV5 training and evaluation"""
    
    # Model configurations
    MODEL_CONFIGS = {
        'mobilenetv5': {
            'name': 'mobilenetv5',
            'params_millions': 2.8,
            'description': 'MobileNetV5 optimized for mobile devices with SE attention',
            'color': 'green'
        },
        'mobilenetv5_small': {
            'name': 'mobilenetv5_small',
            'params_millions': 1.6,
            'description': 'Smaller MobileNetV5 variant for faster inference',
            'color': 'lightgreen'
        }
    }
    
    # Training configurations
    TRAINING_CONFIGS = {
        'quick_test': {
            'batch_size': 16,
            'num_epochs': 5,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'early_stopping_patience': 3
        },
        'standard': {
            'batch_size': 32,
            'num_epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'early_stopping_patience': 10
        },
        'extended': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,
            'early_stopping_patience': 15
        }
    }
    
    # Data configuration
    DATA_CONFIG = {
        'image_size': 70,
        'num_classes': 2,
        'class_names': ['Negative', 'Positive'],
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    
    # Paths
    @staticmethod
    def get_project_root():
        """Get project root directory"""
        return Path(__file__).parent.parent
    
    @staticmethod
    def get_data_dir():
        """Get data directory"""
        return MobileNetV5Config.get_project_root() / 'bioast_dataset'
    
    @staticmethod
    def get_experiments_dir():
        """Get experiments directory"""
        return MobileNetV5Config.get_project_root() / 'experiments'
    
    @staticmethod
    def get_mobilenetv5_experiments_dir():
        """Get MobileNetV5 experiments directory"""
        return MobileNetV5Config.get_experiments_dir() / 'mobilenetv5'
    
    @staticmethod
    def ensure_dirs():
        """Ensure all necessary directories exist"""
        dirs_to_create = [
            MobileNetV5Config.get_mobilenetv5_experiments_dir(),
            MobileNetV5Config.get_mobilenetv5_experiments_dir() / 'models',
            MobileNetV5Config.get_mobilenetv5_experiments_dir() / 'results',
            MobileNetV5Config.get_mobilenetv5_experiments_dir() / 'logs'
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_model_config(model_name):
        """Get model configuration"""
        if model_name not in MobileNetV5Config.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        return MobileNetV5Config.MODEL_CONFIGS[model_name]
    
    @staticmethod
    def get_training_config(config_name):
        """Get training configuration"""
        if config_name not in MobileNetV5Config.TRAINING_CONFIGS:
            raise ValueError(f"Unknown training config: {config_name}")
        return MobileNetV5Config.TRAINING_CONFIGS[config_name]
    
    @staticmethod
    def setup_environment():
        """Setup environment and directories"""
        MobileNetV5Config.ensure_dirs()
        
        # Set environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = '42'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        print("MobileNetV5 environment setup complete")
        print(f"Data directory: {MobileNetV5Config.get_data_dir()}")
        print(f"Experiments directory: {MobileNetV5Config.get_mobilenetv5_experiments_dir()}")


if __name__ == "__main__":
    MobileNetV5Config.setup_environment()