"""
Configuration management module for bioast model system.
"""

from .model_configs import MODEL_CONFIGS, get_model_config, get_all_model_names, get_model_comparison_data
from .training_configs import TRAINING_CONFIGS, get_training_config
from .paths import *

# Add compatibility functions
def get_model_specific_config(model_name):
    """Get model-specific training configuration."""
    from .training_configs import get_model_specific_config as get_training_config_for_model
    return get_training_config_for_model(model_name)

def get_latest_experiment_path(model_name):
    """Get the latest experiment path for a model."""
    import os
    from pathlib import Path
    
    experiments_dir = EXPERIMENTS_DIR
    if not experiments_dir.exists():
        return None
    
    # Find the latest experiment directory containing the model
    latest_experiment = None
    latest_time = 0
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            model_dir = exp_dir / model_name
            if model_dir.exists():
                # Get modification time
                mod_time = os.path.getmtime(model_dir)
                if mod_time > latest_time:
                    latest_time = mod_time
                    latest_experiment = model_dir
    
    return latest_experiment

def get_model_report_path(model_name):
    """Get the report path for a model."""
    return REPORTS_DIR / 'individual' / model_name

__all__ = [
    'MODEL_CONFIGS',
    'get_model_config',
    'get_model_specific_config', 
    'get_all_model_names',
    'get_model_comparison_data',
    'TRAINING_CONFIGS',
    'get_training_config',
    'BASE_DIR',
    'DATA_DIR',
    'EXPERIMENTS_DIR',
    'REPORTS_DIR',
    'MODELS_DIR',
    'get_experiment_path',
    'get_latest_experiment_path',
    'get_model_report_path'
]
