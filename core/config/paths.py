"""
Path configuration and management for the bioast model system.
"""

import os
from datetime import datetime
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = BASE_DIR / 'bioast_dataset'
EXPERIMENTS_DIR = BASE_DIR / 'experiments'
REPORTS_DIR = BASE_DIR / 'reports'
MODELS_DIR = BASE_DIR / 'models'
TRAINING_DIR = BASE_DIR / 'training'
SCRIPTS_DIR = BASE_DIR / 'scripts'
TEMPLATES_DIR = BASE_DIR / 'templates'
DOCS_DIR = BASE_DIR / 'docs'
LEGACY_DIR = BASE_DIR / 'legacy'
VISUALIZATIONS_DIR = BASE_DIR / 'visualizations'

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        REPORTS_DIR,
        REPORTS_DIR / 'individual',
        REPORTS_DIR / 'comparisons', 
        REPORTS_DIR / 'summaries',
        SCRIPTS_DIR,
        TEMPLATES_DIR,
        TEMPLATES_DIR / 'report_templates',
        DOCS_DIR,
        LEGACY_DIR,
        VISUALIZATIONS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_experiment_path(model_name, timestamp=None):
    """
    Get experiment path for a model.
    
    Args:
        model_name (str): Name of the model
        timestamp (str, optional): Custom timestamp. If None, uses current time.
        
    Returns:
        Path: Path to the experiment directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    experiment_name = f'experiment_{timestamp}'
    return EXPERIMENTS_DIR / experiment_name / model_name

def get_latest_experiment_path(model_name):
    """
    Get the latest experiment path for a model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Path: Path to the latest experiment directory, or None if not found
    """
    if not EXPERIMENTS_DIR.exists():
        return None
    
    # Find all experiment directories for this model
    experiment_paths = []
    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith('experiment_'):
            model_path = exp_dir / model_name
            if model_path.exists():
                experiment_paths.append(model_path)
    
    if not experiment_paths:
        return None
    
    # Return the most recent one (sorted by directory name which includes timestamp)
    return sorted(experiment_paths, key=lambda x: x.parent.name)[-1]

def get_model_report_path(model_name, report_type='individual'):
    """
    Get report path for a model.
    
    Args:
        model_name (str): Name of the model
        report_type (str): Type of report ('individual', 'comparison', 'summary')
        
    Returns:
        Path: Path to the report directory
    """
    return REPORTS_DIR / report_type / model_name

def get_comparison_report_path(model_names):
    """
    Get comparison report path for multiple models.
    
    Args:
        model_names (list): List of model names
        
    Returns:
        Path: Path to the comparison report directory
    """
    comparison_name = '_vs_'.join(sorted(model_names))
    return REPORTS_DIR / 'comparisons' / comparison_name

def get_template_path(template_name):
    """
    Get template file path.
    
    Args:
        template_name (str): Name of the template
        
    Returns:
        Path: Path to the template file
    """
    return TEMPLATES_DIR / template_name

def get_legacy_path(filename):
    """
    Get legacy file path.
    
    Args:
        filename (str): Name of the legacy file
        
    Returns:
        Path: Path to the legacy file
    """
    return LEGACY_DIR / filename

# Standard experiment directory structure
EXPERIMENT_STRUCTURE = {
    'config.json': 'Model and training configuration',
    'best_model.pth': 'Best model checkpoint',
    'training_history.json': 'Training history and metrics',
    'test_results.json': 'Test evaluation results',
    'evaluation/': {
        'classification_report.txt': 'Detailed classification metrics',
        'evaluation_results.png': 'Evaluation visualization charts'
    },
    'sample_analysis/': {
        'confidence_analysis.png': 'Confidence distribution analysis',
        'correct_high_conf_samples.png': 'High confidence correct samples',
        'correct_medium_conf_samples.png': 'Medium confidence correct samples', 
        'correct_low_conf_samples.png': 'Low confidence correct samples',
        'incorrect_high_conf_samples.png': 'High confidence incorrect samples',
        'incorrect_medium_conf_samples.png': 'Medium confidence incorrect samples',
        'incorrect_low_conf_samples.png': 'Low confidence incorrect samples'
    },
    'visualizations/': {
        'training_history.png': 'Training history charts',
        'feature_maps.png': 'Feature map visualizations',
        'predictions.png': 'Prediction examples',
        'performance_summary.png': 'Performance summary charts'
    }
}

def create_experiment_structure(experiment_path):
    """
    Create standard experiment directory structure.
    
    Args:
        experiment_path (Path): Path to the experiment directory
    """
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_path / 'evaluation').mkdir(exist_ok=True)
    (experiment_path / 'sample_analysis').mkdir(exist_ok=True)
    (experiment_path / 'visualizations').mkdir(exist_ok=True)

def validate_experiment_structure(experiment_path):
    """
    Validate that an experiment directory has the expected structure.
    
    Args:
        experiment_path (Path): Path to the experiment directory
        
    Returns:
        dict: Validation results with missing files/directories
    """
    missing = []
    
    def check_structure(structure, base_path):
        for item, description in structure.items():
            item_path = base_path / item
            if isinstance(description, dict):
                if not item_path.exists():
                    missing.append(str(item_path))
                else:
                    check_structure(description, item_path)
            else:
                if not item_path.exists():
                    missing.append(str(item_path))
    
    check_structure(EXPERIMENT_STRUCTURE, experiment_path)
    
    return {
        'valid': len(missing) == 0,
        'missing': missing
    }

# Initialize directories on import
ensure_directories()