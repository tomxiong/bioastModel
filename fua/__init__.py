"""
FUA - Flexible Unified Architecture
A flexible and unified architecture for ML model training and optimization
"""

__version__ = "1.0.0"
__author__ = "FUA Development Team"

from .core.data_structures import ModelCapabilities, ModelMetadata, Error, Improvement
from .core.interfaces import (
    ModelInterface, ConfigManager, DataProcessor, FineTuner, AutomationEngine,
    ConfigurationManager, BaseConfigValidator, ConfigValidator
)
from .core.model_adapters import ModelAdapter, ModelFactory, ModelManager
from .core.model_config import ModelConfigurationSystem, ModelConfigurationManager

__all__ = [
    'ModelCapabilities',
    'ModelMetadata', 
    'Error',
    'Improvement',
    'ModelInterface',
    'ConfigManager',
    'DataProcessor',
    'FineTuner',
    'AutomationEngine',
    'ConfigurationManager',
    'BaseConfigValidator',
    'ConfigValidator',
    'ModelAdapter',
    'ModelFactory',
    'ModelManager',
    'ModelConfigurationSystem',
    'ModelConfigurationManager'
]