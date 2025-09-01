"""
Core module for FUA
"""

from .data_structures import ModelCapabilities, ModelMetadata, Error, Improvement
from .interfaces import (
    ModelInterface, ConfigManager, DataProcessor, FineTuner, AutomationEngine,
    ConfigurationManager, BaseConfigValidator, ConfigValidator
)

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
    'ConfigValidator'
]