"""
Abstract interfaces for FUA core components
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from .data_structures import ModelCapabilities, ModelMetadata, Error, Improvement


class ModelInterface(ABC):
    """Abstract interface for all models in FUA"""
    
    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Get model capabilities declaration"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        pass
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure model with given parameters"""
        pass
    
    @abstractmethod
    def train(self, data: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train model with given data"""
        pass
    
    @abstractmethod
    def evaluate(self, data: Any, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> bool:
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """Load model from disk"""
        pass


class ConfigManager(ABC):
    """Abstract interface for configuration management"""
    
    @abstractmethod
    def load_config(self, config_path: str, config_type: str) -> Dict[str, Any]:
        """Load configuration from file"""
        pass
    
    @abstractmethod
    def save_config(self, config: Dict[str, Any], config_path: str, config_type: str) -> bool:
        """Save configuration to file"""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any], config_type: str) -> Tuple[bool, List[str]]:
        """Validate configuration against schema"""
        pass
    
    @abstractmethod
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations"""
        pass
    
    @abstractmethod
    def get_config_schema(self, config_type: str) -> Dict[str, Any]:
        """Get configuration schema for validation"""
        pass


class DataProcessor(ABC):
    """Abstract interface for data processing"""
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported data formats"""
        pass
    
    @abstractmethod
    def preprocess(self, data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Preprocess data for model input"""
        pass
    
    @abstractmethod
    def augment(self, data: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Apply data augmentation"""
        pass
    
    @abstractmethod
    def batch_process(self, data_batch: List[Any], config: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Process batch of data"""
        pass
    
    @abstractmethod
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        pass


class FineTuner(ABC):
    """Abstract interface for model fine-tuning"""
    
    @abstractmethod
    def get_supported_strategies(self) -> List[str]:
        """Get supported fine-tuning strategies"""
        pass
    
    @abstractmethod
    def setup_fine_tuning(self, model: Any, config: Dict[str, Any]) -> bool:
        """Setup fine-tuning configuration"""
        pass
    
    @abstractmethod
    def apply_layerwise_learning_rates(self, model: Any, learning_rates: Dict[str, float]) -> bool:
        """Apply layer-wise learning rates"""
        pass
    
    @abstractmethod
    def apply_adaptive_learning_rates(self, model: Any, config: Dict[str, Any]) -> bool:
        """Apply adaptive learning rates"""
        pass
    
    @abstractmethod
    def monitor_fine_tuning(self, model: Any, metrics: Dict[str, float]) -> List[Improvement]:
        """Monitor fine-tuning and suggest improvements"""
        pass
    
    @abstractmethod
    def get_fine_tuning_config(self, model: Any) -> Dict[str, Any]:
        """Get current fine-tuning configuration"""
        pass


class AutomationEngine(ABC):
    """Abstract interface for automation and improvement"""
    
    @abstractmethod
    def detect_errors(self, model: Any, training_data: Dict[str, Any]) -> List[Error]:
        """Detect training errors and issues"""
        pass
    
    @abstractmethod
    def suggest_improvements(self, errors: List[Error]) -> List[Improvement]:
        """Suggest improvements based on errors"""
        pass
    
    @abstractmethod
    def apply_improvement(self, improvement: Improvement, model: Any) -> bool:
        """Apply improvement to model"""
        pass
    
    @abstractmethod
    def validate_improvement(self, improvement: Improvement, model: Any, validation_data: Any) -> bool:
        """Validate improvement effectiveness"""
        pass
    
    @abstractmethod
    def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation statistics"""
        pass


class ConfigValidator(ABC):
    """Abstract interface for configuration validation"""
    
    @abstractmethod
    def validate_model_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate model configuration"""
        pass
    
    @abstractmethod
    def validate_training_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate training configuration"""
        pass
    
    @abstractmethod
    def validate_data_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data configuration"""
        pass
    
    @abstractmethod
    def get_validation_rules(self, config_type: str) -> Dict[str, Any]:
        """Get validation rules for configuration type"""
        pass


class ConfigurationManager(ConfigManager):
    """Concrete implementation of configuration management"""
    
    def __init__(self):
        self.config_cache = {}
        self.schema_cache = {}
    
    def load_config(self, config_path: str, config_type: str) -> Dict[str, Any]:
        """Load configuration from file"""
        import json
        import os
        
        cache_key = f"{config_path}:{config_type}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.config_cache[cache_key] = config
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: str, config_type: str) -> bool:
        """Save configuration to file"""
        import json
        import os
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            cache_key = f"{config_path}:{config_type}"
            self.config_cache[cache_key] = config
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any], config_type: str) -> Tuple[bool, List[str]]:
        """Validate configuration against schema"""
        validator = BaseConfigValidator()
        if config_type == 'model':
            return validator.validate_model_config(config)
        elif config_type == 'training':
            return validator.validate_training_config(config)
        elif config_type == 'data':
            return validator.validate_data_config(config)
        else:
            return False, [f"Unknown config type: {config_type}"]
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations with deep merge"""
        def deep_merge(base_dict, update_dict):
            result = base_dict.copy()
            for key, value in update_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        merged = {}
        for config in configs:
            merged = deep_merge(merged, config)
        
        return merged
    
    def get_config_schema(self, config_type: str) -> Dict[str, Any]:
        """Get configuration schema for validation"""
        validator = BaseConfigValidator()
        return validator.get_validation_rules(config_type)


class BaseConfigValidator(ConfigValidator):
    """Base implementation of configuration validation"""
    
    def __init__(self):
        self.validation_rules = {
            'model': {
                'required_fields': ['name', 'architecture_type', 'input_size'],
                'optional_fields': ['parameters', 'layers', 'attention_config'],
                'type_checks': {
                    'name': str,
                    'architecture_type': str,
                    'input_size': (list, tuple),
                    'parameters': (int, float),
                    'layers': int
                }
            },
            'training': {
                'required_fields': ['epochs', 'batch_size', 'learning_rate'],
                'optional_fields': ['optimizer', 'scheduler', 'early_stopping'],
                'type_checks': {
                    'epochs': int,
                    'batch_size': int,
                    'learning_rate': (int, float),
                    'optimizer': str
                }
            },
            'data': {
                'required_fields': ['data_path', 'input_size', 'batch_size'],
                'optional_fields': ['augmentation', 'normalization', 'shuffle'],
                'type_checks': {
                    'data_path': str,
                    'input_size': (list, tuple),
                    'batch_size': int,
                    'augmentation': bool
                }
            }
        }
    
    def validate_model_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate model configuration"""
        return self._validate_config(config, 'model')
    
    def validate_training_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate training configuration"""
        return self._validate_config(config, 'training')
    
    def validate_data_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data configuration"""
        return self._validate_config(config, 'data')
    
    def _validate_config(self, config: Dict[str, Any], config_type: str) -> Tuple[bool, List[str]]:
        """Generic configuration validation"""
        errors = []
        
        if config_type not in self.validation_rules:
            return False, [f"Unknown config type: {config_type}"]
        
        rules = self.validation_rules[config_type]
        
        # Check required fields
        for field in rules['required_fields']:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in rules['type_checks'].items():
            if field in config:
                if isinstance(expected_type, tuple):
                    if not isinstance(config[field], expected_type):
                        errors.append(f"Field {field} must be one of types: {expected_type}")
                else:
                    if not isinstance(config[field], expected_type):
                        errors.append(f"Field {field} must be of type: {expected_type}")
        
        return len(errors) == 0, errors
    
    def get_validation_rules(self, config_type: str) -> Dict[str, Any]:
        """Get validation rules for configuration type"""
        return self.validation_rules.get(config_type, {})