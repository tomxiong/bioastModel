"""
FUA Model-Specific Configuration System

This module provides a comprehensive configuration system for model-specific settings,
including validation, inheritance, templates, and automatic configuration generation.
"""

import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from .data_structures import ModelCapabilities, ModelMetadata, Error, Improvement
from .interfaces import ConfigManager
from .interfaces import ConfigurationManager


class ModelConfigurationSystem:
    """Advanced configuration system for model-specific settings"""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
        self.model_configs = {}  # model_type -> config_schema
        self.config_templates = {}  # template_name -> template_config
        self.config_inheritance = {}  # child_type -> parent_type
        
        # Initialize with default configurations
        self._initialize_default_configs()
    
    def register_model_config(self, model_type: str, config_schema: Dict[str, Any]) -> None:
        """Register a configuration schema for a specific model type"""
        self.model_configs[model_type] = config_schema.copy()
        
        # Handle inheritance
        if 'parent_config' in config_schema:
            parent_type = config_schema['parent_config']
            self.config_inheritance[model_type] = parent_type
    
    def register_config_template(self, template_name: str, template_config: Dict[str, Any]) -> None:
        """Register a configuration template"""
        self.config_templates[template_name] = template_config.copy()
    
    def validate_model_config(self, model_type: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a model-specific configuration"""
        if model_type not in self.model_configs:
            return False, [f"Unknown model type: {model_type}"]
        
        schema = self.model_configs[model_type]
        errors = []
        
        # Apply inheritance - get complete schema with parent fields
        complete_schema = self._get_complete_schema(model_type)
        
        # Check required fields
        for field in complete_schema.get('required_fields', []):
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in complete_schema.get('type_checks', {}).items():
            if field in config:
                if isinstance(expected_type, tuple):
                    if not isinstance(config[field], expected_type):
                        errors.append(f"Field {field} must be one of types: {expected_type}")
                else:
                    if not isinstance(config[field], expected_type):
                        errors.append(f"Field {field} must be of type: {expected_type}")
        
        # Check value ranges
        for field, value_range in complete_schema.get('value_ranges', {}).items():
            if field in config:
                min_val, max_val = value_range
                try:
                    field_value = float(config[field])
                    if not (min_val <= field_value <= max_val):
                        errors.append(f"Field {field} must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"Field {field} must be a numeric value")
        
        return len(errors) == 0, errors
    
    def apply_defaults(self, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to a configuration"""
        complete_schema = self._get_complete_schema(model_type)
        result = config.copy()
        
        # Apply defaults from complete schema (including inherited)
        for field, default_value in complete_schema.get('default_values', {}).items():
            if field not in result:
                result[field] = default_value
        
        return result
    
    def merge_model_configs(self, model_type: str, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configurations for a model type"""
        # Start with defaults
        merged_config = self.apply_defaults(model_type, {})
        
        # Deep merge all configurations
        for config in configs:
            merged_config = self._deep_merge(merged_config, config)
        
        return merged_config
    
    def get_model_config_schema(self, model_type: str) -> Dict[str, Any]:
        """Get the complete configuration schema for a model type"""
        return self._get_complete_schema(model_type)
    
    def get_registered_configs(self) -> List[str]:
        """Get list of registered model configuration types"""
        return list(self.model_configs.keys())
    
    def get_available_templates(self) -> List[str]:
        """Get list of available configuration templates"""
        return list(self.config_templates.keys())
    
    def apply_template(self, template_name: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a configuration template to base configuration"""
        if template_name not in self.config_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.config_templates[template_name]
        return self._deep_merge(template.copy(), base_config)
    
    def generate_config_from_capabilities(self, model_name: str, capabilities: ModelCapabilities) -> Dict[str, Any]:
        """Generate configuration based on model capabilities"""
        
        # Determine model type from capabilities
        model_type = self._infer_model_type(capabilities)
        
        # Get base configuration
        base_config = self.apply_defaults(model_type, {'name': model_name})
        
        # Generate training configuration based on capabilities
        training_config = self._generate_training_config(capabilities)
        
        # Generate model-specific configuration
        model_config = self._generate_model_config(capabilities, base_config)
        
        return {
            'name': model_name,
            'model_type': model_type,
            'model': model_config,
            'training': training_config,
            'data': self._generate_data_config(capabilities),
            'capabilities': capabilities.to_dict()
        }
    
    def save_configurations(self, file_path: str) -> bool:
        """Save all model configurations to file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            data = {
                'model_configs': self.model_configs,
                'config_templates': self.config_templates,
                'config_inheritance': self.config_inheritance,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving configurations: {e}")
            return False
    
    def load_configurations(self, file_path: str) -> bool:
        """Load model configurations from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.model_configs = data.get('model_configs', {})
            self.config_templates = data.get('config_templates', {})
            self.config_inheritance = data.get('config_inheritance', {})
            
            return True
        except Exception as e:
            print(f"Error loading configurations: {e}")
            return False
    
    def save_templates(self, file_path: str) -> bool:
        """Save configuration templates to file"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self.config_templates, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving templates: {e}")
            return False
    
    def load_templates(self, file_path: str) -> bool:
        """Load configuration templates from file"""
        try:
            with open(file_path, 'r') as f:
                self.config_templates = json.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading templates: {e}")
            return False
    
    def _get_complete_schema(self, model_type: str) -> Dict[str, Any]:
        """Get complete schema including inherited fields"""
        if model_type not in self.model_configs:
            return {}
        
        schema = self.model_configs[model_type].copy()
        
        # Handle inheritance
        if model_type in self.config_inheritance:
            parent_type = self.config_inheritance[model_type]
            parent_schema = self._get_complete_schema(parent_type)
            
            # Merge parent schema with child schema
            complete_schema = parent_schema.copy()
            
            # Merge required fields
            if 'required_fields' in schema:
                complete_schema['required_fields'] = list(set(
                    complete_schema.get('required_fields', []) + 
                    schema['required_fields']
                ))
            
            # Merge optional fields
            if 'optional_fields' in schema:
                complete_schema['optional_fields'] = list(set(
                    complete_schema.get('optional_fields', []) + 
                    schema['optional_fields']
                ))
            
            # Merge type checks
            if 'type_checks' in schema:
                complete_schema['type_checks'] = {
                    **complete_schema.get('type_checks', {}),
                    **schema['type_checks']
                }
            
            # Merge value ranges
            if 'value_ranges' in schema:
                complete_schema['value_ranges'] = {
                    **complete_schema.get('value_ranges', {}),
                    **schema['value_ranges']
                }
            
            # Merge default values (child overrides parent)
            if 'default_values' in schema:
                complete_schema['default_values'] = {
                    **complete_schema.get('default_values', {}),
                    **schema['default_values']
                }
            
            return complete_schema
        
        return schema
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base_dict.copy()
        
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _infer_model_type(self, capabilities: ModelCapabilities) -> str:
        """Infer model type from capabilities"""
        arch_type = capabilities.computational_complexity
        special_preprocessing = capabilities.special_preprocessing
        
        # Simple inference logic
        if 'bubble_detection' in special_preprocessing:
            return 'airbubble_hybrid_net'
        elif 'transformer' in str(special_preprocessing).lower() or 'attention' in str(special_preprocessing).lower():
            return 'transformer'
        elif 'efficient' in str(arch_type).lower():
            return 'efficientnet'
        elif 'mobile' in str(arch_type).lower():
            return 'mobilenet'
        else:
            return 'cnn'
    
    def _generate_training_config(self, capabilities: ModelCapabilities) -> Dict[str, Any]:
        """Generate training configuration based on capabilities"""
        # Select batch size from recommended range
        min_batch, max_batch = capabilities.recommended_batch_size
        batch_size = min(max_batch, max(min_batch, 32))  # Prefer 32 if in range
        
        # Select optimizer and scheduler from supported options
        optimizer = capabilities.supported_optimizers[0] if capabilities.supported_optimizers else 'adam'
        scheduler = capabilities.supported_schedulers[0] if capabilities.supported_schedulers else 'step'
        
        # Adjust learning rate based on complexity
        base_lr = 0.001
        if capabilities.computational_complexity == 'low':
            base_lr = 0.01
        elif capabilities.computational_complexity == 'high':
            base_lr = 0.0001
        
        return {
            'batch_size': batch_size,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'learning_rate': base_lr,
            'epochs': 50 if capabilities.training_time_estimate == 'fast' else 100,
            'weight_decay': 0.01,
            'early_stopping': {
                'patience': 10,
                'min_delta': 0.001
            }
        }
    
    def _generate_model_config(self, capabilities: ModelCapabilities, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model-specific configuration"""
        min_size, max_size = capabilities.input_size_range
        
        # Use middle of input size range
        input_size = [70, 70]  # Default for this project
        
        return {
            'input_size': input_size,
            'num_classes': 2,  # Binary classification for this project
            'preprocessing': capabilities.special_preprocessing,
            'architecture_params': base_config
        }
    
    def _generate_data_config(self, capabilities: ModelCapabilities) -> Dict[str, Any]:
        """Generate data configuration based on capabilities"""
        min_batch, max_batch = capabilities.recommended_batch_size
        batch_size = min(max_batch, max(min_batch, 32))
        
        return {
            'batch_size': batch_size,
            'input_size': [70, 70],  # Standard for this project
            'preprocessing': capabilities.special_preprocessing,
            'augmentation': True,
            'normalization': True,
            'shuffle': True
        }
    
    def _initialize_default_configs(self):
        """Initialize default model configurations"""
        
        # Default CNN configuration
        cnn_config = {
            'required_fields': ['name', 'architecture_type', 'input_size'],
            'optional_fields': ['layers', 'filters', 'kernel_size', 'activation'],
            'type_checks': {
                'name': str,
                'architecture_type': str,
                'input_size': (list, tuple),
                'layers': int,
                'filters': int,
                'kernel_size': int,
                'activation': str
            },
            'value_ranges': {
                'layers': [1, 100],
                'filters': [1, 1024],
                'kernel_size': [1, 7]
            },
            'default_values': {
                'layers': 10,
                'filters': 32,
                'kernel_size': 3,
                'activation': 'relu'
            }
        }
        
        # Default Transformer configuration
        transformer_config = {
            'required_fields': ['name', 'architecture_type', 'input_size', 'num_heads'],
            'optional_fields': ['num_layers', 'hidden_size', 'dropout', 'num_classes'],
            'type_checks': {
                'name': str,
                'architecture_type': str,
                'input_size': (list, tuple),
                'num_heads': int,
                'num_layers': int,
                'hidden_size': int,
                'dropout': (int, float),
                'num_classes': int
            },
            'value_ranges': {
                'num_heads': [1, 16],
                'num_layers': [1, 24],
                'hidden_size': [128, 4096],
                'dropout': [0.0, 0.5],
                'num_classes': [2, 1000]
            },
            'default_values': {
                'num_layers': 12,
                'hidden_size': 768,
                'dropout': 0.1,
                'num_classes': 2
            }
        }
        
        # Default EfficientNet configuration
        efficientnet_config = {
            'required_fields': ['name', 'architecture_type', 'input_size', 'width_coefficient'],
            'optional_fields': ['depth_coefficient', 'dropout_rate', 'stem_type'],
            'type_checks': {
                'name': str,
                'architecture_type': str,
                'input_size': (list, tuple),
                'width_coefficient': (int, float),
                'depth_coefficient': (int, float),
                'dropout_rate': (int, float),
                'stem_type': str
            },
            'value_ranges': {
                'width_coefficient': [0.5, 2.0],
                'depth_coefficient': [0.5, 2.0],
                'dropout_rate': [0.0, 0.5]
            },
            'default_values': {
                'depth_coefficient': 1.0,
                'dropout_rate': 0.2,
                'stem_type': 'same'
            },
            'parent_config': 'cnn'  # Inherits from CNN
        }
        
        # Register default configurations
        self.register_model_config('cnn', cnn_config)
        self.register_model_config('transformer', transformer_config)
        self.register_model_config('efficientnet', efficientnet_config)
        
        # Register default templates
        self.register_config_template('lightweight', {
            'architecture_type': 'cnn',
            'layers': 5,
            'filters': 16,
            'batch_size': 64,
            'learning_rate': 0.01
        })
        
        self.register_config_template('balanced', {
            'architecture_type': 'cnn',
            'layers': 15,
            'filters': 32,
            'batch_size': 32,
            'learning_rate': 0.001
        })
        
        self.register_config_template('high_performance', {
            'architecture_type': 'cnn',
            'layers': 25,
            'filters': 64,
            'batch_size': 16,
            'learning_rate': 0.0001
        })


class ModelConfigurationManager:
    """High-level manager for model configuration operations"""
    
    def __init__(self, config_system: Optional[ModelConfigurationSystem] = None):
        self.config_system = config_system or ModelConfigurationSystem()
        self.active_configs = {}
    
    def create_model_config(self, model_name: str, model_type: str, 
                          base_config: Optional[Dict[str, Any]] = None,
                          template_name: Optional[str] = None) -> str:
        """Create a model configuration"""
        config_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Ensure model type is registered in the config system
        if model_type not in self.config_system.model_configs:
            # Auto-register with default schema
            default_schema = {
                'required_fields': ['name', 'model_type'],
                'optional_fields': ['parameters', 'layers', 'attention_heads', 'fusion_layers'],
                'type_checks': {
                    'name': str,
                    'model_type': str,
                    'parameters': (int, float),
                    'layers': int,
                    'attention_heads': int,
                    'fusion_layers': int
                }
            }
            self.config_system.register_model_config(model_type, default_schema)
        
        # Start with base config or empty
        config = base_config or {}
        
        # Apply template if specified
        if template_name:
            config = self.config_system.apply_template(template_name, config)
        
        # Apply defaults for model type
        config = self.config_system.apply_defaults(model_type, config)
        
        # Ensure model name is set
        config['name'] = model_name
        config['model_type'] = model_type
        
        # Store active configuration
        self.active_configs[config_id] = {
            'config': config,
            'model_type': model_type,
            'created_at': datetime.now(),
            'last_modified': datetime.now()
        }
        
        return config_id
    
    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get a configuration by ID"""
        if config_id in self.active_configs:
            return self.active_configs[config_id]['config']
        return None
    
    def update_config(self, config_id: str, updates: Dict[str, Any]) -> bool:
        """Update a configuration"""
        if config_id not in self.active_configs:
            return False
        
        config_info = self.active_configs[config_id]
        model_type = config_info['model_type']
        
        # Merge updates
        updated_config = self.config_system.merge_model_configs(model_type, config_info['config'], updates)
        
        # Validate updated configuration
        is_valid, errors = self.config_system.validate_model_config(model_type, updated_config)
        
        if is_valid:
            config_info['config'] = updated_config
            config_info['last_modified'] = datetime.now()
            return True
        else:
            print(f"Configuration validation failed: {errors}")
            return False
    
    def validate_config(self, config_id: str) -> Tuple[bool, List[str]]:
        """Validate a configuration"""
        config_info = self.active_configs.get(config_id)
        if not config_info:
            return False, ["Configuration not found"]
        
        return self.config_system.validate_model_config(config_info['model_type'], config_info['config'])
    
    def save_config(self, config_id: str, file_path: str) -> bool:
        """Save a configuration to file"""
        config = self.get_config(config_id)
        if not config:
            return False
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def load_config(self, file_path: str, model_type: str) -> str:
        """Load a configuration from file"""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Validate loaded configuration
            is_valid, errors = self.config_system.validate_model_config(model_type, config)
            if not is_valid:
                raise ValueError(f"Invalid configuration: {errors}")
            
            # Create new config entry
            config_id = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.active_configs[config_id] = {
                'config': config,
                'model_type': model_type,
                'created_at': datetime.now(),
                'last_modified': datetime.now(),
                'loaded_from': file_path
            }
            
            return config_id
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """List all active configurations"""
        configs = []
        for config_id, info in self.active_configs.items():
            configs.append({
                'config_id': config_id,
                'model_name': info['config'].get('name', 'unknown'),
                'model_type': info['model_type'],
                'created_at': info['created_at'],
                'last_modified': info['last_modified']
            })
        return configs
    
    def apply_template(self, template_name: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a configuration template"""
        return self.config_system.apply_template(template_name, base_config)
    
    def validate_model_config(self, model_type: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a model configuration"""
        return self.config_system.validate_model_config(model_type, config)
    
    def generate_config_from_capabilities(self, model_name: str, 
                                       capabilities: ModelCapabilities) -> str:
        """Generate configuration from model capabilities"""
        config = self.config_system.generate_config_from_capabilities(model_name, capabilities)
        
        config_id = f"generated_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.active_configs[config_id] = {
            'config': config,
            'model_type': config.get('model_type', 'cnn'),
            'created_at': datetime.now(),
            'last_modified': datetime.now(),
            'generated_from': 'capabilities'
        }
        
        return config_id