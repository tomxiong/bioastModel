"""
FUA Model Adapters and Factory Pattern Implementation

This module provides adapters for integrating existing models into the FUA architecture,
along with a factory pattern for creating and managing models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Callable, Type, Union
from datetime import datetime
import os
import json
import importlib

from .data_structures import ModelCapabilities, ModelMetadata, Error, Improvement
from .interfaces import ModelInterface, ConfigManager, ConfigurationManager


class ModelAdapter(ModelInterface):
    """Base adapter class for integrating existing models into FUA architecture"""
    
    def __init__(self, model_name: str, model_factory: Callable, capabilities: ModelCapabilities):
        self.model_name = model_name
        self.model_factory = model_factory
        self.capabilities = capabilities
        self.model = None
        self.config = {}
        self.training_history = []
    
    def get_capabilities(self) -> ModelCapabilities:
        """Get model capabilities"""
        return self.capabilities
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata"""
        if self.model is None:
            # Create model instance to get metadata
            filtered_config = self._filter_model_config(self.config.copy())
            self.model = self.model_factory(num_classes=2, **filtered_config)
        
        return ModelMetadata(
            name=self.model_name,
            version='1.0.0',
            architecture_type=self._get_architecture_type(),
            parameter_count=self._count_parameters(),
            computational_complexity=self._calculate_complexity(),
            memory_usage=self._estimate_memory_usage(),
            supported_input_sizes=self.capabilities.input_size_range,
            performance_metrics=self._get_performance_metrics(),
            training_history=self.training_history,
            creation_date=datetime.now(),
            last_modified=datetime.now(),
            author='FUA Adapter',
            tags=self._get_tags(),
            description=f'FUA adapter for {self.model_name}'
        )
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure model with given parameters"""
        try:
            self.config.update(config)
            
            # Extract num_classes from config or use default
            num_classes = self.config.pop('num_classes', 2)
            
            # Filter out unsupported parameters based on model type (including num_classes)
            filtered_config = self._filter_model_config(self.config.copy())
            
            # Recreate model with filtered configuration
            self.model = self.model_factory(num_classes=num_classes, **filtered_config)
            
            # Restore num_classes to config
            self.config['num_classes'] = num_classes
            
            return True
        except Exception as e:
            print(f"Error configuring model {self.model_name}: {e}")
            return False
    
    def _filter_model_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out unsupported parameters based on model type"""
        model_name_lower = self.model_name.lower()
        
        # Define supported parameters for each model type
        supported_params = {
            'airbubble_hybrid_net': [
                'dropout_rate', 'enable_distortion_correction', 'model_size'
            ],
            'mic_mobilenetv3': [
                'width_mult', 'dropout_rate', 'enable_bubble_detection', 
                'enable_turbidity_analysis', 'model_size'
            ],
            'micro_vit': [
                'embed_dim', 'depth', 'num_heads', 'mlp_ratio', 
                'dropout_rate', 'model_size'
            ]
        }
        
        # Get supported parameters for this model type
        params_to_keep = []
        for model_type, params in supported_params.items():
            if model_type in model_name_lower:
                params_to_keep = params
                break
        
        # If no specific parameters found, keep basic ones
        if not params_to_keep:
            params_to_keep = ['dropout_rate', 'model_size']
        
        # Filter config - note: num_classes is handled separately
        filtered_config = {}
        for key, value in config.items():
            if key in params_to_keep:
                filtered_config[key] = value
            elif key == 'num_classes':
                # Skip num_classes here as it's passed separately
                pass
            else:
                # Silently drop unsupported parameters
                pass
        
        return filtered_config
    
    def train(self, data: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train model with given data"""
        if self.model is None:
            filtered_config = self._filter_model_config(self.config.copy())
            self.model = self.model_factory(num_classes=2, **filtered_config)
        
        # Mock training - in real implementation this would interface with training pipeline
        training_config = config or {}
        epochs = training_config.get('epochs', 10)
        
        # Simulate training process
        mock_results = {
            'loss': 0.1 * (1 + 0.1 * epochs),  # Mock loss increase
            'accuracy': min(0.95, 0.8 + 0.015 * epochs),  # Mock accuracy improvement
            'training_time': epochs * 12.5,  # Mock time per epoch
            'epochs_completed': epochs,
            'model_size_mb': self._estimate_model_size(),
            'learning_rate': training_config.get('learning_rate', 0.001)
        }
        
        # Update training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'config': training_config,
            'results': mock_results
        })
        
        return mock_results
    
    def evaluate(self, data: Any, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate model performance"""
        if metrics is None:
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        # Mock evaluation results
        results = {}
        base_accuracy = 0.9  # Base accuracy for the model
        
        for metric in metrics:
            if metric == 'accuracy':
                results[metric] = min(0.99, base_accuracy + 0.05)
            elif metric == 'f1_score':
                results[metric] = min(0.98, base_accuracy + 0.03)
            elif metric == 'precision':
                results[metric] = min(0.97, base_accuracy + 0.04)
            elif metric == 'recall':
                results[metric] = min(0.96, base_accuracy + 0.02)
            elif metric == 'auc':
                results[metric] = min(0.99, base_accuracy + 0.06)
            else:
                results[metric] = 0.85  # Default for unknown metrics
        
        return results
    
    def save(self, path: str) -> bool:
        """Save model adapter state"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state if available
            model_path = path.replace('.json', '_model.pth')
            if self.model is not None:
                torch.save(self.model.state_dict(), model_path)
            
            # Save adapter metadata
            adapter_data = {
                'model_name': self.model_name,
                'config': self.config,
                'capabilities': self.capabilities.to_dict(),
                'training_history': self.training_history,
                'model_path': model_path if self.model is not None else None
            }
            
            with open(path, 'w') as f:
                json.dump(adapter_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving adapter {self.model_name}: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load model adapter state"""
        try:
            with open(path, 'r') as f:
                adapter_data = json.load(f)
            
            self.model_name = adapter_data['model_name']
            self.config = adapter_data['config']
            self.training_history = adapter_data.get('training_history', [])
            
            # Load model state if available
            if adapter_data.get('model_path') and os.path.exists(adapter_data['model_path']):
                filtered_config = self._filter_model_config(self.config.copy())
                # Extract num_classes from config if present, otherwise use default
                num_classes = filtered_config.pop('num_classes', 2)
                self.model = self.model_factory(num_classes=num_classes, **filtered_config)
                self.model.load_state_dict(torch.load(adapter_data['model_path']))
            
            return True
        except Exception as e:
            print(f"Error loading adapter {self.model_name}: {e}")
            return False
    
    def _get_architecture_type(self) -> str:
        """Determine architecture type from model name"""
        model_name_lower = self.model_name.lower()
        
        if 'cnn' in model_name_lower or 'conv' in model_name_lower:
            return 'cnn'
        elif 'transformer' in model_name_lower or 'vit' in model_name_lower:
            return 'transformer'
        elif 'hybrid' in model_name_lower:
            return 'cnn_transformer_hybrid'
        elif 'efficientnet' in model_name_lower:
            return 'efficientnet'
        elif 'resnet' in model_name_lower:
            return 'resnet'
        elif 'mobilenet' in model_name_lower:
            return 'mobilenet'
        else:
            return 'unknown'
    
    def _count_parameters(self) -> int:
        """Count model parameters"""
        if self.model is None:
            filtered_config = self._filter_model_config(self.config.copy())
            self.model = self.model_factory(num_classes=2, **filtered_config)
        
        return sum(p.numel() for p in self.model.parameters())
    
    def _calculate_complexity(self) -> float:
        """Calculate computational complexity score"""
        param_count = self._count_parameters()
        
        # Simple complexity calculation based on parameter count
        if param_count < 1000000:  # < 1M
            return 1.0
        elif param_count < 5000000:  # < 5M
            return 2.0
        elif param_count < 20000000:  # < 20M
            return 3.0
        else:
            return 4.0
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in MB"""
        param_count = self._count_parameters()
        
        # Rough estimate: 4 bytes per parameter + overhead
        return int((param_count * 4) / (1024 * 1024)) + 10
    
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Get default performance metrics"""
        return {
            'accuracy': 0.9,
            'f1_score': 0.85,
            'precision': 0.88,
            'recall': 0.82
        }
    
    def _get_tags(self) -> List[str]:
        """Get model tags"""
        tags = ['fua_adapter']
        
        model_name_lower = self.model_name.lower()
        if 'cnn' in model_name_lower:
            tags.append('cnn')
        if 'transformer' in model_name_lower:
            tags.append('transformer')
        if 'hybrid' in model_name_lower:
            tags.append('hybrid')
        if 'efficient' in model_name_lower:
            tags.append('efficient')
        if 'lightweight' in model_name_lower or 'mobile' in model_name_lower:
            tags.append('lightweight')
        
        return tags
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in MB"""
        param_count = self._count_parameters()
        return (param_count * 4) / (1024 * 1024)  # 4 bytes per parameter


class ModelFactory:
    """Factory for creating and managing FUA model adapters"""
    
    def __init__(self):
        self.model_registry = {}
        self.adapter_registry = {}
        self._register_default_models()
    
    def register_model(self, model_name: str, model_factory: Callable, 
                      capabilities: Optional[ModelCapabilities] = None) -> None:
        """Register a model factory function"""
        self.model_registry[model_name] = {
            'factory': model_factory,
            'capabilities': capabilities or self._create_default_capabilities(model_name)
        }
    
    def register_adapter(self, adapter_name: str, adapter_class: Type[ModelAdapter]) -> None:
        """Register a model adapter class"""
        self.adapter_registry[adapter_name] = adapter_class
    
    def create_model(self, model_name: str, **kwargs) -> ModelInterface:
        """Create a model instance"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not registered. Available models: {list(self.model_registry.keys())}")
        
        model_info = self.model_registry[model_name]
        
        # Create adapter instance
        adapter = ModelAdapter(
            model_name=model_name,
            model_factory=model_info['factory'],
            capabilities=model_info['capabilities']
        )
        
        # Configure with provided kwargs
        if kwargs:
            adapter.configure(kwargs)
        
        return adapter
    
    def create_adapter(self, adapter_name: str, **kwargs) -> ModelInterface:
        """Create a specific adapter instance"""
        if adapter_name not in self.adapter_registry:
            raise ValueError(f"Adapter '{adapter_name}' not registered. Available adapters: {list(self.adapter_registry.keys())}")
        
        adapter_class = self.adapter_registry[adapter_name]
        return adapter_class(**kwargs)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.model_registry.keys())
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapter names"""
        return list(self.adapter_registry.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a registered model"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model '{model_name}' not registered")
        
        model_info = self.model_registry[model_name]
        return {
            'name': model_name,
            'capabilities': model_info['capabilities'].to_dict(),
            'has_adapter': model_name in self.adapter_registry
        }
    
    def _register_default_models(self):
        """Register default models from the codebase"""
        # Try to import and register models from the models directory
        default_models = [
            'airbubble_hybrid_net',
            'resnet_improved', 
            'efficientnet',
            'mic_mobilenetv3',
            'micro_vit'
        ]
        
        for model_name in default_models:
            try:
                # Try to import the model module
                module_path = f'models.{model_name}'
                module = importlib.import_module(module_path)
                
                # Find the create function
                create_func_name = f'create_{model_name}'
                if hasattr(module, create_func_name):
                    create_func = getattr(module, create_func_name)
                    self.register_model(model_name, create_func)
                    print(f"✓ Registered model: {model_name}")
                else:
                    print(f"⚠ No create function found for: {model_name}")
                    
            except ImportError as e:
                print(f"⚠ Could not import model {model_name}: {e}")
            except Exception as e:
                print(f"⚠ Error registering model {model_name}: {e}")
    
    def _create_default_capabilities(self, model_name: str) -> ModelCapabilities:
        """Create default capabilities for a model"""
        model_name_lower = model_name.lower()
        
        # Default capabilities based on model type
        if 'hybrid' in model_name_lower:
            return ModelCapabilities(
                input_size_range=((60, 60), (80, 80)),
                recommended_batch_size=(16, 64),
                supported_optimizers=['adam', 'sgd'],
                supported_schedulers=['cosine', 'step'],
                special_preprocessing=['bubble_detection', 'multi_scale'],
                memory_requirements={'min_memory': 2048, 'recommended_memory': 4096},
                computational_complexity='high',
                training_time_estimate='slow'
            )
        elif 'efficientnet' in model_name_lower or 'mobilenet' in model_name_lower:
            return ModelCapabilities(
                input_size_range=((60, 60), (80, 80)),
                recommended_batch_size=(32, 128),
                supported_optimizers=['adam', 'rmsprop'],
                supported_schedulers=['cosine', 'plateau'],
                special_preprocessing=['normalization', 'auto_augment'],
                memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
                computational_complexity='medium',
                training_time_estimate='medium'
            )
        elif 'vit' in model_name_lower or 'transformer' in model_name_lower:
            return ModelCapabilities(
                input_size_range=((60, 60), (80, 80)),
                recommended_batch_size=(16, 32),
                supported_optimizers=['adam', 'adamw'],
                supported_schedulers=['cosine', 'linear'],
                special_preprocessing=['patch_extraction', 'positional_encoding'],
                memory_requirements={'min_memory': 1536, 'recommended_memory': 3072},
                computational_complexity='high',
                training_time_estimate='slow'
            )
        else:
            return ModelCapabilities(
                input_size_range=((60, 60), (80, 80)),
                recommended_batch_size=(16, 64),
                supported_optimizers=['adam', 'sgd'],
                supported_schedulers=['step', 'cosine'],
                special_preprocessing=['normalization'],
                memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
                computational_complexity='medium',
                training_time_estimate='medium'
            )


class ModelManager:
    """High-level manager for model operations in FUA"""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.factory = ModelFactory()
        self.config_manager = config_manager or ConfigurationManager()
        self.active_models = {}
    
    def create_model(self, model_name: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Create and manage a model instance"""
        # Use more precise timestamp to avoid ID collisions
        import time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        model_id = f"{model_name}_{timestamp}"
        
        try:
            # Create model instance
            model = self.factory.create_model(model_name, **(config or {}))
            
            # Store in active models
            self.active_models[model_id] = {
                'model': model,
                'config': config or {},
                'created_at': datetime.now(),
                'last_used': datetime.now()
            }
            
            return model_id
        except Exception as e:
            print(f"Error creating model {model_name}: {e}")
            raise ValueError(f"Failed to create model {model_name}: {e}")
    
    def get_model(self, model_id: str) -> Optional[ModelInterface]:
        """Get a model instance by ID"""
        if model_id in self.active_models:
            self.active_models[model_id]['last_used'] = datetime.now()
            return self.active_models[model_id]['model']
        return None
    
    def configure_model(self, model_id: str, config: Dict[str, Any]) -> bool:
        """Configure an existing model"""
        model = self.get_model(model_id)
        if model:
            success = model.configure(config)
            if success:
                self.active_models[model_id]['config'].update(config)
            return success
        return False
    
    def train_model(self, model_id: str, data: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train a model"""
        model = self.get_model(model_id)
        if model:
            return model.train(data, config)
        return {}
    
    def evaluate_model(self, model_id: str, data: Any, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate a model"""
        model = self.get_model(model_id)
        if model:
            return model.evaluate(data, metrics)
        return {}
    
    def save_model(self, model_id: str, path: str) -> bool:
        """Save a model"""
        model = self.get_model(model_id)
        if model:
            return model.save(path)
        return False
    
    def load_model(self, path: str, model_name: str) -> str:
        """Load a model from file"""
        # Use more precise timestamp to avoid ID collisions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        model_id = f"{model_name}_loaded_{timestamp}"
        
        # Create model instance
        model = self.factory.create_model(model_name)
        
        # Load from file
        success = model.load(path)
        if success:
            self.active_models[model_id] = {
                'model': model,
                'config': model.config,
                'created_at': datetime.now(),
                'last_used': datetime.now(),
                'loaded_from': path
            }
            return model_id
        else:
            raise ValueError(f"Failed to load model from {path}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all active models"""
        models = []
        for model_id, info in self.active_models.items():
            model = info['model']
            models.append({
                'model_id': model_id,
                'model_name': model.model_name if hasattr(model, 'model_name') else 'unknown',
                'created_at': info['created_at'],
                'last_used': info['last_used'],
                'config': info['config']
            })
        return models
    
    def cleanup_old_models(self, max_age_hours: int = 24) -> int:
        """Clean up old model instances"""
        current_time = datetime.now()
        cleaned_count = 0
        
        models_to_remove = []
        for model_id, info in self.active_models.items():
            age = (current_time - info['last_used']).total_seconds() / 3600
            if age > max_age_hours:
                models_to_remove.append(model_id)
        
        for model_id in models_to_remove:
            del self.active_models[model_id]
            cleaned_count += 1
        
        return cleaned_count