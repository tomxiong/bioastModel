"""
Unit tests for FUA model-specific configuration system
"""

import unittest
import tempfile
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import FUA components
import fua


class TestModelSpecificConfig(unittest.TestCase):
    """Test cases for model-specific configuration system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = fua.ConfigurationManager()
        self.model_config_system = fua.ModelConfigurationSystem()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_config_registration(self):
        """Test registration of model-specific configurations"""
        
        # Register CNN model configuration
        cnn_config_schema = {
            'required_fields': ['name', 'architecture_type', 'input_size'],
            'optional_fields': ['layers', 'filters', 'kernel_size'],
            'type_checks': {
                'name': str,
                'architecture_type': str,
                'input_size': (list, tuple),
                'layers': int,
                'filters': int,
                'kernel_size': int
            },
            'default_values': {
                'layers': 10,
                'filters': 32,
                'kernel_size': 3
            }
        }
        
        self.model_config_system.register_model_config('cnn', cnn_config_schema)
        
        # Verify registration
        self.assertIn('cnn', self.model_config_system.get_registered_configs())
        
        # Get schema
        schema = self.model_config_system.get_model_config_schema('cnn')
        self.assertEqual(schema['required_fields'], ['name', 'architecture_type', 'input_size'])
    
    def test_model_config_validation(self):
        """Test validation of model-specific configurations"""
        
        # Register transformer model configuration
        transformer_config_schema = {
            'required_fields': ['name', 'architecture_type', 'input_size', 'num_heads'],
            'optional_fields': ['num_layers', 'hidden_size', 'dropout'],
            'type_checks': {
                'name': str,
                'architecture_type': str,
                'input_size': (list, tuple),
                'num_heads': int,
                'num_layers': int,
                'hidden_size': int,
                'dropout': (int, float)
            },
            'value_ranges': {
                'num_heads': [1, 16],  # Must be between 1 and 16
                'dropout': [0.0, 1.0],  # Must be between 0.0 and 1.0
                'num_layers': [1, 24]  # Must be between 1 and 24
            },
            'default_values': {
                'num_layers': 12,
                'hidden_size': 768,
                'dropout': 0.1
            }
        }
        
        self.model_config_system.register_model_config('transformer', transformer_config_schema)
        
        # Test valid configuration
        valid_config = {
            'name': 'test_transformer',
            'architecture_type': 'transformer',
            'input_size': [70, 70],
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1
        }
        
        is_valid, errors = self.model_config_system.validate_model_config('transformer', valid_config)
        self.assertTrue(is_valid, f"Valid config rejected: {errors}")
        
        # Test invalid configuration - value out of range
        invalid_config = valid_config.copy()
        invalid_config['num_heads'] = 32  # Too large
        
        is_valid, errors = self.model_config_system.validate_model_config('transformer', invalid_config)
        self.assertFalse(is_valid)
        self.assertTrue(any('num_heads' in error for error in errors))
        
        # Test configuration with default values applied
        minimal_config = {
            'name': 'minimal_transformer',
            'architecture_type': 'transformer',
            'input_size': [70, 70],
            'num_heads': 4
        }
        
        enhanced_config = self.model_config_system.apply_defaults('transformer', minimal_config)
        self.assertEqual(enhanced_config['num_layers'], 12)  # Default applied
        self.assertEqual(enhanced_config['hidden_size'], 768)  # Default applied
        self.assertEqual(enhanced_config['dropout'], 0.1)  # Default applied
    
    def test_config_inheritance_and_merging(self):
        """Test configuration inheritance and merging"""
        
        # Register base CNN configuration
        base_cnn_schema = {
            'required_fields': ['name', 'architecture_type', 'input_size'],
            'optional_fields': ['layers', 'filters'],
            'type_checks': {
                'name': str,
                'architecture_type': str,
                'input_size': (list, tuple),
                'layers': int,
                'filters': int
            },
            'default_values': {
                'layers': 10,
                'filters': 32
            }
        }
        
        # Register specialized EfficientNet configuration
        efficientnet_schema = {
            'required_fields': ['name', 'architecture_type', 'input_size', 'width_coefficient'],
            'optional_fields': ['depth_coefficient', 'dropout_rate'],
            'type_checks': {
                'name': str,
                'architecture_type': str,
                'input_size': (list, tuple),
                'width_coefficient': (int, float),
                'depth_coefficient': (int, float),
                'dropout_rate': (int, float)
            },
            'parent_config': 'cnn',  # Inherits from CNN
            'default_values': {
                'width_coefficient': 1.0,
                'depth_coefficient': 1.0,
                'dropout_rate': 0.2
            }
        }
        
        self.model_config_system.register_model_config('cnn', base_cnn_schema)
        self.model_config_system.register_model_config('efficientnet', efficientnet_schema)
        
        # Test inheritance
        efficientnet_full_schema = self.model_config_system.get_model_config_schema('efficientnet')
        self.assertIn('layers', efficientnet_full_schema['optional_fields'])  # Inherited from CNN
        self.assertIn('width_coefficient', efficientnet_full_schema['type_checks'])  # Specific to EfficientNet in type_checks
        
        # Test merging configurations
        base_config = {
            'name': 'test_model',
            'architecture_type': 'efficientnet',
            'input_size': [70, 70],
            'layers': 20  # From base CNN schema
        }
        
        specific_config = {
            'width_coefficient': 1.2,
            'depth_coefficient': 1.1
        }
        
        merged_config = self.model_config_system.merge_model_configs('efficientnet', base_config, specific_config)
        self.assertEqual(merged_config['layers'], 20)  # From base
        self.assertEqual(merged_config['width_coefficient'], 1.2)  # From specific
        self.assertEqual(merged_config['depth_coefficient'], 1.1)  # From specific
    
    def test_model_config_templates(self):
        """Test model configuration templates"""
        
        # Register configuration templates
        templates = {
            'lightweight_cnn': {
                'architecture_type': 'cnn',
                'layers': 5,
                'filters': 16,
                'kernel_size': 3,
                'batch_size': 64,
                'learning_rate': 0.01
            },
            'high_performance_cnn': {
                'architecture_type': 'cnn',
                'layers': 20,
                'filters': 64,
                'kernel_size': 3,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'efficientnet_b0': {
                'architecture_type': 'efficientnet',
                'width_coefficient': 1.0,
                'depth_coefficient': 1.0,
                'dropout_rate': 0.2,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        for template_name, template_config in templates.items():
            self.model_config_system.register_config_template(template_name, template_config)
        
        # Test template retrieval
        available_templates = self.model_config_system.get_available_templates()
        self.assertIn('lightweight_cnn', available_templates)
        self.assertIn('high_performance_cnn', available_templates)
        self.assertIn('efficientnet_b0', available_templates)
        
        # Test template application
        lightweight_config = self.model_config_system.apply_template('lightweight_cnn', {'name': 'test_lightweight'})
        self.assertEqual(lightweight_config['layers'], 5)
        self.assertEqual(lightweight_config['filters'], 16)
        self.assertEqual(lightweight_config['batch_size'], 64)
        
        # Test template override
        custom_config = self.model_config_system.apply_template(
            'efficientnet_b0', 
            {'name': 'custom_efficientnet', 'width_coefficient': 1.5}
        )
        self.assertEqual(custom_config['width_coefficient'], 1.5)  # Overridden
        self.assertEqual(custom_config['depth_coefficient'], 1.0)  # From template
        self.assertEqual(custom_config['dropout_rate'], 0.2)  # From template
    
    def test_config_generation_from_model(self):
        """Test automatic configuration generation from model capabilities"""
        
        # Create a model with specific capabilities
        capabilities = fua.ModelCapabilities(
            input_size_range=((60, 60), (80, 80)),
            recommended_batch_size=(16, 64),
            supported_optimizers=['adam', 'sgd'],
            supported_schedulers=['cosine', 'step'],
            special_preprocessing=['normalization'],
            memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
            computational_complexity='medium',
            training_time_estimate='medium'
        )
        
        # Generate configuration based on capabilities
        generated_config = self.model_config_system.generate_config_from_capabilities(
            'test_model', 
            capabilities
        )
        
        # Verify generated configuration
        self.assertIn('name', generated_config)
        self.assertIn('training', generated_config)
        self.assertIn('model', generated_config)
        
        # Check training configuration matches capabilities
        training_config = generated_config['training']
        self.assertIn(training_config['batch_size'], range(16, 65))  # Within recommended range
        self.assertIn(training_config['optimizer'], capabilities.supported_optimizers)
        self.assertIn(training_config['scheduler'], capabilities.supported_schedulers)
        
        # Check model configuration reflects complexity
        model_config = generated_config['model']
        self.assertEqual(model_config['input_size'], [70, 70])  # Default from range
        self.assertIn('num_classes', model_config)  # Check for required field instead
    
    def test_config_validation_with_real_models(self):
        """Test configuration validation with real model types"""
        
        # Register configurations for actual models in the codebase
        model_configs = {
            'airbubble_hybrid_net': {
                'required_fields': ['name', 'architecture_type', 'input_size'],
                'optional_fields': ['attention_heads', 'fusion_layers', 'bubble_detection_weight'],
                'type_checks': {
                    'name': str,
                    'architecture_type': str,
                    'input_size': (list, tuple),
                    'attention_heads': int,
                    'fusion_layers': int,
                    'bubble_detection_weight': (int, float)
                },
                'value_ranges': {
                    'attention_heads': [1, 16],
                    'bubble_detection_weight': [0.0, 1.0]
                },
                'default_values': {
                    'attention_heads': 8,
                    'fusion_layers': 4,
                    'bubble_detection_weight': 0.3
                }
            },
            'mic_mobilenetv3': {
                'required_fields': ['name', 'architecture_type', 'input_size'],
                'optional_fields': ['width_multiplier', 'depth_multiplier', 'dropout_rate'],
                'type_checks': {
                    'name': str,
                    'architecture_type': str,
                    'input_size': (list, tuple),
                    'width_multiplier': (int, float),
                    'depth_multiplier': (int, float),
                    'dropout_rate': (int, float)
                },
                'value_ranges': {
                    'width_multiplier': [0.5, 2.0],
                    'depth_multiplier': [0.5, 2.0],
                    'dropout_rate': [0.0, 0.5]
                },
                'default_values': {
                    'width_multiplier': 1.0,
                    'depth_multiplier': 1.0,
                    'dropout_rate': 0.1
                }
            }
        }
        
        for model_name, config_schema in model_configs.items():
            self.model_config_system.register_model_config(model_name, config_schema)
        
        # Test validation with realistic configurations
        airbubble_config = {
            'name': 'airbubble_test',
            'architecture_type': 'cnn_transformer_hybrid',
            'input_size': [70, 70],
            'attention_heads': 8,
            'fusion_layers': 4,
            'bubble_detection_weight': 0.3
        }
        
        is_valid, errors = self.model_config_system.validate_model_config('airbubble_hybrid_net', airbubble_config)
        self.assertTrue(is_valid, f"AirBubble config validation failed: {errors}")
        
        # Test invalid configuration
        invalid_config = airbubble_config.copy()
        invalid_config['attention_heads'] = 32  # Too large
        
        is_valid, errors = self.model_config_system.validate_model_config('airbubble_hybrid_net', invalid_config)
        self.assertFalse(is_valid)
        self.assertTrue(any('attention_heads' in error for error in errors))


class TestModelConfigurationPersistence(unittest.TestCase):
    """Test cases for configuration persistence"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_system = fua.ModelConfigurationSystem()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_save_and_load(self):
        """Test saving and loading configurations"""
        
        # Register a configuration
        test_schema = {
            'required_fields': ['name', 'type'],
            'optional_fields': ['param1', 'param2'],
            'type_checks': {
                'name': str,
                'type': str,
                'param1': int,
                'param2': (int, float)
            },
            'default_values': {
                'param1': 10,
                'param2': 0.5
            }
        }
        
        self.config_system.register_model_config('test_model', test_schema)
        
        # Save configurations
        config_path = os.path.join(self.temp_dir, 'model_configs.json')
        success = self.config_system.save_configurations(config_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(config_path))
        
        # Load configurations in a new instance
        new_config_system = fua.ModelConfigurationSystem()
        load_success = new_config_system.load_configurations(config_path)
        self.assertTrue(load_success)
        
        # Verify loaded configuration
        loaded_schema = new_config_system.get_model_config_schema('test_model')
        self.assertEqual(loaded_schema['required_fields'], ['name', 'type'])
        self.assertEqual(loaded_schema['default_values']['param1'], 10)
    
    def test_template_save_and_load(self):
        """Test saving and loading configuration templates"""
        
        # Register templates
        templates = {
            'template1': {'param1': 10, 'param2': 0.5},
            'template2': {'param1': 20, 'param2': 1.0}
        }
        
        for name, template in templates.items():
            self.config_system.register_config_template(name, template)
        
        # Save templates
        template_path = os.path.join(self.temp_dir, 'config_templates.json')
        success = self.config_system.save_templates(template_path)
        self.assertTrue(success)
        
        # Load templates
        new_config_system = fua.ModelConfigurationSystem()
        load_success = new_config_system.load_templates(template_path)
        self.assertTrue(load_success)
        
        # Verify loaded templates
        loaded_templates = new_config_system.get_available_templates()
        self.assertIn('template1', loaded_templates)
        self.assertIn('template2', loaded_templates)


if __name__ == '__main__':
    unittest.main()