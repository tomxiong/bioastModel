"""
Unit tests for FUA configuration validator
"""

import unittest
import tempfile
import os
import json
from typing import Dict, Any, List, Tuple

# Import the classes we will test
from fua.core.interfaces import ConfigurationManager, BaseConfigValidator


class TestBaseConfigValidator(unittest.TestCase):
    """Test cases for BaseConfigValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = BaseConfigValidator()
    
    def test_validate_model_config_valid(self):
        """Test validation of valid model configuration"""
        config = {
            'name': 'test_model',
            'architecture_type': 'cnn',
            'input_size': [70, 70],
            'parameters': 1000000,
            'layers': 10
        }
        
        is_valid, errors = self.validator.validate_model_config(config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_model_config_missing_required_fields(self):
        """Test validation with missing required fields"""
        config = {
            'name': 'test_model'
            # Missing architecture_type and input_size
        }
        
        is_valid, errors = self.validator.validate_model_config(config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn('Missing required field: architecture_type', errors)
        self.assertIn('Missing required field: input_size', errors)
    
    def test_validate_model_config_invalid_types(self):
        """Test validation with invalid field types"""
        config = {
            'name': 'test_model',
            'architecture_type': 'cnn',
            'input_size': 'invalid',  # Should be list or tuple
            'parameters': 'invalid'  # Should be int or float
        }
        
        is_valid, errors = self.validator.validate_model_config(config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('input_size' in error for error in errors))
        self.assertTrue(any('parameters' in error for error in errors))
    
    def test_validate_training_config_valid(self):
        """Test validation of valid training configuration"""
        config = {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        is_valid, errors = self.validator.validate_training_config(config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_training_config_missing_required_fields(self):
        """Test validation with missing required fields"""
        config = {
            'epochs': 100
            # Missing batch_size and learning_rate
        }
        
        is_valid, errors = self.validator.validate_training_config(config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn('Missing required field: batch_size', errors)
        self.assertIn('Missing required field: learning_rate', errors)
    
    def test_validate_data_config_valid(self):
        """Test validation of valid data configuration"""
        config = {
            'data_path': '/path/to/data',
            'input_size': [70, 70],
            'batch_size': 32,
            'augmentation': True
        }
        
        is_valid, errors = self.validator.validate_data_config(config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_data_config_invalid_types(self):
        """Test validation with invalid field types"""
        config = {
            'data_path': '/path/to/data',
            'input_size': 'invalid',  # Should be list or tuple
            'batch_size': 'invalid',  # Should be int
            'augmentation': 'invalid'  # Should be bool
        }
        
        is_valid, errors = self.validator.validate_data_config(config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_get_validation_rules(self):
        """Test getting validation rules"""
        model_rules = self.validator.get_validation_rules('model')
        self.assertIn('required_fields', model_rules)
        self.assertIn('optional_fields', model_rules)
        self.assertIn('type_checks', model_rules)
        
        unknown_rules = self.validator.get_validation_rules('unknown')
        self.assertEqual(unknown_rules, {})
    
    def test_validate_unknown_config_type(self):
        """Test validation of unknown config type"""
        config = {'test': 'value'}
        is_valid, errors = self.validator._validate_config(config, 'unknown')
        self.assertFalse(is_valid)
        self.assertIn('Unknown config type: unknown', errors)


class TestConfigurationManager(unittest.TestCase):
    """Test cases for ConfigurationManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = ConfigurationManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        config = {
            'name': 'test_model',
            'architecture_type': 'cnn',
            'input_size': [70, 70]
        }
        
        config_path = os.path.join(self.temp_dir, 'test_config.json')
        
        # Save configuration
        success = self.manager.save_config(config, config_path, 'model')
        self.assertTrue(success)
        self.assertTrue(os.path.exists(config_path))
        
        # Load configuration
        loaded_config = self.manager.load_config(config_path, 'model')
        self.assertEqual(loaded_config, config)
    
    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration"""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.json')
        
        with self.assertRaises(FileNotFoundError):
            self.manager.load_config(nonexistent_path, 'model')
    
    def test_config_caching(self):
        """Test configuration caching"""
        config = {'name': 'test_model'}
        config_path = os.path.join(self.temp_dir, 'test_config.json')
        
        # Save configuration
        self.manager.save_config(config, config_path, 'model')
        
        # Load configuration (should cache it)
        loaded1 = self.manager.load_config(config_path, 'model')
        loaded2 = self.manager.load_config(config_path, 'model')
        
        self.assertEqual(loaded1, loaded2)
        self.assertEqual(loaded1, config)
    
    def test_merge_configs_simple(self):
        """Test simple configuration merging"""
        config1 = {'name': 'model1', 'params': {'lr': 0.001}}
        config2 = {'architecture': 'cnn', 'params': {'batch_size': 32}}
        
        merged = self.manager.merge_configs(config1, config2)
        
        expected = {
            'name': 'model1',
            'architecture': 'cnn',
            'params': {'lr': 0.001, 'batch_size': 32}
        }
        
        self.assertEqual(merged, expected)
    
    def test_merge_configs_deep(self):
        """Test deep configuration merging"""
        config1 = {
            'training': {
                'epochs': 100,
                'optimizer': {'name': 'adam', 'lr': 0.001}
            }
        }
        config2 = {
            'training': {
                'optimizer': {'lr': 0.002, 'weight_decay': 0.01},
                'batch_size': 32
            }
        }
        
        merged = self.manager.merge_configs(config1, config2)
        
        expected = {
            'training': {
                'epochs': 100,
                'optimizer': {'name': 'adam', 'lr': 0.002, 'weight_decay': 0.01},
                'batch_size': 32
            }
        }
        
        self.assertEqual(merged, expected)
    
    def test_validate_config_through_manager(self):
        """Test configuration validation through manager"""
        valid_config = {
            'name': 'test_model',
            'architecture_type': 'cnn',
            'input_size': [70, 70]
        }
        
        is_valid, errors = self.manager.validate_config(valid_config, 'model')
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        invalid_config = {
            'name': 'test_model'
            # Missing required fields
        }
        
        is_valid, errors = self.manager.validate_config(invalid_config, 'model')
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_get_config_schema_through_manager(self):
        """Test getting configuration schema through manager"""
        schema = self.manager.get_config_schema('model')
        self.assertIn('required_fields', schema)
        self.assertIn('optional_fields', schema)
        self.assertIn('type_checks', schema)
    
    def test_validate_unknown_config_type(self):
        """Test validation of unknown config type through manager"""
        config = {'test': 'value'}
        is_valid, errors = self.manager.validate_config(config, 'unknown')
        self.assertFalse(is_valid)
        self.assertIn('Unknown config type: unknown', errors)


class TestConfigurationManagerIntegration(unittest.TestCase):
    """Test cases for ConfigurationManager integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = ConfigurationManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_config_workflow(self):
        """Test complete configuration workflow"""
        # Create multiple configuration files
        base_config = {
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        model_config = {
            'model': {
                'name': 'test_model',
                'architecture_type': 'cnn',
                'input_size': [70, 70]
            }
        }
        
        data_config = {
            'data': {
                'data_path': '/path/to/data',
                'input_size': [70, 70],
                'batch_size': 32
            }
        }
        
        # Save configurations
        base_path = os.path.join(self.temp_dir, 'base_config.json')
        model_path = os.path.join(self.temp_dir, 'model_config.json')
        data_path = os.path.join(self.temp_dir, 'data_config.json')
        
        self.manager.save_config(base_config, base_path, 'training')
        self.manager.save_config(model_config, model_path, 'model')
        self.manager.save_config(data_config, data_path, 'data')
        
        # Load and merge configurations
        loaded_base = self.manager.load_config(base_path, 'training')
        loaded_model = self.manager.load_config(model_path, 'model')
        loaded_data = self.manager.load_config(data_path, 'data')
        
        merged_config = self.manager.merge_configs(loaded_base, loaded_model, loaded_data)
        
        # Validate merged configuration
        is_valid, errors = self.manager.validate_config(merged_config['model'], 'model')
        self.assertTrue(is_valid)
        
        is_valid, errors = self.manager.validate_config(merged_config['training'], 'training')
        self.assertTrue(is_valid)
        
        is_valid, errors = self.manager.validate_config(merged_config['data'], 'data')
        self.assertTrue(is_valid)
        
        # Verify merged structure
        self.assertIn('training', merged_config)
        self.assertIn('model', merged_config)
        self.assertIn('data', merged_config)
        self.assertEqual(merged_config['model']['name'], 'test_model')
        self.assertEqual(merged_config['training']['epochs'], 100)


if __name__ == '__main__':
    unittest.main()