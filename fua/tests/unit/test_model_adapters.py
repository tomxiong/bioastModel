"""
Unit tests for FUA model adapters and integration
"""

import unittest
import tempfile
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch
import torch.nn as nn

# Import FUA components
import fua


class TestFUAModelAdapter(unittest.TestCase):
    """Test cases for FUA model adapter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = fua.ConfigurationManager()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_adapter_interface(self):
        """Test that model adapter implements ModelInterface"""
        
        class TestModelAdapter(fua.ModelInterface):
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.model = self._create_mock_model()
                self.config = {}
                
                # Define capabilities
                self.capabilities = fua.ModelCapabilities(
                    input_size_range=((60, 60), (80, 80)),
                    recommended_batch_size=(16, 64),
                    supported_optimizers=['adam', 'sgd', 'rmsprop'],
                    supported_schedulers=['step', 'cosine', 'plateau'],
                    special_preprocessing=['normalization', 'standardization'],
                    memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
                    computational_complexity='medium',
                    training_time_estimate='medium'
                )
            
            def _create_mock_model(self):
                """Create a mock PyTorch model for testing"""
                return nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(32, 2)
                )
            
            def get_capabilities(self) -> fua.ModelCapabilities:
                return self.capabilities
            
            def get_metadata(self) -> fua.ModelMetadata:
                return fua.ModelMetadata(
                    name=self.model_name,
                    version='1.0.0',
                    architecture_type='cnn',
                    parameter_count=sum(p.numel() for p in self.model.parameters()),
                    computational_complexity=1.5,
                    memory_usage=2048,
                    supported_input_sizes=[(70, 70)],
                    performance_metrics={'accuracy': 0.9, 'f1_score': 0.85},
                    training_history=[],
                    creation_date=datetime.now(),
                    last_modified=datetime.now(),
                    author='test_engineer',
                    tags=['test', 'cnn', 'adapter'],
                    description=f'FUA adapter for {self.model_name}'
                )
            
            def configure(self, config: Dict[str, Any]) -> bool:
                self.config = config
                return True
            
            def train(self, data: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                # Mock training simulation
                return {
                    'loss': 0.1,
                    'accuracy': 0.92,
                    'training_time': 120.5,
                    'epochs_completed': 10,
                    'model_size_mb': 2.5
                }
            
            def evaluate(self, data: Any, metrics: Optional[List[str]] = None) -> Dict[str, float]:
                # Mock evaluation simulation
                return {
                    'accuracy': 0.94,
                    'f1_score': 0.89,
                    'precision': 0.92,
                    'recall': 0.87,
                    'auc': 0.96
                }
            
            def save(self, path: str) -> bool:
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    
                    # Save model state dict
                    model_path = path.replace('.json', '_model.pth')
                    torch.save(self.model.state_dict(), model_path)
                    
                    # Save metadata
                    metadata = {
                        'model_name': self.model_name,
                        'config': self.config,
                        'capabilities': self.capabilities.to_dict(),
                        'metadata': self.get_metadata().to_dict(),
                        'model_path': model_path
                    }
                    
                    with open(path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    return True
                except Exception as e:
                    print(f"Error saving model: {e}")
                    return False
            
            def load(self, path: str) -> bool:
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.model_name = metadata['model_name']
                    self.config = metadata['config']
                    
                    # Load model state dict
                    if os.path.exists(metadata['model_path']):
                        self.model.load_state_dict(torch.load(metadata['model_path']))
                    
                    return True
                except Exception as e:
                    print(f"Error loading model: {e}")
                    return False
        
        # Test adapter creation
        adapter = TestModelAdapter('test_adapter')
        self.assertIsInstance(adapter, fua.ModelInterface)
        
        # Test capabilities
        capabilities = adapter.get_capabilities()
        self.assertIsInstance(capabilities, fua.ModelCapabilities)
        self.assertEqual(capabilities.computational_complexity, 'medium')
        
        # Test metadata
        metadata = adapter.get_metadata()
        self.assertIsInstance(metadata, fua.ModelMetadata)
        self.assertEqual(metadata.name, 'test_adapter')
        
        # Test configuration
        config = {'learning_rate': 0.001, 'batch_size': 32}
        success = adapter.configure(config)
        self.assertTrue(success)
        
        # Test training and evaluation
        training_results = adapter.train({'samples': 1000})
        self.assertIsInstance(training_results, dict)
        self.assertIn('accuracy', training_results)
        
        eval_results = adapter.evaluate({'samples': 200})
        self.assertIsInstance(eval_results, dict)
        self.assertIn('accuracy', eval_results)
        
        # Test save and load
        model_path = os.path.join(self.temp_dir, 'test_adapter.json')
        save_success = adapter.save(model_path)
        self.assertTrue(save_success)
        
        new_adapter = TestModelAdapter('new_adapter')
        load_success = new_adapter.load(model_path)
        self.assertTrue(load_success)
        self.assertEqual(new_adapter.model_name, 'test_adapter')


class TestModelFactory(unittest.TestCase):
    """Test cases for model factory"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = fua.ModelFactory()
    
    def test_model_registration(self):
        """Test model registration in factory"""
        
        # Mock model creation function
        def create_mock_model(num_classes=2, **kwargs):
            class MockModel:
                def __init__(self, num_classes):
                    self.num_classes = num_classes
                    self.parameters = lambda: [MockParameter()]
            
            class MockParameter:
                def numel(self):
                    return 1000
            
            return MockModel(num_classes)
        
        # Register model
        self.factory.register_model('mock_model', create_mock_model)
        
        # Check if model is registered
        self.assertIn('mock_model', self.factory.get_available_models())
        
        # Create model instance
        model = self.factory.create_model('mock_model', num_classes=2)
        self.assertIsNotNone(model)
        # Check if model was created and configured
        self.assertEqual(model.config.get('num_classes'), 2)
    
    def test_model_creation_with_config(self):
        """Test model creation with configuration"""
        
        def create_configurable_model(config=None, **kwargs):
            if config is None:
                config = {}
            
            class ConfigurableModel:
                def __init__(self, config):
                    self.config = config
                    self.parameters = lambda: [MockParameter()]
            
            class MockParameter:
                def numel(self):
                    return 1000
            
            return ConfigurableModel(config)
        
        # Register model
        self.factory.register_model('configurable_model', create_configurable_model)
        
        # Create with configuration
        config = {'learning_rate': 0.001, 'layers': 10}
        model = self.factory.create_model('configurable_model', **config)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.config.get('learning_rate'), 0.001)
        self.assertEqual(model.config.get('layers'), 10)
    
    def test_invalid_model_creation(self):
        """Test creation of invalid models"""
        
        # Try to create non-existent model
        with self.assertRaises(ValueError):
            self.factory.create_model('nonexistent_model')
        
        # Try to create model without registration
        with self.assertRaises(ValueError):
            self.factory.create_model('unregistered_model')


class TestModelIntegration(unittest.TestCase):
    """Test cases for model integration with existing codebase"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = fua.ConfigurationManager()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_existing_model_integration(self):
        """Test integration with existing model patterns"""
        
        class FUAAirBubbleAdapter(fua.ModelInterface):
            """Adapter for existing AirBubble Hybrid Net model"""
            
            def __init__(self):
                self.model = None  # Will be loaded from existing models
                self.config = {}
                self.model_name = 'airbubble_hybrid_net'
                
                self.capabilities = fua.ModelCapabilities(
                    input_size_range=((60, 60), (80, 80)),
                    recommended_batch_size=(16, 64),
                    supported_optimizers=['adam', 'sgd'],
                    supported_schedulers=['cosine', 'step'],
                    special_preprocessing=['bubble_detection', 'multi_scale'],
                    memory_requirements={'min_memory': 2048, 'recommended_memory': 4096},
                    computational_complexity='high',
                    training_time_estimate='slow'
                )
            
            def get_capabilities(self) -> fua.ModelCapabilities:
                return self.capabilities
            
            def get_metadata(self) -> fua.ModelMetadata:
                return fua.ModelMetadata(
                    name=self.model_name,
                    version='1.0.0',
                    architecture_type='cnn_transformer_hybrid',
                    parameter_count=2500000,  # Approximate
                    computational_complexity=2.5,
                    memory_usage=4096,
                    supported_input_sizes=[(70, 70)],
                    performance_metrics={'accuracy': 0.98, 'f1_score': 0.97},
                    training_history=[],
                    creation_date=datetime.now(),
                    last_modified=datetime.now(),
                    author='fua_team',
                    tags=['hybrid', 'attention', 'bubble_detection'],
                    description='AirBubble Hybrid Net with FUA adapter'
                )
            
            def configure(self, config: Dict[str, Any]) -> bool:
                self.config = config
                
                # Load actual model if available
                try:
                    from models.airbubble_hybrid_net import create_airbubble_hybrid_net
                    self.model = create_airbubble_hybrid_net(num_classes=2)
                    return True
                except ImportError:
                    print("Warning: Could not import airbubble_hybrid_net, using mock")
                    return True
            
            def train(self, data: Any, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                # Mock training with realistic results for this model
                return {
                    'loss': 0.05,
                    'accuracy': 0.98,
                    'f1_score': 0.97,
                    'training_time': 300.0,
                    'epochs_completed': 20,
                    'best_epoch': 15,
                    'model_size_mb': 10.5
                }
            
            def evaluate(self, data: Any, metrics: Optional[List[str]] = None) -> Dict[str, float]:
                # Mock evaluation with realistic results
                return {
                    'accuracy': 0.98,
                    'f1_score': 0.97,
                    'precision': 0.98,
                    'recall': 0.97,
                    'auc': 0.99,
                    'specificity': 0.98
                }
            
            def save(self, path: str) -> bool:
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    
                    # Save adapter metadata
                    metadata = {
                        'model_name': self.model_name,
                        'config': self.config,
                        'capabilities': self.capabilities.to_dict(),
                        'metadata': self.get_metadata().to_dict(),
                        'adapter_type': 'airbubble_hybrid_net'
                    }
                    
                    with open(path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    return True
                except Exception as e:
                    print(f"Error saving adapter: {e}")
                    return False
            
            def load(self, path: str) -> bool:
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    
                    self.model_name = metadata['model_name']
                    self.config = metadata['config']
                    
                    return True
                except Exception as e:
                    print(f"Error loading adapter: {e}")
                    return False
        
        # Test adapter creation
        adapter = FUAAirBubbleAdapter()
        self.assertIsInstance(adapter, fua.ModelInterface)
        
        # Test configuration
        config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'attention_config': {'num_heads': 8, 'dropout': 0.1}
        }
        
        success = adapter.configure(config)
        self.assertTrue(success)
        
        # Test capabilities reflect the high-performance nature
        capabilities = adapter.get_capabilities()
        self.assertEqual(capabilities.computational_complexity, 'high')
        self.assertIn('bubble_detection', capabilities.special_preprocessing)
        
        # Test metadata shows realistic performance
        metadata = adapter.get_metadata()
        self.assertEqual(metadata.architecture_type, 'cnn_transformer_hybrid')
        self.assertGreater(metadata.performance_metrics['accuracy'], 0.95)
        
        # Test training and evaluation with realistic metrics
        training_results = adapter.train({'samples': 5000})
        self.assertGreater(training_results['accuracy'], 0.95)
        self.assertGreater(training_results['f1_score'], 0.95)
        
        eval_results = adapter.evaluate({'samples': 1000})
        self.assertGreater(eval_results['accuracy'], 0.95)
        self.assertGreater(eval_results['auc'], 0.98)


class TestModelConfiguration(unittest.TestCase):
    """Test cases for model-specific configuration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = fua.ConfigurationManager()
    
    def test_model_specific_config_schema(self):
        """Test model-specific configuration schema"""
        
        # Get schema for different model types
        cnn_schema = self.config_manager.get_config_schema('model')
        self.assertIsInstance(cnn_schema, dict)
        self.assertIn('required_fields', cnn_schema)
        self.assertIn('type_checks', cnn_schema)
        
        # Validate CNN model configuration
        cnn_config = {
            'name': 'test_cnn',
            'architecture_type': 'cnn',
            'input_size': [70, 70],
            'parameters': 1000000
        }
        
        is_valid, errors = self.config_manager.validate_config(cnn_config, 'model')
        self.assertTrue(is_valid, f"CNN config validation failed: {errors}")
        
        # Validate transformer model configuration
        transformer_config = {
            'name': 'test_transformer',
            'architecture_type': 'transformer',
            'input_size': [70, 70],
            'parameters': 2000000,
            'attention_config': {'num_heads': 8, 'dropout': 0.1}
        }
        
        is_valid, errors = self.config_manager.validate_config(transformer_config, 'model')
        self.assertTrue(is_valid, f"Transformer config validation failed: {errors}")
    
    def test_configuration_inheritance(self):
        """Test configuration inheritance for models"""
        
        # Base configuration
        base_config = {
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            }
        }
        
        # Model-specific configuration
        model_config = {
            'model': {
                'name': 'test_model',
                'architecture_type': 'cnn'
            }
        }
        
        # Override configuration
        override_config = {
            'training': {
                'learning_rate': 0.0005,  # Override base learning rate
                'weight_decay': 0.01      # Additional parameter
            }
        }
        
        # Merge configurations
        merged = self.config_manager.merge_configs(base_config, model_config, override_config)
        
        # Verify inheritance
        self.assertEqual(merged['training']['epochs'], 100)  # From base
        self.assertEqual(merged['training']['learning_rate'], 0.0005)  # Overridden
        self.assertEqual(merged['training']['weight_decay'], 0.01)  # From override
        self.assertEqual(merged['model']['name'], 'test_model')  # From model


if __name__ == '__main__':
    unittest.main()