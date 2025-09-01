"""
Integration tests and performance benchmarks for FUA
"""

import unittest
import time
import tempfile
import os
import json
from typing import Dict, List, Any
from datetime import datetime

# Import FUA components
import fua


class TestFUAIntegration(unittest.TestCase):
    """Integration tests for FUA components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = fua.ConfigurationManager()
        self.validator = fua.BaseConfigValidator()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_model_lifecycle(self):
        """Test complete model lifecycle with FUA components"""
        
        class TestModel(fua.ModelInterface):
            def __init__(self, name: str):
                self.name = name
                self.config = {}
                self.capabilities = fua.ModelCapabilities(
                    input_size_range=((60, 60), (80, 80)),
                    recommended_batch_size=(16, 64),
                    supported_optimizers=['adam', 'sgd'],
                    supported_schedulers=['step', 'cosine'],
                    special_preprocessing=['normalization'],
                    memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
                    computational_complexity='medium',
                    training_time_estimate='medium'
                )
            
            def get_capabilities(self) -> fua.ModelCapabilities:
                return self.capabilities
            
            def get_metadata(self) -> fua.ModelMetadata:
                return fua.ModelMetadata(
                    name=self.name,
                    version='1.0.0',
                    architecture_type='test_cnn',
                    parameter_count=1000000,
                    computational_complexity=1.5,
                    memory_usage=2048,
                    supported_input_sizes=[(70, 70)],
                    performance_metrics={'accuracy': 0.9, 'f1_score': 0.85},
                    training_history=[],
                    creation_date=datetime.now(),
                    last_modified=datetime.now(),
                    author='test_engineer',
                    tags=['test', 'cnn'],
                    description='Test model for integration'
                )
            
            def configure(self, config: Dict[str, Any]) -> bool:
                self.config = config
                return True
            
            def train(self, data: Any, config: Dict[str, Any] = None) -> Dict[str, Any]:
                return {
                    'loss': 0.1,
                    'accuracy': 0.9,
                    'training_time': 120.5,
                    'epochs_completed': 10
                }
            
            def evaluate(self, data: Any, metrics: List[str] = None) -> Dict[str, float]:
                return {
                    'accuracy': 0.92,
                    'f1_score': 0.88,
                    'precision': 0.91,
                    'recall': 0.85
                }
            
            def save(self, path: str) -> bool:
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'w') as f:
                        json.dump({
                            'name': self.name,
                            'config': self.config,
                            'metadata': self.get_metadata().to_dict()
                        }, f)
                    return True
                except Exception:
                    return False
            
            def load(self, path: str) -> bool:
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    self.name = data['name']
                    self.config = data['config']
                    return True
                except Exception:
                    return False
        
        # Create model
        model = TestModel('integration_test_model')
        
        # Test model capabilities
        capabilities = model.get_capabilities()
        self.assertIsInstance(capabilities, fua.ModelCapabilities)
        self.assertEqual(capabilities.computational_complexity, 'medium')
        
        # Test model metadata
        metadata = model.get_metadata()
        self.assertIsInstance(metadata, fua.ModelMetadata)
        self.assertEqual(metadata.name, 'integration_test_model')
        
        # Test configuration
        config = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50
        }
        success = model.configure(config)
        self.assertTrue(success)
        
        # Test training and evaluation
        training_results = model.train({'samples': 1000})
        self.assertIsInstance(training_results, dict)
        self.assertIn('accuracy', training_results)
        
        eval_results = model.evaluate({'samples': 200})
        self.assertIsInstance(eval_results, dict)
        self.assertIn('accuracy', eval_results)
        
        # Test save and load
        model_path = os.path.join(self.temp_dir, 'test_model.json')
        save_success = model.save(model_path)
        self.assertTrue(save_success)
        
        new_model = TestModel('new_model')
        load_success = new_model.load(model_path)
        self.assertTrue(load_success)
        self.assertEqual(new_model.name, 'integration_test_model')
    
    def test_configuration_workflow(self):
        """Test complete configuration workflow"""
        
        # Create multiple configuration files
        base_config = {
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        }
        
        model_config = {
            'model': {
                'name': 'workflow_test_model',
                'architecture_type': 'cnn',
                'input_size': [70, 70],
                'parameters': 1000000
            }
        }
        
        data_config = {
            'data': {
                'data_path': '/path/to/data',
                'input_size': [70, 70],
                'batch_size': 32,
                'augmentation': True
            }
        }
        
        # Save configurations
        base_path = os.path.join(self.temp_dir, 'base.json')
        model_path = os.path.join(self.temp_dir, 'model.json')
        data_path = os.path.join(self.temp_dir, 'data.json')
        
        self.config_manager.save_config(base_config, base_path, 'training')
        self.config_manager.save_config(model_config, model_path, 'model')
        self.config_manager.save_config(data_config, data_path, 'data')
        
        # Load and merge configurations
        loaded_base = self.config_manager.load_config(base_path, 'training')
        loaded_model = self.config_manager.load_config(model_path, 'model')
        loaded_data = self.config_manager.load_config(data_path, 'data')
        
        merged_config = self.config_manager.merge_configs(loaded_base, loaded_model, loaded_data)
        
        # Validate all configurations
        model_valid, model_errors = self.validator.validate_model_config(merged_config['model'])
        training_valid, training_errors = self.validator.validate_training_config(merged_config['training'])
        data_valid, data_errors = self.validator.validate_data_config(merged_config['data'])
        
        self.assertTrue(model_valid, f"Model config validation failed: {model_errors}")
        self.assertTrue(training_valid, f"Training config validation failed: {training_errors}")
        self.assertTrue(data_valid, f"Data config validation failed: {data_errors}")
        
        # Test configuration caching
        cached_base = self.config_manager.load_config(base_path, 'training')
        self.assertEqual(cached_base, loaded_base)
    
    def test_error_and_improvement_workflow(self):
        """Test error detection and improvement workflow"""
        
        class TestAutomationEngine(fua.AutomationEngine):
            def __init__(self):
                self.errors_detected = []
                self.improvements_suggested = []
            
            def detect_errors(self, model: Any, training_data: Dict[str, Any]) -> List[fua.Error]:
                # Simulate error detection
                if training_data.get('loss', 0) > 1.0:
                    error = fua.Error(
                        type='high_loss',
                        severity='high',
                        description='Training loss is too high',
                        model_name=getattr(model, 'name', 'unknown'),
                        timestamp=datetime.now(),
                        metrics={'loss': training_data['loss']},
                        context=training_data
                    )
                    self.errors_detected.append(error)
                    return [error]
                return []
            
            def suggest_improvements(self, errors: List[fua.Error]) -> List[fua.Improvement]:
                improvements = []
                for error in errors:
                    if error.type == 'high_loss':
                        improvement = fua.Improvement(
                            type='adjust_learning_rate',
                            description='Reduce learning rate to stabilize training',
                            implementation=lambda x: x * 0.5,  # Reduce LR by half
                            priority='high',
                            expected_impact='high',
                            implementation_complexity='easy',
                            estimated_time='minutes',
                            risk_level='low'
                        )
                        improvements.append(improvement)
                        self.improvements_suggested.append(improvement)
                return improvements
            
            def apply_improvement(self, improvement: fua.Improvement, model: Any) -> bool:
                # Simulate improvement application
                return True
            
            def validate_improvement(self, improvement: fua.Improvement, model: Any, validation_data: Any) -> bool:
                # Simulate improvement validation
                return True
            
            def get_automation_stats(self) -> Dict[str, Any]:
                return {
                    'errors_detected': len(self.errors_detected),
                    'improvements_suggested': len(self.improvements_suggested),
                    'improvements_applied': len(self.improvements_suggested) // 2  # Assume half applied
                }
        
        # Test automation workflow
        engine = TestAutomationEngine()
        
        # Simulate problematic training data
        training_data = {
            'loss': 1.5,
            'accuracy': 0.6,
            'epoch': 5
        }
        
        # Detect errors
        errors = engine.detect_errors('test_model', training_data)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].type, 'high_loss')
        
        # Suggest improvements
        improvements = engine.suggest_improvements(errors)
        self.assertEqual(len(improvements), 1)
        self.assertEqual(improvements[0].type, 'adjust_learning_rate')
        
        # Get stats
        stats = engine.get_automation_stats()
        self.assertEqual(stats['errors_detected'], 1)
        self.assertEqual(stats['improvements_suggested'], 1)


class TestFUAPerformance(unittest.TestCase):
    """Performance benchmarks for FUA components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_manager = fua.ConfigurationManager()
        self.validator = fua.BaseConfigValidator()
        self.iterations = 1000
    
    def test_configuration_validation_performance(self):
        """Test configuration validation performance"""
        configs = []
        for i in range(self.iterations):
            config = {
                'name': f'model_{i}',
                'architecture_type': 'cnn',
                'input_size': [70, 70],
                'parameters': 1000000 + i * 1000
            }
            configs.append(config)
        
        # Benchmark validation
        start_time = time.time()
        for config in configs:
            self.validator.validate_model_config(config)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / self.iterations
        
        print(f"\n📊 Configuration Validation Performance:")
        print(f"   Total time: {total_time:.4f}s")
        print(f"   Average time per validation: {avg_time:.6f}s")
        print(f"   Validations per second: {self.iterations / total_time:.2f}")
        
        # Performance assertion (should be fast)
        self.assertLess(avg_time, 0.001, "Configuration validation is too slow")
    
    def test_configuration_merge_performance(self):
        """Test configuration merging performance"""
        base_configs = []
        for i in range(self.iterations):
            config = {
                'training': {
                    'epochs': 100 + i,
                    'batch_size': 32,
                    'learning_rate': 0.001
                },
                'model': {
                    'name': f'model_{i}',
                    'layers': i + 10
                }
            }
            base_configs.append(config)
        
        # Benchmark merging
        start_time = time.time()
        for config in base_configs:
            merged = self.config_manager.merge_configs(config, {'additional': 'params'})
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / self.iterations
        
        print(f"\n📊 Configuration Merge Performance:")
        print(f"   Total time: {total_time:.4f}s")
        print(f"   Average time per merge: {avg_time:.6f}s")
        print(f"   Merges per second: {self.iterations / total_time:.2f}")
        
        # Performance assertion
        self.assertLess(avg_time, 0.001, "Configuration merging is too slow")
    
    def test_data_structure_performance(self):
        """Test data structure creation and serialization performance"""
        start_time = time.time()
        
        for i in range(self.iterations):
            # Create data structures
            capabilities = fua.ModelCapabilities(
                input_size_range=((60, 60), (80, 80)),
                recommended_batch_size=(16, 64),
                supported_optimizers=['adam', 'sgd'],
                supported_schedulers=['step', 'cosine'],
                special_preprocessing=['normalization'],
                memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
                computational_complexity='medium',
                training_time_estimate='medium'
            )
            
            metadata = fua.ModelMetadata(
                name=f'model_{i}',
                version='1.0.0',
                architecture_type='cnn',
                parameter_count=1000000,
                computational_complexity=1.5,
                memory_usage=2048,
                supported_input_sizes=[(70, 70)],
                performance_metrics={'accuracy': 0.9, 'f1_score': 0.85},
                training_history=[],
                creation_date=datetime.now(),
                last_modified=datetime.now(),
                author='test',
                tags=['test'],
                description='Test model'
            )
            
            # Test serialization
            capabilities_dict = capabilities.to_dict()
            metadata_dict = metadata.to_dict()
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / self.iterations
        
        print(f"\n📊 Data Structure Performance:")
        print(f"   Total time: {total_time:.4f}s")
        print(f"   Average time per creation: {avg_time:.6f}s")
        print(f"   Creations per second: {self.iterations / total_time:.2f}")
        
        # Performance assertion
        self.assertLess(avg_time, 0.001, "Data structure creation is too slow")


if __name__ == '__main__':
    unittest.main(verbosity=2)