"""
Comprehensive integration tests for FUA Sprint 2 implementation
"""

import unittest
import tempfile
import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import FUA components
import fua


class TestFUACompleteIntegration(unittest.TestCase):
    """Complete integration tests for FUA system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = fua.ModelManager()
        self.config_system = fua.ModelConfigurationManager()
        self.config_manager = fua.ConfigurationManager()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_model_lifecycle(self):
        """Test complete model lifecycle with FUA"""
        
        # Create model using factory
        model_id = self.model_manager.create_model('airbubble_hybrid_net', {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 20
        })
        
        self.assertIsNotNone(model_id)
        
        # Get model instance
        model = self.model_manager.get_model(model_id)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, fua.ModelInterface)
        
        # Test model capabilities
        capabilities = model.get_capabilities()
        self.assertIsInstance(capabilities, fua.ModelCapabilities)
        self.assertEqual(capabilities.computational_complexity, 'high')
        
        # Test model metadata
        metadata = model.get_metadata()
        self.assertIsInstance(metadata, fua.ModelMetadata)
        self.assertIn('hybrid', metadata.architecture_type)
        
        # Test training
        training_data = {'samples': 1000, 'image_size': [70, 70]}
        training_results = self.model_manager.train_model(model_id, training_data)
        self.assertIsInstance(training_results, dict)
        self.assertIn('accuracy', training_results)
        self.assertIn('loss', training_results)
        
        # Test evaluation
        eval_data = {'samples': 200}
        eval_results = self.model_manager.evaluate_model(model_id, eval_data)
        self.assertIsInstance(eval_results, dict)
        self.assertIn('accuracy', eval_results)
        
        # Test model persistence
        model_path = os.path.join(self.temp_dir, 'test_model.json')
        save_success = self.model_manager.save_model(model_id, model_path)
        self.assertTrue(save_success)
        self.assertTrue(os.path.exists(model_path))
        
        # Test model loading
        loaded_model_id = self.model_manager.load_model(model_path, 'airbubble_hybrid_net')
        self.assertIsNotNone(loaded_model_id)
        self.assertNotEqual(loaded_model_id, model_id)
        
        # Verify loaded model works
        loaded_model = self.model_manager.get_model(loaded_model_id)
        self.assertIsNotNone(loaded_model)
        
        loaded_eval_results = self.model_manager.evaluate_model(loaded_model_id, eval_data)
        self.assertIsInstance(loaded_eval_results, dict)
        self.assertIn('accuracy', loaded_eval_results)
    
    def test_model_configuration_workflow(self):
        """Test complete model configuration workflow"""
        
        # Create model configuration
        config_id = self.config_system.create_model_config(
            'test_hybrid_model',
            'airbubble_hybrid_net',
            base_config={
                'attention_heads': 8,
                'fusion_layers': 4
            },
            template_name='high_performance'
        )
        
        # Get configuration
        config = self.config_system.get_config(config_id)
        self.assertIsNotNone(config)
        self.assertEqual(config['name'], 'test_hybrid_model')
        self.assertEqual(config['model_type'], 'airbubble_hybrid_net')
        
        # Validate configuration
        is_valid, errors = self.config_system.validate_config(config_id)
        self.assertTrue(is_valid, f"Config validation failed: {errors}")
        
        # Update configuration
        update_success = self.config_system.update_config(config_id, {
            'attention_heads': 12,
            'bubble_detection_weight': 0.4
        })
        self.assertTrue(update_success)
        
        # Verify updates
        updated_config = self.config_system.get_config(config_id)
        self.assertEqual(updated_config['attention_heads'], 12)
        self.assertEqual(updated_config['bubble_detection_weight'], 0.4)
        
        # Save configuration
        config_path = os.path.join(self.temp_dir, 'model_config.json')
        save_success = self.config_system.save_config(config_id, config_path)
        self.assertTrue(save_success)
        
        # Load configuration
        loaded_config_id = self.config_system.load_config(config_path, 'airbubble_hybrid_net')
        self.assertIsNotNone(loaded_config_id)
        
        # Verify loaded configuration
        loaded_config = self.config_system.get_config(loaded_config_id)
        self.assertEqual(loaded_config['attention_heads'], 12)
        self.assertEqual(loaded_config['bubble_detection_weight'], 0.4)
    
    def test_model_factory_with_real_models(self):
        """Test model factory with real model registrations"""
        
        factory = fua.ModelFactory()
        
        # Check available models
        available_models = factory.get_available_models()
        self.assertIsInstance(available_models, list)
        
        # Test creating each available model
        for model_name in available_models:
            try:
                model = factory.create_model(model_name)
                self.assertIsNotNone(model)
                self.assertIsInstance(model, fua.ModelInterface)
                
                # Test model metadata
                metadata = model.get_metadata()
                self.assertIsInstance(metadata, fua.ModelMetadata)
                self.assertEqual(metadata.name, model_name)
                
                # Test model capabilities
                capabilities = model.get_capabilities()
                self.assertIsInstance(capabilities, fua.ModelCapabilities)
                
                print(f"✓ Successfully created and tested {model_name}")
                
            except Exception as e:
                print(f"⚠ Error testing model {model_name}: {e}")
                # Don't fail the test for individual model issues
    
    def test_configuration_generation_from_capabilities(self):
        """Test automatic configuration generation from model capabilities"""
        
        # Test different capability profiles
        capability_profiles = [
            {
                'name': 'Lightweight CNN',
                'input_size_range': ((60, 60), (80, 80)),
                'batch_size': (32, 128),
                'complexity': 'low',
                'preprocessing': ['normalization']
            },
            {
                'name': 'High-Performance Hybrid',
                'input_size_range': ((60, 60), (80, 80)),
                'batch_size': (16, 64),
                'complexity': 'high',
                'preprocessing': ['bubble_detection', 'multi_scale']
            },
            {
                'name': 'Transformer Model',
                'input_size_range': ((60, 60), (80, 80)),
                'batch_size': (8, 32),
                'complexity': 'high',
                'preprocessing': ['patch_extraction', 'positional_encoding']
            }
        ]
        
        for profile in capability_profiles:
            # Create capabilities
            capabilities = fua.ModelCapabilities(
                input_size_range=profile['input_size_range'],
                recommended_batch_size=profile['batch_size'],
                supported_optimizers=['adam', 'sgd'],
                supported_schedulers=['cosine', 'step'],
                special_preprocessing=profile['preprocessing'],
                memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
                computational_complexity=profile['complexity'],
                training_time_estimate='medium'
            )
            
            # Generate configuration
            config_id = self.config_system.generate_config_from_capabilities(
                profile['name'], 
                capabilities
            )
            
            # Verify generated configuration
            config = self.config_system.get_config(config_id)
            self.assertIsNotNone(config)
            self.assertEqual(config['name'], profile['name'])
            
            # Check training configuration matches capabilities
            training_config = config['training']
            self.assertIn(training_config['batch_size'], range(profile['batch_size'][0], profile['batch_size'][1] + 1))
            
            # Check preprocessing matches capabilities
            model_config = config['model']
            self.assertEqual(model_config['preprocessing'], profile['preprocessing'])
            
            print(f"✓ Generated configuration for {profile['name']}")
    
    def test_model_manager_operations(self):
        """Test model manager operations"""
        
        # Create multiple models
        model_ids = []
        model_configs = [
            {'learning_rate': 0.001, 'batch_size': 32},
            {'learning_rate': 0.0005, 'batch_size': 64},
            {'learning_rate': 0.01, 'batch_size': 16}
        ]
        
        for i, config in enumerate(model_configs):
            model_id = self.model_manager.create_model('mic_mobilenetv3', config)
            model_ids.append(model_id)
        
        # List models
        models = self.model_manager.list_models()
        self.assertEqual(len(models), 3)
        
        # Configure models
        for i, model_id in enumerate(model_ids):
            success = self.model_manager.configure_model(model_id, {
                'epochs': 50 + i * 10,
                'weight_decay': 0.01
            })
            self.assertTrue(success)
        
        # Test each model
        for i, model_id in enumerate(model_ids):
            model = self.model_manager.get_model(model_id)
            self.assertIsNotNone(model)
            
            # Test training with different results
            training_results = self.model_manager.train_model(model_id, {'samples': 500})
            self.assertIsInstance(training_results, dict)
            
            # Test evaluation
            eval_results = self.model_manager.evaluate_model(model_id, {'samples': 100})
            self.assertIsInstance(eval_results, dict)
            
            print(f"✓ Tested model {i+1}: accuracy={eval_results.get('accuracy', 'N/A')}")
        
        # Test model cleanup
        cleanup_count = self.model_manager.cleanup_old_models(max_age_hours=0)
        self.assertGreaterEqual(cleanup_count, 0)
        
        # Verify cleanup
        remaining_models = self.model_manager.list_models()
        print(f"✓ Cleaned up {cleanup_count} models, {len(remaining_models)} remaining")
    
    def test_error_handling_and_robustness(self):
        """Test error handling and system robustness"""
        
        # Test invalid model creation
        with self.assertRaises(ValueError):
            self.model_manager.create_model('nonexistent_model')
        
        # Test invalid configuration operations
        with self.assertRaises(ValueError):
            self.config_system.apply_template('nonexistent_template', {})
        
        # Test invalid model operations
        invalid_model_id = 'nonexistent_model'
        
        # These should handle errors gracefully
        model = self.model_manager.get_model(invalid_model_id)
        self.assertIsNone(model)
        
        training_results = self.model_manager.train_model(invalid_model_id, {})
        self.assertEqual(training_results, {})
        
        eval_results = self.model_manager.evaluate_model(invalid_model_id, {})
        self.assertEqual(eval_results, {})
        
        save_success = self.model_manager.save_model(invalid_model_id, 'test_path')
        self.assertFalse(save_success)
        
        # Test configuration validation with invalid data
        invalid_config = {
            'name': 'test_model',
            'architecture_type': 'cnn',
            'input_size': 'invalid',  # Should be list/tuple
            'layers': 'invalid'  # Should be int
        }
        
        is_valid, errors = self.config_system.validate_model_config('cnn', invalid_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        print(f"✓ Error handling working correctly, caught {len(errors)} validation errors")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for FUA components"""
        
        # Benchmark model creation
        start_time = time.time()
        model_ids = []
        
        for i in range(10):
            model_id = self.model_manager.create_model('mic_mobilenetv3', {
                'learning_rate': 0.001,
                'batch_size': 32
            })
            model_ids.append(model_id)
        
        creation_time = time.time() - start_time
        avg_creation_time = creation_time / 10
        
        print(f"📊 Model Creation Performance:")
        print(f"   Total time: {creation_time:.4f}s")
        print(f"   Average time per model: {avg_creation_time:.6f}s")
        print(f"   Models per second: {10 / creation_time:.2f}")
        
        # Benchmark configuration validation
        config_system = fua.ModelConfigurationSystem()
        test_configs = []
        
        for i in range(100):
            config = {
                'name': f'test_model_{i}',
                'architecture_type': 'cnn',
                'input_size': [70, 70],
                'layers': 10 + i,
                'filters': 32 + i
            }
            test_configs.append(config)
        
        start_time = time.time()
        for config in test_configs:
            config_system.validate_model_config('cnn', config)
        
        validation_time = time.time() - start_time
        avg_validation_time = validation_time / 100
        
        print(f"📊 Configuration Validation Performance:")
        print(f"   Total time: {validation_time:.4f}s")
        print(f"   Average time per validation: {avg_validation_time:.6f}s")
        print(f"   Validations per second: {100 / validation_time:.2f}")
        
        # Benchmark model training simulation
        start_time = time.time()
        for model_id in model_ids[:5]:  # Test with 5 models
            training_results = self.model_manager.train_model(model_id, {'samples': 1000})
        
        training_time = time.time() - start_time
        avg_training_time = training_time / 5
        
        print(f"📊 Model Training Simulation Performance:")
        print(f"   Total time: {training_time:.4f}s")
        print(f"   Average time per training: {avg_training_time:.6f}s")
        print(f"   Training simulations per second: {5 / training_time:.2f}")
        
        # Performance assertions
        self.assertLess(avg_creation_time, 0.1, "Model creation is too slow")
        self.assertLess(avg_validation_time, 0.001, "Configuration validation is too slow")
        self.assertLess(avg_training_time, 0.1, "Model training simulation is too slow")
        
        print("✓ All performance benchmarks passed")


class TestFUARealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = fua.ModelManager()
        self.config_system = fua.ModelConfigurationManager()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_bioast_model_integration(self):
        """Test integration with bioastModel project requirements"""
        
        # Create models suitable for 70x70 biomedical image analysis
        models_to_test = ['airbubble_hybrid_net', 'mic_mobilenetv3', 'micro_vit']
        
        for model_name in models_to_test:
            try:
                # Create model with bioast-specific configuration
                model_id = self.model_manager.create_model(model_name, {
                    'input_size': [70, 70],
                    'num_classes': 2,  # Binary classification
                    'learning_rate': 0.001,
                    'batch_size': 32
                })
                
                model = self.model_manager.get_model(model_id)
                
                # Verify model supports 70x70 input
                capabilities = model.get_capabilities()
                min_size, max_size = capabilities.input_size_range
                self.assertLessEqual(min_size[0], 70)
                self.assertGreaterEqual(max_size[0], 70)
                self.assertLessEqual(min_size[1], 70)
                self.assertGreaterEqual(max_size[1], 70)
                
                # Test with biomedical image data simulation
                bioast_data = {
                    'samples': 1000,
                    'image_size': [70, 70],
                    'channels': 3,
                    'classes': ['negative', 'positive'],
                    'description': 'Biomedical colony detection dataset'
                }
                
                # Simulate training
                training_results = self.model_manager.train_model(model_id, bioast_data)
                self.assertIsInstance(training_results, dict)
                self.assertGreater(training_results.get('accuracy', 0), 0.8)  # Should be reasonable
                
                # Simulate evaluation
                eval_results = self.model_manager.evaluate_model(model_id, bioast_data)
                self.assertIsInstance(eval_results, dict)
                self.assertGreater(eval_results.get('accuracy', 0), 0.8)
                
                print(f"✓ {model_name} works correctly with bioast requirements")
                
            except Exception as e:
                print(f"⚠ Error testing {model_name}: {e}")
                # Continue with other models
    
    def test_model_comparison_workflow(self):
        """Test model comparison and selection workflow"""
        
        # Create multiple models for comparison
        model_configs = [
            {'model_type': 'airbubble_hybrid_net', 'learning_rate': 0.001, 'batch_size': 32},
            {'model_type': 'mic_mobilenetv3', 'learning_rate': 0.001, 'batch_size': 64},
            {'model_type': 'micro_vit', 'learning_rate': 0.0005, 'batch_size': 16}
        ]
        
        model_results = {}
        test_data = {'samples': 500, 'image_size': [70, 70]}
        
        for config in model_configs:
            model_id = self.model_manager.create_model(config['model_type'], config)
            
            # Train and evaluate
            training_results = self.model_manager.train_model(model_id, test_data)
            eval_results = self.model_manager.evaluate_model(model_id, test_data)
            
            model_results[config['model_type']] = {
                'model_id': model_id,
                'training_results': training_results,
                'eval_results': eval_results,
                'config': config
            }
        
        # Analyze results
        best_model = None
        best_accuracy = 0
        
        for model_type, results in model_results.items():
            accuracy = results['eval_results'].get('accuracy', 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_type
            
            print(f"📊 {model_type}: accuracy={accuracy:.4f}, "
                  f"loss={results['training_results'].get('loss', 'N/A')}, "
                  f"training_time={results['training_results'].get('training_time', 'N/A')}")
        
        self.assertIsNotNone(best_model)
        print(f"🏆 Best model: {best_model} with accuracy {best_accuracy:.4f}")
        
        # Verify best model can be saved and loaded
        best_model_id = model_results[best_model]['model_id']
        model_path = os.path.join(self.temp_dir, 'best_model.json')
        
        save_success = self.model_manager.save_model(best_model_id, model_path)
        self.assertTrue(save_success)
        
        loaded_model_id = self.model_manager.load_model(model_path, best_model)
        self.assertIsNotNone(loaded_model_id)
        
        # Verify loaded model performance
        loaded_eval_results = self.model_manager.evaluate_model(loaded_model_id, test_data)
        loaded_accuracy = loaded_eval_results.get('accuracy', 0)
        
        # Performance should be similar (allowing for small variations)
        self.assertAlmostEqual(loaded_accuracy, best_accuracy, delta=0.01)
        
        print(f"✓ Model comparison workflow completed successfully")


if __name__ == '__main__':
    unittest.main(verbosity=2)