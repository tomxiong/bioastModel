"""
Unit tests for FUA core data structures
"""

import unittest
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Import the classes we will test
from fua.core.data_structures import ModelCapabilities, ModelMetadata, Error, Improvement


class TestModelCapabilities(unittest.TestCase):
    """Test cases for ModelCapabilities data class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.capabilities = ModelCapabilities(
            input_size_range=((60, 60), (80, 80)),
            recommended_batch_size=(16, 64),
            supported_optimizers=['adam', 'sgd', 'rmsprop'],
            supported_schedulers=['cosine', 'step', 'plateau'],
            special_preprocessing=['bubble_detection', 'multi_scale'],
            memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
            computational_complexity='medium',
            training_time_estimate='medium'
        )
    
    def test_model_capabilities_creation(self):
        """Test ModelCapabilities instance creation"""
        self.assertIsInstance(self.capabilities, ModelCapabilities)
        self.assertEqual(self.capabilities.input_size_range, ((60, 60), (80, 80)))
        self.assertEqual(self.capabilities.recommended_batch_size, (16, 64))
        self.assertEqual(len(self.capabilities.supported_optimizers), 3)
    
    def test_model_capabilities_to_dict(self):
        """Test ModelCapabilities to_dict conversion"""
        result = self.capabilities.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['input_size_range'], ((60, 60), (80, 80)))
        self.assertEqual(result['computational_complexity'], 'medium')
    
    def test_model_capabilities_from_dict(self):
        """Test ModelCapabilities from_dict creation"""
        data = self.capabilities.to_dict()
        recreated = ModelCapabilities.from_dict(data)
        self.assertEqual(recreated.input_size_range, self.capabilities.input_size_range)
        self.assertEqual(recreated.recommended_batch_size, self.capabilities.recommended_batch_size)
        self.assertEqual(recreated.computational_complexity, self.capabilities.computational_complexity)


class TestModelMetadata(unittest.TestCase):
    """Test cases for ModelMetadata data class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metadata = ModelMetadata(
            name='test_model',
            version='1.0.0',
            architecture_type='cnn_transformer_hybrid',
            parameter_count=1420000,
            computational_complexity=2.5,
            memory_usage=2048,
            supported_input_sizes=[(70, 70), (80, 80)],
            performance_metrics={'accuracy': 0.98, 'f1_score': 0.97},
            training_history=[],
            creation_date=datetime.now(),
            last_modified=datetime.now(),
            author='test_author',
            tags=['hybrid', 'attention', 'cnn'],
            description='Test model for FUA implementation'
        )
    
    def test_model_metadata_creation(self):
        """Test ModelMetadata instance creation"""
        self.assertIsInstance(self.metadata, ModelMetadata)
        self.assertEqual(self.metadata.name, 'test_model')
        self.assertEqual(self.metadata.version, '1.0.0')
        self.assertEqual(self.metadata.parameter_count, 1420000)
    
    def test_model_metadata_to_dict(self):
        """Test ModelMetadata to_dict conversion"""
        result = self.metadata.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['name'], 'test_model')
        self.assertEqual(result['parameter_count'], 1420000)
    
    def test_model_metadata_from_dict(self):
        """Test ModelMetadata from_dict creation"""
        data = self.metadata.to_dict()
        recreated = ModelMetadata.from_dict(data)
        self.assertEqual(recreated.name, self.metadata.name)
        self.assertEqual(recreated.parameter_count, self.metadata.parameter_count)
        self.assertEqual(recreated.architecture_type, self.metadata.architecture_type)


class TestError(unittest.TestCase):
    """Test cases for Error data class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.error = Error(
            type='vanishing_gradients',
            severity='high',
            description='Vanishing gradients detected in training',
            model_name='test_model',
            timestamp=datetime.now(),
            metrics={'gradient_norm': 1e-9, 'layer': 'attention_3'},
            context={'epoch': 15, 'batch_size': 32}
        )
    
    def test_error_creation(self):
        """Test Error instance creation"""
        self.assertIsInstance(self.error, Error)
        self.assertEqual(self.error.type, 'vanishing_gradients')
        self.assertEqual(self.error.severity, 'high')
        self.assertEqual(self.error.model_name, 'test_model')
    
    def test_error_to_dict(self):
        """Test Error to_dict conversion"""
        result = self.error.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'vanishing_gradients')
        self.assertEqual(result['severity'], 'high')
        self.assertIn('timestamp', result)
    
    def test_error_severity_levels(self):
        """Test different severity levels"""
        from datetime import datetime
        low_severity = Error('test', 'low', 'test description', 'model', datetime.now())
        medium_severity = Error('test', 'medium', 'test description', 'model', datetime.now())
        high_severity = Error('test', 'high', 'test description', 'model', datetime.now())
        
        self.assertEqual(low_severity.severity, 'low')
        self.assertEqual(medium_severity.severity, 'medium')
        self.assertEqual(high_severity.severity, 'high')


class TestImprovement(unittest.TestCase):
    """Test cases for Improvement data class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.improvement = Improvement(
            type='increase_regularization',
            description='Increase L2 regularization to combat overfitting',
            implementation=lambda x: x * 2,  # Simple test implementation
            priority='high',
            expected_impact='medium',
            implementation_complexity='easy',
            estimated_time='hours',
            risk_level='low'
        )
    
    def test_improvement_creation(self):
        """Test Improvement instance creation"""
        self.assertIsInstance(self.improvement, Improvement)
        self.assertEqual(self.improvement.type, 'increase_regularization')
        self.assertEqual(self.improvement.priority, 'high')
        self.assertEqual(self.improvement.expected_impact, 'medium')
    
    def test_improvement_to_dict(self):
        """Test Improvement to_dict conversion"""
        result = self.improvement.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['type'], 'increase_regularization')
        self.assertEqual(result['priority'], 'high')
        self.assertEqual(result['implementation_complexity'], 'easy')
    
    def test_improvement_priority_levels(self):
        """Test different priority levels"""
        low_priority = Improvement('test', 'description', lambda x: x, 'low', 'medium', 'easy', 'hours', 'low')
        medium_priority = Improvement('test', 'description', lambda x: x, 'medium', 'medium', 'medium', 'hours', 'medium')
        high_priority = Improvement('test', 'description', lambda x: x, 'high', 'high', 'hard', 'days', 'high')
        
        self.assertEqual(low_priority.priority, 'low')
        self.assertEqual(medium_priority.priority, 'medium')
        self.assertEqual(high_priority.priority, 'high')


if __name__ == '__main__':
    unittest.main()