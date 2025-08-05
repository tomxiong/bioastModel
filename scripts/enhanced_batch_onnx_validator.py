"""
Enhanced Batch ONNX Conversion and Validation System
Converts all models and generates comprehensive JSON/HTML validation reports
"""

import os
import sys
import torch
import numpy as np
import onnxruntime as ort
import time
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import importlib
import traceback
from PIL import Image
import io

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.modern_onnx_converter_base import ModernONNXConverterBase, ModernConversionStrategy
from training.dataset import BioastDataset
from torchvision import transforms

class EnhancedBatchONNXValidator:
    """Enhanced batch ONNX conversion and validation system"""
    
    def __init__(self):
        self.input_shape = (3, 70, 70)
        self.results = {}
        self.start_time = datetime.now()
        
        # Model configurations with detailed metadata
        self.model_configs = {
            'resnet18_improved': {
                'module': 'models.resnet_improved',
                'factory_function': 'create_resnet18_improved',
                'input_shape': (3, 70, 70),
                'priority': 1,
                'expected_accuracy': 97.83,
                'architecture_type': 'CNN',
                'checkpoint_path': 'experiments/experiment_20250802_164948/resnet18_improved/best_model.pth'
            },
            'efficientnet_b0': {
                'module': 'models.efficientnet',
                'factory_function': 'create_efficientnet_b0',
                'input_shape': (3, 224, 224),
                'priority': 2,
                'expected_accuracy': 97.54,
                'architecture_type': 'CNN',
                'checkpoint_path': 'experiments/experiment_20250802_140818/efficientnet_b0/best_model.pth'
            },
            'mic_mobilenetv3': {
                'module': 'models.mic_mobilenetv3',
                'factory_function': 'create_mic_mobilenetv3',
                'input_shape': (3, 70, 70),
                'priority': 3,
                'expected_accuracy': 97.45,
                'architecture_type': 'Mobile CNN',
                'checkpoint_path': 'experiments/experiment_20250803_101438/mic_mobilenetv3/best_model.pth'
            },
            'micro_vit': {
                'module': 'models.micro_vit',
                'factory_function': 'create_micro_vit',
                'input_shape': (3, 70, 70),
                'priority': 4,
                'expected_accuracy': 97.36,
                'architecture_type': 'Vision Transformer',
                'requires_wrapper': True,
                'checkpoint_path': 'experiments/experiment_20250803_102845/micro_vit/best_model.pth'
            },
            'convnext_tiny': {
                'module': 'models.convnext_tiny',
                'factory_function': 'create_convnext_tiny',
                'input_shape': (3, 70, 70),
                'priority': 5,
                'expected_accuracy': None,
                'architecture_type': 'Modern CNN',
                'checkpoint_path': 'experiments/experiment_20250802_231639/convnext_tiny/best_model.pth'
            },
            'vit_tiny': {
                'module': 'models.vit_tiny',
                'factory_function': 'create_vit_tiny',
                'input_shape': (3, 70, 70),
                'priority': 6,
                'expected_accuracy': None,
                'architecture_type': 'Vision Transformer',
                'requires_wrapper': True,
                'checkpoint_path': 'experiments/experiment_20250803_020217/vit_tiny/best_model.pth'
            },
            'coatnet': {
                'module': 'models.coatnet',
                'factory_function': 'create_coatnet',
                'input_shape': (3, 70, 70),
                'priority': 7,
                'expected_accuracy': None,
                'architecture_type': 'Hybrid CNN-Transformer',
                'requires_wrapper': True,
                'checkpoint_path': 'experiments/experiment_20250803_032628/coatnet/best_model.pth'
            }
        }
    
    def create_generic_converter(self, model_name: str, config: Dict) -> ModernONNXConverterBase:
        """Create a generic modern converter for any model"""
        
        class GenericModernConverter(ModernONNXConverterBase):
            def __init__(self, model_name: str, config: Dict):
                super().__init__(model_name)
                self.input_shape = config['input_shape']
                self.config = config
                
                # Customize strategies based on model type
                if config.get('requires_wrapper', False):
                    # Conservative strategies for complex models
                    self.custom_strategies = [
                        ModernConversionStrategy(f"{model_name}_modern_static", 18, False, True, True, False),
                        ModernConversionStrategy(f"{model_name}_modern_dynamic", 18, True, True, True, False),
                        ModernConversionStrategy(f"{model_name}_compat_static", 17, False, True, True, False),
                        ModernConversionStrategy(f"{model_name}_legacy_high", 16, False, False, True, False),
                        ModernConversionStrategy(f"{model_name}_legacy_mid", 13, False, False, True, False),
                        ModernConversionStrategy(f"{model_name}_debug", 18, False, True, True, True),
                    ]
                else:
                    # Full modern strategies for standard models
                    self.custom_strategies = self.modern_strategies
            
            def find_model_checkpoint(self) -> Optional[Path]:
                """Find model checkpoint"""
                checkpoint_path = Path(self.config['checkpoint_path'])
                if checkpoint_path.exists():
                    self.log_message(f"Found checkpoint: {checkpoint_path}")
                    return checkpoint_path
                
                # Fallback to base class method
                return self.find_latest_checkpoint()
            
            def create_model_instance(self):
                """Create model instance"""
                try:
                    module = importlib.import_module(self.config['module'])
                    factory_func = getattr(module, self.config['factory_function'])
                    
                    # Handle different input shapes
                    if self.input_shape == (3, 224, 224):
                        model = factory_func(num_classes=2)  # EfficientNet uses standard 224x224
                    else:
                        model = factory_func(num_classes=2)  # Others use 70x70
                    
                    model.eval()
                    self.log_message(f"Created model using {self.config['factory_function']}")
                    return model
                except Exception as e:
                    self.log_message(f"Failed to create model: {e}", "ERROR")
                    return None
            
            def create_wrapper_if_needed(self, model):
                """Create ONNX wrapper if needed"""
                if not self.config.get('requires_wrapper', False):
                    return model
                
                class GenericONNXWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, x):
                        outputs = self.model(x)
                        
                        if isinstance(outputs, dict):
                            for key in ['classification', 'logits', 'output']:
                                if key in outputs:
                                    return outputs[key]
                            for value in outputs.values():
                                if hasattr(value, 'shape'):
                                    return value
                            return list(outputs.values())[0]
                        
                        if isinstance(outputs, (tuple, list)):
                            return outputs[0]
                        
                        return outputs
                
                wrapped_model = GenericONNXWrapper(model)
                wrapped_model.eval()
                self.log_message("Created ONNX wrapper")
                return wrapped_model
            
            def convert(self) -> bool:
                """Convert model to ONNX"""
                self.log_message(f"Starting modern conversion: {self.model_name}")
                
                # Find checkpoint
                checkpoint_path = self.find_model_checkpoint()
                if checkpoint_path is None:
                    return False
                
                # Create model
                model = self.create_model_instance()
                if model is None:
                    return False
                
                # Load weights
                loaded_model = self.load_model_safely(lambda: model, checkpoint_path)
                if loaded_model is None:
                    return False
                
                # Create wrapper if needed
                final_model = self.create_wrapper_if_needed(loaded_model)
                
                # Convert
                success, conversion_info = self.convert_with_modern_fallback(
                    final_model, self.input_shape, self.custom_strategies
                )
                
                # Save report
                self.save_modern_conversion_report(success, conversion_info)
                
                return success
        
        return GenericModernConverter(model_name, config)
    
    def prepare_test_data(self, input_shape: Tuple, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare balanced test data"""
        print(f"Preparing test data for input shape {input_shape}, samples: {num_samples}")
        
        try:
            transform = transforms.Compose([
                transforms.Resize(input_shape[1:]),  # Resize to match model input
                transforms.ToTensor(),
            ])
            
            dataset = BioastDataset(data_dir='bioast_dataset', split='test', transform=transform)
            print(f"Found {len(dataset)} samples in test dataset")
            
            # Balanced sampling
            negative_samples = []
            positive_samples = []
            
            for i, (data, target) in enumerate(dataset):
                if target == 0:
                    negative_samples.append((data.numpy(), target))
                else:
                    positive_samples.append((data.numpy(), target))
            
            samples_per_class = min(num_samples // 2, len(negative_samples), len(positive_samples))
            
            selected_samples = []
            selected_labels = []
            
            # Add samples from each class
            for i in range(samples_per_class):
                data, label = negative_samples[i]
                selected_samples.append(data)
                selected_labels.append(label)
            
            for i in range(samples_per_class):
                data, label = positive_samples[i]
                selected_samples.append(data)
                selected_labels.append(label)
            
            if selected_samples:
                test_data = np.stack(selected_samples, axis=0)
                test_labels = np.array(selected_labels)
                
                print(f"Loaded balanced test data - Negative: {np.sum(test_labels == 0)}, Positive: {np.sum(test_labels == 1)}")
                return test_data, test_labels
            
        except Exception as e:
            print(f"Cannot load real data: {e}")
            traceback.print_exc()
        
        # Fallback to random data
        test_data = np.random.randn(num_samples, *input_shape).astype(np.float32)
        test_labels = np.random.randint(0, 2, num_samples)
        print(f"Using random test data, shape: {test_data.shape}")
        
        return test_data, test_labels
    
    def convert_single_model(self, model_name: str) -> Dict[str, Any]:
        """Convert and validate a single model"""
        print(f"\n{'='*60}")
        print(f"Converting and validating model: {model_name}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        try:
            config = self.model_configs[model_name]
            
            # Step 1: Convert to ONNX
            print(f"Step 1: Converting {model_name} to ONNX...")
            converter = self.create_generic_converter(model_name, config)
            conversion_success = converter.convert()
            
            if not conversion_success:
                return {
                    'model_name': model_name,
                    'success': False,
                    'error': 'ONNX conversion failed',
                    'duration': (datetime.now() - start_time).total_seconds(),
                    'config': config
                }
            
            # Step 2: Validate performance
            print(f"Step 2: Validating {model_name} performance...")
            validation_result = self.validate_model_performance(model_name, config)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'model_name': model_name,
                'success': True,
                'conversion_success': conversion_success,
                'validation_result': validation_result,
                'duration': duration,
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Model {model_name} completed successfully in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error processing {model_name}: {str(e)}"
            print(f"Error: {error_msg}")
            traceback.print_exc()
            
            return {
                'model_name': model_name,
                'success': False,
                'error': error_msg,
                'duration': duration,
                'config': self.model_configs.get(model_name, {}),
                'timestamp': datetime.now().isoformat()
            }
    
    def validate_model_performance(self, model_name: str, config: Dict) -> Dict[str, Any]:
        """Validate model performance using enhanced validator"""
        try:
            from scripts.validate_onnx_performance import ONNXPerformanceValidator
            
            # Create enhanced validator
            validator = ONNXPerformanceValidator(model_name)
            validator.input_shape = config['input_shape']
            
            # Run validation with real test data
            results = validator.run_full_validation(num_samples=200)
            
            return results
            
        except Exception as e:
            print(f"Performance validation failed for {model_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def convert_all_models(self) -> Dict[str, Any]:
        """Convert and validate all models"""
        print(f"\nStarting batch ONNX conversion and validation...")
        print(f"Models to process: {list(self.model_configs.keys())}")
        
        # Sort by priority
        sorted_models = sorted(
            self.model_configs.keys(),
            key=lambda x: self.model_configs[x]['priority']
        )
        
        # Process each model
        for model_name in sorted_models:
            self.results[model_name] = self.convert_single_model(model_name)
        
        # Generate summary
        total_time = (datetime.now() - self.start_time).total_seconds()
        successful = sum(1 for r in self.results.values() if r['success'])
        failed = len(self.results) - successful
        
        summary = {
            'total_time': total_time,
            'total_models': len(self.results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(self.results) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{'='*60}")
        print(f"Batch processing completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Success rate: {successful}/{len(self.results)} ({summary['success_rate']:.1f}%)")
        print(f"{'='*60}")
        
        return {
            'summary': summary,
            'results': self.results
        }

def main():
    """Main function"""
    validator = EnhancedBatchONNXValidator()
    
    # Convert and validate all models
    batch_results = validator.convert_all_models()
    
    # Save results
    output_dir = Path("reports/batch_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = output_dir / f"batch_validation_{timestamp}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {json_path}")

if __name__ == "__main__":
    main()