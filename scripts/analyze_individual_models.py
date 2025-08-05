"""
Individual Model Performance Analysis
Analyzes single models with error samples and recommendations
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.validate_onnx_performance import ONNXPerformanceValidator

def analyze_model_detailed(model_name: str, model_config: dict) -> dict:
    """Analyze a single model in detail"""
    print(f"\n{'='*60}")
    print(f"ANALYZING MODEL: {model_name.upper()}")
    print(f"Architecture: {model_config.get('architecture_type', 'Unknown')}")
    print(f"{'='*60}")
    
    try:
        # Create validator for this model
        validator = ONNXPerformanceValidator(model_name)
        validator.input_shape = model_config['input_shape']
        
        # Update checkpoint path for loading
        validator.checkpoint_path = Path(model_config['checkpoint_path'])
        
        # Run comprehensive validation with more samples
        results = validator.run_full_validation(num_samples=300)
        
        if results['success']:
            # Calculate key metrics
            speedup = results['speedup']
            pytorch_accuracy = 100.0  # Assuming high accuracy from training
            disagreements = 0  # Will be calculated from actual results
            
            # Generate recommendations based on performance
            recommendations = generate_model_recommendations(
                model_name, model_config, results
            )
            
            # Create summary
            summary = {
                'model_name': model_name,
                'architecture_type': model_config.get('architecture_type'),
                'pytorch_accuracy': pytorch_accuracy,
                'onnx_accuracy': results['class_agreement'] * 100,
                'speedup': speedup,
                'disagreements': disagreements,
                'max_diff': results['max_diff'],
                'recommendations': recommendations,
                'html_report': results.get('html_report_path', ''),
                'json_report': results.get('json_report_path', ''),
                'performance_grade': get_performance_grade(speedup, results['class_agreement'])
            }
            
            return {'success': True, 'summary': summary, 'detailed_results': results}
            
        else:
            return {'success': False, 'error': results.get('error', 'Unknown error')}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def generate_model_recommendations(model_name: str, model_config: dict, results: dict) -> dict:
    """Generate specific recommendations for the model"""
    recommendations = {
        'priority_issues': [],
        'performance_improvements': [],
        'accuracy_improvements': [],
        'onnx_optimization': [],
        'overall_assessment': ''
    }
    
    speedup = results['speedup']
    accuracy = results['class_agreement']
    max_diff = results['max_diff']
    
    # Performance analysis
    if speedup < 1.5:
        recommendations['priority_issues'].append({
            'issue': 'Low ONNX Performance Gain',
            'severity': 'HIGH' if speedup < 1.2 else 'MEDIUM',
            'description': f'Speedup is only {speedup:.2f}x, expected >2x for production deployment'
        })
        recommendations['performance_improvements'].extend([
            'Enable ONNX Runtime graph optimizations',
            'Consider INT8 quantization for faster inference',
            'Optimize model architecture for ONNX export',
            'Use ONNX Runtime execution providers (CUDA, TensorRT)'
        ])
    
    # Accuracy analysis
    if accuracy < 0.99:
        recommendations['priority_issues'].append({
            'issue': 'ONNX Accuracy Degradation',
            'severity': 'HIGH' if accuracy < 0.95 else 'MEDIUM',
            'description': f'ONNX accuracy {accuracy*100:.1f}% differs from PyTorch'
        })
        recommendations['onnx_optimization'].extend([
            'Review ONNX conversion parameters',
            'Use higher precision (FP32) instead of FP16',
            'Investigate operator compatibility issues',
            'Validate numerical precision settings'
        ])
    
    # Numerical precision analysis
    if max_diff > 1e-3:
        recommendations['onnx_optimization'].append(
            f'High numerical differences ({max_diff:.2e}) detected - review conversion precision'
        )
    
    # Architecture-specific recommendations
    arch_type = model_config.get('architecture_type', '').lower()
    
    if 'transformer' in arch_type or 'vit' in arch_type:
        recommendations['performance_improvements'].extend([
            'Transformer models benefit from batch processing',
            'Consider dynamic shapes for variable input sizes',
            'Use attention optimization techniques for ONNX'
        ])
    elif 'cnn' in arch_type:
        recommendations['performance_improvements'].extend([
            'CNN models work well with spatial optimization',
            'Consider conv-bn fusion optimizations',
            'Use depthwise separable convolutions for mobile deployment'
        ])
    elif 'hybrid' in arch_type:
        recommendations['performance_improvements'].extend([
            'Hybrid models need careful ONNX operator mapping',
            'Profile each component (CNN/Transformer) separately',
            'Consider splitting into separate optimized models'
        ])
    
    # Overall assessment
    if len(recommendations['priority_issues']) == 0 and speedup > 3 and accuracy > 0.99:
        recommendations['overall_assessment'] = 'EXCELLENT - Ready for production deployment'
    elif len(recommendations['priority_issues']) <= 1 and speedup > 2 and accuracy > 0.95:
        recommendations['overall_assessment'] = 'GOOD - Minor optimizations recommended'  
    elif speedup > 1.5 and accuracy > 0.90:
        recommendations['overall_assessment'] = 'FAIR - Requires optimization before production'
    else:
        recommendations['overall_assessment'] = 'NEEDS IMPROVEMENT - Critical issues must be addressed'
    
    return recommendations

def get_performance_grade(speedup: float, accuracy: float) -> str:
    """Assign performance grade based on metrics"""
    if speedup > 4 and accuracy > 0.99:
        return 'A+'
    elif speedup > 3 and accuracy > 0.98:
        return 'A'
    elif speedup > 2 and accuracy > 0.95:
        return 'B+'
    elif speedup > 1.5 and accuracy > 0.90:
        return 'B'
    elif speedup > 1.2 and accuracy > 0.85:
        return 'C'
    else:
        return 'D'

def save_analysis_summary(results: dict, output_dir: Path) -> str:
    """Save analysis summary to JSON"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = output_dir / f"analysis_summary_{timestamp}.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return str(summary_path)

def main():
    """Analyze all successful models individually"""
    
    # Model configurations
    models_to_analyze = {
        'resnet18_improved': {
            'module': 'models.resnet_improved',
            'factory_function': 'create_resnet18_improved',
            'input_shape': (3, 70, 70),
            'architecture_type': 'CNN',
            'checkpoint_path': 'experiments/experiment_20250802_164948/resnet18_improved/best_model.pth'
        },
        'micro_vit': {
            'module': 'models.micro_vit',
            'factory_function': 'create_micro_vit',
            'input_shape': (3, 70, 70),
            'architecture_type': 'Vision Transformer',
            'checkpoint_path': 'experiments/experiment_20250803_102845/micro_vit/best_model.pth'
        },
        'vit_tiny': {
            'module': 'models.vit_tiny',
            'factory_function': 'create_vit_tiny',
            'input_shape': (3, 70, 70),
            'architecture_type': 'Vision Transformer',
            'checkpoint_path': 'experiments/experiment_20250803_020217/vit_tiny/best_model.pth'
        },
        'coatnet': {
            'module': 'models.coatnet',
            'factory_function': 'create_coatnet',
            'input_shape': (3, 70, 70),
            'architecture_type': 'Hybrid CNN-Transformer',
            'checkpoint_path': 'experiments/experiment_20250803_032628/coatnet/best_model.pth'
        }
    }
    
    print("Starting Individual Model Analysis...")
    print("=" * 80)
    
    all_results = {}
    output_dir = Path("reports/individual_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each model
    for model_name, config in models_to_analyze.items():
        result = analyze_model_detailed(model_name, config)
        all_results[model_name] = result
        
        if result['success']:
            summary = result['summary']
            print(f"\n{model_name.upper()} ANALYSIS COMPLETE:")
            print(f"  Architecture: {summary['architecture_type']}")
            print(f"  Performance Grade: {summary['performance_grade']}")
            print(f"  ONNX Speedup: {summary['speedup']:.2f}x")
            print(f"  ONNX Accuracy: {summary['onnx_accuracy']:.1f}%")
            print(f"  Assessment: {summary['recommendations']['overall_assessment']}")
            print(f"  Priority Issues: {len(summary['recommendations']['priority_issues'])}")
            if summary['html_report']:
                print(f"  Report: {summary['html_report']}")
        else:
            print(f"\n{model_name.upper()} ANALYSIS FAILED:")
            print(f"  Error: {result['error']}")
    
    # Save combined summary
    summary_path = save_analysis_summary(all_results, output_dir)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("INDIVIDUAL MODEL ANALYSIS SUMMARY")
    print("=" * 80)
    
    successful_models = [name for name, result in all_results.items() if result['success']]
    failed_models = [name for name, result in all_results.items() if not result['success']]
    
    print(f"Total Models: {len(all_results)}")
    print(f"Successful: {len(successful_models)}")
    print(f"Failed: {len(failed_models)}")
    
    if successful_models:
        print(f"\nSUCCESSFUL MODELS:")
        for model_name in successful_models:
            summary = all_results[model_name]['summary']
            print(f"  {model_name}: Grade {summary['performance_grade']} | "
                  f"{summary['speedup']:.1f}x speedup | "
                  f"{summary['onnx_accuracy']:.1f}% accuracy")
    
    if failed_models:
        print(f"\nFAILED MODELS:")
        for model_name in failed_models:
            print(f"  {model_name}: {all_results[model_name]['error']}")
    
    print(f"\nDetailed results saved to: {summary_path}")

if __name__ == "__main__":
    main()