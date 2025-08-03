#!/usr/bin/env python3
"""
Individual Model Analysis Report Generator

This script generates detailed individual analysis reports for each of the 8 trained models,
similar to the EfficientNet-B0 analysis format but more comprehensive.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def find_latest_experiment(model_name: str) -> Optional[Path]:
    """Find the latest experiment directory for a model."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None
    
    model_experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            model_dir = exp_dir / model_name
            if model_dir.exists():
                model_experiments.append(exp_dir)
    
    if not model_experiments:
        return None
    
    latest_exp = max(model_experiments, key=lambda x: x.stat().st_ctime)
    return latest_exp / model_name

def load_model_data(model_name: str) -> Optional[Dict[str, Any]]:
    """Load comprehensive model data."""
    model_dir = find_latest_experiment(model_name)
    if not model_dir:
        return None
    
    # Load training history
    history_file = model_dir / "training_history.json"
    if not history_file.exists():
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Load evaluation results if available
    eval_file = model_dir / "evaluation" / "test_results.json"
    evaluation = None
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            evaluation = json.load(f)
    
    # Load interpretability results if available
    interp_dir = Path("reports/interpretability") / model_name
    interpretability = None
    if interp_dir.exists():
        gradcam_file = interp_dir / "gradcam" / "gradcam_results.json"
        if gradcam_file.exists():
            with open(gradcam_file, 'r') as f:
                interpretability = json.load(f)
    
    return {
        'model_name': model_name,
        'history': history,
        'evaluation': evaluation,
        'interpretability': interpretability,
        'experiment_path': str(model_dir)
    }

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model architecture and parameter information."""
    model_info = {
        'efficientnet_b0': {
            'architecture': 'EfficientNet-B0',
            'type': 'CNN',
            'parameters': 1_560_000,
            'description': 'Efficient convolutional neural network with compound scaling',
            'key_features': ['Compound scaling', 'Mobile-optimized blocks', 'Squeeze-and-excitation'],
            'use_cases': ['General purpose', 'Mobile deployment', 'Transfer learning base']
        },
        'resnet18_improved': {
            'architecture': 'ResNet-18 Improved',
            'type': 'CNN',
            'parameters': 11_260_000,
            'description': 'Enhanced ResNet-18 with improved residual connections',
            'key_features': ['Residual connections', 'Batch normalization', 'Skip connections'],
            'use_cases': ['High accuracy requirements', 'Feature extraction', 'Fine-tuning']
        },
        'convnext_tiny': {
            'architecture': 'ConvNeXt-Tiny',
            'type': 'CNN',
            'parameters': 28_600_000,
            'description': 'Modern CNN architecture inspired by Vision Transformers',
            'key_features': ['Large kernel convolutions', 'Layer normalization', 'GELU activation'],
            'use_cases': ['State-of-the-art accuracy', 'Research applications', 'High-end deployment']
        },
        'vit_tiny': {
            'architecture': 'Vision Transformer Tiny',
            'type': 'Transformer',
            'parameters': 2_720_000,
            'description': 'Lightweight Vision Transformer for image classification',
            'key_features': ['Self-attention mechanism', 'Patch embeddings', 'Position encoding'],
            'use_cases': ['Global context modeling', 'Attention visualization', 'Transformer research']
        },
        'coatnet': {
            'architecture': 'CoAtNet',
            'type': 'Hybrid CNN-Transformer',
            'parameters': 26_042_722,
            'description': 'Hybrid architecture combining convolution and attention',
            'key_features': ['Conv-attention fusion', 'Multi-scale processing', 'Hybrid design'],
            'use_cases': ['Best of both worlds', 'Complex pattern recognition', 'Research applications']
        },
        'mic_mobilenetv3': {
            'architecture': 'MIC MobileNetV3',
            'type': 'Enhanced CNN',
            'parameters': 1_138_137,
            'description': 'MIC-specialized lightweight network with SE attention',
            'key_features': ['MIC optimization', 'SE attention', 'Ultra-lightweight', 'Domain-specific'],
            'use_cases': ['MIC testing', 'Edge deployment', 'Real-time processing']
        },
        'micro_vit': {
            'architecture': 'Micro Vision Transformer',
            'type': 'Enhanced Transformer',
            'parameters': 3_265_402,
            'description': 'Micro-sized ViT optimized for small images',
            'key_features': ['Micro-scale design', 'Efficient attention', 'Small image optimization'],
            'use_cases': ['Small image analysis', 'Efficient transformers', 'MIC applications']
        },
        'airbubble_hybrid_net': {
            'architecture': 'AirBubble Hybrid Network',
            'type': 'Specialized Hybrid',
            'parameters': 750_142,
            'description': 'Specialized network for air bubble detection and correction',
            'key_features': ['Air bubble detection', 'Optical distortion correction', 'Multi-task learning'],
            'use_cases': ['MIC testing with bubbles', 'Quality control', 'Specialized analysis']
        }
    }
    
    return model_info.get(model_name, {
        'architecture': model_name,
        'type': 'Unknown',
        'parameters': 0,
        'description': 'Model information not available',
        'key_features': [],
        'use_cases': []
    })

def analyze_training_performance(history: Dict[str, List]) -> Dict[str, Any]:
    """Analyze training performance metrics."""
    if not history or 'val_acc' not in history:
        return {}
    
    val_acc = history['val_acc']
    train_acc = history['train_acc']
    val_loss = history['val_loss']
    train_loss = history['train_loss']
    
    analysis = {
        'best_val_accuracy': max(val_acc),
        'best_val_epoch': val_acc.index(max(val_acc)) + 1,
        'final_val_accuracy': val_acc[-1],
        'final_train_accuracy': train_acc[-1],
        'total_epochs': len(val_acc),
        'convergence_epoch': None,
        'overfitting_detected': False,
        'training_stability': 'stable'
    }
    
    # Detect convergence (when validation accuracy stops improving significantly)
    for i in range(5, len(val_acc)):
        recent_improvement = max(val_acc[i-5:i]) - max(val_acc[i:i+5]) if i+5 < len(val_acc) else 0
        if recent_improvement < 0.001:  # Less than 0.1% improvement
            analysis['convergence_epoch'] = i + 1
            break
    
    # Detect overfitting (train acc much higher than val acc)
    if len(train_acc) > 0:
        final_gap = train_acc[-1] - val_acc[-1]
        if final_gap > 0.05:  # 5% gap indicates potential overfitting
            analysis['overfitting_detected'] = True
    
    # Assess training stability
    if len(val_acc) > 10:
        val_acc_std = np.std(val_acc[-10:])  # Standard deviation of last 10 epochs
        if val_acc_std > 0.02:
            analysis['training_stability'] = 'unstable'
        elif val_acc_std < 0.005:
            analysis['training_stability'] = 'very_stable'
    
    return analysis

def generate_performance_insights(model_name: str, analysis: Dict[str, Any], model_info: Dict[str, Any]) -> str:
    """Generate performance insights and recommendations."""
    insights = []
    
    # Accuracy insights
    best_acc = analysis.get('best_val_accuracy', 0)
    if best_acc > 0.98:
        insights.append("🏆 **Excellent Performance**: Achieved >98% validation accuracy, indicating strong model capability.")
    elif best_acc > 0.95:
        insights.append("✅ **Good Performance**: Achieved >95% validation accuracy, suitable for most applications.")
    else:
        insights.append("⚠️ **Moderate Performance**: Consider model tuning or architecture improvements.")
    
    # Training efficiency insights
    total_epochs = analysis.get('total_epochs', 0)
    convergence_epoch = analysis.get('convergence_epoch')
    if convergence_epoch and convergence_epoch < total_epochs * 0.5:
        insights.append(f"⚡ **Fast Convergence**: Model converged quickly at epoch {convergence_epoch}, indicating efficient training.")
    elif total_epochs > 30:
        insights.append("🐌 **Slow Convergence**: Model required many epochs to train, consider learning rate adjustment.")
    
    # Overfitting insights
    if analysis.get('overfitting_detected'):
        insights.append("⚠️ **Overfitting Detected**: Training accuracy significantly higher than validation accuracy. Consider regularization.")
    else:
        insights.append("✅ **Good Generalization**: No significant overfitting detected.")
    
    # Stability insights
    stability = analysis.get('training_stability', 'stable')
    if stability == 'very_stable':
        insights.append("📈 **Very Stable Training**: Consistent performance throughout training.")
    elif stability == 'unstable':
        insights.append("📉 **Unstable Training**: High variance in validation accuracy. Consider reducing learning rate.")
    
    # Architecture-specific insights
    arch_type = model_info.get('type', '')
    if arch_type == 'CNN' and best_acc > 0.98:
        insights.append("🔍 **CNN Excellence**: Demonstrates strong convolutional feature extraction capabilities.")
    elif arch_type == 'Transformer' and best_acc > 0.96:
        insights.append("🎯 **Transformer Success**: Shows effective attention mechanism for this task.")
    elif 'Enhanced' in arch_type and best_acc > 0.98:
        insights.append("🚀 **Enhanced Architecture Success**: Domain-specific optimizations proved highly effective.")
    
    return "\n".join(f"- {insight}" for insight in insights)

def generate_individual_report(model_name: str) -> str:
    """Generate comprehensive individual model report."""
    print(f"   Generating report for {model_name}...")
    
    # Load model data
    model_data = load_model_data(model_name)
    if not model_data:
        return f"# {model_name} Analysis Report\n\n❌ **Error**: Could not load model data.\n"
    
    model_info = get_model_info(model_name)
    history = model_data.get('history', {})
    evaluation = model_data.get('evaluation', {})
    interpretability = model_data.get('interpretability', {})
    
    # Analyze performance
    performance_analysis = analyze_training_performance(history)
    
    # Generate report
    report = f"""# {model_info['architecture']} Analysis Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report provides comprehensive analysis for the **{model_info['architecture']}** model, including training performance, architectural insights, and deployment recommendations.

### Key Metrics
- **Best Validation Accuracy**: {performance_analysis.get('best_val_accuracy', 0):.4f}
- **Model Parameters**: {model_info['parameters']:,}
- **Architecture Type**: {model_info['type']}
- **Training Epochs**: {performance_analysis.get('total_epochs', 0)}

## Model Architecture

### Overview
{model_info['description']}

### Key Features
"""
    
    for feature in model_info['key_features']:
        report += f"- **{feature}**\n"
    
    report += f"""
### Technical Specifications
- **Parameters**: {model_info['parameters']:,}
- **Architecture Type**: {model_info['type']}
- **Memory Footprint**: {model_info['parameters'] * 4 / 1024 / 1024:.1f} MB (FP32)
- **Efficiency Ratio**: {performance_analysis.get('best_val_accuracy', 0) / (model_info['parameters'] / 1e6):.3f} (Accuracy per Million Parameters)

## Training Performance Analysis

### Performance Metrics
"""
    
    if performance_analysis:
        report += f"""- **Best Validation Accuracy**: {performance_analysis['best_val_accuracy']:.4f} (Epoch {performance_analysis['best_val_epoch']})
- **Final Validation Accuracy**: {performance_analysis['final_val_accuracy']:.4f}
- **Final Training Accuracy**: {performance_analysis['final_train_accuracy']:.4f}
- **Total Training Epochs**: {performance_analysis['total_epochs']}
- **Training Stability**: {performance_analysis['training_stability'].replace('_', ' ').title()}
"""
        
        if performance_analysis.get('convergence_epoch'):
            report += f"- **Convergence Epoch**: {performance_analysis['convergence_epoch']}\n"
    
    report += f"""
### Performance Insights
{generate_performance_insights(model_name, performance_analysis, model_info)}

## Training History Analysis

"""
    
    if history and 'val_acc' in history:
        val_acc = history['val_acc']
        train_acc = history['train_acc']
        
        # Calculate training statistics
        max_val_acc = max(val_acc)
        min_val_acc = min(val_acc)
        acc_range = max_val_acc - min_val_acc
        
        report += f"""### Accuracy Progression
- **Validation Accuracy Range**: {min_val_acc:.4f} - {max_val_acc:.4f} (Range: {acc_range:.4f})
- **Training Accuracy Range**: {min(train_acc):.4f} - {max(train_acc):.4f}
- **Final Accuracy Gap**: {train_acc[-1] - val_acc[-1]:.4f} (Train - Validation)

### Learning Characteristics
"""
        
        # Analyze learning phases
        if len(val_acc) >= 10:
            early_phase = np.mean(val_acc[:5])
            late_phase = np.mean(val_acc[-5:])
            improvement = late_phase - early_phase
            
            report += f"- **Early Phase Accuracy** (First 5 epochs): {early_phase:.4f}\n"
            report += f"- **Late Phase Accuracy** (Last 5 epochs): {late_phase:.4f}\n"
            report += f"- **Overall Improvement**: {improvement:.4f}\n"
    
    # Add interpretability analysis if available
    if interpretability:
        report += f"""
## Interpretability Analysis

### Grad-CAM Results
"""
        if 'average_cam_max' in interpretability:
            report += f"""- **Average CAM Maximum**: {interpretability['average_cam_max']:.4f}
- **Average CAM Mean**: {interpretability['average_cam_mean']:.4f}
- **Samples Analyzed**: {interpretability.get('num_samples', 'N/A')}

### Attention Pattern Analysis
"""
            cam_max = interpretability['average_cam_max']
            cam_mean = interpretability['average_cam_mean']
            
            if cam_max > 0.8:
                report += "- **Strong Focus**: Model shows strong attention to specific regions\n"
            elif cam_max > 0.6:
                report += "- **Moderate Focus**: Model shows moderate attention concentration\n"
            else:
                report += "- **Distributed Attention**: Model uses more distributed attention patterns\n"
            
            if cam_mean > 0.6:
                report += "- **High Overall Activation**: Model activates broadly across images\n"
            elif cam_mean > 0.4:
                report += "- **Balanced Activation**: Model shows balanced attention distribution\n"
            else:
                report += "- **Selective Activation**: Model is highly selective in attention\n"
    
    # Add evaluation results if available
    if evaluation:
        report += f"""
## Test Set Evaluation

### Performance Metrics
"""
        if 'accuracy' in evaluation:
            report += f"- **Test Accuracy**: {evaluation['accuracy']:.4f}\n"
        if 'precision' in evaluation:
            report += f"- **Precision**: {evaluation['precision']:.4f}\n"
        if 'recall' in evaluation:
            report += f"- **Recall**: {evaluation['recall']:.4f}\n"
        if 'f1_score' in evaluation:
            report += f"- **F1 Score**: {evaluation['f1_score']:.4f}\n"
    
    # Add deployment recommendations
    report += f"""
## Deployment Recommendations

### Recommended Use Cases
"""
    
    for use_case in model_info['use_cases']:
        report += f"- **{use_case}**\n"
    
    report += f"""
### Deployment Considerations

#### Advantages
"""
    
    # Generate advantages based on model characteristics
    advantages = []
    if model_info['parameters'] < 2_000_000:
        advantages.append("Ultra-lightweight design suitable for mobile and edge deployment")
    elif model_info['parameters'] < 5_000_000:
        advantages.append("Lightweight architecture with good efficiency")
    
    if performance_analysis.get('best_val_accuracy', 0) > 0.98:
        advantages.append("Excellent accuracy performance for production use")
    elif performance_analysis.get('best_val_accuracy', 0) > 0.95:
        advantages.append("Good accuracy suitable for most applications")
    
    if performance_analysis.get('training_stability') == 'very_stable':
        advantages.append("Very stable training characteristics")
    
    if 'Enhanced' in model_info['type']:
        advantages.append("Domain-specific optimizations for MIC testing scenarios")
    
    for advantage in advantages:
        report += f"- {advantage}\n"
    
    report += f"""
#### Considerations
"""
    
    # Generate considerations
    considerations = []
    if model_info['parameters'] > 20_000_000:
        considerations.append("Large model size may require significant computational resources")
    
    if performance_analysis.get('overfitting_detected'):
        considerations.append("Shows signs of overfitting - consider regularization in production")
    
    if performance_analysis.get('total_epochs', 0) > 35:
        considerations.append("Requires long training time - consider for batch training scenarios")
    
    if not considerations:
        considerations.append("No significant deployment concerns identified")
    
    for consideration in considerations:
        report += f"- {consideration}\n"
    
    # Add technical details
    report += f"""
## Technical Details

### Model Configuration
- **Experiment Path**: `{model_data['experiment_path']}`
- **Training Framework**: PyTorch
- **Optimization**: AdamW with Cosine Annealing
- **Input Size**: 70x70 RGB images
- **Output Classes**: 2 (Binary classification)

### Performance Summary
"""
    
    if performance_analysis:
        efficiency_score = performance_analysis.get('best_val_accuracy', 0) * 0.6 + \
                          (performance_analysis.get('best_val_accuracy', 0) / (model_info['parameters'] / 1e6)) * 0.3 + \
                          (performance_analysis.get('best_val_accuracy', 0) / performance_analysis.get('total_epochs', 1)) * 0.1
        
        report += f"""- **Overall Efficiency Score**: {efficiency_score:.3f}
- **Parameter Efficiency**: {performance_analysis.get('best_val_accuracy', 0) / (model_info['parameters'] / 1e6):.3f} (Acc/M params)
- **Training Efficiency**: {performance_analysis.get('best_val_accuracy', 0) / performance_analysis.get('total_epochs', 1):.4f} (Acc/Epoch)
"""
    
    report += f"""
## Conclusion

The **{model_info['architecture']}** model demonstrates {
    'excellent' if performance_analysis.get('best_val_accuracy', 0) > 0.98 else
    'good' if performance_analysis.get('best_val_accuracy', 0) > 0.95 else 'moderate'
} performance for biomedical image classification tasks. """
    
    if model_info['parameters'] < 2_000_000 and performance_analysis.get('best_val_accuracy', 0) > 0.98:
        report += "Its combination of high accuracy and lightweight design makes it ideal for production deployment, especially in resource-constrained environments."
    elif performance_analysis.get('best_val_accuracy', 0) > 0.98:
        report += "Its high accuracy makes it suitable for applications where performance is the primary concern."
    elif model_info['parameters'] < 2_000_000:
        report += "Its lightweight design makes it suitable for mobile and edge deployment scenarios."
    else:
        report += "It provides a balanced approach for various deployment scenarios."
    
    report += f"""

### Key Strengths
"""
    
    strengths = []
    if performance_analysis.get('best_val_accuracy', 0) > 0.98:
        strengths.append("High accuracy performance")
    if model_info['parameters'] < 2_000_000:
        strengths.append("Ultra-lightweight architecture")
    if performance_analysis.get('training_stability') in ['stable', 'very_stable']:
        strengths.append("Stable training characteristics")
    if 'Enhanced' in model_info['type']:
        strengths.append("Domain-specific optimizations")
    
    for strength in strengths:
        report += f"- {strength}\n"
    
    report += f"""
---
*Report generated by Individual Model Analysis Script*
*Analysis based on training history, evaluation results, and architectural characteristics*
"""
    
    return report

def main():
    """Generate individual reports for all models."""
    print("📄 Generating Individual Model Analysis Reports")
    print("=" * 50)
    
    # Define all models
    all_models = [
        'efficientnet_b0',
        'resnet18_improved', 
        'convnext_tiny',
        'vit_tiny',
        'coatnet',
        'mic_mobilenetv3',
        'micro_vit',
        'airbubble_hybrid_net'
    ]
    
    # Create reports directory
    reports_dir = Path("reports/individual_analysis")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    generated_reports = []
    
    for model_name in all_models:
        try:
            report_content = generate_individual_report(model_name)
            
            # Save report
            report_file = reports_dir / f"{model_name}_analysis.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            generated_reports.append(report_file)
            print(f"   ✅ {model_name}: Report saved to {report_file}")
            
        except Exception as e:
            print(f"   ❌ {model_name}: Error generating report - {str(e)}")
    
    print(f"\n✅ Generated {len(generated_reports)} individual analysis reports")
    print(f"📁 Reports saved to: {reports_dir.absolute()}")
    
    # Generate index file
    index_content = f"""# Individual Model Analysis Reports

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This directory contains detailed individual analysis reports for all 8 trained models.

## Available Reports

"""
    
    for i, model_name in enumerate(all_models, 1):
        model_info = get_model_info(model_name)
        index_content += f"{i}. **[{model_info['architecture']}]({model_name}_analysis.md)** - {model_info['description']}\n"
    
    index_content += f"""
## Report Contents

Each individual report includes:

- **Executive Summary** - Key metrics and performance overview
- **Model Architecture** - Technical specifications and features
- **Training Performance Analysis** - Detailed training metrics and insights
- **Interpretability Analysis** - Grad-CAM results and attention patterns (if available)
- **Test Set Evaluation** - Performance on held-out test data (if available)
- **Deployment Recommendations** - Use cases and deployment considerations
- **Technical Details** - Configuration and efficiency metrics
- **Conclusion** - Summary and key strengths

## Usage

These reports provide comprehensive analysis for:
- Model selection and comparison
- Deployment planning and optimization
- Performance understanding and debugging
- Research insights and documentation

---
*Generated by Individual Model Analysis Script*
"""
    
    index_file = reports_dir / "README.md"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"📋 Index file created: {index_file}")

if __name__ == "__main__":
    main()