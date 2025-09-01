#!/usr/bin/env python3
"""
Extract error samples from all high-accuracy model checkpoints
"""

import os
import sys
import torch
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_data_loader import create_real_data_loaders
from models.inception_micro import InceptionMicro
from models.mic_mobilenetv3 import MICMobileNetV3
from models.resnet_micro import ResnetMicro

def analyze_model_errors(model_name, model_class, checkpoint_path):
    """Analyze error samples for a specific model"""
    
    print(f"\n🔍 分析 {model_name.upper()} 错误样本...")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return None
    
    print(f"📂 加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Display checkpoint info
    print(f"📋 Checkpoint信息:")
    print(f"   训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"   验证准确率: {checkpoint.get('accuracy', 'N/A'):.4f}%")
    print(f"   验证损失: {checkpoint.get('loss', 'N/A'):.4f}")
    
    # Initialize model
    print(f"🏗️ 初始化 {model_name} 模型...")
    model = model_class(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功")
    
    # Get data loaders (only once for efficiency)
    if not hasattr(analyze_model_errors, 'data_loaders'):
        print(f"📊 加载bioast_dataset...")
        analyze_model_errors.data_loaders = create_real_data_loaders(
            batch_size=32,
            num_workers=4
        )
        train_loader, val_loader, test_loader = analyze_model_errors.data_loaders
        print(f"   训练样本: {len(train_loader.dataset)}")
        print(f"   验证样本: {len(val_loader.dataset)}")
        print(f"   测试样本: {len(test_loader.dataset)}")
    else:
        train_loader, val_loader, test_loader = analyze_model_errors.data_loaders
    
    # Evaluate on validation set
    print(f"🔍 在验证集上评估...")
    
    val_error_samples = []
    val_correct_samples = []
    val_total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get predictions
            _, predicted = torch.max(output, 1)
            
            # Find error samples in this batch
            for i in range(data.size(0)):
                sample_idx = batch_idx * val_loader.batch_size + i
                true_label = target[i].item()
                pred_label = predicted[i].item()
                confidence = torch.softmax(output[i], dim=0)
                max_confidence = torch.max(confidence).item()
                
                sample_info = {
                    'sample_index': sample_idx,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': max_confidence,
                    'is_correct': true_label == pred_label
                }
                
                if true_label != pred_label:
                    val_error_samples.append(sample_info)
                else:
                    val_correct_samples.append(sample_info)
                
                val_total_samples += 1
    
    # Evaluate on test set
    print(f"🔍 在测试集上评估...")
    
    test_error_samples = []
    test_correct_samples = []
    test_total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get predictions
            _, predicted = torch.max(output, 1)
            
            # Find error samples in this batch
            for i in range(data.size(0)):
                sample_idx = batch_idx * test_loader.batch_size + i
                true_label = target[i].item()
                pred_label = predicted[i].item()
                confidence = torch.softmax(output[i], dim=0)
                max_confidence = torch.max(confidence).item()
                
                sample_info = {
                    'sample_index': sample_idx,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': max_confidence,
                    'is_correct': true_label == pred_label
                }
                
                if true_label != pred_label:
                    test_error_samples.append(sample_info)
                else:
                    test_correct_samples.append(sample_info)
                
                test_total_samples += 1
    
    # Calculate metrics
    val_accuracy = len(val_correct_samples) / val_total_samples * 100
    test_accuracy = len(test_correct_samples) / test_total_samples * 100
    
    # Analyze error patterns
    val_false_positives = [s for s in val_error_samples if s['true_label'] == 0 and s['predicted_label'] == 1]
    val_false_negatives = [s for s in val_error_samples if s['true_label'] == 1 and s['predicted_label'] == 0]
    
    test_false_positives = [s for s in test_error_samples if s['true_label'] == 0 and s['predicted_label'] == 1]
    test_false_negatives = [s for s in test_error_samples if s['true_label'] == 1 and s['predicted_label'] == 0]
    
    print(f"📊 验证集结果:")
    print(f"   总样本: {val_total_samples}")
    print(f"   错误样本: {len(val_error_samples)}")
    print(f"   准确率: {val_accuracy:.4f}%")
    print(f"   假阳性: {len(val_false_positives)}")
    print(f"   假阴性: {len(val_false_negatives)}")
    
    print(f"📊 测试集结果:")
    print(f"   总样本: {test_total_samples}")
    print(f"   错误样本: {len(test_error_samples)}")
    print(f"   准确率: {test_accuracy:.4f}%")
    print(f"   假阳性: {len(test_false_positives)}")
    print(f"   假阴性: {len(test_false_negatives)}")
    
    # Sort error samples by confidence
    val_error_samples.sort(key=lambda x: x['confidence'])
    test_error_samples.sort(key=lambda x: x['confidence'])
    
    # Display validation error samples
    if val_error_samples:
        print(f"\n📋 验证集错误样本清单 (按置信度排序):")
        print("-" * 80)
        print(f"{'索引':<6} {'真实标签':<8} {'预测标签':<8} {'置信度':<10} {'错误类型':<12}")
        print("-" * 80)
        
        for sample in val_error_samples:
            error_type = "假阳性" if sample['true_label'] == 0 else "假阴性"
            print(f"{sample['sample_index']:<6} {sample['true_label']:<8} {sample['predicted_label']:<8} {sample['confidence']:<10.4f} {error_type:<12}")
    
    # Prepare analysis report
    analysis_report = {
        'model_name': model_name,
        'checkpoint_path': checkpoint_path,
        'analysis_timestamp': datetime.now().isoformat(),
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'validation_accuracy': float(checkpoint.get('accuracy', 0)),
            'validation_loss': float(checkpoint.get('loss', 0))
        },
        'validation_results': {
            'total_samples': val_total_samples,
            'correct_samples': len(val_correct_samples),
            'error_samples': len(val_error_samples),
            'accuracy': float(val_accuracy),
            'false_positives': len(val_false_positives),
            'false_negatives': len(val_false_negatives)
        },
        'test_results': {
            'total_samples': test_total_samples,
            'correct_samples': len(test_correct_samples),
            'error_samples': len(test_error_samples),
            'accuracy': float(test_accuracy),
            'false_positives': len(test_false_positives),
            'false_negatives': len(test_false_negatives)
        },
        'validation_error_samples': val_error_samples,
        'test_error_samples': test_error_samples
    }
    
    return analysis_report

def main():
    """Main function to analyze all high-accuracy models"""
    
    print("🏆 高精度模型错误样本分析")
    print("=" * 60)
    
    # Define high-accuracy models
    models_to_analyze = [
        {
            'name': 'inception_micro',
            'class': InceptionMicro,
            'checkpoint': 'checkpoints/inception_micro_20250808_000513_best.pth'
        },
        {
            'name': 'mic_mobilenetv3',
            'class': MICMobileNetV3,
            'checkpoint': 'checkpoints/mic_mobilenetv3_20250807_231138_best.pth'
        },
        {
            'name': 'resnet_micro',
            'class': ResnetMicro,
            'checkpoint': 'checkpoints/resnet_micro_20250808_005254_best.pth'
        }
    ]
    
    all_analyses = {}
    
    # Analyze each model
    for model_info in models_to_analyze:
        try:
            analysis = analyze_model_errors(
                model_info['name'],
                model_info['class'],
                model_info['checkpoint']
            )
            
            if analysis:
                all_analyses[model_info['name']] = analysis
                
        except Exception as e:
            print(f"❌ 分析 {model_info['name']} 时出错: {e}")
            continue
    
    # Generate comprehensive report
    comprehensive_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'name': 'bioast_dataset',
            'train_samples': 9094,
            'val_samples': 1316,
            'test_samples': 2614,
            'input_size': '70x70',
            'num_classes': 2
        },
        'models_analyzed': len(all_analyses),
        'individual_analyses': all_analyses,
        'comparative_summary': {}
    }
    
    # Add comparative summary
    if all_analyses:
        summary = {}
        for model_name, analysis in all_analyses.items():
            summary[model_name] = {
                'validation_accuracy': analysis['validation_results']['accuracy'],
                'test_accuracy': analysis['test_results']['accuracy'],
                'validation_errors': analysis['validation_results']['error_samples'],
                'test_errors': analysis['test_results']['error_samples'],
                'validation_false_positives': analysis['validation_results']['false_positives'],
                'validation_false_negatives': analysis['validation_results']['false_negatives'],
                'test_false_positives': analysis['test_results']['false_positives'],
                'test_false_negatives': analysis['test_results']['false_negatives']
            }
        
        comprehensive_report['comparative_summary'] = summary
    
    # Save comprehensive report
    os.makedirs('error_analysis', exist_ok=True)
    report_path = f"error_analysis/high_accuracy_models_error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    print(f"\n💾 综合错误分析报告已保存: {report_path}")
    
    # Print summary
    print(f"\n" + "=" * 60)
    print(f"高精度模型错误样本分析总结")
    print(f"=" * 60)
    
    for model_name, analysis in all_analyses.items():
        print(f"\n📊 {model_name.upper()}:")
        print(f"   验证准确率: {analysis['validation_results']['accuracy']:.4f}%")
        print(f"   测试准确率: {analysis['test_results']['accuracy']:.4f}%")
        print(f"   验证错误样本: {analysis['validation_results']['error_samples']}")
        print(f"   测试错误样本: {analysis['test_results']['error_samples']}")
        print(f"   验证假阳性: {analysis['validation_results']['false_positives']}")
        print(f"   验证假阴性: {analysis['validation_results']['false_negatives']}")
    
    print(f"\n📄 详细报告: {report_path}")
    print(f"=" * 60)
    
    return comprehensive_report

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ 所有高精度模型错误样本分析完成!")
    else:
        print(f"\n❌ 错误样本分析失败!")