#!/usr/bin/env python3
"""
Extract error samples from inception_micro checkpoint training history
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

def load_checkpoint_and_extract_errors():
    """Load inception_micro checkpoint and extract error samples"""
    
    print("🔍 提取inception_micro错误样本清单...")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # Load checkpoint
    checkpoint_path = "checkpoints/inception_micro_20250808_000513_best.pth"
    
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
    print(f"   时间戳: {checkpoint.get('timestamp', 'N/A')}")
    
    # Initialize model
    print(f"\n🏗️ 初始化模型...")
    model = InceptionMicro(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功")
    
    # Get data loaders
    print(f"\n📊 加载bioast_dataset...")
    train_loader, val_loader, test_loader = create_real_data_loaders(
        batch_size=32,
        num_workers=4
    )
    
    print(f"   训练样本: {len(train_loader.dataset)}")
    print(f"   验证样本: {len(val_loader.dataset)}")
    print(f"   测试样本: {len(test_loader.dataset)}")
    
    # Evaluate on validation set to find error samples
    print(f"\n🔍 在验证集上评估模型以找出错误样本...")
    
    error_samples = []
    correct_samples = []
    total_samples = 0
    
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
                    'is_correct': true_label == pred_label,
                    'batch_index': batch_idx,
                    'in_batch_index': i
                }
                
                if true_label != pred_label:
                    error_samples.append(sample_info)
                else:
                    correct_samples.append(sample_info)
                
                total_samples += 1
    
    # Calculate accuracy
    accuracy = len(correct_samples) / total_samples * 100
    error_rate = len(error_samples) / total_samples * 100
    
    print(f"\n📊 验证集评估结果:")
    print(f"   总样本数: {total_samples}")
    print(f"   正确样本: {len(correct_samples)}")
    print(f"   错误样本: {len(error_samples)}")
    print(f"   准确率: {accuracy:.4f}%")
    print(f"   错误率: {error_rate:.4f}%")
    
    # Analyze error patterns
    print(f"\n🔍 错误样本分析:")
    
    # Group errors by type
    false_positives = [s for s in error_samples if s['true_label'] == 0 and s['predicted_label'] == 1]
    false_negatives = [s for s in error_samples if s['true_label'] == 1 and s['predicted_label'] == 0]
    
    print(f"   假阳性 (真实:负样本, 预测:正样本): {len(false_positives)}")
    print(f"   假阴性 (真实:正样本, 预测:负样本): {len(false_negatives)}")
    
    # Sort error samples by confidence (lowest confidence first - most uncertain predictions)
    error_samples.sort(key=lambda x: x['confidence'])
    
    print(f"\n📋 错误样本详细清单 (按置信度排序):")
    print("-" * 80)
    print(f"{'索引':<6} {'真实标签':<8} {'预测标签':<8} {'置信度':<10} {'错误类型':<12}")
    print("-" * 80)
    
    for i, sample in enumerate(error_samples[:20]):  # Show top 20 error samples
        error_type = "假阳性" if sample['true_label'] == 0 else "假阴性"
        print(f"{sample['sample_index']:<6} {sample['true_label']:<8} {sample['predicted_label']:<8} {sample['confidence']:<10.4f} {error_type:<12}")
    
    if len(error_samples) > 20:
        print(f"... 还有 {len(error_samples) - 20} 个错误样本")
    
    # Test set evaluation
    print(f"\n🔍 在测试集上评估模型...")
    
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
                    'is_correct': true_label == pred_label,
                    'batch_index': batch_idx,
                    'in_batch_index': i
                }
                
                if true_label != pred_label:
                    test_error_samples.append(sample_info)
                else:
                    test_correct_samples.append(sample_info)
                
                test_total_samples += 1
    
    # Calculate test accuracy
    test_accuracy = len(test_correct_samples) / test_total_samples * 100
    test_error_rate = len(test_error_samples) / test_total_samples * 100
    
    print(f"\n📊 测试集评估结果:")
    print(f"   总样本数: {test_total_samples}")
    print(f"   正确样本: {len(test_correct_samples)}")
    print(f"   错误样本: {len(test_error_samples)}")
    print(f"   准确率: {test_accuracy:.4f}%")
    print(f"   错误率: {test_error_rate:.4f}%")
    
    # Prepare comprehensive error analysis report
    error_analysis = {
        'model_name': 'inception_micro',
        'checkpoint_path': checkpoint_path,
        'analysis_timestamp': datetime.now().isoformat(),
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'validation_accuracy': float(checkpoint.get('accuracy', 0)),
            'validation_loss': float(checkpoint.get('loss', 0)),
            'timestamp': checkpoint.get('timestamp', 'N/A')
        },
        'dataset_info': {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'input_size': '70x70',
            'num_classes': 2
        },
        'validation_results': {
            'total_samples': total_samples,
            'correct_samples': len(correct_samples),
            'error_samples': len(error_samples),
            'accuracy': float(accuracy),
            'error_rate': float(error_rate),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives)
        },
        'test_results': {
            'total_samples': test_total_samples,
            'correct_samples': len(test_correct_samples),
            'error_samples': len(test_error_samples),
            'accuracy': float(test_accuracy),
            'error_rate': float(test_error_rate),
            'false_positives': len([s for s in test_error_samples if s['true_label'] == 0 and s['predicted_label'] == 1]),
            'false_negatives': len([s for s in test_error_samples if s['true_label'] == 1 and s['predicted_label'] == 0])
        },
        'validation_error_samples': error_samples,
        'test_error_samples': test_error_samples,
        'error_analysis_summary': {
            'validation_error_types': {
                'false_positives': len(false_positives),
                'false_negatives': len(false_negatives)
            },
            'test_error_types': {
                'false_positives': len([s for s in test_error_samples if s['true_label'] == 0 and s['predicted_label'] == 1]),
                'false_negatives': len([s for s in test_error_samples if s['true_label'] == 1 and s['predicted_label'] == 0])
            },
            'most_uncertain_predictions': sorted(error_samples, key=lambda x: x['confidence'])[:10],
            'highest_confidence_errors': sorted(error_samples, key=lambda x: x['confidence'], reverse=True)[:10]
        }
    }
    
    # Save error analysis report
    os.makedirs('error_analysis', exist_ok=True)
    report_path = f"error_analysis/inception_micro_error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    print(f"\n💾 错误分析报告已保存: {report_path}")
    
    # Generate summary report
    print(f"\n" + "=" * 60)
    print(f"INCEPTION_MICRO 错误样本分析总结")
    print(f"=" * 60)
    print(f"模型: inception_micro")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint验证准确率: {checkpoint.get('accuracy', 0):.4f}%")
    print(f"")
    print(f"验证集结果:")
    print(f"  总样本: {total_samples}")
    print(f"  错误样本: {len(error_samples)}")
    print(f"  准确率: {accuracy:.4f}%")
    print(f"  假阳性: {len(false_positives)}")
    print(f"  假阴性: {len(false_negatives)}")
    print(f"")
    print(f"测试集结果:")
    print(f"  总样本: {test_total_samples}")
    print(f"  错误样本: {len(test_error_samples)}")
    print(f"  准确率: {test_accuracy:.4f}%")
    print(f"")
    print(f"错误分析报告: {report_path}")
    print(f"=" * 60)
    
    return error_analysis

if __name__ == "__main__":
    result = load_checkpoint_and_extract_errors()
    if result:
        print(f"\n✅ inception_micro错误样本分析完成!")
    else:
        print(f"\n❌ inception_micro错误样本分析失败!")