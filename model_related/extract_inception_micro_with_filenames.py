#!/usr/bin/env python3
"""
Extract error samples from inception_micro checkpoint with source filenames
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

def get_dataset_filenames():
    """Get all filenames from the bioast_dataset"""
    
    dataset_path = "bioast_dataset"
    
    # Collect all filenames with their labels and splits
    all_files = {}
    
    for split in ['train', 'val', 'test']:
        all_files[split] = {'positive': [], 'negative': []}
        
        # Positive samples
        pos_path = os.path.join(dataset_path, 'positive', split)
        if os.path.exists(pos_path):
            pos_files = [f for f in os.listdir(pos_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_files[split]['positive'] = sorted(pos_files)
        
        # Negative samples
        neg_path = os.path.join(dataset_path, 'negative', split)
        if os.path.exists(neg_path):
            neg_files = [f for f in os.listdir(neg_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_files[split]['negative'] = sorted(neg_files)
    
    return all_files

def create_index_to_filename_mapping():
    """Create mapping from dataset index to filename"""
    
    all_files = get_dataset_filenames()
    
    # Create index mapping for validation set (this is what we need for error analysis)
    val_index_to_filename = {}
    current_index = 0
    
    # First add positive samples (label 1)
    for filename in all_files['val']['positive']:
        val_index_to_filename[current_index] = {
            'filename': filename,
            'label': 1,
            'category': 'positive',
            'full_path': f"bioast_dataset/positive/val/{filename}"
        }
        current_index += 1
    
    # Then add negative samples (label 0)  
    for filename in all_files['val']['negative']:
        val_index_to_filename[current_index] = {
            'filename': filename,
            'label': 0,
            'category': 'negative',
            'full_path': f"bioast_dataset/negative/val/{filename}"
        }
        current_index += 1
    
    # Create test set mapping too
    test_index_to_filename = {}
    current_index = 0
    
    # First add positive samples (label 1)
    for filename in all_files['test']['positive']:
        test_index_to_filename[current_index] = {
            'filename': filename,
            'label': 1,
            'category': 'positive',
            'full_path': f"bioast_dataset/positive/test/{filename}"
        }
        current_index += 1
    
    # Then add negative samples (label 0)
    for filename in all_files['test']['negative']:
        test_index_to_filename[current_index] = {
            'filename': filename,
            'label': 0,
            'category': 'negative',
            'full_path': f"bioast_dataset/negative/test/{filename}"
        }
        current_index += 1
    
    return val_index_to_filename, test_index_to_filename, all_files

def extract_error_samples_with_filenames():
    """Extract error samples with source filenames"""
    
    print("🔍 提取inception_micro错误样本清单 (包含源文件名)...")
    print("=" * 80)
    
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
    
    # Initialize model
    print(f"\n🏗️ 初始化模型...")
    model = InceptionMicro(num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功")
    
    # Create filename mappings
    print(f"\n📁 创建文件名映射...")
    val_index_to_filename, test_index_to_filename, all_files = create_index_to_filename_mapping()
    
    print(f"   验证集正样本: {len(all_files['val']['positive'])}")
    print(f"   验证集负样本: {len(all_files['val']['negative'])}")
    print(f"   测试集正样本: {len(all_files['test']['positive'])}")
    print(f"   测试集负样本: {len(all_files['test']['negative'])}")
    
    # Get data loaders
    print(f"\n📊 加载bioast_dataset...")
    train_loader, val_loader, test_loader = create_real_data_loaders(
        batch_size=32,
        num_workers=4
    )
    
    print(f"   训练样本: {len(train_loader.dataset)}")
    print(f"   验证样本: {len(val_loader.dataset)}")
    print(f"   测试样本: {len(test_loader.dataset)}")
    
    # Evaluate on validation set
    print(f"\n🔍 在验证集上评估模型...")
    
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
                
                # Get filename info
                filename_info = val_index_to_filename.get(sample_idx, {
                    'filename': f'unknown_{sample_idx}',
                    'label': true_label,
                    'category': 'positive' if true_label == 1 else 'negative',
                    'full_path': f'unknown_path_{sample_idx}'
                })
                
                sample_info = {
                    'sample_index': sample_idx,
                    'filename': filename_info['filename'],
                    'full_path': filename_info['full_path'],
                    'category': filename_info['category'],
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
    print(f"🔍 在测试集上评估模型...")
    
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
                
                # Get filename info
                filename_info = test_index_to_filename.get(sample_idx, {
                    'filename': f'unknown_{sample_idx}',
                    'label': true_label,
                    'category': 'positive' if true_label == 1 else 'negative',
                    'full_path': f'unknown_path_{sample_idx}'
                })
                
                sample_info = {
                    'sample_index': sample_idx,
                    'filename': filename_info['filename'],
                    'full_path': filename_info['full_path'],
                    'category': filename_info['category'],
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
    
    print(f"\n📊 验证集评估结果:")
    print(f"   总样本数: {val_total_samples}")
    print(f"   正确样本: {len(val_correct_samples)}")
    print(f"   错误样本: {len(val_error_samples)}")
    print(f"   准确率: {val_accuracy:.4f}%")
    print(f"   错误率: {(100 - val_accuracy):.4f}%")
    print(f"   假阳性: {len(val_false_positives)}")
    print(f"   假阴性: {len(val_false_negatives)}")
    
    print(f"\n📊 测试集评估结果:")
    print(f"   总样本数: {test_total_samples}")
    print(f"   正确样本: {len(test_correct_samples)}")
    print(f"   错误样本: {len(test_error_samples)}")
    print(f"   准确率: {test_accuracy:.4f}%")
    print(f"   错误率: {(100 - test_accuracy):.4f}%")
    print(f"   假阳性: {len(test_false_positives)}")
    print(f"   假阴性: {len(test_false_negatives)}")
    
    # Sort error samples by confidence
    val_error_samples.sort(key=lambda x: x['confidence'])
    test_error_samples.sort(key=lambda x: x['confidence'])
    
    # Display validation error samples with filenames
    print(f"\n📋 验证集错误样本详细清单 (按置信度排序):")
    print("-" * 120)
    print(f"{'索引':<6} {'源文件名':<30} {'真实标签':<8} {'预测标签':<8} {'置信度':<10} {'错误类型':<12}")
    print("-" * 120)
    
    for sample in val_error_samples:
        error_type = "假阳性" if sample['true_label'] == 0 else "假阴性"
        print(f"{sample['sample_index']:<6} {sample['filename']:<30} {sample['true_label']:<8} {sample['predicted_label']:<8} {sample['confidence']:<10.4f} {error_type:<12}")
    
    # Display test error samples (first 20)
    if test_error_samples:
        print(f"\n📋 测试集错误样本详细清单 (前20个，按置信度排序):")
        print("-" * 120)
        print(f"{'索引':<6} {'源文件名':<30} {'真实标签':<8} {'预测标签':<8} {'置信度':<10} {'错误类型':<12}")
        print("-" * 120)
        
        for sample in test_error_samples[:20]:
            error_type = "假阳性" if sample['true_label'] == 0 else "假阴性"
            print(f"{sample['sample_index']:<6} {sample['filename']:<30} {sample['true_label']:<8} {sample['predicted_label']:<8} {sample['confidence']:<10.4f} {error_type:<12}")
        
        if len(test_error_samples) > 20:
            print(f"... 还有 {len(test_error_samples) - 20} 个测试集错误样本")
    
    # Convert all numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Prepare comprehensive report
    error_analysis = {
        'model_name': 'inception_micro',
        'checkpoint_path': checkpoint_path,
        'analysis_timestamp': datetime.now().isoformat(),
        'checkpoint_info': {
            'epoch': int(checkpoint.get('epoch', 0)),
            'validation_accuracy': float(checkpoint.get('accuracy', 0)),
            'validation_loss': float(checkpoint.get('loss', 0)),
            'timestamp': checkpoint.get('timestamp', 'N/A')
        },
        'dataset_info': {
            'name': 'bioast_dataset',
            'train_samples': int(len(train_loader.dataset)),
            'val_samples': int(len(val_loader.dataset)),
            'test_samples': int(len(test_loader.dataset)),
            'val_positive_files': int(len(all_files['val']['positive'])),
            'val_negative_files': int(len(all_files['val']['negative'])),
            'test_positive_files': int(len(all_files['test']['positive'])),
            'test_negative_files': int(len(all_files['test']['negative'])),
            'input_size': '70x70',
            'num_classes': 2
        },
        'validation_results': {
            'total_samples': int(val_total_samples),
            'correct_samples': int(len(val_correct_samples)),
            'error_samples': int(len(val_error_samples)),
            'accuracy': float(val_accuracy),
            'error_rate': float(100 - val_accuracy),
            'false_positives': int(len(val_false_positives)),
            'false_negatives': int(len(val_false_negatives))
        },
        'test_results': {
            'total_samples': int(test_total_samples),
            'correct_samples': int(len(test_correct_samples)),
            'error_samples': int(len(test_error_samples)),
            'accuracy': float(test_accuracy),
            'error_rate': float(100 - test_accuracy),
            'false_positives': int(len(test_false_positives)),
            'false_negatives': int(len(test_false_negatives))
        },
        'validation_error_samples_with_filenames': convert_numpy_types(val_error_samples),
        'test_error_samples_with_filenames': convert_numpy_types(test_error_samples[:50]),  # Limit to first 50 for file size
        'filename_mappings': {
            'validation_files': convert_numpy_types(dict(list(val_index_to_filename.items())[:100])),  # Sample for reference
            'test_files': convert_numpy_types(dict(list(test_index_to_filename.items())[:100]))  # Sample for reference
        }
    }
    
    # Save detailed report
    os.makedirs('error_analysis', exist_ok=True)
    report_path = f"error_analysis/inception_micro_with_filenames_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    print(f"\n💾 详细错误分析报告已保存: {report_path}")
    
    # Generate summary
    print(f"\n" + "=" * 80)
    print(f"INCEPTION_MICRO 错误样本分析总结 (包含源文件名)")
    print(f"=" * 80)
    print(f"模型: inception_micro")
    print(f"数据集: bioast_dataset")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"Checkpoint验证准确率: {checkpoint.get('accuracy', 0):.4f}%")
    print(f"")
    print(f"验证集结果:")
    print(f"  总样本: {val_total_samples}")
    print(f"  错误样本: {len(val_error_samples)}")
    print(f"  准确率: {val_accuracy:.4f}%")
    print(f"  假阳性: {len(val_false_positives)}")
    print(f"  假阴性: {len(val_false_negatives)}")
    print(f"")
    print(f"测试集结果:")
    print(f"  总样本: {test_total_samples}")
    print(f"  错误样本: {len(test_error_samples)}")
    print(f"  准确率: {test_accuracy:.4f}%")
    print(f"  假阳性: {len(test_false_positives)}")
    print(f"  假阴性: {len(test_false_negatives)}")
    print(f"")
    print(f"详细报告: {report_path}")
    print(f"=" * 80)
    
    return error_analysis

if __name__ == "__main__":
    result = extract_error_samples_with_filenames()
    if result:
        print(f"\n✅ inception_micro错误样本分析完成 (包含源文件名)!")
    else:
        print(f"\n❌ inception_micro错误样本分析失败!")