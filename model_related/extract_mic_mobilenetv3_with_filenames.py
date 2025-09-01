#!/usr/bin/env python3
"""
提取mic_mobilenetv3错误样本清单 (包含源文件名)
基于checkpoint训练历史记录的详细分析
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_data_loader import create_real_data_loaders
from models.mic_mobilenetv3 import MICMobileNetV3

def create_filename_mapping():
    """创建数据集索引到文件名的映射"""
    print("📁 创建文件名映射...")
    
    # 数据集路径
    val_pos_path = "bioast_dataset/positive/val"
    val_neg_path = "bioast_dataset/negative/val"
    test_pos_path = "bioast_dataset/positive/test"
    test_neg_path = "bioast_dataset/negative/test"
    
    # 收集验证集文件名
    val_filenames = []
    
    # 正样本
    if os.path.exists(val_pos_path):
        pos_files = sorted([f for f in os.listdir(val_pos_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        for f in pos_files:
            val_filenames.append((f, 1))  # (filename, label)
    
    # 负样本
    if os.path.exists(val_neg_path):
        neg_files = sorted([f for f in os.listdir(val_neg_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        for f in neg_files:
            val_filenames.append((f, 0))  # (filename, label)
    
    # 收集测试集文件名
    test_filenames = []
    
    # 正样本
    if os.path.exists(test_pos_path):
        pos_files = sorted([f for f in os.listdir(test_pos_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        for f in pos_files:
            test_filenames.append((f, 1))  # (filename, label)
    
    # 负样本
    if os.path.exists(test_neg_path):
        neg_files = sorted([f for f in os.listdir(test_neg_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        for f in neg_files:
            test_filenames.append((f, 0))  # (filename, label)
    
    print(f"   验证集正样本: {len([f for f, l in val_filenames if l == 1])}")
    print(f"   验证集负样本: {len([f for f, l in val_filenames if l == 0])}")
    print(f"   测试集正样本: {len([f for f, l in test_filenames if l == 1])}")
    print(f"   测试集负样本: {len([f for f, l in test_filenames if l == 0])}")
    
    return val_filenames, test_filenames

def evaluate_model_with_filenames(model, data_loader, filenames, device, dataset_name="Dataset"):
    """评估模型并返回错误样本的详细信息"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    print(f"🔍 在{dataset_name}上评估模型...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            # 获取预测置信度 (预测类别的概率)
            confidence = probabilities.gather(1, predicted.unsqueeze(1)).squeeze(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    # 计算准确率
    correct = sum(p == t for p, t in zip(all_predictions, all_targets))
    total = len(all_predictions)
    accuracy = 100.0 * correct / total
    
    # 找出错误样本
    error_samples = []
    false_positives = 0
    false_negatives = 0
    
    for i, (pred, true, conf) in enumerate(zip(all_predictions, all_targets, all_confidences)):
        if pred != true:
            filename = filenames[i][0] if i < len(filenames) else f"unknown_{i}"
            error_type = "假阳性" if (true == 0 and pred == 1) else "假阴性"
            
            if error_type == "假阳性":
                false_positives += 1
            else:
                false_negatives += 1
            
            error_samples.append({
                'index': int(i),
                'filename': filename,
                'true_label': int(true),
                'predicted_label': int(pred),
                'confidence': float(conf),
                'error_type': error_type
            })
    
    # 按置信度排序
    error_samples.sort(key=lambda x: x['confidence'])
    
    return {
        'total_samples': int(total),
        'correct_samples': int(correct),
        'error_samples': int(len(error_samples)),
        'accuracy': float(accuracy),
        'error_rate': float(100.0 * len(error_samples) / total),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives),
        'error_details': error_samples
    }

def main():
    print("🔍 提取mic_mobilenetv3错误样本清单 (包含源文件名)...")
    print("=" * 80)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # Checkpoint路径
    checkpoint_path = "checkpoints/mic_mobilenetv3_20250807_231138_best.pth"
    
    # 加载checkpoint
    print(f"📂 加载checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 显示checkpoint信息
    print("📋 Checkpoint信息:")
    print(f"   训练轮数: {checkpoint.get('epoch', 'N/A')}")
    if 'accuracy' in checkpoint:
        print(f"   验证准确率: {checkpoint['accuracy']:.4f}%")
    if 'loss' in checkpoint:
        print(f"   验证损失: {checkpoint['loss']:.4f}")
    
    # 初始化模型
    print("\n🏗️ 初始化mic_mobilenetv3模型...")
    try:
        model = MICMobileNetV3(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 创建文件名映射
    val_filenames, test_filenames = create_filename_mapping()
    
    # 加载数据
    print("\n📊 加载bioast_dataset...")
    try:
        train_loader, val_loader, test_loader = create_real_data_loaders(
            batch_size=32,
            num_workers=4
        )
        print(f"   训练样本: {len(train_loader.dataset)}")
        print(f"   验证样本: {len(val_loader.dataset)}")
        print(f"   测试样本: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 评估验证集
    val_results = evaluate_model_with_filenames(model, val_loader, val_filenames, device, "验证集")
    
    # 评估测试集
    test_results = evaluate_model_with_filenames(model, test_loader, test_filenames, device, "测试集")
    
    # 显示结果
    print(f"\n📊 验证集评估结果:")
    print(f"   总样本数: {val_results['total_samples']}")
    print(f"   正确样本: {val_results['correct_samples']}")
    print(f"   错误样本: {val_results['error_samples']}")
    print(f"   准确率: {val_results['accuracy']:.4f}%")
    print(f"   错误率: {val_results['error_rate']:.4f}%")
    print(f"   假阳性: {val_results['false_positives']}")
    print(f"   假阴性: {val_results['false_negatives']}")
    
    print(f"\n📊 测试集评估结果:")
    print(f"   总样本数: {test_results['total_samples']}")
    print(f"   正确样本: {test_results['correct_samples']}")
    print(f"   错误样本: {test_results['error_samples']}")
    print(f"   准确率: {test_results['accuracy']:.4f}%")
    print(f"   错误率: {test_results['error_rate']:.4f}%")
    print(f"   假阳性: {test_results['false_positives']}")
    print(f"   假阴性: {test_results['false_negatives']}")
    
    # 显示验证集错误样本详情
    print(f"\n📋 验证集错误样本详细清单 (按置信度排序):")
    print("-" * 120)
    print(f"{'索引':<8} {'源文件名':<35} {'真实标签':<8} {'预测标签':<8} {'置信度':<12} {'错误类型'}")
    print("-" * 120)
    
    for error in val_results['error_details']:
        print(f"{error['index']:<8} {error['filename']:<35} {error['true_label']:<8} {error['predicted_label']:<8} {error['confidence']:<12.4f} {error['error_type']}")
    
    # 显示测试集错误样本详情 (前20个)
    print(f"\n📋 测试集错误样本详细清单 (前20个，按置信度排序):")
    print("-" * 120)
    print(f"{'索引':<8} {'源文件名':<35} {'真实标签':<8} {'预测标签':<8} {'置信度':<12} {'错误类型'}")
    print("-" * 120)
    
    for i, error in enumerate(test_results['error_details'][:20]):
        print(f"{error['index']:<8} {error['filename']:<35} {error['true_label']:<8} {error['predicted_label']:<8} {error['confidence']:<12.4f} {error['error_type']}")
    
    if len(test_results['error_details']) > 20:
        print(f"... 还有 {len(test_results['error_details']) - 20} 个测试集错误样本")
    
    # 保存详细报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        'model_name': 'mic_mobilenetv3',
        'dataset': 'bioast_dataset',
        'checkpoint_path': checkpoint_path,
        'analysis_timestamp': timestamp,
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'accuracy': checkpoint.get('accuracy', 'N/A'),
            'loss': checkpoint.get('loss', 'N/A')
        },
        'validation_results': val_results,
        'test_results': test_results,
        'device_used': str(device)
    }
    
    # 保存JSON报告
    os.makedirs('error_analysis', exist_ok=True)
    report_path = f"error_analysis/mic_mobilenetv3_with_filenames_{timestamp}.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细错误分析报告已保存: {report_path}")
    
    # 显示总结
    print("\n" + "=" * 80)
    print("MIC_MOBILENETV3 错误样本分析总结 (包含源文件名)")
    print("=" * 80)
    print(f"模型: mic_mobilenetv3")
    print(f"数据集: bioast_dataset")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")
    if 'accuracy' in checkpoint:
        print(f"Checkpoint验证准确率: {checkpoint['accuracy']:.4f}%")
    
    print(f"\n验证集结果:")
    print(f"  总样本: {val_results['total_samples']}")
    print(f"  错误样本: {val_results['error_samples']}")
    print(f"  准确率: {val_results['accuracy']:.4f}%")
    print(f"  假阳性: {val_results['false_positives']}")
    print(f"  假阴性: {val_results['false_negatives']}")
    
    print(f"\n测试集结果:")
    print(f"  总样本: {test_results['total_samples']}")
    print(f"  错误样本: {test_results['error_samples']}")
    print(f"  准确率: {test_results['accuracy']:.4f}%")
    print(f"  假阳性: {test_results['false_positives']}")
    print(f"  假阴性: {test_results['false_negatives']}")
    
    print(f"\n详细报告: {report_path}")
    print("=" * 80)
    
    print(f"\n✅ mic_mobilenetv3错误样本分析完成 (包含源文件名)!")
    
    return report

if __name__ == "__main__":
    main()