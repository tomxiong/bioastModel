"""
深度分析多任务模型在生长模式和干扰因素识别上的表现
特别关注阴性样本下的气孔识别问题
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter, defaultdict

# 添加项目根路径
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.multitask_mic_mobilenetv3 import create_multitask_mic_mobilenetv3
from training.enhanced_multitask_dataset import create_multitask_dataloaders

def analyze_multitask_performance(experiment_dir="experiments/multitask_grayscale_focused"):
    """深度分析多任务模型性能"""
    
    experiment_path = Path(experiment_dir)
    
    print("🔍 多任务模型深度性能分析")
    print("=" * 60)
    
    # 1. 加载最佳模型
    model_path = experiment_path / "best_model.pth"
    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    print(f"📊 加载最佳模型 (Epoch {checkpoint['epoch']})")
    print(f"   最佳性能: {checkpoint['best_metrics']}")
    
    # 2. 创建数据加载器
    print("\n📊 创建数据加载器...")
    dataloaders = create_multitask_dataloaders(
        data_root="/home/aaa/ws/bioastModel/ds/images",
        annotations_file="m9e1n170.json",
        batch_size=64,
        num_workers=4,
        seed=42
    )
    
    # 3. 重建模型
    dataset = next(iter(dataloaders.values())).dataset
    model = create_multitask_mic_mobilenetv3(
        num_classes=2,
        num_growth_patterns=len(dataset.label_mappings['growth_pattern']),
        num_interference_factors=len(dataset.label_mappings['interference_factors']),
        width_mult=1.0
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"✅ 模型加载完成，设备: {device}")
    
    # 4. 详细评估
    results = {}
    
    for split_name, dataloader in dataloaders.items():
        print(f"\n🔍 评估 {split_name.upper()} 集...")
        
        all_predictions = {
            'classification': [],
            'growth_pattern': [],
            'interference_factors': []
        }
        all_targets = {
            'classification': [],
            'growth_pattern': [],
            'interference_factors': []
        }
        all_growth_levels = []  # 用于分析阴性/阳性
        all_confidences = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(device)
                
                # 前向传播
                outputs = model(images)
                
                # 收集预测
                # 主分类
                main_preds = torch.argmax(outputs['classification'], dim=1).cpu()
                all_predictions['classification'].extend(main_preds.numpy())
                all_targets['classification'].extend(targets['classification'].numpy())
                all_growth_levels.extend(targets['classification'].numpy())
                
                # 生长模式
                pattern_preds = torch.argmax(outputs['growth_pattern'], dim=1).cpu()
                all_predictions['growth_pattern'].extend(pattern_preds.numpy())
                all_targets['growth_pattern'].extend(targets['growth_pattern'].numpy())
                
                # 干扰因素 (多标签)
                interference_preds = (torch.sigmoid(outputs['interference_factors']) > 0.5).cpu()
                all_predictions['interference_factors'].extend(interference_preds.numpy())
                all_targets['interference_factors'].extend(targets['interference_factors'].numpy())
                
                # 主分类置信度
                main_probs = torch.softmax(outputs['classification'], dim=1)
                main_conf = torch.max(main_probs, dim=1)[0].cpu()
                all_confidences.extend(main_conf.numpy())
        
        # 转换为numpy数组
        for key in all_predictions:
            all_predictions[key] = np.array(all_predictions[key])
            all_targets[key] = np.array(all_targets[key])
        
        all_growth_levels = np.array(all_growth_levels)
        all_confidences = np.array(all_confidences)
        
        # 5. 详细分析结果
        split_results = analyze_split_performance(
            all_predictions, all_targets, all_growth_levels, all_confidences,
            dataset.label_mappings, split_name
        )
        
        results[split_name] = split_results
    
    # 6. 生成改进建议
    print("\n" + "="*60)
    print("📈 模型改进建议")
    print("="*60)
    
    generate_improvement_suggestions(results, dataset.label_mappings)
    
    return results

def analyze_split_performance(predictions, targets, growth_levels, confidences, label_mappings, split_name):
    """分析单个数据集划分的性能"""
    
    results = {}
    
    # 1. 主分类分析
    print(f"\n🎯 {split_name} - 主分类分析:")
    main_acc = (predictions['classification'] == targets['classification']).mean()
    print(f"   整体准确率: {main_acc:.4f}")
    
    # 按阴性/阳性分析
    negative_mask = (growth_levels == 0)
    positive_mask = (growth_levels == 1)
    
    neg_acc = (predictions['classification'][negative_mask] == targets['classification'][negative_mask]).mean()
    pos_acc = (predictions['classification'][positive_mask] == targets['classification'][positive_mask]).mean()
    
    print(f"   阴性准确率: {neg_acc:.4f} ({negative_mask.sum()}样本)")
    print(f"   阳性准确率: {pos_acc:.4f} ({positive_mask.sum()}样本)")
    
    results['main_classification'] = {
        'overall_acc': main_acc,
        'negative_acc': neg_acc,
        'positive_acc': pos_acc,
        'negative_count': negative_mask.sum(),
        'positive_count': positive_mask.sum()
    }
    
    # 2. 生长模式分析
    print(f"\n🌱 {split_name} - 生长模式分析:")
    pattern_acc = (predictions['growth_pattern'] == targets['growth_pattern']).mean()
    print(f"   整体准确率: {pattern_acc:.4f}")
    
    # 按阴性/阳性分析生长模式
    neg_pattern_acc = (predictions['growth_pattern'][negative_mask] == targets['growth_pattern'][negative_mask]).mean()
    pos_pattern_acc = (predictions['growth_pattern'][positive_mask] == targets['growth_pattern'][positive_mask]).mean()
    
    print(f"   阴性样本生长模式准确率: {neg_pattern_acc:.4f}")
    print(f"   阳性样本生长模式准确率: {pos_pattern_acc:.4f}")
    
    # 生长模式混淆矩阵
    pattern_labels = list(label_mappings['growth_pattern'].keys())
    pattern_cm = confusion_matrix(targets['growth_pattern'], predictions['growth_pattern'])
    
    # 找出表现最差的生长模式
    pattern_recalls = pattern_cm.diagonal() / pattern_cm.sum(axis=1)
    worst_patterns = np.argsort(pattern_recalls)[:3]
    
    print(f"   表现最差的生长模式:")
    for idx in worst_patterns:
        if idx < len(pattern_labels):
            print(f"     {pattern_labels[idx]}: {pattern_recalls[idx]:.4f}")
    
    results['growth_pattern'] = {
        'overall_acc': pattern_acc,
        'negative_acc': neg_pattern_acc,
        'positive_acc': pos_pattern_acc,
        'confusion_matrix': pattern_cm,
        'worst_patterns': [(pattern_labels[idx], pattern_recalls[idx]) for idx in worst_patterns if idx < len(pattern_labels)]
    }
    
    # 3. 干扰因素分析 (重点关注气孔)
    print(f"\n🔬 {split_name} - 干扰因素分析:")
    
    interference_labels = list(label_mappings['interference_factors'].keys())
    pore_idx = interference_labels.index('pores') if 'pores' in interference_labels else 0
    
    # 整体F1分数
    overall_f1 = f1_score(targets['interference_factors'], predictions['interference_factors'], average='macro', zero_division=0)
    print(f"   整体F1分数: {overall_f1:.4f}")
    
    # 按每个干扰因素分析
    for i, factor in enumerate(interference_labels):
        factor_f1 = f1_score(targets['interference_factors'][:, i], predictions['interference_factors'][:, i], zero_division=0)
        print(f"   {factor} F1: {factor_f1:.4f}")
    
    # 特别分析气孔检测
    print(f"\n🔍 气孔检测深度分析:")
    pore_targets = targets['interference_factors'][:, pore_idx]
    pore_predictions = predictions['interference_factors'][:, pore_idx]
    
    # 整体气孔检测
    pore_f1 = f1_score(pore_targets, pore_predictions, zero_division=0)
    print(f"   气孔整体F1: {pore_f1:.4f}")
    
    # 阴性样本中的气孔检测
    neg_pore_targets = pore_targets[negative_mask]
    neg_pore_predictions = pore_predictions[negative_mask]
    neg_pore_f1 = f1_score(neg_pore_targets, neg_pore_predictions, zero_division=0) if len(neg_pore_targets) > 0 else 0
    
    # 阳性样本中的气孔检测
    pos_pore_targets = pore_targets[positive_mask]
    pos_pore_predictions = pore_predictions[positive_mask]
    pos_pore_f1 = f1_score(pos_pore_targets, pos_pore_predictions, zero_division=0) if len(pos_pore_targets) > 0 else 0
    
    print(f"   阴性样本气孔F1: {neg_pore_f1:.4f} ({negative_mask.sum()}样本)")
    print(f"   阳性样本气孔F1: {pos_pore_f1:.4f} ({positive_mask.sum()}样本)")
    
    # 气孔检测错误分析
    neg_pore_tp = ((neg_pore_targets == 1) & (neg_pore_predictions == 1)).sum()
    neg_pore_fp = ((neg_pore_targets == 0) & (neg_pore_predictions == 1)).sum()
    neg_pore_fn = ((neg_pore_targets == 1) & (neg_pore_predictions == 0)).sum()
    neg_pore_tn = ((neg_pore_targets == 0) & (neg_pore_predictions == 0)).sum()
    
    print(f"   阴性样本气孔混淆矩阵:")
    print(f"     TP: {neg_pore_tp}, FP: {neg_pore_fp}")
    print(f"     FN: {neg_pore_fn}, TN: {neg_pore_tn}")
    
    if neg_pore_tp + neg_pore_fp > 0:
        neg_pore_precision = neg_pore_tp / (neg_pore_tp + neg_pore_fp)
        print(f"     阴性样本气孔精确率: {neg_pore_precision:.4f}")
    
    if neg_pore_tp + neg_pore_fn > 0:
        neg_pore_recall = neg_pore_tp / (neg_pore_tp + neg_pore_fn)
        print(f"     阴性样本气孔召回率: {neg_pore_recall:.4f}")
    
    results['interference_factors'] = {
        'overall_f1': overall_f1,
        'factor_f1s': {factor: f1_score(targets['interference_factors'][:, i], predictions['interference_factors'][:, i], zero_division=0) 
                      for i, factor in enumerate(interference_labels)},
        'pore_analysis': {
            'overall_f1': pore_f1,
            'negative_f1': neg_pore_f1,
            'positive_f1': pos_pore_f1,
            'negative_confusion': {
                'tp': int(neg_pore_tp), 'fp': int(neg_pore_fp),
                'fn': int(neg_pore_fn), 'tn': int(neg_pore_tn)
            }
        }
    }
    
    return results

def generate_improvement_suggestions(results, label_mappings):
    """生成具体的模型改进建议"""
    
    print("\n🎯 针对性改进建议:")
    
    # 分析验证集结果
    val_results = results.get('val', {})
    
    # 1. 生长模式改进建议
    if 'growth_pattern' in val_results:
        pattern_results = val_results['growth_pattern']
        print(f"\n1️⃣ 生长模式优化建议:")
        print(f"   当前准确率: {pattern_results['overall_acc']:.4f}")
        
        if pattern_results['negative_acc'] < pattern_results['positive_acc']:
            print(f"   ⚠️ 阴性样本生长模式识别较弱 ({pattern_results['negative_acc']:.4f} vs {pattern_results['positive_acc']:.4f})")
            print(f"   💡 建议增强阴性样本的生长模式特征提取")
        
        print(f"   📉 表现最差的生长模式:")
        for pattern, recall in pattern_results['worst_patterns']:
            print(f"     • {pattern}: {recall:.4f}")
            
        print(f"   🔧 改进策略:")
        print(f"     • 增加困难样本的数据增强")
        print(f"     • 调整生长模式分类器的网络深度")
        print(f"     • 使用focal loss针对困难类别")
    
    # 2. 气孔检测改进建议
    if 'interference_factors' in val_results:
        interference_results = val_results['interference_factors']
        pore_analysis = interference_results['pore_analysis']
        
        print(f"\n2️⃣ 气孔检测优化建议:")
        print(f"   整体气孔F1: {pore_analysis['overall_f1']:.4f}")
        print(f"   阴性气孔F1: {pore_analysis['negative_f1']:.4f}")
        print(f"   阳性气孔F1: {pore_analysis['positive_f1']:.4f}")
        
        if pore_analysis['negative_f1'] < 0.6:
            print(f"   🚨 阴性样本气孔检测急需改进！")
            
            confusion = pore_analysis['negative_confusion']
            print(f"   📊 阴性样本气孔检测问题分析:")
            
            if confusion['fn'] > confusion['fp']:
                print(f"     • 主要问题: 漏检 (FN: {confusion['fn']} > FP: {confusion['fp']})")
                print(f"     💡 建议: 降低气孔检测阈值，提高召回率")
            else:
                print(f"     • 主要问题: 误检 (FP: {confusion['fp']} > FN: {confusion['fn']})")
                print(f"     💡 建议: 提高气孔检测阈值，提高精确率")
            
            print(f"   🔧 具体改进策略:")
            print(f"     • 专门针对阴性样本增加气孔标注数据")
            print(f"     • 调整气孔检测分类器的损失权重")
            print(f"     • 增加空间注意力机制专注气孔区域")
            print(f"     • 使用难例挖掘(Hard Negative Mining)")
    
    # 3. 整体架构改进建议
    print(f"\n3️⃣ 整体架构改进建议:")
    print(f"   🔄 多任务平衡优化:")
    print(f"     • 当前权重: 生长模式(0.8), 干扰因素(0.6)")
    print(f"     • 建议: 根据阴性样本表现动态调整权重")
    
    print(f"   📊 数据增强策略:")
    print(f"     • 针对阴性样本增加特定的数据增强")
    print(f"     • 气孔相关的几何变换和噪声添加")
    print(f"     • 生长模式边界的对比度增强")
    
    print(f"   🎯 损失函数优化:")
    print(f"     • 考虑类别平衡的Focal Loss调整")
    print(f"     • 引入对抗性损失增强特征区分度")
    print(f"     • 添加一致性正则化项")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', default='experiments/multitask_grayscale_focused',
                       help='实验目录')
    
    args = parser.parse_args()
    
    # 执行深度分析
    results = analyze_multitask_performance(args.experiment_dir)