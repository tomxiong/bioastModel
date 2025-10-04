#!/usr/bin/env python3
"""
评估 MobileNetV3 v0.10.0 模型性能
"""

import torch
import json
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_model():
    # 加载模型
    print('加载 MobileNetV3 v0.10.0 模型...')
    model = create_multilevel_mobilenetv3(model_size='small')
    checkpoint = torch.load(
        'experiments/multilevel_mobilenetv3_v0.10.0/best_model.pth',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'模型加载完成,使用设备: {device}')

    # 加载测试集
    print('\n加载测试数据集...')
    test_dataset = EnhancedMultitaskDataset(
        data_root='ds/images',
        split='test',
        split_file='ds/images/dataset_split_seed44.json',
        annotations_file='m9e1n170_cleaned_round2.json',
        transform=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    # 获取标签顺序
    interference_label_names = list(
        test_dataset.label_mappings['interference_factors'].keys()
    )
    print(f'Interference factors 标签顺序: {interference_label_names}')

    # 评估
    print('\n开始评估...')
    all_preds = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': []
    }
    all_targets = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': []
    }

    with torch.no_grad():
        for batch in test_loader:
            # Batch format: [images_tensor, labels_dict]
            images, labels = batch
            images = images.to(device)
            outputs = model(images)

            # Growth level
            pred_level = torch.argmax(outputs['growth_level'], dim=1).cpu()
            all_preds['growth_level'].extend(pred_level.tolist())
            all_targets['growth_level'].extend(labels['growth_level'].tolist())

            # Growth pattern
            pred_pattern = torch.argmax(outputs['growth_pattern'], dim=1).cpu()
            all_preds['growth_pattern'].extend(pred_pattern.tolist())
            all_targets['growth_pattern'].extend(labels['growth_pattern'].tolist())

            # Interference factors (multi-label, threshold 0.5)
            pred_interference = (
                torch.sigmoid(outputs['interference_factors']) > 0.5
            ).float().cpu()
            all_preds['interference_factors'].append(pred_interference)
            all_targets['interference_factors'].append(
                labels['interference_factors'].float()
            )

    # 拼接 interference factors
    all_preds['interference_factors'] = torch.cat(
        all_preds['interference_factors'], dim=0
    ).numpy()
    all_targets['interference_factors'] = torch.cat(
        all_targets['interference_factors'], dim=0
    ).numpy()

    # 计算指标
    results = {}

    # Growth Level
    print('\n计算 Growth Level 指标...')
    acc_level = accuracy_score(
        all_targets['growth_level'],
        all_preds['growth_level']
    )
    p_level, r_level, f1_level, _ = precision_recall_fscore_support(
        all_targets['growth_level'],
        all_preds['growth_level'],
        average='binary',
        zero_division=0
    )
    results['growth_level'] = {
        'accuracy': float(acc_level),
        'precision': float(p_level),
        'recall': float(r_level),
        'f1': float(f1_level)
    }

    # Growth Pattern
    print('计算 Growth Pattern 指标...')
    acc_pattern = accuracy_score(
        all_targets['growth_pattern'],
        all_preds['growth_pattern']
    )
    p_pattern, r_pattern, f1_pattern, _ = precision_recall_fscore_support(
        all_targets['growth_pattern'],
        all_preds['growth_pattern'],
        average='weighted',
        zero_division=0
    )
    results['growth_pattern'] = {
        'accuracy': float(acc_pattern),
        'precision': float(p_pattern),
        'recall': float(r_pattern),
        'f1': float(f1_pattern)
    }

    # Interference Factors
    print('计算 Interference Factors 指标...')
    results['interference_factors'] = {}
    f1_scores = []

    for i, label in enumerate(interference_label_names):
        p, r, f1, _ = precision_recall_fscore_support(
            all_targets['interference_factors'][:, i],
            all_preds['interference_factors'][:, i],
            average='binary',
            zero_division=0
        )
        acc = accuracy_score(
            all_targets['interference_factors'][:, i],
            all_preds['interference_factors'][:, i]
        )
        results['interference_factors'][label] = {
            'accuracy': float(acc),
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1)
        }
        f1_scores.append(f1)

    results['interference_factors']['overall_f1'] = float(
        sum(f1_scores) / len(f1_scores)
    )

    # 保存结果
    output_path = 'experiments/multilevel_mobilenetv3_v0.10.0/test_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印关键指标
    print('\n' + '='*80)
    print('MobileNetV3 v0.10.0 测试结果 (Threshold 0.5)')
    print('='*80)
    print(f"\nGrowth Level:")
    print(f"  Accuracy:  {acc_level:.4f}")
    print(f"  Precision: {p_level:.4f}")
    print(f"  Recall:    {r_level:.4f}")
    print(f"  F1:        {f1_level:.4f}")

    print(f"\nGrowth Pattern:")
    print(f"  Accuracy:  {acc_pattern:.4f}")
    print(f"  Precision: {p_pattern:.4f}")
    print(f"  Recall:    {r_pattern:.4f}")
    print(f"  F1:        {f1_pattern:.4f}")

    print(f"\nInterference Factors (Overall F1: {results['interference_factors']['overall_f1']:.4f}):")
    for label in interference_label_names:
        metrics = results['interference_factors'][label]
        print(f"  {label:15s} - F1: {metrics['f1']:.4f}, "
              f"P: {metrics['precision']:.4f}, "
              f"R: {metrics['recall']:.4f}, "
              f"Acc: {metrics['accuracy']:.4f}")

    print('='*80)
    print(f'\n✅ 结果已保存到: {output_path}')


if __name__ == '__main__':
    evaluate_model()
