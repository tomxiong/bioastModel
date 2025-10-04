#!/usr/bin/env python3
"""
MobileNetV4 v0.11.0 综合性能评估
包括: 整体性能、混淆矩阵、错误样本分析、业务关键样本分析
"""

import torch
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv4 import create_multilevel_mobilenetv4_small
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def analyze_business_critical_samples(
    all_preds, all_targets, all_patterns, all_levels,
    pattern_mapping, interference_names
):
    """分析业务关键样本性能"""

    # 反向映射
    idx_to_pattern = {v: k for k, v in pattern_mapping.items()}

    # 定义业务关键 pattern
    critical_patterns = ['center_dots', 'weak_scattered_pos']

    # 统计
    stats = {
        'negative_samples': {'total': 0, 'pores_correct': 0, 'pores_fn': []},
        'positive_critical': {'total': 0, 'pores_correct': 0, 'pores_fn': []},
        'other_positive': {'total': 0, 'pores_correct': 0, 'pores_fn': []}
    }

    pores_idx = interference_names.index('pores')

    for i in range(len(all_levels)):
        level = all_levels[i]
        pattern_idx = all_patterns[i]
        pattern_name = idx_to_pattern.get(pattern_idx, 'unknown')

        pores_pred = all_preds['interference_factors'][i, pores_idx]
        pores_target = all_targets['interference_factors'][i, pores_idx]

        if level == 0:  # Negative
            stats['negative_samples']['total'] += 1
            if pores_pred == pores_target:
                stats['negative_samples']['pores_correct'] += 1
            elif pores_target == 1 and pores_pred == 0:  # FN
                stats['negative_samples']['pores_fn'].append({
                    'sample_idx': i,
                    'pattern': pattern_name,
                    'target': int(pores_target),
                    'pred': int(pores_pred)
                })

        elif pattern_name in critical_patterns:  # Positive Critical
            stats['positive_critical']['total'] += 1
            if pores_pred == pores_target:
                stats['positive_critical']['pores_correct'] += 1
            elif pores_target == 1 and pores_pred == 0:  # FN
                stats['positive_critical']['pores_fn'].append({
                    'sample_idx': i,
                    'pattern': pattern_name,
                    'target': int(pores_target),
                    'pred': int(pores_pred)
                })

        else:  # Other Positive
            stats['other_positive']['total'] += 1
            if pores_pred == pores_target:
                stats['other_positive']['pores_correct'] += 1
            elif pores_target == 1 and pores_pred == 0:  # FN
                stats['other_positive']['pores_fn'].append({
                    'sample_idx': i,
                    'pattern': pattern_name,
                    'target': int(pores_target),
                    'pred': int(pores_pred)
                })

    # 计算召回率
    for key in stats:
        if stats[key]['total'] > 0:
            stats[key]['recall'] = stats[key]['pores_correct'] / stats[key]['total']
        else:
            stats[key]['recall'] = 0.0

    return stats


def evaluate_model():
    # 加载模型
    print('='*80)
    print('MobileNetV4 v0.11.0 综合性能评估')
    print('='*80)
    print('\n[1/6] 加载模型...')

    model = create_multilevel_mobilenetv4_small()
    checkpoint = torch.load(
        'experiments/multilevel_mobilenetv4_v0.11.0/best_model.pth',
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'  ✓ 模型加载完成,设备: {device}')

    # 加载测试集
    print('\n[2/6] 加载测试数据集...')
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

    pattern_mapping = test_dataset.label_mappings['growth_pattern']
    level_mapping = test_dataset.label_mappings['growth_level']
    interference_names = list(test_dataset.label_mappings['interference_factors'].keys())

    print(f'  ✓ 测试集加载完成: {len(test_dataset)} 样本')
    print(f'  ✓ Pattern 映射: {len(pattern_mapping)} 类')
    print(f'  ✓ Interference 因子: {interference_names}')

    # 推理
    print('\n[3/6] 模型推理...')
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
    all_probs = {
        'interference_factors': []
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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

            # Interference factors
            interference_probs = torch.sigmoid(outputs['interference_factors']).cpu()
            pred_interference = (interference_probs > 0.5).float()

            all_preds['interference_factors'].append(pred_interference)
            all_targets['interference_factors'].append(labels['interference_factors'].float())
            all_probs['interference_factors'].append(interference_probs)

            if (batch_idx + 1) % 10 == 0:
                print(f'  进度: {batch_idx + 1}/{len(test_loader)} batches')

    # 拼接
    all_preds['interference_factors'] = torch.cat(
        all_preds['interference_factors'], dim=0
    ).numpy()
    all_targets['interference_factors'] = torch.cat(
        all_targets['interference_factors'], dim=0
    ).numpy()
    all_probs['interference_factors'] = torch.cat(
        all_probs['interference_factors'], dim=0
    ).numpy()

    print('  ✓ 推理完成')

    # 计算整体指标
    print('\n[4/6] 计算整体性能指标...')
    results = {'overall': {}, 'tasks': {}}

    # Growth Level
    acc_level = accuracy_score(all_targets['growth_level'], all_preds['growth_level'])
    p_level, r_level, f1_level, _ = precision_recall_fscore_support(
        all_targets['growth_level'], all_preds['growth_level'],
        average='binary', zero_division=0
    )

    cm_level = confusion_matrix(all_targets['growth_level'], all_preds['growth_level'])

    results['tasks']['growth_level'] = {
        'accuracy': float(acc_level),
        'precision': float(p_level),
        'recall': float(r_level),
        'f1': float(f1_level),
        'confusion_matrix': cm_level.tolist()
    }

    # Growth Pattern
    acc_pattern = accuracy_score(all_targets['growth_pattern'], all_preds['growth_pattern'])
    p_pattern, r_pattern, f1_pattern, _ = precision_recall_fscore_support(
        all_targets['growth_pattern'], all_preds['growth_pattern'],
        average='weighted', zero_division=0
    )

    cm_pattern = confusion_matrix(all_targets['growth_pattern'], all_preds['growth_pattern'])

    results['tasks']['growth_pattern'] = {
        'accuracy': float(acc_pattern),
        'precision': float(p_pattern),
        'recall': float(r_pattern),
        'f1': float(f1_pattern),
        'confusion_matrix': cm_pattern.tolist()
    }

    # Interference Factors
    results['tasks']['interference_factors'] = {}
    f1_scores = []

    for i, label in enumerate(interference_names):
        p, r, f1, _ = precision_recall_fscore_support(
            all_targets['interference_factors'][:, i],
            all_preds['interference_factors'][:, i],
            average='binary', zero_division=0
        )
        acc = accuracy_score(
            all_targets['interference_factors'][:, i],
            all_preds['interference_factors'][:, i]
        )

        cm = confusion_matrix(
            all_targets['interference_factors'][:, i],
            all_preds['interference_factors'][:, i]
        )

        results['tasks']['interference_factors'][label] = {
            'accuracy': float(acc),
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'confusion_matrix': cm.tolist()
        }
        f1_scores.append(f1)

    results['tasks']['interference_factors']['overall_f1'] = float(
        sum(f1_scores) / len(f1_scores)
    )

    # 整体准确率
    results['overall']['growth_level_acc'] = float(acc_level)
    results['overall']['growth_pattern_acc'] = float(acc_pattern)
    results['overall']['interference_overall_f1'] = float(sum(f1_scores) / len(f1_scores))
    results['overall']['pores_f1'] = float(results['tasks']['interference_factors']['pores']['f1'])

    print('  ✓ 整体指标计算完成')

    # 错误样本分析
    print('\n[5/6] 错误样本分析...')

    # Pores 错误分析
    pores_idx = interference_names.index('pores')
    pores_errors = {
        'false_negatives': [],
        'false_positives': []
    }

    for i in range(len(all_targets['interference_factors'])):
        target = all_targets['interference_factors'][i, pores_idx]
        pred = all_preds['interference_factors'][i, pores_idx]
        prob = all_probs['interference_factors'][i, pores_idx]

        pattern_idx = all_preds['growth_pattern'][i]
        pattern_name = {v: k for k, v in pattern_mapping.items()}.get(pattern_idx, 'unknown')

        if target == 1 and pred == 0:  # FN
            pores_errors['false_negatives'].append({
                'sample_idx': i,
                'pattern': pattern_name,
                'probability': float(prob),
                'threshold': 0.5
            })
        elif target == 0 and pred == 1:  # FP
            pores_errors['false_positives'].append({
                'sample_idx': i,
                'pattern': pattern_name,
                'probability': float(prob),
                'threshold': 0.5
            })

    results['error_analysis'] = {
        'pores': pores_errors
    }

    print(f'  ✓ Pores FN: {len(pores_errors["false_negatives"])} 个')
    print(f'  ✓ Pores FP: {len(pores_errors["false_positives"])} 个')

    # 业务关键样本分析
    print('\n[6/6] 业务关键样本分析...')

    business_stats = analyze_business_critical_samples(
        all_preds, all_targets,
        all_preds['growth_pattern'],
        all_targets['growth_level'],
        pattern_mapping,
        interference_names
    )

    results['business_critical'] = business_stats

    print(f'  ✓ Negative 样本: {business_stats["negative_samples"]["total"]} 个')
    print(f'    - Pores Recall: {business_stats["negative_samples"]["recall"]:.4f}')
    print(f'  ✓ Positive Critical 样本: {business_stats["positive_critical"]["total"]} 个')
    print(f'    - Pores Recall: {business_stats["positive_critical"]["recall"]:.4f}')
    print(f'  ✓ Other Positive 样本: {business_stats["other_positive"]["total"]} 个')
    print(f'    - Pores Recall: {business_stats["other_positive"]["recall"]:.4f}')

    # 保存结果
    output_path = 'experiments/multilevel_mobilenetv4_v0.11.0/comprehensive_evaluation.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print('\n' + '='*80)
    print('MobileNetV4 v0.11.0 综合评估摘要')
    print('='*80)

    print('\n【整体性能】')
    print(f"  Growth Level Acc:      {results['overall']['growth_level_acc']:.4f}")
    print(f"  Growth Pattern Acc:    {results['overall']['growth_pattern_acc']:.4f}")
    print(f"  Interference Overall:  {results['overall']['interference_overall_f1']:.4f}")
    print(f"  Pores F1:              {results['overall']['pores_f1']:.4f}")

    print('\n【Pores 详细性能】')
    pores_metrics = results['tasks']['interference_factors']['pores']
    print(f"  Accuracy:   {pores_metrics['accuracy']:.4f}")
    print(f"  Precision:  {pores_metrics['precision']:.4f}")
    print(f"  Recall:     {pores_metrics['recall']:.4f}")
    print(f"  F1:         {pores_metrics['f1']:.4f}")

    print('\n【Pores 混淆矩阵】')
    cm = np.array(pores_metrics['confusion_matrix'])
    print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")

    print('\n【业务关键样本 Pores Recall】')
    print(f"  Negative 样本:        {business_stats['negative_samples']['recall']:.4f} ({business_stats['negative_samples']['pores_correct']}/{business_stats['negative_samples']['total']})")
    print(f"  Positive Critical:    {business_stats['positive_critical']['recall']:.4f} ({business_stats['positive_critical']['pores_correct']}/{business_stats['positive_critical']['total']})")
    print(f"  Other Positive:       {business_stats['other_positive']['recall']:.4f} ({business_stats['other_positive']['pores_correct']}/{business_stats['other_positive']['total']})")

    print('\n【错误样本统计】')
    print(f"  Pores FN (漏检):      {len(pores_errors['false_negatives'])} 个")
    print(f"  Pores FP (误检):      {len(pores_errors['false_positives'])} 个")

    print('='*80)
    print(f'\n✅ 完整评估结果已保存到: {output_path}')

    return results


if __name__ == '__main__':
    evaluate_model()
