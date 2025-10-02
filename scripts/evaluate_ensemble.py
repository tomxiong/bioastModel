#!/usr/bin/env python3
"""
Ensemble Evaluation Script for MobileNetV4 v1.3
评估三个模型的集成性能
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv4 import create_multilevel_mobilenetv4_small
from training.multilevel_dataset import create_multilevel_dataloaders


def load_models(model_paths, device):
    """加载三个训练好的模型"""
    models = []
    for model_path in model_paths:
        model = create_multilevel_mobilenetv4_small(
            input_channels=1,
            dropout_rate=0.3
        )

        # 加载权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        models.append(model)
        print(f"Loaded model from {model_path}")

    return models


def ensemble_predict(models, data, device):
    """
    Ensemble预测 (软投票)

    Args:
        models: 模型列表
        data: 输入数据 (images, targets)
        device: 设备

    Returns:
        预测结果字典
    """
    images, _ = data
    images = images.to(device)

    all_predictions = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': []
    }

    # 收集所有模型的预测
    with torch.no_grad():
        for model in models:
            pred = model.predict(images)
            all_predictions['growth_level'].append(pred['growth_level'])
            all_predictions['growth_pattern'].append(pred['growth_pattern'])
            all_predictions['interference_factors'].append(pred['interference_factors'])

    # 对每个任务进行概率平均 (软投票)
    ensemble_pred = {
        'growth_level': torch.mean(torch.stack(all_predictions['growth_level']), dim=0),
        'growth_pattern': torch.mean(torch.stack(all_predictions['growth_pattern']), dim=0),
        'interference_factors': torch.mean(torch.stack(all_predictions['interference_factors']), dim=0)
    }

    return ensemble_pred


def evaluate_single_model(model, test_loader, device):
    """评估单个模型的性能"""
    model.eval()

    correct = {
        'growth_level': 0,
        'growth_pattern': 0,
        'interference_factors': 0
    }
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating single model"):
            images, targets = data
            images = images.to(device)
            labels = {
                'growth_level': targets['growth_level'].to(device),
                'growth_pattern': targets['growth_pattern'].to(device),
                'interference_factors': targets['interference_factors'].to(device)
            }

            predictions = model.predict(images)

            # Growth Level (分类)
            pred_level = torch.argmax(predictions['growth_level'], dim=1)
            correct['growth_level'] += (pred_level == labels['growth_level']).sum().item()

            # Growth Pattern (分类)
            pred_pattern = torch.argmax(predictions['growth_pattern'], dim=1)
            correct['growth_pattern'] += (pred_pattern == labels['growth_pattern']).sum().item()

            # Interference Factors (多标签)
            pred_interference = (predictions['interference_factors'] > 0.5).float()
            correct['interference_factors'] += (pred_interference == labels['interference_factors']).all(dim=1).sum().item()

            total += images.size(0)

    accuracies = {
        'growth_level': correct['growth_level'] / total,
        'growth_pattern': correct['growth_pattern'] / total,
        'interference_factors': correct['interference_factors'] / total
    }

    # 计算加权平均
    weighted_acc = (accuracies['growth_level'] + accuracies['growth_pattern'] + accuracies['interference_factors']) / 3

    return accuracies, weighted_acc


def evaluate_ensemble(models, test_loader, device):
    """评估Ensemble的性能"""
    correct = {
        'growth_level': 0,
        'growth_pattern': 0,
        'interference_factors': 0
    }
    total = 0

    for model in models:
        model.eval()

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating ensemble"):
            images, targets = data
            labels = {
                'growth_level': targets['growth_level'].to(device),
                'growth_pattern': targets['growth_pattern'].to(device),
                'interference_factors': targets['interference_factors'].to(device)
            }

            # Ensemble预测
            predictions = ensemble_predict(models, data, device)

            # Growth Level
            pred_level = torch.argmax(predictions['growth_level'], dim=1)
            correct['growth_level'] += (pred_level == labels['growth_level']).sum().item()

            # Growth Pattern
            pred_pattern = torch.argmax(predictions['growth_pattern'], dim=1)
            correct['growth_pattern'] += (pred_pattern == labels['growth_pattern']).sum().item()

            # Interference Factors
            pred_interference = (predictions['interference_factors'] > 0.5).float()
            correct['interference_factors'] += (pred_interference == labels['interference_factors']).all(dim=1).sum().item()

            total += images.size(0)

    accuracies = {
        'growth_level': correct['growth_level'] / total,
        'growth_pattern': correct['growth_pattern'] / total,
        'interference_factors': correct['interference_factors'] / total
    }

    # 计算加权平均
    weighted_acc = (accuracies['growth_level'] + accuracies['growth_pattern'] + accuracies['interference_factors']) / 3

    return accuracies, weighted_acc


def main():
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV4 Ensemble')
    parser.add_argument('--model1', type=str, required=True, help='Path to model 1')
    parser.add_argument('--model2', type=str, required=True, help='Path to model 2')
    parser.add_argument('--model3', type=str, required=True, help='Path to model 3')
    parser.add_argument('--json_path', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='Path to dataset JSON')
    parser.add_argument('--image_root', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images',
                       help='Root directory for images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 创建数据加载器
    print("Loading dataset...")
    _, _, test_loader, label_info = create_multilevel_dataloaders(
        json_path=args.json_path,
        image_root=args.image_root,
        batch_size=args.batch_size,
        split_ratio=(0.7, 0.15, 0.15)
    )

    print(f"Test set size: {len(test_loader.dataset)}\n")

    # 加载模型
    print("Loading models...")
    model_paths = [args.model1, args.model2, args.model3]
    models = load_models(model_paths, device)
    print()

    # 评估每个单模型
    print("=" * 80)
    print("Evaluating individual models...")
    print("=" * 80)

    single_results = []
    for i, model in enumerate(models, 1):
        print(f"\nModel {i}:")
        acc, weighted_acc = evaluate_single_model(model, test_loader, device)
        print(f"  Growth Level: {acc['growth_level']:.4f}")
        print(f"  Growth Pattern: {acc['growth_pattern']:.4f}")
        print(f"  Interference: {acc['interference_factors']:.4f}")
        print(f"  Weighted Avg: {weighted_acc:.4f}")

        single_results.append({
            'model_id': i,
            'accuracies': {k: float(v) for k, v in acc.items()},
            'weighted_accuracy': float(weighted_acc)
        })

    # 评估Ensemble
    print("\n" + "=" * 80)
    print("Evaluating ensemble...")
    print("=" * 80)

    ensemble_acc, ensemble_weighted_acc = evaluate_ensemble(models, test_loader, device)
    print(f"\nEnsemble Results:")
    print(f"  Growth Level: {ensemble_acc['growth_level']:.4f}")
    print(f"  Growth Pattern: {ensemble_acc['growth_pattern']:.4f}")
    print(f"  Interference: {ensemble_acc['interference_factors']:.4f}")
    print(f"  Weighted Avg: {ensemble_weighted_acc:.4f}")

    # 计算最佳单模型
    best_single_idx = max(range(len(single_results)), key=lambda i: single_results[i]['weighted_accuracy'])
    best_single_acc = single_results[best_single_idx]['weighted_accuracy']

    # 计算Ensemble提升
    improvement = ensemble_weighted_acc - best_single_acc

    print(f"\n" + "=" * 80)
    print(f"Best single model: Model {best_single_idx + 1} ({best_single_acc:.4f})")
    print(f"Ensemble improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    print("=" * 80)

    # 保存结果
    results = {
        'single_models': single_results,
        'ensemble': {
            'accuracies': {k: float(v) for k, v in ensemble_acc.items()},
            'weighted_accuracy': float(ensemble_weighted_acc)
        },
        'best_single_model': {
            'model_id': best_single_idx + 1,
            'weighted_accuracy': float(best_single_acc)
        },
        'improvement': {
            'absolute': float(improvement),
            'percentage': float(improvement * 100)
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
