#!/usr/bin/env python3
"""
MobileNetV4 Error Analysis Tool
错误样本分析工具

分析模型预测错误的样本，生成详细的错误分析报告
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv4 import (
    create_multilevel_mobilenetv4_small,
    create_multilevel_mobilenetv4_medium,
    create_multilevel_mobilenetv4_large
)
from training.multilevel_dataset import create_multilevel_dataloaders


class ErrorAnalyzer:
    """错误样本分析器"""

    def __init__(self, model, test_loader, label_info, device, output_dir):
        self.model = model
        self.test_loader = test_loader
        self.label_info = label_info
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 错误样本存储
        self.errors = {
            'growth_level': [],
            'growth_pattern': [],
            'interference_factors': []
        }

        # 统计信息
        self.stats = {
            'total_samples': 0,
            'correct_samples': {
                'growth_level': 0,
                'growth_pattern': 0,
                'interference_factors': 0
            },
            'error_samples': {
                'growth_level': 0,
                'growth_pattern': 0,
                'interference_factors': 0
            }
        }

    def analyze(self):
        """执行错误分析"""
        print("\n" + "="*80)
        print("开始错误样本分析...")
        print("="*80)

        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="分析进度")):
                images = batch['image'].to(self.device)

                # 获取真实标签
                true_labels = {
                    'growth_level': batch['growth_level'].to(self.device),
                    'growth_pattern': batch['growth_pattern'].to(self.device),
                    'interference_factors': batch['interference_factors'].to(self.device)
                }

                # 获取预测
                predictions = self.model.predict(images)

                # 分析每个样本
                batch_size = images.size(0)
                for i in range(batch_size):
                    sample_idx = batch_idx * self.test_loader.batch_size + i
                    self.stats['total_samples'] += 1

                    # Growth Level分析
                    self._analyze_task(
                        'growth_level',
                        predictions['growth_level'][i],
                        true_labels['growth_level'][i],
                        sample_idx,
                        is_multilabel=False
                    )

                    # Growth Pattern分析
                    self._analyze_task(
                        'growth_pattern',
                        predictions['growth_pattern'][i],
                        true_labels['growth_pattern'][i],
                        sample_idx,
                        is_multilabel=False
                    )

                    # Interference Factors分析
                    self._analyze_task(
                        'interference_factors',
                        predictions['interference_factors'][i],
                        true_labels['interference_factors'][i],
                        sample_idx,
                        is_multilabel=True
                    )

        # 生成报告
        self._generate_report()

        print("\n分析完成！")
        print(f"报告已保存至: {self.output_dir}")

    def _analyze_task(self, task_name, pred, true, sample_idx, is_multilabel=False):
        """分析单个任务的预测"""
        if is_multilabel:
            # 多标签任务
            pred_binary = (pred > 0.5).long()
            is_correct = (pred_binary == true).all().item()
        else:
            # 单标签任务
            pred_class = pred.argmax().item()
            true_class = true.item()
            is_correct = (pred_class == true_class)

        if is_correct:
            self.stats['correct_samples'][task_name] += 1
        else:
            self.stats['error_samples'][task_name] += 1

            # 记录错误样本
            if is_multilabel:
                error_info = {
                    'sample_idx': sample_idx,
                    'true_labels': true.cpu().numpy().tolist(),
                    'pred_scores': pred.cpu().numpy().tolist(),
                    'pred_labels': pred_binary.cpu().numpy().tolist()
                }
            else:
                confidence = pred.max().item()
                error_info = {
                    'sample_idx': sample_idx,
                    'true_class': true.item(),
                    'pred_class': pred.argmax().item(),
                    'confidence': confidence,
                    'true_class_score': pred[true.item()].item(),
                    'all_scores': pred.cpu().numpy().tolist()
                }

            self.errors[task_name].append(error_info)

    def _generate_report(self):
        """生成错误分析报告"""
        report_path = self.output_dir / 'ERROR_ANALYSIS_REPORT.md'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 错误样本分析报告\n\n")
            f.write(f"**总样本数**: {self.stats['total_samples']}\n\n")
            f.write("---\n\n")

            # 总体统计
            f.write("## 总体统计\n\n")
            f.write("| 任务 | 正确样本 | 错误样本 | 准确率 | 错误率 |\n")
            f.write("|------|---------|---------|--------|--------|\n")

            for task in ['growth_level', 'growth_pattern', 'interference_factors']:
                correct = self.stats['correct_samples'][task]
                error = self.stats['error_samples'][task]
                total = self.stats['total_samples']
                acc = correct / total * 100
                err_rate = error / total * 100

                f.write(f"| {task} | {correct} | {error} | {acc:.2f}% | {err_rate:.2f}% |\n")

            f.write("\n---\n\n")

            # 详细错误分析
            for task in ['growth_level', 'growth_pattern', 'interference_factors']:
                self._write_task_analysis(f, task)

        # 保存错误样本JSON
        errors_json_path = self.output_dir / 'error_samples.json'
        with open(errors_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.errors, f, indent=2, ensure_ascii=False)

        # 保存统计信息JSON
        stats_json_path = self.output_dir / 'error_statistics.json'
        with open(stats_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

    def _write_task_analysis(self, f, task_name):
        """写入单个任务的详细分析"""
        f.write(f"## {task_name.replace('_', ' ').title()} 错误分析\n\n")

        errors = self.errors[task_name]
        total_errors = len(errors)

        f.write(f"**错误样本数**: {total_errors}\n\n")

        if total_errors == 0:
            f.write("✅ 无错误样本！\n\n")
            return

        if task_name in ['growth_level', 'growth_pattern']:
            # 单标签任务分析
            self._write_classification_analysis(f, task_name, errors)
        else:
            # 多标签任务分析
            self._write_multilabel_analysis(f, task_name, errors)

        f.write("\n---\n\n")

    def _write_classification_analysis(self, f, task_name, errors):
        """写入分类任务的错误分析"""
        # 统计混淆情况
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        low_confidence_errors = []

        for error in errors:
            true_class = error['true_class']
            pred_class = error['pred_class']
            confidence = error['confidence']

            confusion_matrix[true_class][pred_class] += 1

            if confidence < 0.7:
                low_confidence_errors.append(error)

        # 写入混淆矩阵
        f.write("### 混淆矩阵分析\n\n")
        f.write("| 真实类别 → 预测类别 | 错误次数 |\n")
        f.write("|---------------------|----------|\n")

        confusion_list = []
        for true_cls, pred_dict in confusion_matrix.items():
            for pred_cls, count in pred_dict.items():
                confusion_list.append((count, true_cls, pred_cls))

        confusion_list.sort(reverse=True)

        for count, true_cls, pred_cls in confusion_list[:10]:  # Top 10
            f.write(f"| {true_cls} → {pred_cls} | {count} |\n")

        # 低置信度错误
        f.write(f"\n### 低置信度错误 (confidence < 0.7)\n\n")
        f.write(f"**数量**: {len(low_confidence_errors)} / {len(errors)} ({len(low_confidence_errors)/len(errors)*100:.1f}%)\n\n")

        if low_confidence_errors:
            f.write("| 样本索引 | 真实类别 | 预测类别 | 置信度 |\n")
            f.write("|---------|---------|---------|--------|\n")

            for error in sorted(low_confidence_errors, key=lambda x: x['confidence'])[:20]:
                f.write(f"| {error['sample_idx']} | {error['true_class']} | {error['pred_class']} | {error['confidence']:.3f} |\n")

    def _write_multilabel_analysis(self, f, task_name, errors):
        """写入多标签任务的错误分析"""
        # 统计错误类型
        false_positive = defaultdict(int)
        false_negative = defaultdict(int)

        for error in errors:
            true_labels = error['true_labels']
            pred_labels = error['pred_labels']

            for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
                if true == 0 and pred == 1:
                    false_positive[i] += 1
                elif true == 1 and pred == 0:
                    false_negative[i] += 1

        f.write("### 错误类型分布\n\n")
        f.write("| 标签索引 | False Positive | False Negative |\n")
        f.write("|---------|----------------|----------------|\n")

        for i in range(len(error['true_labels'])):
            fp = false_positive.get(i, 0)
            fn = false_negative.get(i, 0)
            f.write(f"| {i} | {fp} | {fn} |\n")


def parse_args():
    parser = argparse.ArgumentParser(description='MobileNetV4 Error Analysis')

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Model size')
    parser.add_argument('--json_path', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='Path to dataset JSON')
    parser.add_argument('--image_root', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images',
                       help='Image root directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for analysis results')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据加载器
    print("\nLoading dataset...")
    _, _, test_loader, label_info = create_multilevel_dataloaders(
        json_path=args.json_path,
        image_root=args.image_root,
        batch_size=args.batch_size,
        split_ratio=(0.7, 0.15, 0.15)
    )

    # 创建模型
    print("\nLoading model...")
    if args.model_size == 'small':
        model = create_multilevel_mobilenetv4_small()
    elif args.model_size == 'medium':
        model = create_multilevel_mobilenetv4_medium()
    else:
        model = create_multilevel_mobilenetv4_large()

    # 加载权重
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    print(f"Model loaded from: {args.checkpoint}")

    # 创建分析器
    analyzer = ErrorAnalyzer(
        model=model,
        test_loader=test_loader,
        label_info=label_info,
        device=device,
        output_dir=args.output_dir
    )

    # 执行分析
    analyzer.analyze()

    print("\n" + "="*80)
    print("错误分析完成！")
    print(f"结果已保存至: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
