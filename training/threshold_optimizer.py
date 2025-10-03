#!/usr/bin/env python3
"""
Threshold Optimizer for Multi-label Classification
多标签分类阈值优化器

为每个类别找到最优预测阈值，最大化 F1 分数
"""

import numpy as np
from sklearn.metrics import f1_score
import torch
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """
    为每个类别找到最优预测阈值

    在验证集上搜索使 F1 分数最大的阈值
    """

    def __init__(self,
                 num_classes: int = 4,
                 search_range: Tuple[float, float] = (0.05, 0.95),
                 step: float = 0.05,
                 class_names: Optional[List[str]] = None):
        """
        初始化阈值优化器

        Args:
            num_classes: 类别数量
            search_range: 搜索范围 (min, max)
            step: 搜索步长
            class_names: 类别名称列表
        """
        self.num_classes = num_classes
        self.search_range = search_range
        self.step = step
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.optimal_thresholds = [0.5] * num_classes
        self.optimal_f1_scores = [0.0] * num_classes
        self.threshold_search_space = None

    def _generate_search_space(self) -> np.ndarray:
        """生成搜索空间"""
        return np.arange(
            self.search_range[0],
            self.search_range[1] + self.step,
            self.step
        )

    def find_optimal_thresholds(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        verbose: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        在验证集上搜索最优阈值

        Args:
            predictions: (N, C) 预测概率
            targets: (N, C) 真实标签 (0/1)
            verbose: 是否打印详细信息

        Returns:
            optimal_thresholds: 每个类别的最优阈值
            optimal_f1_scores: 每个类别的最优F1分数
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")

        self.threshold_search_space = self._generate_search_space()
        optimal_thresholds = []
        optimal_f1_scores = []

        if verbose:
            print(f"\n🔍 阈值优化搜索:")
            print(f"  搜索范围: [{self.search_range[0]:.2f}, {self.search_range[1]:.2f}]")
            print(f"  搜索步长: {self.step}")
            print(f"  搜索点数: {len(self.threshold_search_space)}")
            print(f"  预测形状: {predictions.shape}")
            print(f"  目标形状: {targets.shape}")
            print("-" * 60)

        for class_idx in range(self.num_classes):
            best_threshold = 0.5
            best_f1 = 0.0
            class_name = self.class_names[class_idx]

            # 统计当前类别的样本分布
            pos_samples = np.sum(targets[:, class_idx])
            neg_samples = len(targets) - pos_samples

            if verbose:
                print(f"\n📊 优化类别 {class_idx} ({class_name}):")
                print(f"    正样本: {pos_samples}, 负样本: {neg_samples}")

            # 搜索最佳阈值
            threshold_results = []
            for threshold in self.threshold_search_space:
                # 使用当前阈值进行预测
                preds_binary = (predictions[:, class_idx] > threshold).astype(int)

                # 计算 F1 分数
                f1 = f1_score(
                    targets[:, class_idx],
                    preds_binary,
                    zero_division=0
                )

                threshold_results.append((threshold, f1))

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            optimal_thresholds.append(best_threshold)
            optimal_f1_scores.append(best_f1)

            if verbose:
                improvement = best_f1 - 0.5  # 与默认阈值 0.5 比较
                print(f"    最优阈值: {best_threshold:.2f}")
                print(f"    最优 F1: {best_f1:.4f}")
                print(f"    改进幅度: {improvement:+.4f}")

                # 显示前5个最佳阈值
                threshold_results.sort(key=lambda x: x[1], reverse=True)
                print(f"    Top 3 阈值:")
                for i, (thresh, f1_val) in enumerate(threshold_results[:3]):
                    marker = "🏆" if thresh == best_threshold else "  "
                    print(f"      {marker} {thresh:.2f}: {f1_val:.4f}")

        self.optimal_thresholds = optimal_thresholds
        self.optimal_f1_scores = optimal_f1_scores

        if verbose:
            print("\n" + "="*60)
            print("🎯 阈值优化总结:")
            for i, (class_name, thresh, f1) in enumerate(zip(
                self.class_names, optimal_thresholds, optimal_f1_scores
            )):
                print(f"  {class_name}: threshold={thresh:.2f}, F1={f1:.4f}")

            overall_f1 = np.mean(optimal_f1_scores)
            print(f"\n  整体 F1: {overall_f1:.4f} ({overall_f1*100:.2f}%)")
            print("="*60)

        return optimal_thresholds, optimal_f1_scores

    def predict_with_optimal_thresholds(self, predictions: np.ndarray) -> np.ndarray:
        """
        使用最优阈值进行预测

        Args:
            predictions: (N, C) 预测概率

        Returns:
            preds_binary: (N, C) 二进制预测 (0/1)
        """
        preds_binary = np.zeros_like(predictions, dtype=int)

        for i, threshold in enumerate(self.optimal_thresholds):
            preds_binary[:, i] = (predictions[:, i] > threshold).astype(int)

        return preds_binary

    def predict_with_custom_thresholds(self, predictions: np.ndarray, thresholds: List[float]) -> np.ndarray:
        """
        使用自定义阈值进行预测

        Args:
            predictions: (N, C) 预测概率
            thresholds: 每个类别的阈值

        Returns:
            preds_binary: (N, C) 二进制预测 (0/1)
        """
        if len(thresholds) != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} thresholds, got {len(thresholds)}")

        preds_binary = np.zeros_like(predictions, dtype=int)

        for i, threshold in enumerate(thresholds):
            preds_binary[:, i] = (predictions[:, i] > threshold).astype(int)

        return preds_binary

    def evaluate_with_thresholds(self, predictions: np.ndarray, targets: np.ndarray,
                               thresholds: List[float]) -> Dict[str, float]:
        """
        使用指定阈值评估性能

        Args:
            predictions: (N, C) 预测概率
            targets: (N, C) 真实标签
            thresholds: 要评估的阈值列表

        Returns:
            results: 包含每个类别 F1 分数的字典
        """
        preds_binary = self.predict_with_custom_thresholds(predictions, thresholds)

        results = {}
        individual_f1s = []

        for i, class_name in enumerate(self.class_names):
            f1 = f1_score(
                targets[:, i],
                preds_binary[:, i],
                zero_division=0
            )
            results[class_name] = f1
            individual_f1s.append(f1)

        results['overall_f1'] = np.mean(individual_f1s)

        return results

    def save(self, filepath: str):
        """保存最优阈值"""
        from datetime import datetime
        save_data = {
            'optimal_thresholds': self.optimal_thresholds,
            'optimal_f1_scores': self.optimal_f1_scores,
            'class_names': self.class_names,
            'search_range': self.search_range,
            'step': self.step,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Threshold optimization results saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'ThresholdOptimizer':
        """加载保存的阈值优化器"""
        with open(filepath, 'r') as f:
            save_data = json.load(f)

        optimizer = cls(
            num_classes=len(save_data['class_names']),
            search_range=tuple(save_data['search_range']),
            step=save_data['step'],
            class_names=save_data['class_names']
        )

        optimizer.optimal_thresholds = save_data['optimal_thresholds']
        optimizer.optimal_f1_scores = save_data['optimal_f1_scores']

        logger.info(f"Threshold optimization results loaded from {filepath}")
        return optimizer

    def get_summary(self) -> Dict:
        """获取优化摘要"""
        return {
            'optimal_thresholds': self.optimal_thresholds,
            'optimal_f1_scores': self.optimal_f1_scores,
            'overall_f1': np.mean(self.optimal_f1_scores),
            'class_names': self.class_names,
            'improvement_over_default': {
                class_name: f1 - 0.5  # 假设默认F1为0.5
                for class_name, f1 in zip(self.class_names, self.optimal_f1_scores)
            }
        }


def analyze_threshold_sensitivity(predictions: np.ndarray, targets: np.ndarray,
                           class_names: List[str], class_idx: int = 0) -> Dict:
    """
    分析特定类别的阈值敏感性

    Args:
        predictions: (N, C) 预测概率
        targets: (N, C) 真实标签
        class_names: 类别名称
        class_idx: 要分析的类别索引

    Returns:
        analysis: 分析结果
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    analysis = {
        'thresholds': thresholds.tolist(),
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': []
    }

    from sklearn.metrics import precision_score, recall_score

    for threshold in thresholds:
        preds_binary = (predictions[:, class_idx] > threshold).astype(int)

        f1 = f1_score(targets[:, class_idx], preds_binary, zero_division=0)
        precision = precision_score(targets[:, class_idx], preds_binary, zero_division=0)
        recall = recall_score(targets[:, class_idx], preds_binary, zero_division=0)

        analysis['f1_scores'].append(f1)
        analysis['precision_scores'].append(precision)
        analysis['recall_scores'].append(recall)

    return analysis


if __name__ == "__main__":
    # 测试阈值优化器
    logging.basicConfig(level=logging.INFO)

    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_classes = 4

    # 生成模拟预测和目标
    predictions = np.random.rand(n_samples, n_classes)
    targets = np.random.randint(0, 2, (n_samples, n_classes))

    # 创建不平衡数据
    targets[:, 0] = (np.random.rand(n_samples) > 0.8).astype(int)  # artifacts (20%)
    targets[:, 1] = (np.random.rand(n_samples) > 0.9).astype(int)  # debris (10%)
    targets[:, 2] = (np.random.rand(n_samples) > 0.99).astype(int)  # contamination (1%)
    targets[:, 3] = (np.random.rand(n_samples) > 0.5).astype(int)  # pores (50%)

    class_names = ['artifacts', 'debris', 'contamination', 'pores']

    print("🧪 测试阈值优化器")
    print("="*60)

    # 创建优化器
    optimizer = ThresholdOptimizer(
        num_classes=n_classes,
        search_range=(0.05, 0.95),
        step=0.05,
        class_names=class_names
    )

    # 搜索最优阈值
    optimal_thresholds, optimal_f1_scores = optimizer.find_optimal_thresholds(
        predictions, targets, verbose=True
    )

    # 测试预测
    test_predictions = np.random.rand(10, n_classes)
    binary_predictions = optimizer.predict_with_optimal_thresholds(test_predictions)

    print(f"\n🔮 测试预测 (10个样本):")
    print(f"  概率预测:\n{test_predictions}")
    print(f"  阈值: {optimal_thresholds}")
    print(f"  二进制预测:\n{binary_predictions}")

    # 保存和加载测试
    test_save_path = 'test_threshold_optimization.json'
    optimizer.save(test_save_path)
    loaded_optimizer = ThresholdOptimizer.load(test_save_path)

    print(f"\n✅ 阈值优化器测试完成")