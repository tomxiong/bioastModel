"""
多任务指标计算模块

提供多任务学习中各种指标的计算功能，包括单标签分类、多标签分类和跨任务一致性指标。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, precision_recall_fscore_support,
    classification_report, confusion_matrix
)


class MultiTaskMetrics:
    """多任务指标计算器"""
    
    def __init__(self, task_names: List[str], task_types: Optional[Dict[str, str]] = None):
        """
        初始化多任务指标计算器
        
        Args:
            task_names: 任务名称列表
            task_types: 任务类型字典，键为任务名，值为'single_label'或'multi_label'
        """
        self.task_names = task_names
        self.task_types = task_types or {}
        
        # 默认任务类型
        for task_name in task_names:
            if task_name not in self.task_types:
                if 'interference' in task_name.lower() or 'factors' in task_name.lower():
                    self.task_types[task_name] = 'multi_label'
                else:
                    self.task_types[task_name] = 'single_label'
    
    def compute_metrics(self, predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        计算所有任务的指标
        
        Args:
            predictions: 预测结果字典
            targets: 真实标签字典
            
        Returns:
            指标结果字典
        """
        results = {}
        
        for task_name in self.task_names:
            if task_name not in predictions or task_name not in targets:
                continue
                
            task_type = self.task_types.get(task_name, 'single_label')
            
            if task_type == 'multi_label':
                results[task_name] = self._compute_multilabel_metrics(
                    predictions[task_name], targets[task_name]
                )
            else:
                results[task_name] = self._compute_singlelabel_metrics(
                    predictions[task_name], targets[task_name]
                )
        
        # 计算整体指标
        results['overall'] = self._compute_overall_metrics(results)
        
        return results
    
    def _compute_singlelabel_metrics(self, preds: torch.Tensor, 
                                   targets: torch.Tensor) -> Dict[str, float]:
        """计算单标签分类指标"""
        # 转换为numpy
        if isinstance(preds, torch.Tensor):
            if preds.dim() > 1:
                preds = torch.argmax(preds, dim=1)
            preds = preds.cpu().numpy()
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # 基本指标
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='weighted', zero_division=0
        )
        
        # 宏平均指标
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, preds, average='macro', zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro)
        }
    
    def _compute_multilabel_metrics(self, preds: torch.Tensor, 
                                  targets: torch.Tensor) -> Dict[str, float]:
        """计算多标签分类指标"""
        # 转换为numpy
        if isinstance(preds, torch.Tensor):
            if preds.dim() > 1 and preds.shape[1] > 1:
                # 如果是logits，应用sigmoid并阈值化
                preds = torch.sigmoid(preds) > 0.5
            preds = preds.cpu().numpy().astype(int)
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy().astype(int)
        
        # 确保是二维数组
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        
        # Hamming Loss
        hamming = hamming_loss(targets, preds)
        
        # 微平均指标
        precision_micro = precision_score(targets, preds, average='micro', zero_division=0)
        recall_micro = recall_score(targets, preds, average='micro', zero_division=0)
        f1_micro = f1_score(targets, preds, average='micro', zero_division=0)
        
        # 宏平均指标
        precision_macro = precision_score(targets, preds, average='macro', zero_division=0)
        recall_macro = recall_score(targets, preds, average='macro', zero_division=0)
        f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
        
        # 样本级准确率（所有标签都正确的比例）
        sample_accuracy = np.mean(np.all(preds == targets, axis=1))
        
        return {
            'hamming_loss': float(hamming),
            'sample_accuracy': float(sample_accuracy),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro)
        }
    
    def _compute_overall_metrics(self, task_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算整体指标"""
        if not task_results:
            return {}
        
        # 收集所有任务的准确率
        accuracies = []
        f1_scores = []
        
        for task_name, metrics in task_results.items():
            if task_name == 'overall':
                continue
                
            if 'accuracy' in metrics:
                accuracies.append(metrics['accuracy'])
            elif 'sample_accuracy' in metrics:
                accuracies.append(metrics['sample_accuracy'])
            
            if 'f1_score' in metrics:
                f1_scores.append(metrics['f1_score'])
            elif 'f1_macro' in metrics:
                f1_scores.append(metrics['f1_macro'])
        
        overall = {}
        if accuracies:
            overall['mean_accuracy'] = float(np.mean(accuracies))
        if f1_scores:
            overall['mean_f1'] = float(np.mean(f1_scores))
        
        return overall
    
    def get_classification_report(self, predictions: Dict[str, torch.Tensor], 
                                targets: Dict[str, torch.Tensor], 
                                class_names: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
        """
        获取详细的分类报告
        
        Args:
            predictions: 预测结果字典
            targets: 真实标签字典
            class_names: 类别名称字典
            
        Returns:
            分类报告字典
        """
        reports = {}
        
        for task_name in self.task_names:
            if task_name not in predictions or task_name not in targets:
                continue
            
            task_type = self.task_types.get(task_name, 'single_label')
            
            if task_type == 'single_label':
                # 转换预测结果
                preds = predictions[task_name]
                if isinstance(preds, torch.Tensor):
                    if preds.dim() > 1:
                        preds = torch.argmax(preds, dim=1)
                    preds = preds.cpu().numpy()
                
                targets_np = targets[task_name]
                if isinstance(targets_np, torch.Tensor):
                    targets_np = targets_np.cpu().numpy()
                
                # 获取类别名称
                target_names = None
                if class_names and task_name in class_names:
                    target_names = class_names[task_name]
                
                # 生成分类报告
                report = classification_report(
                    targets_np, preds, 
                    target_names=target_names,
                    zero_division=0,
                    output_dict=False
                )
                reports[task_name] = report
        
        return reports
    
    def compute_confusion_matrices(self, predictions: Dict[str, torch.Tensor], 
                                 targets: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """计算混淆矩阵"""
        matrices = {}
        
        for task_name in self.task_names:
            if task_name not in predictions or task_name not in targets:
                continue
            
            task_type = self.task_types.get(task_name, 'single_label')
            
            if task_type == 'single_label':
                # 转换预测结果
                preds = predictions[task_name]
                if isinstance(preds, torch.Tensor):
                    if preds.dim() > 1:
                        preds = torch.argmax(preds, dim=1)
                    preds = preds.cpu().numpy()
                
                targets_np = targets[task_name]
                if isinstance(targets_np, torch.Tensor):
                    targets_np = targets_np.cpu().numpy()
                
                # 计算混淆矩阵
                cm = confusion_matrix(targets_np, preds)
                matrices[task_name] = cm
        
        return matrices


def create_multitask_metrics(task_names: List[str], 
                           task_types: Optional[Dict[str, str]] = None) -> MultiTaskMetrics:
    """
    创建多任务指标计算器
    
    Args:
        task_names: 任务名称列表
        task_types: 任务类型字典
        
    Returns:
        MultiTaskMetrics实例
    """
    return MultiTaskMetrics(task_names, task_types)


def calculate_metrics(predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     task_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    计算多任务指标的便捷函数
    
    Args:
        predictions: 预测结果字典
        targets: 真实标签字典
        task_names: 任务名称列表，如果为None则从predictions中获取
        
    Returns:
        指标结果字典
    """
    if task_names is None:
        task_names = list(predictions.keys())
    
    # 创建指标计算器
    metrics_calculator = create_multitask_metrics(task_names)
    
    # 计算指标
    return metrics_calculator.compute_metrics(predictions, targets)