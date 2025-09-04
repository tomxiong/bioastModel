"""
多任务模型评估模块
支持多标签分类的综合评估和可视化
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    hamming_loss, f1_score, jaccard_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MultitaskEvaluator:
    """多任务评估器"""
    
    def __init__(self, 
                 model: nn.Module,
                 task_info: Dict[str, Any],
                 class_mappings: Dict[str, Dict],
                 save_dir: str = "evaluation_results"):
        """
        Args:
            model: 待评估的模型
            task_info: 任务信息
            class_mappings: 类别映射
            save_dir: 结果保存目录
        """
        self.model = model
        self.task_info = task_info
        self.class_mappings = class_mappings
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备
        self.device = next(model.parameters()).device
        
        # 评估结果存储
        self.results = {}
        self.predictions = {}
        self.targets = {}
        self.probabilities = {}
    
    def evaluate(self, 
                test_loader: torch.utils.data.DataLoader,
                save_predictions: bool = True) -> Dict[str, Any]:
        """执行完整评估"""
        print("开始多任务评估...")
        
        # 收集预测结果
        self._collect_predictions(test_loader)
        
        # 评估每个任务
        for task_name in self.task_info['task_names']:
            print(f"\n评估任务: {task_name}")
            task_type = self.task_info['task_types'][task_name]
            
            if task_type == 'multi_label':
                self.results[task_name] = self._evaluate_multilabel_task(task_name)
            else:
                self.results[task_name] = self._evaluate_singlelabel_task(task_name)
        
        # 计算综合得分
        self.results['composite_score'] = self._calculate_composite_score()
        
        # 保存结果
        if save_predictions:
            self._save_predictions()
        
        # 生成报告
        self._generate_report()
        
        # 生成可视化
        self._generate_visualizations()
        
        return self.results
    
    def _collect_predictions(self, test_loader: torch.utils.data.DataLoader):
        """收集模型预测结果"""
        self.model.eval()
        
        # 初始化存储
        for task_name in self.task_info['task_names']:
            self.predictions[task_name] = []
            self.targets[task_name] = []
            self.probabilities[task_name] = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device)
                
                # 模型预测
                outputs = self.model(images)
                
                # 处理每个任务的输出
                for task_name in self.task_info['task_names']:
                    task_output = outputs[task_name]
                    task_target = targets[task_name]
                    
                    # 获取预测概率
                    if self.task_info['task_types'][task_name] == 'multi_label':
                        probs = torch.sigmoid(task_output)
                        preds = (probs > 0.5).cpu().numpy()
                    else:
                        probs = torch.softmax(task_output, dim=1)
                        preds = probs.argmax(dim=1).cpu().numpy()
                    
                    # 存储结果
                    self.predictions[task_name].extend(preds)
                    self.targets[task_name].extend(task_target.cpu().numpy())
                    self.probabilities[task_name].extend(probs.cpu().numpy())
        
        # 转换为numpy数组
        for task_name in self.task_info['task_names']:
            self.predictions[task_name] = np.array(self.predictions[task_name])
            self.targets[task_name] = np.array(self.targets[task_name])
            self.probabilities[task_name] = np.array(self.probabilities[task_name])
    
    def _evaluate_singlelabel_task(self, task_name: str) -> Dict[str, Any]:
        """评估单标签分类任务"""
        y_true = self.targets[task_name]
        y_pred = self.predictions[task_name]
        y_proba = self.probabilities[task_name]
        
        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 分类报告
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self._get_class_names(task_name),
            output_dict=True,
            zero_division=0
        )
        
        # ROC AUC（如果是二分类）
        num_classes = self.task_info['num_classes'][task_name]
        roc_auc = None
        if num_classes == 2:
            roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        
        return {
            'task_type': 'single_label',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'roc_auc': roc_auc,
            'per_class_metrics': self._calculate_per_class_metrics(y_true, y_pred, task_name)
        }
    
    def _evaluate_multilabel_task(self, task_name: str) -> Dict[str, Any]:
        """评估多标签分类任务"""
        y_true = self.targets[task_name]
        y_pred = self.predictions[task_name]
        y_proba = self.probabilities[task_name]
        
        # Hamming Loss
        hamming = hamming_loss(y_true, y_pred)
        
        # F1分数
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
        
        # Jaccard相似度
        jaccard_micro = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
        jaccard_macro = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 精确率和召回率
        precision_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        recall_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 每个标签的指标
        per_label_metrics = {}
        for i, label_name in enumerate(self._get_class_names(task_name)):
            per_label_metrics[label_name] = {
                'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'f1_score': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'support': np.sum(y_true[:, i])
            }
        
        return {
            'task_type': 'multi_label',
            'hamming_loss': hamming,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_samples': f1_samples,
            'jaccard_micro': jaccard_micro,
            'jaccard_macro': jaccard_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'per_label_metrics': per_label_metrics,
            'label_cooccurrence': self._calculate_label_cooccurrence(y_true)
        }
    
    def _get_class_names(self, task_name: str) -> List[str]:
        """获取类别名称"""
        reverse_mapping = self.class_mappings.get(f'{task_name}_reverse', {})
        return [reverse_mapping.get(i, f'class_{i}') for i in sorted(reverse_mapping.keys())]
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    task_name: str) -> Dict[str, Dict]:
        """计算每个类别的指标"""
        class_names = self._get_class_names(task_name)
        per_class = {}
        
        for i, class_name in enumerate(class_names):
            # 二分类指标（当前类别 vs 其他）
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)
            
            precision = precision_score(binary_true, binary_pred, zero_division=0)
            recall = recall_score(binary_true, binary_pred, zero_division=0)
            f1 = f1_score(binary_true, binary_pred, zero_division=0)
            
            per_class[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': np.sum(binary_true)
            }
        
        return per_class
    
    def _calculate_label_cooccurrence(self, y_true: np.ndarray) -> Dict[str, float]:
        """计算标签共现频率"""
        num_labels = y_true.shape[1]
        cooccurrence = {}
        
        for i in range(num_labels):
            for j in range(i + 1, num_labels):
                # 计算标签i和j同时出现的频率
                co_occur = np.sum((y_true[:, i] == 1) & (y_true[:, j] == 1))
                total_i = np.sum(y_true[:, i] == 1)
                total_j = np.sum(y_true[:, j] == 1)
                
                if total_i > 0 and total_j > 0:
                    cooccur_rate_i = co_occur / total_i
                    cooccur_rate_j = co_occur / total_j
                    
                    key = f"{self._get_class_names('interference_mapping')[i]}_&_" \
                          f"{self._get_class_names('interference_mapping')[j]}"
                    cooccurrence[key] = {
                        'cooccurrence_count': int(co_occur),
                        'cooccur_rate_from_i': float(cooccur_rate_i),
                        'cooccur_rate_from_j': float(cooccur_rate_j)
                    }
        
        return cooccurrence
    
    def _calculate_composite_score(self) -> float:
        """计算综合得分"""
        # 使用加权平均F1分数
        weights = {
            'growth_level': 0.3,
            'growth_pattern': 0.3,
            'interference_mapping': 0.2,
            'fine_grained': 0.2
        }
        
        composite_score = 0.0
        for task_name, weight in weights.items():
            if task_name in self.results:
                if task_name == 'interference_mapping':
                    task_score = self.results[task_name]['f1_micro']
                else:
                    task_score = self.results[task_name]['f1_score']
                composite_score += weight * task_score
        
        return composite_score
    
    def _save_predictions(self):
        """保存预测结果"""
        predictions_data = {
            'predictions': {
                task_name: preds.tolist() 
                for task_name, preds in self.predictions.items()
            },
            'targets': {
                task_name: targets.tolist() 
                for task_name, targets in self.targets.items()
            },
            'probabilities': {
                task_name: probs.tolist() 
                for task_name, probs in self.probabilities.items()
            }
        }
        
        with open(self.save_dir / 'predictions.json', 'w') as f:
            json.dump(predictions_data, f, indent=2)
    
    def _generate_report(self):
        """生成评估报告"""
        report = {
            'evaluation_summary': {
                'composite_score': self.results['composite_score'],
                'total_samples': len(list(self.targets.values())[0]),
                'task_count': len(self.task_info['task_names'])
            },
            'task_results': self.results,
            'task_weights': {
                'growth_level': 0.3,
                'growth_pattern': 0.3,
                'interference_mapping': 0.2,
                'fine_grained': 0.2
            }
        }
        
        with open(self.save_dir / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成文本报告
        self._generate_text_report()
    
    def _generate_text_report(self):
        """生成文本格式的报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("多任务模型评估报告")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 摘要
        report_lines.append("评估摘要:")
        report_lines.append(f"  - 综合得分: {self.results['composite_score']:.4f}")
        report_lines.append(f"  - 总样本数: {len(list(self.targets.values())[0])}")
        report_lines.append(f"  - 任务数量: {len(self.task_info['task_names'])}")
        report_lines.append("")
        
        # 各任务结果
        for task_name in self.task_info['task_names']:
            report_lines.append(f"任务: {task_name}")
            report_lines.append("-" * 40)
            
            if task_name in self.results:
                result = self.results[task_name]
                
                if result['task_type'] == 'single_label':
                    report_lines.append(f"  - 准确率: {result['accuracy']:.4f}")
                    report_lines.append(f"  - F1分数: {result['f1_score']:.4f}")
                    report_lines.append(f"  - 精确率: {result['precision']:.4f}")
                    report_lines.append(f"  - 召回率: {result['recall']:.4f}")
                    if result.get('roc_auc'):
                        report_lines.append(f"  - ROC AUC: {result['roc_auc']:.4f}")
                else:
                    report_lines.append(f"  - F1 (Micro): {result['f1_micro']:.4f}")
                    report_lines.append(f"  - F1 (Macro): {result['f1_macro']:.4f}")
                    report_lines.append(f"  - Hamming Loss: {result['hamming_loss']:.4f}")
                    report_lines.append(f"  - Jaccard (Micro): {result['jaccard_micro']:.4f}")
            
            report_lines.append("")
        
        # 保存文本报告
        with open(self.save_dir / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        # 1. 综合性能雷达图
        self._plot_performance_radar()
        
        # 2. 混淆矩阵
        for task_name in self.task_info['task_names']:
            if self.task_info['task_types'][task_name] == 'single_label':
                self._plot_confusion_matrix(task_name)
        
        # 3. 多标签指标热图
        if 'interference_mapping' in self.task_info['task_names']:
            self._plot_multilabel_heatmap('interference_mapping')
        
        # 4. 类别分布对比
        self._plot_class_distribution_comparison()
        
        # 5. ROC曲线（二分类任务）
        for task_name in self.task_info['task_names']:
            if (self.task_info['task_types'][task_name] == 'single_label' and 
                self.task_info['num_classes'][task_name] == 2):
                self._plot_roc_curve(task_name)
    
    def _plot_performance_radar(self):
        """绘制性能雷达图"""
        tasks = self.task_info['task_names']
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # 准备数据
        scores = []
        for task in tasks:
            if task in self.results:
                result = self.results[task]
                if result['task_type'] == 'single_label':
                    scores.append([result.get(m, 0) for m in metrics])
                else:
                    # 对于多标签任务，使用micro指标
                    scores.append([
                        result.get('f1_micro', 0),  # 使用F1作为准确率代理
                        result.get('precision_micro', 0),
                        result.get('recall_micro', 0),
                        result.get('f1_micro', 0)
                    ])
        
        scores = np.array(scores)
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scores = np.concatenate((scores, scores[:, [0]]), axis=1)
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(tasks)))
        
        for idx, (task, color) in enumerate(zip(tasks, colors)):
            ax.plot(angles, scores[idx], 'o-', linewidth=2, label=task, color=color)
            ax.fill(angles, scores[idx], alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['准确率', '精确率', '召回率', 'F1分数'])
        ax.set_ylim(0, 1)
        ax.set_title('多任务性能雷达图', size=16, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, task_name: str):
        """绘制混淆矩阵"""
        if task_name not in self.results:
            return
        
        cm = np.array(self.results[task_name]['confusion_matrix'])
        class_names = self._get_class_names(task_name)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'{task_name} 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'confusion_matrix_{task_name}.png', dpi=300)
        plt.close()
    
    def _plot_multilabel_heatmap(self, task_name: str):
        """绘制多标签指标热图"""
        if task_name not in self.results:
            return
        
        per_label = self.results[task_name]['per_label_metrics']
        class_names = self._get_class_names(task_name)
        
        # 准备数据
        metrics_matrix = []
        for label in class_names:
            metrics_matrix.append([
                per_label[label]['precision'],
                per_label[label]['recall'],
                per_label[label]['f1_score']
            ])
        
        metrics_matrix = np.array(metrics_matrix).T
        
        # 绘制热图
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   xticklabels=class_names,
                   yticklabels=['精确率', '召回率', 'F1分数'])
        plt.title(f'{task_name} 多标签指标热图')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'multilabel_heatmap_{task_name}.png', dpi=300)
        plt.close()
    
    def _plot_class_distribution_comparison(self):
        """绘制类别分布对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, task_name in enumerate(self.task_info['task_names']):
            if idx >= 4:
                break
            
            if task_name not in self.results:
                continue
            
            if self.task_info['task_types'][task_name] == 'single_label':
                # 单标签分布
                class_names = self._get_class_names(task_name)
                unique, counts = np.unique(self.targets[task_name], return_counts=True)
                
                axes[idx].bar(range(len(unique)), counts)
                axes[idx].set_xticks(range(len(unique)))
                axes[idx].set_xticklabels([class_names[i] for i in unique], rotation=45)
                axes[idx].set_title(f'{task_name} 类别分布')
                axes[idx].set_ylabel('样本数')
            else:
                # 多标签分布
                class_names = self._get_class_names(task_name)
                label_counts = np.sum(self.targets[task_name], axis=0)
                
                axes[idx].bar(range(len(class_names)), label_counts)
                axes[idx].set_xticks(range(len(class_names)))
                axes[idx].set_xticklabels(class_names, rotation=45)
                axes[idx].set_title(f'{task_name} 标签分布')
                axes[idx].set_ylabel('样本数')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'class_distribution.png', dpi=300)
        plt.close()
    
    def _plot_roc_curve(self, task_name: str):
        """绘制ROC曲线"""
        if task_name not in self.results:
            return
        
        y_true = self.targets[task_name]
        y_proba = self.probabilities[task_name]
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title(f'{task_name} ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'roc_curve_{task_name}.png', dpi=300)
        plt.close()


def compare_multitask_models(model_results: Dict[str, Dict], 
                           save_dir: str = "model_comparison"):
    """比较多个多任务模型的性能"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备比较数据
    comparison_data = []
    tasks = list(list(model_results.values())[0]['task_results'].keys())
    tasks = [t for t in tasks if t != 'composite_score']
    
    for model_name, results in model_results.items():
        row = {'model': model_name}
        
        # 综合得分
        row['composite_score'] = results['composite_score']
        
        # 各任务得分
        for task in tasks:
            if task in results['task_results']:
                task_result = results['task_results'][task]
                if task_result['task_type'] == 'single_label':
                    row[f'{task}_f1'] = task_result['f1_score']
                    row[f'{task}_accuracy'] = task_result['accuracy']
                else:
                    row[f'{task}_f1'] = task_result['f1_micro']
        
        comparison_data.append(row)
    
    # 创建比较表格
    df = pd.DataFrame(comparison_data)
    
    # 保存比较结果
    df.to_csv(save_dir / 'model_comparison.csv', index=False)
    
    # 绘制比较图
    plt.figure(figsize=(12, 8))
    
    # 综合得分对比
    plt.subplot(2, 2, 1)
    df_sorted = df.sort_values('composite_score', ascending=True)
    plt.barh(df_sorted['model'], df_sorted['composite_score'])
    plt.title('模型综合得分对比')
    plt.xlabel('综合得分')
    
    # 各任务F1分数对比
    task_f1_cols = [col for col in df.columns if col.endswith('_f1')]
    if task_f1_cols:
        plt.subplot(2, 2, 2)
        df_f1 = df[['model'] + task_f1_cols].set_index('model')
        df_f1.plot(kind='bar', ax=plt.gca())
        plt.title('各任务F1分数对比')
        plt.ylabel('F1分数')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 准确率对比（单标签任务）
    accuracy_cols = [col for col in df.columns if col.endswith('_accuracy')]
    if accuracy_cols:
        plt.subplot(2, 2, 3)
        df_acc = df[['model'] + accuracy_cols].set_index('model')
        df_acc.plot(kind='bar', ax=plt.gca())
        plt.title('单标签任务准确率对比')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
    
    # 性能热图
    plt.subplot(2, 2, 4)
    heatmap_data = df.set_index('model')[task_f1_cols]
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r')
    plt.title('模型性能热图')
    plt.tight_layout()
    
    plt.savefig(save_dir / 'model_comparison.png', dpi=300)
    plt.close()
    
    return df


# 使用示例
if __name__ == "__main__":
    # 示例：如何使用评估器
    print("多任务评估模块示例")
    
    # 假设我们有以下数据
    # model = create_multitask_model(...)
    # test_loader = create_dataloader(...)
    # task_info = {...}
    # class_mappings = {...}
    
    # 创建评估器
    # evaluator = MultitaskEvaluator(model, task_info, class_mappings)
    
    # 执行评估
    # results = evaluator.evaluate(test_loader)
    
    print("评估模块已准备就绪")