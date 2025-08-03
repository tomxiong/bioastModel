"""
模型评估器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Args:
            model: 要评估的模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = ['negative', 'positive']
    
    def evaluate(self, data_loader: DataLoader, save_dir: Optional[str] = None) -> Dict:
        """
        全面评估模型
        
        Args:
            data_loader: 数据加载器
            save_dir: 结果保存目录
        
        Returns:
            评估结果字典
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        print("正在评估模型...")
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # 计算各种指标
        results = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # 生成可视化
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._create_visualizations(all_labels, all_predictions, all_probabilities, save_dir)
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict:
        """计算评估指标"""
        
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # 每个类别的指标
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # ROC AUC
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        except:
            auc = 0.0
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 敏感性和特异性（针对二分类）
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)  # 召回率/真正率
        specificity = tn / (tn + fp)  # 真负率
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'classification_report': classification_report(y_true, y_pred, 
                                                         target_names=self.class_names)
        }
        
        return results
    
    def _create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             y_prob: np.ndarray, save_dir: str):
        """创建可视化图表"""
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型评估结果', fontsize=16, fontweight='bold')
        
        # 1. 混淆矩阵
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('混淆矩阵')
        ax1.set_xlabel('预测标签')
        ax1.set_ylabel('真实标签')
        
        # 2. ROC曲线
        ax2 = axes[0, 1]
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])
            ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {auc:.3f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('假正率 (1-特异性)')
            ax2.set_ylabel('真正率 (敏感性)')
            ax2.set_title('ROC曲线')
            ax2.legend(loc="lower right")
        except:
            ax2.text(0.5, 0.5, 'ROC曲线计算失败', ha='center', va='center')
            ax2.set_title('ROC曲线')
        
        # 3. 类别性能对比
        ax3 = axes[1, 0]
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax3.bar(x - width, precision_per_class, width, label='精确率', alpha=0.8)
        ax3.bar(x, recall_per_class, width, label='召回率', alpha=0.8)
        ax3.bar(x + width, f1_per_class, width, label='F1分数', alpha=0.8)
        
        ax3.set_xlabel('类别')
        ax3.set_ylabel('分数')
        ax3.set_title('各类别性能指标')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.class_names)
        ax3.legend()
        ax3.set_ylim([0, 1])
        
        # 4. 预测概率分布
        ax4 = axes[1, 1]
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = y_true == class_idx
            if class_mask.sum() > 0:
                class_probs = y_prob[class_mask, 1]  # positive类的概率
                ax4.hist(class_probs, bins=30, alpha=0.6, label=f'真实{class_name}', 
                        density=True)
        
        ax4.set_xlabel('Positive类预测概率')
        ax4.set_ylabel('密度')
        ax4.set_title('预测概率分布')
        ax4.legend()
        ax4.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='决策边界')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存详细的分类报告
        report_path = os.path.join(save_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 模型评估报告 ===\n\n")
            f.write(f"总体准确率: {accuracy_score(y_true, y_pred):.4f}\n")
            f.write(f"AUC: {roc_auc_score(y_true, y_prob[:, 1]):.4f}\n\n")
            f.write("分类报告:\n")
            f.write(classification_report(y_true, y_pred, target_names=self.class_names))
            f.write(f"\n混淆矩阵:\n{cm}\n")
            
            # 医学相关指标
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
            
            f.write(f"\n医学诊断指标:\n")
            f.write(f"敏感性 (Sensitivity): {sensitivity:.4f}\n")
            f.write(f"特异性 (Specificity): {specificity:.4f}\n")
            f.write(f"阳性预测值 (PPV): {ppv:.4f}\n")
            f.write(f"阴性预测值 (NPV): {npv:.4f}\n")
        
        print(f"评估结果已保存到: {save_dir}")
    
    def compare_models(self, models_results: Dict[str, Dict], save_dir: str):
        """比较多个模型的性能"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 提取指标
        model_names = list(models_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'sensitivity', 'specificity']
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型性能对比', fontsize=16, fontweight='bold')
        
        # 1. 总体性能对比
        ax1 = axes[0, 0]
        metric_values = {metric: [models_results[name][metric] for name in model_names] 
                        for metric in metrics}
        
        x = np.arange(len(model_names))
        width = 0.1
        
        for i, metric in enumerate(['accuracy', 'f1_score', 'auc']):
            ax1.bar(x + i*width, metric_values[metric], width, label=metric, alpha=0.8)
        
        ax1.set_xlabel('模型')
        ax1.set_ylabel('分数')
        ax1.set_title('主要性能指标对比')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend()
        ax1.set_ylim([0, 1])
        
        # 2. 敏感性vs特异性
        ax2 = axes[0, 1]
        sensitivities = [models_results[name]['sensitivity'] for name in model_names]
        specificities = [models_results[name]['specificity'] for name in model_names]
        
        ax2.scatter(specificities, sensitivities, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax2.annotate(name, (specificities[i], sensitivities[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('特异性')
        ax2.set_ylabel('敏感性')
        ax2.set_title('敏感性 vs 特异性')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        
        # 3. 雷达图
        ax3 = axes[1, 0]
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for name in model_names:
            values = [models_results[name][metric] for metric in metrics]
            values += values[:1]  # 闭合
            ax3.plot(angles, values, 'o-', linewidth=2, label=name, alpha=0.7)
            ax3.fill(angles, values, alpha=0.1)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metrics)
        ax3.set_ylim(0, 1)
        ax3.set_title('综合性能雷达图')
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax3.grid(True)
        
        # 4. 性能排名
        ax4 = axes[1, 1]
        
        # 计算综合得分（可以调整权重）
        weights = {'accuracy': 0.2, 'precision': 0.15, 'recall': 0.15, 
                  'f1_score': 0.2, 'auc': 0.15, 'sensitivity': 0.1, 'specificity': 0.05}
        
        scores = {}
        for name in model_names:
            score = sum(models_results[name][metric] * weight 
                       for metric, weight in weights.items())
            scores[name] = score
        
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        names, score_values = zip(*sorted_models)
        
        bars = ax4.barh(range(len(names)), score_values, alpha=0.7)
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names)
        ax4.set_xlabel('综合得分')
        ax4.set_title('模型综合排名')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, score_values)):
            ax4.text(score + 0.01, i, f'{score:.3f}', 
                    va='center', ha='left', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存对比报告
        report_path = os.path.join(save_dir, 'model_comparison_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 模型性能对比报告 ===\n\n")
            
            for i, (name, score) in enumerate(sorted_models):
                f.write(f"排名 {i+1}: {name} (综合得分: {score:.4f})\n")
                for metric in metrics:
                    f.write(f"  {metric}: {models_results[name][metric]:.4f}\n")
                f.write("\n")
        
        print(f"模型对比结果已保存到: {save_dir}")

if __name__ == "__main__":
    # 测试评估器
    from models.efficientnet import create_efficientnet_b0
    from training.dataset import create_data_loaders
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_efficientnet_b0(num_classes=2)
    
    # 创建数据加载器
    data_loaders = create_data_loaders('./bioast_dataset', batch_size=16)
    
    # 创建评估器
    evaluator = ModelEvaluator(model, device)
    
    print("开始测试评估...")
    results = evaluator.evaluate(data_loaders['test'], save_dir='./test_evaluation')
    
    print("评估结果:")
    for key, value in results.items():
        if key not in ['confusion_matrix', 'classification_report']:
            print(f"  {key}: {value}")