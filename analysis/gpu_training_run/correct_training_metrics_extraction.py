#!/usr/bin/env python3
"""
正确GPU训练指标提取脚本
从 experiments/gpu_training_run/training.log 中提取详细的训练指标和历史数据
"""

import re
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

class TrainingMetricsExtractor:
    def __init__(self, log_file_path):
        self.log_file_path = Path(log_file_path)
        self.training_history = {}
        self.detailed_metrics = {}
        
    def extract_all_metrics(self):
        """提取所有训练指标"""
        print("📊 提取训练指标...")
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 提取逐轮训练历史
        self._extract_epoch_history(log_content)
        
        # 提取批次级别的损失
        self._extract_batch_losses(log_content)
        
        # 提取最终测试结果的详细信息
        self._extract_detailed_test_results(log_content)
        
        # 提取数据集详细统计
        self._extract_detailed_dataset_stats(log_content)
        
        print("✅ 指标提取完成")
        
    def _extract_epoch_history(self, log_content):
        """提取每轮的详细历史"""
        epochs_data = []
        
        # 更精确的epoch匹配模式
        epoch_pattern = r"Epoch (\d+)/10.*?Train Loss: ([\d.]+).*?Val Loss: ([\d.]+).*?Val Accuracies: ({.*?}).*?Weighted Accuracy: ([\d.]+).*?Learning Rate: ([\d.]+)"
        
        for match in re.finditer(epoch_pattern, log_content, re.DOTALL):
            epoch_num = int(match.group(1))
            train_loss = float(match.group(2))
            val_loss = float(match.group(3))
            accuracies_str = match.group(4)
            weighted_acc = float(match.group(5))
            learning_rate = float(match.group(6))
            
            # 解析准确率字典
            try:
                accuracies_str = accuracies_str.replace("'", '"')
                accuracies_str = re.sub(r'np\.float64\(([\d.]+)\)', r'\1', accuracies_str)
                accuracies = json.loads(accuracies_str)
            except Exception as e:
                print(f"解析准确率失败 epoch {epoch_num}: {e}")
                accuracies = {}
            
            epoch_data = {
                'epoch': epoch_num,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'weighted_accuracy': weighted_acc,
                'learning_rate': learning_rate,
                'growth_level_acc': accuracies.get('growth_level', 0),
                'growth_pattern_acc': accuracies.get('growth_pattern', 0),
                'interference_factors_acc': accuracies.get('interference_factors', 0)
            }
            
            epochs_data.append(epoch_data)
        
        self.training_history['epochs'] = epochs_data
        
        # 计算训练趋势
        if epochs_data:
            self._calculate_training_trends(epochs_data)
    
    def _calculate_training_trends(self, epochs_data):
        """计算训练趋势"""
        # 提取各指标序列
        train_losses = [e['train_loss'] for e in epochs_data]
        val_losses = [e['val_loss'] for e in epochs_data]
        weighted_accs = [e['weighted_accuracy'] for e in epochs_data]
        growth_level_accs = [e['growth_level_acc'] for e in epochs_data]
        growth_pattern_accs = [e['growth_pattern_acc'] for e in epochs_data]
        interference_accs = [e['interference_factors_acc'] for e in epochs_data]
        
        trends = {
            'train_loss': {
                'initial': train_losses[0],
                'final': train_losses[-1],
                'best': min(train_losses),
                'worst': max(train_losses),
                'improvement': (train_losses[0] - train_losses[-1]) / train_losses[0] * 100,
                'stability': np.std(train_losses[-3:]) if len(train_losses) >= 3 else 0
            },
            'val_loss': {
                'initial': val_losses[0],
                'final': val_losses[-1],
                'best': min(val_losses),
                'worst': max(val_losses),
                'improvement': (val_losses[0] - val_losses[-1]) / val_losses[0] * 100,
                'stability': np.std(val_losses[-3:]) if len(val_losses) >= 3 else 0
            },
            'weighted_accuracy': {
                'initial': weighted_accs[0],
                'final': weighted_accs[-1],
                'best': max(weighted_accs),
                'worst': min(weighted_accs),
                'improvement': (weighted_accs[-1] - weighted_accs[0]) * 100,
                'stability': np.std(weighted_accs[-3:]) if len(weighted_accs) >= 3 else 0
            },
            'growth_level_accuracy': {
                'initial': growth_level_accs[0],
                'final': growth_level_accs[-1],
                'best': max(growth_level_accs),
                'worst': min(growth_level_accs),
                'improvement': (growth_level_accs[-1] - growth_level_accs[0]) * 100,
                'stability': np.std(growth_level_accs[-3:]) if len(growth_level_accs) >= 3 else 0
            },
            'growth_pattern_accuracy': {
                'initial': growth_pattern_accs[0],
                'final': growth_pattern_accs[-1],
                'best': max(growth_pattern_accs),
                'worst': min(growth_pattern_accs),
                'improvement': (growth_pattern_accs[-1] - growth_pattern_accs[0]) * 100,
                'stability': np.std(growth_pattern_accs[-3:]) if len(growth_pattern_accs) >= 3 else 0
            },
            'interference_factors_accuracy': {
                'initial': interference_accs[0],
                'final': interference_accs[-1],
                'best': max(interference_accs),
                'worst': min(interference_accs),
                'improvement': (interference_accs[-1] - interference_accs[0]) * 100,
                'stability': np.std(interference_accs[-3:]) if len(interference_accs) >= 3 else 0
            }
        }
        
        self.training_history['trends'] = trends
    
    def _extract_batch_losses(self, log_content):
        """提取批次级别的损失"""
        batch_losses = []
        
        # 匹配批次损失
        batch_pattern = r"Batch (\d+)/437, Loss: ([\d.]+)"
        
        current_epoch = 1
        for match in re.finditer(batch_pattern, log_content):
            batch_num = int(match.group(1))
            loss = float(match.group(2))
            
            # 如果batch_num为0，说明开始新的epoch
            if batch_num == 0 and batch_losses:
                current_epoch += 1
            
            batch_losses.append({
                'epoch': current_epoch,
                'batch': batch_num,
                'loss': loss
            })
        
        self.training_history['batch_losses'] = batch_losses
        
        # 分析批次损失趋势
        if batch_losses:
            self._analyze_batch_trends(batch_losses)
    
    def _analyze_batch_trends(self, batch_losses):
        """分析批次损失趋势"""
        # 按epoch分组
        epoch_batches = {}
        for batch in batch_losses:
            epoch = batch['epoch']
            if epoch not in epoch_batches:
                epoch_batches[epoch] = []
            epoch_batches[epoch].append(batch['loss'])
        
        batch_analysis = {}
        for epoch, losses in epoch_batches.items():
            if len(losses) > 1:
                batch_analysis[f'epoch_{epoch}'] = {
                    'initial_loss': losses[0],
                    'final_loss': losses[-1],
                    'min_loss': min(losses),
                    'max_loss': max(losses),
                    'avg_loss': np.mean(losses),
                    'std_loss': np.std(losses),
                    'improvement': (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0,
                    'volatility': np.std(losses) / np.mean(losses) * 100 if np.mean(losses) > 0 else 0
                }
        
        self.training_history['batch_analysis'] = batch_analysis
    
    def _extract_detailed_test_results(self, log_content):
        """提取详细的测试结果"""
        test_results = {}
        
        # 生长级别结果
        growth_level_match = re.search(r"growth_level: ({.*?})", log_content, re.DOTALL)
        if growth_level_match:
            result_str = growth_level_match.group(1)
            
            # 提取各项指标
            accuracy_match = re.search(r"'accuracy': ([\d.]+)", result_str)
            precision_match = re.search(r"'precision': ([\d.]+)", result_str)
            recall_match = re.search(r"'recall': ([\d.]+)", result_str)
            f1_match = re.search(r"'f1_score': ([\d.]+)", result_str)
            
            # 提取混淆矩阵
            cm_match = re.search(r"'confusion_matrix': (\[\[.*?\]\])", result_str)
            
            test_results['growth_level'] = {
                'accuracy': float(accuracy_match.group(1)) if accuracy_match else 0,
                'precision': float(precision_match.group(1)) if precision_match else 0,
                'recall': float(recall_match.group(1)) if recall_match else 0,
                'f1_score': float(f1_match.group(1)) if f1_match else 0,
                'confusion_matrix': eval(cm_match.group(1)) if cm_match else []
            }
        
        # 生长模式结果
        growth_pattern_match = re.search(r"growth_pattern: ({.*?})", log_content, re.DOTALL)
        if growth_pattern_match:
            result_str = growth_pattern_match.group(1)
            
            accuracy_match = re.search(r"'accuracy': ([\d.]+)", result_str)
            precision_match = re.search(r"'precision': ([\d.]+)", result_str)
            recall_match = re.search(r"'recall': ([\d.]+)", result_str)
            f1_match = re.search(r"'f1_score': ([\d.]+)", result_str)
            cm_match = re.search(r"'confusion_matrix': (\[\[.*?\]\])", result_str)
            
            test_results['growth_pattern'] = {
                'accuracy': float(accuracy_match.group(1)) if accuracy_match else 0,
                'precision': float(precision_match.group(1)) if precision_match else 0,
                'recall': float(recall_match.group(1)) if recall_match else 0,
                'f1_score': float(f1_match.group(1)) if f1_match else 0,
                'confusion_matrix': eval(cm_match.group(1)) if cm_match else []
            }
        
        # 干扰因素结果
        interference_match = re.search(r"interference_factors: ({.*?})", log_content, re.DOTALL)
        if interference_match:
            result_str = interference_match.group(1)
            
            # 提取各类别准确率
            categories = ['artifacts', 'contamination', 'debris', 'pores']
            interference_results = {}
            
            for category in categories:
                cat_match = re.search(f"'{category}': {{'accuracy': ([\d.]+)}}", result_str)
                if cat_match:
                    interference_results[category] = {
                        'accuracy': float(cat_match.group(1))
                    }
            
            # 总体准确率
            overall_match = re.search(r"'overall_accuracy': np\.float64\(([\d.]+)\)", result_str)
            if overall_match:
                interference_results['overall'] = {
                    'accuracy': float(overall_match.group(1))
                }
            
            test_results['interference_factors'] = interference_results
        
        self.detailed_metrics['test_results'] = test_results
    
    def _extract_detailed_dataset_stats(self, log_content):
        """提取详细的数据集统计信息"""
        dataset_stats = {}
        
        for split in ['TRAIN', 'VAL', 'TEST']:
            split_stats = {}
            split_section = re.search(f"=== {split} Dataset Statistics ===(.*?)===", log_content, re.DOTALL)
            
            if split_section:
                split_content = split_section.group(1)
                
                # 总样本数
                total_match = re.search(r"Total samples: (\d+)", split_content)
                if total_match:
                    split_stats['total_samples'] = int(total_match.group(1))
                
                # 生长级别分布
                growth_level_match = re.search(r"Growth level distribution: ({.*?})", split_content)
                if growth_level_match:
                    growth_level_str = growth_level_match.group(1).replace("'", '"')
                    split_stats['growth_level'] = json.loads(growth_level_str)
                
                # 生长模式分布
                growth_pattern_match = re.search(r"Top 5 growth patterns: ({.*?})", split_content)
                if growth_pattern_match:
                    growth_pattern_str = growth_pattern_match.group(1).replace("'", '"')
                    split_stats['growth_patterns'] = json.loads(growth_pattern_str)
                
                # 干扰因素分布
                interference_match = re.search(r"Top 5 interference factors: ({.*?})", split_content)
                if interference_match:
                    interference_str = interference_match.group(1).replace("'", '"')
                    split_stats['interference_factors'] = json.loads(interference_str)
            
            dataset_stats[split.lower()] = split_stats
        
        self.detailed_metrics['dataset_stats'] = dataset_stats
        
        # 计算类别不平衡比例
        self._calculate_class_imbalance(dataset_stats)
    
    def _calculate_class_imbalance(self, dataset_stats):
        """计算类别不平衡比例"""
        imbalance_analysis = {}
        
        if 'train' in dataset_stats:
            train_stats = dataset_stats['train']
            
            # 生长级别不平衡
            if 'growth_level' in train_stats:
                gl_dist = train_stats['growth_level']
                max_samples = max(gl_dist.values())
                min_samples = min(gl_dist.values())
                imbalance_analysis['growth_level'] = {
                    'max_samples': max_samples,
                    'min_samples': min_samples,
                    'imbalance_ratio': max_samples / min_samples if min_samples > 0 else float('inf'),
                    'distribution': gl_dist
                }
            
            # 生长模式不平衡
            if 'growth_patterns' in train_stats:
                gp_dist = train_stats['growth_patterns']
                max_samples = max(gp_dist.values())
                min_samples = min(gp_dist.values())
                imbalance_analysis['growth_patterns'] = {
                    'max_samples': max_samples,
                    'min_samples': min_samples,
                    'imbalance_ratio': max_samples / min_samples if min_samples > 0 else float('inf'),
                    'distribution': gp_dist
                }
            
            # 干扰因素不平衡
            if 'interference_factors' in train_stats:
                if_dist = train_stats['interference_factors']
                max_samples = max(if_dist.values())
                min_samples = min(if_dist.values())
                imbalance_analysis['interference_factors'] = {
                    'max_samples': max_samples,
                    'min_samples': min_samples,
                    'imbalance_ratio': max_samples / min_samples if min_samples > 0 else float('inf'),
                    'distribution': if_dist
                }
        
        self.detailed_metrics['class_imbalance'] = imbalance_analysis
    
    def generate_detailed_analysis(self):
        """生成详细分析"""
        print("\n📈 生成详细分析...")
        
        analysis = {
            'training_summary': self._summarize_training(),
            'convergence_analysis': self._analyze_convergence(),
            'task_performance_analysis': self._analyze_task_performance(),
            'data_imbalance_impact': self._analyze_imbalance_impact(),
            'training_stability': self._analyze_training_stability()
        }
        
        return analysis
    
    def _summarize_training(self):
        """训练总结"""
        epochs = self.training_history.get('epochs', [])
        if not epochs:
            return {}
        
        return {
            'total_epochs': len(epochs),
            'best_epoch': max(epochs, key=lambda x: x['weighted_accuracy'])['epoch'],
            'best_weighted_accuracy': max(epochs, key=lambda x: x['weighted_accuracy'])['weighted_accuracy'],
            'final_weighted_accuracy': epochs[-1]['weighted_accuracy'],
            'training_efficiency': f"{len(epochs)} epochs in 56.4s",
            'convergence_point': self._find_convergence_point(epochs)
        }
    
    def _find_convergence_point(self, epochs):
        """找到收敛点"""
        if len(epochs) < 3:
            return len(epochs)
        
        # 寻找验证准确率不再显著提升的点
        val_accs = [e['weighted_accuracy'] for e in epochs]
        
        for i in range(2, len(val_accs)):
            # 检查最近3个epoch的改进是否小于1%
            recent_improvement = (val_accs[i] - val_accs[i-2]) * 100
            if recent_improvement < 1.0:
                return i + 1
        
        return len(epochs)
    
    def _analyze_convergence(self):
        """分析收敛情况"""
        trends = self.training_history.get('trends', {})
        epochs = self.training_history.get('epochs', [])
        
        if not trends or not epochs:
            return {}
        
        # 检查过拟合迹象
        val_loss_trend = trends.get('val_loss', {})
        train_loss_trend = trends.get('train_loss', {})
        
        overfitting_signs = []
        if val_loss_trend.get('final', 0) > val_loss_trend.get('best', 0):
            overfitting_signs.append("验证损失在最佳点后上升")
        
        if train_loss_trend.get('final', 0) < val_loss_trend.get('final', 0):
            gap = val_loss_trend.get('final', 0) - train_loss_trend.get('final', 0)
            if gap > 0.2:
                overfitting_signs.append(f"训练-验证损失差距较大 ({gap:.3f})")
        
        return {
            'convergence_status': self._assess_convergence_status(epochs),
            'overfitting_signs': overfitting_signs,
            'loss_trends': {
                'train_loss_improvement': train_loss_trend.get('improvement', 0),
                'val_loss_improvement': val_loss_trend.get('improvement', 0),
                'loss_stability': {
                    'train': train_loss_trend.get('stability', 0),
                    'val': val_loss_trend.get('stability', 0)
                }
            },
            'accuracy_trends': {
                'weighted_acc_improvement': trends.get('weighted_accuracy', {}).get('improvement', 0),
                'task_improvements': {
                    'growth_level': trends.get('growth_level_accuracy', {}).get('improvement', 0),
                    'growth_pattern': trends.get('growth_pattern_accuracy', {}).get('improvement', 0),
                    'interference_factors': trends.get('interference_factors_accuracy', {}).get('improvement', 0)
                }
            }
        }
    
    def _assess_convergence_status(self, epochs):
        """评估收敛状态"""
        if len(epochs) < 5:
            return "insufficient_data"
        
        # 检查最后几个epoch的稳定性
        recent_accs = [e['weighted_accuracy'] for e in epochs[-3:]]
        acc_std = np.std(recent_accs)
        
        if acc_std < 0.005:  # 准确率变化小于0.5%
            return "converged"
        elif acc_std < 0.02:  # 准确率变化小于2%
            return "stabilizing"
        else:
            return "still_changing"
    
    def _analyze_task_performance(self):
        """分析各任务性能"""
        test_results = self.detailed_metrics.get('test_results', {})
        trends = self.training_history.get('trends', {})
        
        task_analysis = {}
        
        # 生长级别分析
        if 'growth_level' in test_results:
            gl_results = test_results['growth_level']
            gl_trends = trends.get('growth_level_accuracy', {})
            
            task_analysis['growth_level'] = {
                'final_metrics': gl_results,
                'training_trends': gl_trends,
                'performance_assessment': self._assess_task_performance('growth_level', gl_results, gl_trends),
                'confusion_matrix_analysis': self._analyze_confusion_matrix(gl_results.get('confusion_matrix', []))
            }
        
        # 生长模式分析
        if 'growth_pattern' in test_results:
            gp_results = test_results['growth_pattern']
            gp_trends = trends.get('growth_pattern_accuracy', {})
            
            task_analysis['growth_pattern'] = {
                'final_metrics': gp_results,
                'training_trends': gp_trends,
                'performance_assessment': self._assess_task_performance('growth_pattern', gp_results, gp_trends),
                'confusion_matrix_analysis': self._analyze_confusion_matrix(gp_results.get('confusion_matrix', []))
            }
        
        # 干扰因素分析
        if 'interference_factors' in test_results:
            if_results = test_results['interference_factors']
            if_trends = trends.get('interference_factors_accuracy', {})
            
            task_analysis['interference_factors'] = {
                'final_metrics': if_results,
                'training_trends': if_trends,
                'performance_assessment': self._assess_task_performance('interference_factors', if_results, if_trends),
                'category_analysis': self._analyze_interference_categories(if_results)
            }
        
        return task_analysis
    
    def _assess_task_performance(self, task_name, results, trends):
        """评估单个任务性能"""
        if task_name == 'interference_factors':
            accuracy = results.get('overall', {}).get('accuracy', 0)
        else:
            accuracy = results.get('accuracy', 0)
        
        improvement = trends.get('improvement', 0)
        stability = trends.get('stability', 0)
        
        assessment = {
            'accuracy_level': self._classify_accuracy(accuracy),
            'improvement_trend': 'improving' if improvement > 2 else 'stable' if improvement > -2 else 'declining',
            'stability_level': 'stable' if stability < 0.01 else 'moderate' if stability < 0.05 else 'unstable',
            'overall_status': self._get_overall_status(accuracy, improvement, stability)
        }
        
        return assessment
    
    def _classify_accuracy(self, accuracy):
        """分类准确率水平"""
        if accuracy >= 0.95:
            return "excellent"
        elif accuracy >= 0.85:
            return "good"
        elif accuracy >= 0.75:
            return "average"
        else:
            return "poor"
    
    def _get_overall_status(self, accuracy, improvement, stability):
        """获取整体状态"""
        if accuracy >= 0.9 and improvement >= 0 and stability < 0.02:
            return "optimal"
        elif accuracy >= 0.8 and improvement >= -1:
            return "good"
        elif accuracy >= 0.7:
            return "acceptable"
        else:
            return "needs_improvement"
    
    def _analyze_confusion_matrix(self, cm):
        """分析混淆矩阵"""
        if not cm or not isinstance(cm, list):
            return {}
        
        cm = np.array(cm)
        if cm.size == 0:
            return {}
        
        # 计算每个类别的精确率和召回率
        num_classes = cm.shape[0]
        class_metrics = {}
        
        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[f'class_{i}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': np.sum(cm[i, :])
            }
        
        # 找出最容易混淆的类别对
        confusion_pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': i,
                        'predicted_class': j,
                        'count': int(cm[i, j]),
                        'percentage': cm[i, j] / np.sum(cm[i, :]) * 100
                    })
        
        # 按混淆数量排序
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'class_metrics': class_metrics,
            'top_confusions': confusion_pairs[:5],  # 前5个最容易混淆的类别对
            'overall_accuracy': np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        }
    
    def _analyze_interference_categories(self, if_results):
        """分析干扰因素各类别"""
        category_analysis = {}
        
        for category, metrics in if_results.items():
            if category == 'overall':
                continue
                
            accuracy = metrics.get('accuracy', 0)
            
            category_analysis[category] = {
                'accuracy': accuracy,
                'performance_level': self._classify_accuracy(accuracy),
                'status': 'excellent' if accuracy > 0.9 else 'good' if accuracy > 0.8 else 'needs_improvement'
            }
        
        return category_analysis
    
    def _analyze_imbalance_impact(self):
        """分析数据不平衡的影响"""
        imbalance = self.detailed_metrics.get('class_imbalance', {})
        test_results = self.detailed_metrics.get('test_results', {})
        
        impact_analysis = {}
        
        for task, imbalance_info in imbalance.items():
            if task in test_results:
                task_results = test_results[task]
                
                impact_analysis[task] = {
                    'imbalance_ratio': imbalance_info.get('imbalance_ratio', 1),
                    'performance_impact': self._assess_imbalance_impact(
                        imbalance_info.get('imbalance_ratio', 1),
                        task_results.get('accuracy', 0) if task != 'interference_factors' else 
                        task_results.get('overall', {}).get('accuracy', 0)
                    ),
                    'distribution': imbalance_info.get('distribution', {}),
                    'recommendations': self._get_imbalance_recommendations(imbalance_info.get('imbalance_ratio', 1))
                }
        
        return impact_analysis
    
    def _assess_imbalance_impact(self, ratio, accuracy):
        """评估不平衡对性能的影响"""
        if ratio < 2:
            return "minimal"
        elif ratio < 5:
            return "moderate" if accuracy > 0.8 else "significant"
        elif ratio < 10:
            return "significant" if accuracy > 0.7 else "severe"
        else:
            return "severe"
    
    def _get_imbalance_recommendations(self, ratio):
        """获取不平衡处理建议"""
        if ratio < 2:
            return ["数据分布相对均衡，无需特殊处理"]
        elif ratio < 5:
            return ["考虑使用类别权重", "轻微的数据增强"]
        elif ratio < 10:
            return ["使用Focal Loss", "重采样技术", "数据增强"]
        else:
            return ["强烈建议使用Focal Loss", "SMOTE或其他过采样技术", "收集更多少数类样本", "考虑集成学习方法"]
    
    def _analyze_training_stability(self):
        """分析训练稳定性"""
        batch_analysis = self.training_history.get('batch_analysis', {})
        trends = self.training_history.get('trends', {})
        
        stability_metrics = {}
        
        # 批次级别稳定性
        if batch_analysis:
            volatilities = []
            for epoch, analysis in batch_analysis.items():
                volatilities.append(analysis.get('volatility', 0))
            
            stability_metrics['batch_level'] = {
                'average_volatility': np.mean(volatilities) if volatilities else 0,
                'volatility_trend': 'improving' if len(volatilities) > 1 and volatilities[-1] < volatilities[0] else 'stable',
                'stability_assessment': 'stable' if np.mean(volatilities) < 10 else 'moderate' if np.mean(volatilities) < 20 else 'unstable'
            }
        
        # Epoch级别稳定性
        epoch_stability = {}
        for metric, trend_data in trends.items():
            stability = trend_data.get('stability', 0)
            epoch_stability[metric] = {
                'stability_score': stability,
                'stability_level': 'stable' if stability < 0.01 else 'moderate' if stability < 0.05 else 'unstable'
            }
        
        stability_metrics['epoch_level'] = epoch_stability
        
        return stability_metrics
    
    def save_detailed_metrics(self, output_path):
        """保存详细指标"""
        print(f"\n💾 保存详细指标: {output_path}")
        
        # 生成完整的指标数据
        complete_metrics = {
            'training_history': self.training_history,
            'detailed_metrics': self.detailed_metrics,
            'analysis': self.generate_detailed_analysis(),
            'extraction_timestamp': datetime.now().isoformat(),
            'source_log': str(self.log_file_path)
        }
        
        # 转换numpy类型为Python原生类型
        complete_metrics = self._convert_numpy_types(complete_metrics)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(complete_metrics, f, indent=2, ensure_ascii=False)
        
        print("✅ 指标保存完成")
        
        return complete_metrics
    
    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def main():
    """主函数"""
    print("📊 正确GPU训练指标提取")
    print("=" * 50)
    
    # 训练日志路径
    log_file = "/home/aaa/ws/bioastModel/experiments/gpu_training_run/training.log"
    
    # 创建提取器
    extractor = TrainingMetricsExtractor(log_file)
    
    # 提取所有指标
    extractor.extract_all_metrics()
    
    # 保存详细指标
    output_path = "/home/aaa/ws/bioastModel/correct_training_detailed_metrics.json"
    complete_metrics = extractor.save_detailed_metrics(output_path)
    
    # 打印关键发现
    print("\n🎯 关键指标摘要:")
    print("-" * 30)
    
    analysis = complete_metrics['analysis']
    
    # 训练总结
    summary = analysis.get('training_summary', {})
    print(f"总轮次: {summary.get('total_epochs', 0)}")
    print(f"最佳轮次: 第{summary.get('best_epoch', 0)}轮")
    print(f"最佳加权准确率: {summary.get('best_weighted_accuracy', 0):.4f}")
    print(f"最终加权准确率: {summary.get('final_weighted_accuracy', 0):.4f}")
    
    # 任务性能
    task_perf = analysis.get('task_performance_analysis', {})
    print(f"\n任务性能:")
    for task, perf in task_perf.items():
        assessment = perf.get('performance_assessment', {})
        print(f"- {task}: {assessment.get('overall_status', 'unknown')}")
    
    # 数据不平衡影响
    imbalance = analysis.get('data_imbalance_impact', {})
    print(f"\n数据不平衡影响:")
    for task, impact in imbalance.items():
        print(f"- {task}: 不平衡比例 {impact.get('imbalance_ratio', 1):.1f}, 影响程度 {impact.get('performance_impact', 'unknown')}")
    
    print(f"\n📊 详细指标已保存至: {output_path}")

if __name__ == "__main__":
    main()