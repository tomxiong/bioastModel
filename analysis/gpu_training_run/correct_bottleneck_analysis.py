#!/usr/bin/env python3
"""
正确GPU训练性能瓶颈分析脚本
基于提取的详细指标识别训练中的关键问题和瓶颈
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

class BottleneckAnalyzer:
    def __init__(self, metrics_file):
        self.metrics_file = Path(metrics_file)
        self.metrics_data = {}
        self.bottlenecks = {}
        self.critical_issues = {}
        
    def load_metrics(self):
        """加载指标数据"""
        print("📊 加载训练指标数据...")
        
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            self.metrics_data = json.load(f)
        
        print("✅ 指标数据加载完成")
    
    def analyze_all_bottlenecks(self):
        """分析所有性能瓶颈"""
        print("\n🔍 开始性能瓶颈分析...")
        
        # 任务级别瓶颈分析
        self._analyze_task_bottlenecks()
        
        # 数据级别瓶颈分析
        self._analyze_data_bottlenecks()
        
        # 训练过程瓶颈分析
        self._analyze_training_process_bottlenecks()
        
        # 模型架构瓶颈分析
        self._analyze_model_bottlenecks()
        
        # 收敛性瓶颈分析
        self._analyze_convergence_bottlenecks()
        
        print("✅ 瓶颈分析完成")
    
    def _analyze_task_bottlenecks(self):
        """分析任务级别瓶颈"""
        print("  🎯 分析任务级别瓶颈...")
        
        task_analysis = self.metrics_data.get('analysis', {}).get('task_performance_analysis', {})
        task_bottlenecks = {}
        
        for task_name, task_data in task_analysis.items():
            assessment = task_data.get('performance_assessment', {})
            final_metrics = task_data.get('final_metrics', {})
            
            # 获取准确率
            if task_name == 'interference_factors':
                accuracy = final_metrics.get('overall', {}).get('accuracy', 0)
            else:
                accuracy = final_metrics.get('accuracy', 0)
            
            overall_status = assessment.get('overall_status', 'unknown')
            accuracy_level = assessment.get('accuracy_level', 'unknown')
            
            # 识别瓶颈
            bottleneck_severity = self._assess_bottleneck_severity(accuracy, overall_status)
            
            task_bottlenecks[task_name] = {
                'accuracy': accuracy,
                'overall_status': overall_status,
                'accuracy_level': accuracy_level,
                'bottleneck_severity': bottleneck_severity,
                'issues': self._identify_task_issues(task_name, task_data),
                'impact_score': self._calculate_task_impact_score(accuracy, bottleneck_severity)
            }
        
        self.bottlenecks['task_level'] = task_bottlenecks
        
        # 识别最严重的任务瓶颈
        worst_task = max(task_bottlenecks.items(), 
                        key=lambda x: x[1]['impact_score'])
        
        self.critical_issues['worst_performing_task'] = {
            'task': worst_task[0],
            'details': worst_task[1],
            'priority': 'critical' if worst_task[1]['bottleneck_severity'] == 'severe' else 'high'
        }
    
    def _assess_bottleneck_severity(self, accuracy, status):
        """评估瓶颈严重程度"""
        if accuracy < 0.5 or status == 'needs_improvement':
            return 'severe'
        elif accuracy < 0.8 or status == 'acceptable':
            return 'moderate'
        elif accuracy < 0.9 or status == 'good':
            return 'minor'
        else:
            return 'none'
    
    def _identify_task_issues(self, task_name, task_data):
        """识别任务特定问题"""
        issues = []
        
        final_metrics = task_data.get('final_metrics', {})
        assessment = task_data.get('performance_assessment', {})
        
        # 准确率相关问题
        if task_name == 'interference_factors':
            accuracy = final_metrics.get('overall', {}).get('accuracy', 0)
            
            # 检查各类别性能
            category_analysis = task_data.get('category_analysis', {})
            poor_categories = []
            for cat, cat_data in category_analysis.items():
                if cat_data.get('performance_level') in ['poor', 'needs_improvement']:
                    poor_categories.append(cat)
            
            if poor_categories:
                issues.append(f"类别性能差: {', '.join(poor_categories)}")
            
            # 检查是否有0准确率问题
            if accuracy == 0:
                issues.append("严重问题: 准确率为0，可能存在标签或损失函数问题")
        else:
            accuracy = final_metrics.get('accuracy', 0)
            
            if accuracy < 0.7:
                issues.append("准确率过低")
            
            # 检查混淆矩阵问题
            cm_analysis = task_data.get('confusion_matrix_analysis', {})
            if cm_analysis:
                top_confusions = cm_analysis.get('top_confusions', [])
                if top_confusions:
                    worst_confusion = top_confusions[0]
                    if worst_confusion['percentage'] > 20:
                        issues.append(f"严重类别混淆: 类别{worst_confusion['true_class']}被误分为类别{worst_confusion['predicted_class']} ({worst_confusion['percentage']:.1f}%)")
        
        # 训练趋势问题
        training_trends = task_data.get('training_trends', {})
        improvement = training_trends.get('improvement', 0)
        stability = training_trends.get('stability', 0)
        
        if improvement < -5:
            issues.append("训练过程中性能下降")
        elif improvement < 2:
            issues.append("训练改进不足")
        
        if stability > 0.05:
            issues.append("训练不稳定")
        
        # 收敛问题
        if assessment.get('improvement_trend') == 'declining':
            issues.append("性能呈下降趋势")
        
        if assessment.get('stability_level') == 'unstable':
            issues.append("训练稳定性差")
        
        return issues
    
    def _calculate_task_impact_score(self, accuracy, severity):
        """计算任务影响分数（越高越严重）"""
        severity_scores = {
            'none': 0,
            'minor': 1,
            'moderate': 3,
            'severe': 5
        }
        
        accuracy_penalty = max(0, (0.9 - accuracy) * 10)  # 准确率低于90%的惩罚
        severity_score = severity_scores.get(severity, 0)
        
        return accuracy_penalty + severity_score
    
    def _analyze_data_bottlenecks(self):
        """分析数据级别瓶颈"""
        print("  📊 分析数据级别瓶颈...")
        
        imbalance_impact = self.metrics_data.get('analysis', {}).get('data_imbalance_impact', {})
        data_bottlenecks = {}
        
        for task, impact_data in imbalance_impact.items():
            imbalance_ratio = impact_data.get('imbalance_ratio', 1)
            performance_impact = impact_data.get('performance_impact', 'minimal')
            distribution = impact_data.get('distribution', {})
            
            # 识别数据问题
            data_issues = []
            
            if imbalance_ratio > 10:
                data_issues.append(f"严重类别不平衡 (比例: {imbalance_ratio:.1f})")
            elif imbalance_ratio > 5:
                data_issues.append(f"中等类别不平衡 (比例: {imbalance_ratio:.1f})")
            
            if performance_impact in ['severe', 'significant']:
                data_issues.append(f"不平衡严重影响性能 ({performance_impact})")
            
            # 检查极小类别
            if distribution:
                min_samples = min(distribution.values())
                max_samples = max(distribution.values())
                
                if min_samples < 10:
                    data_issues.append(f"存在极小类别 (最少样本: {min_samples})")
                
                # 找出样本数过少的类别
                small_classes = [k for k, v in distribution.items() if v < 50]
                if small_classes:
                    data_issues.append(f"样本不足的类别: {', '.join(small_classes)}")
            
            data_bottlenecks[task] = {
                'imbalance_ratio': imbalance_ratio,
                'performance_impact': performance_impact,
                'issues': data_issues,
                'severity': self._assess_data_bottleneck_severity(imbalance_ratio, performance_impact),
                'distribution': distribution
            }
        
        self.bottlenecks['data_level'] = data_bottlenecks
        
        # 识别最严重的数据瓶颈
        severe_data_issues = {k: v for k, v in data_bottlenecks.items() 
                             if v['severity'] in ['severe', 'critical']}
        
        if severe_data_issues:
            worst_data_issue = max(severe_data_issues.items(), 
                                 key=lambda x: x[1]['imbalance_ratio'])
            
            self.critical_issues['worst_data_imbalance'] = {
                'task': worst_data_issue[0],
                'details': worst_data_issue[1],
                'priority': 'critical'
            }
    
    def _assess_data_bottleneck_severity(self, ratio, impact):
        """评估数据瓶颈严重程度"""
        if ratio > 50 or impact == 'severe':
            return 'critical'
        elif ratio > 10 or impact == 'significant':
            return 'severe'
        elif ratio > 5 or impact == 'moderate':
            return 'moderate'
        else:
            return 'minor'
    
    def _analyze_training_process_bottlenecks(self):
        """分析训练过程瓶颈"""
        print("  🔄 分析训练过程瓶颈...")
        
        convergence_analysis = self.metrics_data.get('analysis', {}).get('convergence_analysis', {})
        stability_analysis = self.metrics_data.get('analysis', {}).get('training_stability', {})
        
        process_bottlenecks = {
            'convergence_issues': [],
            'stability_issues': [],
            'optimization_issues': []
        }
        
        # 收敛问题
        convergence_status = convergence_analysis.get('convergence_status', 'unknown')
        overfitting_signs = convergence_analysis.get('overfitting_signs', [])
        
        if convergence_status == 'still_changing':
            process_bottlenecks['convergence_issues'].append("训练未充分收敛")
        
        if overfitting_signs:
            process_bottlenecks['convergence_issues'].extend(overfitting_signs)
        
        # 损失趋势问题
        loss_trends = convergence_analysis.get('loss_trends', {})
        train_improvement = loss_trends.get('train_loss_improvement', 0)
        val_improvement = loss_trends.get('val_loss_improvement', 0)
        
        if train_improvement < 10:
            process_bottlenecks['optimization_issues'].append("训练损失改进不足")
        
        if val_improvement < 5:
            process_bottlenecks['optimization_issues'].append("验证损失改进不足")
        
        if train_improvement > val_improvement + 10:
            process_bottlenecks['convergence_issues'].append("可能存在过拟合")
        
        # 稳定性问题
        if stability_analysis:
            batch_stability = stability_analysis.get('batch_level', {})
            epoch_stability = stability_analysis.get('epoch_level', {})
            
            if batch_stability.get('stability_assessment') == 'unstable':
                process_bottlenecks['stability_issues'].append("批次级别训练不稳定")
            
            unstable_metrics = [k for k, v in epoch_stability.items() 
                              if v.get('stability_level') == 'unstable']
            if unstable_metrics:
                process_bottlenecks['stability_issues'].append(f"不稳定的指标: {', '.join(unstable_metrics)}")
        
        self.bottlenecks['training_process'] = process_bottlenecks
        
        # 评估训练过程瓶颈严重程度
        total_issues = sum(len(issues) for issues in process_bottlenecks.values())
        if total_issues > 3:
            self.critical_issues['training_process_issues'] = {
                'details': process_bottlenecks,
                'total_issues': total_issues,
                'priority': 'high'
            }
    
    def _analyze_model_bottlenecks(self):
        """分析模型架构瓶颈"""
        print("  🏗️ 分析模型架构瓶颈...")
        
        # 基于任务性能差异分析多任务学习问题
        task_analysis = self.bottlenecks.get('task_level', {})
        
        model_bottlenecks = {
            'multi_task_issues': [],
            'capacity_issues': [],
            'architecture_issues': []
        }
        
        # 多任务学习问题
        task_accuracies = {}
        for task, data in task_analysis.items():
            task_accuracies[task] = data['accuracy']
        
        if task_accuracies:
            max_acc = max(task_accuracies.values())
            min_acc = min(task_accuracies.values())
            acc_gap = max_acc - min_acc
            
            if acc_gap > 0.3:
                model_bottlenecks['multi_task_issues'].append(f"任务间性能差异过大 ({acc_gap:.3f})")
            
            # 检查是否有任务完全失败
            failed_tasks = [task for task, acc in task_accuracies.items() if acc < 0.1]
            if failed_tasks:
                model_bottlenecks['multi_task_issues'].append(f"任务完全失败: {', '.join(failed_tasks)}")
            
            # 检查任务权重问题
            poor_tasks = [task for task, acc in task_accuracies.items() if acc < 0.7]
            if len(poor_tasks) > 1:
                model_bottlenecks['multi_task_issues'].append("多个任务性能不佳，可能需要调整任务权重")
        
        # 模型容量问题
        training_summary = self.metrics_data.get('analysis', {}).get('training_summary', {})
        convergence_point = training_summary.get('convergence_point', 10)
        
        if convergence_point < 3:
            model_bottlenecks['capacity_issues'].append("过早收敛，可能模型容量不足")
        elif convergence_point > 8:
            model_bottlenecks['capacity_issues'].append("收敛缓慢，可能需要调整学习率或优化器")
        
        # 架构特定问题（基于MobileNetV3）
        # 检查是否适合多任务学习
        if len([task for task, data in task_analysis.items() 
                if data['bottleneck_severity'] in ['severe', 'moderate']]) > 1:
            model_bottlenecks['architecture_issues'].append("MobileNetV3可能不适合当前多任务学习场景")
        
        self.bottlenecks['model_architecture'] = model_bottlenecks
        
        # 评估模型瓶颈严重程度
        total_issues = sum(len(issues) for issues in model_bottlenecks.values())
        if total_issues > 2:
            self.critical_issues['model_architecture_issues'] = {
                'details': model_bottlenecks,
                'total_issues': total_issues,
                'priority': 'medium'
            }
    
    def _analyze_convergence_bottlenecks(self):
        """分析收敛性瓶颈"""
        print("  📈 分析收敛性瓶颈...")
        
        training_history = self.metrics_data.get('training_history', {})
        epochs = training_history.get('epochs', [])
        
        convergence_bottlenecks = {
            'convergence_speed': 'unknown',
            'final_performance': 'unknown',
            'stability': 'unknown',
            'issues': []
        }
        
        if epochs:
            # 分析收敛速度
            best_epoch = max(epochs, key=lambda x: x['weighted_accuracy'])['epoch']
            total_epochs = len(epochs)
            
            if best_epoch <= 3:
                convergence_bottlenecks['convergence_speed'] = 'too_fast'
                convergence_bottlenecks['issues'].append("收敛过快，可能欠拟合")
            elif best_epoch >= total_epochs - 2:
                convergence_bottlenecks['convergence_speed'] = 'too_slow'
                convergence_bottlenecks['issues'].append("收敛过慢，可能需要更多训练轮次")
            else:
                convergence_bottlenecks['convergence_speed'] = 'normal'
            
            # 分析最终性能
            final_acc = epochs[-1]['weighted_accuracy']
            best_acc = best_epoch_data = max(epochs, key=lambda x: x['weighted_accuracy'])['weighted_accuracy']
            
            if final_acc < best_acc - 0.02:
                convergence_bottlenecks['issues'].append("最终性能低于最佳性能，存在过拟合")
            
            if best_acc < 0.85:
                convergence_bottlenecks['final_performance'] = 'poor'
                convergence_bottlenecks['issues'].append("最佳性能不足")
            elif best_acc < 0.92:
                convergence_bottlenecks['final_performance'] = 'moderate'
            else:
                convergence_bottlenecks['final_performance'] = 'good'
            
            # 分析稳定性
            if len(epochs) >= 5:
                recent_accs = [e['weighted_accuracy'] for e in epochs[-5:]]
                acc_std = np.std(recent_accs)
                
                if acc_std > 0.02:
                    convergence_bottlenecks['stability'] = 'unstable'
                    convergence_bottlenecks['issues'].append("训练后期不稳定")
                else:
                    convergence_bottlenecks['stability'] = 'stable'
        
        self.bottlenecks['convergence'] = convergence_bottlenecks
    
    def prioritize_bottlenecks(self):
        """优先级排序瓶颈"""
        print("\n🎯 瓶颈优先级排序...")
        
        all_bottlenecks = []
        
        # 任务级别瓶颈
        for task, data in self.bottlenecks.get('task_level', {}).items():
            severity = data['bottleneck_severity']
            impact_score = data['impact_score']
            
            priority_score = self._calculate_priority_score(severity, impact_score, 'task')
            
            all_bottlenecks.append({
                'type': 'task',
                'name': task,
                'severity': severity,
                'impact_score': impact_score,
                'priority_score': priority_score,
                'issues': data['issues'],
                'details': data
            })
        
        # 数据级别瓶颈
        for task, data in self.bottlenecks.get('data_level', {}).items():
            severity = data['severity']
            imbalance_ratio = data['imbalance_ratio']
            
            priority_score = self._calculate_priority_score(severity, imbalance_ratio, 'data')
            
            all_bottlenecks.append({
                'type': 'data',
                'name': f"{task}_data_imbalance",
                'severity': severity,
                'impact_score': imbalance_ratio,
                'priority_score': priority_score,
                'issues': data['issues'],
                'details': data
            })
        
        # 训练过程瓶颈
        process_issues = self.bottlenecks.get('training_process', {})
        total_process_issues = sum(len(issues) for issues in process_issues.values())
        
        if total_process_issues > 0:
            severity = 'severe' if total_process_issues > 3 else 'moderate'
            priority_score = self._calculate_priority_score(severity, total_process_issues, 'process')
            
            all_bottlenecks.append({
                'type': 'training_process',
                'name': 'training_process_issues',
                'severity': severity,
                'impact_score': total_process_issues,
                'priority_score': priority_score,
                'issues': [issue for issues in process_issues.values() for issue in issues],
                'details': process_issues
            })
        
        # 模型架构瓶颈
        model_issues = self.bottlenecks.get('model_architecture', {})
        total_model_issues = sum(len(issues) for issues in model_issues.values())
        
        if total_model_issues > 0:
            severity = 'moderate' if total_model_issues > 2 else 'minor'
            priority_score = self._calculate_priority_score(severity, total_model_issues, 'model')
            
            all_bottlenecks.append({
                'type': 'model_architecture',
                'name': 'model_architecture_issues',
                'severity': severity,
                'impact_score': total_model_issues,
                'priority_score': priority_score,
                'issues': [issue for issues in model_issues.values() for issue in issues],
                'details': model_issues
            })
        
        # 按优先级分数排序
        all_bottlenecks.sort(key=lambda x: x['priority_score'], reverse=True)
        
        self.bottlenecks['prioritized'] = all_bottlenecks
        
        return all_bottlenecks
    
    def _calculate_priority_score(self, severity, impact_score, bottleneck_type):
        """计算优先级分数"""
        severity_weights = {
            'critical': 10,
            'severe': 8,
            'moderate': 5,
            'minor': 2,
            'none': 0
        }
        
        type_weights = {
            'task': 1.0,
            'data': 0.8,
            'process': 0.6,
            'model': 0.4
        }
        
        severity_score = severity_weights.get(severity, 0)
        type_weight = type_weights.get(bottleneck_type, 0.5)
        
        # 归一化影响分数
        normalized_impact = min(impact_score / 10, 1.0) if isinstance(impact_score, (int, float)) else 0.5
        
        return severity_score * type_weight + normalized_impact * 5
    
    def generate_bottleneck_report(self):
        """生成瓶颈分析报告"""
        print("\n📝 生成瓶颈分析报告...")
        
        prioritized_bottlenecks = self.prioritize_bottlenecks()
        
        report = {
            'analysis_summary': {
                'total_bottlenecks': len(prioritized_bottlenecks),
                'critical_issues': len([b for b in prioritized_bottlenecks if b['severity'] in ['critical', 'severe']]),
                'analysis_timestamp': datetime.now().isoformat(),
                'source_metrics': str(self.metrics_file)
            },
            'critical_bottlenecks': prioritized_bottlenecks[:5],  # 前5个最严重的瓶颈
            'detailed_analysis': {
                'task_level': self.bottlenecks.get('task_level', {}),
                'data_level': self.bottlenecks.get('data_level', {}),
                'training_process': self.bottlenecks.get('training_process', {}),
                'model_architecture': self.bottlenecks.get('model_architecture', {}),
                'convergence': self.bottlenecks.get('convergence', {})
            },
            'all_bottlenecks': prioritized_bottlenecks,
            'recommendations': self._generate_immediate_recommendations(prioritized_bottlenecks)
        }
        
        return report
    
    def _generate_immediate_recommendations(self, bottlenecks):
        """生成即时改进建议"""
        recommendations = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_strategies': []
        }
        
        # 基于最严重的瓶颈生成建议
        top_bottlenecks = bottlenecks[:3]
        
        for bottleneck in top_bottlenecks:
            if bottleneck['type'] == 'task':
                if bottleneck['severity'] in ['critical', 'severe']:
                    if 'interference_factors' in bottleneck['name']:
                        recommendations['immediate_actions'].append("检查interference_factors任务的标签和损失函数配置")
                        recommendations['immediate_actions'].append("验证interference_factors数据预处理流程")
                    
                    recommendations['short_term_improvements'].append(f"调整{bottleneck['name']}任务权重")
                    recommendations['short_term_improvements'].append(f"为{bottleneck['name']}任务使用Focal Loss")
            
            elif bottleneck['type'] == 'data':
                if bottleneck['details']['imbalance_ratio'] > 10:
                    recommendations['immediate_actions'].append("实施数据重采样策略")
                    recommendations['short_term_improvements'].append("使用SMOTE或其他过采样技术")
                
                recommendations['long_term_strategies'].append("收集更多少数类样本")
            
            elif bottleneck['type'] == 'training_process':
                if 'overfitting' in str(bottleneck['issues']).lower():
                    recommendations['immediate_actions'].append("添加更强的正则化")
                    recommendations['short_term_improvements'].append("实施早停策略")
                
                if 'unstable' in str(bottleneck['issues']).lower():
                    recommendations['immediate_actions'].append("降低学习率")
                    recommendations['short_term_improvements'].append("使用学习率调度器")
            
            elif bottleneck['type'] == 'model_architecture':
                recommendations['long_term_strategies'].append("考虑使用更适合多任务学习的架构")
                recommendations['short_term_improvements'].append("调整任务特定的头部结构")
        
        return recommendations
    
    def save_bottleneck_analysis(self, output_path):
        """保存瓶颈分析结果"""
        print(f"\n💾 保存瓶颈分析: {output_path}")
        
        report = self.generate_bottleneck_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("✅ 瓶颈分析保存完成")
        
        return report

def main():
    """主函数"""
    print("🔍 正确GPU训练性能瓶颈分析")
    print("=" * 50)
    
    # 指标文件路径
    metrics_file = "/home/aaa/ws/bioastModel/correct_training_detailed_metrics.json"
    
    # 创建分析器
    analyzer = BottleneckAnalyzer(metrics_file)
    
    # 加载指标
    analyzer.load_metrics()
    
    # 分析所有瓶颈
    analyzer.analyze_all_bottlenecks()
    
    # 保存分析结果
    output_path = "/home/aaa/ws/bioastModel/correct_training_bottleneck_analysis.json"
    report = analyzer.save_bottleneck_analysis(output_path)
    
    # 打印关键发现
    print("\n🎯 关键瓶颈发现:")
    print("-" * 30)
    
    critical_bottlenecks = report['critical_bottlenecks']
    
    for i, bottleneck in enumerate(critical_bottlenecks[:3], 1):
        print(f"{i}. {bottleneck['name']} ({bottleneck['type']})")
        print(f"   严重程度: {bottleneck['severity']}")
        print(f"   优先级分数: {bottleneck['priority_score']:.2f}")
        print(f"   主要问题: {', '.join(bottleneck['issues'][:2])}")
        print()
    
    # 打印即时建议
    recommendations = report['recommendations']
    print("🚀 即时改进建议:")
    print("-" * 20)
    
    for action in recommendations['immediate_actions'][:3]:
        print(f"• {action}")
    
    print(f"\n📊 详细分析已保存至: {output_path}")

if __name__ == "__main__":
    main()