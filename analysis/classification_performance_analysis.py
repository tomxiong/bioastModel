#!/usr/bin/env python3
"""
分类性能深度分析脚本
基于训练历史数据和数据分布，分析各类别的实际分类性能
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import pandas as pd

def load_training_history(history_path):
    """加载训练历史数据"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

def analyze_task_performance_trends(history):
    """分析各任务性能趋势"""
    task_accuracies = history['task_accuracies']
    
    analysis = {}
    for task, accuracies in task_accuracies.items():
        final_acc = accuracies[-1]
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        
        # 计算稳定性（后10轮的标准差）
        stability = np.std(accuracies[-10:]) if len(accuracies) >= 10 else np.std(accuracies)
        
        # 计算学习趋势（最后5轮vs前5轮的平均值）
        early_avg = np.mean(accuracies[:5])
        late_avg = np.mean(accuracies[-5:])
        improvement = late_avg - early_avg
        
        analysis[task] = {
            'final_accuracy': final_acc,
            'max_accuracy': max_acc,
            'min_accuracy': min_acc,
            'accuracy_range': max_acc - min_acc,
            'stability': stability,
            'improvement': improvement,
            'learning_efficiency': improvement / len(accuracies)
        }
    
    return analysis

def analyze_loss_convergence(history):
    """分析损失收敛情况"""
    individual_losses = history['individual_losses']
    
    convergence_analysis = {}
    for task, losses in individual_losses.items():
        if task == 'confidence':  # 跳过confidence任务
            continue
            
        initial_loss = losses[0]
        final_loss = losses[-1]
        min_loss = min(losses)
        
        # 计算收敛速度（损失减少50%所需轮数）
        target_loss = initial_loss * 0.5
        convergence_epoch = None
        for i, loss in enumerate(losses):
            if loss <= target_loss:
                convergence_epoch = i + 1
                break
        
        # 计算过拟合风险（最小损失后的损失增加）
        min_loss_idx = losses.index(min_loss)
        if min_loss_idx < len(losses) - 5:  # 至少有5轮后续数据
            post_min_losses = losses[min_loss_idx:]
            overfitting_risk = (max(post_min_losses) - min_loss) / min_loss
        else:
            overfitting_risk = 0
        
        convergence_analysis[task] = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'min_loss': min_loss,
            'loss_reduction': (initial_loss - final_loss) / initial_loss,
            'convergence_epoch': convergence_epoch,
            'overfitting_risk': overfitting_risk,
            'convergence_quality': 'good' if overfitting_risk < 0.1 else 'concerning'
        }
    
    return convergence_analysis

def calculate_class_performance_estimates(task_accuracies, data_distribution):
    """基于任务准确率和数据分布估算各类别性能"""
    
    # Growth Pattern类别性能估算
    growth_pattern_acc = task_accuracies['growth_pattern'][-1]  # 最终准确率78.99%
    
    # 基于数据分布和类别不平衡程度估算各类别性能
    growth_pattern_dist = {
        'clean': 5590, 'clustered': 5335, 'weak_scattered': 3314,
        'heavy_growth': 1702, 'focal': 1572, 'litter_center_dots': 876,
        'strong_scattered': 663, 'center_dots': 602, 'weak_scattered_pos': 253,
        'scattered': 36, 'irregular': 35, 'default_positive': 16
    }
    
    total_samples = sum(growth_pattern_dist.values())
    
    # 估算各类别性能（基于样本数量和整体准确率）
    class_performance = {}
    for class_name, count in growth_pattern_dist.items():
        # 样本比例
        ratio = count / total_samples
        
        # 基于样本数量估算性能（样本越多，性能越好）
        if count >= 1000:  # 大类别
            estimated_acc = growth_pattern_acc + 0.1  # 高于平均
        elif count >= 500:  # 中等类别
            estimated_acc = growth_pattern_acc
        elif count >= 100:  # 小类别
            estimated_acc = growth_pattern_acc - 0.15
        elif count >= 50:  # 很小类别
            estimated_acc = growth_pattern_acc - 0.25
        else:  # 极小类别
            estimated_acc = max(0.1, growth_pattern_acc - 0.4)  # 几乎无法学习
        
        class_performance[class_name] = {
            'sample_count': count,
            'sample_ratio': ratio,
            'estimated_accuracy': estimated_acc,
            'performance_level': 'excellent' if estimated_acc > 0.85 else
                               'good' if estimated_acc > 0.7 else
                               'poor' if estimated_acc > 0.4 else 'critical'
        }
    
    return class_performance

def analyze_interference_factors_performance(task_accuracies):
    """分析干扰因素检测性能"""
    interference_acc = task_accuracies['interference_factors'][-1]  # 最终准确率77.43%
    
    # 基于报告中的具体数据
    factor_performance = {
        'pores': {
            'accuracy': 0.7477,  # 报告中的具体数据
            'sample_count': 7450,
            'sample_ratio': 0.7546,
            'issue': '样本过多导致模型过度依赖，但准确率偏低'
        },
        'artifacts': {
            'accuracy': 0.9263,  # 报告中的数据
            'sample_count': 1484,
            'sample_ratio': 0.1503,
            'issue': '性能良好'
        },
        'debris': {
            'accuracy': 0.953,   # 报告中的数据
            'sample_count': 907,
            'sample_ratio': 0.0919,
            'issue': '性能优秀'
        },
        'contamination': {
            'accuracy': 0.9983,  # 报告中的数据，但可能不准确
            'sample_count': 32,
            'sample_ratio': 0.0032,
            'issue': '样本极少，实际检测能力可能很弱'
        }
    }
    
    return factor_performance

def generate_performance_report(task_analysis, convergence_analysis, class_performance, interference_performance):
    """生成性能分析报告"""
    
    report = []
    report.append("# 分类性能深度分析报告\n")
    
    # 1. 任务级别性能分析
    report.append("## 1. 任务级别性能分析\n")
    for task, metrics in task_analysis.items():
        report.append(f"### {task.upper()}任务")
        report.append(f"- **最终准确率**: {metrics['final_accuracy']:.4f}")
        report.append(f"- **最高准确率**: {metrics['max_accuracy']:.4f}")
        report.append(f"- **准确率波动范围**: {metrics['accuracy_range']:.4f}")
        report.append(f"- **稳定性** (标准差): {metrics['stability']:.4f}")
        report.append(f"- **学习改进幅度**: {metrics['improvement']:.4f}")
        report.append(f"- **学习效率**: {metrics['learning_efficiency']:.6f}")
        
        # 性能评估
        if metrics['final_accuracy'] > 0.9:
            performance_level = "优秀"
        elif metrics['final_accuracy'] > 0.8:
            performance_level = "良好"
        elif metrics['final_accuracy'] > 0.7:
            performance_level = "一般"
        else:
            performance_level = "需要改进"
        
        report.append(f"- **性能评级**: {performance_level}\n")
    
    # 2. 收敛性分析
    report.append("## 2. 损失收敛性分析\n")
    for task, metrics in convergence_analysis.items():
        report.append(f"### {task.upper()}任务收敛情况")
        report.append(f"- **初始损失**: {metrics['initial_loss']:.6f}")
        report.append(f"- **最终损失**: {metrics['final_loss']:.6f}")
        report.append(f"- **损失减少比例**: {metrics['loss_reduction']:.2%}")
        report.append(f"- **收敛轮数** (50%损失): {metrics['convergence_epoch'] or 'N/A'}")
        report.append(f"- **过拟合风险**: {metrics['overfitting_risk']:.4f}")
        report.append(f"- **收敛质量**: {metrics['convergence_quality']}\n")
    
    # 3. Growth Pattern类别性能分析
    report.append("## 3. Growth Pattern类别性能详细分析\n")
    
    # 按性能分组
    excellent_classes = []
    good_classes = []
    poor_classes = []
    critical_classes = []
    
    for class_name, metrics in class_performance.items():
        if metrics['performance_level'] == 'excellent':
            excellent_classes.append((class_name, metrics))
        elif metrics['performance_level'] == 'good':
            good_classes.append((class_name, metrics))
        elif metrics['performance_level'] == 'poor':
            poor_classes.append((class_name, metrics))
        else:
            critical_classes.append((class_name, metrics))
    
    report.append("### 优秀类别 (预估准确率 > 85%)")
    for class_name, metrics in excellent_classes:
        report.append(f"- **{class_name}**: {metrics['sample_count']}样本, 预估准确率{metrics['estimated_accuracy']:.2%}")
    
    report.append("\n### 良好类别 (预估准确率 70-85%)")
    for class_name, metrics in good_classes:
        report.append(f"- **{class_name}**: {metrics['sample_count']}样本, 预估准确率{metrics['estimated_accuracy']:.2%}")
    
    report.append("\n### 较差类别 (预估准确率 40-70%)")
    for class_name, metrics in poor_classes:
        report.append(f"- **{class_name}**: {metrics['sample_count']}样本, 预估准确率{metrics['estimated_accuracy']:.2%}")
    
    report.append("\n### 关键问题类别 (预估准确率 < 40%)")
    for class_name, metrics in critical_classes:
        report.append(f"- **{class_name}**: {metrics['sample_count']}样本, 预估准确率{metrics['estimated_accuracy']:.2%}")
    
    # 4. 干扰因素性能分析
    report.append("\n## 4. 干扰因素检测性能分析\n")
    for factor, metrics in interference_performance.items():
        report.append(f"### {factor.upper()}")
        report.append(f"- **准确率**: {metrics['accuracy']:.2%}")
        report.append(f"- **样本数量**: {metrics['sample_count']}")
        report.append(f"- **样本比例**: {metrics['sample_ratio']:.2%}")
        report.append(f"- **问题分析**: {metrics['issue']}\n")
    
    return '\n'.join(report)

def create_performance_visualizations(task_analysis, class_performance, output_dir):
    """创建性能可视化图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 任务性能对比图
    plt.figure(figsize=(12, 8))
    tasks = list(task_analysis.keys())
    final_accs = [task_analysis[task]['final_accuracy'] for task in tasks]
    max_accs = [task_analysis[task]['max_accuracy'] for task in tasks]
    stabilities = [task_analysis[task]['stability'] for task in tasks]
    
    x = np.arange(len(tasks))
    width = 0.25
    
    plt.bar(x - width, final_accs, width, label='最终准确率', alpha=0.8)
    plt.bar(x, max_accs, width, label='最高准确率', alpha=0.8)
    plt.bar(x + width, stabilities, width, label='稳定性(标准差)', alpha=0.8)
    
    plt.xlabel('任务')
    plt.ylabel('数值')
    plt.title('各任务性能对比')
    plt.xticks(x, tasks, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'task_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Growth Pattern类别性能分布
    plt.figure(figsize=(15, 10))
    
    class_names = list(class_performance.keys())
    sample_counts = [class_performance[name]['sample_count'] for name in class_names]
    estimated_accs = [class_performance[name]['estimated_accuracy'] for name in class_names]
    
    # 创建散点图：样本数量 vs 预估准确率
    colors = ['red' if acc < 0.4 else 'orange' if acc < 0.7 else 'green' for acc in estimated_accs]
    
    plt.scatter(sample_counts, estimated_accs, c=colors, s=100, alpha=0.7)
    
    for i, name in enumerate(class_names):
        plt.annotate(name, (sample_counts[i], estimated_accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('样本数量 (log scale)')
    plt.ylabel('预估准确率')
    plt.title('Growth Pattern类别：样本数量 vs 预估性能')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 添加性能区域标识
    plt.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, label='优秀 (>85%)')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='良好 (70-85%)')
    plt.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='关键问题 (<40%)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'growth_pattern_performance_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载训练历史数据
    history_path = "/home/aaa/ws/bioastModel/experiments/resnet34_gpu_optimized_20250919_021208/training_history.json"
    history = load_training_history(history_path)
    
    # 分析任务性能趋势
    task_analysis = analyze_task_performance_trends(history)
    
    # 分析损失收敛情况
    convergence_analysis = analyze_loss_convergence(history)
    
    # 计算类别性能估算
    class_performance = calculate_class_performance_estimates(history['task_accuracies'], None)
    
    # 分析干扰因素性能
    interference_performance = analyze_interference_factors_performance(history['task_accuracies'])
    
    # 生成报告
    report = generate_performance_report(task_analysis, convergence_analysis, 
                                       class_performance, interference_performance)
    
    # 保存报告
    output_dir = Path("performance_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "classification_performance_analysis.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 创建可视化图表
    create_performance_visualizations(task_analysis, class_performance, output_dir)
    
    print("✅ 分类性能分析完成！")
    print(f"📊 报告已保存到: {output_dir / 'classification_performance_analysis.md'}")
    print(f"📈 可视化图表已保存到: {output_dir}/")
    
    # 输出关键发现
    print("\n🔍 关键发现:")
    print("1. 任务性能排序:")
    sorted_tasks = sorted(task_analysis.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
    for task, metrics in sorted_tasks:
        print(f"   - {task}: {metrics['final_accuracy']:.2%}")
    
    print("\n2. 关键问题类别:")
    critical_classes = [(name, metrics) for name, metrics in class_performance.items() 
                       if metrics['performance_level'] == 'critical']
    for class_name, metrics in critical_classes:
        print(f"   - {class_name}: {metrics['sample_count']}样本, 预估准确率{metrics['estimated_accuracy']:.1%}")

if __name__ == "__main__":
    main()