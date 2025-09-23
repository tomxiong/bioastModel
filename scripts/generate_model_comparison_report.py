#!/usr/bin/env python3
"""
生成多任务模型性能对比报告
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

def load_training_history(experiment_dir):
    """加载训练历史"""
    history_file = os.path.join(experiment_dir, 'train_history.json')
    config_file = os.path.join(experiment_dir, 'config.json')
    
    history = None
    config = None
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    return history, config

def find_all_experiments():
    """查找所有实验"""
    experiments = []
    
    # 搜索模式
    patterns = [
        "experiments/fixed_efficientnet_b0_multitask_*",
        "experiments/resnet34_gpu_optimized_*",
        "experiments/fixed_mobilenetv3_multitask_*"
    ]
    
    for pattern in patterns:
        for exp_dir in glob.glob(pattern):
            if os.path.isdir(exp_dir):
                history, config = load_training_history(exp_dir)
                
                if history and config:
                    # 推断模型类型
                    model_type = None
                    if 'efficientnet_b0' in exp_dir:
                        model_type = 'EfficientNet-B0'
                    elif 'resnet34' in exp_dir:
                        model_type = 'ResNet-34'
                    elif 'mobilenetv3' in exp_dir:
                        model_type = 'MobileNetV3'
                    
                    if model_type:
                        experiments.append({
                            'model_type': model_type,
                            'experiment_dir': exp_dir,
                            'history': history,
                            'config': config
                        })
    
    return experiments

def analyze_experiment(experiment):
    """分析单个实验"""
    history = experiment['history']
    config = experiment['config']
    
    # 基本信息
    best_accuracy = max(history.get('val_accuracy', [0]))
    best_epoch = history['val_accuracy'].index(best_accuracy) + 1 if history.get('val_accuracy') else 0
    total_epochs = len(history.get('val_accuracy', []))
    final_accuracy = history['val_accuracy'][-1] if history.get('val_accuracy') else 0
    
    # 任务特定准确率
    task_accuracies = {}
    for task in ['growth_level', 'growth_pattern', 'interference_factors', 'microbe_type']:
        if task in history.get('task_accuracies', {}):
            task_history = history['task_accuracies'][task]
            if task_history:
                task_accuracies[task] = {
                    'best': max(task_history),
                    'final': task_history[-1],
                    'best_epoch': task_history.index(max(task_history)) + 1
                }
    
    # 训练配置
    training_config = {
        'batch_size': config.get('batch_size', 'Unknown'),
        'learning_rate': config.get('learning_rate', 'Unknown'),
        'epochs': config.get('epochs', total_epochs)
    }
    
    # 收敛分析
    if len(history.get('val_accuracy', [])) > 5:
        # 计算收敛速度 (达到80%最佳性能需要的epoch数)
        target_acc = best_accuracy * 0.8
        convergence_epoch = None
        for i, acc in enumerate(history['val_accuracy']):
            if acc >= target_acc:
                convergence_epoch = i + 1
                break
        
        # 计算稳定性 (最后5个epoch的标准差)
        last_5_acc = history['val_accuracy'][-5:]
        stability = np.std(last_5_acc) if len(last_5_acc) >= 5 else np.inf
    else:
        convergence_epoch = None
        stability = np.inf
    
    return {
        'model_type': experiment['model_type'],
        'experiment_dir': experiment['experiment_dir'],
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'final_accuracy': final_accuracy,
        'total_epochs': total_epochs,
        'task_accuracies': task_accuracies,
        'training_config': training_config,
        'convergence_epoch': convergence_epoch,
        'stability': stability,
        'history': history
    }

def create_training_curves_plot(experiments, output_dir):
    """创建训练曲线图"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('多任务模型训练曲线对比', fontsize=16, fontweight='bold')
    
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD']
    
    # 整体准确率
    ax = axes[0, 0]
    for i, exp in enumerate(experiments):
        history = exp['history']
        if 'val_accuracy' in history:
            epochs = range(1, len(history['val_accuracy']) + 1)
            ax.plot(epochs, history['val_accuracy'], 
                   label=exp['model_type'], color=colors[i % len(colors)], linewidth=2)
    
    ax.set_title('验证准确率', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('准确率 (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 训练损失
    ax = axes[0, 1]
    for i, exp in enumerate(experiments):
        history = exp['history']
        if 'train_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 
                   label=exp['model_type'], color=colors[i % len(colors)], linewidth=2)
    
    ax.set_title('训练损失', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 验证损失
    ax = axes[1, 0]
    for i, exp in enumerate(experiments):
        history = exp['history']
        if 'val_loss' in history:
            epochs = range(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'], 
                   label=exp['model_type'], color=colors[i % len(colors)], linewidth=2)
    
    ax.set_title('验证损失', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('损失')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 最佳准确率对比
    ax = axes[1, 1]
    models = [exp['model_type'] for exp in experiments]
    accuracies = [max(exp['history'].get('val_accuracy', [0])) for exp in experiments]
    
    bars = ax.bar(models, accuracies, color=colors[:len(models)], alpha=0.8)
    ax.set_title('最佳验证准确率对比', fontweight='bold')
    ax.set_ylabel('准确率 (%)')
    ax.set_ylim(0, max(accuracies) * 1.1)
    
    # 在柱状图上添加数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_task_performance_plot(experiments, output_dir):
    """创建各任务性能对比图"""
    tasks = ['growth_level', 'growth_pattern', 'interference_factors', 'microbe_type']
    task_names = ['Growth Level', 'Growth Pattern', 'Interference Factors', 'Microbe Type']
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('各任务性能对比', fontsize=16, fontweight='bold')
    
    colors = ['#2E86C1', '#E74C3C', '#28B463', '#F39C12', '#8E44AD']
    
    for idx, (task, task_name) in enumerate(zip(tasks, task_names)):
        ax = axes[idx // 2, idx % 2]
        
        models = []
        best_accs = []
        final_accs = []
        
        for exp in experiments:
            if task in exp['history'].get('task_accuracies', {}):
                task_history = exp['history']['task_accuracies'][task]
                if task_history:
                    models.append(exp['model_type'])
                    best_accs.append(max(task_history))
                    final_accs.append(task_history[-1])
        
        if models:
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, best_accs, width, label='最佳', 
                          color=colors[0], alpha=0.8)
            bars2 = ax.bar(x + width/2, final_accs, width, label='最终', 
                          color=colors[1], alpha=0.8)
            
            ax.set_title(f'{task_name}', fontweight='bold')
            ax.set_ylabel('准确率 (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'task_performance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_summary_table(analyzed_experiments):
    """创建汇总表格"""
    data = []
    
    for exp in analyzed_experiments:
        row = {
            '模型': exp['model_type'],
            '最佳准确率(%)': f"{exp['best_accuracy']:.2f}",
            '最佳轮次': exp['best_epoch'],
            '最终准确率(%)': f"{exp['final_accuracy']:.2f}",
            '总轮次': exp['total_epochs'],
            '批次大小': exp['training_config']['batch_size'],
            '学习率': exp['training_config']['learning_rate'],
            '收敛轮次': exp['convergence_epoch'] if exp['convergence_epoch'] else '-',
            '稳定性': f"{exp['stability']:.3f}" if exp['stability'] != np.inf else '-'
        }
        
        # 添加各任务最佳准确率
        for task in ['growth_level', 'growth_pattern', 'interference_factors', 'microbe_type']:
            if task in exp['task_accuracies']:
                row[f'{task}_best'] = f"{exp['task_accuracies'][task]['best']:.2f}%"
            else:
                row[f'{task}_best'] = '-'
        
        data.append(row)
    
    return data

def generate_markdown_report(analyzed_experiments, plots, output_dir):
    """生成Markdown报告"""
    report_lines = []
    
    # 标题和概述
    report_lines.append("# 多任务模型性能对比报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## 概述")
    report_lines.append("")
    report_lines.append(f"本报告对比了 {len(analyzed_experiments)} 个多任务深度学习模型的训练性能：")
    for exp in analyzed_experiments:
        report_lines.append(f"- **{exp['model_type']}**: 最佳验证准确率 {exp['best_accuracy']:.2f}%")
    report_lines.append("")
    
    # 排序模型
    sorted_exps = sorted(analyzed_experiments, key=lambda x: x['best_accuracy'], reverse=True)
    
    # 性能排名
    report_lines.append("## 🏆 性能排名")
    report_lines.append("")
    for i, exp in enumerate(sorted_exps, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        report_lines.append(f"{emoji} **{exp['model_type']}**: {exp['best_accuracy']:.2f}%")
    report_lines.append("")
    
    # 详细对比表格
    report_lines.append("## 📊 详细性能对比")
    report_lines.append("")
    
    # 基本信息表格
    report_lines.append("### 基本训练信息")
    report_lines.append("")
    report_lines.append("| 模型 | 最佳准确率 | 最佳轮次 | 最终准确率 | 总轮次 | 批次大小 | 学习率 | 收敛轮次 | 稳定性 |")
    report_lines.append("|------|------------|----------|------------|--------|----------|--------|----------|--------|")
    
    for exp in sorted_exps:
        convergence = exp['convergence_epoch'] if exp['convergence_epoch'] else '-'
        stability = f"{exp['stability']:.3f}" if exp['stability'] != np.inf else '-'
        report_lines.append(f"| {exp['model_type']} | {exp['best_accuracy']:.2f}% | {exp['best_epoch']} | "
                          f"{exp['final_accuracy']:.2f}% | {exp['total_epochs']} | "
                          f"{exp['training_config']['batch_size']} | {exp['training_config']['learning_rate']} | "
                          f"{convergence} | {stability} |")
    report_lines.append("")
    
    # 各任务性能表格
    report_lines.append("### 各任务最佳性能")
    report_lines.append("")
    report_lines.append("| 模型 | Growth Level | Growth Pattern | Interference Factors | Microbe Type |")
    report_lines.append("|------|--------------|----------------|---------------------|--------------|")
    
    for exp in sorted_exps:
        row_data = [exp['model_type']]
        for task in ['growth_level', 'growth_pattern', 'interference_factors', 'microbe_type']:
            if task in exp['task_accuracies']:
                row_data.append(f"{exp['task_accuracies'][task]['best']:.2f}%")
            else:
                row_data.append('-')
        report_lines.append("| " + " | ".join(row_data) + " |")
    report_lines.append("")
    
    # 训练曲线图
    report_lines.append("## 📈 训练曲线")
    report_lines.append("")
    if 'training_curves' in plots:
        report_lines.append(f"![训练曲线对比]({os.path.basename(plots['training_curves'])})")
    report_lines.append("")
    
    # 任务性能图
    report_lines.append("## 🎯 各任务性能对比")
    report_lines.append("")
    if 'task_performance' in plots:
        report_lines.append(f"![各任务性能对比]({os.path.basename(plots['task_performance'])})")
    report_lines.append("")
    
    # 详细分析
    report_lines.append("## 🔍 详细分析")
    report_lines.append("")
    
    best_model = sorted_exps[0]
    report_lines.append(f"### 最佳模型: {best_model['model_type']}")
    report_lines.append("")
    report_lines.append(f"- **最佳验证准确率**: {best_model['best_accuracy']:.2f}% (第{best_model['best_epoch']}轮)")
    report_lines.append(f"- **训练配置**: 批次大小={best_model['training_config']['batch_size']}, "
                       f"学习率={best_model['training_config']['learning_rate']}")
    
    if best_model['convergence_epoch']:
        report_lines.append(f"- **收敛速度**: 第{best_model['convergence_epoch']}轮达到80%最佳性能")
    
    report_lines.append("")
    report_lines.append("**各任务表现**:")
    for task in ['growth_level', 'growth_pattern', 'interference_factors', 'microbe_type']:
        if task in best_model['task_accuracies']:
            task_data = best_model['task_accuracies'][task]
            report_lines.append(f"- **{task}**: {task_data['best']:.2f}% (第{task_data['best_epoch']}轮)")
    report_lines.append("")
    
    # 改进建议
    report_lines.append("## 💡 改进建议")
    report_lines.append("")
    
    # 找出性能较差的任务
    all_task_perfs = {}
    for task in ['growth_level', 'growth_pattern', 'interference_factors', 'microbe_type']:
        task_accs = []
        for exp in analyzed_experiments:
            if task in exp['task_accuracies']:
                task_accs.append(exp['task_accuracies'][task]['best'])
        if task_accs:
            all_task_perfs[task] = max(task_accs)
    
    if all_task_perfs:
        worst_task = min(all_task_perfs.keys(), key=lambda k: all_task_perfs[k])
        report_lines.append(f"1. **{worst_task}任务表现相对较差** (最佳: {all_task_perfs[worst_task]:.2f}%)")
        report_lines.append("   - 建议增加针对性的数据增强")
        report_lines.append("   - 考虑调整损失函数权重")
        report_lines.append("   - 探索任务特定的架构优化")
        report_lines.append("")
    
    report_lines.append("2. **继续模型优化**")
    report_lines.append("   - 尝试更大的模型架构")
    report_lines.append("   - 实施集成学习策略") 
    report_lines.append("   - 进行超参数微调")
    report_lines.append("")
    
    report_lines.append("3. **数据质量提升**")
    report_lines.append("   - 分析错误样本")
    report_lines.append("   - 增加困难样本的标注")
    report_lines.append("   - 平衡各类别数据分布")
    report_lines.append("")
    
    # 保存报告
    report_path = os.path.join(output_dir, 'model_comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    return report_path

def main():
    print("🚀 生成多任务模型性能对比报告")
    print("=" * 60)
    
    # 查找实验
    experiments = find_all_experiments()
    
    if not experiments:
        print("❌ 未找到完整的训练实验!")
        return
    
    print(f"发现 {len(experiments)} 个训练实验:")
    for exp in experiments:
        print(f"  - {exp['model_type']}: {exp['experiment_dir']}")
    
    # 分析实验
    print("\n分析实验结果...")
    analyzed_experiments = [analyze_experiment(exp) for exp in experiments]
    
    # 创建输出目录
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成图表
    print("生成训练曲线图...")
    training_curves_path = create_training_curves_plot(experiments, output_dir)
    
    print("生成任务性能对比图...")
    task_performance_path = create_task_performance_plot(experiments, output_dir)
    
    plots = {
        'training_curves': training_curves_path,
        'task_performance': task_performance_path
    }
    
    # 生成报告
    print("生成Markdown报告...")
    report_path = generate_markdown_report(analyzed_experiments, plots, output_dir)
    
    # 保存JSON数据
    json_data = {
        'generation_time': datetime.now().isoformat(),
        'experiments': analyzed_experiments,
        'plots': plots
    }
    
    json_path = os.path.join(output_dir, 'model_comparison_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print(f"\n🎉 报告生成完成!")
    print(f"Markdown报告: {report_path}")
    print(f"JSON数据: {json_path}")
    print(f"训练曲线图: {training_curves_path}")
    print(f"任务性能图: {task_performance_path}")
    
    # 显示简要结果
    print(f"\n📊 性能排名:")
    sorted_exps = sorted(analyzed_experiments, key=lambda x: x['best_accuracy'], reverse=True)
    for i, exp in enumerate(sorted_exps, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(f"  {emoji} {exp['model_type']}: {exp['best_accuracy']:.2f}%")

if __name__ == "__main__":
    main()