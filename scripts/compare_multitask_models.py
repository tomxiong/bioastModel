#!/usr/bin/env python3
"""
多任务模型对比分析脚本
对比GrayColonyNet和EfficientNet-B0两个多任务模型的综合性能
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def find_experiment_results():
    """查找两个模型的实验结果"""
    experiments_dir = Path(project_root) / 'experiments'
    
    # 查找GrayColonyNet实验
    gray_dirs = [d for d in experiments_dir.iterdir() 
                if d.is_dir() and 'gray_colony_net' in d.name]
    gray_experiment = sorted(gray_dirs, key=lambda x: x.name)[-1] if gray_dirs else None
    
    # 查找EfficientNet-B0实验
    efficient_dirs = [d for d in experiments_dir.iterdir() 
                     if d.is_dir() and 'efficientnet_b0' in d.name]
    efficient_experiment = sorted(efficient_dirs, key=lambda x: x.name)[-1] if efficient_dirs else None
    
    return gray_experiment, efficient_experiment


def load_experiment_data(experiment_dir, model_name):
    """加载实验数据"""
    print(f"加载 {model_name} 实验数据: {experiment_dir}")
    
    data = {
        'model_name': model_name,
        'experiment_dir': experiment_dir,
        'training_history': None,
        'test_metrics': None,
        'error_analysis': None,
        'model_info': None
    }
    
    # 加载训练历史
    history_file = experiment_dir / 'training_history.json'
    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            data['training_history'] = json.load(f)
    
    # 查找测试指标文件
    test_files = list(experiment_dir.glob('test_metrics_*.json'))
    if test_files:
        latest_test_file = sorted(test_files, key=lambda x: x.name)[-1]
        with open(latest_test_file, 'r', encoding='utf-8') as f:
            data['test_metrics'] = json.load(f)
    
    # 查找错误分析文件
    error_files = list(experiment_dir.glob('error_pattern_analysis*.json'))
    if error_files:
        latest_error_file = sorted(error_files, key=lambda x: x.name)[-1]
        with open(latest_error_file, 'r', encoding='utf-8') as f:
            data['error_analysis'] = json.load(f)
    
    # 查找模型信息
    config_file = experiment_dir / 'config.json'
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            data['model_info'] = json.load(f)
    
    return data


def compare_training_curves(gray_data, efficient_data, output_dir):
    """对比训练曲线"""
    print("生成训练曲线对比...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Curves Comparison: GrayColonyNet vs EfficientNet-B0', fontsize=16)
    
    # 提取训练历史
    gray_history = gray_data.get('training_history', {})
    efficient_history = efficient_data.get('training_history', {})
    
    # 训练损失对比
    if gray_history and efficient_history:
        axes[0, 0].plot(gray_history.get('train_loss', []), 'b-', label='GrayColonyNet', linewidth=2)
        axes[0, 0].plot(efficient_history.get('train_loss', []), 'r-', label='EfficientNet-B0', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 验证损失对比
        axes[0, 1].plot(gray_history.get('val_loss', []), 'b--', label='GrayColonyNet', linewidth=2)
        axes[0, 1].plot(efficient_history.get('val_loss', []), 'r--', label='EfficientNet-B0', linewidth=2)
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 训练准确率对比
        axes[1, 0].plot(gray_history.get('train_acc', []), 'b-', label='GrayColonyNet', linewidth=2)
        axes[1, 0].plot(efficient_history.get('train_acc', []), 'r-', label='EfficientNet-B0', linewidth=2)
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 验证准确率对比
        axes[1, 1].plot(gray_history.get('val_acc', []), 'b--', label='GrayColonyNet', linewidth=2)
        axes[1, 1].plot(efficient_history.get('val_acc', []), 'r--', label='EfficientNet-B0', linewidth=2)
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = output_dir / 'training_curves_comparison.png'
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return curve_path


def compare_task_performance(gray_data, efficient_data, output_dir):
    """对比各任务性能"""
    print("生成任务性能对比...")
    
    # 提取测试指标，处理None情况
    gray_metrics = gray_data.get('test_metrics') or {}
    efficient_metrics = efficient_data.get('test_metrics') or {}
    
    task_names = ['growth_level', 'growth_pattern', 'interference_factors', 'fine_grained']
    
    # 准备数据
    comparison_data = []
    for task in task_names:
        gray_acc = gray_metrics.get(task, {}).get('accuracy', 0)
        efficient_acc = efficient_metrics.get(task, {}).get('accuracy', 0)
        
        comparison_data.append({
            'Task': task.replace('_', ' ').title(),
            'GrayColonyNet': gray_acc * 100,
            'EfficientNet-B0': efficient_acc * 100
        })
    
    # 添加整体准确率
    gray_overall = gray_metrics.get('overall', {}).get('accuracy', 0)
    efficient_overall = efficient_metrics.get('overall', {}).get('accuracy', 0)
    comparison_data.append({
        'Task': 'Overall',
        'GrayColonyNet': gray_overall * 100,
        'EfficientNet-B0': efficient_overall * 100
    })
    
    df = pd.DataFrame(comparison_data)
    
    # 绘制对比图
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(df))
    width = 0.35
    
    ax.bar(x - width/2, df['GrayColonyNet'], width, label='GrayColonyNet', alpha=0.8)
    ax.bar(x + width/2, df['EfficientNet-B0'], width, label='EfficientNet-B0', alpha=0.8)
    
    ax.set_xlabel('Tasks')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Task Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Task'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (gray_val, eff_val) in enumerate(zip(df['GrayColonyNet'], df['EfficientNet-B0'])):
        ax.text(i - width/2, gray_val + 1, f'{gray_val:.1f}%', ha='center', fontsize=9)
        ax.text(i + width/2, eff_val + 1, f'{eff_val:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    performance_path = output_dir / 'task_performance_comparison.png'
    plt.savefig(performance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return performance_path, df


def compare_error_patterns(gray_data, efficient_data, output_dir):
    """对比错误模式"""
    print("分析错误模式对比...")
    
    gray_errors = gray_data.get('error_analysis', {})
    efficient_errors = efficient_data.get('error_analysis', {})
    
    # 错误总数对比
    gray_total_errors = gray_errors.get('total_errors', 0)
    efficient_total_errors = efficient_errors.get('total_errors', 0)
    
    # 各任务错误数对比
    error_comparison = []
    task_names = ['growth_level', 'growth_pattern', 'interference_factors', 'fine_grained']
    
    for task in task_names:
        gray_task_errors = gray_errors.get('error_by_task', {}).get(f'{task}_errors', 0)
        efficient_task_errors = efficient_errors.get('error_by_task', {}).get(task, 0)
        
        error_comparison.append({
            'Task': task.replace('_', ' ').title(),
            'GrayColonyNet': gray_task_errors,
            'EfficientNet-B0': efficient_task_errors
        })
    
    df_errors = pd.DataFrame(error_comparison)
    
    # 绘制错误数对比
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(df_errors))
    width = 0.35
    
    ax.bar(x - width/2, df_errors['GrayColonyNet'], width, label='GrayColonyNet', alpha=0.8, color='orange')
    ax.bar(x + width/2, df_errors['EfficientNet-B0'], width, label='EfficientNet-B0', alpha=0.8, color='red')
    
    ax.set_xlabel('Tasks')
    ax.set_ylabel('Number of Errors')
    ax.set_title('Error Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df_errors['Task'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (gray_val, eff_val) in enumerate(zip(df_errors['GrayColonyNet'], df_errors['EfficientNet-B0'])):
        ax.text(i - width/2, gray_val + 0.5, str(gray_val), ha='center', fontsize=9)
        ax.text(i + width/2, eff_val + 0.5, str(eff_val), ha='center', fontsize=9)
    
    plt.tight_layout()
    error_path = output_dir / 'error_distribution_comparison.png'
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return error_path, {'gray_total': gray_total_errors, 'efficient_total': efficient_total_errors}


def generate_comprehensive_report(gray_data, efficient_data, comparison_results, output_dir):
    """生成综合对比报告"""
    
    report_path = output_dir / 'comprehensive_model_comparison.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 多任务模型综合对比分析报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 模型概述\n\n")
        f.write("### GrayColonyNet (CNN-Transformer混合架构)\n")
        f.write("- **架构特点**: 结合CNN和Transformer的混合多任务学习架构\n")
        f.write("- **设计理念**: 利用CNN提取局部特征，Transformer建模全局依赖关系\n")
        f.write("- **任务头设计**: 层次化多任务头，考虑任务间依赖关系\n\n")
        
        f.write("### EfficientNet-B0多任务版 (改进EfficientNet架构)\n")
        f.write("- **架构特点**: 基于EfficientNet-B0的多任务学习架构\n")
        f.write("- **设计理念**: 采用复合缩放和跨任务注意力机制\n")
        f.write("- **任务头设计**: 层次化设计 + 自适应任务权重学习\n\n")
        
        # 性能对比
        f.write("## 性能对比\n\n")
        
        # 整体性能，处理None情况
        gray_metrics = gray_data.get('test_metrics') or {}
        efficient_metrics = efficient_data.get('test_metrics') or {}
        
        gray_overall = gray_metrics.get('overall', {}).get('accuracy', 0) * 100
        efficient_overall = efficient_metrics.get('overall', {}).get('accuracy', 0) * 100
        
        f.write("### 整体性能\n")
        f.write(f"- **GrayColonyNet**: {gray_overall:.2f}%\n")
        f.write(f"- **EfficientNet-B0**: {efficient_overall:.2f}%\n")
        
        if efficient_overall > gray_overall:
            f.write(f"- **优势**: EfficientNet-B0领先 {efficient_overall - gray_overall:.2f}%\n\n")
        elif gray_overall > efficient_overall:
            f.write(f"- **优势**: GrayColonyNet领先 {gray_overall - efficient_overall:.2f}%\n\n")
        else:
            f.write("- **结果**: 两个模型性能相当\n\n")
        
        # 各任务详细对比
        f.write("### 各任务性能详细对比\n\n")
        performance_df = comparison_results.get('performance_df')
        if performance_df is not None:
            f.write("| 任务 | GrayColonyNet | EfficientNet-B0 | 差异 |\n")
            f.write("|------|---------------|-----------------|------|\n")
            for _, row in performance_df.iterrows():
                gray_val = row['GrayColonyNet']
                eff_val = row['EfficientNet-B0']
                diff = eff_val - gray_val
                diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
                f.write(f"| {row['Task']} | {gray_val:.2f}% | {eff_val:.2f}% | {diff_str} |\n")
        f.write("\n")
        
        # 错误分析对比
        f.write("## 错误分析对比\n\n")
        error_stats = comparison_results.get('error_stats', {})
        gray_errors = error_stats.get('gray_total', 0)
        efficient_errors = error_stats.get('efficient_total', 0)
        
        f.write(f"### 错误样本数量\n")
        f.write(f"- **GrayColonyNet**: {gray_errors} 个错误样本\n")
        f.write(f"- **EfficientNet-B0**: {efficient_errors} 个错误样本\n")
        
        if efficient_errors < gray_errors:
            f.write(f"- **改进**: EfficientNet-B0减少了 {gray_errors - efficient_errors} 个错误样本\n\n")
        elif gray_errors < efficient_errors:
            f.write(f"- **退化**: EfficientNet-B0增加了 {efficient_errors - gray_errors} 个错误样本\n\n")
        else:
            f.write("- **结果**: 两个模型错误数量相同\n\n")
        
        # 训练效率对比
        f.write("## 训练效率分析\n\n")
        
        # 收敛性分析
        gray_history = gray_data.get('training_history', {})
        efficient_history = efficient_data.get('training_history', {})
        
        if gray_history and efficient_history:
            gray_epochs = len(gray_history.get('val_loss', []))
            efficient_epochs = len(efficient_history.get('val_loss', []))
            
            f.write(f"### 训练轮数\n")
            f.write(f"- **GrayColonyNet**: {gray_epochs} 轮\n")
            f.write(f"- **EfficientNet-B0**: {efficient_epochs} 轮\n\n")
            
            # 最终验证损失
            gray_final_loss = gray_history.get('val_loss', [])[-1] if gray_history.get('val_loss') else 0
            efficient_final_loss = efficient_history.get('val_loss', [])[-1] if efficient_history.get('val_loss') else 0
            
            f.write(f"### 最终验证损失\n")
            f.write(f"- **GrayColonyNet**: {gray_final_loss:.4f}\n")
            f.write(f"- **EfficientNet-B0**: {efficient_final_loss:.4f}\n\n")
        
        # 优缺点分析
        f.write("## 优缺点分析\n\n")
        
        f.write("### GrayColonyNet\n")
        f.write("**优点**:\n")
        f.write("- CNN-Transformer混合架构设计新颖\n")
        f.write("- 能够同时捕获局部和全局特征\n")
        f.write("- 层次化任务设计合理\n\n")
        
        f.write("**缺点**:\n")
        f.write("- 架构复杂度较高\n")
        f.write("- 训练时间可能较长\n")
        f.write("- 参数数量较多\n\n")
        
        f.write("### EfficientNet-B0多任务版\n")
        f.write("**优点**:\n")
        f.write("- 基于成熟的EfficientNet架构\n")
        f.write("- 复合缩放策略在小图像上效果好\n")
        f.write("- 自适应任务权重学习\n")
        f.write("- 跨任务注意力机制\n\n")
        
        f.write("**缺点**:\n")
        f.write("- 依赖预训练权重的效果\n")
        f.write("- 任务头参数较多\n\n")
        
        # 结论和建议
        f.write("## 结论和建议\n\n")
        
        if efficient_overall > gray_overall:
            f.write("### 推荐模型: EfficientNet-B0多任务版\n\n")
            f.write("**理由**:\n")
            f.write(f"1. 整体准确率更高 ({efficient_overall:.2f}% vs {gray_overall:.2f}%)\n")
            f.write("2. 基于成熟的EfficientNet架构，稳定性好\n")
            f.write("3. 在70×70小图像上表现优异\n\n")
        else:
            f.write("### 推荐模型: GrayColonyNet\n\n")
            f.write("**理由**:\n")
            f.write(f"1. 整体准确率更高或相当 ({gray_overall:.2f}% vs {efficient_overall:.2f}%)\n")
            f.write("2. 创新的混合架构设计\n")
            f.write("3. 能够很好地处理多任务关系\n\n")
        
        f.write("### 进一步改进建议\n")
        f.write("1. **模型集成**: 将两个模型进行集成，利用各自优势\n")
        f.write("2. **架构优化**: 结合两个模型的优点设计新架构\n")
        f.write("3. **数据增强**: 进一步优化数据增强策略\n")
        f.write("4. **超参调优**: 使用更系统的超参数优化方法\n")
        f.write("5. **知识蒸馏**: 使用较好的模型指导较差模型的训练\n\n")
        
        f.write("---\n")
        f.write("*本报告基于实验数据自动生成*\n")
    
    return report_path


def main():
    """主函数"""
    print("=== 多任务模型综合对比分析 ===")
    
    # 1. 查找实验结果
    gray_experiment, efficient_experiment = find_experiment_results()
    
    if not gray_experiment:
        print("✗ 未找到GrayColonyNet实验结果")
        return 1
    
    if not efficient_experiment:
        print("✗ 未找到EfficientNet-B0实验结果") 
        return 1
    
    print(f"GrayColonyNet实验: {gray_experiment}")
    print(f"EfficientNet-B0实验: {efficient_experiment}")
    
    # 2. 加载实验数据
    gray_data = load_experiment_data(gray_experiment, 'GrayColonyNet')
    efficient_data = load_experiment_data(efficient_experiment, 'EfficientNet-B0')
    
    # 3. 创建输出目录
    output_dir = Path(project_root) / 'reports' / 'model_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 生成对比分析
    comparison_results = {}
    
    try:
        # 训练曲线对比
        if gray_data.get('training_history') and efficient_data.get('training_history'):
            curve_path = compare_training_curves(gray_data, efficient_data, output_dir)
            comparison_results['curves'] = curve_path
            print(f"✓ 训练曲线对比: {curve_path}")
        
        # 任务性能对比
        if gray_data.get('test_metrics') and efficient_data.get('test_metrics'):
            performance_path, performance_df = compare_task_performance(gray_data, efficient_data, output_dir)
            comparison_results['performance'] = performance_path
            comparison_results['performance_df'] = performance_df
            print(f"✓ 任务性能对比: {performance_path}")
        
        # 错误模式对比
        if gray_data.get('error_analysis') and efficient_data.get('error_analysis'):
            error_path, error_stats = compare_error_patterns(gray_data, efficient_data, output_dir)
            comparison_results['errors'] = error_path
            comparison_results['error_stats'] = error_stats
            print(f"✓ 错误模式对比: {error_path}")
        
        # 综合报告
        report_path = generate_comprehensive_report(gray_data, efficient_data, comparison_results, output_dir)
        print(f"✓ 综合报告: {report_path}")
        
        print("\n=== 对比分析完成 ===")
        print(f"报告目录: {output_dir}")
        print(f"主报告: {report_path}")
        
    except Exception as e:
        print(f"✗ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())