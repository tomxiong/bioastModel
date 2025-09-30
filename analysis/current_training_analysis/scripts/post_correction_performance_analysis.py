#!/usr/bin/env python3
"""
Post-Correction Performance Analysis Script
标签修正后性能分析脚本

分析标签修正后的模型性能，并与之前的结果进行对比
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_test_results(results_path):
    """加载测试结果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_training_history(history_path):
    """加载训练历史"""
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_performance_improvement():
    """分析性能改进情况"""
    
    # 当前实验结果路径
    current_results_path = "/home/aaa/ws/bioastModel/experiments/multilevel_mobilenetv3_ds/test_results.json"
    current_history_path = "/home/aaa/ws/bioastModel/experiments/multilevel_mobilenetv3_ds/training_history.json"
    
    # 加载当前结果
    current_results = load_test_results(current_results_path)
    current_history = load_training_history(current_history_path)
    
    # 之前的性能基准（从之前的分析报告中获取）
    previous_performance = {
        'growth_level': 0.9807,  # 98.07%
        'growth_pattern': 0.7667,  # 76.67%
        'interference_factors': 0.8620  # 86.20%
    }
    
    # 当前性能
    current_performance = {
        'growth_level': current_results['growth_level']['accuracy'],
        'growth_pattern': current_results['growth_pattern']['accuracy'],
        'interference_factors': current_results['interference_factors']['overall_accuracy']
    }
    
    # 计算改进幅度
    improvements = {}
    for task in previous_performance:
        prev_acc = previous_performance[task]
        curr_acc = current_performance[task]
        improvement = curr_acc - prev_acc
        improvement_pct = (improvement / prev_acc) * 100
        improvements[task] = {
            'previous': prev_acc,
            'current': curr_acc,
            'absolute_improvement': improvement,
            'relative_improvement_pct': improvement_pct
        }
    
    return improvements, current_results, current_history

def create_performance_comparison_plot(improvements):
    """创建性能对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 准确率对比柱状图
    tasks = list(improvements.keys())
    previous_accs = [improvements[task]['previous'] for task in tasks]
    current_accs = [improvements[task]['current'] for task in tasks]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    ax1.bar(x - width/2, previous_accs, width, label='修正前', alpha=0.8, color='lightcoral')
    ax1.bar(x + width/2, current_accs, width, label='修正后', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('任务类型')
    ax1.set_ylabel('准确率')
    ax1.set_title('标签修正前后准确率对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Growth Level', 'Growth Pattern', 'Interference Factors'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (prev, curr) in enumerate(zip(previous_accs, current_accs)):
        ax1.text(i - width/2, prev + 0.01, f'{prev:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, curr + 0.01, f'{curr:.3f}', ha='center', va='bottom')
    
    # 2. 改进幅度图
    improvements_abs = [improvements[task]['absolute_improvement'] for task in tasks]
    colors = ['green' if imp > 0 else 'red' for imp in improvements_abs]
    
    bars = ax2.bar(tasks, improvements_abs, color=colors, alpha=0.7)
    ax2.set_xlabel('任务类型')
    ax2.set_ylabel('准确率改进 (绝对值)')
    ax2.set_title('各任务准确率改进幅度')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements_abs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.005),
                f'{imp:+.4f}', ha='center', va='bottom' if height > 0 else 'top')
    
    # 3. 相对改进百分比
    improvements_pct = [improvements[task]['relative_improvement_pct'] for task in tasks]
    colors_pct = ['green' if imp > 0 else 'red' for imp in improvements_pct]
    
    bars_pct = ax3.bar(tasks, improvements_pct, color=colors_pct, alpha=0.7)
    ax3.set_xlabel('任务类型')
    ax3.set_ylabel('相对改进 (%)')
    ax3.set_title('各任务相对改进百分比')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, imp in zip(bars_pct, improvements_pct):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                f'{imp:+.2f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 4. 性能热力图
    performance_data = np.array([previous_accs, current_accs])
    im = ax4.imshow(performance_data, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)
    
    ax4.set_xticks(range(len(tasks)))
    ax4.set_xticklabels(['Growth Level', 'Growth Pattern', 'Interference Factors'])
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['修正前', '修正后'])
    ax4.set_title('性能热力图')
    
    # 添加数值标签
    for i in range(len(tasks)):
        for j in range(2):
            text = ax4.text(i, j, f'{performance_data[j, i]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    
    return fig

def generate_detailed_analysis_report(improvements, current_results, current_history):
    """生成详细分析报告"""
    
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "experiment_name": "multilevel_mobilenetv3_ds",
        "label_correction_impact": {
            "summary": "标签修正对模型性能的影响分析",
            "corrections_made": {
                "default_positive_to_clustered": 16,
                "focal_to_clustered": 1572,
                "total_corrections": 1588
            }
        },
        "performance_comparison": improvements,
        "current_detailed_results": current_results,
        "training_summary": {
            "total_epochs": len(current_history['train_loss']),
            "best_epoch": current_history['val_accuracy']['growth_level'].index(max(current_history['val_accuracy']['growth_level'])) + 1,
            "final_train_loss": current_history['train_loss'][-1],
            "final_val_loss": current_history['val_loss'][-1],
            "early_stopping": True if len(current_history['train_loss']) < 50 else False
        },
        "key_findings": [],
        "recommendations": []
    }
    
    # 分析关键发现
    for task, data in improvements.items():
        if data['absolute_improvement'] > 0.01:  # 改进超过1%
            report['key_findings'].append(f"{task}任务显著改进: {data['relative_improvement_pct']:.2f}%")
        elif data['absolute_improvement'] < -0.01:  # 下降超过1%
            report['key_findings'].append(f"{task}任务性能下降: {data['relative_improvement_pct']:.2f}%")
        else:
            report['key_findings'].append(f"{task}任务性能基本稳定")
    
    # 生成建议
    growth_pattern_improvement = improvements['growth_pattern']['relative_improvement_pct']
    if growth_pattern_improvement > 5:
        report['recommendations'].append("Growth Pattern分类显著改进，建议继续优化数据质量")
    elif growth_pattern_improvement < -2:
        report['recommendations'].append("Growth Pattern分类性能下降，需要检查标签修正的合理性")
    
    # 检查是否有过拟合
    if report['training_summary']['early_stopping']:
        report['recommendations'].append("模型提前停止训练，说明泛化能力良好")
    
    return report

def save_analysis_results(improvements, report, output_dir):
    """保存分析结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存性能对比图
    fig = create_performance_comparison_plot(improvements)
    fig.savefig(output_dir / 'post_correction_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # 保存详细报告
    with open(output_dir / 'post_correction_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 生成Markdown报告
    md_content = generate_markdown_report(report)
    with open(output_dir / 'post_correction_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"分析结果已保存到: {output_dir}")

def generate_markdown_report(report):
    """生成Markdown格式报告"""
    
    md_content = f"""# 标签修正后性能分析报告

## 分析概要
- **分析时间**: {report['analysis_timestamp']}
- **实验名称**: {report['experiment_name']}
- **标签修正数量**: {report['label_correction_impact']['corrections_made']['total_corrections']}条

## 标签修正详情
- `default_positive` → `clustered`: {report['label_correction_impact']['corrections_made']['default_positive_to_clustered']}条
- `focal` → `clustered`: {report['label_correction_impact']['corrections_made']['focal_to_clustered']}条

## 性能对比分析

### 各任务性能变化

| 任务类型 | 修正前准确率 | 修正后准确率 | 绝对改进 | 相对改进(%) |
|---------|-------------|-------------|----------|------------|
"""
    
    for task, data in report['performance_comparison'].items():
        md_content += f"| {task} | {data['previous']:.4f} | {data['current']:.4f} | {data['absolute_improvement']:+.4f} | {data['relative_improvement_pct']:+.2f}% |\n"
    
    md_content += f"""
## 训练过程分析
- **总训练轮数**: {report['training_summary']['total_epochs']}
- **最佳轮数**: {report['training_summary']['best_epoch']}
- **最终训练损失**: {report['training_summary']['final_train_loss']:.4f}
- **最终验证损失**: {report['training_summary']['final_val_loss']:.4f}
- **提前停止**: {'是' if report['training_summary']['early_stopping'] else '否'}

## 关键发现
"""
    
    for finding in report['key_findings']:
        md_content += f"- {finding}\n"
    
    md_content += "\n## 建议和下一步行动\n"
    
    for recommendation in report['recommendations']:
        md_content += f"- {recommendation}\n"
    
    md_content += f"""
## 详细测试结果

### Growth Level 分类
- **准确率**: {report['current_detailed_results']['growth_level']['accuracy']:.4f}
- **精确率**: {report['current_detailed_results']['growth_level']['precision']:.4f}
- **召回率**: {report['current_detailed_results']['growth_level']['recall']:.4f}
- **F1分数**: {report['current_detailed_results']['growth_level']['f1_score']:.4f}

### Growth Pattern 分类
- **准确率**: {report['current_detailed_results']['growth_pattern']['accuracy']:.4f}
- **精确率**: {report['current_detailed_results']['growth_pattern']['precision']:.4f}
- **召回率**: {report['current_detailed_results']['growth_pattern']['recall']:.4f}
- **F1分数**: {report['current_detailed_results']['growth_pattern']['f1_score']:.4f}

### Interference Factors 分类
- **整体准确率**: {report['current_detailed_results']['interference_factors']['overall_accuracy']:.4f}
"""
    
    for factor, acc in report['current_detailed_results']['interference_factors'].items():
        if factor != 'overall_accuracy':
            md_content += f"- **{factor}准确率**: {acc['accuracy']:.4f}\n"
    
    return md_content

def main():
    """主函数"""
    print("开始标签修正后性能分析...")
    
    # 分析性能改进
    improvements, current_results, current_history = analyze_performance_improvement()
    
    # 生成详细分析报告
    report = generate_detailed_analysis_report(improvements, current_results, current_history)
    
    # 保存分析结果
    output_dir = "/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports"
    save_analysis_results(improvements, report, output_dir)
    
    # 打印关键结果
    print("\n=== 标签修正后性能分析结果 ===")
    for task, data in improvements.items():
        print(f"{task}:")
        print(f"  修正前: {data['previous']:.4f}")
        print(f"  修正后: {data['current']:.4f}")
        print(f"  改进: {data['absolute_improvement']:+.4f} ({data['relative_improvement_pct']:+.2f}%)")
        print()
    
    print("分析完成！")

if __name__ == "__main__":
    main()