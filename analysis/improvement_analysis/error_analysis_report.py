#!/usr/bin/env python3
"""
错误样本分析脚本
分析Growth Pattern和Pores检测的误分类情况
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_growth_pattern_confusion_matrix():
    """分析Growth Pattern的混淆矩阵"""
    
    # Growth Pattern混淆矩阵 (从test_results.json提取)
    confusion_matrix = np.array([
        [57, 5, 27, 0, 0, 0, 0, 1, 0, 0, 5, 1],      # center_dots
        [2, 634, 0, 0, 0, 0, 0, 36, 0, 1, 147, 0],   # clean
        [4, 0, 762, 0, 0, 12, 0, 0, 0, 9, 2, 1],     # clustered
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # default_positive
        [8, 1, 235, 0, 0, 0, 0, 0, 0, 3, 1, 1],      # focal
        [1, 0, 14, 0, 0, 209, 0, 0, 0, 13, 0, 6],    # heavy_growth
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 1],        # irregular
        [18, 16, 0, 0, 0, 0, 0, 79, 0, 0, 27, 0],    # litter_center_dots
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 3],        # scattered
        [0, 0, 17, 0, 0, 4, 0, 0, 0, 78, 2, 7],      # strong_scattered
        [3, 20, 1, 0, 0, 0, 0, 13, 0, 0, 465, 1],    # weak_scattered
        [0, 0, 2, 0, 0, 2, 0, 0, 0, 3, 14, 16]       # weak_scattered_pos
    ])
    
    # 类别标签
    labels = [
        'center_dots', 'clean', 'clustered', 'default_positive', 'focal',
        'heavy_growth', 'irregular', 'litter_center_dots', 'scattered',
        'strong_scattered', 'weak_scattered', 'weak_scattered_pos'
    ]
    
    # 计算每个类别的性能指标
    class_performance = {}
    total_samples = confusion_matrix.sum()
    
    for i, label in enumerate(labels):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = total_samples - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_performance[label] = {
            'samples': confusion_matrix[i, :].sum(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    return confusion_matrix, labels, class_performance

def analyze_major_misclassifications():
    """分析主要的误分类情况"""
    
    confusion_matrix, labels, class_performance = analyze_growth_pattern_confusion_matrix()
    
    print("=== Growth Pattern 错误分析 ===\n")
    
    # 找出性能最差的类别
    worst_performers = sorted(class_performance.items(), key=lambda x: x[1]['f1_score'])[:5]
    
    print("性能最差的5个类别:")
    for label, metrics in worst_performers:
        print(f"{label:20} - F1: {metrics['f1_score']:.3f}, 样本数: {metrics['samples']:4d}, "
              f"精确率: {metrics['precision']:.3f}, 召回率: {metrics['recall']:.3f}")
    
    print("\n主要误分类模式:")
    
    # 分析混淆矩阵中的主要错误
    for i, true_label in enumerate(labels):
        row = confusion_matrix[i, :]
        total_samples = row.sum()
        if total_samples == 0:
            continue
            
        # 找出被误分类最多的情况
        misclassified = [(j, count) for j, count in enumerate(row) if j != i and count > 0]
        misclassified.sort(key=lambda x: x[1], reverse=True)
        
        if misclassified and total_samples > 10:  # 只分析样本数较多的类别
            print(f"\n{true_label} (总样本: {total_samples}):")
            correct = confusion_matrix[i, i]
            print(f"  正确分类: {correct} ({correct/total_samples*100:.1f}%)")
            
            for j, count in misclassified[:3]:  # 显示前3个误分类
                if count > total_samples * 0.05:  # 只显示误分类率>5%的情况
                    print(f"  误分为 {labels[j]}: {count} ({count/total_samples*100:.1f}%)")

def analyze_pores_detection():
    """分析Pores检测问题"""
    
    print("\n=== Pores 检测分析 ===\n")
    
    # Pores检测准确率为86.20%，相对较低
    pores_accuracy = 0.862
    
    print(f"当前Pores检测准确率: {pores_accuracy:.1%}")
    print("可能的问题:")
    print("1. Pores与其他干扰因子(artifacts, debris)在视觉上相似")
    print("2. Pores的标注可能存在主观性，边界模糊")
    print("3. 数据集中Pores样本的多样性不足")
    print("4. 模型对细微纹理特征的学习能力有限")
    
    # 与其他干扰因子对比
    interference_accuracies = {
        'artifacts': 0.9253,
        'contamination': 0.9983,
        'debris': 0.953,
        'pores': 0.862
    }
    
    print(f"\n干扰因子检测性能对比:")
    for factor, acc in sorted(interference_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {factor:12}: {acc:.1%}")

def generate_improvement_suggestions():
    """生成改进建议"""
    
    print("\n=== 改进建议 ===\n")
    
    print("Growth Pattern 改进策略:")
    print("1. 数据增强:")
    print("   - 针对样本数少的类别(default_positive, irregular, scattered)进行过采样")
    print("   - 使用更多样化的数据增强技术(旋转、缩放、颜色变换)")
    print("   - 生成合成样本来平衡类别分布")
    
    print("\n2. 模型架构优化:")
    print("   - 增加注意力机制来关注关键区域")
    print("   - 使用多尺度特征融合")
    print("   - 考虑使用更大的模型(MobileNetV3-large)")
    
    print("\n3. 训练策略:")
    print("   - 使用类别权重来处理不平衡数据")
    print("   - 实施渐进式训练(先训练简单类别，再训练困难类别)")
    print("   - 增加训练轮数，使用更小的学习率")
    
    print("\nPores 检测改进策略:")
    print("1. 数据质量:")
    print("   - 重新审查Pores标注的一致性")
    print("   - 增加更多高质量的Pores样本")
    print("   - 建立更清晰的Pores定义标准")
    
    print("\n2. 特征工程:")
    print("   - 使用纹理特征提取(LBP, GLCM)")
    print("   - 添加边缘检测预处理")
    print("   - 考虑使用多模态输入(原图+边缘图)")
    
    print("\n3. 模型改进:")
    print("   - 使用专门的纹理分析网络")
    print("   - 实施困难样本挖掘")
    print("   - 考虑使用集成学习方法")

def create_visualization():
    """创建可视化图表"""
    
    confusion_matrix, labels, class_performance = analyze_growth_pattern_confusion_matrix()
    
    # 创建混淆矩阵热图
    plt.figure(figsize=(12, 10))
    
    # 归一化混淆矩阵
    normalized_cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(normalized_cm, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                cbar_kws={'label': 'Normalized Frequency'})
    
    plt.title('Growth Pattern Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'growth_pattern_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建类别性能对比图
    plt.figure(figsize=(14, 8))
    
    categories = list(class_performance.keys())
    f1_scores = [class_performance[cat]['f1_score'] for cat in categories]
    sample_counts = [class_performance[cat]['samples'] for cat in categories]
    
    # 创建双轴图
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    color = 'tab:blue'
    ax1.set_xlabel('Growth Pattern Categories')
    ax1.set_ylabel('F1 Score', color=color)
    bars1 = ax1.bar(categories, f1_scores, color=color, alpha=0.7, label='F1 Score')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sample Count', color=color)
    line = ax2.plot(categories, sample_counts, color=color, marker='o', linewidth=2, label='Sample Count')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Growth Pattern Performance vs Sample Count')
    plt.xticks(rotation=45, ha='right')
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'growth_pattern_performance_vs_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化图表已保存到: {output_dir}")

def main():
    """主函数"""
    print("开始错误样本分析...\n")
    
    # 分析Growth Pattern混淆矩阵
    analyze_major_misclassifications()
    
    # 分析Pores检测问题
    analyze_pores_detection()
    
    # 生成改进建议
    generate_improvement_suggestions()
    
    # 创建可视化
    create_visualization()
    
    print("\n错误分析完成!")

if __name__ == "__main__":
    main()