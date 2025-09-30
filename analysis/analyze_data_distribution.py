#!/usr/bin/env python3
"""
数据分布分析脚本
重点分析：
1. negative样本中pores的分布情况
2. center_dots vs litter_center_dots的差异
3. 各类别的样本不平衡情况
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import argparse

def load_data(json_path):
    """加载JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 提取annotations部分
    if isinstance(data, dict) and 'annotations' in data:
        return data['annotations']
    return data

def analyze_growth_level_distribution(data):
    """分析growth_level分布"""
    growth_levels = [item['features']['growth_level'] for item in data]
    counter = Counter(growth_levels)
    
    print("=== Growth Level 分布 ===")
    for level, count in counter.items():
        print(f"{level}: {count} ({count/len(data)*100:.2f}%)")
    
    return counter

def analyze_growth_pattern_distribution(data):
    """分析growth_pattern分布"""
    growth_patterns = [item['features']['growth_pattern'] for item in data]
    counter = Counter(growth_patterns)
    
    print("\n=== Growth Pattern 分布 ===")
    for pattern, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{pattern}: {count} ({count/len(data)*100:.2f}%)")
    
    return counter

def analyze_interference_factors_distribution(data):
    """分析interference_factors分布"""
    all_factors = []
    for item in data:
        factors = item['features'].get('interference_factors', [])
        all_factors.extend(factors)
    
    counter = Counter(all_factors)
    
    print("\n=== Interference Factors 分布 ===")
    for factor, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{factor}: {count} ({count/len(all_factors)*100:.2f}%)")
    
    return counter

def analyze_negative_pores_distribution(data):
    """重点分析negative样本中pores的分布"""
    negative_samples = [item for item in data if item['features']['growth_level'] == 'negative']
    negative_with_pores = [item for item in negative_samples 
                          if 'pores' in item['features'].get('interference_factors', [])]
    
    print(f"\n=== Negative样本中Pores分析 ===")
    print(f"总negative样本数: {len(negative_samples)}")
    print(f"含pores的negative样本数: {len(negative_with_pores)}")
    print(f"negative样本中pores比例: {len(negative_with_pores)/len(negative_samples)*100:.2f}%")
    
    # 分析negative+pores样本的growth_pattern分布
    negative_pores_patterns = [item['features']['growth_pattern'] for item in negative_with_pores]
    pattern_counter = Counter(negative_pores_patterns)
    
    print("\nNegative+Pores样本的Growth Pattern分布:")
    for pattern, count in sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} ({count/len(negative_with_pores)*100:.2f}%)")
    
    return negative_with_pores, pattern_counter

def analyze_center_dots_vs_litter_center_dots(data):
    """分析center_dots vs litter_center_dots的差异"""
    center_dots_samples = [item for item in data if item['features']['growth_pattern'] == 'center_dots']
    litter_center_dots_samples = [item for item in data if item['features']['growth_pattern'] == 'litter_center_dots']
    
    print(f"\n=== Center Dots vs Litter Center Dots 分析 ===")
    print(f"center_dots样本数: {len(center_dots_samples)}")
    print(f"litter_center_dots样本数: {len(litter_center_dots_samples)}")
    
    # 分析growth_level分布
    center_dots_levels = Counter([item['features']['growth_level'] for item in center_dots_samples])
    litter_center_dots_levels = Counter([item['features']['growth_level'] for item in litter_center_dots_samples])
    
    print("\nCenter Dots的Growth Level分布:")
    for level, count in center_dots_levels.items():
        print(f"  {level}: {count} ({count/len(center_dots_samples)*100:.2f}%)")
    
    print("\nLitter Center Dots的Growth Level分布:")
    for level, count in litter_center_dots_levels.items():
        print(f"  {level}: {count} ({count/len(litter_center_dots_samples)*100:.2f}%)")
    
    # 分析interference_factors差异
    center_dots_factors = []
    litter_center_dots_factors = []
    
    for item in center_dots_samples:
        center_dots_factors.extend(item['features'].get('interference_factors', []))
    
    for item in litter_center_dots_samples:
        litter_center_dots_factors.extend(item['features'].get('interference_factors', []))
    
    center_dots_factor_counter = Counter(center_dots_factors)
    litter_center_dots_factor_counter = Counter(litter_center_dots_factors)
    
    print("\nCenter Dots的Interference Factors分布:")
    for factor, count in sorted(center_dots_factor_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {factor}: {count}")
    
    print("\nLitter Center Dots的Interference Factors分布:")
    for factor, count in sorted(litter_center_dots_factor_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"  {factor}: {count}")
    
    return center_dots_samples, litter_center_dots_samples

def create_visualizations(data, output_dir):
    """创建可视化图表"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Growth Level分布饼图
    growth_levels = [item['features']['growth_level'] for item in data]
    level_counter = Counter(growth_levels)
    
    plt.figure(figsize=(8, 6))
    plt.pie(level_counter.values(), labels=level_counter.keys(), autopct='%1.1f%%')
    plt.title('Growth Level Distribution')
    plt.savefig(output_dir / 'growth_level_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Growth Pattern分布条形图
    growth_patterns = [item['features']['growth_pattern'] for item in data]
    pattern_counter = Counter(growth_patterns)
    
    plt.figure(figsize=(12, 8))
    patterns, counts = zip(*sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True))
    plt.bar(range(len(patterns)), counts)
    plt.xticks(range(len(patterns)), patterns, rotation=45, ha='right')
    plt.title('Growth Pattern Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(output_dir / 'growth_pattern_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Interference Factors分布
    all_factors = []
    for item in data:
        all_factors.extend(item['features'].get('interference_factors', []))
    
    factor_counter = Counter(all_factors)
    
    plt.figure(figsize=(10, 6))
    factors, counts = zip(*sorted(factor_counter.items(), key=lambda x: x[1], reverse=True))
    plt.bar(factors, counts)
    plt.title('Interference Factors Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'interference_factors_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 交叉分析热力图
    # Growth Level vs Growth Pattern
    cross_data = defaultdict(lambda: defaultdict(int))
    for item in data:
        cross_data[item['features']['growth_level']][item['features']['growth_pattern']] += 1
    
    # 转换为DataFrame
    df_cross = pd.DataFrame(cross_data).fillna(0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(df_cross, annot=True, fmt='g', cmap='Blues')
    plt.title('Growth Level vs Growth Pattern Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'growth_level_vs_pattern_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_optimization_recommendations(data):
    """生成优化建议"""
    print("\n" + "="*50)
    print("=== 优化建议 ===")
    
    # 分析negative+pores的情况
    negative_samples = [item for item in data if item['features']['growth_level'] == 'negative']
    negative_with_pores = [item for item in negative_samples 
                          if 'pores' in item['features'].get('interference_factors', [])]
    
    pores_ratio = len(negative_with_pores) / len(negative_samples) if len(negative_samples) > 0 else 0
    
    print(f"1. Negative+Pores权重调整建议:")
    print(f"   - 当前negative样本中pores比例: {pores_ratio:.3f}")
    if pores_ratio < 0.1:
        print(f"   - 建议对negative+pores样本增加3-5倍权重")
    elif pores_ratio < 0.2:
        print(f"   - 建议对negative+pores样本增加2-3倍权重")
    else:
        print(f"   - 建议对negative+pores样本增加1.5-2倍权重")
    
    # 分析center_dots vs litter_center_dots
    center_dots_samples = [item for item in data if item['features']['growth_pattern'] == 'center_dots']
    litter_center_dots_samples = [item for item in data if item['features']['growth_pattern'] == 'litter_center_dots']
    
    center_positive = len([item for item in center_dots_samples if item['features']['growth_level'] == 'positive'])
    litter_negative = len([item for item in litter_center_dots_samples if item['features']['growth_level'] == 'negative'])
    
    print(f"\n2. Center Dots区分优化建议:")
    print(f"   - center_dots中positive样本: {center_positive}/{len(center_dots_samples)}")
    print(f"   - litter_center_dots中negative样本: {litter_negative}/{len(litter_center_dots_samples)}")
    print(f"   - 建议增加对比学习损失，强化positive center_dots与negative litter_center_dots的区分")
    print(f"   - 建议在特征提取层添加注意力机制，关注点状特征的空间分布")
    
    # 类别不平衡分析
    growth_patterns = [item['features']['growth_pattern'] for item in data]
    pattern_counter = Counter(growth_patterns)
    max_count = max(pattern_counter.values())
    min_count = min(pattern_counter.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n3. 类别平衡优化建议:")
    print(f"   - 最大类别样本数: {max_count}, 最小类别样本数: {min_count}")
    print(f"   - 不平衡比例: {imbalance_ratio:.2f}")
    if imbalance_ratio > 10:
        print(f"   - 建议使用Focal Loss或Class-Balanced Loss")
        print(f"   - 建议对少数类样本进行数据增强")

def main():
    parser = argparse.ArgumentParser(description='数据分布分析')
    parser.add_argument('--json_path', type=str, 
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='JSON数据文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/aaa/ws/bioastModel/data_analysis_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("开始数据分布分析...")
    
    # 加载数据
    data = load_data(args.json_path)
    print(f"总样本数: {len(data)}")
    
    # 基础分布分析
    analyze_growth_level_distribution(data)
    analyze_growth_pattern_distribution(data)
    analyze_interference_factors_distribution(data)
    
    # 重点分析
    analyze_negative_pores_distribution(data)
    analyze_center_dots_vs_litter_center_dots(data)
    
    # 创建可视化
    create_visualizations(data, args.output_dir)
    print(f"\n可视化图表已保存到: {args.output_dir}")
    
    # 生成优化建议
    generate_optimization_recommendations(data)

if __name__ == "__main__":
    main()