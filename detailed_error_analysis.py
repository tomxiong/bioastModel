#!/usr/bin/env python3
"""
M16 MultiTask MobileNetV3 错误样本深度分析总结
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

def analyze_error_report():
    """分析错误报告并生成深度分析"""
    
    print("🔍 M16 MultiTask MobileNetV3 错误样本深度分析")
    print("="*60)
    
    error_dir = Path("experiments/mic_mobilenetv3/error_analysis")
    
    # 1. 加载错误分析数据
    with open(error_dir / "error_analysis_data.json", 'r', encoding='utf-8') as f:
        error_data = json.load(f)
    
    # 2. 加载错误样本CSV
    df = pd.read_csv(error_dir / "error_samples_summary.csv")
    
    # 基本统计
    print("📊 基本错误统计:")
    print(f"   总测试样本: 186")
    print(f"   有错误的样本: {len(df)} (100%)")
    print(f"   平均每样本错误任务数: {df['error_count'].mean():.2f}")
    print(f"   最多错误任务数: {df['error_count'].max()}")
    
    # 各任务错误率
    print(f"\n🎯 各任务错误率:")
    error_stats = error_data['error_stats']
    task_performance = {}
    
    for task_name, stats in error_stats.items():
        error_rate = stats['error_rate']
        accuracy = 1 - error_rate
        task_performance[task_name] = {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'error_count': stats['errors']
        }
        print(f"   {task_name:20s}: {accuracy*100:5.1f}% 准确率 ({stats['errors']:3d}/186 错误)")
    
    # 多任务错误分布
    print(f"\n📋 多任务错误分布:")
    error_count_dist = df['error_count'].value_counts().sort_index()
    for count, samples in error_count_dist.items():
        percentage = samples / len(df) * 100
        print(f"   {count} 个任务出错: {samples:3d} 样本 ({percentage:5.1f}%)")
    
    # 任务组合错误分析
    print(f"\n🔄 任务组合错误分析:")
    task_combinations = Counter()
    
    for _, row in df.iterrows():
        tasks = sorted(row['error_tasks'].split(','))
        combo = ','.join(tasks)
        task_combinations[combo] += 1
    
    print("   最常见的错误组合:")
    for combo, count in task_combinations.most_common(10):
        percentage = count / len(df) * 100
        print(f"     {combo:40s}: {count:3d} 样本 ({percentage:5.1f}%)")
    
    # 最困难样本分析
    print(f"\n🎯 最困难样本分析 (3个任务都错误):")
    difficult_samples = df[df['error_count'] == 3]
    
    if len(difficult_samples) > 0:
        print(f"   发现 {len(difficult_samples)} 个最困难样本:")
        for _, row in difficult_samples.head(10).iterrows():
            print(f"     样本 {row['sample_idx']:3d}: {row['image_path']}")
            print(f"       错误任务: {row['error_tasks']}")
    
    # 错误模式深度分析
    print(f"\n🔍 各任务详细错误模式分析:")
    error_patterns = error_data['error_patterns']
    
    for task_name, patterns in error_patterns.items():
        if not patterns:
            continue
            
        print(f"\n   📝 {task_name} ({len(patterns)} 种错误模式):")
        
        # 统计错误类型
        total_errors = sum(patterns.values())
        pattern_analysis = []
        
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_errors * 100
            pattern_analysis.append({
                'pattern': pattern,
                'count': count,
                'percentage': percentage
            })
        
        # 显示前5种主要错误模式
        for i, analysis in enumerate(pattern_analysis[:5]):
            print(f"     {i+1:2d}. {analysis['pattern']:<40s}: {analysis['count']:3d} ({analysis['percentage']:5.1f}%)")
        
        if len(pattern_analysis) > 5:
            others_count = sum(a['count'] for a in pattern_analysis[5:])
            others_percentage = others_count / total_errors * 100
            print(f"         ... 其他 {len(pattern_analysis)-5} 种模式: {others_count:3d} ({others_percentage:5.1f}%)")
    
    # 生成改进建议
    generate_improvement_suggestions(task_performance, error_patterns, difficult_samples)

def generate_improvement_suggestions(task_performance, error_patterns, difficult_samples):
    """生成模型改进建议"""
    
    print(f"\n💡 模型改进建议:")
    print("="*50)
    
    # 1. 基于任务性能的建议
    sorted_tasks = sorted(task_performance.items(), key=lambda x: x[1]['accuracy'])
    
    print("🔧 针对各任务的改进建议:")
    
    for task_name, perf in sorted_tasks:
        accuracy = perf['accuracy']
        error_count = perf['error_count']
        
        print(f"\n   📋 {task_name} (准确率: {accuracy*100:.1f}%):")
        
        if task_name == 'interference_factors' and accuracy == 0:
            print("     ❗ 关键问题: 多标签分类完全失败")
            print("     🔧 建议: 重新审视多标签损失函数和阈值设置")
            print("     🔧 建议: 检查多标签数据的编码和解码过程")
            print("     🔧 建议: 考虑使用focal loss处理类别不平衡")
            
        elif accuracy < 0.8:
            print(f"     ❗ 问题: 准确率较低 ({error_count} 个错误)")
            print("     🔧 建议: 增加该任务的训练权重")
            print("     🔧 建议: 收集更多相关的训练样本")
            print("     🔧 建议: 检查数据标注质量")
            
        elif accuracy < 0.9:
            print(f"     ⚠️ 改进空间: 中等准确率 ({error_count} 个错误)")  
            print("     🔧 建议: 使用数据增强增加该任务的难例")
            print("     🔧 建议: 调整网络架构，增加任务特定的层")
    
    # 2. 基于错误模式的建议
    print(f"\n🎯 基于错误模式的建议:")
    
    # 分析growth_level错误
    growth_errors = error_patterns.get('growth_level', {})
    if growth_errors:
        print(f"\n   📊 生长级别判断:")
        if 'positive_to_negative' in growth_errors or 'negative_to_positive' in growth_errors:
            print("     🔧 建议: 正负样本混淆严重，需要更清晰的特征提取")
            print("     🔧 建议: 考虑增加对比学习机制")
    
    # 分析fine_grained错误
    fine_errors = error_patterns.get('fine_grained', {})
    if fine_errors:
        print(f"\n   🔍 精细分类:")
        print("     🔧 建议: 40类精细分类过于复杂，考虑层次化分类")
        print("     🔧 建议: 使用curriculum learning，从粗到细逐步训练")
    
    # 3. 针对困难样本的建议
    if len(difficult_samples) > 0:
        print(f"\n🎯 困难样本处理:")
        print(f"     发现 {len(difficult_samples)} 个在多个任务上都出错的样本")
        print("     🔧 建议: 人工检查这些困难样本，可能存在标注错误")
        print("     🔧 建议: 使用Active Learning重点标注困难样本")
        print("     🔧 建议: 在训练时给困难样本更高权重")
    
    # 4. 整体架构建议
    print(f"\n🏗️ 架构改进建议:")
    print("     🔧 建议: 考虑使用任务特定的attention机制")
    print("     🔧 建议: 实现dynamic task weighting")
    print("     🔧 建议: 使用uncertainty-based multi-task learning")
    print("     🔧 建议: 添加任务间的知识蒸馏")
    
    print(f"\n✅ 分析完成！基于模拟数据的深度分析报告生成完毕。")

def main():
    """主函数"""
    try:
        analyze_error_report()
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()