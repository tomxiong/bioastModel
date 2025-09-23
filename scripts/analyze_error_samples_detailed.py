#!/usr/bin/env python3
"""
详细错误样本分析脚本
分析每个错误样本在各个任务上的具体错误情况
"""

import json
import os
from collections import defaultdict

def analyze_detailed_errors():
    """分析详细错误样本"""
    
    # 读取详细错误数据
    error_file = "/home/aaa/ws/bioastModel/experiments/ni_multitask_efficientnet_b0_20250905_004007/detailed_errors_20250905_004514.json"
    
    with open(error_file, 'r', encoding='utf-8') as f:
        error_samples = json.load(f)
    
    print("# EfficientNet-B0 多任务模型错误样本详细分析\n")
    print(f"## 总体概况")
    print(f"- **总错误样本数**: {len(error_samples)}")
    print(f"- **测试样本总数**: 186")
    print(f"- **错误率**: {len(error_samples)/186*100:.2f}%")
    print(f"- **整体准确率**: {(186-len(error_samples))/186*100:.2f}%\n")
    
    # 按任务分类错误
    task_errors = {
        'growth_level': [],
        'growth_pattern': [], 
        'fine_grained': [],
        'interference_factors': []
    }
    
    # 按全景图分组
    panoramic_errors = defaultdict(list)
    
    # 按错误任务组合分组
    error_combinations = defaultdict(list)
    
    for sample in error_samples:
        sample_id = sample['image_id']
        panoramic_id = sample['panoramic_id']
        error_tasks = sample['error_tasks']
        
        # 按任务分类
        for task in error_tasks:
            task_errors[task].append({
                'sample_id': sample_id,
                'panoramic_id': panoramic_id,
                'pred': sample['predictions'][task],
                'target': sample['targets'][task]
            })
        
        # 按全景图分组
        panoramic_errors[panoramic_id].append({
            'sample_id': sample_id,
            'error_tasks': error_tasks,
            'predictions': sample['predictions'],
            'targets': sample['targets']
        })
        
        # 按错误组合分组
        combo = '+'.join(sorted(error_tasks))
        error_combinations[combo].append(sample_id)
    
    print("## 1. 按任务类型错误详情\n")
    
    for task, errors in task_errors.items():
        if not errors:
            continue
            
        print(f"### {task.upper()} 错误详情")
        print(f"- **错误数量**: {len(errors)}")
        
        if task == 'interference_factors':
            print("- **错误类型**: 多标签分类错误")
            print("- **常见模式**: ")
            for i, error in enumerate(errors[:10]):  # 只显示前10个
                pred_labels = [i for i, v in enumerate(error['pred']) if v]
                target_labels = [i for i, v in enumerate(error['target']) if v]
                print(f"  - `{error['sample_id']}`: 预测{pred_labels} → 真实{target_labels}")
        else:
            # 统计错误模式
            error_patterns = defaultdict(int)
            for error in errors:
                pattern = f"{error['target']} → {error['pred']}"
                error_patterns[pattern] += 1
            
            print("- **主要错误模式**: ")
            for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - `{pattern}`: {count}次")
        
        print(f"- **涉及样本**: ")
        for i, error in enumerate(errors[:10]):  # 只显示前10个样本
            print(f"  - `{error['sample_id']}` (来自 {error['panoramic_id']})")
        
        if len(errors) > 10:
            print(f"  - ...还有{len(errors)-10}个样本")
        print()
    
    print("## 2. 按全景图错误分布\n")
    
    # 按错误数量排序
    sorted_panoramics = sorted(panoramic_errors.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("### 高错误率全景图 (Top 15)")
    for i, (panoramic_id, samples) in enumerate(sorted_panoramics[:15]):
        print(f"{i+1:2d}. **{panoramic_id}**: {len(samples)}个错误样本")
        
        # 统计该全景图的任务错误分布
        task_count = defaultdict(int)
        for sample in samples:
            for task in sample['error_tasks']:
                task_count[task] += 1
        
        task_summary = []
        for task, count in task_count.items():
            task_summary.append(f"{task}({count})")
        
        print(f"     - 任务分布: {', '.join(task_summary)}")
        
        # 显示前5个错误样本的详情
        print(f"     - 样本详情:")
        for j, sample in enumerate(samples[:5]):
            error_tasks_str = ', '.join(sample['error_tasks'])
            print(f"       * `{sample['sample_id']}`: {error_tasks_str}")
        
        if len(samples) > 5:
            print(f"       * ...还有{len(samples)-5}个样本")
        print()
    
    print("## 3. 错误任务组合分析\n")
    
    print("### 多任务同时错误情况")
    sorted_combinations = sorted(error_combinations.items(), key=lambda x: len(x[1]), reverse=True)
    
    for combo, samples in sorted_combinations:
        print(f"- **{combo}**: {len(samples)}个样本")
        
        # 显示前10个样本
        sample_list = []
        for i, sample_id in enumerate(samples[:10]):
            sample_list.append(f"`{sample_id}`")
        
        print(f"  - 样本: {', '.join(sample_list)}")
        if len(samples) > 10:
            print(f"  - ...还有{len(samples)-10}个样本")
        print()
    
    print("## 4. 关键发现总结\n")
    
    # Fine Grained 错误分析
    fine_grained_errors = task_errors['fine_grained']
    print("### Fine Grained 分类错误重点分析")
    print(f"- **错误数量**: {len(fine_grained_errors)} (占总错误的 {len(fine_grained_errors)/len(error_samples)*100:.1f}%)")
    
    # 统计最常见的错误模式
    fg_patterns = defaultdict(int)
    for error in fine_grained_errors:
        pattern = f"{error['target']} → {error['pred']}"
        fg_patterns[pattern] += 1
    
    print("- **Top 5 错误模式**:")
    for pattern, count in sorted(fg_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - `{pattern}`: {count}次 ({count/len(fine_grained_errors)*100:.1f}%)")
    
    # 分析类别偏向
    pred_dist = defaultdict(int)
    target_dist = defaultdict(int)
    for error in fine_grained_errors:
        pred_dist[error['pred']] += 1
        target_dist[error['target']] += 1
    
    print("- **预测偏向分析**:")
    for pred_class, count in sorted(pred_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - 类别 {pred_class}: {count}次被错误预测")
    
    print("- **真实类别分析**:")
    for target_class, count in sorted(target_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - 类别 {target_class}: {count}次被误判")
    
    print("\n### 问题全景图分析")
    print("- **SE系列** (SE10000052, SE10000078): 生物样本存在特殊干扰")
    print("- **NF系列** (NF10000033, NF10000034): 可能存在标注质量问题")
    print("- **建议**: 重点检查这些全景图的图像质量和标注准确性")

if __name__ == "__main__":
    analyze_detailed_errors()