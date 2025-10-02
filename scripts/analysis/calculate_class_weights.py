#!/usr/bin/env python3
"""
计算基于实际数据分布的类别权重
"""

import json
import numpy as np
from collections import Counter

def calculate_class_weights(json_path):
    """计算类别权重"""
    print(f"正在分析文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 收集growth_pattern
    growth_patterns = []
    annotations = data.get('annotations', [])
    
    for annotation in annotations:
        features = annotation.get('features', {})
        growth_pattern = features.get('growth_pattern')
        if growth_pattern:
            growth_patterns.append(growth_pattern)
    
    # 统计分布
    pattern_counter = Counter(growth_patterns)
    total_samples = len(growth_patterns)
    
    print("Growth Pattern 分布:")
    pattern_names = []
    pattern_counts = []
    
    for pattern, count in pattern_counter.most_common():
        percentage = (count / total_samples) * 100
        print(f"  {pattern}: {count} ({percentage:.2f}%)")
        pattern_names.append(pattern)
        pattern_counts.append(count)
    
    # 计算权重 (使用inverse frequency)
    pattern_counts = np.array(pattern_counts)
    weights = total_samples / (len(pattern_names) * pattern_counts)
    
    # 归一化权重到合理范围
    weights = weights / weights.min()
    
    print("\n计算的类别权重:")
    weight_dict = {}
    for i, (pattern, weight) in enumerate(zip(pattern_names, weights)):
        print(f"  {pattern}: {weight:.3f}")
        weight_dict[pattern] = weight
    
    # 按照标准顺序排列权重
    standard_order = [
        'clustered', 'clean', 'weak_scattered', 'heavy_growth', 
        'litter_center_dots', 'strong_scattered', 'center_dots', 
        'weak_scattered_pos', 'scattered', 'irregular'
    ]
    
    ordered_weights = []
    print("\n按标准顺序的权重:")
    for pattern in standard_order:
        if pattern in weight_dict:
            weight = weight_dict[pattern]
            ordered_weights.append(weight)
            print(f"  {pattern}: {weight:.3f}")
        else:
            print(f"  {pattern}: 未找到")
    
    # 生成PyTorch tensor代码
    print("\nPyTorch权重tensor代码:")
    weights_str = ", ".join([f"{w:.3f}" for w in ordered_weights])
    print(f"torch.tensor([{weights_str}], device=self.device)")
    
    return ordered_weights, weight_dict

if __name__ == "__main__":
    json_path = "/home/aaa/ws/bioastModel/ds/images/m9e1n170.json"
    weights, weight_dict = calculate_class_weights(json_path)