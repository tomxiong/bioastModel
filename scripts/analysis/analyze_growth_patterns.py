#!/usr/bin/env python3
"""
分析数据集中growth_pattern的分类数量和分布
"""

import json
import sys
from collections import Counter

def analyze_growth_patterns(json_path):
    """分析growth_pattern的分类"""
    print(f"正在分析文件: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"数据集名称: {data.get('name', 'Unknown')}")
        print(f"总标注数量: {data.get('total_annotations', 0)}")
        print(f"保存标注数量: {data.get('saved_annotations', 0)}")
        print()
        
        # 统计growth_pattern
        growth_patterns = []
        growth_levels = []
        interference_factors_list = []
        
        annotations = data.get('annotations', [])
        print(f"实际处理的标注数量: {len(annotations)}")
        
        for annotation in annotations:
            features = annotation.get('features', {})
            
            # 收集growth_pattern
            growth_pattern = features.get('growth_pattern')
            if growth_pattern:
                growth_patterns.append(growth_pattern)
            
            # 收集growth_level
            growth_level = features.get('growth_level')
            if growth_level:
                growth_levels.append(growth_level)
            
            # 收集interference_factors
            interference_factors = features.get('interference_factors', [])
            for factor in interference_factors:
                interference_factors_list.append(factor)
        
        # 统计growth_pattern分布
        pattern_counter = Counter(growth_patterns)
        print("=== Growth Pattern 分析 ===")
        print(f"总的growth_pattern类别数量: {len(pattern_counter)}")
        print("各类别分布:")
        for pattern, count in pattern_counter.most_common():
            percentage = (count / len(growth_patterns)) * 100
            print(f"  {pattern}: {count} ({percentage:.2f}%)")
        
        print()
        
        # 统计growth_level分布
        level_counter = Counter(growth_levels)
        print("=== Growth Level 分析 ===")
        print(f"总的growth_level类别数量: {len(level_counter)}")
        print("各类别分布:")
        for level, count in level_counter.most_common():
            percentage = (count / len(growth_levels)) * 100
            print(f"  {level}: {count} ({percentage:.2f}%)")
        
        print()
        
        # 统计interference_factors分布
        if interference_factors_list:
            factor_counter = Counter(interference_factors_list)
            print("=== Interference Factors 分析 ===")
            print(f"总的interference_factors类别数量: {len(factor_counter)}")
            print("各类别分布:")
            for factor, count in factor_counter.most_common():
                percentage = (count / len(interference_factors_list)) * 100
                print(f"  {factor}: {count} ({percentage:.2f}%)")
        else:
            print("=== Interference Factors 分析 ===")
            print("没有发现interference_factors数据")
        
        print()
        
        # 显示一些样本数据
        print("=== 样本数据 ===")
        for i, annotation in enumerate(annotations[:5]):
            features = annotation.get('features', {})
            print(f"样本 {i+1}:")
            print(f"  image_id: {annotation.get('image_id')}")
            print(f"  growth_level: {features.get('growth_level')}")
            print(f"  growth_pattern: {features.get('growth_pattern')}")
            print(f"  interference_factors: {features.get('interference_factors', [])}")
            print()
        
        return {
            'growth_patterns': dict(pattern_counter),
            'growth_levels': dict(level_counter),
            'interference_factors': dict(factor_counter) if interference_factors_list else {}
        }
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    json_path = "/home/aaa/ws/bioastModel/ds/images/m9e1n170.json"
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    results = analyze_growth_patterns(json_path)
    
    if results:
        print("=== 分析完成 ===")
        print(f"Growth Pattern 类别数: {len(results['growth_patterns'])}")
        print(f"Growth Level 类别数: {len(results['growth_levels'])}")
        print(f"Interference Factors 类别数: {len(results['interference_factors'])}")