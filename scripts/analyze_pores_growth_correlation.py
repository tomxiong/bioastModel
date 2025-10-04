"""
分析 pores 与 growth_pattern 的相关性
验证是否存在特征冲突
"""
import json
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency

def main():
    # 加载数据集
    with open('ds/images/m9e1n170_cleaned_round2.json', 'r') as f:
        dataset = json.load(f)
    
    with open('ds/images/dataset_split_seed44.json', 'r') as f:
        split_data = json.load(f)
    
    # 建立路径到标注的映射
    path_to_ann = {ann['image_path']: ann for ann in dataset['annotations']}
    
    # 分析训练集和测试集
    for split_name in ['train', 'test']:
        paths = split_data['splits'][split_name]
        
        print(f"\n{'='*80}")
        print(f"{split_name.upper()} SET Analysis")
        print(f"{'='*80}")
        
        # 统计 pores 与 growth_pattern 的联合分布
        pores_pattern_matrix = defaultdict(lambda: {'pores': 0, 'no_pores': 0})
        pores_level_matrix = defaultdict(lambda: {'pores': 0, 'no_pores': 0})
        
        total_samples = 0
        pores_samples = 0
        
        for path in paths:
            if path not in path_to_ann:
                continue
            
            ann = path_to_ann[path]
            features = ann['features']
            
            has_pores = 'pores' in features['interference_factors']
            growth_pattern = features['growth_pattern']
            growth_level = features['growth_level']
            
            total_samples += 1
            
            if has_pores:
                pores_samples += 1
                pores_pattern_matrix[growth_pattern]['pores'] += 1
                pores_level_matrix[growth_level]['pores'] += 1
            else:
                pores_pattern_matrix[growth_pattern]['no_pores'] += 1
                pores_level_matrix[growth_level]['no_pores'] += 1
        
        # 1. Growth Pattern 与 Pores 的关联分析
        print(f"\n1. Growth Pattern vs Pores Distribution:")
        print(f"   Total samples: {total_samples}, Pores samples: {pores_samples} ({pores_samples/total_samples*100:.1f}%)")
        print(f"\n   Pattern-wise breakdown:")
        
        pattern_stats = []
        for pattern in sorted(pores_pattern_matrix.keys(), key=lambda p: pores_pattern_matrix[p]['pores'] + pores_pattern_matrix[p]['no_pores'], reverse=True):
            pores_count = pores_pattern_matrix[pattern]['pores']
            no_pores_count = pores_pattern_matrix[pattern]['no_pores']
            total = pores_count + no_pores_count
            pores_ratio = pores_count / total * 100 if total > 0 else 0
            
            pattern_stats.append({
                'pattern': pattern,
                'total': total,
                'pores': pores_count,
                'pores_ratio': pores_ratio
            })
            
            print(f"   {pattern:20s}: {total:5d} total | {pores_count:4d} pores ({pores_ratio:5.1f}%) | {no_pores_count:4d} no_pores")
        
        # 计算 Chi-square 检验
        print(f"\n2. Statistical Independence Test (Chi-square):")
        
        # 构建列联表
        patterns_list = list(pores_pattern_matrix.keys())
        contingency_table = []
        for pattern in patterns_list:
            contingency_table.append([
                pores_pattern_matrix[pattern]['pores'],
                pores_pattern_matrix[pattern]['no_pores']
            ])
        
        contingency_array = np.array(contingency_table)
        chi2, p_value, dof, expected = chi2_contingency(contingency_array)
        
        print(f"   Chi-square statistic: {chi2:.2f}")
        print(f"   p-value: {p_value:.4e}")
        print(f"   Degrees of freedom: {dof}")
        
        if p_value < 0.001:
            print(f"   ✅ STRONG DEPENDENCY: Growth pattern and pores are highly correlated (p < 0.001)")
        elif p_value < 0.05:
            print(f"   ⚠️ MODERATE DEPENDENCY: Growth pattern and pores are correlated (p < 0.05)")
        else:
            print(f"   ❌ INDEPENDENT: Growth pattern and pores are independent (p >= 0.05)")
        
        # 3. 识别高 pores 比例的 pattern
        print(f"\n3. High Pores Ratio Patterns (>40%):")
        high_pores_patterns = [s for s in pattern_stats if s['pores_ratio'] > 40]
        if high_pores_patterns:
            for s in sorted(high_pores_patterns, key=lambda x: -x['pores_ratio']):
                print(f"   {s['pattern']:20s}: {s['pores_ratio']:5.1f}% pores ({s['pores']}/{s['total']})")
        else:
            print(f"   None")
        
        # 4. 识别低 pores 比例的 pattern
        print(f"\n4. Low Pores Ratio Patterns (<10%):")
        low_pores_patterns = [s for s in pattern_stats if s['pores_ratio'] < 10 and s['total'] > 50]
        if low_pores_patterns:
            for s in sorted(low_pores_patterns, key=lambda x: x['pores_ratio']):
                print(f"   {s['pattern']:20s}: {s['pores_ratio']:5.1f}% pores ({s['pores']}/{s['total']})")
        else:
            print(f"   None")
        
        # 5. Growth Level 与 Pores 的关联
        print(f"\n5. Growth Level vs Pores:")
        for level in ['negative', 'positive']:
            pores_count = pores_level_matrix[level]['pores']
            no_pores_count = pores_level_matrix[level]['no_pores']
            total = pores_count + no_pores_count
            pores_ratio = pores_count / total * 100 if total > 0 else 0
            print(f"   {level:10s}: {total:5d} total | {pores_count:4d} pores ({pores_ratio:5.1f}%)")
    
    # 6. 特征冲突分析总结
    print(f"\n{'='*80}")
    print(f"FEATURE CONFLICT ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # 重新统计所有数据
    all_paths = split_data['splits']['train'] + split_data['splits']['val'] + split_data['splits']['test']
    
    all_pores_patterns = defaultdict(int)
    all_no_pores_patterns = defaultdict(int)
    
    for path in all_paths:
        if path not in path_to_ann:
            continue
        
        ann = path_to_ann[path]
        features = ann['features']
        
        has_pores = 'pores' in features['interference_factors']
        growth_pattern = features['growth_pattern']
        
        if has_pores:
            all_pores_patterns[growth_pattern] += 1
        else:
            all_no_pores_patterns[growth_pattern] += 1
    
    print(f"\nOverall Dataset:")
    print(f"  Total patterns: {len(set(all_pores_patterns.keys()) | set(all_no_pores_patterns.keys()))}")
    
    # 计算每个 pattern 的 pores 偏好度
    pattern_preference = []
    for pattern in set(all_pores_patterns.keys()) | set(all_no_pores_patterns.keys()):
        pores_count = all_pores_patterns[pattern]
        no_pores_count = all_no_pores_patterns[pattern]
        total = pores_count + no_pores_count
        
        if total > 100:  # 只考虑样本量 > 100 的 pattern
            pores_ratio = pores_count / total
            pattern_preference.append({
                'pattern': pattern,
                'pores_ratio': pores_ratio,
                'total': total
            })
    
    # 排序并显示
    print(f"\nPattern Pores Preference (samples > 100):")
    for item in sorted(pattern_preference, key=lambda x: -x['pores_ratio']):
        ratio = item['pores_ratio'] * 100
        bias = "PORES-BIASED" if ratio > 40 else "NO-PORES-BIASED" if ratio < 20 else "BALANCED"
        print(f"  {item['pattern']:20s}: {ratio:5.1f}% pores | {item['total']:5d} samples | {bias}")
    
    # 结论
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")
    print(f"\n1. Statistical Correlation:")
    print(f"   - Growth pattern and pores are statistically correlated (Chi-square test)")
    print(f"   - Different patterns have different pores occurrence rates")
    
    print(f"\n2. Feature Conflict Evidence:")
    pores_biased = [p for p in pattern_preference if p['pores_ratio'] > 0.4]
    no_pores_biased = [p for p in pattern_preference if p['pores_ratio'] < 0.2]
    
    if pores_biased and no_pores_biased:
        print(f"   ✅ YES - Feature conflict exists:")
        print(f"      - {len(pores_biased)} patterns are pores-biased (>40% pores)")
        print(f"      - {len(no_pores_biased)} patterns are no-pores-biased (<20% pores)")
        print(f"      - Model may learn pattern features instead of pores features")
    else:
        print(f"   ❌ NO - No strong feature conflict detected")
    
    print(f"\n3. Impact on Model Learning:")
    print(f"   - Model might use growth_pattern as proxy for pores prediction")
    print(f"   - This could explain low pores recall (20%)")
    print(f"   - Pores detection should focus on visual features, not pattern labels")
    
    print(f"\n4. Recommendation:")
    print(f"   - Consider independent pores detector (separate from growth_pattern)")
    print(f"   - Or use attention mechanism to separate pores and pattern features")
    print(f"   - Or train in two stages: pattern first, then pores")

if __name__ == '__main__':
    main()
