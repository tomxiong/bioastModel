"""
基于正确业务逻辑分析 pores 需求
"""
import json
from collections import defaultdict

def main():
    # 加载数据
    with open('ds/images/m9e1n170_cleaned_round2.json', 'r') as f:
        dataset = json.load(f)
    
    with open('ds/images/dataset_split_seed44.json', 'r') as f:
        split_data = json.load(f)
    
    path_to_ann = {ann['image_path']: ann for ann in dataset['annotations']}
    
    print("="*80)
    print("业务需求完整分析")
    print("="*80)
    
    for split_name in ['train', 'test']:
        paths = split_data['splits'][split_name]
        
        print(f"\n{split_name.upper()} SET:")
        print("-"*80)
        
        # 按业务需求分类统计
        negative_pores = defaultdict(int)
        negative_no_pores = defaultdict(int)
        positive_critical_pores = defaultdict(int)
        positive_critical_no_pores = defaultdict(int)
        positive_other_pores = defaultdict(int)
        
        total_samples = 0
        
        for path in paths:
            if path not in path_to_ann:
                continue
            
            ann = path_to_ann[path]
            features = ann['features']
            
            has_pores = 'pores' in features['interference_factors']
            growth_level = features['growth_level']
            growth_pattern = features['growth_pattern']
            
            total_samples += 1
            
            if growth_level == 'negative':
                # Negative 样本：所有都需要检测 pores
                if has_pores:
                    negative_pores[growth_pattern] += 1
                else:
                    negative_no_pores[growth_pattern] += 1
            
            elif growth_level == 'positive':
                # Positive 样本：仅关键 pattern 需要检测 pores
                if growth_pattern in ['center_dots', 'weak_scattered_pos']:
                    if has_pores:
                        positive_critical_pores[growth_pattern] += 1
                    else:
                        positive_critical_no_pores[growth_pattern] += 1
                else:
                    if has_pores:
                        positive_other_pores[growth_pattern] += 1
        
        # 统计
        total_negative_pores = sum(negative_pores.values())
        total_negative_no_pores = sum(negative_no_pores.values())
        total_negative = total_negative_pores + total_negative_no_pores
        
        total_positive_critical_pores = sum(positive_critical_pores.values())
        total_positive_critical_no_pores = sum(positive_critical_no_pores.values())
        total_positive_critical = total_positive_critical_pores + total_positive_critical_no_pores
        
        total_positive_other_pores = sum(positive_other_pores.values())
        
        # 业务关键样本
        business_critical_pores = total_negative_pores + total_positive_critical_pores
        business_total = total_samples
        
        print(f"\n1. NEGATIVE 样本（需要检测 pores）:")
        print(f"   总 Negative: {total_negative} ({total_negative/total_samples*100:.1f}%)")
        print(f"   ├─ 有 pores: {total_negative_pores} ({total_negative_pores/total_negative*100:.1f}%)")
        print(f"   └─ 无 pores: {total_negative_no_pores} ({total_negative_no_pores/total_negative*100:.1f}%)")
        
        print(f"\n   Negative + Pores 的 Pattern 分布:")
        for pattern, count in sorted(negative_pores.items(), key=lambda x: -x[1]):
            total_pattern = count + negative_no_pores.get(pattern, 0)
            ratio = count / total_pattern * 100 if total_pattern > 0 else 0
            print(f"   {pattern:20s}: {count:4d}/{total_pattern:4d} ({ratio:5.1f}%)")
        
        print(f"\n2. POSITIVE 关键 Pattern（需要检测 pores）:")
        print(f"   center_dots + weak_scattered_pos: {total_positive_critical}")
        print(f"   ├─ 有 pores: {total_positive_critical_pores} ({total_positive_critical_pores/total_positive_critical*100:.1f}%)")
        print(f"   └─ 无 pores: {total_positive_critical_no_pores} ({total_positive_critical_no_pores/total_positive_critical*100:.1f}%)")
        
        print(f"\n   详细:")
        for pattern in ['center_dots', 'weak_scattered_pos']:
            pores = positive_critical_pores.get(pattern, 0)
            no_pores = positive_critical_no_pores.get(pattern, 0)
            total_pattern = pores + no_pores
            ratio = pores / total_pattern * 100 if total_pattern > 0 else 0
            print(f"   {pattern:20s}: {pores:4d}/{total_pattern:4d} ({ratio:5.1f}%)")
        
        print(f"\n3. POSITIVE 其他 Pattern（不需要检测 pores）:")
        print(f"   其他 patterns 的 pores: {total_positive_other_pores} (可忽略)")
        
        print(f"\n{'='*80}")
        print(f"业务关键统计")
        print(f"{'='*80}")
        print(f"需要检测 pores 的样本:")
        print(f"  1. 所有 Negative: {total_negative} ({total_negative/total_samples*100:.1f}%)")
        print(f"     - 其中有 pores: {total_negative_pores}")
        print(f"  2. Positive 关键 Pattern: {total_positive_critical} ({total_positive_critical/total_samples*100:.1f}%)")
        print(f"     - 其中有 pores: {total_positive_critical_pores}")
        print(f"\n总业务关键 pores: {business_critical_pores}/{total_samples} ({business_critical_pores/total_samples*100:.1f}%)")
        print(f"  - Negative pores: {total_negative_pores} ({total_negative_pores/business_critical_pores*100:.1f}%)")
        print(f"  - Positive critical pores: {total_positive_critical_pores} ({total_positive_critical_pores/business_critical_pores*100:.1f}%)")
    
    # 测试集业务目标
    print(f"\n{'='*80}")
    print(f"业务目标（测试集）")
    print(f"{'='*80}")
    
    # 重新计算测试集数据
    test_paths = split_data['splits']['test']
    test_negative_pores = 0
    test_positive_critical_pores = 0
    test_total_critical_pores = 0
    
    for path in test_paths:
        if path not in path_to_ann:
            continue
        ann = path_to_ann[path]
        features = ann['features']
        
        has_pores = 'pores' in features['interference_factors']
        if not has_pores:
            continue
        
        growth_level = features['growth_level']
        growth_pattern = features['growth_pattern']
        
        if growth_level == 'negative':
            test_negative_pores += 1
            test_total_critical_pores += 1
        elif growth_pattern in ['center_dots', 'weak_scattered_pos']:
            test_positive_critical_pores += 1
            test_total_critical_pores += 1
    
    print(f"\n需要检测的 pores 样本:")
    print(f"  Negative pores: {test_negative_pores}")
    print(f"  Positive critical pores: {test_positive_critical_pores}")
    print(f"  总计: {test_total_critical_pores}")
    
    # v0.9.9 当前性能
    current_detected = 172  # 从之前分析得知
    current_recall = current_detected / test_total_critical_pores
    
    print(f"\nv0.9.9 当前性能:")
    print(f"  检测到的 pores: {current_detected}")
    print(f"  业务关键 Recall: {current_recall*100:.1f}% ({current_detected}/{test_total_critical_pores})")
    
    target_recall = 0.75
    target_detected = int(test_total_critical_pores * target_recall)
    
    print(f"\n业务目标:")
    print(f"  目标 Recall: {target_recall*100:.0f}%")
    print(f"  需要检测: {target_detected}/{test_total_critical_pores}")
    
    if current_recall >= target_recall:
        print(f"\n✅ 已达到业务目标")
    else:
        gap = target_detected - current_detected
        print(f"\n⚠️ 还需额外检测: {gap} 个样本")
        print(f"   提升空间: {(target_recall - current_recall)*100:.1f} 个百分点")

if __name__ == '__main__':
    main()
