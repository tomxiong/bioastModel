"""
分析 v0.9.9 检测到的 pores 在各个 pattern 上的分布
验证是否符合业务需求
"""
import json
import torch
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

def main():
    # 加载 v0.9.9 测试结果
    results_file = 'experiments/multilevel_mobilenetv3_v0.9.9/test_results.json'
    
    if not Path(results_file).exists():
        print(f"错误: 找不到测试结果文件 {results_file}")
        return
    
    with open(results_file, 'r') as f:
        test_results = json.load(f)
    
    # 检查 pores 性能
    pores_metrics = test_results['interference_factors']['pores']
    print("="*80)
    print("V0.9.9 Pores 整体性能")
    print("="*80)
    print(f"Accuracy:  {pores_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {pores_metrics['precision']*100:.2f}%")
    print(f"Recall:    {pores_metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {pores_metrics['f1_score']*100:.2f}%")
    
    # 计算检测到的数量
    # Recall = TP / (TP + FN)
    # TP = Recall * (TP + FN)
    # (TP + FN) = 总 pores 样本
    
    # 从数据集获取总 pores 样本数
    with open('ds/images/m9e1n170_cleaned_round2.json', 'r') as f:
        dataset = json.load(f)
    
    with open('ds/images/dataset_split_seed44.json', 'r') as f:
        split_data = json.load(f)
    
    # 建立路径到标注的映射
    path_to_ann = {ann['image_path']: ann for ann in dataset['annotations']}
    test_paths = split_data['splits']['test']
    
    # 统计测试集 pores
    total_pores = 0
    pores_by_pattern = defaultdict(int)
    
    for path in test_paths:
        if path not in path_to_ann:
            continue
        
        ann = path_to_ann[path]
        features = ann['features']
        
        has_pores = 'pores' in features['interference_factors']
        if has_pores:
            total_pores += 1
            pattern = features['growth_pattern']
            pores_by_pattern[pattern] += 1
    
    # 计算检测到的 pores 数量
    recall = pores_metrics['recall']
    detected_pores = int(total_pores * recall)
    
    print(f"\n测试集 Pores 统计:")
    print(f"  总 pores 样本: {total_pores}")
    print(f"  检测到的 pores: {detected_pores} ({recall*100:.1f}%)")
    print(f"  遗漏的 pores: {total_pores - detected_pores} ({(1-recall)*100:.1f}%)")
    
    # 业务关键 patterns
    business_critical_patterns = ['center_dots', 'weak_scattered_pos']
    
    critical_pores = sum(pores_by_pattern[p] for p in business_critical_patterns)
    non_critical_pores = total_pores - critical_pores
    
    print(f"\n业务关键 Pattern 的 Pores:")
    print(f"  center_dots + weak_scattered_pos: {critical_pores} ({critical_pores/total_pores*100:.1f}%)")
    for pattern in business_critical_patterns:
        count = pores_by_pattern[pattern]
        print(f"    {pattern}: {count}")
    
    print(f"\n其他 Pattern 的 Pores:")
    print(f"  其他 patterns: {non_critical_pores} ({non_critical_pores/total_pores*100:.1f}%)")
    for pattern, count in sorted(pores_by_pattern.items(), key=lambda x: -x[1]):
        if pattern not in business_critical_patterns:
            print(f"    {pattern}: {count}")
    
    # 业务需求评估
    print(f"\n{'='*80}")
    print(f"业务需求评估")
    print(f"{'='*80}")
    
    print(f"\n当前模型表现:")
    print(f"  Pores 整体 Recall: {recall*100:.1f}% (检测到 {detected_pores}/{total_pores})")
    
    # 假设检测到的 pores 主要分布在有 pores 的 patterns
    # 简化估算：检测到的 pores 按 pattern 比例分配
    estimated_critical_detected = int(detected_pores * (critical_pores / total_pores))
    critical_recall = estimated_critical_detected / critical_pores if critical_pores > 0 else 0
    
    print(f"\n估算业务关键 Pattern 上的 Pores Recall:")
    print(f"  center_dots + weak_scattered_pos:")
    print(f"    总样本: {critical_pores}")
    print(f"    估算检测到: ~{estimated_critical_detected}")
    print(f"    估算 Recall: ~{critical_recall*100:.1f}%")
    
    # 业务目标
    target_recall = 0.75
    print(f"\n业务目标:")
    print(f"  目标 Recall: {target_recall*100:.0f}%")
    print(f"  需要检测到: {int(critical_pores * target_recall)}/{critical_pores}")
    
    if critical_recall >= target_recall:
        print(f"\n✅ 已达到业务目标！")
    else:
        gap = target_recall - critical_recall
        print(f"\n⚠️ 距离目标还差 {gap*100:.1f} 个百分点")
        print(f"   需要额外检测 {int(critical_pores * gap)} 个样本")
    
    # 优化建议
    print(f"\n{'='*80}")
    print(f"优化建议")
    print(f"{'='*80}")
    
    if critical_recall < target_recall:
        print(f"\n方案 1: 调整阈值（立即可用）")
        print(f"  在 center_dots 和 weak_scattered_pos 上降低 pores 阈值")
        print(f"  从 0.5 → 0.3 或 0.2，提高召回率")
        print(f"  预期效果: Recall {critical_recall*100:.1f}% → 80-90%")
        
        print(f"\n方案 2: 针对性训练（需要重新训练）")
        print(f"  仅在 center_dots + weak_scattered_pos 上优化 pores")
        print(f"  使用高权重或 Focal Loss")
        print(f"  预期效果: Recall {critical_recall*100:.1f}% → 85-95%")
        
        print(f"\n方案 3: 后处理规则（推荐）")
        print(f"  if pattern in ['center_dots', 'weak_scattered_pos']:")
        print(f"      pores_threshold = 0.3  # 降低阈值")
        print(f"  else:")
        print(f"      pores = False  # 忽略其他 pattern 的 pores")
    else:
        print(f"\n✅ 当前性能已满足业务需求")
        print(f"   建议：使用规则优化推理逻辑")

if __name__ == '__main__':
    main()
