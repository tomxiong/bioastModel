#!/usr/bin/env python3
"""
分析固定数据集划分的样本分布情况

检查各个任务的样本分布是否均衡:
1. Growth Level (二分类): positive/negative
2. Growth Pattern (10分类): 各种生长模式
3. Interference Factors (多标签): artifacts, debris, contamination, pores
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List

# 添加项目根路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_annotations(data_root: str, annotations_file: str = "m9e1n170.json"):
    """加载标注文件"""
    ann_path = Path(data_root) / annotations_file
    with open(ann_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_split_file(split_file: str):
    """加载固定划分文件"""
    with open(split_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_distribution(data_root: str, split_file: str, annotations_file: str = "m9e1n170.json"):
    """分析数据集分布"""

    # 加载数据
    annotations_data = load_annotations(data_root, annotations_file)
    split_data = load_split_file(split_file)

    # 创建 image_path 到 annotation 的映射
    ann_map = {ann['image_path']: ann for ann in annotations_data['annotations']}

    # 分析结果
    results = {
        'train': {'total': 0, 'growth_level': Counter(), 'growth_pattern': Counter(), 'interference': defaultdict(int)},
        'val': {'total': 0, 'growth_level': Counter(), 'growth_pattern': Counter(), 'interference': defaultdict(int)},
        'test': {'total': 0, 'growth_level': Counter(), 'growth_pattern': Counter(), 'interference': defaultdict(int)}
    }

    # 分析每个划分
    for split in ['train', 'val', 'test']:
        image_paths = split_data['splits'][split]
        results[split]['total'] = len(image_paths)

        for img_path in image_paths:
            if img_path not in ann_map:
                continue

            ann = ann_map[img_path]
            features = ann['features']

            # Growth Level
            growth_level = features.get('growth_level', 'unknown')
            results[split]['growth_level'][growth_level] += 1

            # Growth Pattern
            growth_pattern = features.get('growth_pattern', 'unknown')
            results[split]['growth_pattern'][growth_pattern] += 1

            # Interference Factors
            interference = features.get('interference_factors', [])
            for factor in interference:
                results[split]['interference'][factor] += 1

    return results, split_data


def print_distribution_report(results: Dict, split_data: Dict):
    """打印分布分析报告"""

    print("=" * 100)
    print("固定数据集分布分析报告")
    print("=" * 100)

    # 总体统计
    print("\n【总体统计】")
    print("-" * 100)
    total_train = results['train']['total']
    total_val = results['val']['total']
    total_test = results['test']['total']
    total_all = total_train + total_val + total_test

    print(f"训练集: {total_train:,} 样本 ({total_train/total_all*100:.2f}%)")
    print(f"验证集: {total_val:,} 样本 ({total_val/total_all*100:.2f}%)")
    print(f"测试集: {total_test:,} 样本 ({total_test/total_all*100:.2f}%)")
    print(f"总计:   {total_all:,} 样本")

    # Growth Level 分布
    print("\n【Growth Level 分布 (二分类)】")
    print("-" * 100)
    print(f"{'类别':<15} {'训练集':>12} {'比例':>10} {'验证集':>12} {'比例':>10} {'测试集':>12} {'比例':>10} {'总计':>12} {'比例':>10}")
    print("-" * 100)

    all_levels = set()
    for split in ['train', 'val', 'test']:
        all_levels.update(results[split]['growth_level'].keys())

    for level in sorted(all_levels):
        train_count = results['train']['growth_level'][level]
        val_count = results['val']['growth_level'][level]
        test_count = results['test']['growth_level'][level]
        total_count = train_count + val_count + test_count

        print(f"{level:<15} {train_count:>12,} {train_count/total_train*100:>9.2f}% "
              f"{val_count:>12,} {val_count/total_val*100:>9.2f}% "
              f"{test_count:>12,} {test_count/total_test*100:>9.2f}% "
              f"{total_count:>12,} {total_count/total_all*100:>9.2f}%")

    # 计算不平衡度
    pos_train = results['train']['growth_level'].get('positive', 0)
    neg_train = results['train']['growth_level'].get('negative', 0)
    imbalance_train = max(pos_train, neg_train) / min(pos_train, neg_train) if min(pos_train, neg_train) > 0 else float('inf')

    print(f"\n不平衡度分析:")
    print(f"  训练集: positive/negative = {pos_train}/{neg_train} (比例 1:{imbalance_train:.2f})")

    pos_val = results['val']['growth_level'].get('positive', 0)
    neg_val = results['val']['growth_level'].get('negative', 0)
    imbalance_val = max(pos_val, neg_val) / min(pos_val, neg_val) if min(pos_val, neg_val) > 0 else float('inf')
    print(f"  验证集: positive/negative = {pos_val}/{neg_val} (比例 1:{imbalance_val:.2f})")

    pos_test = results['test']['growth_level'].get('positive', 0)
    neg_test = results['test']['growth_level'].get('negative', 0)
    imbalance_test = max(pos_test, neg_test) / min(pos_test, neg_test) if min(pos_test, neg_test) > 0 else float('inf')
    print(f"  测试集: positive/negative = {pos_test}/{neg_test} (比例 1:{imbalance_test:.2f})")

    if imbalance_train < 1.2 and imbalance_val < 1.2 and imbalance_test < 1.2:
        print(f"  ✅ Growth Level 分布均衡 (比例接近 1:1)")
    else:
        print(f"  ⚠️  Growth Level 存在轻微不平衡")

    # Growth Pattern 分布
    print("\n【Growth Pattern 分布 (10分类)】")
    print("-" * 100)
    print(f"{'类别':<25} {'训练集':>12} {'比例':>10} {'验证集':>12} {'比例':>10} {'测试集':>12} {'比例':>10} {'总计':>12} {'比例':>10}")
    print("-" * 100)

    all_patterns = set()
    for split in ['train', 'val', 'test']:
        all_patterns.update(results[split]['growth_pattern'].keys())

    pattern_totals = []
    for pattern in sorted(all_patterns):
        train_count = results['train']['growth_pattern'][pattern]
        val_count = results['val']['growth_pattern'][pattern]
        test_count = results['test']['growth_pattern'][pattern]
        total_count = train_count + val_count + test_count
        pattern_totals.append((pattern, total_count))

        print(f"{pattern:<25} {train_count:>12,} {train_count/total_train*100:>9.2f}% "
              f"{val_count:>12,} {val_count/total_val*100:>9.2f}% "
              f"{test_count:>12,} {test_count/total_test*100:>9.2f}% "
              f"{total_count:>12,} {total_count/total_all*100:>9.2f}%")

    # 分析 Growth Pattern 不平衡度
    pattern_totals.sort(key=lambda x: x[1], reverse=True)
    max_pattern = pattern_totals[0][1]
    min_pattern = pattern_totals[-1][1]

    print(f"\n不平衡度分析:")
    print(f"  最多样本类别: {pattern_totals[0][0]} ({pattern_totals[0][1]:,} 样本)")
    print(f"  最少样本类别: {pattern_totals[-1][0]} ({pattern_totals[-1][1]:,} 样本)")
    print(f"  不平衡比: 1:{max_pattern/min_pattern:.2f}")

    # 识别稀有类别 (< 1% 样本)
    rare_patterns = [p for p, c in pattern_totals if c/total_all < 0.01]
    if rare_patterns:
        print(f"  ⚠️  稀有类别 (<1% 样本): {', '.join(rare_patterns)}")
        print(f"     建议: 对这些类别使用数据增强或类别权重")

    # Interference Factors 分布
    print("\n【Interference Factors 分布 (多标签)】")
    print("-" * 100)
    print(f"{'类别':<20} {'训练集':>12} {'比例':>10} {'验证集':>12} {'比例':>10} {'测试集':>12} {'比例':>10} {'总计':>12} {'比例':>10}")
    print("-" * 100)

    all_factors = set()
    for split in ['train', 'val', 'test']:
        all_factors.update(results[split]['interference'].keys())

    factor_totals = []
    for factor in sorted(all_factors):
        train_count = results['train']['interference'][factor]
        val_count = results['val']['interference'][factor]
        test_count = results['test']['interference'][factor]
        total_count = train_count + val_count + test_count
        factor_totals.append((factor, total_count))

        print(f"{factor:<20} {train_count:>12,} {train_count/total_train*100:>9.2f}% "
              f"{val_count:>12,} {val_count/total_val*100:>9.2f}% "
              f"{test_count:>12,} {test_count/total_test*100:>9.2f}% "
              f"{total_count:>12,} {total_count/total_all*100:>9.2f}%")

    # 计算正负样本比
    print(f"\n正负样本比分析:")
    for factor in sorted(all_factors):
        pos_total = sum(results[split]['interference'][factor] for split in ['train', 'val', 'test'])
        neg_total = total_all - pos_total
        imbalance = neg_total / pos_total if pos_total > 0 else float('inf')

        # 训练集比例
        train_pos = results['train']['interference'][factor]
        train_neg = total_train - train_pos
        train_imbalance = train_neg / train_pos if train_pos > 0 else float('inf')

        print(f"  {factor:<20} 正样本: {pos_total:>6,} | 负样本: {neg_total:>6,} | 比例 1:{imbalance:.2f} (训练集 1:{train_imbalance:.2f})")

        # 根据不平衡度给出建议
        if imbalance > 100:
            print(f"     ❌ 严重不平衡 (>1:100) - 需要强力类别权重 (建议 pos_weight={int(imbalance/5)}-{int(imbalance/3)})")
        elif imbalance > 20:
            print(f"     ⚠️  中度不平衡 (1:20-1:100) - 建议类别权重 (pos_weight={int(imbalance/4)}-{int(imbalance/2)})")
        elif imbalance > 5:
            print(f"     🟡 轻度不平衡 (1:5-1:20) - 可选类别权重 (pos_weight={int(imbalance/3)}-{int(imbalance)})")
        else:
            print(f"     ✅ 相对均衡 (<1:5)")

    # 检查训练集、验证集、测试集分布一致性
    print("\n【数据集分布一致性检查】")
    print("-" * 100)

    # Growth Pattern 一致性
    print("\nGrowth Pattern 分布一致性:")
    for pattern in sorted(all_patterns):
        train_ratio = results['train']['growth_pattern'][pattern] / total_train * 100
        val_ratio = results['val']['growth_pattern'][pattern] / total_val * 100
        test_ratio = results['test']['growth_pattern'][pattern] / total_test * 100

        # 计算方差
        ratios = [train_ratio, val_ratio, test_ratio]
        variance = np.var(ratios)

        if variance > 1.0:  # 方差大于1说明分布不一致
            print(f"  ⚠️  {pattern:<25} 分布不一致 (train: {train_ratio:.2f}%, val: {val_ratio:.2f}%, test: {test_ratio:.2f}%, var: {variance:.2f})")

    # Interference Factors 一致性
    print("\nInterference Factors 分布一致性:")
    for factor in sorted(all_factors):
        train_ratio = results['train']['interference'][factor] / total_train * 100
        val_ratio = results['val']['interference'][factor] / total_val * 100
        test_ratio = results['test']['interference'][factor] / total_test * 100

        ratios = [train_ratio, val_ratio, test_ratio]
        variance = np.var(ratios)

        if variance > 0.5:
            print(f"  ⚠️  {factor:<20} 分布不一致 (train: {train_ratio:.2f}%, val: {val_ratio:.2f}%, test: {test_ratio:.2f}%, var: {variance:.2f})")

    # 总结建议
    print("\n【优化建议】")
    print("-" * 100)

    # Interference Factors 建议
    print("\n1. Interference Factors 优化建议:")
    current_weights = [3.0, 5.0, 20.0, 1.0]  # [artifacts, debris, contamination, pores]

    for i, factor in enumerate(['artifacts', 'debris', 'contamination', 'pores']):
        if factor in all_factors:
            pos_total = sum(results[split]['interference'][factor] for split in ['train', 'val', 'test'])
            neg_total = total_all - pos_total
            imbalance = neg_total / pos_total if pos_total > 0 else float('inf')

            suggested_weight = min(max(int(imbalance / 4), 1), 50)
            current_weight = current_weights[i] if i < len(current_weights) else 1.0

            if abs(suggested_weight - current_weight) > 2:
                print(f"   {factor:<20} 当前权重: {current_weight:>5.1f} → 建议权重: {suggested_weight:>5} (不平衡比 1:{imbalance:.2f})")
            else:
                print(f"   {factor:<20} 当前权重: {current_weight:>5.1f} ✅ (合理)")

    # Growth Pattern 建议
    print("\n2. Growth Pattern 优化建议:")
    if rare_patterns:
        print(f"   稀有类别需要数据增强: {', '.join(rare_patterns)}")
        print(f"   建议策略:")
        print(f"   - 使用更强的数据增强 (旋转、翻转、亮度调整)")
        print(f"   - 考虑使用 Focal Loss 关注难分类样本")
        print(f"   - 增加这些类别的采样权重")

    # 数据增强建议
    print("\n3. 数据增强建议:")
    print("   针对 Interference Factors 的不平衡:")
    print("   - contamination (1:1499): 强烈建议对正样本进行过采样或合成")
    print("   - debris (1:21): 建议增强正样本多样性")
    print("   - artifacts (1:12): 适度数据增强")

    print("\n" + "=" * 100)


def main():
    # 数据路径
    data_root = "ds/images"
    split_file = "ds/images/dataset_split_seed42.json"
    annotations_file = "m9e1n170.json"

    # 分析分布
    results, split_data = analyze_distribution(data_root, split_file, annotations_file)

    # 打印报告
    print_distribution_report(results, split_data)

    # 保存分析结果
    output_file = "dataset_distribution_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'distribution': results,
            'split_info': split_data.get('metadata', {})
        }, f, indent=2, ensure_ascii=False)

    print(f"\n分析结果已保存至: {output_file}")


if __name__ == '__main__':
    main()
