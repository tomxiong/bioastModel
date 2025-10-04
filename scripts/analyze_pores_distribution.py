"""
分析 v0.9.8 数据集中 pores 的分布和相关性
"""
import json
from collections import Counter

def main():
    """分析 pores 样本分布"""

    # 加载数据集
    with open('ds/images/m9e1n170_cleaned_round2.json', 'r') as f:
        dataset = json.load(f)

    with open('ds/images/dataset_split_seed44.json', 'r') as f:
        split_data = json.load(f)

    # 获取划分集合（使用图像路径）
    train_paths = set(split_data['splits']['train'])
    val_paths = set(split_data['splits']['val'])
    test_paths = set(split_data['splits']['test'])

    stats = {
        'train': {'total': 0, 'pores': 0, 'neg_pores': 0, 'pos_pores': 0, 'patterns': Counter()},
        'val': {'total': 0, 'pores': 0, 'neg_pores': 0, 'pos_pores': 0, 'patterns': Counter()},
        'test': {'total': 0, 'pores': 0, 'neg_pores': 0, 'pos_pores': 0, 'patterns': Counter()}
    }

    for ann in dataset['annotations']:
        image_path = ann['image_path']
        
        # 确定属于哪个集合
        if image_path in train_paths:
            split = 'train'
        elif image_path in val_paths:
            split = 'val'
        elif image_path in test_paths:
            split = 'test'
        else:
            continue

        stats[split]['total'] += 1

        has_pores = 'pores' in ann['features']['interference_factors']
        growth_level = ann['features']['growth_level']
        growth_pattern = ann['features']['growth_pattern']

        if has_pores:
            stats[split]['pores'] += 1
            if growth_level == 'negative':
                stats[split]['neg_pores'] += 1
            else:
                stats[split]['pos_pores'] += 1
                stats[split]['patterns'][growth_pattern] += 1

    # 打印统计
    print("=" * 80)
    print("Dataset Pores Distribution (m9e1n170_cleaned_round2.json)")
    print("=" * 80)

    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        print(f"\n[{split_name.upper()} SET]")
        print(f"  Total samples: {s['total']}")
        if s['total'] > 0:
            print(f"  Pores samples: {s['pores']} ({s['pores']/s['total']*100:.2f}%)")
            if s['pores'] > 0:
                print(f"  +- Negative + Pores: {s['neg_pores']} ({s['neg_pores']/s['pores']*100:.1f}% of pores)")
                print(f"  +- Positive + Pores: {s['pos_pores']} ({s['pos_pores']/s['pores']*100:.1f}% of pores)")

                if s['patterns']:
                    print(f"\n  Positive + Pores Growth Pattern Distribution:")
                    for pattern, count in s['patterns'].most_common():
                        print(f"    {pattern}: {count} ({count/s['pos_pores']*100:.1f}%)")

    # 分析整体数据集
    print("\n" + "=" * 80)
    print("Overall Dataset Statistics")
    print("=" * 80)

    total_samples = sum(s['total'] for s in stats.values())
    total_pores = sum(s['pores'] for s in stats.values())
    total_neg_pores = sum(s['neg_pores'] for s in stats.values())
    total_pos_pores = sum(s['pos_pores'] for s in stats.values())

    print(f"Total samples: {total_samples}")
    print(f"Total Pores: {total_pores} ({total_pores/total_samples*100:.2f}%)")
    print(f"Negative + Pores: {total_neg_pores} ({total_neg_pores/total_pores*100:.1f}%)")
    print(f"Positive + Pores: {total_pos_pores} ({total_pos_pores/total_pores*100:.1f}%)")

    # 合并所有 positive pores 的 pattern
    all_patterns = Counter()
    for s in stats.values():
        all_patterns.update(s['patterns'])

    if all_patterns:
        print(f"\nAll Positive + Pores Growth Pattern Distribution:")
        for pattern, count in all_patterns.most_common():
            print(f"  {pattern}: {count} ({count/total_pos_pores*100:.1f}%)")

    # 检查数据质量
    print("\n" + "=" * 80)
    print("Data Quality Assessment")
    print("=" * 80)

    pores_purity = total_neg_pores / total_pores * 100
    print(f"Pores Purity (Negative ratio): {pores_purity:.1f}%")

    if pores_purity >= 90:
        print("OK - Pores purity >= 90%")
    elif pores_purity >= 80:
        print("WARNING - Pores purity >= 80%")
    else:
        print("ERROR - Pores purity < 80%")

    # 分析可能的问题
    if total_pos_pores > 0:
        print(f"\nWARNING: Still {total_pos_pores} Positive + Pores samples")
        print("These samples might be:")
        print("  1. Valid edge cases (center_dots, weak_scattered)")
        print("  2. Uncleaned conflicting annotations")
        print("  3. Actual positive samples with pores")

    # 保存详细统计
    output = {
        'dataset': 'm9e1n170_cleaned_round2.json',
        'total_samples': total_samples,
        'total_pores': total_pores,
        'pores_purity': pores_purity,
        'split_stats': {
            k: {**v, 'patterns': dict(v['patterns'])}
            for k, v in stats.items()
        },
        'positive_pores_patterns': dict(all_patterns)
    }

    output_file = 'experiments/multilevel_mobilenetv3_v0.9.8/pores_distribution_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed statistics saved to: {output_file}")

if __name__ == '__main__':
    main()
