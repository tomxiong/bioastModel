#!/usr/bin/env python3
"""
创建清理后的数据集

移除规则:
- growth_level = positive
- growth_pattern = clustered
- interference_factors 包含 pores

操作: 从 interference_factors 中移除 pores
"""

import json
from pathlib import Path
from collections import Counter


def load_dataset(json_path):
    """加载数据集"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def clean_dataset(data):
    """清理数据集,移除符合条件的 pores 标注"""

    cleaned_annotations = []
    removed_count = 0
    total_pores_before = 0
    total_pores_after = 0

    # 统计信息
    stats = {
        'total_samples': len(data['annotations']),
        'removed_pores_count': 0,
        'affected_samples': 0,
        'conditions': {
            'positive_clustered_pores': 0,
            'other_pores': 0
        }
    }

    for ann in data['annotations']:
        features = ann['features']
        growth_level = features.get('growth_level', '')
        growth_pattern = features.get('growth_pattern', '')
        interference = features.get('interference_factors', [])

        # 统计原始 pores
        if 'pores' in interference:
            total_pores_before += 1

        # 检查是否符合移除条件
        should_remove = (
            growth_level == 'positive' and
            growth_pattern == 'clustered' and
            'pores' in interference
        )

        # 创建新的标注
        new_ann = ann.copy()
        new_features = features.copy()
        new_interference = interference.copy()

        if should_remove:
            # 移除 pores
            new_interference.remove('pores')
            removed_count += 1
            stats['conditions']['positive_clustered_pores'] += 1
        elif 'pores' in interference:
            stats['conditions']['other_pores'] += 1

        # 更新 interference_factors
        new_features['interference_factors'] = new_interference
        new_ann['features'] = new_features

        cleaned_annotations.append(new_ann)

        # 统计清理后的 pores
        if 'pores' in new_interference:
            total_pores_after += 1

    # 更新统计
    stats['removed_pores_count'] = removed_count
    stats['affected_samples'] = removed_count
    stats['total_pores_before'] = total_pores_before
    stats['total_pores_after'] = total_pores_after

    # 创建新的数据集
    cleaned_data = data.copy()
    cleaned_data['annotations'] = cleaned_annotations

    return cleaned_data, stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "="*80)
    print("数据集清理统计")
    print("="*80)

    print(f"\n总样本数: {stats['total_samples']}")
    print(f"\nPores 统计:")
    print(f"  清理前 pores 样本数: {stats['total_pores_before']}")
    print(f"  清理后 pores 样本数: {stats['total_pores_after']}")
    print(f"  移除的 pores 标注数: {stats['removed_pores_count']}")
    print(f"  移除比例: {stats['removed_pores_count'] / stats['total_pores_before'] * 100:.1f}%")

    print(f"\n移除条件分析:")
    print(f"  positive + clustered + pores: {stats['conditions']['positive_clustered_pores']}")
    print(f"  其他 pores (保留):           {stats['conditions']['other_pores']}")

    print(f"\n受影响的样本:")
    print(f"  移除了 pores 的样本数: {stats['affected_samples']}")
    print(f"  占总样本比例: {stats['affected_samples'] / stats['total_samples'] * 100:.1f}%")


def verify_cleaning(original_data, cleaned_data):
    """验证清理结果"""
    print("\n" + "="*80)
    print("验证清理结果")
    print("="*80)

    # 检查样本数量
    original_count = len(original_data['annotations'])
    cleaned_count = len(cleaned_data['annotations'])

    print(f"\n样本数量检查:")
    print(f"  原始数据集: {original_count}")
    print(f"  清理后数据集: {cleaned_count}")

    if original_count == cleaned_count:
        print(f"  ✅ 样本数量一致")
    else:
        print(f"  ❌ 样本数量不一致!")
        return False

    # 抽样检查
    print(f"\n抽样检查 (前5个符合条件的样本):")
    checked = 0

    for orig_ann, clean_ann in zip(original_data['annotations'], cleaned_data['annotations']):
        orig_features = orig_ann['features']
        clean_features = clean_ann['features']

        orig_interference = orig_features.get('interference_factors', [])
        clean_interference = clean_features.get('interference_factors', [])

        # 检查是否符合移除条件
        if (orig_features.get('growth_level') == 'positive' and
            orig_features.get('growth_pattern') == 'clustered' and
            'pores' in orig_interference):

            if 'pores' not in clean_interference:
                print(f"  ✅ {orig_ann['image_path']}: pores 已移除")
                checked += 1

                # 显示其他 interference
                if clean_interference:
                    print(f"     保留的 interference: {clean_interference}")
                else:
                    print(f"     interference 为空")
            else:
                print(f"  ❌ {orig_ann['image_path']}: pores 未移除!")
                return False

            if checked >= 5:
                break

    if checked == 0:
        print(f"  (未找到符合条件的样本)")

    print(f"\n✅ 验证通过!")
    return True


def main():
    # 配置
    original_json = "ds/images/m9e1n170.json"
    output_json = "ds/images/m9e1n170_cleaned.json"

    print("="*80)
    print("创建清理后的数据集")
    print("="*80)
    print(f"\n原始数据集: {original_json}")
    print(f"输出数据集: {output_json}")

    print(f"\n移除规则:")
    print(f"  - growth_level = positive")
    print(f"  - growth_pattern = clustered")
    print(f"  - interference_factors 包含 pores")
    print(f"  → 从 interference_factors 中移除 pores")

    # 加载数据
    print(f"\n加载原始数据集...")
    original_data = load_dataset(original_json)
    print(f"  总样本数: {len(original_data['annotations'])}")

    # 清理数据
    print(f"\n清理数据集...")
    cleaned_data, stats = clean_dataset(original_data)

    # 打印统计
    print_statistics(stats)

    # 验证结果
    if not verify_cleaning(original_data, cleaned_data):
        print("\n❌ 验证失败,终止保存")
        return

    # 保存清理后的数据集
    print(f"\n保存清理后的数据集...")
    output_path = Path(output_json)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 已保存至: {output_path}")

    # 保存统计信息
    stats_path = output_path.parent / "dataset_cleaning_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 统计信息已保存至: {stats_path}")

    print("\n" + "="*80)
    print("✅ 数据集清理完成!")
    print("="*80)

    print(f"\n使用清理后的数据集训练:")
    print(f"  1. 更新数据集路径为: {output_json}")
    print(f"  2. 重新创建数据集划分:")
    print(f"     python scripts/create_fixed_dataset_split.py \\")
    print(f"       --json-path {output_json} \\")
    print(f"       --seed 44 \\")
    print(f"       --output-dir ds/images")
    print(f"  3. 训练新模型 v0.9.8")


if __name__ == '__main__':
    main()
