#!/usr/bin/env python3
"""
创建固定的数据集划分

将数据集按照固定的随机种子划分为 train/val/test，
并保存划分索引，确保所有模型使用相同的数据集划分。

用法:
    python scripts/create_fixed_dataset_split.py
    python scripts/create_fixed_dataset_split.py --seed 42 --train-ratio 0.7
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def create_fixed_split(
    json_path: str,
    image_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: str = None
):
    """
    创建固定的数据集划分

    Args:
        json_path: 标注文件路径
        image_root: 图像根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        output_dir: 输出目录
    """
    print("="*80)
    print("创建固定数据集划分")
    print("="*80)
    print(f"标注文件: {json_path}")
    print(f"图像根目录: {image_root}")
    print(f"划分比例: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"随机种子: {seed}")
    print("="*80)

    # 验证比例和为1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"划分比例之和必须为1.0，当前为 {train_ratio + val_ratio + test_ratio}")

    # 设置随机种子
    random.seed(seed)

    # 加载标注
    json_path = Path(json_path)
    image_root = Path(image_root)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data['annotations']
    print(f"\n总标注数: {len(annotations)}")

    # 过滤存在的图像文件
    valid_annotations = []
    missing_files = 0

    for ann in annotations:
        image_path = image_root / ann['image_path']
        if image_path.exists():
            valid_annotations.append(ann)
        else:
            missing_files += 1

    print(f"有效图像: {len(valid_annotations)}")
    print(f"缺失图像: {missing_files}")

    # 按 growth_level 分层抽样
    print("\n按 growth_level 分层抽样...")
    negative_samples = [ann for ann in valid_annotations
                       if ann['features']['growth_level'] == 'negative']
    positive_samples = [ann for ann in valid_annotations
                       if ann['features']['growth_level'] == 'positive']

    print(f"  negative: {len(negative_samples)}")
    print(f"  positive: {len(positive_samples)}")

    def split_samples(samples, ratios, split_name):
        """划分样本"""
        # 打乱顺序（使用固定种子）
        random.shuffle(samples)

        n = len(samples)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        splits = {
            'train': samples[:train_end],
            'val': samples[train_end:val_end],
            'test': samples[val_end:]
        }

        # 打印统计
        print(f"\n{split_name} 划分:")
        for split, split_samples in splits.items():
            print(f"  {split}: {len(split_samples)}")

        return splits

    # 划分 negative 和 positive
    ratios = (train_ratio, val_ratio, test_ratio)
    neg_splits = split_samples(negative_samples, ratios, "negative")
    pos_splits = split_samples(positive_samples, ratios, "positive")

    # 合并并打乱
    train_samples = neg_splits['train'] + pos_splits['train']
    val_samples = neg_splits['val'] + pos_splits['val']
    test_samples = neg_splits['test'] + pos_splits['test']

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    # 统计各划分的分布
    def get_statistics(samples, split_name):
        """获取样本统计信息"""
        stats = {
            'total': len(samples),
            'growth_level': defaultdict(int),
            'growth_pattern': defaultdict(int),
            'interference_factors': defaultdict(int)
        }

        for ann in samples:
            features = ann['features']
            stats['growth_level'][features['growth_level']] += 1
            stats['growth_pattern'][features['growth_pattern']] += 1

            # 统计干扰因素
            for factor in features.get('interference_factors', []):
                stats['interference_factors'][factor] += 1
            if not features.get('interference_factors'):
                stats['interference_factors']['none'] += 1

        # 转换为普通字典
        stats['growth_level'] = dict(stats['growth_level'])
        stats['growth_pattern'] = dict(stats['growth_pattern'])
        stats['interference_factors'] = dict(stats['interference_factors'])

        return stats

    train_stats = get_statistics(train_samples, 'train')
    val_stats = get_statistics(val_samples, 'val')
    test_stats = get_statistics(test_samples, 'test')

    # 打印详细统计
    print("\n" + "="*80)
    print("数据集划分统计")
    print("="*80)

    for split_name, stats in [('TRAIN', train_stats), ('VAL', val_stats), ('TEST', test_stats)]:
        print(f"\n=== {split_name} ===")
        print(f"总样本数: {stats['total']}")
        print(f"Growth Level: {stats['growth_level']}")
        print(f"Growth Pattern (Top 5): {dict(sorted(stats['growth_pattern'].items(), key=lambda x: x[1], reverse=True)[:5])}")
        print(f"Interference Factors (Top 5): {dict(sorted(stats['interference_factors'].items(), key=lambda x: x[1], reverse=True)[:5])}")

    # 创建索引列表（使用image_path作为唯一标识）
    train_indices = [ann['image_path'] for ann in train_samples]
    val_indices = [ann['image_path'] for ann in val_samples]
    test_indices = [ann['image_path'] for ann in test_samples]

    # 保存划分
    if output_dir is None:
        output_dir = image_root
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_file = output_dir / f"dataset_split_seed{seed}.json"

    split_data = {
        'metadata': {
            'created_at': timestamp,
            'json_path': str(json_path),
            'image_root': str(image_root),
            'seed': seed,
            'ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            },
            'total_samples': len(valid_annotations),
            'missing_files': missing_files
        },
        'splits': {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        },
        'statistics': {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats
        }
    }

    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ 固定数据集划分已保存!")
    print("="*80)
    print(f"输出文件: {split_file}")
    print(f"\n使用方法:")
    print(f"  在训练脚本中指定: --split-file {split_file}")
    print(f"  或在数据集类中加载此文件")
    print("="*80)

    # 创建一个符号链接到最新的划分文件
    latest_link = output_dir / "dataset_split_latest.json"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(split_file.name)
    print(f"\n符号链接已创建: {latest_link} -> {split_file.name}")

    return split_file


def main():
    parser = argparse.ArgumentParser(
        description='创建固定的数据集划分',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--json-path', type=str,
                       default='ds/images/m9e1n170.json',
                       help='标注文件路径')
    parser.add_argument('--image-root', type=str,
                       default='ds/images',
                       help='图像根目录')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认使用image_root）')

    args = parser.parse_args()

    create_fixed_split(
        json_path=args.json_path,
        image_root=args.image_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
