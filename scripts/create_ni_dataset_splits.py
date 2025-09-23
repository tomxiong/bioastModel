#!/usr/bin/env python3
"""
创建基于ds/ni/m16.json的多任务训练数据集划分脚本
支持多任务标注：growth_level, growth_pattern, interference_factors, fine_grained分类

数据划分比例：
- 训练集：70%
- 验证集：20%
- 测试集：10%
"""

import json
import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_m16_annotations(json_path: str) -> Dict:
    """加载m16.json标注文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_dataset_distribution(annotations: List[Dict]) -> Dict:
    """分析数据集分布"""
    stats = {
        'total_samples': len(annotations),
        'growth_level_dist': Counter(),
        'growth_pattern_dist': Counter(),
        'interference_dist': Counter(),
        'panoramic_dist': Counter(),
        'microbe_type_dist': Counter()
    }
    
    # 统计各字段分布
    for ann in annotations:
        features = ann['features']
        
        stats['growth_level_dist'][features['growth_level']] += 1
        stats['growth_pattern_dist'][features['growth_pattern']] += 1
        stats['microbe_type_dist'][features['microbe_type']] += 1
        stats['panoramic_dist'][ann['panoramic_id']] += 1
        
        # 统计干扰因素
        interference_factors = features.get('interference_factors', [])
        if not interference_factors:
            stats['interference_dist']['none'] += 1
        else:
            for factor in interference_factors:
                stats['interference_dist'][factor] += 1
    
    return stats

def create_fine_grained_labels(annotations: List[Dict]) -> List[Dict]:
    """
    基于现有标注创建精细分类标签
    结合growth_level, growth_pattern, interference_factors创建15类精细分类
    """
    updated_annotations = []
    
    fine_grained_mapping = {}
    label_counter = 0
    
    for ann in annotations.copy():
        features = ann['features']
        
        # 构建精细分类键
        growth_level = features['growth_level']
        growth_pattern = features['growth_pattern']
        interference = features.get('interference_factors', [])
        
        # 创建精细分类标签
        if growth_level == 'positive':
            if growth_pattern == 'clustered':
                if not interference:
                    fine_label = 'positive_cluster_no_pores'
                elif 'pores' in interference:
                    fine_label = 'positive_cluster_with_pores'
                else:
                    fine_label = 'positive_cluster_overlapping_pores'
            elif growth_pattern == 'default_positive':
                fine_label = 'positive_cluster_no_pores'
            else:
                fine_label = 'positive_other'
        elif growth_level == 'negative':
            if growth_pattern == 'clean':
                if not interference:
                    fine_label = 'negative_clean_no_pores'
                elif 'pores' in interference:
                    fine_label = 'negative_clean_with_pores'
                else:
                    fine_label = 'negative_clean_other'
            else:
                fine_label = 'negative_other'
        else:
            # weak_growth (如果存在)
            fine_label = 'weak_growth_other'
        
        # 特殊干扰情况
        if 'debris' in interference:
            fine_label = 'with_debris'
        
        # 添加到映射
        if fine_label not in fine_grained_mapping:
            fine_grained_mapping[fine_label] = label_counter
            label_counter += 1
        
        # 更新标注
        ann['features']['fine_grained'] = fine_label
        ann['features']['fine_grained_id'] = fine_grained_mapping[fine_label]
        
        updated_annotations.append(ann)
    
    return updated_annotations, fine_grained_mapping

def stratified_split(annotations: List[Dict], 
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.2,
                    test_ratio: float = 0.1,
                    random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分层采样划分数据集
    保证各类别在训练集、验证集、测试集中的比例一致
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例和必须为1"
    
    # 按growth_level进行分层
    stratify_groups = defaultdict(list)
    for i, ann in enumerate(annotations):
        key = ann['features']['growth_level']
        stratify_groups[key].append(i)
    
    train_indices, val_indices, test_indices = [], [], []
    
    for group, indices in stratify_groups.items():
        random.shuffle(indices)
        
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # 随机打乱
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    train_set = [annotations[i] for i in train_indices]
    val_set = [annotations[i] for i in val_indices]
    test_set = [annotations[i] for i in test_indices]
    
    return train_set, val_set, test_set

def create_dataset_structure(base_dir: str, ni_dir: str, 
                           train_set: List[Dict], 
                           val_set: List[Dict], 
                           test_set: List[Dict],
                           copy_images: bool = True):
    """
    创建标准的数据集目录结构并复制图片
    
    结构:
    dataset_ni_multitask/
    ├── train/
    ├── val/
    ├── test/
    ├── train_annotations.json
    ├── val_annotations.json
    ├── test_annotations.json
    └── dataset_info.json
    """
    
    dataset_dir = Path(base_dir) / "dataset_ni_multitask"
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制图片并保存标注
    splits = {
        'train': train_set,
        'val': val_set, 
        'test': test_set
    }
    
    split_stats = {}
    
    for split_name, split_data in splits.items():
        split_dir = dataset_dir / split_name
        
        # 统计信息
        split_stats[split_name] = {
            'total_samples': len(split_data),
            'growth_level_dist': Counter(),
            'growth_pattern_dist': Counter(),
            'interference_dist': Counter(),
            'fine_grained_dist': Counter()
        }
        
        # 保存标注和复制图片
        for ann in split_data:
            # 复制图片
            src_path = Path(ni_dir) / ann['image_path']
            dst_path = split_dir / f"{ann['image_id']}.png"
            
            if copy_images and src_path.exists():
                shutil.copy2(src_path, dst_path)
            elif not src_path.exists():
                print(f"警告: 图片不存在 {src_path}")
            
            # 更新路径信息
            ann['local_image_path'] = f"{split_name}/{ann['image_id']}.png"
            
            # 更新统计
            features = ann['features']
            split_stats[split_name]['growth_level_dist'][features['growth_level']] += 1
            split_stats[split_name]['growth_pattern_dist'][features['growth_pattern']] += 1
            split_stats[split_name]['fine_grained_dist'][features['fine_grained']] += 1
            
            interference = features.get('interference_factors', [])
            if not interference:
                split_stats[split_name]['interference_dist']['none'] += 1
            else:
                for factor in interference:
                    split_stats[split_name]['interference_dist'][factor] += 1
        
        # 保存标注文件
        ann_file = dataset_dir / f"{split_name}_annotations.json"
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 创建{split_name}集: {len(split_data)}个样本")
    
    return dataset_dir, split_stats

def save_dataset_info(dataset_dir: Path, split_stats: Dict, fine_grained_mapping: Dict):
    """保存数据集信息"""
    
    # 创建标签映射
    label_mappings = {
        'growth_level': {
            'negative': 0,
            'positive': 1,
            'weak_growth': 2  # 如果存在
        },
        'growth_pattern': {
            'clean': 0,
            'clustered': 1,
            'default_positive': 2,
            'scattered': 3,
            'small_dots': 4,
            'ring_shaped': 5,
            'irregular': 6,
            'mixed': 7,
            'sparse': 8
        },
        'interference_factors': {
            'none': 0,
            'pores': 1,
            'debris': 2,
            'artifacts': 3,
            'contamination': 4
        },
        'fine_grained': fine_grained_mapping
    }
    
    # 数据集信息
    dataset_info = {
        'name': 'NI_Multitask_Dataset',
        'description': '基于ds/ni/m16.json的多任务菌落检测数据集',
        'created_at': str(datetime.now()),
        'tasks': {
            'growth_level': {
                'type': 'single_label_classification',
                'num_classes': len(label_mappings['growth_level']),
                'class_names': list(label_mappings['growth_level'].keys())
            },
            'growth_pattern': {
                'type': 'single_label_classification', 
                'num_classes': len(label_mappings['growth_pattern']),
                'class_names': list(label_mappings['growth_pattern'].keys())
            },
            'interference_factors': {
                'type': 'multi_label_classification',
                'num_classes': len(label_mappings['interference_factors']),
                'class_names': list(label_mappings['interference_factors'].keys())
            },
            'fine_grained': {
                'type': 'single_label_classification',
                'num_classes': len(label_mappings['fine_grained']),
                'class_names': list(label_mappings['fine_grained'].keys())
            }
        },
        'label_mappings': label_mappings,
        'split_statistics': split_stats,
        'image_format': 'PNG',
        'image_size': '70x70 (expected)',
        'color_mode': 'grayscale'
    }
    
    # 保存数据集信息
    info_file = dataset_dir / 'dataset_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    return dataset_info

def main():
    """主函数"""
    print("=== 创建NI多任务数据集划分 ===")
    
    # 配置路径
    project_root = Path(__file__).parent.parent
    ni_dir = project_root / "ds" / "ni"
    m16_json = ni_dir / "m16.json"
    output_dir = project_root
    
    if not m16_json.exists():
        print(f"错误: 找不到标注文件 {m16_json}")
        return
    
    # 1. 加载标注数据
    print(f"加载标注数据: {m16_json}")
    data = load_m16_annotations(str(m16_json))
    annotations = data['annotations']
    
    # 2. 分析数据分布
    print(f"\n原始数据集统计:")
    original_stats = analyze_dataset_distribution(annotations)
    print(f"  总样本数: {original_stats['total_samples']}")
    print(f"  生长级别分布: {dict(original_stats['growth_level_dist'])}")
    print(f"  生长模式分布: {dict(original_stats['growth_pattern_dist'])}")
    print(f"  干扰因素分布: {dict(original_stats['interference_dist'])}")
    print(f"  全景图分布: {len(original_stats['panoramic_dist'])}个全景图")
    
    # 3. 创建精细分类标签
    print(f"\n创建精细分类标签...")
    annotations, fine_grained_mapping = create_fine_grained_labels(annotations)
    print(f"  精细分类类别数: {len(fine_grained_mapping)}")
    print(f"  精细分类映射: {fine_grained_mapping}")
    
    # 4. 划分数据集
    print(f"\n数据集划分...")
    train_set, val_set, test_set = stratified_split(
        annotations, 
        train_ratio=0.7,
        val_ratio=0.2, 
        test_ratio=0.1,
        random_seed=42
    )
    
    print(f"  训练集: {len(train_set)}个样本 ({len(train_set)/len(annotations)*100:.1f}%)")
    print(f"  验证集: {len(val_set)}个样本 ({len(val_set)/len(annotations)*100:.1f}%)")  
    print(f"  测试集: {len(test_set)}个样本 ({len(test_set)/len(annotations)*100:.1f}%)")
    
    # 5. 创建数据集目录结构
    print(f"\n创建数据集目录结构...")
    dataset_dir, split_stats = create_dataset_structure(
        str(output_dir),
        str(ni_dir),
        train_set, 
        val_set, 
        test_set,
        copy_images=True
    )
    
    # 6. 保存数据集信息
    print(f"\n保存数据集信息...")
    dataset_info = save_dataset_info(dataset_dir, split_stats, fine_grained_mapping)
    
    print(f"\n✓ 数据集创建完成")
    print(f"  数据集目录: {dataset_dir}")
    print(f"  数据集信息文件: {dataset_dir}/dataset_info.json")
    
    # 7. 输出详细统计
    print(f"\n=== 详细统计信息 ===")
    for split_name, stats in split_stats.items():
        print(f"\n{split_name.upper()}集 ({stats['total_samples']}个样本):")
        print(f"  生长级别: {dict(stats['growth_level_dist'])}")
        print(f"  生长模式: {dict(stats['growth_pattern_dist'])}")
        print(f"  干扰因素: {dict(stats['interference_dist'])}")
        print(f"  精细分类: {dict(stats['fine_grained_dist'])}")
    
    print(f"\n任务配置:")
    for task_name, task_config in dataset_info['tasks'].items():
        print(f"  {task_name}: {task_config['type']}, {task_config['num_classes']}类")
    
    return dataset_dir

if __name__ == "__main__":
    try:
        dataset_dir = main()
        print(f"\n🎉 数据集划分完成! 数据集位于: {dataset_dir}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()