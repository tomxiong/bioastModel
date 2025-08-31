#!/usr/bin/env python3
"""
增强的多任务学习数据集
支持生长级别、生长模式、干扰因素和精细分类的多标签分类
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class EnhancedMultiTaskNIDataset(Dataset):
    """增强的多任务学习数据集"""
    
    def __init__(self, json_path: str, image_dir: str, split: str = 'train',
                 image_size: tuple = (70, 70), transform=None):
        """
        Args:
            json_path: m13.json文件路径
            image_dir: 图片根目录
            split: 数据集分割 ('train', 'val', 'test')
            image_size: 目标图片尺寸
            transform: 数据增强变换
        """
        self.json_path = json_path
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.split = split
        
        # 加载多任务数据
        self.samples = self._load_enhanced_multitask_data()
        
        # 数据增强
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # 获取任务信息
        self.task_info = self._get_enhanced_task_info()
        
        print(f"{split}集样本数: {len(self.samples)}")
        print(f"生长级别类别数: {self.task_info['growth_level']['num_classes']}")
        print(f"生长模式类别数: {self.task_info['growth_pattern']['num_classes']}")
        print(f"干扰因素类别数: {self.task_info['interference_factors']['num_classes']}")
        print(f"精细分类类别数: {self.task_info['fine_grained']['num_classes']}")
        print(f"生长级别分布: {self.task_info['growth_level']['distribution']}")
        print(f"生长模式分布: {self.task_info['growth_pattern']['distribution']}")
        print(f"干扰因素分布: {self.task_info['interference_factors']['distribution']}")
        print(f"精细分类分布: {self.task_info['fine_grained']['distribution']}")
    
    def _load_enhanced_multitask_data(self):
        """加载增强的多任务数据"""
        # 首先尝试加载m16.json的分割
        split_file = Path('dataset_splits/m16_enhanced_splits.json')
        
        if split_file.exists():
            return self._load_pre_split_data(split_file)
        
        # 如果没有m16分割，尝试旧的分割
        old_split_file = Path('dataset_splits/enhanced_multitask_splits.json')
        if old_split_file.exists():
            return self._load_pre_split_data(old_split_file)
        
        # 都没有则创建新的分割
        print(f"创建增强的多任务数据分割...")
        return self._create_enhanced_multitask_split()
    
    def _load_pre_split_data(self, split_file: Path):
        """加载预分割的数据"""
        with open(split_file, 'r', encoding='utf-8') as f:
            splits_data = json.load(f)
        
        # 获取当前分割的样本
        split_samples = splits_data.get(self.split, [])
        
        # 转换为完整格式
        full_samples = []
        for sample in split_samples:
            image_path = self.image_dir / sample['image_path']
            if image_path.exists():
                # 重新编码标签以适应新的分类
                features = sample['features']
                
                # 生长级别编码
                growth_level_mapping = {'negative': 0, 'positive': 1, 'weak_growth': 2}
                growth_level_label = growth_level_mapping.get(features['growth_level'], 0)
                
                # 生长模式编码
                growth_pattern_mapping = {
                    'clean': 0, 'clustered': 1, 'scattered': 2, 'heavy_growth': 3,
                    'small_dots': 4, 'irregular_areas': 5, 'light_gray': 6,
                    'default_positive': 7, 'default_weak_growth': 8
                }
                growth_pattern_label = growth_pattern_mapping.get(features['growth_pattern'], 0)
                
                # 干扰因素编码
                interference_mapping = {'pores': 0, 'debris': 1, 'artifacts': 2}
                interference_labels = [0] * 3
                for factor in features['interference_factors']:
                    if factor in interference_mapping:
                        interference_labels[interference_mapping[factor]] = 1
                
                # 精细分类编码
                fine_grained_label = self._generate_fine_grained_label(
                    features['growth_level'], 
                    features['growth_pattern'], 
                    features['interference_factors']
                )
                
                full_sample = {
                    'image_path': str(image_path),
                    'growth_level_label': growth_level_label,
                    'growth_pattern_label': growth_pattern_label,
                    'interference_labels': interference_labels,
                    'fine_grained_label': fine_grained_label
                }
                full_samples.append(full_sample)
        
        print(f"从 {split_file} 加载了 {len(full_samples)} 个 {self.split} 样本")
        return full_samples
    
    def _create_enhanced_multitask_split(self):
        """创建增强的多任务数据分割"""
        print(f"创建增强的多任务数据分割...")
        
        # 检查是否已经有预分割的数据
        split_file = Path("dataset_splits/m16_enhanced_splits.json")
        if split_file.exists():
            print(f"加载预分割的数据: {split_file}")
            return self._load_pre_split_data()
        
        # 加载原始数据
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 定义分类映射 (更新为m16.json的新分类)
        growth_level_mapping = {
            'negative': 0, 'positive': 1, 'weak_growth': 2
        }
        
        growth_pattern_mapping = {
            'clean': 0, 'clustered': 1, 'scattered': 2, 'heavy_growth': 3,
            'small_dots': 4, 'irregular_areas': 5, 'light_gray': 6,
            'default_positive': 7, 'default_weak_growth': 8
        }
        
        interference_mapping = {
            'pores': 0, 'debris': 1, 'artifacts': 2
        }
        
        # 过滤和准备数据
        all_samples = []
        for annotation in data['annotations']:
            if annotation['hole_number'] < 25:
                continue
            
            features = annotation['features']
            image_path = self.image_dir / annotation['image_path']
            
            if not image_path.exists():
                continue
            
            # 获取基础标签
            growth_level = features.get('growth_level', 'negative')
            growth_pattern = features.get('growth_pattern', 'clean')
            interference_factors = features.get('interference_factors', [])
            
            # 检查是否包含有效标签
            if (growth_level in growth_level_mapping and 
                growth_pattern in growth_pattern_mapping):
                
                # 生成干扰因素标签（多标签）
                interference_labels = [0] * 8
                for factor in interference_factors:
                    if factor in interference_mapping:
                        interference_labels[interference_mapping[factor]] = 1
                
                # 生成精细分类标签
                fine_grained_label = self._generate_fine_grained_label(
                    growth_level, growth_pattern, interference_factors
                )
                
                sample = {
                    'image_path': str(image_path),
                    'growth_level_label': growth_level_mapping[growth_level],
                    'growth_pattern_label': growth_pattern_mapping[growth_pattern],
                    'interference_labels': interference_labels,
                    'fine_grained_label': fine_grained_label
                }
                all_samples.append(sample)
        
        print(f"有效增强多任务样本数: {len(all_samples)}")
        
        # 创建分层分割
        splits = self._stratified_enhanced_multitask_split(all_samples)
        
        # 保存分割
        self._save_enhanced_multitask_splits(splits)
        
        # 返回当前分割的样本
        return splits.get(self.split, [])
    
    def _generate_fine_grained_label(self, growth_level: str, growth_pattern: str, 
                                   interference_factors: List[str]):
        """生成精细分类标签 - 基于m16.json的新干扰因素分类"""
        # 定义精细分类映射 (更新为包含artifacts)
        fine_grained_patterns = [
            'negative_clean', 'negative_with_pores', 'negative_with_debris', 'negative_with_artifacts',
            'positive_clustered_clean', 'positive_clustered_with_pores', 'positive_clustered_with_debris', 'positive_clustered_with_artifacts',
            'positive_scattered_clean', 'positive_scattered_with_pores', 'positive_scattered_with_debris', 'positive_scattered_with_artifacts',
            'positive_heavy_growth_clean', 'positive_heavy_growth_with_pores', 'positive_heavy_growth_with_debris', 'positive_heavy_growth_with_artifacts',
            'positive_small_dots_clean', 'positive_small_dots_with_pores', 'positive_small_dots_with_debris', 'positive_small_dots_with_artifacts',
            'positive_irregular_clean', 'positive_irregular_with_pores', 'positive_irregular_with_debris', 'positive_irregular_with_artifacts',
            'positive_light_gray_clean', 'positive_light_gray_with_pores', 'positive_light_gray_with_debris', 'positive_light_gray_with_artifacts',
            'weak_growth_small_dots_clean', 'weak_growth_small_dots_with_pores', 'weak_growth_small_dots_with_debris', 'weak_growth_small_dots_with_artifacts',
            'weak_growth_default_clean', 'weak_growth_default_with_pores', 'weak_growth_default_with_debris', 'weak_growth_default_with_artifacts'
        ]
        
        # 根据特征生成标签
        if growth_level == 'negative':
            if 'pores' in interference_factors:
                return 1  # negative_with_pores
            elif 'debris' in interference_factors:
                return 2  # negative_with_debris
            elif 'artifacts' in interference_factors:
                return 3  # negative_with_artifacts
            else:
                return 0  # negative_clean
        
        elif growth_level == 'positive':
            if growth_pattern == 'clustered':
                if 'pores' in interference_factors:
                    return 5  # positive_clustered_with_pores
                elif 'debris' in interference_factors:
                    return 6  # positive_clustered_with_debris
                elif 'artifacts' in interference_factors:
                    return 7  # positive_clustered_with_artifacts
                else:
                    return 4  # positive_clustered_clean
            elif growth_pattern == 'scattered':
                if 'pores' in interference_factors:
                    return 9  # positive_scattered_with_pores
                elif 'debris' in interference_factors:
                    return 10  # positive_scattered_with_debris
                elif 'artifacts' in interference_factors:
                    return 11  # positive_scattered_with_artifacts
                else:
                    return 8  # positive_scattered_clean
            elif growth_pattern == 'heavy_growth':
                if 'pores' in interference_factors:
                    return 13  # positive_heavy_growth_with_pores
                elif 'debris' in interference_factors:
                    return 14  # positive_heavy_growth_with_debris
                elif 'artifacts' in interference_factors:
                    return 15  # positive_heavy_growth_with_artifacts
                else:
                    return 12  # positive_heavy_growth_clean
            elif growth_pattern == 'small_dots':
                if 'pores' in interference_factors:
                    return 17  # positive_small_dots_with_pores
                elif 'debris' in interference_factors:
                    return 18  # positive_small_dots_with_debris
                elif 'artifacts' in interference_factors:
                    return 19  # positive_small_dots_with_artifacts
                else:
                    return 16  # positive_small_dots_clean
            elif growth_pattern == 'irregular_areas':
                if 'pores' in interference_factors:
                    return 21  # positive_irregular_with_pores
                elif 'debris' in interference_factors:
                    return 22  # positive_irregular_with_debris
                elif 'artifacts' in interference_factors:
                    return 23  # positive_irregular_with_artifacts
                else:
                    return 20  # positive_irregular_clean
            elif growth_pattern == 'light_gray':
                if 'pores' in interference_factors:
                    return 25  # positive_light_gray_with_pores
                elif 'debris' in interference_factors:
                    return 26  # positive_light_gray_with_debris
                elif 'artifacts' in interference_factors:
                    return 27  # positive_light_gray_with_artifacts
                else:
                    return 24  # positive_light_gray_clean
            elif growth_pattern == 'default_positive':
                if 'pores' in interference_factors:
                    return 29  # positive_default_with_pores
                elif 'debris' in interference_factors:
                    return 30  # positive_default_with_debris
                elif 'artifacts' in interference_factors:
                    return 31  # positive_default_with_artifacts
                else:
                    return 28  # positive_default_clean
            else:
                return 4  # 默认为聚集型清洁
        
        elif growth_level == 'weak_growth':
            if growth_pattern == 'small_dots':
                if 'pores' in interference_factors:
                    return 33  # weak_growth_small_dots_with_pores
                elif 'debris' in interference_factors:
                    return 34  # weak_growth_small_dots_with_debris
                elif 'artifacts' in interference_factors:
                    return 35  # weak_growth_small_dots_with_artifacts
                else:
                    return 32  # weak_growth_small_dots_clean
            elif growth_pattern == 'default_weak_growth':
                if 'pores' in interference_factors:
                    return 37  # weak_growth_default_with_pores
                elif 'debris' in interference_factors:
                    return 38  # weak_growth_default_with_debris
                elif 'artifacts' in interference_factors:
                    return 39  # weak_growth_default_with_artifacts
                else:
                    return 36  # weak_growth_default_clean
            else:
                return 32  # 默认为small_dots清洁
        
        return 0  # 默认
    
    def _stratified_enhanced_multitask_split(self, samples):
        """增强的多任务分层分割"""
        # 使用生长级别 + 生长模式的组合作为分层依据
        composite_keys = {}
        for sample in samples:
            key = (sample['growth_level_label'], sample['growth_pattern_label'])
            if key not in composite_keys:
                composite_keys[key] = []
            composite_keys[key].append(sample)
        
        print(f"发现 {len(composite_keys)} 种不同的标签组合")
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        for key, key_samples in composite_keys.items():
            growth_level_label, growth_pattern_label = key
            print(f"  组合 ({growth_level_label}, {growth_pattern_label}): {len(key_samples)} 样本")
            
            if len(key_samples) < 3:
                # 样本太少，全部加入训练集
                train_samples.extend(key_samples)
                print(f"    -> 全部加入训练集 (样本不足)")
                continue
            
            # 分层分割: 70% train, 15% val, 15% test
            train, temp = train_test_split(
                key_samples, 
                test_size=0.3, 
                random_state=42
            )
            
            val, test = train_test_split(
                temp, 
                test_size=0.5, 
                random_state=42
            )
            
            train_samples.extend(train)
            val_samples.extend(val)
            test_samples.extend(test)
            
            print(f"    -> 训练: {len(train)}, 验证: {len(val)}, 测试: {len(test)}")
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
    def _save_enhanced_multitask_splits(self, splits):
        """保存增强的多任务分割结果"""
        split_dir = Path('dataset_splits')
        split_dir.mkdir(exist_ok=True)
        
        # 转换为可序列化的格式
        serializable_splits = {}
        for split_name, samples in splits.items():
            serializable_splits[split_name] = [
                {
                    'image_path': str(Path(s['image_path']).relative_to(self.image_dir)),
                    'growth_level_label': s['growth_level_label'],
                    'growth_pattern_label': s['growth_pattern_label'],
                    'interference_labels': s['interference_labels'],
                    'fine_grained_label': s['fine_grained_label']
                }
                for s in samples
            ]
        
        output_file = split_dir / 'enhanced_multitask_splits.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_splits, f, indent=2, ensure_ascii=False)
        
        print(f"增强的多任务分割结果已保存到: {output_file}")
    
    def _get_enhanced_task_info(self):
        """获取增强的任务信息"""
        if not self.samples:
            return {
                'growth_level': {'num_classes': 0, 'distribution': {}},
                'growth_pattern': {'num_classes': 0, 'distribution': {}},
                'interference_factors': {'num_classes': 0, 'distribution': {}},
                'fine_grained': {'num_classes': 0, 'distribution': {}}
            }
        
        # 生长级别信息
        growth_level_labels = [s['growth_level_label'] for s in self.samples]
        growth_level_distribution = self._get_label_distribution(
            growth_level_labels, 
            ['negative', 'positive', 'weak_growth']
        )
        
        # 生长模式信息 (更新为包含default类别)
        growth_pattern_labels = [s['growth_pattern_label'] for s in self.samples]
        growth_pattern_distribution = self._get_label_distribution(
            growth_pattern_labels,
            ['clean', 'clustered', 'scattered', 'heavy_growth', 'small_dots', 'irregular_areas', 'light_gray', 'default_positive', 'default_weak_growth']
        )
        
        # 干扰因素信息 (更新为m16.json的新分类)
        interference_labels = [s['interference_labels'] for s in self.samples]
        interference_distribution = self._get_multilabel_distribution(
            interference_labels,
            ['pores', 'debris', 'artifacts']
        )
        
        # 精细分类信息 (更新为40类)
        fine_grained_labels = [s['fine_grained_label'] for s in self.samples]
        fine_grained_distribution = self._get_label_distribution(
            fine_grained_labels,
            ['negative_clean', 'negative_with_pores', 'negative_with_debris', 'negative_with_artifacts',
             'positive_clustered_clean', 'positive_clustered_with_pores', 'positive_clustered_with_debris', 'positive_clustered_with_artifacts',
             'positive_scattered_clean', 'positive_scattered_with_pores', 'positive_scattered_with_debris', 'positive_scattered_with_artifacts',
             'positive_heavy_growth_clean', 'positive_heavy_growth_with_pores', 'positive_heavy_growth_with_debris', 'positive_heavy_growth_with_artifacts',
             'positive_small_dots_clean', 'positive_small_dots_with_pores', 'positive_small_dots_with_debris', 'positive_small_dots_with_artifacts',
             'positive_irregular_clean', 'positive_irregular_with_pores', 'positive_irregular_with_debris', 'positive_irregular_with_artifacts',
             'positive_light_gray_clean', 'positive_light_gray_with_pores', 'positive_light_gray_with_debris', 'positive_light_gray_with_artifacts',
             'positive_default_clean', 'positive_default_with_pores', 'positive_default_with_debris', 'positive_default_with_artifacts',
             'weak_growth_small_dots_clean', 'weak_growth_small_dots_with_pores', 'weak_growth_small_dots_with_debris', 'weak_growth_small_dots_with_artifacts',
             'weak_growth_default_clean', 'weak_growth_default_with_pores', 'weak_growth_default_with_debris', 'weak_growth_default_with_artifacts']
        )
        
        return {
            'growth_level': {
                'num_classes': len(growth_level_distribution),
                'distribution': growth_level_distribution
            },
            'growth_pattern': {
                'num_classes': len(growth_pattern_distribution),
                'distribution': growth_pattern_distribution
            },
            'interference_factors': {
                'num_classes': len(interference_distribution),
                'distribution': interference_distribution
            },
            'fine_grained': {
                'num_classes': len(fine_grained_distribution),
                'distribution': fine_grained_distribution
            }
        }
    
    def _get_label_distribution(self, labels, class_names):
        """获取标签分布"""
        if not labels:
            return {}
        
        label_counts = {}
        for i, class_name in enumerate(class_names):
            count = sum(1 for label in labels if label == i)
            if count > 0:
                label_counts[class_name] = count
        
        return label_counts
    
    def _get_multilabel_distribution(self, labels_list, class_names):
        """获取多标签分布"""
        if not labels_list:
            return {}
        
        label_counts = {}
        for i, class_name in enumerate(class_names):
            count = sum(1 for labels in labels_list if labels[i] == 1)
            if count > 0:
                label_counts[class_name] = count
        
        return label_counts
    
    def _get_default_transform(self):
        """获取默认数据增强"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图片
        image = Image.open(sample['image_path']).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'growth_level_label': sample['growth_level_label'],
            'growth_pattern_label': sample['growth_pattern_label'],
            'interference_labels': torch.tensor(sample['interference_labels'], dtype=torch.float),
            'fine_grained_label': sample['fine_grained_label']
        }

def enhanced_multitask_collate_fn(batch):
    """增强的多任务数据整理函数"""
    images = torch.stack([item['image'] for item in batch])
    growth_level_labels = torch.tensor([item['growth_level_label'] for item in batch])
    growth_pattern_labels = torch.tensor([item['growth_pattern_label'] for item in batch])
    interference_labels = torch.stack([item['interference_labels'] for item in batch])
    fine_grained_labels = torch.tensor([item['fine_grained_label'] for item in batch])
    
    return {
        'image': images,
        'growth_level_label': growth_level_labels,
        'growth_pattern_label': growth_pattern_labels,
        'interference_labels': interference_labels,
        'fine_grained_label': fine_grained_labels
    }

def create_enhanced_multitask_dataloaders(json_path: str, image_dir: str, 
                                          batch_size: int = 32,
                                          image_size: tuple = (70, 70), 
                                          num_workers: int = 4):
    """创建增强的多任务数据加载器"""
    
    # 创建数据集
    train_dataset = EnhancedMultiTaskNIDataset(json_path, image_dir, split='train',
                                             image_size=image_size)
    val_dataset = EnhancedMultiTaskNIDataset(json_path, image_dir, split='val',
                                           image_size=image_size)
    test_dataset = EnhancedMultiTaskNIDataset(json_path, image_dir, split='test',
                                            image_size=image_size)
    
    # 获取任务信息
    task_info = train_dataset.task_info
    
    # 验证数据集一致性
    print("\n=== 增强的多任务数据集一致性验证 ===")
    
    def get_sample_paths(dataset):
        return set(s['image_path'] for s in dataset.samples)
    
    train_paths = get_sample_paths(train_dataset)
    val_paths = get_sample_paths(val_dataset)
    test_paths = get_sample_paths(test_dataset)
    
    # 检查重叠
    train_val_overlap = train_paths & val_paths
    train_test_overlap = train_paths & test_paths
    val_test_overlap = val_paths & test_paths
    
    if train_val_overlap:
        print(f"[WARNING] 训练集和验证集有 {len(train_val_overlap)} 个重叠样本")
    else:
        print("[OK] 训练集和验证集无重叠")
    
    if train_test_overlap:
        print(f"[WARNING] 训练集和测试集有 {len(train_test_overlap)} 个重叠样本")
    else:
        print("[OK] 训练集和测试集无重叠")
    
    if val_test_overlap:
        print(f"[WARNING] 验证集和测试集有 {len(val_test_overlap)} 个重叠样本")
    else:
        print("[OK] 验证集和测试集无重叠")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=enhanced_multitask_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=enhanced_multitask_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=enhanced_multitask_collate_fn
    )
    
    return train_loader, val_loader, test_loader, task_info

def test_enhanced_multitask_dataset():
    """测试增强的多任务数据集"""
    print("=== 测试增强的多任务数据集 ===")
    
    try:
        train_loader, val_loader, test_loader, task_info = create_enhanced_multitask_dataloaders(
            json_path='ni/m13.json',
            image_dir='ni',
            batch_size=4
        )
        
        print("[OK] 增强的多任务数据集创建成功")
        print(f"  生长级别: {task_info['growth_level']['num_classes']} 类")
        print(f"  生长模式: {task_info['growth_pattern']['num_classes']} 类")
        print(f"  干扰因素: {task_info['interference_factors']['num_classes']} 类")
        print(f"  精细分类: {task_info['fine_grained']['num_classes']} 类")
        print(f"  训练集: {len(train_loader.dataset)} 样本")
        print(f"  验证集: {len(val_loader.dataset)} 样本")
        print(f"  测试集: {len(test_loader.dataset)} 样本")
        
        # 测试数据加载
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:
                print(f"批次数据形状:")
                print(f"  图像: {batch['image'].shape}")
                print(f"  生长级别标签: {batch['growth_level_label'].shape}")
                print(f"  生长模式标签: {batch['growth_pattern_label'].shape}")
                print(f"  干扰因素标签: {batch['interference_labels'].shape}")
                print(f"  精细分类标签: {batch['fine_grained_label'].shape}")
                print(f"  生长级别标签: {batch['growth_level_label']}")
                print(f"  生长模式标签: {batch['growth_pattern_label']}")
                print(f"  干扰因素标签: {batch['interference_labels']}")
                print(f"  精细分类标签: {batch['fine_grained_label']}")
                break
        
        print("[OK] 增强的多任务数据集测试成功")
        
    except Exception as e:
        print(f"[ERROR] 增强的多任务数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_multitask_dataset()