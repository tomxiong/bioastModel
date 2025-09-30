#!/usr/bin/env python3
"""
Multi-level Dataset Loader for Bacterial Image Classification
多层分类数据集加载器，用于处理m9e1n170.json数据
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class MultiLevelBacterialDataset(Dataset):
    """
    Multi-level bacterial image dataset
    支持四层分类任务的数据集
    """
    
    def __init__(self, 
                 json_path: str,
                 image_root: str,
                 transform: Optional[transforms.Compose] = None,
                 split: str = 'train',
                 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Args:
            json_path: JSON标注文件路径
            image_root: 图像根目录
            transform: 图像变换
            split: 数据集分割 ('train', 'val', 'test')
            split_ratio: 训练/验证/测试集比例
        """
        self.json_path = json_path
        self.image_root = image_root
        self.transform = transform
        self.split = split
        
        # 加载数据
        self._load_data()
        
        # 创建标签映射
        self._create_label_mappings()
        
        # 分割数据集
        self._split_dataset(split_ratio)
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_statistics()
    
    def _load_data(self):
        """加载JSON数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        logger.info(f"Loaded {len(self.annotations)} annotations from {self.json_path}")
    
    def _create_label_mappings(self):
        """创建标签映射"""
        # Growth level mapping
        self.growth_level_map = {'negative': 0, 'positive': 1}
        
        # Growth pattern mapping - 收集所有可能的值
        growth_patterns = set()
        for ann in self.annotations:
            growth_patterns.add(ann['features']['growth_pattern'])
        
        self.growth_pattern_map = {pattern: idx for idx, pattern in enumerate(sorted(growth_patterns))}
        
        # Interference factors mapping - 多标签
        interference_factors = set()
        for ann in self.annotations:
            factors = ann['features']['interference_factors']
            if factors:
                interference_factors.update(factors)
        
        self.interference_factors_map = {factor: idx for idx, factor in enumerate(sorted(interference_factors))}
        
        logger.info(f"Growth patterns: {len(self.growth_pattern_map)} classes")
        logger.info(f"Interference factors: {len(self.interference_factors_map)} classes")
        
        # 保存映射信息
        self.label_info = {
            'growth_level': self.growth_level_map,
            'growth_pattern': self.growth_pattern_map,
            'interference_factors': self.interference_factors_map
        }
    
    def _split_dataset(self, split_ratio: Tuple[float, float, float]):
        """分割数据集"""
        np.random.seed(42)  # 确保可重现
        
        total_samples = len(self.annotations)
        indices = np.random.permutation(total_samples)
        
        train_end = int(total_samples * split_ratio[0])
        val_end = train_end + int(total_samples * split_ratio[1])
        
        if self.split == 'train':
            selected_indices = indices[:train_end]
        elif self.split == 'val':
            selected_indices = indices[train_end:val_end]
        else:  # test
            selected_indices = indices[val_end:]
        
        self.samples = [self.annotations[i] for i in selected_indices]
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        # Growth level统计
        growth_levels = [sample['features']['growth_level'] for sample in self.samples]
        growth_level_counts = Counter(growth_levels)
        
        # Growth pattern统计
        growth_patterns = [sample['features']['growth_pattern'] for sample in self.samples]
        growth_pattern_counts = Counter(growth_patterns)
        
        # Interference factors统计
        interference_factors = []
        for sample in self.samples:
            factors = sample['features']['interference_factors']
            if factors:
                interference_factors.extend(factors)
            else:
                interference_factors.append('none')
        interference_counts = Counter(interference_factors)
        
        logger.info(f"\n=== {self.split.upper()} Dataset Statistics ===")
        logger.info(f"Total samples: {len(self.samples)}")
        logger.info(f"Growth level distribution: {dict(growth_level_counts)}")
        logger.info(f"Top 5 growth patterns: {dict(growth_pattern_counts.most_common(5))}")
        logger.info(f"Top 5 interference factors: {dict(interference_counts.most_common(5))}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        获取单个样本
        
        Returns:
            image: 图像张量 [1, 70, 70]
            targets: 标签字典
        """
        sample = self.samples[idx]
        
        # 加载图像
        image_path = os.path.join(self.image_root, sample['image_path'])
        
        try:
            # 加载为灰度图像
            image = Image.open(image_path).convert('L')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            else:
                # 默认变换
                image = transforms.ToTensor()(image)
            
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # 返回零图像
            image = torch.zeros(1, 70, 70)
        
        # 准备标签
        features = sample['features']
        
        targets = {}
        
        # Growth level (二分类)
        targets['growth_level'] = torch.tensor(
            self.growth_level_map[features['growth_level']], 
            dtype=torch.long
        )
        
        # Growth pattern (多分类)
        targets['growth_pattern'] = torch.tensor(
            self.growth_pattern_map[features['growth_pattern']], 
            dtype=torch.long
        )
        
        # Interference factors (多标签)
        interference_vector = torch.zeros(len(self.interference_factors_map), dtype=torch.float32)
        if features['interference_factors']:
            for factor in features['interference_factors']:
                if factor in self.interference_factors_map:
                    interference_vector[self.interference_factors_map[factor]] = 1.0
        
        targets['interference_factors'] = interference_vector
        
        return image, targets
    
    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """计算类别权重用于处理不平衡数据"""
        weights = {}
        
        # Growth level weights
        growth_levels = [sample['features']['growth_level'] for sample in self.samples]
        growth_level_counts = Counter(growth_levels)
        total_samples = len(self.samples)
        
        growth_level_weights = []
        for class_name in ['negative', 'positive']:
            count = growth_level_counts.get(class_name, 1)
            weight = total_samples / (len(self.growth_level_map) * count)
            growth_level_weights.append(weight)
        
        weights['growth_level'] = torch.tensor(growth_level_weights, dtype=torch.float32)
        
        # Growth pattern weights
        growth_patterns = [sample['features']['growth_pattern'] for sample in self.samples]
        growth_pattern_counts = Counter(growth_patterns)
        
        growth_pattern_weights = []
        for pattern in sorted(self.growth_pattern_map.keys()):
            count = growth_pattern_counts.get(pattern, 1)
            weight = total_samples / (len(self.growth_pattern_map) * count)
            growth_pattern_weights.append(weight)
        
        weights['growth_pattern'] = torch.tensor(growth_pattern_weights, dtype=torch.float32)
        
        return weights

def get_transforms(split: str = 'train') -> transforms.Compose:
    """获取数据变换"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图像归一化
        ])
    else:
        return transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

def create_multilevel_dataloaders(json_path: str,
                                 image_root: str,
                                 batch_size: int = 32,
                                 num_workers: int = 4,
                                 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建多层分类数据加载器
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = MultiLevelBacterialDataset(
        json_path=json_path,
        image_root=image_root,
        transform=get_transforms('train'),
        split='train',
        split_ratio=split_ratio
    )
    
    val_dataset = MultiLevelBacterialDataset(
        json_path=json_path,
        image_root=image_root,
        transform=get_transforms('val'),
        split='val',
        split_ratio=split_ratio
    )
    
    test_dataset = MultiLevelBacterialDataset(
        json_path=json_path,
        image_root=image_root,
        transform=get_transforms('test'),
        split='test',
        split_ratio=split_ratio
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.label_info

if __name__ == "__main__":
    # 测试数据集
    logging.basicConfig(level=logging.INFO)
    
    json_path = "/home/aaa/ws/bioastModel/ds/images/m9e1n170.json"
    image_root = "/home/aaa/ws/bioastModel/ds/images"
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
        json_path=json_path,
        image_root=image_root,
        batch_size=8
    )
    
    print("=== Label Information ===")
    for task, mapping in label_info.items():
        print(f"{task}: {len(mapping)} classes")
        if len(mapping) <= 10:  # 只显示少量类别
            print(f"  {mapping}")
    
    # 测试数据加载
    print("\n=== Testing Data Loading ===")
    for i, (images, targets) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        for task, target in targets.items():
            print(f"  {task} shape: {target.shape}")
        
        if i >= 2:  # 只测试前3个batch
            break