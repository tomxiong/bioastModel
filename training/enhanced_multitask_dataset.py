"""
增强多任务数据集加载器
基于 ds/images/m9e1n170.json 的增强标注数据
支持2分类growth_level和多模式growth_pattern的多任务学习
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from PIL import Image
import cv2
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EnhancedMultitaskDataset(Dataset):
    """
    增强多任务数据集类
    
    支持任务：
    1. growth_level: 生长级别 (2类: negative, positive)
    2. growth_pattern: 生长模式 (12类，基于实际数据分布)
    3. interference_factors: 干扰因素 (4类多标签)
    4. microbe_type: 微生物类型 (当前仅支持细菌)
    """
    
    def __init__(self, 
                 data_root: str,
                 annotations_file: str = "m9e1n170.json",
                 split: str = 'train',
                 split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (70, 70),
                 seed: int = 42):
        """
        Args:
            data_root: 数据集根目录 (/home/aaa/ws/bioastModel/ds/images)
            annotations_file: 标注文件名
            split: 数据集划分 ('train', 'val', 'test')
            split_ratio: 训练/验证/测试集比例
            transform: 数据增强变换
            target_size: 目标图像尺寸
            seed: 随机种子
        """
        self.data_root = Path(data_root)
        self.annotations_file = annotations_file
        self.split = split
        self.split_ratio = split_ratio
        self.target_size = target_size
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载完整标注数据
        self.full_annotations = self._load_annotations()
        
        # 创建标签映射
        self.label_mappings = self._create_label_mappings()
        
        # 数据集划分
        self.annotations = self._split_dataset()
        
        # 设置变换
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        # 输出数据集信息
        self._print_dataset_info()
        
    def _load_annotations(self) -> Dict[str, Any]:
        """加载标注数据"""
        ann_file = self.data_root / self.annotations_file
        if not ann_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {ann_file}")
            
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data
    
    def _create_label_mappings(self) -> Dict[str, Dict]:
        """基于实际数据创建标签映射"""
        mappings = {}
        
        # Growth Level (2分类)
        mappings['growth_level'] = {
            'negative': 0,
            'positive': 1
        }
        
        # Growth Pattern (基于数据统计)
        patterns = set()
        for ann in self.full_annotations['annotations']:
            pattern = ann['features'].get('growth_pattern', '')
            if pattern:
                patterns.add(pattern)
        
        patterns = sorted(list(patterns))
        mappings['growth_pattern'] = {pattern: idx for idx, pattern in enumerate(patterns)}
        
        # Interference Factors (多标签)
        interference_types = ['pores', 'artifacts', 'debris', 'contamination']
        mappings['interference_factors'] = {factor: idx for idx, factor in enumerate(interference_types)}
        
        # Microbe Type
        mappings['microbe_type'] = {
            'bacteria': 0,
            'fungi': 1,
            'virus': 2,
            'other': 3
        }
        
        return mappings
    
    def _split_dataset(self) -> List[Dict]:
        """数据集划分"""
        annotations = self.full_annotations['annotations']
        
        # 过滤存在的图像文件
        valid_annotations = []
        missing_files = 0
        for ann in annotations:
            image_path = self.data_root / ann['image_path']
            if image_path.exists():
                valid_annotations.append(ann)
            else:
                missing_files += 1
        
        print(f"原始标注: {len(annotations)}个, 有效图像: {len(valid_annotations)}个, 缺失图像: {missing_files}个")
        
        # 按growth_level分层抽样
        negative_samples = [ann for ann in valid_annotations if ann['features']['growth_level'] == 'negative']
        positive_samples = [ann for ann in valid_annotations if ann['features']['growth_level'] == 'positive']
        
        def split_samples(samples, ratios):
            random.shuffle(samples)
            n = len(samples)
            train_end = int(n * ratios[0])
            val_end = train_end + int(n * ratios[1])
            
            return {
                'train': samples[:train_end],
                'val': samples[train_end:val_end],
                'test': samples[val_end:]
            }
        
        neg_split = split_samples(negative_samples, self.split_ratio)
        pos_split = split_samples(positive_samples, self.split_ratio)
        
        # 合并对应的split
        split_data = neg_split[self.split] + pos_split[self.split]
        random.shuffle(split_data)
        
        return split_data
    
    def _get_default_transform(self) -> transforms.Compose:
        """获取默认的数据变换 - 适配灰度图"""
        if self.split == 'train':
            # 训练时的数据增强 - 灰度图单通道
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 移除饱和度和色调
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图单通道归一化
            ])
        else:
            # 验证/测试时的基础变换
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
    
    def _load_image(self, image_path: str) -> Image.Image:
        """加载图像为灰度图"""
        full_path = self.data_root / image_path
        if not full_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {full_path}")
        
        # 使用PIL加载为灰度图像
        image = Image.open(full_path).convert('L')  # 'L'模式为灰度图
        return image
    
    def _encode_labels(self, features: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """编码标签 - 适配优化的多级MobileNetV3模型"""
        labels = {}
        
        # Growth Level (2分类: negative=0, positive=1)
        growth_level = features.get('growth_level', 'negative')
        labels['growth_level'] = torch.tensor(
            self.label_mappings['growth_level'].get(growth_level, 0),
            dtype=torch.long
        )
        
        # Growth Pattern (多分类)
        growth_pattern = features.get('growth_pattern', 'clean')
        labels['growth_pattern'] = torch.tensor(
            self.label_mappings['growth_pattern'].get(growth_pattern, 0),
            dtype=torch.long
        )
        
        # Interference Factors (多标签)
        interference_factors = features.get('interference_factors', [])
        interference_vector = torch.zeros(len(self.label_mappings['interference_factors']))
        for factor in interference_factors:
            if factor in self.label_mappings['interference_factors']:
                idx = self.label_mappings['interference_factors'][factor]
                interference_vector[idx] = 1.0
        labels['interference_factors'] = interference_vector
        
        # Microbe Type (可选)
        labels['microbe_type'] = torch.tensor(
            self.label_mappings['microbe_type'].get(features.get('microbe_type', 'bacteria'), 0),
            dtype=torch.long
        )
        
        # 保存置信度
        labels['confidence'] = torch.tensor(features.get('confidence', 1.0), dtype=torch.float32)
        
        return labels
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """获取单个样本 - 返回(image, labels)格式适配现有训练代码"""
        annotation = self.annotations[idx]
        
        # 加载图像
        image = self._load_image(annotation['image_path'])
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 编码标签
        labels = self._encode_labels(annotation['features'])
        
        return image, labels
    
    def _print_dataset_info(self):
        """打印数据集信息"""
        print(f"\n=== {self.split.upper()}数据集信息 ===")
        print(f"样本总数: {len(self.annotations)}")
        
        # Growth Level分布
        growth_levels = [ann['features']['growth_level'] for ann in self.annotations]
        growth_counter = Counter(growth_levels)
        print(f"Growth Level分布: {dict(growth_counter)}")
        
        # Growth Pattern分布
        growth_patterns = [ann['features']['growth_pattern'] for ann in self.annotations]
        pattern_counter = Counter(growth_patterns)
        print(f"Growth Pattern分布 (Top 5): {dict(pattern_counter.most_common(5))}")
        
        # Interference Factors分布
        all_factors = []
        for ann in self.annotations:
            all_factors.extend(ann['features'].get('interference_factors', []))
        factor_counter = Counter(all_factors)
        print(f"Interference Factors分布: {dict(factor_counter)}")
        
        print(f"图像尺寸: {self.target_size}")
        print(f"标签映射数量:")
        for task, mapping in self.label_mappings.items():
            print(f"  {task}: {len(mapping)}类")
    
    def get_num_classes(self) -> Dict[str, int]:
        """获取各任务的类别数量"""
        return {
            'growth_level': len(self.label_mappings['growth_level']),
            'growth_pattern': len(self.label_mappings['growth_pattern']),
            'interference_factors': len(self.label_mappings['interference_factors']),
            'microbe_type': len(self.label_mappings['microbe_type'])
        }
    
    def get_class_weights(self, task: str) -> torch.Tensor:
        """计算类别权重，用于处理类别不平衡"""
        if task == 'growth_level':
            labels = [ann['features']['growth_level'] for ann in self.annotations]
        elif task == 'growth_pattern':
            labels = [ann['features']['growth_pattern'] for ann in self.annotations]
        elif task == 'microbe_type':
            labels = [ann['features']['microbe_type'] for ann in self.annotations]
        else:
            return torch.ones(len(self.label_mappings[task]))
        
        counter = Counter(labels)
        total = len(labels)
        weights = []
        
        for label in self.label_mappings[task].keys():
            count = counter.get(label, 1)
            weight = total / (len(self.label_mappings[task]) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.annotations),
            'image_size': self.target_size,
            'tasks': list(self.label_mappings.keys()),
            'class_distributions': {}
        }
        
        for task in self.label_mappings.keys():
            if task == 'interference_factors':
                # 多标签任务的统计
                all_factors = []
                for ann in self.annotations:
                    all_factors.extend(ann['features'].get('interference_factors', []))
                stats['class_distributions'][task] = dict(Counter(all_factors))
            else:
                # 单标签任务的统计
                labels = [ann['features'].get(task, '') for ann in self.annotations]
                stats['class_distributions'][task] = dict(Counter(labels))
        
        return stats


def create_multitask_dataloaders(data_root: str,
                                annotations_file: str = "m9e1n170.json",
                                batch_size: int = 32,
                                num_workers: int = 4,
                                split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                                seed: int = 42) -> Dict[str, DataLoader]:
    """
    创建多任务数据加载器
    
    Args:
        data_root: 数据集根目录
        annotations_file: 标注文件名
        batch_size: 批次大小
        num_workers: 数据加载进程数
        split_ratio: 数据集划分比例
        seed: 随机种子
    
    Returns:
        包含train/val/test的DataLoader字典
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = EnhancedMultitaskDataset(
            data_root=data_root,
            annotations_file=annotations_file,
            split=split,
            split_ratio=split_ratio,
            seed=seed
        )
        
        # 训练集使用shuffle，验证/测试集不使用
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')  # 训练时丢弃最后一个不完整批次
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders


if __name__ == "__main__":
    # 测试数据集加载
    data_root = "/home/aaa/ws/bioastModel/ds/images"
    
    print("创建多任务数据加载器...")
    dataloaders = create_multitask_dataloaders(
        data_root=data_root,
        batch_size=16,
        num_workers=2
    )
    
    print("\n测试数据加载...")
    for split, dataloader in dataloaders.items():
        print(f"\n{split.upper()}集测试:")
        batch = next(iter(dataloader))
        images, labels = batch
        
        print(f"  图像batch形状: {images.shape}")
        print(f"  标签键: {list(labels.keys())}")
        for task, task_labels in labels.items():
            print(f"    {task}: {task_labels.shape}")
        
        # 仅测试第一个批次
        break
    
    print("\n数据集加载器创建完成！")