"""
NI多任务数据集加载器
支持基于ds/ni创建的多任务数据集
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NIMultitaskDataset(Dataset):
    """
    NI多任务数据集类
    
    支持任务：
    1. growth_level: 生长级别 (3类)
    2. growth_pattern: 生长模式 (9类)
    3. interference_factors: 干扰因素 (多标签)
    4. fine_grained: 精细分类 (8类)
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (70, 70),
                 grayscale: bool = True):
        """
        Args:
            data_root: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据增强变换
            target_size: 目标图像尺寸
            grayscale: 是否转为灰度图
        """
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        self.grayscale = grayscale
        
        # 加载数据集信息
        self.dataset_info = self._load_dataset_info()
        self.label_mappings = self.dataset_info['label_mappings']
        
        # 加载标注数据
        self.annotations = self._load_annotations()
        
        # 设置变换
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
            
        print(f"✓ 加载{split}集: {len(self.annotations)}个样本")
        
    def _load_dataset_info(self) -> Dict:
        """加载数据集信息"""
        info_file = self.data_root / 'dataset_info.json'
        if not info_file.exists():
            raise FileNotFoundError(f"找不到数据集信息文件: {info_file}")
            
        with open(info_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_annotations(self) -> List[Dict]:
        """加载标注数据"""
        ann_file = self.data_root / f'{self.split}_annotations.json'
        if not ann_file.exists():
            raise FileNotFoundError(f"找不到标注文件: {ann_file}")
            
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            
        return annotations
    
    def _get_default_transform(self) -> transforms.Compose:
        """获取默认的数据变换"""
        transform_list = []
        
        # 基础变换
        if self.split == 'train':
            # 训练时的数据增强
            transform_list.extend([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            # 验证/测试时只做基础变换
            transform_list.extend([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
            ])
        
        # 转为tensor并归一化
        if self.grayscale:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5] if self.grayscale else [0.5, 0.5, 0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        full_path = self.data_root / image_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"找不到图像文件: {full_path}")
        
        # 使用OpenCV加载图像
        if self.grayscale:
            image = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = np.expand_dims(image, axis=2)  # 添加通道维度
        else:
            image = cv2.imread(str(full_path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise ValueError(f"无法读取图像: {full_path}")
            
        return image
    
    def _encode_labels(self, annotation: Dict) -> Dict[str, torch.Tensor]:
        """编码标签"""
        features = annotation['features']
        labels = {}
        
        # 1. 生长级别 (单标签分类)
        growth_level = features['growth_level']
        labels['growth_level'] = torch.tensor(
            self.label_mappings['growth_level'][growth_level], 
            dtype=torch.long
        )
        
        # 2. 生长模式 (单标签分类)
        growth_pattern = features['growth_pattern']
        if growth_pattern in self.label_mappings['growth_pattern']:
            pattern_id = self.label_mappings['growth_pattern'][growth_pattern]
        else:
            # 未知模式映射到0 (clean)
            pattern_id = 0
            
        labels['growth_pattern'] = torch.tensor(pattern_id, dtype=torch.long)
        
        # 3. 干扰因素 (多标签分类)
        interference_factors = features.get('interference_factors', [])
        interference_vector = torch.zeros(len(self.label_mappings['interference_factors']), dtype=torch.float32)
        
        if not interference_factors:
            # 无干扰因素
            interference_vector[self.label_mappings['interference_factors']['none']] = 1.0
        else:
            for factor in interference_factors:
                if factor in self.label_mappings['interference_factors']:
                    factor_id = self.label_mappings['interference_factors'][factor]
                    interference_vector[factor_id] = 1.0
                    
        labels['interference_factors'] = interference_vector
        
        # 4. 精细分类 (单标签分类)
        fine_grained = features['fine_grained']
        labels['fine_grained'] = torch.tensor(
            self.label_mappings['fine_grained'][fine_grained],
            dtype=torch.long
        )
        
        return labels
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """获取单个样本"""
        annotation = self.annotations[idx]
        
        # 加载图像
        image_path = annotation['local_image_path']
        image = self._load_image(image_path)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        # 编码标签
        labels = self._encode_labels(annotation)
        
        # 添加额外信息
        labels['image_id'] = annotation['image_id']
        labels['panoramic_id'] = annotation['panoramic_id']
        
        return image, labels
    
    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """计算类别权重用于处理不平衡数据"""
        from collections import Counter
        
        weights = {}
        
        # 统计各任务的类别分布
        growth_level_counts = Counter()
        growth_pattern_counts = Counter()
        fine_grained_counts = Counter()
        
        for ann in self.annotations:
            features = ann['features']
            growth_level_counts[features['growth_level']] += 1
            growth_pattern_counts[features['growth_pattern']] += 1
            fine_grained_counts[features['fine_grained']] += 1
        
        # 计算权重
        def compute_class_weights(counts, num_classes, label_mapping):
            total_samples = sum(counts.values())
            weights = torch.ones(num_classes, dtype=torch.float32)
            
            for class_name, class_id in label_mapping.items():
                if class_name in counts:
                    # 使用倒数频率作为权重
                    weights[class_id] = total_samples / (num_classes * counts[class_name])
            
            return weights
        
        # 生长级别权重
        weights['growth_level'] = compute_class_weights(
            growth_level_counts, 
            len(self.label_mappings['growth_level']),
            self.label_mappings['growth_level']
        )
        
        # 生长模式权重
        weights['growth_pattern'] = compute_class_weights(
            growth_pattern_counts,
            len(self.label_mappings['growth_pattern']), 
            self.label_mappings['growth_pattern']
        )
        
        # 精细分类权重
        weights['fine_grained'] = compute_class_weights(
            fine_grained_counts,
            len(self.label_mappings['fine_grained']),
            self.label_mappings['fine_grained']
        )
        
        return weights
    
    def get_dataset_statistics(self) -> Dict:
        """获取数据集统计信息"""
        from collections import Counter
        
        stats = {
            'total_samples': len(self.annotations),
            'growth_level_dist': Counter(),
            'growth_pattern_dist': Counter(),
            'fine_grained_dist': Counter(),
            'interference_dist': Counter(),
            'panoramic_dist': Counter()
        }
        
        for ann in self.annotations:
            features = ann['features']
            
            stats['growth_level_dist'][features['growth_level']] += 1
            stats['growth_pattern_dist'][features['growth_pattern']] += 1 
            stats['fine_grained_dist'][features['fine_grained']] += 1
            stats['panoramic_dist'][ann['panoramic_id']] += 1
            
            # 统计干扰因素
            interference = features.get('interference_factors', [])
            if not interference:
                stats['interference_dist']['none'] += 1
            else:
                for factor in interference:
                    stats['interference_dist'][factor] += 1
        
        return stats


def create_ni_dataloaders(data_root: str,
                         batch_size: int = 32,
                         num_workers: int = 4,
                         pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建NI多任务数据加载器
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 创建数据集
    train_dataset = NIMultitaskDataset(data_root, split='train')
    val_dataset = NIMultitaskDataset(data_root, split='val')
    test_dataset = NIMultitaskDataset(data_root, split='test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


# 测试代码
def test_dataset():
    """测试数据集功能"""
    print("=== 测试NI多任务数据集 ===")
    
    # 数据集路径
    data_root = "/home/aaa/ws/bioastModel/dataset_ni_multitask"
    
    if not Path(data_root).exists():
        print(f"错误: 找不到数据集目录 {data_root}")
        return
    
    # 创建数据集
    train_dataset = NIMultitaskDataset(data_root, split='train')
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"标签映射: {train_dataset.label_mappings}")
    
    # 测试单个样本
    image, labels = train_dataset[0]
    print(f"\n样本测试:")
    print(f"  图像形状: {image.shape}")
    print(f"  图像数据类型: {image.dtype}")
    print(f"  图像数值范围: [{image.min().item():.3f}, {image.max().item():.3f}]")
    
    print(f"  标签:")
    for task_name, label_tensor in labels.items():
        if isinstance(label_tensor, torch.Tensor):
            print(f"    {task_name}: {label_tensor} (shape: {label_tensor.shape})")
        else:
            print(f"    {task_name}: {label_tensor}")
    
    # 测试数据加载器
    train_loader, val_loader, test_loader = create_ni_dataloaders(data_root, batch_size=4)
    
    print(f"\n数据加载器测试:")
    print(f"  训练集批次数: {len(train_loader)}")
    print(f"  验证集批次数: {len(val_loader)}")
    print(f"  测试集批次数: {len(test_loader)}")
    
    # 测试一个批次
    for batch_images, batch_labels in train_loader:
        print(f"\n批次测试:")
        print(f"  批次图像形状: {batch_images.shape}")
        
        for task_name, task_labels in batch_labels.items():
            if isinstance(task_labels, torch.Tensor):
                print(f"    {task_name}: {task_labels.shape}")
        
        break
    
    # 获取类别权重
    class_weights = train_dataset.get_class_weights()
    print(f"\n类别权重:")
    for task_name, weights in class_weights.items():
        print(f"  {task_name}: {weights}")
    
    # 获取数据集统计
    stats = train_dataset.get_dataset_statistics()
    print(f"\n训练集统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  生长级别分布: {dict(stats['growth_level_dist'])}")
    print(f"  精细分类分布: {dict(stats['fine_grained_dist'])}")
    
    print(f"\n✓ NI多任务数据集测试通过")


if __name__ == "__main__":
    test_dataset()