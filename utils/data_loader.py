#!/usr/bin/env python3
"""
Data Loader Utilities for Enhanced MultiLevel MobileNetV3
提供数据加载器创建功能，支持多任务学习
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import numpy as np

def create_data_loaders(data_dir: str, 
                       batch_size: int = 32, 
                       num_workers: int = 4,
                       image_size: Tuple[int, int] = (70, 70)) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        num_workers: 数据加载进程数
        image_size: 图像尺寸
        
    Returns:
        train_loader, val_loader
    """
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"⚠️  数据目录 {data_dir} 不存在，使用模拟数据")
        return create_synthetic_data_loaders(batch_size, num_workers, image_size)
    
    # 尝试加载真实数据
    try:
        # 这里可以根据实际数据格式实现数据加载
        # 目前使用模拟数据作为示例
        return create_synthetic_data_loaders(batch_size, num_workers, image_size)
        
    except Exception as e:
        print(f"⚠️  加载真实数据失败: {e}")
        print("🧪 使用模拟数据进行训练")
        return create_synthetic_data_loaders(batch_size, num_workers, image_size)


def create_synthetic_data_loaders(batch_size: int = 32, 
                                 num_workers: int = 4,
                                 image_size: Tuple[int, int] = (70, 70)) -> Tuple[DataLoader, DataLoader]:
    """
    创建模拟数据加载器用于测试和演示
    
    Args:
        batch_size: 批次大小
        num_workers: 数据加载进程数
        image_size: 图像尺寸
        
    Returns:
        train_loader, val_loader
    """
    
    # 创建模拟数据
    num_train_samples = 1000
    num_val_samples = 200
    
    # 训练数据
    train_images = torch.randn(num_train_samples, 1, image_size[0], image_size[1])
    train_labels = {
        'growth_level': torch.randint(0, 2, (num_train_samples,)),
        'growth_pattern': torch.randint(0, 12, (num_train_samples,)),
        'interference_factors': torch.randint(0, 2, (num_train_samples, 4)).float()  # 多标签二分类
    }
    
    # 验证数据
    val_images = torch.randn(num_val_samples, 1, image_size[0], image_size[1])
    val_labels = {
        'growth_level': torch.randint(0, 2, (num_val_samples,)),
        'growth_pattern': torch.randint(0, 12, (num_val_samples,)),
        'interference_factors': torch.randint(0, 2, (num_val_samples, 4)).float()  # 多标签二分类
    }
    
    # 创建数据集
    train_dataset = MultiTaskDataset(train_images, train_labels)
    val_dataset = MultiTaskDataset(val_images, val_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"📊 模拟数据加载器创建完成:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {image_size}")
    
    return train_loader, val_loader


class MultiTaskDataset(torch.utils.data.Dataset):
    """多任务数据集类"""
    
    def __init__(self, images: torch.Tensor, labels: Dict[str, torch.Tensor]):
        """
        初始化多任务数据集
        
        Args:
            images: 图像张量 [N, C, H, W]
            labels: 标签字典 {'task_name': tensor}
        """
        self.images = images
        self.labels = labels
        
        # 验证数据一致性
        num_samples = len(images)
        for task_name, task_labels in labels.items():
            assert len(task_labels) == num_samples, f"标签 {task_name} 长度不匹配"
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            image, labels_dict
        """
        image = self.images[idx]
        labels = {task_name: task_labels[idx] for task_name, task_labels in self.labels.items()}
        
        return image, labels


def create_real_data_loaders(data_dir: str,
                           batch_size: int = 32,
                           num_workers: int = 4,
                           image_size: Tuple[int, int] = (70, 70)) -> Tuple[DataLoader, DataLoader]:
    """
    创建真实数据加载器（待实现）
    
    这个函数需要根据实际的数据格式和存储结构来实现
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        num_workers: 数据加载进程数
        image_size: 图像尺寸
        
    Returns:
        train_loader, val_loader
    """
    
    # TODO: 实现真实数据加载逻辑
    # 1. 扫描数据目录
    # 2. 加载图像和标签
    # 3. 数据预处理和增强
    # 4. 创建数据集和数据加载器
    
    raise NotImplementedError("真实数据加载器尚未实现，请使用create_data_loaders函数")


if __name__ == "__main__":
    # 测试数据加载器
    print("🧪 测试数据加载器...")
    
    train_loader, val_loader = create_data_loaders("test_data", batch_size=16)
    
    # 测试数据加载
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签: {list(labels.keys())}")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print("✅ 数据加载器测试完成")