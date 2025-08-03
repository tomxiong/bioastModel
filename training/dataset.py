"""
数据集加载和预处理模块
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any

class BioastDataset(Dataset):
    """生物抗菌素敏感性测试数据集"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None):
        """
        Args:
            data_dir: 数据集根目录
            split: 数据分割 ('train', 'val', 'test')
            transform: 图像变换
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 类别映射
        self.class_to_idx = {'negative': 0, 'positive': 1}
        self.idx_to_class = {0: 'negative', 1: 'positive'}
        
        # 加载文件路径和标签
        self.samples = self._load_samples()
        
        print(f"加载 {split} 数据集: {len(self.samples)} 个样本")
        self._print_class_distribution()
    
    def _load_samples(self):
        """加载样本路径和标签"""
        samples = []
        
        for class_name in ['negative', 'positive']:
            class_dir = os.path.join(self.data_dir, class_name, self.split)
            if not os.path.exists(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.png'):
                    file_path = os.path.join(class_dir, filename)
                    samples.append((file_path, class_idx))
        
        return samples
    
    def _print_class_distribution(self):
        """打印类别分布"""
        class_counts = {0: 0, 1: 0}
        for _, label in self.samples:
            class_counts[label] += 1
        
        print(f"  - Negative: {class_counts[0]} 样本")
        print(f"  - Positive: {class_counts[1]} 样本")
        print(f"  - 平衡度: {min(class_counts.values())/max(class_counts.values()):.3f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(file_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败 {file_path}: {e}")
            # 返回黑色图像作为备用
            image = Image.new('RGB', (70, 70), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """计算类别权重用于处理不平衡数据"""
        class_counts = {0: 0, 1: 0}
        for _, label in self.samples:
            class_counts[label] += 1
        
        total = len(self.samples)
        weights = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        
        return torch.tensor([weights[0], weights[1]], dtype=torch.float32)

def get_transforms(split: str = 'train', image_size: int = 70) -> transforms.Compose:
    """获取数据变换"""
    
    if split == 'train':
        # 训练时的数据增强
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证和测试时的变换
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, 
                       image_size: int = 70) -> Dict[str, DataLoader]:
    """创建数据加载器"""
    
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        # 创建数据集
        transform = get_transforms(split, image_size)
        dataset = BioastDataset(data_dir, split, transform)
        
        # 创建数据加载器
        shuffle = (split == 'train')
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        data_loaders[split] = data_loader
    
    return data_loaders

def analyze_dataset_statistics(data_dir: str):
    """分析数据集统计信息"""
    print("=== 数据集统计分析 ===")
    
    # 创建数据加载器（不使用数据增强）
    stats_loaders = {}
    for split in ['train', 'val', 'test']:
        transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor()
        ])
        dataset = BioastDataset(data_dir, split, transform)
        stats_loaders[split] = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 计算每个分割的统计信息
    for split, loader in stats_loaders.items():
        print(f"\n{split.upper()} 集统计:")
        
        pixel_values = []
        labels = []
        
        for images, batch_labels in loader:
            pixel_values.append(images.numpy())
            labels.extend(batch_labels.numpy())
        
        if pixel_values:
            pixel_values = np.concatenate(pixel_values, axis=0)
            
            print(f"  样本数量: {len(labels)}")
            print(f"  图像形状: {pixel_values.shape[1:]}")
            print(f"  像素值范围: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
            print(f"  像素均值: {pixel_values.mean():.3f}")
            print(f"  像素标准差: {pixel_values.std():.3f}")
            
            # 按类别统计
            labels = np.array(labels)
            for class_idx in [0, 1]:
                class_mask = labels == class_idx
                if class_mask.sum() > 0:
                    class_pixels = pixel_values[class_mask]
                    class_name = 'negative' if class_idx == 0 else 'positive'
                    print(f"  {class_name} 类别:")
                    print(f"    数量: {class_mask.sum()}")
                    print(f"    像素均值: {class_pixels.mean():.3f}")
                    print(f"    像素标准差: {class_pixels.std():.3f}")

if __name__ == "__main__":
    # 测试数据加载器
    data_dir = "./bioast_dataset"
    
    print("测试数据加载器...")
    data_loaders = create_data_loaders(data_dir, batch_size=8)
    
    for split, loader in data_loaders.items():
        print(f"\n{split} 数据加载器:")
        print(f"  批次数量: {len(loader)}")
        
        # 测试一个批次
        for images, labels in loader:
            print(f"  批次形状: {images.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  标签分布: {torch.bincount(labels)}")
            break
    
    # 分析数据集统计
    analyze_dataset_statistics(data_dir)