"""
MIC数据加载器
用于加载和预处理96孔板MIC测试数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Tuple, List, Optional
import json
from sklearn.model_selection import train_test_split
import logging

class MICDataLoader:
    """MIC数据加载器"""
    
    def __init__(self, data_dir: str = "data", image_size: Tuple[int, int] = (70, 70)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.logger = logging.getLogger(__name__)
        
        # 数据缓存
        self._train_data = None
        self._val_data = None
        self._test_data = None
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        # 检查是否有真实数据目录
        if os.path.exists(self.data_dir):
            self.logger.info(f"Loading data from {self.data_dir}")
            self._load_real_data()
        else:
            self.logger.warning(f"Data directory {self.data_dir} not found, generating synthetic data")
            self._generate_synthetic_data()
    
    def _load_real_data(self):
        """加载真实数据"""
        positive_dir = os.path.join(self.data_dir, 'positive')
        negative_dir = os.path.join(self.data_dir, 'negative')
        
        if not (os.path.exists(positive_dir) and os.path.exists(negative_dir)):
            self.logger.warning("Positive or negative directories not found, generating synthetic data")
            self._generate_synthetic_data()
            return
        
        # 加载图像文件
        images = []
        labels = []
        
        # 加载positive样本（递归搜索所有子目录）
        for root, dirs, files in os.walk(positive_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, filename)
                    image = cv2.imread(img_path)
                    if image is not None:
                        # 调整大小到指定尺寸
                        image = cv2.resize(image, self.image_size)
                        # 转换BGR到RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # 归一化到[0,1]
                        image = image.astype(np.float32) / 255.0
                        images.append(image)
                        labels.append(1)  # positive
        
        # 加载negative样本（递归搜索所有子目录）
        for root, dirs, files in os.walk(negative_dir):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, filename)
                    image = cv2.imread(img_path)
                    if image is not None:
                        # 调整大小到指定尺寸
                        image = cv2.resize(image, self.image_size)
                        # 转换BGR到RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # 归一化到[0,1]
                        image = image.astype(np.float32) / 255.0
                        images.append(image)
                        labels.append(0)  # negative
        
        if len(images) == 0:
            self.logger.warning("No valid images found, generating synthetic data")
            self._generate_synthetic_data()
            return
        
        images = np.array(images)
        labels = np.array(labels)
        
        self.logger.info(f"Loaded {len(images)} real images ({np.sum(labels)} positive, {len(labels) - np.sum(labels)} negative)")
        
        # 检查是否有预定义的数据分割
        split_file = os.path.join(self.data_dir, 'test_split.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            # 这里可以根据split_info进行数据分割
            # 暂时使用随机分割
        
        # 划分数据集
        # 训练集: 70%, 验证集: 15%, 测试集: 15%
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=0.15, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
        )
        
        self._train_data = (X_train, y_train)
        self._val_data = (X_val, y_val)
        self._test_data = (X_test, y_test)
        
        self.logger.info(f"Split into {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples")
    
    def _generate_synthetic_data(self):
        """生成合成数据用于训练和测试"""
        self.logger.info("Generating synthetic MIC data...")
        
        # 生成合成图像数据
        num_samples = 2000
        images = []
        labels = []
        
        for i in range(num_samples):
            # 生成70x70的合成图像
            image = self._generate_synthetic_image()
            images.append(image)
            
            # 生成标签 (0: negative, 1: positive)
            label = np.random.randint(0, 2)
            labels.append(label)
        
        images = np.array(images)
        labels = np.array(labels)
        
        # 划分数据集
        # 训练集: 70%, 验证集: 15%, 测试集: 15%
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=0.15, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 ≈ 0.15/0.85
        )
        
        self._train_data = (X_train, y_train)
        self._val_data = (X_val, y_val)
        self._test_data = (X_test, y_test)
        
        self.logger.info(f"Generated {len(X_train)} training samples")
        self.logger.info(f"Generated {len(X_val)} validation samples")
        self.logger.info(f"Generated {len(X_test)} test samples")
    
    def _generate_synthetic_image(self) -> np.ndarray:
        """生成单个合成图像"""
        # 创建基础图像
        image = np.random.normal(0.5, 0.1, (70, 70, 3))
        
        # 添加一些结构
        # 添加圆形区域（模拟孔板）
        center = (35, 35)
        radius = 30
        y, x = np.ogrid[:70, :70]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # 在圆形区域内添加变化
        image[mask] += np.random.normal(0, 0.05, image[mask].shape)
        
        # 随机添加一些"气孔"效果
        if np.random.random() < 0.3:  # 30%概率有气孔
            bubble_x = np.random.randint(10, 60)
            bubble_y = np.random.randint(10, 60)
            bubble_radius = np.random.randint(2, 8)
            
            y, x = np.ogrid[:70, :70]
            bubble_mask = (x - bubble_x)**2 + (y - bubble_y)**2 <= bubble_radius**2
            image[bubble_mask] = np.minimum(image[bubble_mask] + 0.3, 1.0)
        
        # 添加浊度效果
        turbidity = np.random.uniform(0.8, 1.2)
        image *= turbidity
        
        # 添加噪声
        noise = np.random.normal(0, 0.02, image.shape)
        image += noise
        
        # 确保值在[0, 1]范围内
        image = np.clip(image, 0, 1)
        
        return image.astype(np.float32)
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取训练数据"""
        return self._train_data
    
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取验证数据"""
        return self._val_data
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取测试数据"""
        return self._test_data
    
    def get_data_info(self) -> dict:
        """获取数据信息"""
        train_images, train_labels = self._train_data
        val_images, val_labels = self._val_data
        test_images, test_labels = self._test_data
        
        return {
            'train_samples': len(train_images),
            'val_samples': len(val_images),
            'test_samples': len(test_images),
            'image_shape': train_images[0].shape,
            'num_classes': len(np.unique(train_labels)),
            'class_distribution': {
                'train': np.bincount(train_labels).tolist(),
                'val': np.bincount(val_labels).tolist(),
                'test': np.bincount(test_labels).tolist()
            }
        }

class MICDataset(Dataset):
    """MIC数据集类"""
    
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 转换为tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # 确保图像格式为 (C, H, W)
        if len(image.shape) == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

def create_data_loaders(data_loader: MICDataLoader, 
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 获取数据
    train_images, train_labels = data_loader.get_train_data()
    val_images, val_labels = data_loader.get_val_data()
    test_images, test_labels = data_loader.get_test_data()
    
    # 创建数据集
    train_dataset = MICDataset(train_images, train_labels)
    val_dataset = MICDataset(val_images, val_labels)
    test_dataset = MICDataset(test_images, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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
    
    return train_loader, val_loader, test_loader

# 测试代码
if __name__ == "__main__":
    # 创建数据加载器
    data_loader = MICDataLoader()
    
    # 获取数据信息
    info = data_loader.get_data_info()
    print("Data Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 创建PyTorch数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(data_loader)
    
    print(f"\nDataLoader Info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # 测试一个批次
    for images, labels in train_loader:
        print(f"  Batch shape: {images.shape}")
        print(f"  Label shape: {labels.shape}")
        break