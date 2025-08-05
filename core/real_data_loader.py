"""
真实数据加载器 - 处理bioast_dataset的复杂目录结构
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Tuple, List, Optional
import json
import logging

class RealDataLoader:
    """真实数据加载器"""
    
    def __init__(self, data_dir: str = "bioast_dataset", image_size: Tuple[int, int] = (70, 70)):
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
        """加载真实数据"""
        self.logger.info(f"Loading real data from {self.data_dir}")
        
        # 检查数据目录结构
        positive_dir = os.path.join(self.data_dir, 'positive')
        negative_dir = os.path.join(self.data_dir, 'negative')
        
        if not (os.path.exists(positive_dir) and os.path.exists(negative_dir)):
            raise ValueError(f"Data directories not found: {positive_dir}, {negative_dir}")
        
        # 加载训练、验证、测试数据
        self._train_data = self._load_split_data('train')
        self._val_data = self._load_split_data('val')
        self._test_data = self._load_split_data('test')
        
        train_images, train_labels = self._train_data
        val_images, val_labels = self._val_data
        test_images, test_labels = self._test_data
        
        self.logger.info(f"Loaded {len(train_images)} training samples")
        self.logger.info(f"Loaded {len(val_images)} validation samples")
        self.logger.info(f"Loaded {len(test_images)} test samples")
    
    def _load_split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载指定分割的数据"""
        images = []
        labels = []
        
        # 加载positive样本
        pos_split_dir = os.path.join(self.data_dir, 'positive', split)
        if os.path.exists(pos_split_dir):
            for filename in os.listdir(pos_split_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(pos_split_dir, filename)
                    image = self._load_image(img_path)
                    if image is not None:
                        images.append(image)
                        labels.append(1)  # positive
        
        # 加载negative样本
        neg_split_dir = os.path.join(self.data_dir, 'negative', split)
        if os.path.exists(neg_split_dir):
            for filename in os.listdir(neg_split_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(neg_split_dir, filename)
                    image = self._load_image(img_path)
                    if image is not None:
                        images.append(image)
                        labels.append(0)  # negative
        
        if len(images) == 0:
            self.logger.warning(f"No images found for {split} split")
            # 返回空数组但保持正确的形状
            return np.empty((0, *self.image_size, 3), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
        images = np.array(images)
        labels = np.array(labels)
        
        pos_count = np.sum(labels)
        neg_count = len(labels) - pos_count
        self.logger.info(f"{split} split: {len(images)} images ({pos_count} positive, {neg_count} negative)")
        
        return images, labels
    
    def _load_image(self, img_path: str) -> Optional[np.ndarray]:
        """加载单个图像"""
        try:
            image = cv2.imread(img_path)
            if image is None:
                return None
            
            # 调整大小到指定尺寸
            image = cv2.resize(image, self.image_size)
            # 转换BGR到RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 归一化到[0,1]
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            self.logger.warning(f"Failed to load image {img_path}: {e}")
            return None
    
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
            'image_shape': train_images[0].shape if len(train_images) > 0 else None,
            'num_classes': 2,
            'class_distribution': {
                'train': np.bincount(train_labels, minlength=2).tolist() if len(train_labels) > 0 else [0, 0],
                'val': np.bincount(val_labels, minlength=2).tolist() if len(val_labels) > 0 else [0, 0],
                'test': np.bincount(test_labels, minlength=2).tolist() if len(test_labels) > 0 else [0, 0]
            }
        }

class RealDataset(Dataset):
    """真实数据集类"""
    
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

def create_real_data_loaders(data_dir: str = "bioast_dataset",
                            image_size: Tuple[int, int] = (70, 70),
                            batch_size: int = 32,
                            num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建真实数据加载器"""
    
    # 创建数据加载器
    data_loader = RealDataLoader(data_dir, image_size)
    
    # 获取数据
    train_images, train_labels = data_loader.get_train_data()
    val_images, val_labels = data_loader.get_val_data()
    test_images, test_labels = data_loader.get_test_data()
    
    # 创建数据集
    train_dataset = RealDataset(train_images, train_labels)
    val_dataset = RealDataset(val_images, val_labels)
    test_dataset = RealDataset(test_images, test_labels)
    
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

if __name__ == "__main__":
    # 测试真实数据加载器
    try:
        data_loader = RealDataLoader()
        info = data_loader.get_data_info()
        print("Real Data Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 创建PyTorch数据加载器
        train_loader, val_loader, test_loader = create_real_data_loaders()
        
        print(f"\nDataLoader Info:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
    except Exception as e:
        print(f"Error loading real data: {e}")