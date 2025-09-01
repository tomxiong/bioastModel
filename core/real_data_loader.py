"""
Real Biomedical Data Loader
用于加载bioast_dataset中的真实70x70生物医学图像数据
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import Tuple, List, Optional
import json
import logging

class RealBiomedicalDataLoader:
    """真实生物医学数据加载器 - 使用bioast_dataset"""
    
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
        positive_dir = os.path.join(self.data_dir, 'positive')
        negative_dir = os.path.join(self.data_dir, 'negative')
        
        if not (os.path.exists(positive_dir) and os.path.exists(negative_dir)):
            raise ValueError(f"Real data directories not found in {self.data_dir}")
        
        # 检查是否有预分割的train/val/test结构
        train_pos_dir = os.path.join(positive_dir, 'train')
        val_pos_dir = os.path.join(positive_dir, 'val')
        test_pos_dir = os.path.join(positive_dir, 'test')
        
        if os.path.exists(train_pos_dir) and os.path.exists(val_pos_dir) and os.path.exists(test_pos_dir):
            self.logger.info("Found pre-split dataset structure, loading train/val/test splits")
            self._load_presplit_data()
        else:
            raise ValueError("Expected pre-split train/val/test structure not found")
    
    def _load_presplit_data(self):
        """加载预分割的数据"""
        # 加载训练数据
        train_images, train_labels = self._load_split_data('train')
        val_images, val_labels = self._load_split_data('val') 
        test_images, test_labels = self._load_split_data('test')
        
        self._train_data = (train_images, train_labels)
        self._val_data = (val_images, val_labels)
        self._test_data = (test_images, test_labels)
        
        self.logger.info(f"✅ Loaded real biomedical data:")
        self.logger.info(f"  📊 Train: {len(train_images)} samples ({np.sum(train_labels)} positive, {len(train_labels) - np.sum(train_labels)} negative)")
        self.logger.info(f"  📊 Val: {len(val_images)} samples ({np.sum(val_labels)} positive, {len(val_labels) - np.sum(val_labels)} negative)")
        self.logger.info(f"  📊 Test: {len(test_images)} samples ({np.sum(test_labels)} positive, {len(test_labels) - np.sum(test_labels)} negative)")
        self.logger.info(f"  📐 Image size: {self.image_size}")
    
    def _load_split_data(self, split_name: str):
        """加载指定分割的数据"""
        images = []
        labels = []
        
        # 加载positive样本
        pos_split_dir = os.path.join(self.data_dir, 'positive', split_name)
        if os.path.exists(pos_split_dir):
            pos_files = [f for f in os.listdir(pos_split_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.logger.info(f"Loading {len(pos_files)} positive {split_name} samples...")
            
            for filename in pos_files:
                img_path = os.path.join(pos_split_dir, filename)
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
        
        # 加载negative样本
        neg_split_dir = os.path.join(self.data_dir, 'negative', split_name)
        if os.path.exists(neg_split_dir):
            neg_files = [f for f in os.listdir(neg_split_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.logger.info(f"Loading {len(neg_files)} negative {split_name} samples...")
            
            for filename in neg_files:
                img_path = os.path.join(neg_split_dir, filename)
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
        
        return np.array(images), np.array(labels)
    
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
            'dataset_type': 'real_biomedical',
            'data_source': self.data_dir,
            'train_samples': len(train_images),
            'val_samples': len(val_images),
            'test_samples': len(test_images),
            'total_samples': len(train_images) + len(val_images) + len(test_images),
            'image_shape': train_images[0].shape,
            'image_size': self.image_size,
            'num_classes': len(np.unique(train_labels)),
            'class_distribution': {
                'train': {
                    'positive': int(np.sum(train_labels)),
                    'negative': int(len(train_labels) - np.sum(train_labels))
                },
                'val': {
                    'positive': int(np.sum(val_labels)),
                    'negative': int(len(val_labels) - np.sum(val_labels))
                },
                'test': {
                    'positive': int(np.sum(test_labels)),
                    'negative': int(len(test_labels) - np.sum(test_labels))
                }
            }
        }

class BiomedicalDataset(Dataset):
    """生物医学数据集类"""
    
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

def create_real_data_loaders(batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建真实数据加载器"""
    
    # 创建数据加载器实例
    data_loader = RealBiomedicalDataLoader()
    
    # 获取数据
    train_images, train_labels = data_loader.get_train_data()
    val_images, val_labels = data_loader.get_val_data()
    test_images, test_labels = data_loader.get_test_data()
    
    # 创建数据集
    train_dataset = BiomedicalDataset(train_images, train_labels)
    val_dataset = BiomedicalDataset(val_images, val_labels)
    test_dataset = BiomedicalDataset(test_images, test_labels)
    
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
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧬 Testing Real Biomedical Data Loader")
    print("=" * 50)
    
    # 创建数据加载器
    data_loader = RealBiomedicalDataLoader()
    
    # 获取数据信息
    info = data_loader.get_data_info()
    print("\n📊 Dataset Information:")
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # 创建PyTorch数据加载器
    train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=16)
    
    print(f"\n🔄 DataLoader Information:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # 测试一个批次
    print(f"\n🧪 Testing batch loading...")
    for images, labels in train_loader:
        print(f"  ✅ Batch shape: {images.shape}")
        print(f"  ✅ Label shape: {labels.shape}")
        print(f"  ✅ Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  ✅ Labels: {labels[:8].tolist()}")
        break
    
    print(f"\n🎉 Real biomedical data loader test completed!")