"""
FUA 数据处理管道

提供自动化的数据增强、预处理和质量检查功能
"""

import cv2
import numpy as np
import albumentations as A
from typing import Dict, Any, List, Optional, Tuple, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BioAstDataProcessor:
    """生物医学图像数据处理器"""
    
    def __init__(self, image_size: tuple = (70, 70)):
        self.image_size = image_size
        self.transforms = self._create_transforms()
        self.quality_metrics = []
    
    def _create_transforms(self) -> Dict[str, A.Compose]:
        """创建数据增强变换"""
        transforms = {
            'train': A.Compose([
                A.Resize(*self.image_size),
                A.RandomRotate90(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ]),
            'val': A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ]),
            'test': A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ])
        }
        return transforms
    
    def process_image(self, image_path: str, mode: str = 'train') -> np.ndarray:
        """处理单张图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        transformed = self.transforms[mode](image=image)
        return transformed['image']
    
    def check_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """检查图像质量"""
        metrics = {}
        
        # 计算清晰度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 计算亮度
        metrics['brightness'] = np.mean(image)
        
        # 计算对比度
        metrics['contrast'] = np.std(image)
        
        # 检测空泡（适用于生物医学图像）
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_bubble = np.array([0, 0, 200])
        upper_bubble = np.array([180, 30, 255])
        bubble_mask = cv2.inRange(hsv, lower_bubble, upper_bubble)
        metrics['bubble_ratio'] = np.sum(bubble_mask > 0) / bubble_mask.size
        
        return metrics
    
    def create_dataset(self, 
                      data_dir: str,
                      mode: str = 'train') -> Dataset:
        """创建数据集"""
        return BioAstDataset(data_dir, self, mode)
    
    def create_dataloader(self,
                        dataset: Dataset,
                        batch_size: int = 32,
                        shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )


class BioAstDataset(Dataset):
    """生物医学数据集"""
    
    def __init__(self, data_dir: str, processor: BioAstDataProcessor, mode: str):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.mode = mode
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """加载样本数据"""
        samples = []
        
        # 遍历正负样本文件夹
        for class_idx, class_name in enumerate(['negative', 'positive']):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    samples.append((str(img_path), class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # 处理图像
        image = self.processor.process_image(img_path, self.mode)
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, label


# 工厂函数
def create_data_processor(image_size: tuple = (70, 70)) -> BioAstDataProcessor:
    """创建数据处理器"""
    return BioAstDataProcessor(image_size)
