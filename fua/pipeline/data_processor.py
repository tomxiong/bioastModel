"""
FUA 数据处理管道

提供自动化的数据增强、预处理和质量检查功能
"""

import cv2
import numpy as np
import albumentations as A
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """数据处理模式"""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    INFERENCE = "inference"


class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class ProcessingResult:
    """处理结果"""
    image: np.ndarray
    metrics: Dict[str, float]
    quality_level: QualityLevel
    processing_time: float
    warnings: List[str]


@dataclass
class DatasetStats:
    """数据集统计信息"""
    total_images: int
    class_distribution: Dict[str, int]
    quality_distribution: Dict[str, int]
    average_metrics: Dict[str, float]
    processing_errors: int


class BioAstDataProcessor:
    """生物医学图像数据处理器"""
    
    def __init__(self, 
                 image_size: tuple = (70, 70),
                 quality_thresholds: Optional[Dict[str, float]] = None,
                 enable_auto_augment: bool = True,
                 num_workers: int = 4):
        self.image_size = image_size
        self.quality_thresholds = quality_thresholds or {
            'sharpness': {'excellent': 100, 'good': 50, 'acceptable': 20},
            'brightness': {'min': 30, 'max': 220},
            'contrast': {'min': 20, 'excellent': 50},
            'bubble_ratio': {'max': 0.3}
        }
        self.enable_auto_augment = enable_auto_augment
        self.num_workers = num_workers
        
        # 创建增强策略
        self.transforms = self._create_transforms()
        self.auto_augment = self._create_auto_augment()
        
        # 统计信息
        self.processing_stats = {
            'total_processed': 0,
            'quality_distribution': {level.value: 0 for level in QualityLevel},
            'errors': 0
        }
    
    def _create_transforms(self) -> Dict[str, A.Compose]:
        """创建数据增强变换"""
        base_transforms = [
            A.Resize(*self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
        ]
        
        transforms = {
            'train': A.Compose([
                A.Resize(*self.image_size),
                
                # 几何变换
                A.RandomRotate90(p=0.3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomScale(scale_limit=0.1, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                                 rotate_limit=15, p=0.3),
                
                # 颜色变换
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                         contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, 
                             saturation=0.2, hue=0.1, p=0.3),
                A.HueSaturationValue(p=0.3),
                
                # 噪声和模糊
                A.GaussianBlur(p=0.2),
                A.GaussNoise(p=0.1),
                A.MotionBlur(p=0.1),
                
                # 高级增强
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, 
                               min_holes=1, fill_value=0, p=0.2),
                A.GridDistortion(p=0.1),
                A.OpticalDistortion(p=0.1),
                
                # 标准化
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ]),
            
            'val': A.Compose(base_transforms),
            
            'test': A.Compose(base_transforms),
            
            'inference': A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ])
        }
        
        return transforms
    
    def _create_auto_augment(self) -> A.Compose:
        """创建自动增强策略"""
        if not self.enable_auto_augment:
            return None
            
        return A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ], p=0.7),
            
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.ColorJitter(p=1),
                A.HueSaturationValue(p=1),
            ], p=0.5),
            
            A.OneOf([
                A.GaussianBlur(p=1),
                A.GaussNoise(p=1),
                A.MotionBlur(p=1),
            ], p=0.3),
        ])
    
    def process_image(self, 
                     image_path: str, 
                     mode: ProcessingMode = ProcessingMode.TRAIN,
                     return_metrics: bool = True) -> Union[np.ndarray, ProcessingResult]:
        """处理单张图像"""
        start_time = time.time()
        warnings_list = []
        
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 转换颜色空间
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 质量检查
            if return_metrics:
                metrics = self.check_image_quality(image)
                quality_level = self._assess_quality_level(metrics, warnings_list)
            else:
                metrics = {}
                quality_level = QualityLevel.GOOD
            
            # 应用变换
            if mode == ProcessingMode.TRAIN and self.auto_augment:
                # 应用自动增强
                augmented = self.auto_augment(image=image)
                image = augmented['image']
            
            transformed = self.transforms[mode.value](image=image)
            processed_image = transformed['image']
            
            processing_time = time.time() - start_time
            
            # 更新统计
            self.processing_stats['total_processed'] += 1
            self.processing_stats['quality_distribution'][quality_level.value] += 1
            
            if return_metrics:
                return ProcessingResult(
                    image=processed_image,
                    metrics=metrics,
                    quality_level=quality_level,
                    processing_time=processing_time,
                    warnings=warnings_list
                )
            else:
                return processed_image
                
        except Exception as e:
            self.processing_stats['errors'] += 1
            logger.error(f"处理图像失败 {image_path}: {e}")
            raise
    
    def check_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """检查图像质量"""
        metrics = {}
        
        # 基础指标
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 清晰度（拉普拉斯方差）
        metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 亮度
        metrics['brightness'] = np.mean(image)
        
        # 对比度（标准差）
        metrics['contrast'] = np.std(image)
        
        # 曝光度
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        metrics['exposure'] = np.mean(hsv[:, :, 2])
        
        # 饱和度
        metrics['saturation'] = np.mean(hsv[:, :, 1])
        
        # 检测空泡
        lower_bubble = np.array([0, 0, 200])
        upper_bubble = np.array([180, 30, 255])
        bubble_mask = cv2.inRange(hsv, lower_bubble, upper_bubble)
        metrics['bubble_ratio'] = np.sum(bubble_mask > 0) / bubble_mask.size
        
        # 噪声水平（使用局部标准差）
        kernel_size = 3
        local_std = cv2.filter2D(gray.astype(float), -1, 
                               np.ones((kernel_size, kernel_size)) / (kernel_size ** 2))
        metrics['noise_level'] = np.std(local_std - gray.astype(float))
        
        # 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        metrics['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 模糊检测
        metrics['blur_score'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return metrics
    
    def _assess_quality_level(self, 
                             metrics: Dict[str, float], 
                             warnings: List[str]) -> QualityLevel:
        """评估质量等级"""
        score = 0
        max_score = 0
        
        # 清晰度评分
        sharpness = metrics['sharpness']
        if sharpness >= self.quality_thresholds['sharpness']['excellent']:
            score += 3
        elif sharpness >= self.quality_thresholds['sharpness']['good']:
            score += 2
        elif sharpness >= self.quality_thresholds['sharpness']['acceptable']:
            score += 1
        max_score += 3
        
        # 亮度评分
        brightness = metrics['brightness']
        if (self.quality_thresholds['brightness']['min'] <= brightness <= 
            self.quality_thresholds['brightness']['max']):
            score += 2
        else:
            warnings.append(f"亮度异常: {brightness:.1f}")
        max_score += 2
        
        # 对比度评分
        contrast = metrics['contrast']
        if contrast >= self.quality_thresholds['contrast']['excellent']:
            score += 2
        elif contrast >= self.quality_thresholds['contrast']['min']:
            score += 1
        else:
            warnings.append(f"对比度过低: {contrast:.1f}")
        max_score += 2
        
        # 空泡评分
        bubble_ratio = metrics['bubble_ratio']
        if bubble_ratio <= 0.1:
            score += 3
        elif bubble_ratio <= self.quality_thresholds['bubble_ratio']['max']:
            score += 1
        else:
            warnings.append(f"空泡比例过高: {bubble_ratio:.2%}")
        max_score += 3
        
        # 确定等级
        quality_ratio = score / max_score
        if quality_ratio >= 0.8:
            return QualityLevel.EXCELLENT
        elif quality_ratio >= 0.6:
            return QualityLevel.GOOD
        elif quality_ratio >= 0.4:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR
    
    def process_batch(self, 
                     image_paths: List[str],
                     mode: ProcessingMode = ProcessingMode.TRAIN,
                     parallel: bool = True) -> List[ProcessingResult]:
        """批量处理图像"""
        if parallel and len(image_paths) > 10:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self.process_image, path, mode, True) 
                          for path in image_paths]
                results = [future.result() for future in futures]
        else:
            results = [self.process_image(path, mode, True) for path in image_paths]
        
        return results
    
    def analyze_dataset(self, data_dir: str) -> DatasetStats:
        """分析整个数据集"""
        data_dir = Path(data_dir)
        all_images = []
        class_counts = {'negative': 0, 'positive': 0}
        
        # 收集所有图像
        for class_name in ['negative', 'positive']:
            class_dir = data_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*'))
                images = [img for img in images 
                         if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                all_images.extend([(str(img), class_name) for img in images])
                class_counts[class_name] = len(images)
        
        # 批量处理
        image_paths = [item[0] for item in all_images]
        results = self.process_batch(image_paths, ProcessingMode.TEST)
        
        # 统计质量分布
        quality_dist = {level.value: 0 for level in QualityLevel}
        avg_metrics = {}
        
        for result in results:
            quality_dist[result.quality_level.value] += 1
            
            for metric_name, value in result.metrics.items():
                if metric_name not in avg_metrics:
                    avg_metrics[metric_name] = []
                avg_metrics[metric_name].append(value)
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in avg_metrics.items()}
        
        return DatasetStats(
            total_images=len(all_images),
            class_distribution=class_counts,
            quality_distribution=quality_dist,
            average_metrics=avg_metrics,
            processing_errors=self.processing_stats['errors']
        )
    
    def create_balanced_splits(self, 
                             data_dir: str,
                             val_ratio: float = 0.2,
                             test_ratio: float = 0.1,
                             random_state: int = 42) -> Dict[str, List[str]]:
        """创建平衡的数据划分"""
        data_dir = Path(data_dir)
        
        # 收集数据
        negative_paths = list((data_dir / 'negative').glob('*'))
        positive_paths = list((data_dir / 'positive').glob('*'))
        
        # 过滤有效图像
        def filter_images(paths):
            return [str(p) for p in paths 
                   if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        negative_paths = filter_images(negative_paths)
        positive_paths = filter_images(positive_paths)
        
        # 首先划分测试集
        neg_remaining, neg_test = train_test_split(
            negative_paths, test_size=test_ratio, random_state=random_state
        )
        pos_remaining, pos_test = train_test_split(
            positive_paths, test_size=test_ratio, random_state=random_state
        )
        
        # 然后划分验证集
        val_adjusted = val_ratio / (1 - test_ratio)
        neg_train, neg_val = train_test_split(
            neg_remaining, test_size=val_adjusted, random_state=random_state
        )
        pos_train, pos_val = train_test_split(
            pos_remaining, test_size=val_adjusted, random_state=random_state
        )
        
        return {
            'train': neg_train + pos_train,
            'val': neg_val + pos_val,
            'test': neg_test + pos_test
        }
    
    def export_quality_report(self, 
                            data_dir: str,
                            output_path: str,
                            format: str = 'json'):
        """导出质量报告"""
        stats = self.analyze_dataset(data_dir)
        
        report = {
            'dataset_stats': {
                'total_images': stats.total_images,
                'class_distribution': stats.class_distribution,
                'quality_distribution': stats.quality_distribution,
                'average_metrics': stats.average_metrics,
                'processing_errors': stats.processing_errors
            },
            'quality_thresholds': self.quality_thresholds,
            'processing_stats': self.processing_stats
        }
        
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        else:
            # 生成HTML报告
            self._generate_html_report(report, output_path)
        
        logger.info(f"质量报告已导出到: {output_path}")
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: str):
        """生成HTML质量报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>数据集质量报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 5px 0; }}
                .quality-excellent {{ color: green; }}
                .quality-good {{ color: blue; }}
                .quality-acceptable {{ color: orange; }}
                .quality-poor {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>数据集质量报告</h1>
            
            <div class="section">
                <h2>基本信息</h2>
                <p>总图像数: {report['dataset_stats']['total_images']}</p>
                <p>处理错误: {report['dataset_stats']['processing_errors']}</p>
            </div>
            
            <div class="section">
                <h2>类别分布</h2>
        """
        
        for class_name, count in report['dataset_stats']['class_distribution'].items():
            html_content += f"<p>{class_name}: {count}</p>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>质量分布</h2>
        """
        
        for quality_level, count in report['dataset_stats']['quality_distribution'].items():
            html_content += f'<p class="quality-{quality_level}">{quality_level}: {count}</p>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def create_dataset(self, 
                      data_dir: str,
                      mode: ProcessingMode = ProcessingMode.TRAIN,
                      quality_filter: Optional[QualityLevel] = None) -> Dataset:
        """创建数据集"""
        return BioAstDataset(data_dir, self, mode, quality_filter)
    
    def create_dataloader(self,
                        dataset: Dataset,
                        batch_size: int = 32,
                        shuffle: bool = True,
                        num_workers: Optional[int] = None) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers or self.num_workers,
            pin_memory=True,
            persistent_workers=True if (num_workers or self.num_workers) > 0 else False
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """获取处理摘要"""
        return {
            'total_processed': self.processing_stats['total_processed'],
            'quality_distribution': self.processing_stats['quality_distribution'],
            'error_rate': (self.processing_stats['errors'] / 
                          max(1, self.processing_stats['total_processed'])),
            'image_size': self.image_size,
            'quality_thresholds': self.quality_thresholds,
            'auto_augment_enabled': self.enable_auto_augment
        }


class BioAstDataset(Dataset):
    """生物医学数据集"""
    
    def __init__(self, 
                 data_dir: str, 
                 processor: BioAstDataProcessor, 
                 mode: ProcessingMode = ProcessingMode.TRAIN,
                 quality_filter: Optional[QualityLevel] = None):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.mode = mode
        self.quality_filter = quality_filter
        self.samples = self._load_samples()
        
        # 缓存处理结果
        self.cache = {}
        self.use_cache = mode != ProcessingMode.TRAIN
    
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
        
        # 检查缓存
        if self.use_cache and img_path in self.cache:
            image = self.cache[img_path]
        else:
            # 处理图像
            result = self.processor.process_image(img_path, self.mode, True)
            
            # 质量过滤
            if (self.quality_filter and 
                result.quality_level.value < self.quality_filter.value):
                # 如果质量不达标，返回一个零张量
                image = np.zeros((*self.processor.image_size, 3), dtype=np.float32)
            else:
                image = result.image
            
            # 缓存结果
            if self.use_cache:
                self.cache[img_path] = image
        
        # 转换为张量
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, label
    
    def get_quality_stats(self) -> Dict[str, int]:
        """获取数据集质量统计"""
        if not hasattr(self, '_quality_stats'):
            self._quality_stats = {level.value: 0 for level in QualityLevel}
            
            # 评估前100个样本的质量分布
            for i in range(min(100, len(self))):
                img_path, _ = self.samples[i]
                result = self.processor.process_image(img_path, self.mode, True)
                self._quality_stats[result.quality_level.value] += 1
        
        return self._quality_stats


class DataPipeline:
    """数据处理管道"""
    
    def __init__(self, 
                 data_dir: str,
                 processor: Optional[BioAstDataProcessor] = None,
                 auto_split: bool = True,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1):
        self.data_dir = Path(data_dir)
        self.processor = processor or BioAstDataProcessor()
        self.auto_split = auto_split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # 数据集
        self.datasets = {}
        self.dataloaders = {}
        
        # 初始化
        if auto_split:
            self._setup_datasets()
    
    def _setup_datasets(self):
        """设置数据集"""
        # 创建平衡划分
        splits = self.processor.create_balanced_splits(
            str(self.data_dir),
            self.val_ratio,
            self.test_ratio
        )
        
        # 为每个划分创建临时目录
        temp_dirs = {}
        for split_name, paths in splits.items():
            temp_dir = self.data_dir.parent / f'temp_{split_name}'
            temp_dirs[split_name] = temp_dir
            
            # 创建目录结构
            for class_name in ['negative', 'positive']:
                (temp_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            for img_path in paths:
                img_path = Path(img_path)
                class_name = img_path.parent.name
                dest_path = temp_dir / class_name / img_path.name
                if not dest_path.exists():
                    import shutil
                    shutil.copy2(img_path, dest_path)
        
        # 创建数据集
        for split_name in ['train', 'val', 'test']:
            mode = ProcessingMode.TRAIN if split_name == 'train' else ProcessingMode.VAL
            self.datasets[split_name] = self.processor.create_dataset(
                str(temp_dirs[split_name]), 
                mode
            )
    
    def get_dataloader(self, 
                      split: str,
                      batch_size: int = 32,
                      shuffle: Optional[bool] = None) -> DataLoader:
        """获取数据加载器"""
        if split not in self.datasets:
            raise ValueError(f"未知的数据划分: {split}")
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        if split not in self.dataloaders:
            self.dataloaders[split] = self.processor.create_dataloader(
                self.datasets[split],
                batch_size=batch_size,
                shuffle=shuffle
            )
        
        return self.dataloaders[split]
    
    def analyze_all_datasets(self) -> Dict[str, DatasetStats]:
        """分析所有数据集"""
        stats = {}
        
        if self.auto_split:
            temp_dirs = {
                'train': self.data_dir.parent / 'temp_train',
                'val': self.data_dir.parent / 'temp_val',
                'test': self.data_dir.parent / 'temp_test'
            }
            
            for split_name, temp_dir in temp_dirs.items():
                if temp_dir.exists():
                    stats[split_name] = self.processor.analyze_dataset(str(temp_dir))
        else:
            stats['all'] = self.processor.analyze_dataset(str(self.data_dir))
        
        return stats
    
    def cleanup(self):
        """清理临时文件"""
        if self.auto_split:
            for split_name in ['train', 'val', 'test']:
                temp_dir = self.data_dir.parent / f'temp_{split_name}'
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)


# 工厂函数
def create_data_processor(image_size: tuple = (70, 70),
                         quality_thresholds: Optional[Dict[str, float]] = None,
                         enable_auto_augment: bool = True,
                         num_workers: int = 4) -> BioAstDataProcessor:
    """创建数据处理器"""
    return BioAstDataProcessor(
        image_size=image_size,
        quality_thresholds=quality_thresholds,
        enable_auto_augment=enable_auto_augment,
        num_workers=num_workers
    )


def create_data_pipeline(data_dir: str,
                        image_size: tuple = (70, 70),
                        auto_split: bool = True,
                        val_ratio: float = 0.2,
                        test_ratio: float = 0.1) -> DataPipeline:
    """创建数据处理管道"""
    return DataPipeline(
        data_dir=data_dir,
        processor=create_data_processor(image_size),
        auto_split=auto_split,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
