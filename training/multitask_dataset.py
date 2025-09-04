"""
多任务生物图像数据集加载器
支持生长级别、生长模式、干扰因素和精细分类的多标签数据
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging


class MultitaskBioastDataset(Dataset):
    """多任务生物图像数据集"""
    
    def __init__(self, 
                 annotation_file: str, 
                 image_root: str,
                 split: str = None,
                 transform: Optional[transforms.Compose] = None,
                 augment: bool = False):
        """
        Args:
            annotation_file: JSON标注文件路径
            image_root: 图像根目录
            split: 数据分割 ('train', 'val', 'test')
            transform: 图像变换
            augment: 是否使用数据增强
        """
        self.annotation_file = annotation_file
        self.image_root = Path(image_root)
        self.transform = transform
        self.augment = augment
        self.split = split
        
        # 类别映射
        self.mappings = self._create_mappings()
        
        # 加载标注数据
        self.annotations = self._load_annotations()
        
        # 数据统计
        self.stats = self._calculate_stats()
        
        print(f"加载多任务数据集: {len(self.annotations)} 个样本")
        self._print_stats()
    
    def _create_mappings(self) -> Dict[str, Dict]:
        """创建类别映射"""
        mappings = {
            'growth_level': {
                'negative': 0,
                'positive': 1,
                'weak_growth': 2
            },
            'growth_pattern': {
                'clean': 0,
                'clustered': 1,
                'scattered': 2,
                'heavy_growth': 3,
                'small_dots': 4,
                'irregular_areas': 5,
                'light_gray': 6,
                'default_positive': 7,
                'default_weak_growth': 8
            },
            'interference_mapping': {
                'pores': 0,
                'debris': 1,
                'artifacts': 2
            },
            'fine_grained': self._generate_fine_grained_mapping()
        }
        
        # 创建反向映射
        for task_name in mappings.keys():
            if task_name != 'fine_grained':
                mappings[f'{task_name}_reverse'] = {
                    v: k for k, v in mappings[task_name].items()
                }
        
        return mappings
    
    def _generate_fine_grained_mapping(self) -> Dict[str, int]:
        """生成40个精细类别的映射"""
        mapping = {}
        idx = 0
        
        # 阴性样本变体 (3类)
        mapping['negative_clean'] = idx
        mapping['negative_pores'] = idx + 1
        mapping['negative_debris'] = idx + 2
        idx += 3
        
        # 阳性聚集型变体 (9类)
        for interference in ['pores', 'debris', 'artifacts', 'none', 'multiple']:
            mapping[f'positive_clustered_{interference}'] = idx
            idx += 1
        
        # 阳性分散型变体 (3类)
        mapping['positive_scattered_pores'] = idx
        mapping['positive_scattered_debris'] = idx + 1
        mapping['positive_scattered_artifacts'] = idx + 2
        idx += 3
        
        # 重度生长变体 (3类)
        mapping['heavy_growth_pores'] = idx
        mapping['heavy_growth_debris'] = idx + 1
        mapping['heavy_growth_artifacts'] = idx + 2
        idx += 3
        
        # 弱生长小点型变体 (4类)
        mapping['weak_growth_small_dots_pores'] = idx
        mapping['weak_growth_small_dots_debris'] = idx + 1
        mapping['weak_growth_small_dots_artifacts'] = idx + 2
        mapping['weak_growth_small_dots_clean'] = idx + 3
        idx += 3
        
        # 不规则区域变体 (6类)
        for interference in ['pores', 'debris', 'artifacts', 'none']:
            mapping[f'irregular_areas_{interference}'] = idx
            idx += 1
        
        # 浅灰色变体 (6类)
        for interference in ['pores', 'debris', 'artifacts', 'none']:
            mapping[f'light_gray_{interference}'] = idx
            idx += 1
        
        # 其他组合填充到40类
        while idx < 40:
            mapping[f'combination_{idx}'] = idx
            idx += 1
        
        return mapping
    
    def _load_annotations(self) -> List[Dict]:
        """加载标注数据"""
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"标注文件不存在: {self.annotation_file}")
        
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 过滤指定分割的数据
        if self.split:
            annotations = [ann for ann in annotations if ann.get('split') == self.split]
        
        # 验证数据完整性
        validated_annotations = []
        for ann in annotations:
            if self._validate_annotation(ann):
                validated_annotations.append(ann)
            else:
                logging.warning(f"跳过无效标注: {ann.get('image_id', 'unknown')}")
        
        return validated_annotations
    
    def _validate_annotation(self, annotation: Dict) -> bool:
        """验证标注数据的有效性"""
        required_fields = ['image_id', 'file_path', 'annotations']
        
        # 检查必需字段
        for field in required_fields:
            if field not in annotation:
                return False
        
        # 检查图像文件是否存在
        image_path = self.image_root / annotation['file_path']
        if not image_path.exists():
            return False
        
        # 检查标注完整性
        ann_data = annotation['annotations']
        required_tasks = ['growth_level', 'growth_pattern', 'interference_mapping']
        
        for task in required_tasks:
            if task not in ann_data:
                return False
        
        return True
    
    def _calculate_stats(self) -> Dict:
        """计算数据集统计信息"""
        stats = {
            'total_samples': len(self.annotations),
            'tasks': {}
        }
        
        # 初始化统计结构
        for task_name in ['growth_level', 'growth_pattern', 'interference_mapping', 'fine_grained']:
            if task_name == 'interference_mapping':
                stats['tasks'][task_name] = {
                    'label_counts': [0, 0, 0],
                    'multilabel_samples': 0,
                    'avg_labels_per_sample': 0.0
                }
            else:
                num_classes = len(self.mappings[task_name])
                stats['tasks'][task_name] = {
                    'class_counts': [0] * num_classes,
                    'class_distribution': {}
                }
        
        # 统计各类别数量
        for ann in self.annotations:
            ann_data = ann['annotations']
            
            # 生长级别
            gl_label = self.mappings['growth_level'][ann_data['growth_level']]
            stats['tasks']['growth_level']['class_counts'][gl_label] += 1
            
            # 生长模式
            gp_label = self.mappings['growth_pattern'][ann_data['growth_pattern']]
            stats['tasks']['growth_pattern']['class_counts'][gp_label] += 1
            
            # 干扰因素（多标签）
            interference_labels = [0] * 3
            for interference in ann_data['interference_mapping']:
                if interference in self.mappings['interference_mapping']:
                    label_idx = self.mappings['interference_mapping'][interference]
                    interference_labels[label_idx] = 1
                    stats['tasks']['interference_mapping']['label_counts'][label_idx] += 1
            
            num_labels = sum(interference_labels)
            if num_labels > 1:
                stats['tasks']['interference_mapping']['multilabel_samples'] += 1
            
            # 精细分类
            fg_key = self._generate_fine_grained_key(ann_data)
            if fg_key in self.mappings['fine_grained']:
                fg_label = self.mappings['fine_grained'][fg_key]
                stats['tasks']['fine_grained']['class_counts'][fg_label] += 1
        
        # 计算分布和百分比
        for task_name, task_stats in stats['tasks'].items():
            if task_name != 'interference_mapping':
                total = sum(task_stats['class_counts'])
                for i, count in enumerate(task_stats['class_counts']):
                    task_stats['class_distribution'][i] = {
                        'count': count,
                        'percentage': count / total * 100 if total > 0 else 0
                    }
            else:
                total_samples = len(self.annotations)
                total_labels = sum(task_stats['label_counts'])
                task_stats['avg_labels_per_sample'] = total_labels / total_samples if total_samples > 0 else 0
        
        return stats
    
    def _generate_fine_grained_key(self, ann_data: Dict) -> str:
        """根据标注生成精细分类的键"""
        gl = ann_data['growth_level']
        gp = ann_data['growth_pattern']
        im = ann_data['interference_mapping']
        
        # 简化的组合逻辑
        if gl == 'negative':
            interference = im[0] if im else 'clean'
            return f'negative_{interference}'
        elif gl == 'positive':
            if gp in ['clustered', 'scattered']:
                interference = im[0] if im else 'none'
                return f'positive_{gp}_{interference}'
            else:
                return f'{gl}_{gp}'
        else:  # weak_growth
            return f'{gl}_{gp}'
    
    def _print_stats(self):
        """打印数据集统计信息"""
        print("\n=== 数据集统计 ===")
        print(f"总样本数: {self.stats['total_samples']}")
        
        for task_name, task_stats in self.stats['tasks'].items():
            print(f"\n{task_name}:")
            
            if task_name == 'interference_mapping':
                print(f"  - 标签分布: {task_stats['label_counts']}")
                print(f"  - 多标签样本: {task_stats['multilabel_samples']}")
                print(f"  - 平均标签数/样本: {task_stats['avg_labels_per_sample']:.2f}")
            else:
                print(f"  - 类别数: {len(task_stats['class_counts'])}")
                for class_idx, dist in task_stats['class_distribution'].items():
                    print(f"  - 类别 {class_idx}: {dist['count']} 样本 ({dist['percentage']:.1f}%)")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        annotation = self.annotations[idx]
        
        # 加载图像
        image_path = self.image_root / annotation['file_path']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"加载图像失败 {image_path}: {e}")
            # 返回黑色图像作为备用
            image = Image.new('RGB', (70, 70), (0, 0, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 准备多任务标签
        labels = self._prepare_labels(annotation['annotations'])
        
        return image, labels
    
    def _prepare_labels(self, ann_data: Dict) -> Dict[str, torch.Tensor]:
        """准备多任务标签"""
        labels = {}
        
        # 生长级别 (单标签)
        labels['growth_level'] = torch.tensor(
            self.mappings['growth_level'][ann_data['growth_level']],
            dtype=torch.long
        )
        
        # 生长模式 (单标签)
        labels['growth_pattern'] = torch.tensor(
            self.mappings['growth_pattern'][ann_data['growth_pattern']],
            dtype=torch.long
        )
        
        # 干扰因素 (多标签)
        interference_labels = torch.zeros(3, dtype=torch.float)
        for interference in ann_data['interference_mapping']:
            if interference in self.mappings['interference_mapping']:
                label_idx = self.mappings['interference_mapping'][interference]
                interference_labels[label_idx] = 1.0
        labels['interference_mapping'] = interference_labels
        
        # 精细分类 (单标签)
        fg_key = self._generate_fine_grained_key(ann_data)
        fg_label = self.mappings['fine_grained'].get(fg_key, 0)  # 默认为0
        labels['fine_grained'] = torch.tensor(fg_label, dtype=torch.long)
        
        return labels
    
    def get_class_weights(self, task_name: str) -> torch.Tensor:
        """获取类别权重用于处理不平衡数据"""
        if task_name == 'interference_mapping':
            # 多标签任务的权重
            pos_counts = np.array(self.stats['tasks'][task_name]['label_counts'])
            neg_counts = len(self.annotations) - pos_counts
            
            # 计算正负样本权重
            pos_weights = neg_counts / (pos_counts + 1e-6)
            return torch.tensor(pos_weights, dtype=torch.float)
        else:
            # 单标签任务的权重
            class_counts = np.array(self.stats['tasks'][task_name]['class_counts'])
            total = len(self.annotations)
            
            # 计算各类别权重
            weights = total / (len(class_counts) * (class_counts + 1e-6))
            return torch.tensor(weights, dtype=torch.float)
    
    def get_task_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        return {
            'task_names': ['growth_level', 'growth_pattern', 'interference_mapping', 'fine_grained'],
            'num_classes': {
                'growth_level': 3,
                'growth_pattern': 9,
                'interference_mapping': 3,
                'fine_grained': 40
            },
            'task_types': {
                'growth_level': 'single_label',
                'growth_pattern': 'single_label',
                'interference_mapping': 'multi_label',
                'fine_grained': 'single_label'
            },
            'mappings': self.mappings,
            'stats': self.stats
        }


def get_multitask_transforms(split: str = 'train', image_size: int = 70) -> transforms.Compose:
    """获取多任务数据变换"""
    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_multitask_dataloaders(annotation_file: str,
                               image_root: str,
                               batch_size: int = 32,
                               num_workers: int = 4) -> Dict[str, DataLoader]:
    """创建多任务数据加载器"""
    
    # 创建数据集
    train_dataset = MultitaskBioastDataset(
        annotation_file=annotation_file,
        image_root=image_root,
        split='train',
        transform=get_multitask_transforms('train'),
        augment=True
    )
    
    val_dataset = MultitaskBioastDataset(
        annotation_file=annotation_file,
        image_root=image_root,
        split='val',
        transform=get_multitask_transforms('val'),
        augment=False
    )
    
    test_dataset = MultitaskBioastDataset(
        annotation_file=annotation_file,
        image_root=image_root,
        split='test',
        transform=get_multitask_transforms('test'),
        augment=False
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
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'dataset_info': train_dataset.get_task_info()
    }


# 使用示例
if __name__ == "__main__":
    # 示例用法
    annotation_file = "bioast_dataset/annotations/multitask_annotations.json"
    image_root = "bioast_dataset/images"
    
    # 创建数据加载器
    dataloaders = create_multitask_dataloaders(
        annotation_file=annotation_file,
        image_root=image_root,
        batch_size=16
    )
    
    # 查看数据集信息
    print("\n数据集信息:")
    info = dataloaders['dataset_info']
    print(f"任务名称: {info['task_names']}")
    print(f"类别数量: {info['num_classes']}")
    
    # 测试数据加载
    for batch_idx, (images, labels) in enumerate(dataloaders['train']):
        print(f"\n批次 {batch_idx}:")
        print(f"图像形状: {images.shape}")
        print("标签形状:")
        for task_name, label_tensor in labels.items():
            print(f"  {task_name}: {label_tensor.shape}")
        
        if batch_idx >= 2:  # 只显示前3个批次
            break