#!/usr/bin/env python3
"""
Enhanced Multi-Level MobileNetV3 Training Script
使用增强版多级MobileNetV3模型进行训练，包含Growth Pattern类别权重、Focal Loss和Pores专用功能
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_multilevel_mobilenetv3 import (
    EnhancedMultiLevelMobileNetV3, 
    PoresSpecificAugmentation
)
from training.enhanced_multitask_dataset import create_multitask_dataloaders
from utils.metrics import calculate_metrics

class EnhancedTrainer:
    """增强版多级分类训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建实验目录
        self.experiment_dir = Path(config['experiment_dir'])
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self.writer = SummaryWriter(self.experiment_dir / 'tensorboard')
        
        # Growth Pattern类别权重 (0.019-5.219)
        self.growth_pattern_weights = torch.tensor([
            1.000, 1.238, 2.089, 4.068, 7.903, 10.442,
            11.500, 27.364, 192.306, 197.800
        ], device=self.device)
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 初始化数据增强
        self.pores_augmentation = PoresSpecificAugmentation()
        
        # 训练状态
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.training_history = []
        
    def _create_model(self) -> EnhancedMultiLevelMobileNetV3:
        """创建增强版模型"""
        model = EnhancedMultiLevelMobileNetV3(
            model_size=self.config.get('model_size', 'small'),
            input_channels=self.config.get('input_channels', 1),
            dropout_rate=self.config.get('dropout_rate', 0.2),
            use_pores_attention=self.config.get('use_pores_attention', True),
            growth_pattern_weights=self.growth_pattern_weights
        )
        
        model = model.to(self.device)
        
        print(f"✅ 创建增强版模型: {model.get_model_info()['model_name']}")
        print(f"📊 模型参数量: {model.get_model_info()['total_parameters']:,}")
        print(f"🎯 特殊功能: {', '.join(model.get_model_info()['features'])}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.config['num_classes'].keys()}
        task_losses['pores_detection'] = 0.0
        
        num_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            
            # 转换标签格式
            batch_labels = {}
            for task_name in self.config['num_classes'].keys():
                if task_name in labels:
                    if task_name == 'interference_factors':
                        # interference_factors是多标签任务，确保维度正确
                        batch_labels[task_name] = labels[task_name].to(self.device).float()
                    else:
                        batch_labels[task_name] = labels[task_name].to(self.device)
            
            # 添加pores检测标签（基于growth_pattern）
            if 'growth_pattern' in batch_labels:
                # 假设类别11是pores相关的
                pores_labels = (batch_labels['growth_pattern'] == 11).long()
                batch_labels['pores_detection'] = pores_labels
            
            # 应用Pores特定数据增强
            if np.random.random() < 0.3:  # 30%概率应用对比度增强
                images = self.pores_augmentation.enhance_pores_contrast(images)
            if np.random.random() < 0.2:  # 20%概率应用边缘增强
                images = self.pores_augmentation.pores_edge_enhancement(images)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss_dict = self.model.compute_enhanced_loss(outputs, batch_labels)
            total_batch_loss = loss_dict['total']
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累积损失
            total_loss += total_batch_loss.item()
            for task_name, loss_value in loss_dict.items():
                if task_name != 'total' and task_name in task_losses:
                    task_losses[task_name] += loss_value.item()
            
            # 打印进度
            if batch_idx % 50 == 0:
                print(f"Batch [{batch_idx}/{num_batches}] - "
                      f"Loss: {total_batch_loss.item():.4f}")
        
        # 计算平均损失
        avg_losses = {
            'total': total_loss / num_batches,
            **{task: loss / num_batches for task, loss in task_losses.items()}
        }
        
        return avg_losses
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.config['num_classes'].keys()}
        task_losses['pores_detection'] = 0.0
        
        all_predictions = {task: [] for task in self.config['num_classes'].keys()}
        all_predictions['pores_detection'] = []
        all_labels = {task: [] for task in self.config['num_classes'].keys()}
        all_labels['pores_detection'] = []
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                
                # 转换标签格式
                batch_labels = {}
                for task_name in self.config['num_classes'].keys():
                    if task_name in labels:
                        if task_name == 'interference_factors':
                            # interference_factors是多标签任务，确保维度正确
                            batch_labels[task_name] = labels[task_name].to(self.device).float()
                            all_labels[task_name].extend(labels[task_name].cpu().numpy())
                        else:
                            batch_labels[task_name] = labels[task_name].to(self.device)
                            all_labels[task_name].extend(labels[task_name].cpu().numpy())
                
                # 添加pores检测标签
                if 'growth_pattern' in batch_labels:
                    pores_labels = (batch_labels['growth_pattern'] == 11).long()
                    batch_labels['pores_detection'] = pores_labels
                    all_labels['pores_detection'].extend(pores_labels.cpu().numpy())
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                loss_dict = self.model.compute_enhanced_loss(outputs, batch_labels)
                total_loss += loss_dict['total'].item()
                
                for task_name, loss_value in loss_dict.items():
                    if task_name != 'total' and task_name in task_losses:
                        task_losses[task_name] += loss_value.item()
                
                # 收集预测结果
                predictions = self.model.predict(images)
                for task_name, pred in predictions.items():
                    if task_name in all_predictions:
                        all_predictions[task_name].extend(pred.cpu().numpy())
        
        # 计算平均损失
        avg_losses = {
            'total': total_loss / num_batches,
            **{task: loss / num_batches for task, loss in task_losses.items()}
        }
        
        # 计算准确率
        accuracies = {}
        for task_name in all_predictions.keys():
            if len(all_labels[task_name]) > 0:
                predictions_array = np.array(all_predictions[task_name])
                labels_array = np.array(all_labels[task_name])
                
                # 处理不同任务的预测格式
                if task_name == 'growth_level':
                    # 对于growth_level，预测是概率分布，需要取argmax
                    if predictions_array.ndim > 1:
                        predictions_array = np.argmax(predictions_array, axis=1)
                elif task_name == 'growth_pattern':
                    # 对于growth_pattern，预测是概率分布，需要取argmax
                    if predictions_array.ndim > 1:
                        predictions_array = np.argmax(predictions_array, axis=1)
                elif task_name == 'interference_factors':
                    # 对于interference_factors，这是多标签分类，使用阈值0.5
                    if predictions_array.ndim > 1:
                        predictions_array = (predictions_array > 0.5).astype(int)
                    # 计算每个样本的准确匹配
                    if predictions_array.shape == labels_array.shape:
                        correct = np.sum(np.all(predictions_array == labels_array, axis=1))
                        total = len(all_labels[task_name])
                        accuracies[task_name] = correct / total
                        continue
                elif task_name == 'pores_detection':
                    # 对于pores_detection，预测是概率分布，需要取argmax
                    if predictions_array.ndim > 1:
                        predictions_array = np.argmax(predictions_array, axis=1)
                
                # 对于单标签分类任务
                if task_name != 'interference_factors':
                    correct = np.sum(predictions_array == labels_array)
                    total = len(all_labels[task_name])
                    accuracies[task_name] = correct / total
        
        return avg_losses, accuracies
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print(f"🚀 开始训练增强版多级MobileNetV3模型")
        print(f"📱 设备: {self.device}")
        print(f"🔄 训练轮数: {self.config.get('epochs', 100)}")
        print(f"📊 批次大小: {self.config.get('batch_size', 32)}")
        print("-" * 60)
        
        epochs = self.config.get('epochs', 100)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # 训练阶段
            train_losses = self.train_epoch(train_loader)
            
            # 验证阶段
            val_losses, val_accuracies = self.validate_epoch(val_loader)
            
            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    # 使用总体准确率作为指标
                    overall_acc = np.mean(list(val_accuracies.values()))
                    self.scheduler.step(overall_acc)
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 记录训练历史
            epoch_info = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            }
            self.training_history.append(epoch_info)
            
            # 计算总体准确率
            overall_acc = np.mean(list(val_accuracies.values()))
            
            # 打印训练信息
            print(f"Epoch [{epoch+1}/{epochs}] ({epoch_time:.1f}s)")
            print(f"  训练损失: {train_losses['total']:.4f}")
            print(f"  验证损失: {val_losses['total']:.4f}")
            print(f"  总体准确率: {overall_acc:.4f}")
            
            # 打印各任务准确率
            for task_name, acc in val_accuracies.items():
                print(f"    {task_name}: {acc:.4f}")
            
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train_Total', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Val_Total', val_losses['total'], epoch)
            self.writer.add_scalar('Accuracy/Overall', overall_acc, epoch)
            
            for task_name, acc in val_accuracies.items():
                self.writer.add_scalar(f'Accuracy/{task_name}', acc, epoch)
            
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 保存最佳模型
            if overall_acc > self.best_accuracy:
                self.best_accuracy = overall_acc
                self.save_checkpoint(epoch + 1, is_best=True)
                print(f"  🎉 新的最佳准确率: {self.best_accuracy:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, is_best=False)
            
            print("-" * 60)
        
        print(f"✅ 训练完成！最佳准确率: {self.best_accuracy:.4f}")
        
        # 保存训练历史
        self.save_training_history()
        
        # 关闭TensorBoard
        self.writer.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_accuracy': self.best_accuracy,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # 保存常规检查点
        checkpoint_path = self.experiment_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.experiment_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型: {best_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.experiment_dir / 'enhanced_training_history.json'
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        history_data = convert_numpy(self.training_history)
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 保存训练历史: {history_path}")


def main():
    """主函数"""
    # 训练配置
    config = {
        'experiment_dir': 'experiments/enhanced_multilevel_mobilenetv3',
        'data_dir': '/home/aaa/ws/bioastModel/ds/images',  # 更新为正确的数据路径
        'json_path': '/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',  # JSON标注文件路径
        'num_classes': {
            'growth_level': 2,
            'growth_pattern': 10,  # 更新为实际的10个类别
            'interference_factors': 4
        },
        'model_size': 'small',  # 'small' or 'large'
        'input_channels': 1,
        'dropout_rate': 0.2,
        'use_pores_attention': True,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine'
    }
    
    print("🔧 增强版多级MobileNetV3训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 60)
    
    # 创建训练器
    trainer = EnhancedTrainer(config)
    
    # 创建数据加载器
    try:
        dataloaders = create_multitask_dataloaders(
            data_root=config['data_dir'],
            annotations_file='m9e1n170.json',
            batch_size=config['batch_size'],
            num_workers=4,
            split_ratio=(0.7, 0.15, 0.15),
            seed=42
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        print(f"📂 数据加载完成:")
        print(f"  训练样本: {len(train_loader.dataset)}")
        print(f"  验证样本: {len(val_loader.dataset)}")
        print("-" * 60)
        
        # 开始训练
        trainer.train(train_loader, val_loader)
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("💡 请确保数据路径正确并实现了create_multitask_dataloaders函数")
        
        # 创建模拟数据进行测试
        print("🧪 使用模拟数据进行测试...")
        from utils.data_loader import create_synthetic_data_loaders
        
        # 使用已有的模拟数据加载器
        train_loader, val_loader = create_synthetic_data_loaders(
            batch_size=config['batch_size'],
            num_workers=4
        )
        
        print(f"🧪 模拟数据创建完成:")
        print(f"  训练样本: {len(train_loader.dataset)}")
        print(f"  验证样本: {len(val_loader.dataset)}")
        print("-" * 60)
        
        # 开始训练
        trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()