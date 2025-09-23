"""
增强多任务模型训练脚本
支持多种网络架构的训练、验证和保存
基于新的数据集和改进的模型架构
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp as amp

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据集和模型
from training.enhanced_multitask_dataset import create_multitask_dataloaders
from models.enhanced_efficientnet_b0_multitask import create_enhanced_efficientnet_b0_multitask

class MultiTaskTrainer:
    """多任务模型训练器"""
    
    def __init__(self, 
                 model_name: str,
                 model: nn.Module,
                 dataloaders: Dict[str, DataLoader],
                 config: Dict,
                 experiment_dir: str):
        self.model_name = model_name
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = amp.GradScaler() if config.get('use_amp', False) else None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # 日志记录
        self.train_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.task_history = {task: {'train_loss': [], 'val_loss': [], 'val_acc': []} 
                           for task in ['growth_level', 'growth_pattern', 'interference_factors', 'microbe_type']}
        
        # TensorBoard
        self.writer = SummaryWriter(self.experiment_dir / 'tensorboard')
        
        print(f"✓ 训练器初始化完成")
        print(f"  模型: {model_name}")
        print(f"  设备: {self.device}")
        print(f"  实验目录: {self.experiment_dir}")
        print(f"  混合精度: {'启用' if self.scaler else '禁用'}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        if self.config.get('optimizer', 'adamw').lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-4),
                betas=(0.9, 0.999)
            )
        elif self.config.get('optimizer', 'adamw').lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 1e-3),
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.get('scheduler', 'cosine').lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('learning_rate', 1e-4) * 0.01
            )
        elif self.config.get('scheduler', 'cosine').lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=self.config.get('lr_patience', 10),
                verbose=True
            )
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_history.keys()}
        task_corrects = {task: 0 for task in self.task_history.keys()}
        task_totals = {task: 0 for task in self.task_history.keys()}
        
        num_batches = len(self.dataloaders['train'])
        
        for batch_idx, (images, targets) in enumerate(self.dataloaders['train']):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.scaler:
                with amp.autocast():
                    outputs = self.model(images)
                    loss, individual_losses = self.model.compute_loss(outputs, targets, self.current_epoch)
            else:
                outputs = self.model(images)
                loss, individual_losses = self.model.compute_loss(outputs, targets, self.current_epoch)
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算各任务准确率
            with torch.no_grad():
                for task in task_losses.keys():
                    if task in individual_losses:
                        task_losses[task] += individual_losses[task].item()
                    
                    if task in outputs and task in targets:
                        if task == 'interference_factors':
                            # 多标签任务
                            preds = (torch.sigmoid(outputs[task]) > 0.5).float()
                            correct = (preds == targets[task]).all(dim=1).sum().item()
                        else:
                            # 单标签任务
                            preds = torch.argmax(outputs[task], dim=1)
                            correct = (preds == targets[task]).sum().item()
                        
                        task_corrects[task] += correct
                        task_totals[task] += images.size(0)
            
            # 进度显示
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # 计算平均损失和准确率
        avg_loss = total_loss / num_batches
        task_metrics = {}
        for task in task_losses.keys():
            task_metrics[task] = {
                'loss': task_losses[task] / num_batches,
                'acc': task_corrects[task] / max(task_totals[task], 1) * 100
            }
        
        return avg_loss, task_metrics
    
    def validate(self) -> Tuple[float, Dict[str, float], float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.task_history.keys()}
        task_corrects = {task: 0 for task in self.task_history.keys()}
        task_totals = {task: 0 for task in self.task_history.keys()}
        
        num_batches = len(self.dataloaders['val'])
        
        with torch.no_grad():
            for images, targets in self.dataloaders['val']:
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(images)
                loss, individual_losses = self.model.compute_loss(outputs, targets, self.current_epoch)
                
                total_loss += loss.item()
                
                # 计算各任务准确率
                for task in task_losses.keys():
                    if task in individual_losses:
                        task_losses[task] += individual_losses[task].item()
                    
                    if task in outputs and task in targets:
                        if task == 'interference_factors':
                            # 多标签任务
                            preds = (torch.sigmoid(outputs[task]) > 0.5).float()
                            correct = (preds == targets[task]).all(dim=1).sum().item()
                        else:
                            # 单标签任务
                            preds = torch.argmax(outputs[task], dim=1)
                            correct = (preds == targets[task]).sum().item()
                        
                        task_corrects[task] += correct
                        task_totals[task] += images.size(0)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / num_batches
        task_metrics = {}
        overall_acc = 0.0
        
        for task in task_losses.keys():
            acc = task_corrects[task] / max(task_totals[task], 1) * 100
            task_metrics[task] = {
                'loss': task_losses[task] / num_batches,
                'acc': acc
            }
            overall_acc += acc
        
        overall_acc /= len(task_losses)
        
        return avg_loss, task_metrics, overall_acc
    
    def train(self):
        """完整训练流程"""
        print(f"\n开始训练 {self.model_name}")
        print(f"训练参数: {self.config}")
        
        start_time = time.time()
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 15)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_metrics = self.train_epoch()
            
            # 验证
            val_loss, val_metrics, val_acc = self.validate()
            
            # 更新学习率
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 记录历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            for task in self.task_history.keys():
                if task in train_metrics:
                    self.task_history[task]['train_loss'].append(train_metrics[task]['loss'])
                    self.task_history[task]['val_loss'].append(val_metrics[task]['loss'])
                    self.task_history[task]['val_acc'].append(val_metrics[task]['acc'])
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Val_Overall', val_acc, epoch)
            
            for task in train_metrics.keys():
                self.writer.add_scalar(f'Loss/Train_{task}', train_metrics[task]['loss'], epoch)
                self.writer.add_scalar(f'Loss/Val_{task}', val_metrics[task]['loss'], epoch)
                self.writer.add_scalar(f'Accuracy/Val_{task}', val_metrics[task]['acc'], epoch)
            
            # 显示结果
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证准确率: {val_acc:.2f}%")
            print("各任务准确率:")
            for task, metrics in val_metrics.items():
                print(f"  {task}: {metrics['acc']:.2f}%")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint('best')
                print(f"✓ 保存最佳模型 (准确率: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= patience:
                print(f"\n早停触发 (patience: {patience})")
                break
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'epoch_{epoch+1}')
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n训练完成!")
        print(f"训练时间: {training_time:.2f}秒")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        
        # 保存最终模型和历史
        self._save_checkpoint('final')
        self._save_training_history()
        self.writer.close()
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'epochs_trained': self.current_epoch + 1
        }
    
    def _save_checkpoint(self, suffix: str):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.experiment_dir / f'{self.model_name}_{suffix}.pth')
    
    def _save_training_history(self):
        """保存训练历史"""
        history = {
            'train_history': self.train_history,
            'task_history': self.task_history,
            'config': self.config,
            'model_name': self.model_name
        }
        
        with open(self.experiment_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)


def create_model(model_name: str, num_classes: Dict[str, int]) -> nn.Module:
    """创建指定的模型"""
    if model_name.lower() == 'enhanced_efficientnet_b0':
        return create_enhanced_efficientnet_b0_multitask(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='增强多任务模型训练')
    parser.add_argument('--model', type=str, default='enhanced_efficientnet_b0',
                       choices=['enhanced_efficientnet_b0'], 
                       help='模型架构')
    parser.add_argument('--data_root', type=str, 
                       default='/home/aaa/ws/bioastModel/ds/images',
                       help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--patience', type=int, default=15, help='早停patience')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--experiment_name', type=str, default=None, help='实验名称')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'use_amp': args.use_amp,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'num_workers': args.num_workers
    }
    
    # 类别数配置
    num_classes = {
        'growth_level': 2,
        'growth_pattern': 12, 
        'interference_factors': 4,
        'microbe_type': 4
    }
    
    print("=" * 60)
    print("增强多任务模型训练")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据集: {args.data_root}")
    print(f"配置: {config}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    dataloaders = create_multitask_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
    model = create_model(args.model, num_classes)
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        experiment_name = f"{args.experiment_name}_{timestamp}"
    else:
        experiment_name = f"{args.model}_multitask_{timestamp}"
    
    experiment_dir = f"experiments/{experiment_name}"
    
    # 创建训练器
    trainer = MultiTaskTrainer(
        model_name=args.model,
        model=model,
        dataloaders=dataloaders,
        config=config,
        experiment_dir=experiment_dir
    )
    
    # 开始训练
    results = trainer.train()
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳验证准确率: {results['best_val_acc']:.2f}%")
    print(f"训练时间: {results['training_time']:.2f}秒")
    print(f"实验目录: {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()