#!/usr/bin/env python3
"""
训练修复版MobileNetV3多任务模型
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

from training.enhanced_multitask_dataset import EnhancedMultitaskDataset
from models.fixed_mobilenetv3_multitask import create_fixed_mobilenetv3_multitask

def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

class MobileNetV3Trainer:
    def __init__(self, model, device, train_loader, val_loader, num_classes, 
                 learning_rate=0.001, experiment_dir=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        
        # 优化器和调度器
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 实验目录
        self.experiment_dir = experiment_dir
        if experiment_dir:
            os.makedirs(experiment_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'logs'))
        else:
            self.writer = None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'task_accuracies': {task: [] for task in num_classes.keys()}
        }
        
        # 最佳性能跟踪
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 移动数据到设备
            image = batch['image'].to(self.device)
            targets = {
                task: batch[task].to(self.device) 
                for task in self.num_classes.keys()
            }
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(image)
            
            # 计算损失
            loss, individual_losses = self.model.compute_loss(predictions, targets, epoch)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}")
                
                if self.writer:
                    step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('Loss/Train_Batch', loss.item(), step)
                    for task, task_loss in individual_losses.items():
                        self.writer.add_scalar(f'Loss/Train_{task}', task_loss.item(), step)
        
        avg_loss = total_loss / num_batches
        self.scheduler.step()
        
        return avg_loss
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        task_correct = {task: 0 for task in self.num_classes.keys()}
        task_total = {task: 0 for task in self.num_classes.keys()}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移动数据到设备
                image = batch['image'].to(self.device)
                targets = {
                    task: batch[task].to(self.device) 
                    for task in self.num_classes.keys()
                }
                
                # 前向传播
                predictions = self.model(image)
                
                # 计算损失
                loss, _ = self.model.compute_loss(predictions, targets, epoch)
                total_loss += loss.item()
                num_batches += 1
                
                # 计算准确率
                for task in self.num_classes.keys():
                    if task == 'interference_factors':
                        # 多标签分类
                        pred_binary = (torch.sigmoid(predictions[task]) > 0.5).float()
                        correct = (pred_binary == targets[task].float()).all(dim=1).sum()
                    else:
                        # 单标签分类
                        _, predicted = torch.max(predictions[task], 1)
                        correct = (predicted == targets[task]).sum()
                    
                    task_correct[task] += correct.item()
                    task_total[task] += targets[task].size(0)
        
        # 计算平均损失和准确率
        avg_loss = total_loss / num_batches
        task_accuracies = {
            task: 100.0 * task_correct[task] / task_total[task] 
            for task in self.num_classes.keys()
        }
        overall_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
        
        return avg_loss, overall_accuracy, task_accuracies
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        if not self.experiment_dir:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'train_history': self.train_history
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.experiment_dir, 'latest.pth'))
        
        # 如果是最佳模型，保存副本
        if is_best:
            torch.save(checkpoint, os.path.join(self.experiment_dir, 'best.pth'))
            print(f"New best model saved with accuracy: {self.best_val_accuracy:.4f}%")
    
    def train(self, num_epochs=30):
        """完整训练流程"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_accuracy, task_accuracies = self.validate(epoch)
            
            # 更新历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_accuracy)
            for task, acc in task_accuracies.items():
                self.train_history['task_accuracies'][task].append(acc)
            
            # 记录到TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/Overall', val_accuracy, epoch)
                for task, acc in task_accuracies.items():
                    self.writer.add_scalar(f'Accuracy/{task}', acc, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Overall Accuracy: {val_accuracy:.4f}%")
            print("Task Accuracies:")
            for task, acc in task_accuracies.items():
                print(f"  {task}: {acc:.4f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 检查是否是最佳模型
            is_best = val_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_accuracy
                self.best_epoch = epoch
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 每5个epoch打印最佳结果
            if epoch % 5 == 0:
                print(f"Best accuracy so far: {self.best_val_accuracy:.4f}% (Epoch {self.best_epoch})")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}% (Epoch {self.best_epoch})")
        
        # 保存训练历史
        if self.experiment_dir:
            with open(os.path.join(self.experiment_dir, 'train_history.json'), 'w') as f:
                json.dump(self.train_history, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        return self.train_history

def main():
    parser = argparse.ArgumentParser(description='Train Fixed MobileNetV3 MultiTask Model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data_root', type=str, default='ds/images', help='Data root directory')
    
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device()
    
    # 创建实验目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"fixed_mobilenetv3_multitask_{timestamp}"
    experiment_dir = f"experiments/{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 保存实验配置
    config = {
        'model': 'fixed_mobilenetv3_multitask',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'data_root': args.data_root,
        'experiment_name': experiment_name,
        'timestamp': timestamp
    }
    
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment: {experiment_name}")
    print(f"Configuration: {config}")
    
    # 创建数据集
    print("Creating datasets...")
    train_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='train'
    )
    
    val_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 获取类别数量
    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes: {num_classes}")
    
    # 创建模型
    print("Creating model...")
    model = create_fixed_mobilenetv3_multitask(
        num_classes=num_classes,
        dropout_rate=0.3,
        use_attention=True,
        use_label_smoothing=True,
        freeze_backbone=False,
        use_pretrained=True
    )
    
    model = model.to(device)
    
    # 创建训练器
    trainer = MobileNetV3Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        learning_rate=args.lr,
        experiment_dir=experiment_dir
    )
    
    # 开始训练
    train_history = trainer.train(num_epochs=args.epochs)
    
    print(f"\nTraining complete! Results saved to: {experiment_dir}")

if __name__ == "__main__":
    main()