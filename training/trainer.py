"""
模型训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
import os
import json
from collections import defaultdict

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 save_dir: str = './checkpoints'):
        """
        Args:
            model: 要训练的模型
            device: 训练设备
            save_dir: 模型保存目录
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练历史
        self.history = defaultdict(list)
        
        # 最佳模型跟踪
        self.best_val_acc = 0.0
        self.best_model_path = None
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, scheduler_type: str = 'cosine',
              early_stopping_patience: int = 10, class_weights: Optional[torch.Tensor] = None) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            scheduler_type: 学习率调度器类型
            early_stopping_patience: 早停耐心值
            class_weights: 类别权重
        
        Returns:
            训练历史字典
        """
        
        # 设置损失函数
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 设置优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=weight_decay)
        
        # 设置学习率调度器
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
        elif scheduler_type == 'reduce':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=5)
        else:
            scheduler = None
        
        # 早停计数器
        early_stopping_counter = 0
        
        print(f"开始训练，共 {num_epochs} 轮")
        print(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
        print(f"学习率: {learning_rate}, 权重衰减: {weight_decay}")
        print(f"调度器: {scheduler_type}, 早停耐心: {early_stopping_patience}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # 验证阶段
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # 更新学习率
            if scheduler is not None:
                if scheduler_type == 'reduce':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, self.best_model_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # 打印进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查
            if early_stopping_counter >= early_stopping_patience:
                print(f"早停触发！验证准确率在 {early_stopping_patience} 轮内没有改善")
                break
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"训练完成！最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"最佳模型保存在: {self.best_model_path}")
        
        return dict(self.history)
    
    def _train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                    optimizer: optim.Optimizer) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 打印进度（每100个batch）
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def load_best_model(self):
        """加载最佳模型"""
        if self.best_model_path and os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载最佳模型，验证准确率: {checkpoint['val_acc']:.4f}")
        else:
            print("未找到最佳模型文件")
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                       filename: str = 'checkpoint.pth'):
        """保存检查点"""
        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': dict(self.history),
            'best_val_acc': self.best_val_acc
        }, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, filename: str = 'checkpoint.pth') -> int:
        """加载检查点"""
        checkpoint_path = os.path.join(self.save_dir, filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.history = defaultdict(list, checkpoint.get('history', {}))
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"检查点已加载: {checkpoint_path}")
            return checkpoint['epoch']
        else:
            print(f"检查点文件不存在: {checkpoint_path}")
            return 0

if __name__ == "__main__":
    # 测试训练器
    from models.efficientnet import create_efficientnet_b0
    from training.dataset import create_data_loaders
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_efficientnet_b0(num_classes=2)
    
    # 创建数据加载器
    data_loaders = create_data_loaders('./bioast_dataset', batch_size=16)
    
    # 创建训练器
    trainer = ModelTrainer(model, device, save_dir='./test_checkpoints')
    
    # 获取类别权重
    train_dataset = data_loaders['train'].dataset
    class_weights = train_dataset.get_class_weights()
    
    print("开始测试训练...")
    history = trainer.train(
        data_loaders['train'], 
        data_loaders['val'],
        num_epochs=2,  # 测试用少量epoch
        learning_rate=0.001,
        class_weights=class_weights
    )
    
    print("训练历史:")
    for key, values in history.items():
        print(f"  {key}: {values}")