"""
DenseNet-121 使用真实数据训练脚本
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.densenet_wrapper import DenseNet121
from core.real_data_loader import create_real_data_loaders
from core.training_utils import EarlyStopping, ModelCheckpoint, calculate_metrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DenseNetRealDataTrainer:
    """DenseNet-121 真实数据训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建实验目录
        self.experiment_dir = Path(f"experiments/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}/densenet121_real")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"实验目录: {self.experiment_dir}")
        logging.info(f"使用设备: {self.device}")
    
    def create_model(self):
        """创建模型"""
        model = DenseNet121(num_classes=self.config['num_classes'])
        model = model.to(self.device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"模型总参数数量: {total_params:,}")
        logging.info(f"可训练参数数量: {trainable_params:,}")
        
        return model
    
    def create_data_loaders(self):
        """创建真实数据加载器"""
        train_loader, val_loader, test_loader = create_real_data_loaders(
            data_dir=self.config['data_dir'],
            image_size=(self.config['input_size'], self.config['input_size']),
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers']
        )
        
        logging.info(f"训练集大小: {len(train_loader.dataset)}")
        logging.info(f"验证集大小: {len(val_loader.dataset)}")
        logging.info(f"测试集大小: {len(test_loader.dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_optimizer_and_scheduler(self, model):
        """创建优化器和学习率调度器"""
        # 使用AdamW优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            eps=1e-8
        )
        
        # 使用余弦退火调度器，适合真实数据训练
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%"
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, model, val_loader, criterion):
        """验证模型"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """主训练循环"""
        logging.info("开始训练 DenseNet-121 (真实数据)...")
        
        # 创建模型
        model = self.create_model()
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model)
        
        # 创建损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 创建早停和检查点保存器
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        checkpoint = ModelCheckpoint(
            self.experiment_dir,
            monitor='val_accuracy',
            mode='max'
        )
        
        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(
                model, train_loader, optimizer, criterion, epoch
            )
            
            # 验证
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # 更新学习率
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(current_lr)
            
            # 日志输出
            logging.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f}"
            )
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'config': self.config
                }, self.experiment_dir / 'best_model.pth')
                logging.info(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
            
            # 保存检查点
            checkpoint.save(model, optimizer, epoch, val_acc, val_loss)
            
            # 早停检查
            if early_stopping.should_stop(val_loss):
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 保存训练历史
        with open(self.experiment_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # 最终测试
        if test_loader:
            logging.info("开始最终测试...")
            # 加载最佳模型
            checkpoint_data = torch.load(self.experiment_dir / 'best_model.pth')
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            test_loss, test_acc = self.validate(model, test_loader, criterion)
            logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # 保存测试结果
            test_results = {
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'best_val_accuracy': best_val_acc,
                'data_type': 'real_data',
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset),
                'test_samples': len(test_loader.dataset)
            }
            with open(self.experiment_dir / "test_results.json", "w") as f:
                json.dump(test_results, f, indent=2)
        
        logging.info("训练完成!")
        logging.info(f"最佳验证准确率: {best_val_acc:.2f}%")
        
        return model, history

def main():
    """主函数"""
    # 训练配置
    config = {
        'model_name': 'densenet121_real',
        'num_classes': 2,
        'input_size': 70,
        'batch_size': 32,  # 真实数据更多，可以用更大的batch size
        'epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'data_dir': 'bioast_dataset',
        'num_workers': 4
    }
    
    # 创建训练器
    trainer = DenseNetRealDataTrainer(config)
    
    # 开始训练
    model, history = trainer.train()
    
    return model, history

if __name__ == "__main__":
    main()