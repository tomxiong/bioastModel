#!/usr/bin/env python3
"""
修复NaN损失问题的稳定训练脚本
基于调试结果优化训练配置
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import argparse

# 导入模型和数据集
from models.enhanced_multitask_mobilenetv3 import (
    create_enhanced_multitask_mobilenetv3, 
    get_class_definitions
)
from enhanced_multitask_ni_dataset import EnhancedMultiTaskNIDataset

class StableM16MultiTaskTrainer:
    """稳定的m16多任务训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置混合精度训练
        self.use_amp = config.get('use_amp', False)
        self.scaler = None
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        
        # 设置日志
        self.setup_logging()
        
        # 创建模型
        self.model = self.create_model()
        
        # 创建数据加载器
        self.train_loader, self.val_loader, self.test_loader, self.task_info = self.create_dataloaders()
        
        # 创建损失函数
        self.create_loss_functions()
        
        # 创建优化器
        self.optimizer = self.create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self.create_scheduler()
        
        # 创建实验目录
        self.create_experiment_dir()
        
        # 记录最佳性能
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
    def setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"stable_m16_multitask_training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_model(self):
        """创建模型"""
        model = create_enhanced_multitask_mobilenetv3(
            growth_level_classes=3,  # negative, positive, weak_growth
            growth_pattern_classes=9,  # clean, clustered, scattered, heavy_growth, small_dots, irregular_areas, light_gray, default_positive, default_weak_growth
            interference_classes=3,   # pores, debris, artifacts
            fine_grained_classes=40,  # 40种精细分类
            width_mult=self.config.get('width_mult', 1.0),
            dropout_rate=self.config.get('dropout_rate', 0.2)
        )
        
        model = model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"模型参数量: {total_params/1e6:.2f}M")
        self.logger.info(f"可训练参数量: {trainable_params/1e6:.2f}M")
        
        return model
    
    def create_dataloaders(self):
        """创建数据加载器"""
        # 使用m16.json数据集
        json_path = "ni/m16.json"
        image_dir = "ni"
        
        # 创建数据集
        train_dataset = EnhancedMultiTaskNIDataset(
            json_path=json_path,
            image_dir=image_dir,
            split='train',
            image_size=(70, 70)
        )
        
        val_dataset = EnhancedMultiTaskNIDataset(
            json_path=json_path,
            image_dir=image_dir,
            split='val',
            image_size=(70, 70)
        )
        
        test_dataset = EnhancedMultiTaskNIDataset(
            json_path=json_path,
            image_dir=image_dir,
            split='test',
            image_size=(70, 70)
        )
        
        # 创建数据加载器 - 使用优化配置
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=stable_multitask_collate_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=stable_multitask_collate_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=stable_multitask_collate_fn,
            persistent_workers=True if num_workers > 0 else False
        )
        
        task_info = train_dataset.task_info
        
        self.logger.info(f"训练样本数: {len(train_dataset)}")
        self.logger.info(f"验证样本数: {len(val_dataset)}")
        self.logger.info(f"测试样本数: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, task_info
    
    def create_loss_functions(self):
        """创建损失函数"""
        # 分类任务使用交叉熵损失
        self.growth_level_criterion = nn.CrossEntropyLoss().to(self.device)
        self.growth_pattern_criterion = nn.CrossEntropyLoss().to(self.device)
        self.fine_grained_criterion = nn.CrossEntropyLoss().to(self.device)
        
        # 干扰因素使用多标签二元交叉熵损失
        self.interference_criterion = nn.BCEWithLogitsLoss().to(self.device)
        
        # 任务权重
        self.task_weights = self.config.get('task_weights', {
            'growth_level': 1.0,
            'growth_pattern': 1.0,
            'interference_factors': 0.5,
            'fine_grained': 1.0
        })
        
    def create_optimizer(self):
        """创建优化器"""
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    def create_scheduler(self):
        """创建学习率调度器"""
        epochs = self.config.get('epochs', 50)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        
        return scheduler
    
    def create_experiment_dir(self):
        """创建实验目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(f"experiments/stable_m16_multitask_mobilenetv3_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_file = self.exp_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"实验目录: {self.exp_dir}")
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        growth_level_correct = 0
        growth_pattern_correct = 0
        fine_grained_correct = 0
        total_samples = 0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # 数据移至设备
                images = batch['image'].to(self.device)
                growth_level_labels = batch['growth_level_label'].to(self.device)
                growth_pattern_labels = batch['growth_pattern_label'].to(self.device)
                interference_labels = batch['interference_labels'].to(self.device)
                fine_grained_labels = batch['fine_grained_label'].to(self.device)
                
                # 检查数据有效性
                if torch.isnan(images).any() or torch.isinf(images).any():
                    self.logger.warning(f"批次 {batch_idx} 图像包含NaN或Inf，跳过")
                    continue
                
                self.optimizer.zero_grad()
                
                # 前向传播 - 支持混合精度
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)
                        
                        # 检查模型输出
                        skip_batch = False
                        for key, output in outputs.items():
                            if torch.isnan(output).any() or torch.isinf(output).any():
                                self.logger.warning(f"批次 {batch_idx} 输出 {key} 包含NaN或Inf，跳过")
                                skip_batch = True
                                break
                        
                        if skip_batch:
                            continue
                        
                        # 计算损失
                        loss = self.compute_loss(outputs, batch)
                        
                        # 检查损失
                        if torch.isnan(loss) or torch.isinf(loss) or loss > 100:
                            self.logger.warning(f"批次 {batch_idx} 损失异常: {loss.item():.6f}，跳过")
                            continue
                    
                    # 反向传播 - 混合精度
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # 更新权重
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 标准精度训练
                    outputs = self.model(images)
                    
                    # 检查模型输出
                    skip_batch = False
                    for key, output in outputs.items():
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            self.logger.warning(f"批次 {batch_idx} 输出 {key} 包含NaN或Inf，跳过")
                            skip_batch = True
                            break
                    
                    if skip_batch:
                        continue
                    
                    # 计算损失
                    loss = self.compute_loss(outputs, batch)
                    
                    # 检查损失
                    if torch.isnan(loss) or torch.isinf(loss) or loss > 100:
                        self.logger.warning(f"批次 {batch_idx} 损失异常: {loss.item():.6f}，跳过")
                        continue
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # 更新权重
                    self.optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
                
                # 计算准确率
                _, growth_level_pred = torch.max(outputs['growth_level'], 1)
                _, growth_pattern_pred = torch.max(outputs['growth_pattern'], 1)
                _, fine_grained_pred = torch.max(outputs['fine_grained'], 1)
                
                growth_level_correct += (growth_level_pred == growth_level_labels).sum().item()
                growth_pattern_correct += (growth_pattern_pred == growth_pattern_labels).sum().item()
                fine_grained_correct += (fine_grained_pred == fine_grained_labels).sum().item()
                
                total_samples += images.size(0)
                
                if batch_idx % 50 == 0:
                    self.logger.info(
                        f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )
                    
            except Exception as e:
                self.logger.warning(f"批次 {batch_idx} 训练失败: {e}")
                continue
        
        if valid_batches == 0:
            self.logger.error(f"Epoch {epoch} 没有有效的训练批次!")
            return float('nan'), 0, 0, 0, 0
        
        avg_loss = total_loss / valid_batches
        growth_level_acc = growth_level_correct / total_samples * 100
        growth_pattern_acc = growth_pattern_correct / total_samples * 100
        fine_grained_acc = fine_grained_correct / total_samples * 100
        combined_acc = (growth_level_acc + growth_pattern_acc + fine_grained_acc) / 3
        
        return avg_loss, growth_level_acc, growth_pattern_acc, fine_grained_acc, combined_acc
    
    def validate(self, epoch: int):
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        growth_level_correct = 0
        growth_pattern_correct = 0
        fine_grained_correct = 0
        total_samples = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    # 数据移至设备
                    images = batch['image'].to(self.device)
                    growth_level_labels = batch['growth_level_label'].to(self.device)
                    growth_pattern_labels = batch['growth_pattern_label'].to(self.device)
                    interference_labels = batch['interference_labels'].to(self.device)
                    fine_grained_labels = batch['fine_grained_label'].to(self.device)
                    
                    # 前向传播
                    outputs = self.model(images)
                    
                    # 计算损失
                    loss = self.compute_loss(outputs, batch)
                    
                    # 检查损失
                    if torch.isnan(loss) or torch.isinf(loss) or loss > 100:
                        continue
                    
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    # 计算准确率
                    _, growth_level_pred = torch.max(outputs['growth_level'], 1)
                    _, growth_pattern_pred = torch.max(outputs['growth_pattern'], 1)
                    _, fine_grained_pred = torch.max(outputs['fine_grained'], 1)
                    
                    growth_level_correct += (growth_level_pred == growth_level_labels).sum().item()
                    growth_pattern_correct += (growth_pattern_pred == growth_pattern_labels).sum().item()
                    fine_grained_correct += (fine_grained_pred == fine_grained_labels).sum().item()
                    
                    total_samples += images.size(0)
                    
                except Exception as e:
                    self.logger.warning(f"验证批次 {batch_idx} 失败: {e}")
                    continue
        
        if valid_batches == 0:
            self.logger.error(f"Epoch {epoch} 没有有效的验证批次!")
            return float('nan'), 0, 0, 0, 0
        
        avg_loss = total_loss / valid_batches
        growth_level_acc = growth_level_correct / total_samples * 100
        growth_pattern_acc = growth_pattern_correct / total_samples * 100
        fine_grained_acc = fine_grained_correct / total_samples * 100
        combined_acc = (growth_level_acc + growth_pattern_acc + fine_grained_acc) / 3
        
        # 保存最佳模型
        if combined_acc > self.best_val_accuracy:
            self.best_val_accuracy = combined_acc
            self.best_epoch = epoch
            self.save_model('best.pth')
        
        return avg_loss, growth_level_acc, growth_pattern_acc, fine_grained_acc, combined_acc
    
    def compute_loss(self, outputs, batch):
        """计算多任务损失"""
        growth_level_labels = batch['growth_level_label'].to(self.device)
        growth_pattern_labels = batch['growth_pattern_label'].to(self.device)
        interference_labels = batch['interference_labels'].to(self.device)
        fine_grained_labels = batch['fine_grained_label'].to(self.device)
        
        # 计算各任务损失
        growth_level_loss = self.growth_level_criterion(
            outputs['growth_level'], growth_level_labels
        )
        
        growth_pattern_loss = self.growth_pattern_criterion(
            outputs['growth_pattern'], growth_pattern_labels
        )
        
        interference_loss = self.interference_criterion(
            outputs['interference_factors'], interference_labels
        )
        
        fine_grained_loss = self.fine_grained_criterion(
            outputs['fine_grained'], fine_grained_labels
        )
        
        # 加权总损失
        total_loss = (
            self.task_weights['growth_level'] * growth_level_loss +
            self.task_weights['growth_pattern'] * growth_pattern_loss +
            self.task_weights['interference_factors'] * interference_loss +
            self.task_weights['fine_grained'] * fine_grained_loss
        )
        
        return total_loss
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = self.exp_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'task_info': self.task_info
        }, model_path)
        
        self.logger.info(f"模型已保存到: {model_path}")
    
    def train(self):
        """训练模型"""
        self.logger.info("=== 优化的m16多任务MobileNetV3训练开始 ===")
        self.logger.info(f"模型: Enhanced MobileNetV3-MultiTask (optimized m16)")
        self.logger.info(f"生长级别类别数: {self.model.growth_level_classes}")
        self.logger.info(f"生长模式类别数: {self.model.growth_pattern_classes}")
        self.logger.info(f"干扰因素类别数: {self.model.interference_classes}")
        self.logger.info(f"精细分类类别数: {self.model.fine_grained_classes}")
        self.logger.info(f"批次大小: {self.config['batch_size']}")
        self.logger.info(f"学习率: {self.config['learning_rate']}")
        self.logger.info(f"模型宽度倍数: {self.config['width_mult']}")
        self.logger.info(f"Dropout率: {self.config['dropout_rate']}")
        self.logger.info(f"训练轮数: {self.config['epochs']}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"混合精度训练: {'启用' if self.use_amp else '禁用'}")
        self.logger.info(f"数据加载器工作进程: {self.config.get('num_workers', 0)}")
        self.logger.info(f"任务权重: {self.task_weights}")
        self.logger.info("=" * 57)
        
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            train_loss, train_gl_acc, train_gp_acc, train_fg_acc, train_comb_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_gl_acc, val_gp_acc, val_fg_acc, val_comb_acc = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: GL={train_gl_acc:.2f}%, GP={train_gp_acc:.2f}%, FG={train_fg_acc:.2f}%, Comb={train_comb_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: GL={val_gl_acc:.2f}%, GP={val_gp_acc:.2f}%, FG={val_fg_acc:.2f}%, Comb={val_comb_acc:.2f}% - "
                f"Time: {time.time()-start_time:.1f}s - LR: {current_lr:.6f}"
            )
            
            # 保存检查点
            if epoch % 10 == 0:
                self.save_model(f'epoch_{epoch}.pth')
            
            # 检查是否需要提前停止
            if np.isnan(train_loss) or np.isnan(val_loss):
                self.logger.error(f"Epoch {epoch} 检测到NaN损失，停止训练")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"训练完成，总时间: {total_time:.1f}秒")
        self.logger.info(f"最佳验证准确率: {self.best_val_accuracy:.2f}% (Epoch {self.best_epoch})")
        self.logger.info("训练完成！")

def stable_multitask_collate_fn(batch):
    """稳定的多任务数据整理函数"""
    images = torch.stack([item['image'] for item in batch])
    growth_level_labels = torch.tensor([item['growth_level_label'] for item in batch])
    growth_pattern_labels = torch.tensor([item['growth_pattern_label'] for item in batch])
    interference_labels = torch.stack([item['interference_labels'] for item in batch])
    fine_grained_labels = torch.tensor([item['fine_grained_label'] for item in batch])
    
    return {
        'image': images,
        'growth_level_label': growth_level_labels,
        'growth_pattern_label': growth_pattern_labels,
        'interference_labels': interference_labels,
        'fine_grained_label': fine_grained_labels
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='稳定的m16多任务MobileNetV3训练')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.003, help='学习率')
    parser.add_argument('--width_mult', type=float, default=1.2, help='模型宽度倍数')
    parser.add_argument('--dropout_rate', type=float, default=0.15, help='Dropout率')
    parser.add_argument('--use_amp', action='store_true', help='使用混合精度训练')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'width_mult': args.width_mult,
        'dropout_rate': args.dropout_rate,
        'weight_decay': 1e-4,
        'use_amp': args.use_amp,
        'num_workers': args.num_workers,
        'task_weights': {
            'growth_level': 1.0,
            'growth_pattern': 1.0,
            'interference_factors': 0.5,
            'fine_grained': 1.0
        }
    }
    
    # 创建训练器并训练
    trainer = StableM16MultiTaskTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()