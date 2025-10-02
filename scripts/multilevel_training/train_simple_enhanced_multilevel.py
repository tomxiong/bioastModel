#!/usr/bin/env python3
"""
Simple Enhanced Multi-level MobileNetV3 Training Script
简单增强版多层分类MobileNetV3训练脚本

基于原始训练脚本的简化版本，专注于测试简单优化策略的效果
"""

import os
import sys
import argparse
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.simple_enhanced_multilevel_mobilenetv3 import create_simple_enhanced_multilevel_mobilenetv3
from training.multilevel_dataset import create_multilevel_dataloaders
from utils.metrics import calculate_metrics


class SimpleEnhancedTrainer:
    """简单增强版训练器"""
    
    def __init__(self, model, device, experiment_dir):
        self.model = model
        self.device = device
        self.experiment_dir = experiment_dir
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # 数据移到设备
            images = images.to(self.device)
            targets = {key: value.to(self.device) for key, value in targets.items()}
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss, task_losses = self.model.compute_loss(outputs, targets, epoch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印进度
            if batch_idx % 50 == 0:
                progress = 100.0 * batch_idx / num_batches
                gp_loss = task_losses.get('growth_pattern', torch.tensor(0)).item()
                if_loss = task_losses.get('interference_factors', torch.tensor(0)).item()
                lr = optimizer.param_groups[0]['lr']
                
                print(f"训练进度: {progress:.1f}% | "
                      f"总损失: {loss.item():.4f} | "
                      f"GP损失: {gp_loss:.4f} | "
                      f"IF损失: {if_loss:.4f} | "
                      f"学习率: {lr:.6f}")
        
        return total_loss / num_batches
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        all_outputs = {task: [] for task in ['microbe_type', 'growth_level', 'growth_pattern', 'interference_factors']}
        all_targets = {task: [] for task in ['microbe_type', 'growth_level', 'growth_pattern', 'interference_factors']}
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = {key: value.to(self.device) for key, value in targets.items()}
                
                outputs = self.model(images)
                loss, _ = self.model.compute_loss(outputs, targets)
                total_loss += loss.item()
                
                # 收集预测和目标
                for task in all_outputs.keys():
                    if task in outputs and task in targets:
                        all_outputs[task].append(outputs[task].cpu())
                        all_targets[task].append(targets[task].cpu())
        
        # 计算指标
        metrics = {}
        for task in all_outputs.keys():
            if all_outputs[task]:
                task_outputs = torch.cat(all_outputs[task], dim=0)
                task_targets = torch.cat(all_targets[task], dim=0)
                
                if task == 'interference_factors':
                    # 多标签分类
                    predictions = torch.sigmoid(task_outputs) > 0.5
                    accuracy = (predictions == task_targets.bool()).float().mean().item()
                else:
                    # 单标签分类
                    predictions = torch.argmax(task_outputs, dim=1)
                    accuracy = (predictions == task_targets).float().mean().item()
                
                metrics[task] = accuracy
        
        # 计算总体准确率
        overall_accuracy = sum(metrics.values()) / len(metrics) if metrics else 0.0
        
        return total_loss / len(val_loader), metrics, overall_accuracy
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate):
        """训练模型"""
        # 设置优化器和调度器
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        print(f"开始训练 - 总共 {num_epochs} 个epoch")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()
            
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            
            # 验证
            val_loss, val_metrics, overall_acc = self.validate(val_loader)
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(overall_acc)
            self.train_history['learning_rates'].append(current_lr)
            
            # 保存最佳模型
            if overall_acc > self.best_val_acc:
                self.best_val_acc = overall_acc
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), 
                          os.path.join(self.experiment_dir, 'best_model.pth'))
            
            # 打印epoch结果
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            print(f"\nEpoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print("验证准确率:")
            for task, acc in val_metrics.items():
                print(f"  {task}: {acc:.4f}")
            print(f"总体准确率: {overall_acc:.4f}")
            print(f"学习率: {current_lr:.6f}")
            print(f"最佳准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
            print("-" * 60)
        
        # 保存训练历史
        history_path = os.path.join(self.experiment_dir, 'simple_enhanced_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print("🎉 训练完成!")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"💾 训练历史已保存到: {history_path}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Simple Enhanced Multi-level MobileNetV3 Training')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录')
    parser.add_argument('--json_path', type=str, 
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='JSON标注文件路径')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'large'],
                       help='MobileNetV3模型大小')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout率')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='是否冻结backbone')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    
    # 数据分割
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    
    # 实验设置
    parser.add_argument('--experiment_name', type=str, default='simple_enhanced_multilevel_mobilenetv3',
                       help='实验名称')
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='实验目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (cpu/cuda/auto)')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 创建实验目录
    experiment_dir = os.path.join(args.experiment_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
        json_path=args.json_path,
        image_root=args.data_root,
        batch_size=args.batch_size,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        num_workers=4
    )
    
    print(f"数据集大小: 训练={len(train_loader.dataset)}, "
          f"验证={len(val_loader.dataset)}, 测试={len(test_loader.dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = create_simple_enhanced_multilevel_mobilenetv3(
        model_size=args.model_size,
        dropout_rate=args.dropout_rate,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # 创建训练器
    trainer = SimpleEnhancedTrainer(model, device, experiment_dir)
    
    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()