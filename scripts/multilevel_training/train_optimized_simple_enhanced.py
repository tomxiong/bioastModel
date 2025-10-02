#!/usr/bin/env python3
"""
优化版Simple Enhanced Multi-level MobileNetV3训练脚本
包含增加的训练轮次、任务权重调整和早停机制
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.simple_enhanced_multilevel_mobilenetv3 import create_simple_enhanced_multilevel_mobilenetv3
from training.multilevel_dataset import create_multilevel_dataloaders
from utils.metrics import calculate_metrics


class OptimizedSimpleEnhancedTrainer:
    """优化版Simple Enhanced训练器"""
    
    def __init__(self, model, device, experiment_dir, config):
        self.model = model
        self.device = device
        self.experiment_dir = experiment_dir
        self.config = config
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': [],
            'task_losses': {
                'growth_level': [],
                'growth_pattern': [],
                'interference_factors': []
            }
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 任务权重 - 平衡权重配置 (基于性能差距分析)
        self.task_weights = {
            'growth_level': 1.0,
            'growth_pattern': 1.0,  # 从1.5降至1.0，实现完全平衡
            'interference_factors': 1.0  # 从2.0降至1.0，避免过度关注
        }
        
        print(f"📊 任务权重设置: {self.task_weights}")
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(experiment_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def compute_weighted_loss(self, outputs, targets):
        """计算加权损失"""
        total_loss = 0.0
        task_losses = {}
        
        # 计算各任务损失
        # Growth level 和 Growth pattern 使用交叉熵损失
        criterion_ce = nn.CrossEntropyLoss()
        growth_level_loss = criterion_ce(outputs['growth_level'], targets['growth_level'])
        growth_pattern_loss = criterion_ce(outputs['growth_pattern'], targets['growth_pattern'])
        
        # Interference factors 使用二元交叉熵损失（多标签分类）
        criterion_bce = nn.BCEWithLogitsLoss()
        interference_factors_loss = criterion_bce(outputs['interference_factors'], targets['interference_factors'])
        
        # 应用权重
        weighted_growth_level = growth_level_loss * self.task_weights['growth_level']
        weighted_growth_pattern = growth_pattern_loss * self.task_weights['growth_pattern']
        weighted_interference = interference_factors_loss * self.task_weights['interference_factors']
        
        # 总损失
        total_loss = weighted_growth_level + weighted_growth_pattern + weighted_interference
        
        # 记录各任务损失
        task_losses = {
            'growth_level': growth_level_loss.item(),
            'growth_pattern': growth_pattern_loss.item(),
            'interference_factors': interference_factors_loss.item()
        }
        
        return total_loss, task_losses
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_task_losses = {'growth_level': 0.0, 'growth_pattern': 0.0, 'interference_factors': 0.0}
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            images, targets = batch  # batch是(images, targets)的元组
            images = images.to(self.device)
            
            # targets已经是字典格式，直接移动到设备
            targets = {
                task: target.to(self.device) for task, target in targets.items()
            }
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(images)
            
            # 计算加权损失
            loss, task_losses = self.compute_weighted_loss(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            for task, task_loss in task_losses.items():
                total_task_losses[task] += task_loss
            
            # 打印进度
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in total_task_losses.items()}
        
        return avg_loss, avg_task_losses
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = {'growth_level': [], 'growth_pattern': [], 'interference_factors': []}
        all_targets = {'growth_level': [], 'growth_pattern': [], 'interference_factors': []}
        
        with torch.no_grad():
            for batch in val_loader:
                images, targets = batch  # batch是(images, targets)的元组
                images = images.to(self.device)
                
                # targets已经是字典格式，直接移动到设备
                targets = {
                    task: target.to(self.device) for task, target in targets.items()
                }
                
                outputs = self.model(images)
                loss, _ = self.compute_weighted_loss(outputs, targets)
                total_loss += loss.item()
                
                # 收集预测和目标
                for task in all_predictions.keys():
                    if task == 'interference_factors':
                        # 多标签分类：使用sigmoid + 阈值
                        preds = torch.sigmoid(outputs[task]) > 0.5
                        all_predictions[task].extend(preds.cpu().numpy())
                        all_targets[task].extend(targets[task].cpu().numpy())
                    else:
                        # 多分类：使用argmax
                        preds = torch.argmax(outputs[task], dim=1)
                        all_predictions[task].extend(preds.cpu().numpy())
                        all_targets[task].extend(targets[task].cpu().numpy())
        
        # 计算准确率
        val_metrics = {}
        for task in all_predictions.keys():
            if task == 'interference_factors':
                # 多标签分类：计算每个样本的准确率（所有标签都正确才算正确）
                preds_array = np.array(all_predictions[task])
                targets_array = np.array(all_targets[task])
                accuracy = np.mean(np.all(preds_array == targets_array, axis=1))
            else:
                # 多分类：计算准确率
                accuracy = np.mean(np.array(all_predictions[task]) == np.array(all_targets[task]))
            val_metrics[f'{task}_accuracy'] = accuracy
        
        overall_accuracy = np.mean(list(val_metrics.values()))
        avg_val_loss = total_loss / len(val_loader)
        
        return avg_val_loss, val_metrics, overall_accuracy
    
    def early_stopping_check(self, val_acc):
        """早停检查"""
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            return False  # 不停止
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config['patience']:
                return True  # 停止训练
        return False
    
    def train(self, train_loader, val_loader, num_epochs, learning_rate):
        """训练模型"""
        # 设置优化器和调度器
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 使用CosineAnnealingLR调度器，更平滑的学习率衰减
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        print(f"🚀 开始优化训练 - 总共 {num_epochs} 个epoch")
        print(f"📊 任务权重: {self.task_weights}")
        print(f"⏰ 早停patience: {self.config['patience']}")
        print(f"📈 学习率调度器: CosineAnnealingLR")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            epoch_start_time = datetime.now()
            
            # 训练
            train_loss, train_task_losses = self.train_epoch(train_loader, optimizer, epoch)
            
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
            
            for task, loss in train_task_losses.items():
                self.train_history['task_losses'][task].append(loss)
            
            # 保存最佳模型
            if overall_acc > self.best_val_acc:
                self.best_val_acc = overall_acc
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), 
                          os.path.join(self.experiment_dir, 'best_model.pth'))
                print(f"💾 保存最佳模型 (准确率: {overall_acc:.4f})")
            
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            
            # 打印epoch结果
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"整体准确率: {overall_acc:.4f}")
            print(f"各任务准确率:")
            for task, acc in val_metrics.items():
                print(f"  {task}: {acc:.4f}")
            print(f"学习率: {current_lr:.6f}")
            print(f"用时: {epoch_time:.2f}s")
            print(f"最佳准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
            print("-" * 60)
            
            # 早停检查
            if self.early_stopping_check(overall_acc):
                print(f"🛑 早停触发! 在epoch {epoch+1}停止训练")
                print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
                break
        
        # 保存训练历史
        history_path = os.path.join(self.experiment_dir, 'optimized_training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"\n✅ 训练完成!")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"📁 模型和历史已保存到: {self.experiment_dir}")
        
        return self.best_val_acc


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Optimized Simple Enhanced Multi-level MobileNetV3 Training')
    
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
    
    # 优化的训练参数 (基于性能差距分析优化)
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=20,  # 从40降至20轮，避免过度训练
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=8,  # 从15降至8，更快收敛
                       help='早停patience')
    
    # 任务权重 (平衡权重配置)
    parser.add_argument('--growth_level_weight', type=float, default=1.0,
                       help='生长水平任务权重')
    parser.add_argument('--growth_pattern_weight', type=float, default=1.0,  # 从2.0降至1.0
                       help='生长模式任务权重')
    parser.add_argument('--interference_weight', type=float, default=1.0,  # 从1.5降至1.0
                       help='干扰因子任务权重')
    
    # 数据分割
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='测试集比例')
    
    # 实验设置
    parser.add_argument('--experiment_name', type=str, default='optimized_simple_enhanced',
                       help='实验名称')
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='实验目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备 (cpu/cuda/auto)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  使用设备: {device}")
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.experiment_dir, f"{args.experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 训练配置
    config = {
        'patience': args.patience,
        'task_weights': {
            'growth_level': args.growth_level_weight,
            'growth_pattern': args.growth_pattern_weight,
            'interference_factors': args.interference_weight
        }
    }
    
    # 创建数据加载器
    print("📊 创建数据加载器...")
    train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
        json_path=args.json_path,
        image_root=args.data_root,
        batch_size=args.batch_size,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        num_workers=4
    )
    
    print(f"📈 数据集大小: 训练={len(train_loader.dataset)}, "
          f"验证={len(val_loader.dataset)}, 测试={len(test_loader.dataset)}")
    
    # 创建模型
    print("🏗️  创建模型...")
    model = create_simple_enhanced_multilevel_mobilenetv3(
        model_size=args.model_size,
        dropout_rate=args.dropout_rate,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")
    
    # 创建优化训练器
    trainer = OptimizedSimpleEnhancedTrainer(model, device, experiment_dir, config)
    
    # 开始训练
    best_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    print(f"\n🎉 训练完成! 最佳验证准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()