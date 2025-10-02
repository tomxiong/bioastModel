#!/usr/bin/env python3
"""
Optimized Simple Enhanced Multi-level MobileNetV3 Large Model Training
基于简单优化版本的Large模型训练脚本

基于train_optimized_simple_enhanced.py，将模型从small改为large以提升性能
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


class OptimizedSimpleEnhancedLargeTrainer:
    """优化的简单增强版Large模型训练器"""
    
    def __init__(self, model, device, experiment_dir, config):
        self.model = model
        self.device = device
        self.experiment_dir = experiment_dir
        self.config = config
        
        # 任务权重配置 - 基于性能分析优化，全部平衡为1.0
        self.task_weights = {
            'growth_level': 1.0,      # 从1.0保持不变
            'growth_pattern': 1.0,    # 从2.0降至1.0，避免过度关注
            'interference_factors': 1.0  # 从1.5降至1.0，平衡任务
        }
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.train_history = {
            'train_loss': [], 'val_loss': [], 'val_accuracy': [],
            'learning_rates': [], 'task_losses': {
                'growth_level': [], 'growth_pattern': [], 'interference_factors': []
            }
        }
        
        # 早停相关
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_epoch = 0
        
        print(f"🎯 任务权重配置: {self.task_weights}")
        print(f"📁 实验目录: {experiment_dir}")

    def compute_weighted_loss(self, outputs, targets):
        """计算加权多任务损失"""
        total_loss = 0.0
        task_losses = {}
        
        # 计算各任务损失
        growth_level_loss = self.criterion(outputs['growth_level'], targets['growth_level'])
        growth_pattern_loss = self.criterion(outputs['growth_pattern'], targets['growth_pattern'])
        interference_loss = self.criterion(outputs['interference_factors'], targets['interference_factors'])
        
        # 应用任务权重
        weighted_growth_level = growth_level_loss * self.task_weights['growth_level']
        weighted_growth_pattern = growth_pattern_loss * self.task_weights['growth_pattern']
        weighted_interference = interference_loss * self.task_weights['interference_factors']
        
        total_loss = weighted_growth_level + weighted_growth_pattern + weighted_interference
        
        task_losses = {
            'growth_level': growth_level_loss.item(),
            'growth_pattern': growth_pattern_loss.item(),
            'interference_factors': interference_loss.item()
        }
        
        return total_loss, task_losses

    def train_epoch(self, train_loader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        epoch_task_losses = {'growth_level': 0.0, 'growth_pattern': 0.0, 'interference_factors': 0.0}
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            outputs = self.model(data)
            
            loss, task_losses = self.compute_weighted_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            for task, task_loss in task_losses.items():
                epoch_task_losses[task] += task_loss
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        avg_task_losses = {task: loss / len(train_loader) for task, loss in epoch_task_losses.items()}
        
        return {
            'loss': avg_loss,
            'task_losses': avg_task_losses
        }

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = {'growth_level': [], 'growth_pattern': [], 'interference_factors': []}
        all_targets = {'growth_level': [], 'growth_pattern': [], 'interference_factors': []}
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(data)
                loss, _ = self.compute_weighted_loss(outputs, targets)
                total_loss += loss.item()
                
                # 收集预测和真实标签
                for task in all_predictions.keys():
                    if task == 'interference_factors':
                        # 多标签任务：使用sigmoid阈值
                        preds = (torch.sigmoid(outputs[task]) > 0.5).float()
                        all_predictions[task].extend(preds.cpu().numpy())
                    else:
                        # 单标签任务：使用argmax
                        preds = torch.argmax(outputs[task], dim=1)
                        all_predictions[task].extend(preds.cpu().numpy())
                    all_targets[task].extend(targets[task].cpu().numpy())
        
        # 计算各任务准确率
        task_accuracies = {}
        for task in all_predictions.keys():
            if task == 'interference_factors':
                # 多标签任务：计算每个样本的准确率（所有标签都正确才算正确）
                preds_array = np.array(all_predictions[task])
                targets_array = np.array(all_targets[task])
                # 对于多标签，计算每个样本是否完全匹配
                correct = np.sum(np.all(preds_array == targets_array, axis=1))
                total = len(all_predictions[task])
                task_accuracies[task] = correct / total
            else:
                # 单标签任务
                correct = np.sum(np.array(all_predictions[task]) == np.array(all_targets[task]))
                total = len(all_predictions[task])
                task_accuracies[task] = correct / total
        
        overall_accuracy = np.mean(list(task_accuracies.values()))
        avg_loss = total_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': overall_accuracy,
            'task_accuracies': task_accuracies,
            'predictions': all_predictions,
            'targets': all_targets
        }

    def early_stopping_check(self, val_acc):
        """早停检查"""
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            return True  # 保存模型
        else:
            self.patience_counter += 1
            return False

    def train(self, train_loader, val_loader, num_epochs, learning_rate):
        """主训练循环"""
        print(f"\n🚀 开始训练Large模型 (目标: >91% 准确率)")
        print(f"📊 训练参数: epochs={num_epochs}, lr={learning_rate}, patience={self.config['patience']}")
        
        # 优化器和调度器
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 使用CosineAnnealingLR替代ReduceLROnPlateau，实现更平滑的学习率衰减
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            # 训练
            train_results = self.train_epoch(train_loader, optimizer, epoch+1)
            
            # 验证
            val_results = self.validate(val_loader)
            
            # 更新学习率 (CosineAnnealingLR)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_results['loss'])
            self.train_history['val_loss'].append(val_results['loss'])
            self.train_history['val_accuracy'].append(val_results['accuracy'])
            self.train_history['learning_rates'].append(current_lr)
            
            for task in self.train_history['task_losses'].keys():
                self.train_history['task_losses'][task].append(train_results['task_losses'][task])
            
            # 打印结果
            print(f"训练损失: {train_results['loss']:.4f}")
            print(f"验证损失: {val_results['loss']:.4f}")
            print(f"验证准确率: {val_results['accuracy']:.4f}")
            print(f"学习率: {current_lr:.6f}")
            print(f"任务准确率: {val_results['task_accuracies']}")
            
            # 早停检查
            if self.early_stopping_check(val_results['accuracy']):
                print(f"🎯 新的最佳准确率: {val_results['accuracy']:.4f}")
                self.best_epoch = epoch + 1
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                    'config': self.config
                }, os.path.join(self.experiment_dir, 'best_model.pth'))
            
            # 检查是否需要早停
            if self.patience_counter >= self.config['patience']:
                print(f"🛑 早停触发! 最佳准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
                break
        
        # 保存训练历史
        with open(os.path.join(self.experiment_dir, 'optimized_training_history.json'), 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        print(f"\n✅ 训练完成!")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"📈 训练效率: {self.best_epoch} epochs (vs 目标 20 epochs)")
        
        return self.best_val_acc


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Optimized Simple Enhanced Multi-level MobileNetV3 Large Training')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录')
    parser.add_argument('--json_path', type=str, 
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='JSON标注文件路径')
    
    # 模型参数 - 固定为large
    parser.add_argument('--model_size', type=str, default='large', choices=['small', 'large'],
                       help='MobileNetV3模型大小 (固定为large)')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout率')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='是否冻结backbone')
    
    # 优化的训练参数 (基于性能差距分析优化)
    parser.add_argument('--batch_size', type=int, default=24,  # Large模型内存占用更大，减小batch_size
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=20,  # 保持20轮，避免过度训练
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=8e-4,  # 稍微降低学习率适应大模型
                       help='学习率')
    parser.add_argument('--patience', type=int, default=8,  # 保持8，更快收敛
                       help='早停patience')
    
    # 平衡的任务权重 (基于性能分析，全部设为1.0)
    parser.add_argument('--growth_level_weight', type=float, default=1.0,
                       help='Growth level任务权重')
    parser.add_argument('--growth_pattern_weight', type=float, default=1.0,  # 从2.0降至1.0
                       help='Growth pattern任务权重')
    parser.add_argument('--interference_weight', type=float, default=1.0,  # 从1.5降至1.0
                       help='Interference factors任务权重')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/optimized_simple_enhanced_large_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 训练配置
    config = {
        'model_size': args.model_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'patience': args.patience,
        'task_weights': {
            'growth_level': args.growth_level_weight,
            'growth_pattern': args.growth_pattern_weight,
            'interference_factors': args.interference_weight
        }
    }
    
    print(f"📋 训练配置: {config}")
    
    # 创建数据加载器
    print("📂 创建数据加载器...")
    json_path = os.path.join(args.data_root, "m9e1n170.json")
    image_root = args.data_root
    
    train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
        json_path=json_path,
        image_root=image_root,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print(f"📊 数据集大小: 训练={len(train_loader.dataset)}, 验证={len(val_loader.dataset)}")
    
    # 创建Large模型
    print("🏗️  创建Large模型...")
    model = create_simple_enhanced_multilevel_mobilenetv3(
        model_size=args.model_size,  # 'large'
        dropout_rate=args.dropout_rate,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Large模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")
    print(f"📈 相比Small模型参数量增加约 {(total_params / 2.5e6 - 1) * 100:.1f}%")
    
    # 创建优化训练器
    trainer = OptimizedSimpleEnhancedLargeTrainer(model, device, experiment_dir, config)
    
    # 开始训练
    print(f"\n🎯 目标: 通过Large模型提升准确率至 >91%")
    best_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    print(f"\n🎉 Large模型训练完成! 最佳验证准确率: {best_acc:.4f}")
    
    # 性能对比提示
    print(f"\n📊 性能对比:")
    print(f"   - Simple优化版本: 91.61%")
    print(f"   - Immediate优化版本: 87.89%")
    print(f"   - Large模型版本: {best_acc:.2%}")
    
    if best_acc > 0.9161:
        print(f"🎯 成功! Large模型超越了Simple优化版本!")
    elif best_acc > 0.8789:
        print(f"📈 改善! Large模型优于Immediate优化版本!")
    else:
        print(f"⚠️  Large模型未达到预期，可能需要进一步优化")


if __name__ == "__main__":
    main()