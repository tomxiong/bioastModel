#!/usr/bin/env python3
"""
Optimized Multi-Level MobileNetV3 Training Script
优化版多层分类MobileNetV3训练脚本，专门针对growth_pattern和interference_factors进行优化
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
from collections import Counter
import argparse
from sklearn.utils.class_weight import compute_class_weight

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.optimized_multilevel_mobilenetv3 import create_optimized_multilevel_mobilenetv3
from training.enhanced_multitask_dataset import create_multitask_dataloaders
from utils.metrics import calculate_metrics

class OptimizedMultiLevelTrainer:
    """优化版多级分类训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建实验目录
        self.experiment_dir = Path(config.get('experiment_dir', 'experiments/optimized_multilevel'))
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.writer = SummaryWriter(log_dir=self.experiment_dir / 'logs')
        
        # 训练历史
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'task_weights': [],
            'learning_rates': []
        }
        
        # 最佳性能跟踪
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.current_epoch = 0
        
        print(f"🚀 优化版多级MobileNetV3训练器初始化完成")
        print(f"📱 设备: {self.device}")
        print(f"📁 实验目录: {self.experiment_dir}")
    
    def analyze_dataset_distribution(self, data_loader: DataLoader) -> Dict[str, Dict]:
        """分析数据集分布并计算类别权重"""
        print("📊 分析数据集分布...")
        
        task_distributions = {
            'growth_level': [],
            'growth_pattern': [],
            'interference_factors': []
        }
        
        # 收集所有标签
        for batch_idx, (images, targets) in enumerate(data_loader):
            for task_name in task_distributions.keys():
                if task_name in targets:
                    if task_name == 'interference_factors':
                        # 多标签任务
                        task_distributions[task_name].extend(targets[task_name].cpu().numpy())
                    else:
                        # 单标签任务
                        task_distributions[task_name].extend(targets[task_name].cpu().numpy().tolist())
        
        # 计算分布统计和类别权重
        distribution_info = {}
        class_weights = {}
        
        for task_name, labels in task_distributions.items():
            if task_name == 'interference_factors':
                # 多标签任务的分布分析
                labels_array = np.array(labels)
                pos_counts = labels_array.sum(axis=0)
                neg_counts = len(labels) - pos_counts
                
                distribution_info[task_name] = {
                    'positive_counts': pos_counts.tolist(),
                    'negative_counts': neg_counts.tolist(),
                    'total_samples': len(labels)
                }
                
                # 为多标签任务计算权重（简化处理）
                pos_weights = neg_counts / (pos_counts + 1e-8)
                class_weights[task_name] = torch.FloatTensor(pos_weights).to(self.device)
                
            else:
                # 单标签任务
                label_counts = Counter(labels)
                unique_labels = sorted(label_counts.keys())
                
                distribution_info[task_name] = {
                    'class_counts': {str(k): v for k, v in label_counts.items()},
                    'total_samples': len(labels),
                    'num_classes': len(unique_labels)
                }
                
                # 计算类别权重
                if len(unique_labels) > 1:
                    weights = compute_class_weight(
                        'balanced',
                        classes=np.array(unique_labels),
                        y=np.array(labels)
                    )
                    class_weights[task_name] = torch.FloatTensor(weights).to(self.device)
                else:
                    class_weights[task_name] = torch.ones(len(unique_labels)).to(self.device)
        
        # 打印分布信息
        print("\n📈 数据集分布分析结果:")
        for task_name, info in distribution_info.items():
            print(f"\n{task_name.upper()}:")
            if task_name == 'interference_factors':
                print(f"  总样本数: {info['total_samples']}")
                for i, (pos, neg) in enumerate(zip(info['positive_counts'], info['negative_counts'])):
                    print(f"  类别 {i}: 正样本={pos}, 负样本={neg}, 权重={class_weights[task_name][i]:.3f}")
            else:
                print(f"  总样本数: {info['total_samples']}")
                print(f"  类别数: {info['num_classes']}")
                for class_id, count in info['class_counts'].items():
                    weight = class_weights[task_name][int(class_id)] if class_id.isdigit() else 0
                    print(f"  类别 {class_id}: {count} 样本, 权重={weight:.3f}")
        
        return distribution_info, class_weights
    
    def create_model_and_optimizer(self, class_weights: Dict[str, torch.Tensor]):
        """创建模型和优化器"""
        print("🏗️ 创建优化版模型...")
        
        # 创建模型
        self.model = create_optimized_multilevel_mobilenetv3(
            model_size=self.config.get('model_size', 'small'),
            input_channels=self.config.get('input_channels', 1),
            dropout_rate=self.config.get('dropout_rate', 0.3),
            use_focal_loss=self.config.get('use_focal_loss', True),
            use_asymmetric_loss=self.config.get('use_asymmetric_loss', True),
            use_task_attention=self.config.get('use_task_attention', True),
            focal_alpha=self.config.get('focal_alpha', 1.0),
            focal_gamma=self.config.get('focal_gamma', 2.0)
        ).to(self.device)
        
        # 设置类别权重
        self.model.set_class_weights(class_weights)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=self.config.get('scheduler_T0', 10), 
            T_mult=self.config.get('scheduler_T_mult', 2),
            eta_min=self.config.get('scheduler_eta_min', 1e-6)
        )
        
        # 打印模型信息
        model_info = self.model.get_model_info()
        print(f"📱 模型: {model_info['model_name']}")
        print(f"🔢 总参数: {model_info['total_parameters']:,}")
        print(f"🎯 可训练参数: {model_info['trainable_parameters']:,}")
        print(f"🎛️ 优化功能: {model_info['optimizations']}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'growth_level': 0.0,
            'growth_pattern': 0.0,
            'interference_factors': 0.0
        }
        
        num_batches = len(train_loader)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # 前向传播
            outputs = self.model(images)
            
            # 计算损失
            losses = self.model.compute_loss(outputs, targets, epoch=self.current_epoch)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累积损失
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # 打印进度
            if batch_idx % 20 == 0:
                progress = 100.0 * batch_idx / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"训练进度: {progress:.1f}% | "
                      f"总损失: {losses['total'].item():.4f} | "
                      f"GP损失: {losses['growth_pattern'].item():.4f} | "
                      f"IF损失: {losses['interference_factors'].item():.4f} | "
                      f"学习率: {current_lr:.6f}")
        
        # 平均损失
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, float]]:
        """验证一个epoch"""
        self.model.eval()
        
        epoch_losses = {
            'total': 0.0,
            'growth_level': 0.0,
            'growth_pattern': 0.0,
            'interference_factors': 0.0
        }
        
        all_predictions = {
            'growth_level': [],
            'growth_pattern': [],
            'interference_factors': []
        }
        
        all_targets = {
            'growth_level': [],
            'growth_pattern': [],
            'interference_factors': []
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                outputs = self.model(images)
                predictions = self.model.predict(images)
                
                # 计算损失
                losses = self.model.compute_loss(outputs, targets, epoch=self.current_epoch)
                
                # 累积损失
                for key in epoch_losses.keys():
                    if key in losses:
                        epoch_losses[key] += losses[key].item()
                
                # 收集预测和目标
                for task_name in all_predictions.keys():
                    if task_name in predictions and task_name in targets:
                        all_predictions[task_name].append(predictions[task_name].cpu())
                        all_targets[task_name].append(targets[task_name].cpu())
        
        # 平均损失
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        # 计算准确率
        accuracies = {}
        for task_name in all_predictions.keys():
            if all_predictions[task_name]:
                pred_tensor = torch.cat(all_predictions[task_name], dim=0)
                target_tensor = torch.cat(all_targets[task_name], dim=0)
                
                if task_name == 'interference_factors':
                    # 多标签准确率
                    pred_binary = (pred_tensor > 0.5).float()
                    accuracies[task_name] = (pred_binary == target_tensor).float().mean().item()
                else:
                    # 单标签准确率
                    pred_classes = torch.argmax(pred_tensor, dim=1)
                    accuracies[task_name] = (pred_classes == target_tensor).float().mean().item()
        
        return epoch_losses, accuracies
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        print(f"🚀 开始训练优化版多级MobileNetV3模型")
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
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - start_time
            
            # 计算总体准确率
            overall_accuracy = np.mean(list(val_accuracies.values()))
            
            # 记录训练历史
            epoch_info = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'overall_accuracy': overall_accuracy,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            
            self.history['train_losses'].append(train_losses)
            self.history['val_losses'].append(val_losses)
            self.history['val_accuracies'].append(val_accuracies)
            self.history['learning_rates'].append(current_lr)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train_Total', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Val_Total', val_losses['total'], epoch)
            self.writer.add_scalar('Accuracy/Overall', overall_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            for task_name, acc in val_accuracies.items():
                self.writer.add_scalar(f'Accuracy/{task_name}', acc, epoch)
            
            # 保存最佳模型
            if overall_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = overall_accuracy
                self.best_epoch = epoch + 1
                
                # 保存模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_accuracy': self.best_val_accuracy,
                    'config': self.config
                }, self.experiment_dir / 'best_model.pth')
                
                print(f"💾 保存最佳模型 (Epoch {epoch + 1}, 准确率: {overall_accuracy:.4f})")
            
            # 打印epoch结果
            print(f"\nEpoch {epoch + 1}/{epochs} - {epoch_time:.2f}s")
            print(f"训练损失: {train_losses['total']:.4f}")
            print(f"验证损失: {val_losses['total']:.4f}")
            print(f"验证准确率:")
            for task_name, acc in val_accuracies.items():
                print(f"  {task_name}: {acc:.4f}")
            print(f"总体准确率: {overall_accuracy:.4f}")
            print(f"学习率: {current_lr:.6f}")
            print(f"最佳准确率: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
            print("-" * 60)
            
            # 定期保存训练历史
            if (epoch + 1) % 10 == 0:
                self.save_training_history()
        
        # 训练完成
        print(f"🎉 训练完成!")
        print(f"🏆 最佳验证准确率: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
        
        # 保存最终训练历史
        self.save_training_history()
        self.writer.close()
    
    def save_training_history(self):
        """保存训练历史"""
        history_file = self.experiment_dir / 'optimized_training_history.json'
        
        # 转换为可序列化格式
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    serializable_history[key] = value
                else:
                    serializable_history[key] = [float(v) if isinstance(v, (int, float, np.number)) else v for v in value]
            else:
                serializable_history[key] = value
        
        # 添加最佳性能信息
        serializable_history['best_performance'] = {
            'best_val_accuracy': float(self.best_val_accuracy),
            'best_epoch': int(self.best_epoch),
            'total_epochs': len(self.history['train_losses'])
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        print(f"💾 训练历史已保存到: {history_file}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Optimized Multi-Level MobileNetV3 Training')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, 
                       default='/home/aaa/ws/bioastModel/ds/images',
                       help='Root directory of dataset')
    parser.add_argument('--annotations_file', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='Path to annotations JSON file')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'large'], help='Model size')
    parser.add_argument('--input_channels', type=int, default=1,
                       help='Number of input channels')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # 优化参数
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                       help='Use Focal Loss for growth_pattern')
    parser.add_argument('--use_asymmetric_loss', action='store_true', default=True,
                       help='Use Asymmetric Loss for interference_factors')
    parser.add_argument('--use_task_attention', action='store_true', default=True,
                       help='Use task-specific attention')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                       help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # 数据分割参数
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                       help='Train/Val/Test split ratios')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # 实验参数
    parser.add_argument('--experiment_dir', type=str, 
                       default='experiments/optimized_multilevel_mobilenetv3',
                       help='Experiment directory')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 配置
    config = {
        'data_root': args.data_root,
        'annotations_file': args.annotations_file,
        'model_size': args.model_size,
        'input_channels': args.input_channels,
        'dropout_rate': args.dropout_rate,
        'use_focal_loss': args.use_focal_loss,
        'use_asymmetric_loss': args.use_asymmetric_loss,
        'use_task_attention': args.use_task_attention,
        'focal_alpha': args.focal_alpha,
        'focal_gamma': args.focal_gamma,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'split_ratio': args.split_ratio,
        'seed': args.seed,
        'experiment_dir': args.experiment_dir
    }
    
    print("🚀 优化版多级MobileNetV3训练开始")
    print(f"📊 配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    try:
        # 创建数据加载器
        print("📁 创建数据加载器...")
        dataloaders = create_multitask_dataloaders(
            data_root=config['data_root'],
            annotations_file=config['annotations_file'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            split_ratio=config['split_ratio'],
            seed=config['seed']
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders['test']
        label_info = dataloaders.get('label_info', {})
        
        print(f"✅ 数据加载器创建成功")
        print(f"📊 训练集: {len(train_loader.dataset)} 样本")
        print(f"📊 验证集: {len(val_loader.dataset)} 样本")
        print(f"📊 测试集: {len(test_loader.dataset)} 样本")
        
        # 创建训练器
        trainer = OptimizedMultiLevelTrainer(config)
        
        # 分析数据集分布
        distribution_info, class_weights = trainer.analyze_dataset_distribution(train_loader)
        
        # 创建模型和优化器
        trainer.create_model_and_optimizer(class_weights)
        
        # 开始训练
        trainer.train(train_loader, val_loader)
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())