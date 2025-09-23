#!/usr/bin/env python3
"""
GPU优化的多任务模型训练脚本
针对RTX 3090 24GB显存优化
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.enhanced_multitask_dataset import EnhancedMultitaskDataset
from models.fixed_efficientnet_b0_multitask import create_fixed_efficientnet_b0_multitask
from models.resnet34_multitask import create_resnet34_multitask


class GPUOptimizedMultiTaskTrainer:
    """GPU优化的多任务训练器"""
    
    def __init__(self, 
                 model_name: str,
                 data_root: str,
                 config: Dict[str, Any],
                 experiment_dir: str):
        self.model_name = model_name
        self.data_root = data_root
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GPU优化设置
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # 优化cudnn性能
            torch.backends.cudnn.deterministic = False  # 允许非确定性算法以提高性能
        
        # 创建实验目录
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 设置TensorBoard
        self.writer = SummaryWriter(os.path.join(experiment_dir, 'tensorboard'))
        
        # 初始化数据加载器
        self._setup_data_loaders()
        
        # 初始化模型
        self._setup_model()
        
        # 初始化优化器和调度器
        self._setup_optimizer()
        
        # 训练历史
        self.train_history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'task_accuracies': {task: [] for task in self.num_classes.keys()},
            'individual_losses': {task: [] for task in self.num_classes.keys()}
        }
        # 添加confidence任务的历史记录
        self.train_history['individual_losses']['confidence'] = []
        
        print(f"✓ GPU优化训练器初始化完成")
        print(f"  模型: {model_name}")
        print(f"  设备: {self.device}")
        print(f"  GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "")
        print(f"  实验目录: {experiment_dir}")
        print(f"  混合精度: {'启用' if config.get('use_amp', False) else '禁用'}")
        print(f"  批次大小: {config['batch_size']}")
    
    def _setup_data_loaders(self):
        """设置GPU优化的数据加载器"""
        print("创建GPU优化数据加载器...")
        
        # 创建数据集
        train_dataset = EnhancedMultitaskDataset(
            self.data_root, split='train'
        )
        val_dataset = EnhancedMultitaskDataset(
            self.data_root, split='val'
        )
        test_dataset = EnhancedMultitaskDataset(
            self.data_root, split='test'
        )
        
        # 保存类别数量信息
        self.num_classes = train_dataset.get_num_classes()
        
        # GPU优化的数据加载器参数
        dataloader_kwargs = {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config.get('num_workers', 8),  # 增加worker数量
            'pin_memory': True,  # 固定内存，加速GPU传输
            'persistent_workers': True,  # 保持worker进程
            'prefetch_factor': 4,  # 预取因子
        }
        
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **dataloader_kwargs
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **{k: v for k, v in dataloader_kwargs.items() if k != 'drop_last'}
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **{k: v for k, v in dataloader_kwargs.items() if k != 'drop_last'}
        )
        
        print(f"GPU优化数据加载器创建完成:")
        print(f"  训练集: {len(self.train_loader)} batches")
        print(f"  验证集: {len(self.val_loader)} batches")
        print(f"  测试集: {len(self.test_loader)} batches")
        print(f"  批次大小: {self.config['batch_size']}")
        print(f"  Worker数量: {dataloader_kwargs['num_workers']}")
        print(f"  类别数量: {self.num_classes}")
    
    def _setup_model(self):
        """设置模型"""
        print(f"创建GPU优化模型: {self.model_name}")
        
        if self.model_name == 'fixed_efficientnet_b0':
            self.model = create_fixed_efficientnet_b0_multitask(
                num_classes=self.num_classes,
                dropout_rate=0.3,
                use_pretrained=True
            )
        elif self.model_name == 'resnet34':
            self.model = create_resnet34_multitask(
                num_classes=self.num_classes,
                dropout_rate=0.3,
                use_attention=True,
                use_pretrained=True
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_name}")
        
        self.model = self.model.to(self.device)
        
        # 计算模型参数量和显存使用
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 估算模型显存使用 (粗略估计)
        model_memory = total_params * 4 / (1024**3)  # 4 bytes per parameter
        batch_memory = self.config['batch_size'] * 1 * 70 * 70 * 4 / (1024**3)  # 输入tensor
        
        print(f"模型参数:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  估算模型显存: {model_memory:.2f}GB")
        print(f"  估算批次显存: {batch_memory:.3f}GB")
        print(f"  预计总显存使用: {model_memory + batch_memory * 2:.2f}GB")
    
    def _setup_optimizer(self):
        """设置优化器和调度器"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        
        # GPU优化的优化器参数
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4),
                eps=1e-8,  # 数值稳定性
                amsgrad=True  # 改进的AdamW
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 1e-4),
                eps=1e-8,
                amsgrad=True
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config.get('weight_decay', 1e-4),
                nesterov=True  # Nesterov动量
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        # 设置学习率调度器
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['epochs'],
                eta_min=self.config['learning_rate'] * 0.01  # 最小学习率
            )
        elif scheduler_name == 'cosine_warm':
            # 带预热的余弦调度器
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['epochs'] // 4,
                T_mult=2,
                eta_min=self.config['learning_rate'] * 0.01
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config['epochs'] // 3, 
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5, 
                patience=8,
                min_lr=self.config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = None
        
        # 混合精度训练
        if self.config.get('use_amp', False):
            self.scaler = torch.cuda.amp.GradScaler()
            print("✓ 启用混合精度训练 (AMP)")
        else:
            self.scaler = None
        
        print(f"优化器: {optimizer_name}")
        print(f"调度器: {scheduler_name}")
        print(f"学习率: {self.config['learning_rate']}")
        print(f"权重衰减: {self.config.get('weight_decay', 1e-4)}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        individual_losses_sum = {task: 0.0 for task in self.num_classes.keys()}
        individual_losses_sum['confidence'] = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # GPU优化的数据传输
            images = batch['image'].to(self.device, non_blocking=True)
            targets = {key: value.to(self.device, non_blocking=True) 
                      for key, value in batch.items() 
                      if key != 'image' and key != 'original_path'}
            
            self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            
            if self.scaler:
                # 混合精度训练
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss, individual_losses = self.model.compute_loss(predictions, targets, epoch)
                
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准精度训练
                predictions = self.model(images)
                loss, individual_losses = self.model.compute_loss(predictions, targets, epoch)
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 累计各任务损失
            for task, task_loss in individual_losses.items():
                individual_losses_sum[task] += task_loss.item()
            
            # 打印进度 (减少频率以提高性能)
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_individual_losses = {task: loss_sum / num_batches 
                               for task, loss_sum in individual_losses_sum.items()}
        
        return avg_loss, avg_individual_losses
    
    def validate(self, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        task_correct = {task: 0 for task in self.num_classes.keys()}
        task_total = {task: 0 for task in self.num_classes.keys()}
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                targets = {key: value.to(self.device, non_blocking=True) 
                          for key, value in batch.items() 
                          if key != 'image' and key != 'original_path'}
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images)
                        loss, _ = self.model.compute_loss(predictions, targets, epoch)
                else:
                    predictions = self.model(images)
                    loss, _ = self.model.compute_loss(predictions, targets, epoch)
                
                total_loss += loss.item()
                
                # 计算准确率
                batch_size = images.size(0)
                total_samples += batch_size
                
                for task in self.num_classes.keys():
                    if task == 'interference_factors':
                        # 多标签任务
                        pred_binary = (torch.sigmoid(predictions[task]) > 0.5).float()
                        correct = (pred_binary == targets[task]).all(dim=1).sum().item()
                    else:
                        # 标准分类任务
                        pred_classes = torch.argmax(predictions[task], dim=1)
                        correct = (pred_classes == targets[task]).sum().item()
                    
                    task_correct[task] += correct
                    task_total[task] += batch_size
                
                # 整体准确率（所有任务都正确才算正确）
                all_correct = 0
                for i in range(batch_size):
                    sample_correct = True
                    for task in self.num_classes.keys():
                        if task == 'interference_factors':
                            pred_binary = (torch.sigmoid(predictions[task][i]) > 0.5).float()
                            if not torch.equal(pred_binary, targets[task][i]):
                                sample_correct = False
                                break
                        else:
                            pred_class = torch.argmax(predictions[task][i])
                            if pred_class != targets[task][i]:
                                sample_correct = False
                                break
                    if sample_correct:
                        all_correct += 1
                
                total_correct += all_correct
        
        avg_loss = total_loss / len(self.val_loader)
        overall_accuracy = total_correct / total_samples
        task_accuracies = {task: task_correct[task] / task_total[task] 
                          for task in self.num_classes.keys()}
        
        return avg_loss, overall_accuracy, task_accuracies
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
            'config': self.config,
            'num_classes': self.num_classes
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.experiment_dir, 'latest.pth'))
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(self.experiment_dir, 'best.pth'))
            print(f"  ✓ 保存最佳模型 (epoch {epoch})")
    
    def train(self):
        """训练主循环"""
        print(f"\n开始GPU优化训练 {self.model_name}")
        print(f"训练参数: {self.config}")
        print("=" * 60)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            print("-" * 50)
            
            # 训练
            train_loss, train_individual_losses = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_accuracy, task_accuracies = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_accuracy)
                else:
                    self.scheduler.step()
            
            # 记录历史
            self.train_history['train_losses'].append(train_loss)
            self.train_history['val_losses'].append(val_loss)
            self.train_history['val_accuracies'].append(val_accuracy)
            
            for task, acc in task_accuracies.items():
                self.train_history['task_accuracies'][task].append(acc)
            
            for task, loss in train_individual_losses.items():
                self.train_history['individual_losses'][task].append(loss)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Val', val_accuracy, epoch)
            
            for task, acc in task_accuracies.items():
                self.writer.add_scalar(f'Accuracy/{task}', acc, epoch)
            
            for task, loss in train_individual_losses.items():
                self.writer.add_scalar(f'Loss/{task}', loss, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # GPU显存使用情况
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar('GPU/Memory_Used_GB', memory_used, epoch)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证准确率: {val_accuracy:.2%}")
            print("各任务准确率:")
            for task, acc in task_accuracies.items():
                print(f"  {task}: {acc:.2%}")
            
            if torch.cuda.is_available():
                print(f"GPU显存使用: {memory_used:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # 检查是否是最佳模型
            is_best = val_accuracy > best_val_acc
            if is_best:
                best_val_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 早停检查
            if patience_counter >= self.config.get('patience', 15):
                print(f"\n早停触发! 验证准确率连续 {patience_counter} 个epoch未改善")
                break
        
        # 保存训练历史
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)
        
        self.writer.close()
        
        print(f"\n✓ GPU优化训练完成!")
        print(f"  最佳验证准确率: {best_val_acc:.2%}")
        print(f"  实验目录: {self.experiment_dir}")
        
        return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='GPU优化多任务模型训练')
    parser.add_argument('--model', type=str, default='resnet34',
                       choices=['fixed_efficientnet_b0', 'resnet34'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,  # 增大默认批次
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='启用混合精度训练')
    parser.add_argument('--data_root', type=str, 
                       default='/home/aaa/ws/bioastModel/ds/images',
                       help='数据根目录')
    
    args = parser.parse_args()
    
    # GPU优化配置
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'patience': 12,  # 稍微减少patience
        'use_amp': args.use_amp,
        'optimizer': 'adamw',
        'scheduler': 'cosine_warm',  # 使用带预热的余弦调度器
        'num_workers': 8,  # 增加数据加载worker
    }
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{args.model}_gpu_optimized_{timestamp}"
    
    print("=" * 70)
    print("🚀 GPU优化多任务模型训练")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"数据集: {args.data_root}")
    print(f"配置: {config}")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n🎮 GPU信息:")
        print(f"  设备: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练")
    
    print()
    
    # 创建训练器并开始训练
    trainer = GPUOptimizedMultiTaskTrainer(
        model_name=args.model,
        data_root=args.data_root,
        config=config,
        experiment_dir=experiment_dir
    )
    
    best_acc = trainer.train()
    
    print(f"\n🎉 最终结果: 最佳验证准确率 {best_acc:.2%}")


if __name__ == "__main__":
    main()