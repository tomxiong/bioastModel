#!/usr/bin/env python3
"""
Multi-level Trainer for Bacterial Image Classification
多层分类模型训练器
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
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.multilevel_mobilenetv3 import MultiLevelMobileNetV3
from training.multilevel_dataset import create_multilevel_dataloaders

logger = logging.getLogger(__name__)

class MultiLevelTrainer:
    """多层分类模型训练器"""
    
    def __init__(self,
                 model: MultiLevelMobileNetV3,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 label_info: Dict,
                 device: torch.device,
                 experiment_dir: str,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_info = label_info
        self.device = device
        self.experiment_dir = experiment_dir
        
        # 创建实验目录
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'logs'))
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': {},
            'learning_rate': []
        }
        
        # 初始化任务准确率历史
        for task in self.model.num_classes.keys():
            self.history['val_accuracy'][task] = []
        
        # 最佳性能跟踪
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.patience = 15
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized. Experiment dir: {experiment_dir}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.model.num_classes.keys()}
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # 移动到设备
            images = images.to(self.device)
            targets = {task: target.to(self.device) for task, target in targets.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            losses = self.model.compute_loss(outputs, targets)
            
            # 反向传播
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累计损失
            total_loss += losses['total'].item()
            for task in task_losses.keys():
                if task in losses:
                    task_losses[task] += losses[task].item()
            
            num_batches += 1
            
            # 打印进度
            if batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx}/{len(self.train_loader)}, "
                          f"Loss: {losses['total'].item():.4f}")
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        return {'total': avg_loss, **avg_task_losses}
    
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.model.num_classes.keys()}
        
        # 收集预测和真实标签
        all_predictions = {task: [] for task in self.model.num_classes.keys()}
        all_targets = {task: [] for task in self.model.num_classes.keys()}
        
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # 移动到设备
                images = images.to(self.device)
                targets = {task: target.to(self.device) for task, target in targets.items()}
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                losses = self.model.compute_loss(outputs, targets)
                
                total_loss += losses['total'].item()
                for task in task_losses.keys():
                    if task in losses:
                        task_losses[task] += losses[task].item()
                
                # 收集预测结果
                for task in self.model.num_classes.keys():
                    if task == 'interference_factors':
                        # 多标签分类
                        preds = torch.sigmoid(outputs[task]) > 0.5
                        all_predictions[task].append(preds.cpu())
                        all_targets[task].append(targets[task].cpu())
                    else:
                        # 多分类
                        preds = torch.argmax(outputs[task], dim=1)
                        all_predictions[task].append(preds.cpu())
                        all_targets[task].append(targets[task].cpu())
                
                num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        # 计算准确率
        accuracies = {}
        for task in self.model.num_classes.keys():
            preds = torch.cat(all_predictions[task], dim=0).numpy()
            targets_np = torch.cat(all_targets[task], dim=0).numpy()
            
            if task == 'interference_factors':
                # 多标签准确率（每个标签的平均准确率）
                accuracies[task] = np.mean([
                    accuracy_score(targets_np[:, i], preds[:, i])
                    for i in range(targets_np.shape[1])
                ])
            else:
                # 多分类准确率
                accuracies[task] = accuracy_score(targets_np, preds)
        
        return {'total': avg_loss, **avg_task_losses}, accuracies
    
    def train(self, num_epochs: int = 100, save_best: bool = True) -> Dict:
        """训练模型"""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_losses = self.train_epoch()
            
            # 验证
            val_losses, val_accuracies = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['learning_rate'].append(current_lr)
            
            for task, acc in val_accuracies.items():
                self.history['val_accuracy'][task].append(acc)
            
            # 计算综合准确率（加权平均）
            weighted_accuracy = sum(
                acc * self.model.task_weights[task] 
                for task, acc in val_accuracies.items()
            ) / sum(self.model.task_weights.values())
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Val', val_losses['total'], epoch)
            self.writer.add_scalar('Accuracy/Weighted', weighted_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            for task, acc in val_accuracies.items():
                self.writer.add_scalar(f'Accuracy/{task}', acc, epoch)
            
            # 早停检查
            if weighted_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = weighted_accuracy
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # 打印进度
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_losses['total']:.4f}")
            logger.info(f"  Val Loss: {val_losses['total']:.4f}")
            logger.info(f"  Val Accuracies: {val_accuracies}")
            logger.info(f"  Weighted Accuracy: {weighted_accuracy:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            
            # 早停
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch+1}")
        
        # 保存训练历史
        self.save_history()
        
        return self.history
    
    def evaluate(self) -> Dict:
        """在测试集上评估模型"""
        logger.info("Evaluating on test set...")
        
        # 加载最佳模型
        best_model_path = os.path.join(self.experiment_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            logger.info("Loaded best model for evaluation")
        
        self.model.eval()
        
        # 收集预测和真实标签
        all_predictions = {task: [] for task in self.model.num_classes.keys()}
        all_targets = {task: [] for task in self.model.num_classes.keys()}
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = {task: target.to(self.device) for task, target in targets.items()}
                
                outputs = self.model(images)
                
                for task in self.model.num_classes.keys():
                    if task == 'interference_factors':
                        preds = torch.sigmoid(outputs[task]) > 0.5
                        all_predictions[task].append(preds.cpu())
                        all_targets[task].append(targets[task].cpu())
                    else:
                        preds = torch.argmax(outputs[task], dim=1)
                        all_predictions[task].append(preds.cpu())
                        all_targets[task].append(targets[task].cpu())
        
        # 计算详细指标
        results = {}
        for task in self.model.num_classes.keys():
            preds = torch.cat(all_predictions[task], dim=0).numpy()
            targets_np = torch.cat(all_targets[task], dim=0).numpy()
            
            if task == 'interference_factors':
                # 多标签指标
                task_results = {}
                for i, factor in enumerate(sorted(self.label_info[task].keys())):
                    acc = accuracy_score(targets_np[:, i], preds[:, i])
                    task_results[factor] = {'accuracy': acc}
                
                # 整体准确率
                task_results['overall_accuracy'] = np.mean([
                    accuracy_score(targets_np[:, i], preds[:, i])
                    for i in range(targets_np.shape[1])
                ])
            else:
                # 多分类指标
                accuracy = accuracy_score(targets_np, preds)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets_np, preds, average='weighted', zero_division=0
                )
                
                task_results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # 混淆矩阵
                cm = confusion_matrix(targets_np, preds)
                task_results['confusion_matrix'] = cm.tolist()
            
            results[task] = task_results
        
        # 保存评估结果
        results_path = os.path.join(self.experiment_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Test evaluation completed")
        return results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history
        }
        
        if is_best:
            torch.save(self.model.state_dict(), os.path.join(self.experiment_dir, 'best_model.pth'))
            torch.save(checkpoint, os.path.join(self.experiment_dir, 'best_checkpoint.pth'))
        
        torch.save(checkpoint, os.path.join(self.experiment_dir, 'latest_checkpoint.pth'))
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        axes[0, 1].plot(self.history['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # 准确率曲线
        for task, accuracies in self.history['val_accuracy'].items():
            axes[1, 0].plot(accuracies, label=task)
        axes[1, 0].set_title('Validation Accuracy by Task')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 综合准确率
        weighted_accuracies = []
        for i in range(len(self.history['val_accuracy']['growth_level'])):
            weighted_acc = sum(
                self.history['val_accuracy'][task][i] * self.model.task_weights[task]
                for task in self.model.task_weights.keys()
            ) / sum(self.model.task_weights.values())
            weighted_accuracies.append(weighted_acc)
        
        axes[1, 1].plot(weighted_accuracies)
        axes[1, 1].set_title('Weighted Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'training_curves.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    # 测试训练器
    logging.basicConfig(level=logging.INFO)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 数据路径
    json_path = "/home/aaa/ws/bioastModel/ds/images/m9e1n170.json"
    image_root = "/home/aaa/ws/bioastModel/ds/images"
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
        json_path=json_path,
        image_root=image_root,
        batch_size=32
    )
    
    # 创建模型
    from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
    model = create_multilevel_mobilenetv3(model_size='small', input_channels=1)
    
    # 创建训练器
    trainer = MultiLevelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_info=label_info,
        device=device,
        experiment_dir='experiments/multilevel_test'
    )
    
    # 训练模型
    history = trainer.train(num_epochs=5)  # 短期测试
    
    # 评估模型
    results = trainer.evaluate()
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    print("Training and evaluation completed!")