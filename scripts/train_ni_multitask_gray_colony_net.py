#!/usr/bin/env python3
"""
训练NI多任务GrayColonyNet模型
基于dataset_ni_multitask数据集进行多任务学习
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_gray_colony_net import create_multitask_gray_colony_net
from training.ni_multitask_dataset import create_ni_dataloaders
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score, roc_auc_score

class MultitaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, 
                 task_weights: Dict[str, float] = None,
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        # 任务权重
        self.task_weights = task_weights or {
            'growth_level': 1.0,
            'growth_pattern': 0.8,
            'interference_factors': 0.6,
            'fine_grained': 1.2
        }
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # 是否使用Focal Loss
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """Focal Loss for handling class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            
        Returns:
            total_loss, individual_losses
        """
        losses = {}
        
        # 1. 生长级别损失
        if self.use_focal_loss:
            losses['growth_level'] = self.focal_loss(
                predictions['growth_level'], 
                targets['growth_level'],
                alpha=self.focal_alpha,
                gamma=self.focal_gamma
            )
        else:
            losses['growth_level'] = self.ce_loss(
                predictions['growth_level'], 
                targets['growth_level']
            )
        
        # 2. 生长模式损失
        losses['growth_pattern'] = self.ce_loss(
            predictions['growth_pattern'],
            targets['growth_pattern']
        )
        
        # 3. 干扰因素损失 (多标签)
        losses['interference_factors'] = self.bce_loss(
            predictions['interference_mapping'],
            targets['interference_factors']
        )
        
        # 4. 精细分类损失
        if 'fine_grained_refined' in predictions:
            # 使用融合后的精细分类输出
            fine_pred = predictions['fine_grained_refined']
        else:
            fine_pred = predictions['fine_grained']
            
        if self.use_focal_loss:
            losses['fine_grained'] = self.focal_loss(
                fine_pred,
                targets['fine_grained'],
                alpha=self.focal_alpha * 1.5,  # 精细分类更难，增加alpha
                gamma=self.focal_gamma
            )
        else:
            losses['fine_grained'] = self.ce_loss(fine_pred, targets['fine_grained'])
        
        # 5. 辅助损失 (如果有)
        if 'pore_confidence' in predictions and 'has_pores' in targets:
            # 气孔置信度辅助损失
            pore_target = targets['has_pores'].float().unsqueeze(1)
            losses['pore_auxiliary'] = F.binary_cross_entropy(
                predictions['pore_confidence'], pore_target
            ) * 0.1  # 辅助损失权重较小
        
        # 计算总损失
        total_loss = 0
        for task, loss in losses.items():
            if task in self.task_weights:
                total_loss += self.task_weights[task] * loss
            else:
                total_loss += loss * 0.1  # 辅助损失默认权重
        
        return total_loss, losses


class MultitaskTrainer:
    """多任务训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 experiment_dir: str):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader 
        self.test_loader = test_loader
        self.device = device
        
        # 实验目录
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志设置
        self.setup_logging()
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.experiment_dir / 'tensorboard')
        
        # 损失函数
        self.criterion = MultitaskLoss(use_focal_loss=True)
        
        # 最佳指标跟踪
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'epoch': 0
        }
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
    def setup_logging(self):
        """设置日志"""
        log_file = self.experiment_dir / 'training.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, optimizer: optim.Optimizer) -> Tuple[float, Dict[str, float]]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = defaultdict(float)
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # 移动到设备
            images = images.to(self.device)
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    targets[key] = value.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss, individual_losses = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            for task, task_loss in individual_losses.items():
                task_losses[task] += task_loss.item()
            
            # 收集预测和目标用于指标计算
            self._collect_predictions(outputs, targets, all_predictions, all_targets)
            
            # 打印进度
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_task_losses = {task: loss/len(self.train_loader) for task, loss in task_losses.items()}
        
        # 计算训练指标
        train_metrics = self._compute_metrics(all_predictions, all_targets)
        
        return avg_loss, avg_task_losses, train_metrics
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = defaultdict(float)
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # 移动到设备
                images = images.to(self.device)
                for key, value in targets.items():
                    if isinstance(value, torch.Tensor):
                        targets[key] = value.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                loss, individual_losses = self.criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item()
                for task, task_loss in individual_losses.items():
                    task_losses[task] += task_loss.item()
                
                # 收集预测和目标
                self._collect_predictions(outputs, targets, all_predictions, all_targets)
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_task_losses = {task: loss/len(self.val_loader) for task, loss in task_losses.items()}
        
        # 计算验证指标
        val_metrics = self._compute_metrics(all_predictions, all_targets)
        
        return avg_loss, avg_task_losses, val_metrics
    
    def _collect_predictions(self, outputs, targets, all_predictions, all_targets):
        """收集预测和真实标签用于指标计算"""
        # 生长级别
        pred_growth_level = outputs['growth_level'].argmax(dim=1).cpu().numpy()
        all_predictions['growth_level'].extend(pred_growth_level)
        all_targets['growth_level'].extend(targets['growth_level'].cpu().numpy())
        
        # 生长模式
        pred_growth_pattern = outputs['growth_pattern'].argmax(dim=1).cpu().numpy()
        all_predictions['growth_pattern'].extend(pred_growth_pattern)
        all_targets['growth_pattern'].extend(targets['growth_pattern'].cpu().numpy())
        
        # 精细分类
        fine_output = outputs.get('fine_grained_refined', outputs['fine_grained'])
        pred_fine_grained = fine_output.argmax(dim=1).cpu().numpy()
        all_predictions['fine_grained'].extend(pred_fine_grained)
        all_targets['fine_grained'].extend(targets['fine_grained'].cpu().numpy())
        
        # 干扰因素 (多标签)
        pred_interference = torch.sigmoid(outputs['interference_mapping']).detach().cpu().numpy()
        all_predictions['interference_factors'].extend(pred_interference)
        all_targets['interference_factors'].extend(targets['interference_factors'].cpu().numpy())
    
    def _compute_metrics(self, all_predictions, all_targets) -> Dict[str, float]:
        """计算各任务的评估指标"""
        metrics = {}
        
        # 单标签分类任务
        for task in ['growth_level', 'growth_pattern', 'fine_grained']:
            if task in all_predictions:
                y_true = np.array(all_targets[task])
                y_pred = np.array(all_predictions[task])
                
                # 准确率
                acc = accuracy_score(y_true, y_pred)
                metrics[f'{task}_accuracy'] = acc
                
                # 精确率、召回率、F1
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted', zero_division=0
                )
                metrics[f'{task}_precision'] = precision
                metrics[f'{task}_recall'] = recall  
                metrics[f'{task}_f1'] = f1
        
        # 多标签分类 (干扰因素)
        if 'interference_factors' in all_predictions:
            y_true = np.array(all_targets['interference_factors'])
            y_pred_prob = np.array(all_predictions['interference_factors'])
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # 计算每个标签的Average Precision
            try:
                ap_scores = []
                for i in range(y_true.shape[1]):
                    if y_true[:, i].sum() > 0:  # 只有正样本才计算AP
                        ap = average_precision_score(y_true[:, i], y_pred_prob[:, i])
                        ap_scores.append(ap)
                
                if ap_scores:
                    metrics['interference_factors_mAP'] = np.mean(ap_scores)
                else:
                    metrics['interference_factors_mAP'] = 0.0
            except:
                metrics['interference_factors_mAP'] = 0.0
        
        # 计算综合准确率
        task_accuracies = [metrics.get(f'{task}_accuracy', 0) 
                          for task in ['growth_level', 'growth_pattern', 'fine_grained']]
        metrics['overall_accuracy'] = np.mean(task_accuracies)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, optimizer: optim.Optimizer, 
                       val_loss: float, val_metrics: Dict[str, float], 
                       is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'history': self.history,
            'best_metrics': self.best_metrics
        }
        
        # 保存最新检查点
        latest_path = self.experiment_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，保存最佳检查点
        if is_best:
            best_path = self.experiment_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"✓ 保存最佳模型 (epoch {epoch})")
    
    def train(self, num_epochs: int = 100, learning_rate: float = 0.001):
        """开始训练"""
        self.logger.info("=== 开始多任务训练 ===")
        self.logger.info(f"训练集: {len(self.train_loader)} 批次")
        self.logger.info(f"验证集: {len(self.val_loader)} 批次")
        self.logger.info(f"设备: {self.device}")
        
        # 优化器和学习率调度器
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # 训练循环
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss, train_task_losses, train_metrics = self.train_epoch(optimizer)
            
            # 验证
            val_loss, val_task_losses, val_metrics = self.validate_epoch()
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            
            for task, loss in train_task_losses.items():
                self.writer.add_scalar(f'Task_Loss_Train/{task}', loss, epoch)
            
            for task, loss in val_task_losses.items():
                self.writer.add_scalar(f'Task_Loss_Val/{task}', loss, epoch)
            
            for metric, value in val_metrics.items():
                self.writer.add_scalar(f'Metrics_Val/{metric}', value, epoch)
            
            # 检查是否是最佳模型
            is_best = val_metrics['overall_accuracy'] > self.best_metrics['val_accuracy']
            if is_best:
                self.best_metrics['val_loss'] = val_loss
                self.best_metrics['val_accuracy'] = val_metrics['overall_accuracy']
                self.best_metrics['epoch'] = epoch
            
            # 保存检查点
            self.save_checkpoint(epoch, optimizer, val_loss, val_metrics, is_best)
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)\n"
                f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                f"  Val Accuracy: {val_metrics['overall_accuracy']:.4f}\n"
                f"  Growth Level Acc: {val_metrics.get('growth_level_accuracy', 0):.4f}\n"
                f"  Growth Pattern Acc: {val_metrics.get('growth_pattern_accuracy', 0):.4f}\n"
                f"  Fine Grained Acc: {val_metrics.get('fine_grained_accuracy', 0):.4f}\n"
                f"  Interference mAP: {val_metrics.get('interference_factors_mAP', 0):.4f}\n"
                f"  LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # 每10个epoch保存一次历史
            if (epoch + 1) % 10 == 0:
                self.save_training_history()
        
        self.logger.info(f"\n✓ 训练完成!")
        self.logger.info(f"最佳验证准确率: {self.best_metrics['val_accuracy']:.4f} (epoch {self.best_metrics['epoch']})")
        
        # 保存最终历史
        self.save_training_history()
        self.writer.close()
    
    def save_training_history(self):
        """保存训练历史"""
        history_file = self.experiment_dir / 'training_history.json'
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)


def test_model_performance(model: nn.Module, 
                          test_loader: DataLoader, 
                          device: torch.device,
                          save_dir: str) -> Dict[str, Any]:
    """测试模型性能并生成详细报告"""
    model.eval()
    
    all_predictions = defaultdict(list)
    all_targets = defaultdict(list)
    error_samples = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    targets[key] = value.to(device)
            
            outputs = model(images)
            
            # 收集预测结果
            batch_predictions = {}
            
            # 生长级别
            growth_level_pred = outputs['growth_level'].argmax(dim=1)
            batch_predictions['growth_level'] = growth_level_pred.cpu().numpy()
            all_predictions['growth_level'].extend(batch_predictions['growth_level'])
            all_targets['growth_level'].extend(targets['growth_level'].cpu().numpy())
            
            # 生长模式
            growth_pattern_pred = outputs['growth_pattern'].argmax(dim=1)
            batch_predictions['growth_pattern'] = growth_pattern_pred.cpu().numpy()
            all_predictions['growth_pattern'].extend(batch_predictions['growth_pattern'])
            all_targets['growth_pattern'].extend(targets['growth_pattern'].cpu().numpy())
            
            # 精细分类
            fine_output = outputs.get('fine_grained_refined', outputs['fine_grained'])
            fine_grained_pred = fine_output.argmax(dim=1)
            batch_predictions['fine_grained'] = fine_grained_pred.cpu().numpy()
            all_predictions['fine_grained'].extend(batch_predictions['fine_grained'])
            all_targets['fine_grained'].extend(targets['fine_grained'].cpu().numpy())
            
            # 记录错误样本
            for i in range(len(images)):
                sample_errors = {}
                sample_errors['image_id'] = targets['image_id'][i]
                sample_errors['panoramic_id'] = targets['panoramic_id'][i]
                
                # 检查各任务是否预测错误
                if batch_predictions['growth_level'][i] != targets['growth_level'][i].cpu().item():
                    sample_errors['growth_level_error'] = {
                        'predicted': int(batch_predictions['growth_level'][i]),
                        'actual': int(targets['growth_level'][i].cpu().item())
                    }
                
                if batch_predictions['fine_grained'][i] != targets['fine_grained'][i].cpu().item():
                    sample_errors['fine_grained_error'] = {
                        'predicted': int(batch_predictions['fine_grained'][i]),
                        'actual': int(targets['fine_grained'][i].cpu().item())
                    }
                
                if sample_errors.get('growth_level_error') or sample_errors.get('fine_grained_error'):
                    error_samples.append(sample_errors)
    
    # 计算详细指标
    test_report = {}
    
    # 单标签分类指标
    for task in ['growth_level', 'growth_pattern', 'fine_grained']:
        y_true = np.array(all_targets[task])
        y_pred = np.array(all_predictions[task])
        
        # 准确率
        acc = accuracy_score(y_true, y_pred)
        test_report[f'{task}_accuracy'] = acc
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        test_report[f'{task}_confusion_matrix'] = cm.tolist()
        
        # 分类报告
        try:
            cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            test_report[f'{task}_classification_report'] = cls_report
        except:
            test_report[f'{task}_classification_report'] = {}
    
    # 总体指标
    test_report['overall_accuracy'] = np.mean([
        test_report['growth_level_accuracy'],
        test_report['growth_pattern_accuracy'], 
        test_report['fine_grained_accuracy']
    ])
    
    test_report['total_samples'] = len(all_targets['growth_level'])
    test_report['error_samples_count'] = len(error_samples)
    test_report['error_rate'] = len(error_samples) / len(all_targets['growth_level'])
    
    # 保存测试报告
    save_path = Path(save_dir)
    
    # 保存详细报告
    report_file = save_path / 'test_performance_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)
    
    # 保存错误样本
    error_file = save_path / 'error_samples_analysis.json'
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_samples, f, ensure_ascii=False, indent=2)
    
    return test_report, error_samples


def main():
    """主训练函数"""
    print("=== NI多任务GrayColonyNet训练 ===")
    
    # 配置
    config = {
        'data_root': '/home/aaa/ws/bioastModel/dataset_ni_multitask',
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'num_workers': 4,
        'feature_dim': 128,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"使用设备: {config['device']}")
    
    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"experiments/ni_multitask_gray_colony_net_{timestamp}"
    
    # 检查数据集是否存在
    if not Path(config['data_root']).exists():
        print(f"错误: 找不到数据集目录 {config['data_root']}")
        print("请先运行 scripts/create_ni_dataset_splits.py 创建数据集")
        return
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = create_ni_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 创建模型
    print("创建多任务GrayColonyNet模型...")
    model = create_multitask_gray_colony_net(
        feature_dim=config['feature_dim'],
        enable_background_filter=True,
        dropout_rate=0.2
    )
    
    model_info = model.get_model_info()
    print(f"模型参数量: {model_info['trainable_parameters']:,}")
    print(f"任务配置: {model_info['tasks']}")
    
    # 创建训练器
    trainer = MultitaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config['device'],
        experiment_dir=experiment_dir
    )
    
    # 保存配置
    config_file = Path(experiment_dir) / 'config.json'
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        # 转换device为字符串以便JSON序列化
        config_copy = config.copy()
        config_copy['device'] = str(config['device'])
        json.dump(config_copy, f, ensure_ascii=False, indent=2)
    
    # 开始训练
    trainer.train(
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate']
    )
    
    # 加载最佳模型进行测试
    print("\n=== 加载最佳模型进行测试 ===")
    best_checkpoint_path = Path(experiment_dir) / 'best_checkpoint.pth'
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 加载最佳模型 (epoch {checkpoint['epoch']})")
    
    # 测试模型性能
    print("测试模型性能...")
    test_report, error_samples = test_model_performance(
        model, test_loader, config['device'], experiment_dir
    )
    
    print(f"\n=== 测试结果 ===")
    print(f"总体准确率: {test_report['overall_accuracy']:.4f}")
    print(f"生长级别准确率: {test_report['growth_level_accuracy']:.4f}")
    print(f"生长模式准确率: {test_report['growth_pattern_accuracy']:.4f}")
    print(f"精细分类准确率: {test_report['fine_grained_accuracy']:.4f}")
    print(f"错误样本数: {test_report['error_samples_count']}/{test_report['total_samples']} ({test_report['error_rate']:.2%})")
    
    print(f"\n✓ 训练完成!")
    print(f"实验目录: {experiment_dir}")
    print(f"测试报告: {experiment_dir}/test_performance_report.json")
    print(f"错误样本分析: {experiment_dir}/error_samples_analysis.json")
    
    return experiment_dir


if __name__ == "__main__":
    experiment_dir = main()