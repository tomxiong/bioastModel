#!/usr/bin/env python3
"""
Improved Multi-level Trainer for Bacterial Image Classification
改进版多层分类模型训练器
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
from training.threshold_optimizer import ThresholdOptimizer

logger = logging.getLogger(__name__)

class ImprovedMultiLevelTrainer:
    """改进版多层分类模型训练器"""
    
    def __init__(self,
                 model: MultiLevelMobileNetV3,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 label_info: Dict,
                 device: torch.device,
                 experiment_dir: str,
                 learning_rate: float = 0.002,  # 提高初始学习率
                 weight_decay: float = 0.01,
                 warmup_epochs: int = 5,  # 预热轮数
                 patience: int = 10,  # 减少patience以更早停止
                 interference_class_weights: Optional[List[float]] = None,  # Interference类别权重
                 task_weights: Optional[List[float]] = None,  # 🆕 v0.9.3: 任务权重 [growth_level, growth_pattern, interference]
                 optimize_thresholds: bool = False,  # 是否优化阈值
                 pattern_conditional_pores_loss = None):  # 🆕 v0.10.0: Pattern-Conditional Pores Loss
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.label_info = label_info
        self.device = device
        self.experiment_dir = experiment_dir
        self.warmup_epochs = warmup_epochs
        
        # 创建实验目录
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 设置优化器 - 使用更好的参数
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 改进的学习率调度器 - 使用预热 + 余弦退火
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        self.main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=50,  # 50个epoch的余弦周期
            eta_min=1e-6
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
        
        # 改进的早停机制
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.patience = patience
        self.patience_counter = 0
        self.min_delta = 0.001  # 最小改进阈值

        # 初始化 Interference 类别权重
        if interference_class_weights is None:
            # 默认权重：基于类别不平衡度
            # 顺序: [artifacts, debris, contamination, pores]
            # 不平衡度: [1:12.33, 1:21.55, 1:1499, 1:1.73]
            self.interference_pos_weights = torch.tensor([
                3.0,   # artifacts (1:12.33 → 权重 3.0)
                5.0,   # debris (1:21.55 → 权重 5.0)
                20.0,  # contamination (1:1499 → 权重 20.0)
                1.0    # pores (1:1.73 → 权重 1.0，待分析)
            ]).to(device)
        else:
            self.interference_pos_weights = torch.tensor(
                interference_class_weights
            ).to(device)

        # 创建带权重的 Interference 损失函数
        self.criterion_interference = nn.BCEWithLogitsLoss(
            pos_weight=self.interference_pos_weights,
            reduction='mean'
        )

        # 🆕 v0.9.3: 任务权重
        if task_weights is None:
            # 默认权重 (均衡)
            self.task_weights = [1.0, 1.0, 1.0]
        else:
            self.task_weights = task_weights

        # 🆕 v0.9.3: 初始化阈值优化器
        self.optimize_thresholds = optimize_thresholds
        if self.optimize_thresholds:
            self.threshold_optimizer = ThresholdOptimizer(
                num_classes=4,
                search_range=(0.05, 0.95),
                step=0.05,
                class_names=list(label_info['interference_factors'].keys())
            )
        else:
            self.threshold_optimizer = None

        # 🆕 v0.10.0: Pattern-Conditional Pores Loss
        self.pattern_conditional_pores_loss = pattern_conditional_pores_loss

        logger.info(f"Improved Trainer initialized. Experiment dir: {experiment_dir}")
        logger.info(f"Warmup epochs: {warmup_epochs}, Patience: {patience}")
        logger.info(f"Task weights: {self.task_weights} (growth_level, growth_pattern, interference)")
        logger.info(f"Interference class weights: {self.interference_pos_weights.tolist()}")
        if self.optimize_thresholds:
            logger.info(f"Threshold optimization: ENABLED")
        if self.pattern_conditional_pores_loss:
            logger.info(f"Pattern-Conditional Pores Loss: ENABLED")
    
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
            if self.pattern_conditional_pores_loss:
                # 使用 Pattern-Conditional Interference Loss
                # 手动计算各任务损失
                losses = {}

                # Growth level loss
                if 'growth_level' in outputs and 'growth_level' in targets:
                    losses['growth_level'] = nn.CrossEntropyLoss()(
                        outputs['growth_level'],
                        targets['growth_level']
                    )

                # Growth pattern loss
                if 'growth_pattern' in outputs and 'growth_pattern' in targets:
                    losses['growth_pattern'] = nn.CrossEntropyLoss()(
                        outputs['growth_pattern'],
                        targets['growth_pattern']
                    )

                # Interference factors loss with pattern-conditional weighting
                if 'interference_factors' in outputs and 'interference_factors' in targets:
                    losses['interference_factors'] = self.pattern_conditional_pores_loss(
                        interference_pred=outputs['interference_factors'],
                        interference_target=targets['interference_factors'],
                        pattern_pred=outputs['growth_pattern'],
                        growth_level=targets['growth_level']
                    )

                # 计算总损失
                total_batch_loss = sum(
                    self.task_weights[i] * losses[task]
                    for i, task in enumerate(['growth_level', 'growth_pattern', 'interference_factors'])
                    if task in losses
                )
                losses['total'] = total_batch_loss
            else:
                # 使用默认损失
                custom_criterions = {
                    'interference_factors': self.criterion_interference
                }
                losses = self.model.compute_loss(outputs, targets, custom_criterions)
                total_batch_loss = losses['total']
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度裁剪 - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 累计损失
            total_loss += total_batch_loss.item()
            for task in self.model.num_classes.keys():
                task_losses[task] += losses[task].item()
            
            num_batches += 1
        
        # 计算平均损失
        avg_losses = {
            'total': total_loss / num_batches,
            **{task: loss / num_batches for task, loss in task_losses.items()}
        }
        
        return avg_losses
    
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.model.num_classes.keys()}
        
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
                if self.pattern_conditional_pores_loss:
                    # 使用 Pattern-Conditional Interference Loss
                    losses = {}

                    # Growth level loss
                    if 'growth_level' in outputs and 'growth_level' in targets:
                        losses['growth_level'] = nn.CrossEntropyLoss()(
                            outputs['growth_level'],
                            targets['growth_level']
                        )

                    # Growth pattern loss
                    if 'growth_pattern' in outputs and 'growth_pattern' in targets:
                        losses['growth_pattern'] = nn.CrossEntropyLoss()(
                            outputs['growth_pattern'],
                            targets['growth_pattern']
                        )

                    # Interference factors loss with pattern-conditional weighting
                    if 'interference_factors' in outputs and 'interference_factors' in targets:
                        losses['interference_factors'] = self.pattern_conditional_pores_loss(
                            interference_pred=outputs['interference_factors'],
                            interference_target=targets['interference_factors'],
                            pattern_pred=outputs['growth_pattern'],
                            growth_level=targets['growth_level']
                        )

                    # 计算总损失
                    total_batch_loss = sum(
                        self.task_weights[i] * losses[task]
                        for i, task in enumerate(['growth_level', 'growth_pattern', 'interference_factors'])
                        if task in losses
                    )
                    losses['total'] = total_batch_loss
                else:
                    # 使用默认损失
                    custom_criterions = {
                        'interference_factors': self.criterion_interference
                    }
                    losses = self.model.compute_loss(outputs, targets, custom_criterions)
                    total_batch_loss = losses['total']
                
                # 累计损失
                total_loss += total_batch_loss.item()
                for task in self.model.num_classes.keys():
                    task_losses[task] += losses[task].item()
                
                # 收集预测和目标
                predictions = self.model.predict(images)
                for task in self.model.num_classes.keys():
                    all_predictions[task].append(predictions[task].cpu())
                    all_targets[task].append(targets[task].cpu())
                
                num_batches += 1
        
        # 计算平均损失
        avg_losses = {
            'total': total_loss / num_batches,
            **{task: loss / num_batches for task, loss in task_losses.items()}
        }
        
        # 计算评估指标
        metrics = {}
        for task in self.model.num_classes.keys():
            preds = torch.cat(all_predictions[task], dim=0).numpy()
            targets_np = torch.cat(all_targets[task], dim=0).numpy()

            if task == 'interference_factors':
                # 多标签分类 - 使用 F1 分数而非准确率
                from sklearn.metrics import f1_score
                preds_binary = (preds > 0.5).astype(int)

                # 计算每个类别的 F1 分数
                f1_scores = []
                for i in range(targets_np.shape[1]):
                    f1 = f1_score(
                        targets_np[:, i],
                        preds_binary[:, i],
                        zero_division=0
                    )
                    f1_scores.append(f1)

                # 使用平均 F1 分数作为指标
                metrics[task] = np.mean(f1_scores)
            else:
                # 多分类任务 - 使用准确率
                preds_classes = np.argmax(preds, axis=1)
                metrics[task] = accuracy_score(targets_np, preds_classes)

        return avg_losses, metrics
    
    def train(self, num_epochs: int = 100, save_best: bool = True) -> Dict:
        """训练模型"""
        logger.info(f"Starting improved training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            train_losses = self.train_epoch()
            
            # 验证阶段
            val_losses, val_metrics = self.validate()

            # 学习率调度 - 改进的调度策略
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.main_scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # 计算加权评估指标
            # 注意: interference_factors 现在是 F1 分数，其他是准确率
            # 🆕 v0.9.3: 使用可配置的任务权重
            task_weight_dict = {
                'growth_level': self.task_weights[0],
                'growth_pattern': self.task_weights[1],
                'interference_factors': self.task_weights[2]
            }
            weighted_metric = sum(
                val_metrics[task] * task_weight_dict[task]
                for task in val_metrics.keys()
            ) / sum(task_weight_dict.values())

            # 记录历史
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['learning_rate'].append(current_lr)

            for task, metric in val_metrics.items():
                self.history['val_accuracy'][task].append(metric)

            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/Val', val_losses['total'], epoch)
            self.writer.add_scalar('Metric/Weighted', weighted_metric, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            for task, metric in val_metrics.items():
                # interference_factors 是 F1，其他是准确率
                metric_name = 'F1' if task == 'interference_factors' else 'Accuracy'
                self.writer.add_scalar(f'{metric_name}/{task}', metric, epoch)
            
            # 改进的早停检查
            improvement = weighted_metric - self.best_val_accuracy
            if improvement > self.min_delta:
                self.best_val_accuracy = weighted_metric
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
            logger.info(f"  Val Metrics: {val_metrics}")
            logger.info(f"    (Note: interference_factors is F1 score, others are accuracy)")
            logger.info(f"  Weighted Metric: {weighted_metric:.4f} (best: {self.best_val_accuracy:.4f})")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            logger.info(f"  Patience: {self.patience_counter}/{self.patience}")
            
            # 早停
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                logger.info(f"No improvement for {self.patience} epochs")
                break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch+1}")
        
        # 保存训练历史
        self.save_history()
        
        return self.history
    
    def evaluate_with_thresholds(self, use_optimal_thresholds: bool = True,
                               custom_thresholds: Optional[List[float]] = None) -> Dict:
        """
        使用阈值优化评估模型 (v0.9.3新增功能)

        Args:
            use_optimal_thresholds: 是否使用最优阈值
            custom_thresholds: 自定义阈值 [artifacts, debris, contamination, pores]

        Returns:
            包含优化后指标的字典
        """
        logger.info("Starting threshold-optimized evaluation...")
        self.model.eval()

        all_predictions = {task: [] for task in self.model.num_classes.keys()}
        all_targets = {task: [] for task in self.model.num_classes.keys()}

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = {task: target.to(self.device) for task, target in targets.items()}

                predictions = self.model.predict(images)

                for task in self.model.num_classes.keys():
                    all_predictions[task].append(predictions[task].cpu())
                    all_targets[task].append(targets[task].cpu())

        # 计算详细指标
        results = {}
        for task in self.model.num_classes.keys():
            preds = torch.cat(all_predictions[task], dim=0).numpy()
            targets_np = torch.cat(all_targets[task], dim=0).numpy()

            if task == 'interference_factors':
                # 🆕 v0.9.3: 支持阈值优化的多标签指标
                from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

                # 选择阈值策略
                if use_optimal_thresholds and self.threshold_optimizer:
                    if custom_thresholds:
                        preds_binary = self.threshold_optimizer.predict_with_custom_thresholds(preds, custom_thresholds)
                    else:
                        preds_binary = self.threshold_optimizer.predict_with_optimal_thresholds(preds)
                    used_thresholds = custom_thresholds if custom_thresholds else self.threshold_optimizer.optimal_thresholds
                else:
                    # 使用默认阈值 0.5
                    preds_binary = (preds > 0.5).astype(int)
                    used_thresholds = [0.5, 0.5, 0.5, 0.5]

                task_results = {}
                threshold_info = {}

                # 计算每个类别的详细指标
                for i, factor in enumerate(sorted(self.label_info[task].keys())):
                    acc = accuracy_score(targets_np[:, i], preds_binary[:, i])
                    f1 = f1_score(targets_np[:, i], preds_binary[:, i], zero_division=0)
                    precision, recall, _, _ = precision_recall_fscore_support(
                        targets_np[:, i], preds_binary[:, i], average='binary', zero_division=0
                    )
                    task_results[factor] = {
                        'accuracy': acc,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall
                    }
                    threshold_info[factor] = used_thresholds[i]

                # 整体指标 - 使用 F1 分数
                f1_scores = []
                for i in range(targets_np.shape[1]):
                    f1 = f1_score(targets_np[:, i], preds_binary[:, i], zero_division=0)
                    f1_scores.append(f1)

                task_results['overall_f1'] = np.mean(f1_scores)
                task_results['overall_accuracy'] = np.mean([
                    accuracy_score(targets_np[:, i], preds_binary[:, i])
                    for i in range(targets_np.shape[1])
                ])

                # 添加阈值信息
                task_results['thresholds'] = threshold_info
                task_results['threshold_strategy'] = 'optimal' if use_optimal_thresholds else 'default'
            else:
                # 多分类指标 - 将概率转换为类别预测
                from sklearn.metrics import accuracy_score as acc_score
                preds_classes = np.argmax(preds, axis=1)
                accuracy = acc_score(targets_np, preds_classes)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets_np, preds_classes, average='weighted', zero_division=0
                )

                task_results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }

                # 混淆矩阵
                cm = confusion_matrix(targets_np, preds_classes)
                task_results['confusion_matrix'] = cm.tolist()

            results[task] = task_results

        return results

    def optimize_thresholds_on_validation(self) -> Tuple[List[float], List[float]]:
        """
        在验证集上优化阈值

        Returns:
            optimal_thresholds: 最优阈值列表
            optimal_f1_scores: 对应的F1分数列表
        """
        logger.info("Starting threshold optimization on validation set...")
        self.model.eval()

        all_val_predictions = []
        all_val_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = {task: target.to(self.device) for task, target in targets.items()}

                predictions = self.model.predict(images)

                # 收集 Interference 任务的预测和目标
                all_val_predictions.append(predictions['interference_factors'].cpu())
                all_val_targets.append(targets['interference_factors'].cpu())

        # 合并所有验证数据
        val_predictions = torch.cat(all_val_predictions, dim=0).numpy()
        val_targets = torch.cat(all_val_targets, dim=0).numpy()

        # 使用阈值优化器找到最优阈值
        optimal_thresholds, optimal_f1_scores = self.threshold_optimizer.find_optimal_thresholds(
            val_predictions,
            val_targets,
            verbose=True
        )

        # 保存优化结果
        threshold_save_path = os.path.join(self.experiment_dir, 'optimal_thresholds.json')
        self.threshold_optimizer.save(threshold_save_path)

        logger.info(f"Optimal thresholds saved to {threshold_save_path}")

        return optimal_thresholds, optimal_f1_scores

    def evaluate(self) -> Dict:
        """评估模型"""
        logger.info("Starting model evaluation...")
        self.model.eval()
        
        all_predictions = {task: [] for task in self.model.num_classes.keys()}
        all_targets = {task: [] for task in self.model.num_classes.keys()}
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = {task: target.to(self.device) for task, target in targets.items()}
                
                predictions = self.model.predict(images)
                
                for task in self.model.num_classes.keys():
                    all_predictions[task].append(predictions[task].cpu())
                    all_targets[task].append(targets[task].cpu())
        
        # 计算详细指标
        results = {}
        for task in self.model.num_classes.keys():
            preds = torch.cat(all_predictions[task], dim=0).numpy()
            targets_np = torch.cat(all_targets[task], dim=0).numpy()
            
            if task == 'interference_factors':
                # 多标签指标 - 将概率转换为二进制预测
                from sklearn.metrics import f1_score
                preds_binary = (preds > 0.5).astype(int)
                task_results = {}

                # 计算每个类别的详细指标
                for i, factor in enumerate(sorted(self.label_info[task].keys())):
                    acc = accuracy_score(targets_np[:, i], preds_binary[:, i])
                    f1 = f1_score(targets_np[:, i], preds_binary[:, i], zero_division=0)
                    precision, recall, _, _ = precision_recall_fscore_support(
                        targets_np[:, i], preds_binary[:, i], average='binary', zero_division=0
                    )
                    task_results[factor] = {
                        'accuracy': acc,
                        'f1_score': f1,
                        'precision': precision,
                        'recall': recall
                    }

                # 整体指标 - 使用 F1 分数
                f1_scores = []
                for i in range(targets_np.shape[1]):
                    f1 = f1_score(targets_np[:, i], preds_binary[:, i], zero_division=0)
                    f1_scores.append(f1)

                task_results['overall_f1'] = np.mean(f1_scores)
                task_results['overall_accuracy'] = np.mean([
                    accuracy_score(targets_np[:, i], preds_binary[:, i])
                    for i in range(targets_np.shape[1])
                ])
            else:
                # 多分类指标 - 将概率转换为类别预测
                preds_classes = np.argmax(preds, axis=1)
                accuracy = accuracy_score(targets_np, preds_classes)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets_np, preds_classes, average='weighted', zero_division=0
                )
                
                task_results = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # 混淆矩阵
                cm = confusion_matrix(targets_np, preds_classes)
                task_results['confusion_matrix'] = cm.tolist()
            
            results[task] = task_results
        
        # 保存评估结果
        results_path = os.path.join(self.experiment_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Evaluation completed")
        return results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict(),
            'main_scheduler_state_dict': self.main_scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.experiment_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            logger.info(f"Best model saved at epoch {epoch+1}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.experiment_dir, 'improved_training_history.json')
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
        
        # 准确率曲线
        for task, accuracies in self.history['val_accuracy'].items():
            axes[0, 1].plot(accuracies, label=f'{task}')
        axes[0, 1].set_title('Validation Accuracy by Task')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(self.history['learning_rate'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # 加权准确率
        # 🆕 v0.9.3: 使用可配置的任务权重
        task_weight_dict = {
            'growth_level': self.task_weights[0],
            'growth_pattern': self.task_weights[1],
            'interference_factors': self.task_weights[2]
        }
        weighted_acc = []
        for i in range(len(self.history['val_accuracy']['growth_level'])):
            acc = sum(
                self.history['val_accuracy'][task][i] * task_weight_dict[task]
                for task in task_weight_dict.keys()
            ) / sum(task_weight_dict.values())
            weighted_acc.append(acc)
        
        axes[1, 1].plot(weighted_acc)
        axes[1, 1].set_title('Weighted Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Weighted Accuracy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'improved_training_curves.png'), dpi=300)
        plt.close()
        
        logger.info("Training curves saved")


if __name__ == "__main__":
    # 测试改进版训练器
    logging.basicConfig(level=logging.INFO)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 数据路径
    json_path = "/home/aaa/ws/bioastModel/ds/images/m9e1n170.json"
    image_root = "/home/aaa/ws/bioastModel/ds/images"
    
    # 创建数据加载器 - 增加批次大小
    train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
        json_path=json_path,
        image_root=image_root,
        batch_size=64  # 增加批次大小
    )
    
    # 创建模型
    from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
    model = create_multilevel_mobilenetv3(model_size='small', input_channels=1)
    
    # 创建改进版训练器
    trainer = ImprovedMultiLevelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_info=label_info,
        device=device,
        experiment_dir='experiments/improved_multilevel'
    )
    
    # 训练模型
    history = trainer.train(num_epochs=50)
    
    # 评估模型
    results = trainer.evaluate()
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    print("Improved training and evaluation completed!")