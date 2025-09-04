"""
多任务训练流水线
支持多个分类任务的联合训练
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss, f1_score

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.multitask_dataset import MultitaskBioastDataset, create_multitask_dataloaders
from core.config.model_configs import get_model_config


class MultitaskLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.task_weights = task_weights or {
            'growth_level': 1.0,
            'growth_pattern': 1.0,
            'interference_mapping': 0.5,
            'fine_grained': 1.0
        }
        
        # 任务特定的损失函数
        self.criterion = {
            'growth_level': nn.CrossEntropyLoss(),
            'growth_pattern': nn.CrossEntropyLoss(),
            'interference_mapping': nn.BCEWithLogitsLoss(),
            'fine_grained': nn.CrossEntropyLoss()
        }
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        losses = {}
        
        for task_name in outputs.keys():
            task_output = outputs[task_name]
            task_target = targets[task_name]
            
            # 计算任务损失
            if task_name == 'interference_mapping':
                # 多标签损失
                task_loss = self.criterion[task_name](task_output, task_target.float())
            else:
                # 单标签损失
                task_loss = self.criterion[task_name](task_output, task_target)
            
            # 加权累加
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * task_loss
            losses[task_name] = task_loss.item()
        
        losses['total'] = total_loss.item()
        return total_loss, losses


class MultitaskTrainer:
    """多任务训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any]):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练配置
        self.epochs = config.get('epochs', 100)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.use_amp = config.get('use_amp', True)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        
        # 初始化组件
        self.criterion = MultitaskLoss(config.get('task_weights'))
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 日志和记录
        self.logger = self._setup_logger()
        self.writer = SummaryWriter(config.get('log_dir', 'runs/multitask'))
        
        # 指标历史
        self.train_history = {task: [] for task in model.heads.keys()}
        self.val_history = {task: [] for task in model.heads.keys()}
        self.loss_history = {'train': [], 'val': []}
        
        # 最佳模型
        self.best_val_score = 0.0
        self.best_model_path = None
        
        print(f"训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"使用AMP: {self.use_amp}")
        print(f"任务权重: {self.criterion.task_weights}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.epochs,
            eta_min=self.learning_rate * 0.01
        )
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('MultitaskTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {task: [] for task in self.model.heads.keys()}
        epoch_losses['total'] = []
        
        # 任务指标收集
        task_predictions = {task: [] for task in self.model.heads.keys()}
        task_targets = {task: [] for task in self.model.heads.keys()}
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss, task_losses = self.criterion(outputs, targets)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.gradient_clip
                    )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss, task_losses = self.criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.gradient_clip
                    )
                
                # 更新参数
                self.optimizer.step()
            
            # 记录损失
            epoch_losses['total'].append(task_losses['total'])
            for task_name, task_loss in task_losses.items():
                if task_name != 'total':
                    epoch_losses[task_name].append(task_loss)
            
            # 收集预测结果用于计算指标
            for task_name in outputs.keys():
                if task_name == 'interference_mapping':
                    # 多标签预测
                    preds = torch.sigmoid(outputs[task_name]) > 0.5
                else:
                    # 单标签预测
                    preds = outputs[task_name].argmax(dim=1)
                
                task_predictions[task_name].extend(preds.cpu().numpy())
                task_targets[task_name].extend(targets[task_name].cpu().numpy())
            
            # 日志输出
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                    f'Loss: {task_losses["total"]:.4f}'
                )
        
        # 计算epoch平均损失
        avg_losses = {
            task: np.mean(losses) 
            for task, losses in epoch_losses.items()
        }
        
        # 计算各任务指标
        epoch_metrics = self._calculate_metrics(task_predictions, task_targets)
        
        return avg_losses, epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        epoch_losses = {task: [] for task in self.model.heads.keys()}
        epoch_losses['total'] = []
        
        # 任务指标收集
        task_predictions = {task: [] for task in self.model.heads.keys()}
        task_targets = {task: [] for task in self.model.heads.keys()}
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                outputs = self.model(images)
                loss, task_losses = self.criterion(outputs, targets)
                
                # 记录损失
                epoch_losses['total'].append(task_losses['total'])
                for task_name, task_loss in task_losses.items():
                    if task_name != 'total':
                        epoch_losses[task_name].append(task_loss)
                
                # 收集预测结果
                for task_name in outputs.keys():
                    if task_name == 'interference_mapping':
                        # 多标签预测
                        preds = torch.sigmoid(outputs[task_name]) > 0.5
                    else:
                        # 单标签预测
                        preds = outputs[task_name].argmax(dim=1)
                    
                    task_predictions[task_name].extend(preds.cpu().numpy())
                    task_targets[task_name].extend(targets[task_name].cpu().numpy())
        
        # 计算平均损失
        avg_losses = {
            task: np.mean(losses) 
            for task, losses in epoch_losses.items()
        }
        
        # 计算指标
        epoch_metrics = self._calculate_metrics(task_predictions, task_targets)
        
        return avg_losses, epoch_metrics
    
    def _calculate_metrics(self, predictions: Dict[str, List], 
                          targets: Dict[str, List]) -> Dict[str, Dict]:
        """计算各任务的指标"""
        metrics = {}
        
        for task_name in predictions.keys():
            if task_name == 'interference_mapping':
                # 多标签指标
                metrics[task_name] = self._calculate_multilabel_metrics(
                    predictions[task_name], 
                    targets[task_name]
                )
            else:
                # 单标签指标
                metrics[task_name] = self._calculate_singlelabel_metrics(
                    predictions[task_name], 
                    targets[task_name]
                )
        
        return metrics
    
    def _calculate_singlelabel_metrics(self, preds: List, targets: List) -> Dict:
        """计算单标签分类指标"""
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _calculate_multilabel_metrics(self, preds: List, targets: List) -> Dict:
        """计算多标签分类指标"""
        # 转换为numpy数组
        preds = np.array(preds)
        targets = np.array(targets)
        
        # Hamming Loss
        hamming = hamming_loss(targets, preds)
        
        # F1分数
        f1_micro = f1_score(targets, preds, average='micro', zero_division=0)
        f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
        
        # 精确率和召回率
        precision_micro = f1_score(targets, preds, average='micro', zero_division=0)
        recall_micro = f1_score(targets, preds, average='micro', zero_division=0)
        
        return {
            'hamming_loss': hamming,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro
        }
    
    def train(self):
        """完整训练流程"""
        self.logger.info("开始多任务训练...")
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # 训练
            train_losses, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_losses, val_metrics = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.loss_history['train'].append(train_losses['total'])
            self.loss_history['val'].append(val_losses['total'])
            
            # 计算综合得分（加权平均F1）
            val_score = self._calculate_composite_score(val_metrics)
            
            # 保存最佳模型
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self._save_model(epoch, 'best')
            
            # 定期保存检查点
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                self._save_model(epoch, f'checkpoint_{epoch}')
            
            # 日志输出
            epoch_time = time.time() - start_time
            self.logger.info(
                f'Epoch {epoch}/{self.epochs} [{epoch_time:.1f}s] '
                f'Train Loss: {train_losses["total"]:.4f} '
                f'Val Loss: {val_losses["total"]:.4f} '
                f'Val Score: {val_score:.4f} '
                f'Best: {self.best_val_score:.4f}'
            )
            
            # TensorBoard记录
            self._log_to_tensorboard(epoch, train_losses, val_losses, 
                                     train_metrics, val_metrics)
            
            # 打印各任务指标
            self._log_task_metrics(epoch, train_metrics, val_metrics)
        
        self.logger.info("训练完成!")
        self._save_final_report()
        
        return {
            'best_val_score': self.best_val_score,
            'loss_history': self.loss_history,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """计算综合得分"""
        # 使用加权平均F1分数
        weights = {
            'growth_level': 0.3,
            'growth_pattern': 0.3,
            'interference_mapping': 0.2,
            'fine_grained': 0.2
        }
        
        composite_score = 0.0
        for task_name, task_metrics in metrics.items():
            if task_name == 'interference_mapping':
                task_score = task_metrics['f1_micro']
            else:
                task_score = task_metrics['f1_score']
            
            composite_score += weights.get(task_name, 0.25) * task_score
        
        return composite_score
    
    def _save_model(self, epoch: int, suffix: str):
        """保存模型"""
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        
        model_path = f'checkpoints/multitask_model_{suffix}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, model_path)
        
        if suffix == 'best':
            self.best_model_path = model_path
    
    def _log_to_tensorboard(self, epoch: int, train_losses: Dict, 
                           val_losses: Dict, train_metrics: Dict, val_metrics: Dict):
        """记录到TensorBoard"""
        # 损失
        self.writer.add_scalar('Loss/Train', train_losses['total'], epoch)
        self.writer.add_scalar('Loss/Val', val_losses['total'], epoch)
        
        # 学习率
        self.writer.add_scalar('Learning_Rate', 
                              self.optimizer.param_groups[0]['lr'], epoch)
        
        # 各任务损失和指标
        for task_name in self.model.heads.keys():
            # 损失
            self.writer.add_scalar(f'Loss/{task_name}/Train', 
                                  train_losses[task_name], epoch)
            self.writer.add_scalar(f'Loss/{task_name}/Val', 
                                  val_losses[task_name], epoch)
            
            # 指标
            if task_name == 'interference_mapping':
                self.writer.add_scalar(f'Metrics/{task_name}/Val_F1_Micro', 
                                      val_metrics[task_name]['f1_micro'], epoch)
                self.writer.add_scalar(f'Metrics/{task_name}/Val_Hamming_Loss', 
                                      val_metrics[task_name]['hamming_loss'], epoch)
            else:
                self.writer.add_scalar(f'Metrics/{task_name}/Val_Accuracy', 
                                      val_metrics[task_name]['accuracy'], epoch)
                self.writer.add_scalar(f'Metrics/{task_name}/Val_F1', 
                                      val_metrics[task_name]['f1_score'], epoch)
    
    def _log_task_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """打印各任务指标"""
        print(f"\nEpoch {epoch} 任务指标:")
        print("-" * 60)
        print(f"{'Task':<20} {'Train Acc':<10} {'Val Acc':<10} {'Val F1':<10}")
        print("-" * 60)
        
        for task_name in self.model.heads.keys():
            if task_name == 'interference_mapping':
                train_acc = train_metrics[task_name].get('precision_micro', 0)
                val_acc = val_metrics[task_name].get('precision_micro', 0)
                val_f1 = val_metrics[task_name].get('f1_micro', 0)
            else:
                train_acc = train_metrics[task_name].get('accuracy', 0)
                val_acc = val_metrics[task_name].get('accuracy', 0)
                val_f1 = val_metrics[task_name].get('f1_score', 0)
            
            print(f"{task_name:<20} {train_acc:<10.3f} {val_acc:<10.3f} {val_f1:<10.3f}")
        print("-" * 60)
    
    def _save_final_report(self):
        """保存最终报告"""
        report = {
            'config': self.config,
            'best_val_score': self.best_val_score,
            'loss_history': self.loss_history,
            'final_train_metrics': self.train_history[-1] if self.train_history else {},
            'final_val_metrics': self.val_history[-1] if self.val_history else {},
            'best_model_path': self.best_model_path
        }
        
        report_path = f'reports/multitask_training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"训练报告已保存: {report_path}")


# 使用示例
if __name__ == "__main__":
    # 配置示例
    config = {
        'epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'use_amp': True,
        'gradient_clip': 1.0,
        'task_weights': {
            'growth_level': 1.0,
            'growth_pattern': 1.0,
            'interference_mapping': 0.5,
            'fine_grained': 1.0
        },
        'log_dir': 'runs/multitask_experiment'
    }
    
    # 创建数据加载器
    annotation_file = "bioast_dataset/annotations/multitask_annotations.json"
    image_root = "bioast_dataset/images"
    
    dataloaders = create_multitask_dataloaders(
        annotation_file=annotation_file,
        image_root=image_root,
        batch_size=config['batch_size']
    )
    
    # 创建模型（这里需要根据实际模型架构实现）
    # model = create_multitask_model(...)
    
    # 创建训练器
    # trainer = MultitaskTrainer(
    #     model=model,
    #     train_loader=dataloaders['train'],
    #     val_loader=dataloaders['val'],
    #     config=config
    # )
    
    # 开始训练
    # results = trainer.train()
    
    print("多任务训练示例")