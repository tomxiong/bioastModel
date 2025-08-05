"""
训练工具类
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Optional, Dict, Any

class EarlyStopping:
    """早停工具类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        """
        Args:
            patience: 耐心值，连续多少个epoch没有改善就停止
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def should_stop(self, val_loss: float) -> bool:
        """
        检查是否应该停止训练
        
        Args:
            val_loss: 当前验证损失
            
        Returns:
            是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def update(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        更新早停状态
        
        Args:
            val_loss: 当前验证损失
            model: 模型
            
        Returns:
            是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
            return False

class ModelCheckpoint:
    """模型检查点保存工具类"""
    
    def __init__(self, save_dir: str, monitor: str = 'val_accuracy', 
                 mode: str = 'max', save_best_only: bool = True,
                 filename: str = 'best_model.pth'):
        """
        Args:
            save_dir: 保存目录
            monitor: 监控指标
            mode: 模式，'max'表示越大越好，'min'表示越小越好
            save_best_only: 是否只保存最佳模型
            filename: 文件名
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename = filename
        
        if mode == 'max':
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best
        else:
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best
    
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, val_accuracy: float, val_loss: float,
             additional_info: Optional[Dict[str, Any]] = None):
        """
        保存模型检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            val_accuracy: 验证准确率
            val_loss: 验证损失
            additional_info: 额外信息
        """
        # 确定当前分数
        if self.monitor == 'val_accuracy':
            current_score = val_accuracy
        elif self.monitor == 'val_loss':
            current_score = val_loss
        else:
            current_score = val_accuracy  # 默认使用准确率
        
        # 检查是否需要保存
        should_save = not self.save_best_only or self.is_better(current_score, self.best_score)
        
        if should_save:
            if self.is_better(current_score, self.best_score):
                self.best_score = current_score
            
            # 准备保存数据
            save_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'best_score': self.best_score,
                'monitor': self.monitor,
                'mode': self.mode
            }
            
            if additional_info:
                save_data.update(additional_info)
            
            # 保存文件
            save_path = self.save_dir / self.filename
            torch.save(save_data, save_path)
            
            print(f"模型已保存: {save_path} (epoch {epoch}, {self.monitor}: {current_score:.4f})")

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        指标字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 计算基本指标
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = correct / total
    
    # 计算每个类别的指标
    unique_labels = np.unique(y_true)
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    
    for label in unique_labels:
        # True Positive, False Positive, False Negative
        tp = ((y_true == label) & (y_pred == label)).sum()
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        
        # 计算精确率、召回率、F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class[f'precision_class_{label}'] = precision
        recall_per_class[f'recall_class_{label}'] = recall
        f1_per_class[f'f1_class_{label}'] = f1
    
    # 宏平均
    macro_precision = np.mean(list(precision_per_class.values()))
    macro_recall = np.mean(list(recall_per_class.values()))
    macro_f1 = np.mean(list(f1_per_class.values()))
    
    # 组合所有指标
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        **precision_per_class,
        **recall_per_class,
        **f1_per_class
    }
    
    return metrics

class LearningRateScheduler:
    """学习率调度器包装类"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler_type: str = 'cosine',
                 **scheduler_kwargs):
        """
        Args:
            optimizer: 优化器
            scheduler_type: 调度器类型
            **scheduler_kwargs: 调度器参数
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_kwargs
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **scheduler_kwargs
            )
        elif scheduler_type == 'reduce':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_kwargs
            )
        elif scheduler_type == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, **scheduler_kwargs
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """更新学习率"""
        if self.scheduler_type == 'reduce' and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """获取最后的学习率"""
        return self.scheduler.get_last_lr()

class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            self.metrics[key] = value
    
    def get_current(self, key: str) -> float:
        """获取当前指标值"""
        return self.metrics.get(key, 0.0)
    
    def get_history(self, key: str) -> list:
        """获取指标历史"""
        return self.history.get(key, [])
    
    def get_best(self, key: str, mode: str = 'max') -> float:
        """获取最佳指标值"""
        history = self.get_history(key)
        if not history:
            return 0.0
        
        if mode == 'max':
            return max(history)
        else:
            return min(history)
    
    def save(self, filepath: str):
        """保存指标历史"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: str):
        """加载指标历史"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        
        # 更新当前指标为最新值
        for key, values in self.history.items():
            if values:
                self.metrics[key] = values[-1]

if __name__ == "__main__":
    # 测试早停
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    losses = [1.0, 0.8, 0.7, 0.69, 0.68, 0.67, 0.66]
    for i, loss in enumerate(losses):
        should_stop = early_stopping.should_stop(loss)
        print(f"Epoch {i+1}: Loss {loss:.2f}, Should stop: {should_stop}")
        if should_stop:
            break
    
    # 测试指标计算
    y_true = torch.tensor([0, 1, 0, 1, 1, 0])
    y_pred = torch.tensor([0, 1, 1, 1, 0, 0])
    
    metrics = calculate_metrics(y_true, y_pred)
    print("\n指标测试:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")