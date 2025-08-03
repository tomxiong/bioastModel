"""
假阴性控制优化脚本
目标：将假阴性率从2.43%降至1.5%以下
重点：实施不平衡学习策略和优化损失函数权重
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_airbubble_detector import EnhancedAirBubbleDetector
from core.data_loader import MICDataLoader

class FalseNegativeOptimizer:
    """假阴性控制优化器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self.setup_logging()
        
        # 初始化模型
        self.model = EnhancedAirBubbleDetector().to(self.device)
        
        # 加载预训练的气孔检测器
        if 'pretrained_model_path' in config:
            self.load_pretrained_model(config['pretrained_model_path'])
        
        # 优化器配置
        self.setup_optimizer()
        
        # 损失函数配置
        self.setup_loss_functions()
        
        # 数据加载器
        self.setup_data_loaders()
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_fnr': [],
            'val_fnr': [],
            'train_fpr': [],
            'val_fpr': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_fnr = float('inf')
        self.patience_counter = 0
    
    def setup_logging(self):
        """设置日志系统"""
        log_dir = "experiments/false_negative_optimization"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"fn_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_pretrained_model(self, model_path: str):
        """加载预训练的气孔检测器"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded pretrained model from {model_path}")
        else:
            self.logger.warning(f"Pretrained model not found at {model_path}")
    
    def setup_optimizer(self):
        """设置优化器"""
        # 使用较小的学习率进行微调
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.0001),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    
    def setup_loss_functions(self):
        """设置损失函数"""
        # 计算类别权重以处理不平衡数据
        data_loader = MICDataLoader()
        train_images, train_labels = data_loader.get_train_data()
        
        # 计算类别权重
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        
        # 额外增加正类（有气孔）的权重以减少假阴性
        positive_class_boost = self.config.get('positive_class_boost', 2.0)
        if len(class_weights) > 1:
            class_weights[1] *= positive_class_boost  # 假设1是正类
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        self.logger.info(f"Class weights: {class_weights}")
        
        # 主要损失函数：加权交叉熵
        self.primary_loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # 辅助损失函数：Focal Loss用于处理困难样本
        self.focal_loss = FocalLoss(
            alpha=class_weights_tensor,
            gamma=self.config.get('focal_gamma', 2.0)
        )
        
        # 假阴性惩罚损失
        self.fn_penalty_loss = FalseNegativePenaltyLoss(
            penalty_weight=self.config.get('fn_penalty_weight', 5.0)
        )
    
    def setup_data_loaders(self):
        """设置数据加载器"""
        data_loader = MICDataLoader()
        
        # 获取训练数据
        train_images, train_labels = data_loader.get_train_data()
        val_images, val_labels = data_loader.get_val_data()
        test_images, test_labels = data_loader.get_test_data()
        
        # 创建加权采样器以平衡类别
        train_class_counts = np.bincount(train_labels)
        train_weights = 1.0 / train_class_counts[train_labels]
        
        # 增加正类样本的采样权重
        positive_boost = self.config.get('positive_sampling_boost', 1.5)
        train_weights[train_labels == 1] *= positive_boost
        
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
        
        # 创建数据集
        train_dataset = MICDataset(train_images, train_labels, augment=True)
        val_dataset = MICDataset(val_images, val_labels, augment=False)
        test_dataset = MICDataset(test_images, test_labels, augment=False)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
    
    def compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算详细指标"""
        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # 计算各种指标
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # 假阴性率和假阳性率
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            # 多类别情况
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted')
            recall = recall_score(labels, predictions, average='weighted')
            f1 = f1_score(labels, predictions, average='weighted')
            fnr = 1 - recall  # 简化计算
            fpr = 0  # 多类别情况下不易计算
            specificity = 0
        
        return {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'specificity': specificity * 100,
            'f1': f1 * 100,
            'fnr': fnr * 100,
            'fpr': fpr * 100
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            logits = outputs['classification']
            
            # 计算多个损失
            primary_loss = self.primary_loss(logits, labels)
            focal_loss = self.focal_loss(logits, labels)
            fn_penalty = self.fn_penalty_loss(logits, labels)
            
            # 组合损失
            total_loss_batch = (
                self.config.get('primary_loss_weight', 1.0) * primary_loss +
                self.config.get('focal_loss_weight', 0.5) * focal_loss +
                self.config.get('fn_penalty_weight', 0.3) * fn_penalty
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += total_loss_batch.item()
            _, predicted = torch.max(logits.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {total_loss_batch.item():.4f}'
                )
        
        # 计算指标
        metrics = self.compute_metrics(np.array(all_predictions), np.array(all_labels))
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                logits = outputs['classification']
                
                # 计算损失
                loss = self.primary_loss(logits, labels)
                total_loss += loss.item()
                
                # 预测
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算指标
        metrics = self.compute_metrics(np.array(all_predictions), np.array(all_labels))
        metrics['loss'] = total_loss / len(self.val_loader)
        
        # 计算AUC（如果是二分类）
        if len(np.unique(all_labels)) == 2:
            all_probabilities = np.array(all_probabilities)
            if all_probabilities.shape[1] == 2:
                auc = roc_auc_score(all_labels, all_probabilities[:, 1])
                metrics['auc'] = auc * 100
        
        return metrics
    
    def optimize(self):
        """执行假阴性优化"""
        self.logger.info("Starting False Negative Optimization...")
        self.logger.info(f"Target: Reduce FNR from 2.43% to < 1.5%")
        
        num_epochs = self.config.get('num_epochs', 50)
        patience = self.config.get('patience', 10)
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step(val_metrics['fnr'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_fnr'].append(train_metrics['fnr'])
            self.training_history['val_fnr'].append(val_metrics['fnr'])
            self.training_history['train_fpr'].append(train_metrics['fpr'])
            self.training_history['val_fpr'].append(val_metrics['fpr'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            
            # 日志输出
            self.logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.2f}%, "
                f"FNR: {train_metrics['fnr']:.2f}%, "
                f"FPR: {train_metrics['fpr']:.2f}%"
            )
            self.logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.2f}%, "
                f"FNR: {val_metrics['fnr']:.2f}%, "
                f"FPR: {val_metrics['fpr']:.2f}%"
            )
            if 'auc' in val_metrics:
                self.logger.info(f"Val AUC: {val_metrics['auc']:.2f}%")
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型（基于FNR）
            if val_metrics['fnr'] < self.best_fnr:
                self.best_fnr = val_metrics['fnr']
                self.patience_counter = 0
                self.save_best_model(epoch, val_metrics)
                self.logger.info(f"New best FNR: {self.best_fnr:.2f}%")
                
                # 检查是否达到目标
                if val_metrics['fnr'] <= 1.5:
                    self.logger.info("🎉 TARGET ACHIEVED! FNR <= 1.5%")
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # 最终测试
        self.final_evaluation()
        
        # 生成报告
        self.generate_optimization_report()
    
    def save_best_model(self, epoch: int, metrics: Dict[str, float]):
        """保存最佳模型"""
        save_dir = "experiments/false_negative_optimization"
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, "best_fn_optimized_model.pth")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, model_path)
        self.logger.info(f"Best model saved: {model_path}")
    
    def final_evaluation(self):
        """最终评估"""
        self.logger.info("Performing final evaluation on test set...")
        
        # 加载最佳模型
        model_path = "experiments/false_negative_optimization/best_fn_optimized_model.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                logits = outputs['classification']
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 计算最终指标
        final_metrics = self.compute_metrics(np.array(all_predictions), np.array(all_labels))
        
        self.logger.info("=== FINAL TEST RESULTS ===")
        self.logger.info(f"Accuracy: {final_metrics['accuracy']:.2f}%")
        self.logger.info(f"Precision: {final_metrics['precision']:.2f}%")
        self.logger.info(f"Recall: {final_metrics['recall']:.2f}%")
        self.logger.info(f"F1-Score: {final_metrics['f1']:.2f}%")
        self.logger.info(f"False Negative Rate: {final_metrics['fnr']:.2f}%")
        self.logger.info(f"False Positive Rate: {final_metrics['fpr']:.2f}%")
        
        # 目标达成检查
        baseline_fnr = 2.43
        target_fnr = 1.5
        improvement = baseline_fnr - final_metrics['fnr']
        
        self.logger.info(f"\n=== TARGET ACHIEVEMENT ===")
        self.logger.info(f"Baseline FNR: {baseline_fnr:.2f}%")
        self.logger.info(f"Current FNR: {final_metrics['fnr']:.2f}%")
        self.logger.info(f"Improvement: -{improvement:.2f}%")
        self.logger.info(f"Target FNR: ≤ {target_fnr:.2f}%")
        
        if final_metrics['fnr'] <= target_fnr:
            self.logger.info("✅ TARGET ACHIEVED!")
        else:
            self.logger.info("❌ Target not achieved, further optimization needed")
        
        return final_metrics
    
    def generate_optimization_report(self):
        """生成优化报告"""
        report_dir = "experiments/false_negative_optimization"
        os.makedirs(report_dir, exist_ok=True)
        
        # 绘制训练曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # FNR曲线
        axes[0, 0].plot(self.training_history['train_fnr'], label='Train FNR', color='red')
        axes[0, 0].plot(self.training_history['val_fnr'], label='Val FNR', color='blue')
        axes[0, 0].axhline(y=1.5, color='green', linestyle='--', label='Target (1.5%)')
        axes[0, 0].axhline(y=2.43, color='orange', linestyle='--', label='Baseline (2.43%)')
        axes[0, 0].set_title('False Negative Rate Optimization')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('FNR (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # FPR曲线
        axes[0, 1].plot(self.training_history['train_fpr'], label='Train FPR', color='red')
        axes[0, 1].plot(self.training_history['val_fpr'], label='Val FPR', color='blue')
        axes[0, 1].set_title('False Positive Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('FPR (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 准确率曲线
        axes[1, 0].plot(self.training_history['train_acc'], label='Train Accuracy', color='red')
        axes[1, 0].plot(self.training_history['val_acc'], label='Val Accuracy', color='blue')
        axes[1, 0].set_title('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 损失曲线
        axes[1, 1].plot(self.training_history['train_loss'], label='Train Loss', color='red')
        axes[1, 1].plot(self.training_history['val_loss'], label='Val Loss', color='blue')
        axes[1, 1].set_title('Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(report_dir, "fn_optimization_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Optimization curves saved: {plot_path}")

class FocalLoss(nn.Module):
    """Focal Loss用于处理困难样本"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FalseNegativePenaltyLoss(nn.Module):
    """假阴性惩罚损失"""
    
    def __init__(self, penalty_weight=5.0):
        super().__init__()
        self.penalty_weight = penalty_weight
    
    def forward(self, inputs, targets):
        # 计算预测概率
        probs = torch.softmax(inputs, dim=1)
        
        # 找到真正例（targets == 1）
        positive_mask = (targets == 1)
        
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=inputs.device)
        
        # 对于正样本，惩罚预测为负类的概率
        positive_probs = probs[positive_mask]
        negative_predictions = positive_probs[:, 0]  # 预测为负类的概率
        
        # 惩罚损失：预测为负类的概率越高，损失越大
        penalty_loss = self.penalty_weight * negative_predictions.mean()
        
        return penalty_loss

class MICDataset(torch.utils.data.Dataset):
    """简化的MIC数据集"""
    
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 转换为tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # 确保图像格式正确
        if len(image.shape) == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        
        # 简单的数据增强
        if self.augment:
            if np.random.random() > 0.5:
                image = torch.flip(image, [2])  # 水平翻转
            if np.random.random() > 0.5:
                image = torch.flip(image, [1])  # 垂直翻转
        
        # 标准化
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        return image, torch.tensor(label, dtype=torch.long)

def main():
    """主函数"""
    config = {
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 50,
        'patience': 10,
        'positive_class_boost': 2.0,
        'positive_sampling_boost': 1.5,
        'focal_gamma': 2.0,
        'fn_penalty_weight': 5.0,
        'primary_loss_weight': 1.0,
        'focal_loss_weight': 0.5,
        'fn_penalty_weight': 0.3,
        'pretrained_model_path': 'experiments/enhanced_airbubble_detector/enhanced_airbubble_*_best.pth'
    }
    
    # 创建优化器
    optimizer = FalseNegativeOptimizer(config)
    
    # 开始优化
    optimizer.optimize()

if __name__ == "__main__":
    main()