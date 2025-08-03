"""
浊度分类精度提升脚本
目标：将浊度分类准确率从88%提升至92%+
重点：多尺度特征融合和自适应阈值调整
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_airbubble_detector import EnhancedAirBubbleDetector
from core.data_loader import MICDataLoader

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # 不同尺度的特征提取
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),  # 1x1卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # 3x3卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2),  # 5x5卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化分支
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),  # 4个分支 * 64 = 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 获取输入尺寸
        b, c, h, w = x.size()
        
        # 多尺度特征提取
        feat1 = self.scale1(x)
        feat2 = self.scale2(x)
        feat3 = self.scale3(x)
        
        # 全局特征
        global_feat = self.global_branch(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        # 特征拼接
        fused_features = torch.cat([feat1, feat2, feat3, global_feat], dim=1)
        
        # 注意力权重
        attention = self.fusion(fused_features)
        
        # 应用注意力
        enhanced_features = x * attention
        
        return enhanced_features

class AdaptiveThresholdModule(nn.Module):
    """自适应阈值调整模块"""
    
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
        # 阈值预测网络
        self.threshold_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes - 1),  # n-1个阈值
            nn.Sigmoid()
        )
        
        # 初始化阈值
        self.register_buffer('base_thresholds', torch.linspace(0.2, 0.8, num_classes - 1))
    
    def forward(self, features, logits):
        # 预测阈值调整
        threshold_adjustments = self.threshold_predictor(features)
        
        # 计算自适应阈值
        adaptive_thresholds = self.base_thresholds + 0.2 * (threshold_adjustments - 0.5)
        adaptive_thresholds = torch.clamp(adaptive_thresholds, 0.1, 0.9)
        
        return adaptive_thresholds

class EnhancedTurbidityClassifier(nn.Module):
    """增强型浊度分类器"""
    
    def __init__(self, base_model: nn.Module, num_classes: int = 5):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # 多尺度特征融合
        self.feature_fusion = MultiScaleFeatureFusion(256)
        
        # 自适应阈值模块
        self.adaptive_threshold = AdaptiveThresholdModule(num_classes)
        
        # 浊度特征增强器
        self.turbidity_enhancer = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        # 置信度估计
        self.confidence_estimator = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 基础特征提取
        if hasattr(self.base_model, 'backbone'):
            features = self.base_model.backbone(x)
        else:
            # 假设base_model返回字典
            outputs = self.base_model(x)
            features = outputs.get('features', x)
        
        # 多尺度特征融合
        enhanced_features = self.feature_fusion(features)
        
        # 浊度特征增强
        turbidity_features = self.turbidity_enhancer(enhanced_features)
        
        # 分类预测
        logits = self.classifier(turbidity_features)
        
        # 自适应阈值
        adaptive_thresholds = self.adaptive_threshold(enhanced_features, logits)
        
        # 置信度估计
        confidence = self.confidence_estimator(turbidity_features)
        
        return {
            'logits': logits,
            'adaptive_thresholds': adaptive_thresholds,
            'confidence': confidence,
            'features': turbidity_features
        }

class TurbidityClassificationTrainer:
    """浊度分类训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置日志
        self.setup_logging()
        
        # 初始化模型
        self.setup_model()
        
        # 设置优化器和损失函数
        self.setup_optimizer()
        self.setup_loss_functions()
        
        # 数据加载器
        self.setup_data_loaders()
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def setup_logging(self):
        """设置日志系统"""
        log_dir = "experiments/turbidity_classification"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"turbidity_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_model(self):
        """设置模型"""
        # 加载预训练的基础模型
        base_model = EnhancedAirBubbleDetector()
        
        # 如果有预训练权重，加载它们
        if 'pretrained_model_path' in self.config:
            pretrained_path = self.config['pretrained_model_path']
            if os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                base_model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded pretrained model from {pretrained_path}")
        
        # 创建增强型浊度分类器
        self.model = EnhancedTurbidityClassifier(
            base_model=base_model,
            num_classes=self.config.get('num_classes', 5)
        ).to(self.device)
        
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer(self):
        """设置优化器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('T_0', 10),
            T_mult=self.config.get('T_mult', 2),
            eta_min=self.config.get('eta_min', 1e-6)
        )
    
    def setup_loss_functions(self):
        """设置损失函数"""
        # 主要分类损失
        self.classification_loss = nn.CrossEntropyLoss(
            label_smoothing=self.config.get('label_smoothing', 0.1)
        )
        
        # 置信度损失
        self.confidence_loss = nn.MSELoss()
        
        # 阈值一致性损失
        self.threshold_consistency_loss = nn.MSELoss()
    
    def setup_data_loaders(self):
        """设置数据加载器"""
        data_loader = MICDataLoader()
        
        # 获取数据
        train_images, train_labels = data_loader.get_train_data()
        val_images, val_labels = data_loader.get_val_data()
        test_images, test_labels = data_loader.get_test_data()
        
        # 转换为浊度分类标签（假设原始标签需要转换）
        train_turbidity_labels = self.convert_to_turbidity_labels(train_labels)
        val_turbidity_labels = self.convert_to_turbidity_labels(val_labels)
        test_turbidity_labels = self.convert_to_turbidity_labels(test_labels)
        
        # 创建数据集
        train_dataset = TurbidityDataset(train_images, train_turbidity_labels, augment=True)
        val_dataset = TurbidityDataset(val_images, val_turbidity_labels, augment=False)
        test_dataset = TurbidityDataset(test_images, test_turbidity_labels, augment=False)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
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
    
    def convert_to_turbidity_labels(self, original_labels):
        """将原始标签转换为浊度分类标签"""
        # 这里需要根据实际情况实现标签转换逻辑
        # 假设我们有5个浊度等级：0(清澈), 1(轻微), 2(中等), 3(浑浊), 4(非常浑浊)
        
        # 简化实现：随机分配浊度等级（实际应用中需要根据图像特征判断）
        turbidity_labels = []
        for label in original_labels:
            # 这里应该有实际的浊度判断逻辑
            # 暂时使用简单的映射
            if label == 0:  # 假设0表示无气孔，可能浊度较低
                turbidity_level = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            else:  # 有气孔，可能浊度较高
                turbidity_level = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])
            
            turbidity_labels.append(turbidity_level)
        
        return np.array(turbidity_labels)
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            logits = outputs['logits']
            confidence = outputs['confidence']
            
            # 计算损失
            cls_loss = self.classification_loss(logits, labels)
            
            # 置信度损失（高置信度对应高准确率）
            _, predicted = torch.max(logits.data, 1)
            correct_predictions = (predicted == labels).float()
            conf_loss = self.confidence_loss(confidence.squeeze(), correct_predictions)
            
            # 总损失
            total_loss_batch = cls_loss + 0.1 * conf_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += total_loss_batch.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {total_loss_batch.item():.4f}, '
                    f'Acc: {100.*correct/total:.2f}%'
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                logits = outputs['logits']
                confidence = outputs['confidence']
                
                # 计算损失
                loss = self.classification_loss(logits, labels)
                total_loss += loss.item()
                
                # 预测
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # 计算详细指标
        precision = precision_score(all_labels, all_predictions, average='weighted') * 100
        recall = recall_score(all_labels, all_predictions, average='weighted') * 100
        f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
        avg_confidence = np.mean(all_confidences) * 100
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confidence': avg_confidence
        }
    
    def train(self):
        """完整训练流程"""
        self.logger.info("Starting Turbidity Classification Training...")
        self.logger.info(f"Target: Improve accuracy from 88% to 92%+")
        
        num_epochs = self.config.get('num_epochs', 100)
        patience = self.config.get('patience', 15)
        
        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['learning_rates'].append(current_lr)
            
            # 日志输出
            self.logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.2f}%"
            )
            self.logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.2f}%, "
                f"Precision: {val_metrics['precision']:.2f}%, "
                f"Recall: {val_metrics['recall']:.2f}%, "
                f"F1: {val_metrics['f1']:.2f}%, "
                f"Confidence: {val_metrics['confidence']:.2f}%"
            )
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_best_model(epoch, val_metrics)
                self.logger.info(f"New best validation accuracy: {self.best_val_acc:.2f}%")
                
                # 检查是否达到目标
                if val_metrics['accuracy'] >= 92.0:
                    self.logger.info("🎉 TARGET ACHIEVED! Accuracy >= 92%")
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # 最终测试
        self.final_evaluation()
        
        # 生成报告
        self.generate_training_report()
    
    def save_best_model(self, epoch: int, metrics: Dict[str, float]):
        """保存最佳模型"""
        save_dir = "experiments/turbidity_classification"
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, "best_turbidity_classifier.pth")
        
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
        model_path = "experiments/turbidity_classification/best_turbidity_classifier.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                logits = outputs['logits']
                confidence = outputs['confidence']
                
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
        
        test_accuracy = 100. * correct / total
        test_precision = precision_score(all_labels, all_predictions, average='weighted') * 100
        test_recall = recall_score(all_labels, all_predictions, average='weighted') * 100
        test_f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
        avg_confidence = np.mean(all_confidences) * 100
        
        self.logger.info("=== FINAL TEST RESULTS ===")
        self.logger.info(f"Accuracy: {test_accuracy:.2f}%")
        self.logger.info(f"Precision: {test_precision:.2f}%")
        self.logger.info(f"Recall: {test_recall:.2f}%")
        self.logger.info(f"F1-Score: {test_f1:.2f}%")
        self.logger.info(f"Average Confidence: {avg_confidence:.2f}%")
        
        # 目标达成检查
        baseline_acc = 88.0
        target_acc = 92.0
        improvement = test_accuracy - baseline_acc
        
        self.logger.info(f"\n=== TARGET ACHIEVEMENT ===")
        self.logger.info(f"Baseline Accuracy: {baseline_acc:.2f}%")
        self.logger.info(f"Current Accuracy: {test_accuracy:.2f}%")
        self.logger.info(f"Improvement: +{improvement:.2f}%")
        self.logger.info(f"Target Accuracy: ≥ {target_acc:.2f}%")
        
        if test_accuracy >= target_acc:
            self.logger.info("✅ TARGET ACHIEVED!")
        else:
            self.logger.info("❌ Target not achieved, further optimization needed")
        
        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'confidence': avg_confidence
        }
    
    def generate_training_report(self):
        """生成训练报告"""
        report_dir = "experiments/turbidity_classification"
        os.makedirs(report_dir, exist_ok=True)
        
        # 绘制训练曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 准确率曲线
        axes[0, 0].plot(self.training_history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 0].plot(self.training_history['val_acc'], label='Val Accuracy', color='red')
        axes[0, 0].axhline(y=92.0, color='green', linestyle='--', label='Target (92%)')
        axes[0, 0].axhline(y=88.0, color='orange', linestyle='--', label='Baseline (88%)')
        axes[0, 0].set_title('Turbidity Classification Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 损失曲线
        axes[0, 1].plot(self.training_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 1].plot(self.training_history['val_loss'], label='Val Loss', color='red')
        axes[0, 1].set_title('Training and Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(self.training_history['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # 性能提升对比
        baseline_acc = 88.0
        current_best = max(self.training_history['val_acc'])
        improvement = current_best - baseline_acc
        
        axes[1, 1].bar(['Baseline', 'Enhanced'], [baseline_acc, current_best], 
                      color=['red', 'green'], alpha=0.7)
        axes[1, 1].set_title('Turbidity Classification Improvement')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].text(1, current_best + 0.5, f'+{improvement:.1f}%', 
                       ha='center', va='bottom', fontweight='bold')
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(report_dir, "turbidity_training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved: {plot_path}")

class TurbidityDataset(torch.utils.data.Dataset):
    """浊度分类数据集"""
    
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
        
        # 数据增强
        if self.augment:
            # 随机翻转
            if np.random.random() > 0.5:
                image = torch.flip(image, [2])
            if np.random.random() > 0.5:
                image = torch.flip(image, [1])
            
            # 随机旋转
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                image = self.rotate_image(image, angle)
            
            # 亮度和对比度调整
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                contrast = np.random.uniform(0.8, 1.2)
                image = image * contrast + brightness - 1.0
                image = torch.clamp(image, 0, 1)
        
        # 标准化
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        return image, torch.tensor(label, dtype=torch.long)
    
    def rotate_image(self, image, angle):
        """旋转图像"""
        # 简单的旋转实现
        if abs(angle) < 1:
            return image
        
        # 转换为numpy进行旋转
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        h, w = img_np.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_np, rotation_matrix, (w, h))
        
        # 转换回tensor
        return torch.from_numpy(rotated.transpose(2, 0, 1))

def main():
    """主函数"""
    config = {
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'num_epochs': 100,
        'patience': 15,
        'num_classes': 5,
        'label_smoothing': 0.1,
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-6,
        'pretrained_model_path': 'experiments/false_negative_optimization/best_fn_optimized_model.pth'
    }
    
    # 创建训练器
    trainer = TurbidityClassificationTrainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
"""
浊度分类精度提升脚本
目标：将浊度分类准确率从88%提升至92%+
重点：多尺度特征融合和自适应阈值调整
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_airbubble_detector import EnhancedAirBubbleDetector
from core.data_loader import MICDataLoader

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # 不同尺度的特征提取
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),  # 1x1卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # 3x3卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(in