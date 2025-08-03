"""
修复版简化气孔检测器训练脚本
基于训练数据分析的改进方案
目标：解决过拟合问题，提升泛化能力
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from datetime import datetime
import logging
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import MICDataLoader, MICDataset

class SimplifiedAirBubbleDetector(nn.Module):
    """简化版气孔检测器 - 解决过拟合问题"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        # 大幅简化的特征提取器 (目标: <100k参数)
        self.features = nn.Sequential(
            # 第一层: 保持分辨率
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 第二层: 轻微下采样
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 35x35
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # 第三层: 特征提取
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            
            # 第四层: 进一步下采样
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 18x18
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 简化的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # 计算参数数量
        self.param_count = sum(p.numel() for p in self.parameters())
        print(f"Simplified model parameters: {self.param_count:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

class ImprovedDataGenerator:
    """改进的数据生成器 - 提高合成数据质量"""
    
    def __init__(self):
        self.bubble_templates = self._generate_bubble_templates()
    
    def _generate_bubble_templates(self):
        """生成多样化的气孔模板"""
        templates = []
        sizes = [3, 4, 5, 6, 7, 8, 10]
        
        for size in sizes:
            # 圆形气孔
            template = np.zeros((size, size))
            center = size // 2
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    if dist <= center:
                        intensity = 1.0 - (dist / center) * 0.5
                        template[y, x] = intensity
            templates.append(template)
        
        return templates
    
    def generate_improved_image(self) -> np.ndarray:
        """生成改进的合成图像"""
        # 基础图像
        image = np.random.normal(0.5, 0.05, (70, 70, 3))
        
        # 添加孔板结构
        center = (35, 35)
        radius = 32
        y, x = np.ogrid[:70, :70]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # 在孔板内添加变化
        image[mask] += np.random.normal(0, 0.02, image[mask].shape)
        
        # 随机添加气孔
        has_bubble = np.random.random() < 0.5  # 50%概率有气孔
        if has_bubble:
            bubble_template = self.bubble_templates[np.random.randint(len(self.bubble_templates))]
            t_h, t_w = bubble_template.shape
            
            # 确保气孔在孔板内
            max_x = min(55, 70 - t_w)
            max_y = min(55, 70 - t_h)
            start_x = np.random.randint(15, max_x)
            start_y = np.random.randint(15, max_y)
            
            # 应用气孔效果
            for ch in range(3):
                region = image[start_y:start_y+t_h, start_x:start_x+t_w, ch]
                image[start_y:start_y+t_h, start_x:start_x+t_w, ch] = np.maximum(region, bubble_template * 0.9)
        
        # 添加轻微噪声
        noise = np.random.normal(0, 0.01, image.shape)
        image += noise
        
        # 确保值在[0, 1]范围内
        image = np.clip(image, 0, 1)
        
        return image.astype(np.float32), int(has_bubble)

class SimplifiedTrainer:
    """简化版训练器 - 专注解决过拟合"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = "experiments/simplified_airbubble_detector"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 创建改进的模型
        self.model = SimplifiedAirBubbleDetector().to(self.device)
        
        # 损失函数 (添加标签平滑)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 优化器 (降低学习率，增加权重衰减)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0005,  # 降低学习率
            weight_decay=1e-3,  # 增加权重衰减
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=1e-6
        )
        
        # 训练配置
        self.batch_size = 32
        self.num_epochs = 30
        self.patience = 8
        
        # 训练历史
        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.save_dir, f"simplified_training_{timestamp}.log")
        
        # 清除之前的处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_data(self):
        """生成改进的训练数据"""
        self.logger.info("Generating improved synthetic data...")
        
        data_generator = ImprovedDataGenerator()
        
        # 生成更多样本
        num_samples = 3000
        images = []
        labels = []
        
        for i in range(num_samples):
            image, label = data_generator.generate_improved_image()
            images.append(image)
            labels.append(label)
            
            if (i + 1) % 500 == 0:
                self.logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        images = np.array(images)
        labels = np.array(labels)
        
        # 确保类别平衡
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        
        min_class_size = min(len(pos_indices), len(neg_indices))
        balanced_indices = np.concatenate([
            pos_indices[:min_class_size],
            neg_indices[:min_class_size]
        ])
        
        images = images[balanced_indices]
        labels = labels[balanced_indices]
        
        # 重新打乱
        shuffle_indices = np.random.permutation(len(images))
        images = images[shuffle_indices]
        labels = labels[shuffle_indices]
        
        self.logger.info(f"Balanced dataset: {len(images)} samples")
        self.logger.info(f"Class distribution: {np.bincount(labels)}")
        
        return images, labels
    
    def prepare_data(self):
        """准备数据"""
        images, labels = self.generate_data()
        
        # 划分数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # 创建数据集
        train_dataset = MICDataset(X_train, y_train)
        val_dataset = MICDataset(X_val, y_val)
        test_dataset = MICDataset(X_test, y_test)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        self.logger.info(f"Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return {
            'loss': total_loss / len(self.train_loader), 
            'accuracy': 100. * correct / total
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        precision = precision_score(all_labels, all_predictions, average='weighted') * 100
        recall = recall_score(all_labels, all_predictions, average='weighted') * 100
        f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self):
        """完整训练流程"""
        self.logger.info("Starting simplified air bubble detector training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.param_count:,}")
        
        # 准备数据
        self.prepare_data()
        
        # 训练循环
        for epoch in range(self.num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['train_acc'].append(train_metrics['accuracy'])
            self.train_history['val_acc'].append(val_metrics['accuracy'])
            self.train_history['learning_rates'].append(current_lr)
            
            # 日志输出
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%"
            )
            self.logger.info(
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Val F1: {val_metrics['f1']:.2f}%"
            )
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # 计算训练/验证差距
            acc_gap = train_metrics['accuracy'] - val_metrics['accuracy']
            self.logger.info(f"Train/Val Gap: {acc_gap:.2f}%")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                self.logger.info(f"New best validation accuracy: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        self.logger.info(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # 测试最佳模型
        self.test_model()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history
        }
        
        checkpoint_path = os.path.join(self.save_dir, "simplified_airbubble_best.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def test_model(self):
        """测试模型性能"""
        self.logger.info("Testing simplified model...")
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = 100. * correct / total
        test_precision = precision_score(all_labels, all_predictions, average='weighted') * 100
        test_recall = recall_score(all_labels, all_predictions, average='weighted') * 100
        test_f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
        
        self.logger.info("Test Results:")
        self.logger.info(f"Accuracy: {test_accuracy:.2f}%")
        self.logger.info(f"Precision: {test_precision:.2f}%")
        self.logger.info(f"Recall: {test_recall:.2f}%")
        self.logger.info(f"F1-Score: {test_f1:.2f}%")
        
        # 与原始模型比较
        improvement = test_accuracy - 52.0
        self.logger.info(f"Improvement over original: {improvement:.2f}%")
        
        # 目标达成情况
        target_achievement = "Achieved" if test_accuracy >= 92.0 else "Not Achieved"
        self.logger.info(f"Target Achievement (92%): {target_achievement}")
        
        # 保存结果
        results = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'improvement_over_original': improvement,
            'target_achievement': test_accuracy >= 92.0,
            'best_val_accuracy': self.best_val_acc
        }
        
        results_path = os.path.join(self.save_dir, "simplified_test_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results

def main():
    """主函数"""
    trainer = SimplifiedTrainer()
    trainer.train()

if __name__ == "__main__":
    main()