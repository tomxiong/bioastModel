"""
训练增强型气孔检测器
目标：将气孔检测精度从85%提升至92%+
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_airbubble_detector import (
    EnhancedAirBubbleDetector, 
    AirBubbleLoss, 
    PhysicsBasedAugmentation
)
from core.data_loader import MICDataLoader

class AirBubbleTrainingConfig:
    """气孔检测训练配置"""
    
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.weight_decay = 1e-4
        self.patience = 15
        self.min_delta = 0.001
        
        # 损失权重
        self.classification_weight = 1.0
        self.localization_weight = 0.5
        self.uncertainty_weight = 0.1
        
        # 数据增强
        self.use_physics_augmentation = True
        self.augmentation_probability = 0.7
        
        # 模型参数
        self.input_channels = 3
        self.num_classes = 2
        
        # 保存路径
        self.save_dir = "experiments/enhanced_airbubble_detector"
        self.model_name = f"enhanced_airbubble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class EnhancedMICDataset(Dataset):
    """增强型MIC数据集，专门用于气孔检测"""
    
    def __init__(self, data_loader: MICDataLoader, 
                 split: str = 'train',
                 use_augmentation: bool = True,
                 augmentation_prob: float = 0.7):
        self.data_loader = data_loader
        self.split = split
        self.use_augmentation = use_augmentation and (split == 'train')
        self.augmentation_prob = augmentation_prob
        
        if self.use_augmentation:
            self.physics_augmentation = PhysicsBasedAugmentation()
        
        # 加载数据
        if split == 'train':
            self.images, self.labels = data_loader.get_train_data()
        elif split == 'val':
            self.images, self.labels = data_loader.get_val_data()
        else:
            self.images, self.labels = data_loader.get_test_data()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 转换为tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        
        # 确保图像格式正确 (C, H, W)
        if len(image.shape) == 3 and image.shape[0] != 3:
            image = image.permute(2, 0, 1)
        
        # 应用物理模型数据增强
        if self.use_augmentation and np.random.random() < self.augmentation_prob:
            image = self.physics_augmentation.apply_augmentation(image)
        
        # 标准化
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'index': idx
        }

class AirBubbleTrainer:
    """气孔检测训练器"""
    
    def __init__(self, config: AirBubbleTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # 初始化模型
        self.model = EnhancedAirBubbleDetector(
            input_channels=config.input_channels,
            num_classes=config.num_classes
        ).to(self.device)
        
        # 损失函数
        self.criterion = AirBubbleLoss(
            classification_weight=config.classification_weight,
            localization_weight=config.localization_weight,
            uncertainty_weight=config.uncertainty_weight
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.patience // 2,
            min_lr=1e-6
        )
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.config.save_dir, f"{self.config.model_name}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """准备数据"""
        self.logger.info("Preparing data...")
        
        # 加载数据
        data_loader = MICDataLoader()
        
        # 创建数据集
        train_dataset = EnhancedMICDataset(
            data_loader, 
            split='train',
            use_augmentation=self.config.use_physics_augmentation,
            augmentation_prob=self.config.augmentation_probability
        )
        
        val_dataset = EnhancedMICDataset(
            data_loader, 
            split='val',
            use_augmentation=False
        )
        
        test_dataset = EnhancedMICDataset(
            data_loader, 
            split='test',
            use_augmentation=False
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            
            # 计算损失
            targets = {'labels': labels}
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs['classification'].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                self.logger.info(
                    f'Batch {batch_idx}/{len(self.train_loader)}, '
                    f'Loss: {loss.item():.4f}, '
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
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                targets = {'labels': labels}
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs['classification'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # 计算详细指标
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    def train(self):
        """完整训练流程"""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 准备数据
        self.prepare_data()
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step(val_metrics['loss'])
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
                f"Val Precision: {val_metrics['precision']:.2f}%, "
                f"Val Recall: {val_metrics['recall']:.2f}%, "
                f"Val F1: {val_metrics['f1']:.2f}%"
            )
            self.logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                self.logger.info(f"New best validation accuracy: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        # 训练完成
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # 最终测试
        self.test_model()
        
        # 保存训练历史
        self.save_training_history()
        
        # 生成训练报告
        self.generate_training_report()
    
    def test_model(self):
        """测试模型性能"""
        self.logger.info("Testing model...")
        
        # 加载最佳模型
        best_model_path = os.path.join(self.config.save_dir, f"{self.config.model_name}_best.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs['classification'].data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_uncertainties.extend(outputs['uncertainty'].cpu().numpy())
        
        test_accuracy = 100. * correct / total
        test_precision = precision_score(all_labels, all_predictions, average='weighted') * 100
        test_recall = recall_score(all_labels, all_predictions, average='weighted') * 100
        test_f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
        
        self.logger.info(f"Test Results:")
        self.logger.info(f"Accuracy: {test_accuracy:.2f}%")
        self.logger.info(f"Precision: {test_precision:.2f}%")
        self.logger.info(f"Recall: {test_recall:.2f}%")
        self.logger.info(f"F1-Score: {test_f1:.2f}%")
        self.logger.info(f"Average Uncertainty: {np.mean(all_uncertainties):.4f}")
        
        return {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'uncertainty': np.mean(all_uncertainties)
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'train_history': self.train_history
        }
        
        if is_best:
            checkpoint_path = os.path.join(self.config.save_dir, f"{self.config.model_name}_best.pth")
        else:
            checkpoint_path = os.path.join(self.config.save_dir, f"{self.config.model_name}_epoch_{epoch}.pth")
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.save_dir, f"{self.config.model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        self.logger.info(f"Training history saved: {history_path}")
    
    def generate_training_report(self):
        """生成训练报告"""
        # 绘制训练曲线
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(self.train_history['val_acc'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(self.train_history['learning_rates'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # 性能提升
        baseline_acc = 85.0  # 基线气孔检测精度
        current_best = max(self.train_history['val_acc'])
        improvement = current_best - baseline_acc
        
        axes[1, 1].bar(['Baseline', 'Enhanced'], [baseline_acc, current_best], 
                      color=['red', 'green'], alpha=0.7)
        axes[1, 1].set_title('Air Bubble Detection Accuracy Improvement')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].text(1, current_best + 1, f'+{improvement:.1f}%', 
                       ha='center', va='bottom', fontweight='bold')
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.config.save_dir, f"{self.config.model_name}_training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved: {plot_path}")
        
        # 生成文本报告
        report_path = os.path.join(self.config.save_dir, f"{self.config.model_name}_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Enhanced Air Bubble Detector Training Report\n\n")
            f.write(f"**Model**: {self.config.model_name}\n")
            f.write(f"**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Device**: {self.device}\n\n")
            
            f.write(f"## Configuration\n")
            f.write(f"- Batch Size: {self.config.batch_size}\n")
            f.write(f"- Learning Rate: {self.config.learning_rate}\n")
            f.write(f"- Epochs: {len(self.train_history['train_loss'])}\n")
            f.write(f"- Physics Augmentation: {self.config.use_physics_augmentation}\n\n")
            
            f.write(f"## Results\n")
            f.write(f"- Best Validation Accuracy: {self.best_val_acc:.2f}%\n")
            f.write(f"- Baseline Accuracy: 85.0%\n")
            f.write(f"- Improvement: +{self.best_val_acc - 85.0:.1f}%\n")
            f.write(f"- Target Achievement: {'✅ Achieved' if self.best_val_acc >= 92.0 else '❌ Not Achieved'} (Target: 92%+)\n\n")
            
            f.write(f"## Training Progress\n")
            f.write(f"- Final Train Loss: {self.train_history['train_loss'][-1]:.4f}\n")
            f.write(f"- Final Val Loss: {self.train_history['val_loss'][-1]:.4f}\n")
            f.write(f"- Final Train Accuracy: {self.train_history['train_acc'][-1]:.2f}%\n")
            f.write(f"- Final Val Accuracy: {self.train_history['val_acc'][-1]:.2f}%\n\n")
            
            f.write(f"## Model Architecture\n")
            f.write(f"- Input Channels: {self.config.input_channels}\n")
            f.write(f"- Number of Classes: {self.config.num_classes}\n")
            f.write(f"- Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"- Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n\n")
            
            f.write(f"## Next Steps\n")
            if self.best_val_acc >= 92.0:
                f.write(f"✅ **Target Achieved!** The enhanced air bubble detector has successfully reached the target accuracy of 92%+.\n\n")
                f.write(f"**Recommended Actions:**\n")
                f.write(f"1. Deploy the model for integration testing\n")
                f.write(f"2. Proceed to Phase 1 Week 3-4: False Negative Control Optimization\n")
                f.write(f"3. Collect more diverse test data for robustness validation\n")
                f.write(f"4. Begin preparation for multi-task learning framework\n")
            else:
                f.write(f"❌ **Target Not Achieved.** Current accuracy: {self.best_val_acc:.2f}%, Target: 92%+\n\n")
                f.write(f"**Recommended Actions:**\n")
                f.write(f"1. Increase training data with more air bubble examples\n")
                f.write(f"2. Fine-tune physics-based augmentation parameters\n")
                f.write(f"3. Experiment with different loss function weights\n")
                f.write(f"4. Consider ensemble methods or model architecture adjustments\n")
        
        self.logger.info(f"Training report saved: {report_path}")

def main():
    """主函数"""
    # 创建配置
    config = AirBubbleTrainingConfig()
    
    # 创建训练器
    trainer = AirBubbleTrainer(config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
