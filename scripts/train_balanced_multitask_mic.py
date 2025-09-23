"""
针对阴性气孔识别和阳性样本干扰因素检测的改进训练脚本
专门解决当前模型在阳性样本气孔检测失效和其他干扰因素未学会的问题
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

# 添加项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.multitask_mic_mobilenetv3 import create_multitask_mic_mobilenetv3
from training.enhanced_multitask_dataset import create_multitask_dataloaders

class BalancedMultiTaskLoss(nn.Module):
    """平衡的多任务损失函数，专门解决阳性样本气孔检测问题"""
    
    def __init__(self, task_weights=None, focal_params=None):
        super().__init__()
        
        # 调整权重，重点关注干扰因素平衡
        if task_weights is None:
            task_weights = {
                'classification': 1.0,       
                'aux_classification': 0.3,   
                'growth_pattern': 0.6,       # 降低生长模式权重
                'interference_factors': 1.0, # 大幅提升干扰因素权重
                'quality': 0.2               
            }
        
        self.task_weights = task_weights
        
        # Focal Loss参数
        if focal_params is None:
            focal_params = {'alpha': 0.75, 'gamma': 2.0}
        
        self.focal_alpha = focal_params['alpha']
        self.focal_gamma = focal_params['gamma']
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        # 针对干扰因素的平衡损失权重
        # 根据数据分布设置权重 [pores, artifacts, debris, contamination]
        self.interference_weights = torch.tensor([1.0, 5.0, 8.0, 20.0])  # 对稀少类别给予更高权重
        
    def focal_loss(self, inputs, targets):
        """Focal Loss实现"""
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def balanced_bce_loss(self, inputs, targets):
        """平衡的多标签二分类损失"""
        # 动态调整权重到正确设备
        weights = self.interference_weights.to(inputs.device)
        
        # 为每个样本计算加权损失
        losses = []
        for i in range(inputs.size(1)):  # 遍历每个干扰因素
            factor_input = inputs[:, i]
            factor_target = targets[:, i]
            
            # 计算该因素的损失
            factor_loss = nn.BCEWithLogitsLoss(reduction='none')(factor_input, factor_target)
            
            # 应用类别权重
            weighted_loss = factor_loss * weights[i]
            losses.append(weighted_loss.mean())
        
        return torch.stack(losses).mean()
    
    def stratified_interference_loss(self, inputs, targets, main_labels):
        """分层的干扰因素损失：分别处理阴性和阳性样本"""
        negative_mask = (main_labels == 0)
        positive_mask = (main_labels == 1)
        
        total_loss = 0.0
        
        # 阴性样本的干扰因素损失 (权重1.0)
        if negative_mask.sum() > 0:
            neg_inputs = inputs[negative_mask]
            neg_targets = targets[negative_mask]
            neg_loss = self.balanced_bce_loss(neg_inputs, neg_targets)
            total_loss += 1.0 * neg_loss
        
        # 阳性样本的干扰因素损失 (权重2.0，强化学习)
        if positive_mask.sum() > 0:
            pos_inputs = inputs[positive_mask]
            pos_targets = targets[positive_mask]
            pos_loss = self.balanced_bce_loss(pos_inputs, pos_targets)
            total_loss += 2.0 * pos_loss  # 对阳性样本给予更高权重
        
        return total_loss
    
    def forward(self, outputs, targets):
        """计算平衡的多任务损失"""
        total_loss = 0.0
        loss_components = {}
        
        # 获取主分类标签用于分层
        main_labels = targets['classification']
        
        # 1. 主分类任务 (Focal Loss)
        if 'classification' in outputs and 'classification' in targets:
            loss = self.focal_loss(outputs['classification'], targets['classification'])
            weighted_loss = self.task_weights['classification'] * loss
            total_loss += weighted_loss
            loss_components['classification'] = loss.item()
        
        # 2. 辅助分类任务
        if 'aux_classification' in outputs and 'aux_classification' in targets:
            loss = self.ce_loss(outputs['aux_classification'], targets['aux_classification'])
            weighted_loss = self.task_weights['aux_classification'] * loss
            total_loss += weighted_loss
            loss_components['aux_classification'] = loss.item()
        
        # 3. 生长模式分类 (降低权重)
        if 'growth_pattern' in outputs and 'growth_pattern' in targets:
            loss = self.focal_loss(outputs['growth_pattern'], targets['growth_pattern'])
            weighted_loss = self.task_weights['growth_pattern'] * loss
            total_loss += weighted_loss
            loss_components['growth_pattern'] = loss.item()
        
        # 4. 干扰因素多标签 (重点优化，分层处理)
        if 'interference_factors' in outputs and 'interference_factors' in targets:
            loss = self.stratified_interference_loss(
                outputs['interference_factors'], 
                targets['interference_factors'],
                main_labels
            )
            weighted_loss = self.task_weights['interference_factors'] * loss
            total_loss += weighted_loss
            loss_components['interference_factors'] = loss.item()
        
        # 5. 质量评估
        if 'quality' in outputs and 'quality' in targets:
            quality_preds = outputs['quality'].squeeze()
            quality_targets = targets['quality'].float()
            loss = self.mse_loss(quality_preds, quality_targets)
            weighted_loss = self.task_weights['quality'] * loss
            total_loss += weighted_loss
            loss_components['quality'] = loss.item()
        
        return total_loss, loss_components

class AdvancedMultiTaskTrainer:
    """高级多任务训练器，专门解决阳性样本气孔检测问题"""
    
    def __init__(self, config: dict, data_root: str, experiment_name: str = None):
        self.config = config
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建实验目录
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"balanced_multitask_{timestamp}"
        
        self.experiment_dir = Path("experiments") / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"🚀 高级多任务训练器初始化")
        print(f"📁 实验目录: {self.experiment_dir}")
        print(f"🔧 设备: {self.device}")
        print(f"📊 数据源: {data_root}")
        
        # 初始化组件
        self.model = None
        self.dataloaders = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_metrics = {
            'main_acc': 0.0, 
            'pattern_acc': 0.0, 
            'interference_f1': 0.0,
            'pore_negative_f1': 0.0,  # 新增：阴性样本气孔F1
            'pore_positive_f1': 0.0   # 新增：阳性样本气孔F1
        }
        self.training_history = []
        
    def setup_data(self):
        """设置数据加载器"""
        print("📊 创建平衡的多任务数据加载器...")
        
        self.dataloaders = create_multitask_dataloaders(
            data_root=self.data_root,
            annotations_file="m9e1n170.json",
            batch_size=self.config['batch_size'],
            num_workers=4,
            split_ratio=(0.7, 0.15, 0.15),
            seed=42
        )
        
        print(f"✅ 数据加载器创建完成")
        for split, loader in self.dataloaders.items():
            print(f"   {split}: {len(loader)} 批次, {len(loader.dataset)} 样本")
    
    def setup_model(self):
        """设置模型"""
        print("🧠 创建优化的多任务MIC MobileNetV3...")
        
        # 获取数据集信息确定类别数
        dataset = next(iter(self.dataloaders.values())).dataset
        
        self.model = create_multitask_mic_mobilenetv3(
            num_classes=2,
            num_growth_patterns=len(dataset.label_mappings['growth_pattern']),
            num_interference_factors=len(dataset.label_mappings['interference_factors']),
            width_mult=1.0
        )
        self.model.to(self.device)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ 模型创建完成")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
    
    def setup_optimizer_and_scheduler(self):
        """设置优化器和调度器"""
        print("⚙️ 设置优化器和学习率调度器...")
        
        # 使用更小的学习率确保稳定训练
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用更平滑的学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        print(f"✅ 优化器设置完成")
        print(f"   学习率: {self.config['learning_rate']}")
        print(f"   权重衰减: {self.config['weight_decay']}")
    
    def setup_loss_function(self):
        """设置平衡的多任务损失函数"""
        print("📉 设置平衡的多任务损失函数...")
        
        # 调整任务权重，重点关注干扰因素
        task_weights = {
            'classification': 1.0,
            'aux_classification': 0.3,
            'growth_pattern': 0.6,       # 降低
            'interference_factors': 1.0, # 提升
            'quality': 0.2
        }
        
        self.criterion = BalancedMultiTaskLoss(
            task_weights=task_weights,
            focal_params={'alpha': 0.75, 'gamma': 2.0}
        )
        
        print(f"✅ 平衡多任务损失函数设置完成")
        print(f"   任务权重: {task_weights}")
        print(f"   干扰因素分层权重: pores(1.0), artifacts(5.0), debris(8.0), contamination(20.0)")
    
    def compute_detailed_metrics(self, outputs, targets):
        """计算详细的多任务指标，包括分层气孔检测"""
        metrics = {}
        
        # 主分类准确率
        if 'classification' in outputs and 'classification' in targets:
            preds = torch.argmax(outputs['classification'], dim=1)
            correct = (preds == targets['classification']).sum().item()
            total = targets['classification'].size(0)
            metrics['main_acc'] = correct / total
        
        # 生长模式准确率
        if 'growth_pattern' in outputs and 'growth_pattern' in targets:
            preds = torch.argmax(outputs['growth_pattern'], dim=1)
            correct = (preds == targets['growth_pattern']).sum().item()
            total = targets['growth_pattern'].size(0)
            metrics['pattern_acc'] = correct / total
        
        # 干扰因素F1分数 (多标签)
        if 'interference_factors' in outputs and 'interference_factors' in targets:
            preds = torch.sigmoid(outputs['interference_factors']) > 0.5
            targets_np = targets['interference_factors'].cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            # 整体F1
            f1_scores = []
            for i in range(targets_np.shape[0]):
                if targets_np[i].sum() > 0 or preds_np[i].sum() > 0:
                    f1 = f1_score(targets_np[i], preds_np[i], average='macro', zero_division=0)
                    f1_scores.append(f1)
            
            metrics['interference_f1'] = np.mean(f1_scores) if f1_scores else 0.0
            
            # 分层气孔检测分析
            main_labels = targets['classification'].cpu().numpy()
            negative_mask = (main_labels == 0)
            positive_mask = (main_labels == 1)
            
            # 气孔是第0个干扰因素
            pore_targets = targets_np[:, 0]
            pore_preds = preds_np[:, 0]
            
            # 阴性样本气孔F1
            if negative_mask.sum() > 0:
                neg_pore_targets = pore_targets[negative_mask]
                neg_pore_preds = pore_preds[negative_mask]
                metrics['pore_negative_f1'] = f1_score(neg_pore_targets, neg_pore_preds, zero_division=0)
            else:
                metrics['pore_negative_f1'] = 0.0
            
            # 阳性样本气孔F1
            if positive_mask.sum() > 0:
                pos_pore_targets = pore_targets[positive_mask]
                pos_pore_preds = pore_preds[positive_mask]
                metrics['pore_positive_f1'] = f1_score(pos_pore_targets, pos_pore_preds, zero_division=0)
            else:
                metrics['pore_positive_f1'] = 0.0
        
        return metrics
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        epoch_metrics = {
            'main_acc': 0.0, 'pattern_acc': 0.0, 'interference_f1': 0.0,
            'pore_negative_f1': 0.0, 'pore_positive_f1': 0.0
        }
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.dataloaders['train']):
            images = images.to(self.device)
            
            # 将目标数据移到GPU
            for key in targets:
                targets[key] = targets[key].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss, loss_components = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            batch_metrics = self.compute_detailed_metrics(outputs, targets)
            
            for key in epoch_metrics:
                if key in batch_metrics:
                    epoch_metrics[key] += batch_metrics[key]
            
            num_batches += 1
            
            # 打印进度
            if batch_idx % 100 == 0:
                progress = 100. * batch_idx / len(self.dataloaders['train'])
                print(f'训练 Epoch {epoch}: [{batch_idx}/{len(self.dataloaders["train"])} ({progress:.1f}%)]\\t'
                      f'损失: {loss.item():.6f}')
        
        # 更新学习率
        if self.scheduler:
            self.scheduler.step()
        
        # 计算平均指标
        avg_loss = total_loss / len(self.dataloaders['train'])
        for key in epoch_metrics:
            epoch_metrics[key] = epoch_metrics[key] / num_batches
        
        return avg_loss, epoch_metrics
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        epoch_metrics = {
            'main_acc': 0.0, 'pattern_acc': 0.0, 'interference_f1': 0.0,
            'pore_negative_f1': 0.0, 'pore_positive_f1': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.dataloaders['val']:
                images = images.to(self.device)
                
                # 将目标数据移到GPU
                for key in targets:
                    targets[key] = targets[key].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                loss, _ = self.criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item()
                batch_metrics = self.compute_detailed_metrics(outputs, targets)
                
                for key in epoch_metrics:
                    if key in batch_metrics:
                        epoch_metrics[key] += batch_metrics[key]
                
                num_batches += 1
        
        # 计算平均指标
        avg_loss = total_loss / len(self.dataloaders['val'])
        for key in epoch_metrics:
            epoch_metrics[key] = epoch_metrics[key] / num_batches
        
        return avg_loss, epoch_metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        # 保存最新检查点
        latest_path = self.experiment_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.experiment_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 保存最佳模型: {best_path}")
            print(f"   主分类: {metrics['main_acc']:.4f}")
            print(f"   生长模式: {metrics['pattern_acc']:.4f}")
            print(f"   干扰F1: {metrics['interference_f1']:.4f}")
            print(f"   阴性气孔F1: {metrics['pore_negative_f1']:.4f}")
            print(f"   阳性气孔F1: {metrics['pore_positive_f1']:.4f}")
    
    def train(self):
        """完整训练流程"""
        print(f"\\n🚀 开始平衡多任务训练 - {self.config['num_epochs']} epochs")
        print("=" * 80)
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\\n📊 Epoch {epoch}/{self.config['num_epochs']}")
            print("-" * 50)
            
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # 记录历史
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            # 添加所有指标
            for key in train_metrics:
                history_entry[f'train_{key}'] = train_metrics[key]
            for key in val_metrics:
                history_entry[f'val_{key}'] = val_metrics[key]
            
            self.training_history.append(history_entry)
            
            # 输出结果
            print(f"训练 - 损失: {train_loss:.6f}")
            print(f"      主分类: {train_metrics['main_acc']:.4f}, 生长模式: {train_metrics['pattern_acc']:.4f}")
            print(f"      干扰F1: {train_metrics['interference_f1']:.4f}")
            print(f"      阴性气孔F1: {train_metrics['pore_negative_f1']:.4f}, 阳性气孔F1: {train_metrics['pore_positive_f1']:.4f}")
            
            print(f"验证 - 损失: {val_loss:.6f}")
            print(f"      主分类: {val_metrics['main_acc']:.4f}, 生长模式: {val_metrics['pattern_acc']:.4f}")
            print(f"      干扰F1: {val_metrics['interference_f1']:.4f}")
            print(f"      阴性气孔F1: {val_metrics['pore_negative_f1']:.4f}, 阳性气孔F1: {val_metrics['pore_positive_f1']:.4f}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # 检查是否最佳 (重点关注阳性气孔F1改进)
            combined_score = (
                val_metrics['main_acc'] + 
                val_metrics['pattern_acc'] + 
                val_metrics['interference_f1'] + 
                val_metrics['pore_positive_f1'] * 2  # 给阳性气孔F1更高权重
            ) / 5
            
            best_combined = (
                self.best_metrics['main_acc'] + 
                self.best_metrics['pattern_acc'] + 
                self.best_metrics['interference_f1'] + 
                self.best_metrics['pore_positive_f1'] * 2
            ) / 5
            
            is_best = combined_score > best_combined
            if is_best:
                self.best_metrics.update(val_metrics)
                print(f"🎯 新的最佳综合性能: {combined_score:.4f}")
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 保存训练历史
            with open(self.experiment_dir / 'training_history.json', 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        print(f"\\n🎉 平衡多任务训练完成！")
        print(f"📈 最佳性能:")
        print(f"   主分类准确率: {self.best_metrics['main_acc']:.4f}")
        print(f"   生长模式准确率: {self.best_metrics['pattern_acc']:.4f}")
        print(f"   干扰因素F1: {self.best_metrics['interference_f1']:.4f}")
        print(f"   阴性气孔F1: {self.best_metrics['pore_negative_f1']:.4f}")
        print(f"   阳性气孔F1: {self.best_metrics['pore_positive_f1']:.4f}")
        print(f"💾 模型保存在: {self.experiment_dir}")
        
        return self.best_metrics

def main():
    parser = argparse.ArgumentParser(description='Balanced Multi-Task MIC Training')
    parser.add_argument('--data_root', type=str, default='/home/aaa/ws/bioastModel/ds/images',
                       help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='学习率 (更小的学习率确保稳定)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='权重衰减')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'description': 'Balanced multi-task training focusing on positive sample pore detection'
    }
    
    print(f"🔧 平衡训练配置:")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Weight Decay: {config['weight_decay']}")
    print(f"📊 数据源: {args.data_root}")
    
    # 创建训练器
    trainer = AdvancedMultiTaskTrainer(config, args.data_root, args.experiment_name)
    
    # 设置组件
    trainer.setup_data()
    trainer.setup_model()
    trainer.setup_optimizer_and_scheduler()
    trainer.setup_loss_function()
    
    # 开始训练
    best_metrics = trainer.train()
    
    print(f"\\n🎊 平衡多任务训练结束！")
    print(f"   主分类准确率: {best_metrics['main_acc']:.4f}")
    print(f"   生长模式准确率: {best_metrics['pattern_acc']:.4f}")
    print(f"   干扰因素F1: {best_metrics['interference_f1']:.4f}")
    print(f"   阴性气孔F1: {best_metrics['pore_negative_f1']:.4f}")
    print(f"   阳性气孔F1: {best_metrics['pore_positive_f1']:.4f}")

if __name__ == '__main__':
    main()