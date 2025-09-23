"""
基于新增强多任务数据集训练MIC MobileNetV3
使用ds/images/m9e1n170.json的19,994个增强标注数据
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.enhanced_mic_mobilenetv3 import create_enhanced_mic_mobilenetv3
from training.enhanced_multitask_dataset import create_multitask_dataloaders
from core.config.training_configs import get_model_specific_config

class EnhancedMICTrainer:
    """基于新数据集的增强MIC训练器"""
    
    def __init__(self, config: dict, data_root: str, experiment_name: str = None):
        self.config = config
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建实验目录
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"enhanced_mic_multitask_{timestamp}"
        
        self.experiment_dir = Path("experiments") / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"🚀 初始化增强MIC训练器")
        print(f"📁 实验目录: {self.experiment_dir}")
        print(f"🔧 设备: {self.device}")
        print(f"📊 数据源: {data_root}")
        
        # 初始化组件
        self.model = None
        self.dataloaders = None
        self.optimizer = None
        self.scheduler = None
        self.loss_functions = None
        self.best_val_acc = 0.0
        self.training_history = []
        
    def setup_data(self):
        """设置数据加载器"""
        print("📊 创建数据加载器...")
        
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
        print("🧠 创建增强MIC MobileNetV3模型...")
        
        self.model = create_enhanced_mic_mobilenetv3(num_classes=2)
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
        
        # 优化器
        if self.config['optimizer'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config['optimizer']}")
        
        # 学习率调度器
        if self.config['scheduler'] == 'cosine_with_restarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, self.config['num_epochs'] // 4),
                T_mult=2,
                eta_min=self.config['learning_rate'] * 0.01
            )
        else:
            raise ValueError(f"不支持的调度器: {self.config['scheduler']}")
        
        print(f"✅ 优化器和调度器设置完成")
        print(f"   优化器: {self.config['optimizer']}")
        print(f"   学习率: {self.config['learning_rate']}")
        print(f"   权重衰减: {self.config['weight_decay']}")
        
    def setup_loss_functions(self):
        """设置损失函数"""
        print("📉 设置多任务损失函数...")
        
        self.loss_functions = {}
        
        # 主分类损失 - Focal Loss
        if self.config.get('focal_loss', True):
            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.75, gamma=2.0):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    
                def forward(self, inputs, targets):
                    ce_loss = nn.CrossEntropyLoss()(inputs, targets)
                    pt = torch.exp(-ce_loss)
                    focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
                    return focal_loss
                    
            self.loss_functions['classification'] = FocalLoss(
                alpha=self.config.get('focal_alpha', 0.75),
                gamma=self.config.get('focal_gamma', 2.0)
            )
        else:
            self.loss_functions['classification'] = nn.CrossEntropyLoss()
        
        # 辅助分类损失
        self.loss_functions['aux_classification'] = nn.CrossEntropyLoss()
        
        # 气泡检测损失
        self.loss_functions['bubble_detection'] = nn.BCEWithLogitsLoss()
        
        # 浊度分析损失  
        self.loss_functions['turbidity'] = nn.BCEWithLogitsLoss()
        
        # 质量评估损失
        self.loss_functions['quality'] = nn.MSELoss()
        
        print(f"✅ 损失函数设置完成")
        print(f"   主分类: {'Focal Loss' if self.config.get('focal_loss', True) else 'CrossEntropy'}")
        print(f"   多任务权重: {self.config.get('multitask_weights', {})}")
    
    def compute_multitask_loss(self, outputs, targets):
        """计算多任务损失"""
        total_loss = 0.0
        loss_components = {}
        
        # 获取多任务权重
        weights = self.config.get('multitask_weights', {
            'classification': 1.0,
            'aux_classification': 0.5,
            'bubble_detection': 0.3,
            'turbidity': 0.2,
            'quality': 0.1
        })
        
        # 主分类损失
        if 'classification' in outputs and 'classification' in targets:
            loss = self.loss_functions['classification'](outputs['classification'], targets['classification'])
            weighted_loss = weights['classification'] * loss
            total_loss += weighted_loss
            loss_components['classification'] = loss.item()
        
        # 辅助分类损失
        if 'aux_classification' in outputs and 'aux_classification' in targets:
            loss = self.loss_functions['aux_classification'](outputs['aux_classification'], targets['aux_classification'])
            weighted_loss = weights['aux_classification'] * loss
            total_loss += weighted_loss
            loss_components['aux_classification'] = loss.item()
        
        # 气泡检测损失
        if 'bubble_detection' in outputs and 'bubble_detection' in targets:
            bubble_logits = outputs['bubble_detection'].squeeze()
            bubble_targets = targets['bubble_detection'].float()
            loss = self.loss_functions['bubble_detection'](bubble_logits, bubble_targets)
            weighted_loss = weights['bubble_detection'] * loss
            total_loss += weighted_loss
            loss_components['bubble_detection'] = loss.item()
        
        # 浊度分析损失
        if 'turbidity' in outputs and 'turbidity' in targets:
            turbidity_logits = outputs['turbidity'].squeeze()
            turbidity_targets = targets['turbidity'].float()
            loss = self.loss_functions['turbidity'](turbidity_logits, turbidity_targets)
            weighted_loss = weights['turbidity'] * loss
            total_loss += weighted_loss
            loss_components['turbidity'] = loss.item()
        
        # 质量评估损失
        if 'quality' in outputs and 'quality' in targets:
            quality_preds = outputs['quality'].squeeze()
            quality_targets = targets['quality'].float()
            
            # 确保维度匹配
            if quality_preds.dim() == 2 and quality_preds.size(1) > 1:
                # 如果模型输出多个质量分数，取平均值
                quality_preds = quality_preds.mean(dim=1)
            
            loss = self.loss_functions['quality'](quality_preds, quality_targets)
            weighted_loss = weights['quality'] * loss
            total_loss += weighted_loss
            loss_components['quality'] = loss.item()
        
        return total_loss, loss_components
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, targets) in enumerate(self.dataloaders['train']):
            images = images.to(self.device)
            
            # 将目标数据移到GPU
            for key in targets:
                targets[key] = targets[key].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss, loss_components = self.compute_multitask_loss(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('gradient_clip_norm', 0) > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_norm'])
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(outputs['classification'], dim=1)
            correct_predictions += (predictions == targets['classification']).sum().item()
            total_samples += targets['classification'].size(0)
            
            # 打印进度
            if batch_idx % 100 == 0:
                progress = 100. * batch_idx / len(self.dataloaders['train'])
                print(f'训练 Epoch {epoch}: [{batch_idx}/{len(self.dataloaders["train"])} ({progress:.1f}%)]\\t'
                      f'损失: {loss.item():.6f}')
        
        # 更新学习率
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / len(self.dataloaders['train'])
        accuracy = 100. * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, targets in self.dataloaders['val']:
                images = images.to(self.device)
                
                # 将目标数据移到GPU
                for key in targets:
                    targets[key] = targets[key].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                loss, _ = self.compute_multitask_loss(outputs, targets)
                
                # 统计
                total_loss += loss.item()
                predictions = torch.argmax(outputs['classification'], dim=1)
                correct_predictions += (predictions == targets['classification']).sum().item()
                total_samples += targets['classification'].size(0)
        
        avg_loss = total_loss / len(self.dataloaders['val'])
        accuracy = 100. * correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
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
    
    def train(self):
        """完整训练流程"""
        print(f"\\n🚀 开始训练 - {self.config['num_epochs']} epochs")
        print("=" * 60)
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\\n📊 Epoch {epoch}/{self.config['num_epochs']}")
            print("-" * 40)
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # 记录历史
            history_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(history_entry)
            
            # 输出结果
            print(f"训练 - 损失: {train_loss:.6f}, 准确率: {train_acc:.2f}%")
            print(f"验证 - 损失: {val_loss:.6f}, 准确率: {val_acc:.2f}%")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # 保存检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                print(f"🎯 新的最佳验证准确率: {val_acc:.2f}%")
            
            self.save_checkpoint(epoch, is_best)
            
            # 保存训练历史
            with open(self.experiment_dir / 'training_history.json', 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        print(f"\\n🎉 训练完成！")
        print(f"📈 最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"💾 模型保存在: {self.experiment_dir}")
        
        return self.best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Enhanced MIC MobileNetV3 with New Dataset')
    parser.add_argument('--config', type=str, default='enhanced_mic_mobilenetv3_optimized',
                       help='训练配置名称')
    parser.add_argument('--data_root', type=str, default='/home/aaa/ws/bioastModel/ds/images',
                       help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (覆盖配置)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小 (覆盖配置)')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率 (覆盖配置)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_model_specific_config(args.config)
    
    # 覆盖配置参数
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['learning_rate'] = args.lr
    
    print(f"🔧 使用配置: {args.config}")
    print(f"📊 数据源: {args.data_root}")
    print(f"⚙️ 训练参数: epochs={config['num_epochs']}, batch_size={config['batch_size']}, lr={config['learning_rate']}")
    
    # 创建训练器
    trainer = EnhancedMICTrainer(config, args.data_root, args.experiment_name)
    
    # 设置组件
    trainer.setup_data()
    trainer.setup_model()
    trainer.setup_optimizer_and_scheduler()
    trainer.setup_loss_functions()
    
    # 开始训练
    best_acc = trainer.train()
    
    print(f"\\n🎊 训练结束！最佳准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()