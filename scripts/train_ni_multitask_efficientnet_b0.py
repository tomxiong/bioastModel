#!/usr/bin/env python3
"""
EfficientNet-B0多任务模型训练脚本
使用相同的NI数据集训练改进版EfficientNet-B0架构
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 导入必要模块
from models.multitask_efficientnet_b0 import create_multitask_efficientnet_b0_standard
from training.ni_multitask_dataset import NIMultitaskDataset


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultitaskLoss(nn.Module):
    """Multi-task learning loss"""
    def __init__(self, task_weights=None, use_focal_loss=False, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.task_weights = task_weights or {}
        self.use_focal_loss = use_focal_loss
        
        if use_focal_loss:
            self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        losses = {}
        
        # Growth level loss
        weight = self.task_weights.get('growth_level', 1.0)
        losses['growth_level'] = weight * self.ce_loss(predictions['growth_level'], targets['growth_level'])
        
        # Growth pattern loss  
        weight = self.task_weights.get('growth_pattern', 1.0)
        losses['growth_pattern'] = weight * self.ce_loss(predictions['growth_pattern'], targets['growth_pattern'])
        
        # Interference factors loss (multi-label)
        weight = self.task_weights.get('interference_factors', 1.0)
        losses['interference_factors'] = weight * self.bce_loss(
            predictions['interference_factors'], 
            targets['interference_factors'].float()
        )
        
        # Fine-grained loss
        weight = self.task_weights.get('fine_grained', 1.0)
        losses['fine_grained'] = weight * self.ce_loss(predictions['fine_grained'], targets['fine_grained'])
        
        return losses


class TaskWeightLearner(nn.Module):
    """自适应任务权重学习器"""
    def __init__(self, num_tasks=4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses):
        weights = torch.exp(-self.log_vars)
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += weights[i] * loss + self.log_vars[i]
        return total_loss, weights


def create_data_loaders(dataset_path, batch_size=16, num_workers=4):
    """创建数据加载器"""
    print("创建数据加载器...")
    
    # 训练集
    train_dataset = NIMultitaskDataset(
        data_root=dataset_path,
        split='train'
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    
    # 验证集
    val_dataset = NIMultitaskDataset(
        data_root=dataset_path,
        split='val'
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"批次大小: {batch_size}")
    
    return train_loader, val_loader


def validate_epoch(model, val_loader, criterion, device, task_weight_learner=None):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    task_losses = defaultdict(float)
    task_correct = defaultdict(int)
    task_totals = defaultdict(int)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            # 只处理张量类型的任务标签
            task_targets = {}
            for task, target in targets.items():
                if isinstance(target, torch.Tensor):
                    task_targets[task] = target.to(device)
            targets = task_targets
            
            # 前向传播
            outputs = model(images)
            losses = criterion(outputs, targets)
            
            # 计算总损失
            if task_weight_learner is not None:
                batch_loss, _ = task_weight_learner([
                    losses['growth_level'], losses['growth_pattern'],
                    losses['interference_factors'], losses['fine_grained']
                ])
            else:
                batch_loss = sum(losses.values())
            
            total_loss += batch_loss.item()
            
            # 记录各任务损失和准确率
            for task, loss in losses.items():
                task_losses[task] += loss.item()
                
                # 计算准确率
                if task == 'interference_factors':
                    # 多标签分类用阈值
                    pred = (torch.sigmoid(outputs[task]) > 0.5).float()
                    correct = (pred == targets[task]).all(dim=1).sum().item()
                else:
                    # 单标签分类
                    pred = outputs[task].argmax(dim=1)
                    correct = (pred == targets[task]).sum().item()
                
                task_correct[task] += correct
                task_totals[task] += len(images)
    
    # 计算平均指标
    avg_loss = total_loss / len(val_loader)
    avg_task_losses = {task: loss / len(val_loader) for task, loss in task_losses.items()}
    task_accuracies = {task: task_correct[task] / task_totals[task] 
                      for task in task_correct.keys()}
    
    return avg_loss, avg_task_losses, task_accuracies


def train_epoch(model, train_loader, criterion, optimizer, device, 
               task_weight_learner=None, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    task_losses = defaultdict(float)
    task_correct = defaultdict(int)
    task_totals = defaultdict(int)
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        # 只处理张量类型的任务标签
        task_targets = {}
        for task, target in targets.items():
            if isinstance(target, torch.Tensor):
                task_targets[task] = target.to(device)
        targets = task_targets
        
        optimizer.zero_grad()
        
        if scaler:
            # 混合精度训练
            with torch.cuda.amp.autocast():
                outputs = model(images)
                losses = criterion(outputs, targets)
                
                if task_weight_learner is not None:
                    batch_loss, task_weights = task_weight_learner([
                        losses['growth_level'], losses['growth_pattern'],
                        losses['interference_factors'], losses['fine_grained']
                    ])
                else:
                    batch_loss = sum(losses.values())
            
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准训练
            outputs = model(images)
            losses = criterion(outputs, targets)
            
            if task_weight_learner is not None:
                batch_loss, task_weights = task_weight_learner([
                    losses['growth_level'], losses['growth_pattern'],
                    losses['interference_factors'], losses['fine_grained']
                ])
            else:
                batch_loss = sum(losses.values())
            
            batch_loss.backward()
            optimizer.step()
        
        total_loss += batch_loss.item()
        
        # 记录各任务指标
        for task, loss in losses.items():
            task_losses[task] += loss.item()
            
            # 计算准确率
            if task == 'interference_factors':
                pred = (torch.sigmoid(outputs[task]) > 0.5).float()
                correct = (pred == targets[task]).all(dim=1).sum().item()
            else:
                pred = outputs[task].argmax(dim=1)
                correct = (pred == targets[task]).sum().item()
            
            task_correct[task] += correct
            task_totals[task] += len(images)
        
        # 打印进度
        if batch_idx % 20 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {batch_loss.item():.4f}')
    
    # 计算平均指标
    avg_loss = total_loss / len(train_loader)
    avg_task_losses = {task: loss / len(train_loader) for task, loss in task_losses.items()}
    task_accuracies = {task: task_correct[task] / task_totals[task] 
                      for task in task_correct.keys()}
    
    return avg_loss, avg_task_losses, task_accuracies


def save_training_curves(history, save_dir):
    """保存训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EfficientNet-B0 MultiTask Training Progress', fontsize=16)
    
    # 总损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 各任务损失
    task_colors = ['blue', 'red', 'green', 'orange']
    tasks = ['growth_level', 'growth_pattern', 'interference_factors', 'fine_grained']
    
    for i, (task, color) in enumerate(zip(tasks, task_colors)):
        if i < 2:
            ax = axes[0, i+1]
        else:
            ax = axes[1, i-2]
        
        ax.plot(epochs, [h[task] for h in history['train_task_losses']], 
                f'{color[0]}-', label=f'Train {task}')
        ax.plot(epochs, [h[task] for h in history['val_task_losses']], 
                f'{color[0]}--', label=f'Val {task}')
        ax.set_title(f'{task.replace("_", " ").title()} Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
    
    # 总准确率
    axes[1, 2].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1, 2].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1, 2].set_title('Overall Accuracy')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主训练函数"""
    print("=== EfficientNet-B0 多任务训练开始 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练参数
    config = {
        'batch_size': 16,
        'num_epochs': 100,
        'initial_lr': 1e-4,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'use_amp': True,  # 混合精度
        'use_adaptive_weights': True,  # 自适应任务权重
        'patience': 15,  # 早停耐心
        'warmup_epochs': 5,  # 预热轮数
    }
    
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"ni_multitask_efficientnet_b0_{timestamp}"
    experiment_dir = os.path.join(project_root, 'experiments', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"实验目录: {experiment_dir}")
    
    # 保存配置
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 数据集路径
    dataset_path = os.path.join(project_root, 'dataset_ni_multitask')
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        dataset_path, config['batch_size'], config['num_workers']
    )
    
    # 创建模型
    print("创建EfficientNet-B0多任务模型...")
    model = create_multitask_efficientnet_b0_standard(
        pretrained=False  # 只传递需要修改的参数
    )
    model = model.to(device)
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"模型参数: {model_info['total_parameters']:,}")
    print(f"主干参数: {model_info['backbone_parameters']:,}")
    print(f"任务头参数: {model_info['task_head_parameters']:,}")
    
    # 创建损失函数
    criterion = MultitaskLoss(
        task_weights={
            'growth_level': 1.0,
            'growth_pattern': 1.0, 
            'interference_factors': 1.0,
            'fine_grained': 2.0  # 精细分类权重更高
        },
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
    
    # 自适应任务权重学习器
    task_weight_learner = None
    if config['use_adaptive_weights']:
        task_weight_learner = TaskWeightLearner(num_tasks=4).to(device)
        print("启用自适应任务权重学习")
    
    # 创建优化器
    params_to_optimize = list(model.parameters())
    if task_weight_learner:
        params_to_optimize.extend(list(task_weight_learner.parameters()))
    
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=config['initial_lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8,
        min_lr=1e-7
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None
    
    # 训练历史记录
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_task_losses': [], 'val_task_losses': [],
        'train_task_accs': [], 'val_task_accs': [],
        'lr_history': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n开始训练 {config['num_epochs']} 轮...")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # 训练阶段
        train_loss, train_task_losses, train_task_accs = train_epoch(
            model, train_loader, criterion, optimizer, device,
            task_weight_learner, scaler
        )
        
        # 验证阶段
        val_loss, val_task_losses, val_task_accs = validate_epoch(
            model, val_loader, criterion, device, task_weight_learner
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算总体准确率
        train_acc = np.mean(list(train_task_accs.values()))
        val_acc = np.mean(list(val_task_accs.values()))
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_task_losses'].append(train_task_losses)
        history['val_task_losses'].append(val_task_losses)
        history['train_task_accs'].append(train_task_accs)
        history['val_task_accs'].append(val_task_accs)
        history['lr_history'].append(current_lr)
        
        # 打印结果
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        print(f"训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"学习率: {current_lr:.6f}")
        
        print("各任务验证准确率:")
        for task, acc in val_task_accs.items():
            print(f"  {task}: {acc:.4f}")
        
        # 打印任务权重
        if task_weight_learner is not None:
            with torch.no_grad():
                weights = torch.exp(-task_weight_learner.log_vars)
                print("当前任务权重:")
                tasks = ['growth_level', 'growth_pattern', 'interference_factors', 'fine_grained']
                for i, task in enumerate(tasks):
                    print(f"  {task}: {weights[i].item():.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'task_weight_learner_state_dict': task_weight_learner.state_dict() if task_weight_learner else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_task_accs': val_task_accs,
                'model_info': model_info
            }, os.path.join(experiment_dir, 'best_model.pth'))
            
            print(f"✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= config['patience']:
            print(f"早停触发 (patience: {config['patience']})")
            break
        
        # 保存训练历史
        with open(os.path.join(experiment_dir, 'training_history.json'), 'w') as f:
            # 转换numpy类型以便JSON序列化
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value[0] if value else None, dict):
                    serializable_history[key] = value
                else:
                    serializable_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            json.dump(serializable_history, f, indent=2)
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'task_weight_learner_state_dict': task_weight_learner.state_dict() if task_weight_learner else None,
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': val_loss,
        'val_task_accs': val_task_accs,
        'model_info': model_info
    }, os.path.join(experiment_dir, 'final_model.pth'))
    
    # 绘制训练曲线
    save_training_curves(history, experiment_dir)
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最终验证准确率: {val_acc:.4f}")
    print(f"实验结果保存在: {experiment_dir}")


if __name__ == "__main__":
    main()