"""
Air Bubble Hybrid Network 模型训练脚本
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.airbubble_hybrid_net import create_airbubble_hybrid_net
from core.config.model_configs import get_model_config
from core.data_loader import create_data_loaders
from core.training_utils import EarlyStopping, ModelCheckpoint, calculate_metrics

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class AirBubbleHybridNetTrainer:
    """Air Bubble Hybrid Network 训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建实验目录
        self.experiment_dir = Path(f"experiments/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}/airbubble_hybrid_net")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"实验目录: {self.experiment_dir}")
        logging.info(f"使用设备: {self.device}")
    
    def create_model(self):
        """创建模型"""
        model = create_airbubble_hybrid_net(
            num_classes=self.config['num_classes'],
            model_size='base',
            dropout_rate=self.config['dropout_rate'],
            enable_distortion_correction=True
        )
        
        model = model.to(self.device)
        
        # 打印模型信息
        model_info = model.get_model_info()
        logging.info(f"模型参数数量: {model_info['total_parameters']:,}")
        logging.info(f"特性: {model_info['features']}")
        
        return model
    
    def create_data_loaders(self):
        """创建数据加载器"""
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            input_size=self.config['input_size'],
            num_workers=self.config['num_workers']
        )
        
        logging.info(f"训练集大小: {len(train_loader.dataset)}")
        logging.info(f"验证集大小: {len(val_loader.dataset)}")
        logging.info(f"测试集大小: {len(test_loader.dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def create_optimizer_and_scheduler(self, model):
        """创建优化器和学习率调度器"""
        # 为不同部分使用不同的学习率
        cnn_params = []
        transformer_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'transformer' in name:
                transformer_params.append(param)
            elif any(head in name for head in ['classification_head', 'turbidity_head', 'bubble_param_head', 'quality_head']):
                head_params.append(param)
            else:
                cnn_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': cnn_params, 'lr': self.config['learning_rate']},
            {'params': transformer_params, 'lr': self.config['learning_rate'] * 0.5},  # Transformer用较小学习率
            {'params': head_params, 'lr': self.config['learning_rate'] * 2.0}  # Head用较大学习率
        ], weight_decay=self.config['weight_decay'])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        return optimizer, scheduler
    
    def create_loss_functions(self):
        """创建损失函数"""
        # 主分类损失（4类：正常、异常、气泡干扰、其他）
        classification_loss = nn.CrossEntropyLoss()
        
        # 浊度回归损失
        turbidity_loss = nn.MSELoss()
        
        # 气泡参数回归损失
        bubble_param_loss = nn.MSELoss()
        
        # 质量评估损失
        quality_loss = nn.CrossEntropyLoss()
        
        # 气泡检测损失
        bubble_detection_loss = nn.BCELoss()
        
        return {
            'classification': classification_loss,
            'turbidity': turbidity_loss,
            'bubble_param': bubble_param_loss,
            'quality': quality_loss,
            'bubble_detection': bubble_detection_loss
        }
    
    def compute_multi_task_loss(self, outputs, targets, loss_functions):
        """计算多任务损失"""
        losses = {}
        
        # 分类损失
        if 'classification' in outputs:
            losses['classification'] = loss_functions['classification'](
                outputs['classification'], targets['labels']
            )
        
        # 浊度损失
        if 'turbidity' in outputs and 'turbidity' in targets:
            losses['turbidity'] = loss_functions['turbidity'](
                outputs['turbidity'].squeeze(), targets['turbidity']
            )
        
        # 气泡参数损失
        if 'bubble_params' in outputs and 'bubble_params' in targets:
            losses['bubble_param'] = loss_functions['bubble_param'](
                outputs['bubble_params'], targets['bubble_params']
            )
        
        # 质量评估损失
        if 'quality' in outputs and 'quality' in targets:
            losses['quality'] = loss_functions['quality'](
                outputs['quality'], targets['quality']
            )
        
        # 气泡检测损失
        if 'bubble_analysis' in outputs and 'bubble_mask' in targets:
            bubble_mask = outputs['bubble_analysis']['bubble_mask']
            # 全局平均池化到图像级别
            bubble_pred = torch.mean(bubble_mask.view(bubble_mask.size(0), -1), dim=1)
            losses['bubble_detection'] = loss_functions['bubble_detection'](
                bubble_pred, targets['bubble_mask']
            )
        
        # 总损失（加权组合）
        total_loss = losses['classification']
        if 'turbidity' in losses:
            total_loss += 0.3 * losses['turbidity']
        if 'bubble_param' in losses:
            total_loss += 0.2 * losses['bubble_param']
        if 'quality' in losses:
            total_loss += 0.15 * losses['quality']
        if 'bubble_detection' in losses:
            total_loss += 0.25 * losses['bubble_detection']
        
        losses['total'] = total_loss
        return losses
    
    def train_epoch(self, model, train_loader, optimizer, loss_functions, epoch):
        """训练一个epoch"""
        model.train()
        total_losses = {
            'total': 0, 'classification': 0, 'turbidity': 0, 
            'bubble_param': 0, 'quality': 0, 'bubble_detection': 0
        }
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device)
            
            # 处理目标数据
            if isinstance(targets, torch.Tensor):
                targets = {'labels': targets.to(self.device)}
            else:
                targets = {k: v.to(self.device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            
            # 计算损失
            losses = self.compute_multi_task_loss(outputs, targets, loss_functions)
            
            # 反向传播
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            for key, loss in losses.items():
                if key in total_losses:
                    total_losses[key] += loss.item()
            
            # 计算准确率
            if 'classification' in outputs:
                _, predicted = outputs['classification'].max(1)
                total += targets['labels'].size(0)
                correct += predicted.eq(targets['labels']).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%" if total > 0 else "0%"
            })
        
        # 计算平均损失
        avg_losses = {k: v / len(train_loader) for k, v in total_losses.items()}
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_losses, accuracy
    
    def validate(self, model, val_loader, loss_functions):
        """验证模型"""
        model.eval()
        total_losses = {
            'total': 0, 'classification': 0, 'turbidity': 0, 
            'bubble_param': 0, 'quality': 0, 'bubble_detection': 0
        }
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.device)
                
                # 处理目标数据
                if isinstance(targets, torch.Tensor):
                    targets = {'labels': targets.to(self.device)}
                else:
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # 前向传播
                outputs = model(data)
                
                # 计算损失
                losses = self.compute_multi_task_loss(outputs, targets, loss_functions)
                
                # 统计
                for key, loss in losses.items():
                    if key in total_losses:
                        total_losses[key] += loss.item()
                
                # 计算准确率
                if 'classification' in outputs:
                    _, predicted = outputs['classification'].max(1)
                    total += targets['labels'].size(0)
                    correct += predicted.eq(targets['labels']).sum().item()
        
        # 计算平均损失
        avg_losses = {k: v / len(val_loader) for k, v in total_losses.items()}
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_losses, accuracy
    
    def train(self):
        """主训练循环"""
        logging.info("开始训练 Air Bubble Hybrid Network...")
        
        # 创建模型
        model = self.create_model()
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model)
        
        # 创建损失函数
        loss_functions = self.create_loss_functions()
        
        # 创建早停和检查点保存器
        early_stopping = EarlyStopping(patience=12, min_delta=0.001)
        checkpoint = ModelCheckpoint(
            self.experiment_dir,
            monitor='val_accuracy',
            mode='max'
        )
        
        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            # 训练
            train_losses, train_acc = self.train_epoch(
                model, train_loader, optimizer, loss_functions, epoch
            )
            
            # 验证
            val_losses, val_acc = self.validate(model, val_loader, loss_functions)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_losses['total'])
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_losses['total'])
            history['val_acc'].append(val_acc)
            
            # 日志输出
            logging.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Train Loss: {train_losses['total']:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_losses['total']:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # 详细损失信息
            if epoch % 5 == 0:
                logging.info(f"  详细损失 - 分类: {train_losses['classification']:.4f}, "
                           f"浊度: {train_losses.get('turbidity', 0):.4f}, "
                           f"气泡: {train_losses.get('bubble_detection', 0):.4f}")
            
            # 保存检查点
            checkpoint.save(model, optimizer, epoch, val_acc, val_losses['total'])
            
            # 早停检查
            if early_stopping.should_stop(val_losses['total']):
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # 保存训练历史
        with open(self.experiment_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # 最终测试
        if test_loader:
            logging.info("开始最终测试...")
            test_losses, test_acc = self.validate(model, test_loader, loss_functions)
            logging.info(f"Test Loss: {test_losses['total']:.4f}, Test Acc: {test_acc:.2f}%")
        
        logging.info("训练完成!")
        return model, history

def main():
    """主函数"""
    # 训练配置
    config = {
        'model_name': 'airbubble_hybrid_net',
        'num_classes': 4,  # 包括气泡干扰类
        'input_size': 70,
        'batch_size': 24,  # 混合模型较大，使用较小batch size
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'dropout_rate': 0.2,
        'data_dir': 'data',
        'num_workers': 4
    }
    
    # 创建训练器
    trainer = AirBubbleHybridNetTrainer(config)
    
    # 开始训练
    model, history = trainer.train()
    
    return model, history

if __name__ == "__main__":
    main()