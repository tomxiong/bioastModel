#!/usr/bin/env python3
"""
单独训练指定模型的脚本
支持训练新增的模型，而不需要训练全部模型
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from training.dataset import BioastDataset
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from training.visualizer import TrainingVisualizer

# 导入所有模型
from models.mic_mobilenetv3 import create_mic_mobilenetv3, MODEL_CONFIG as MIC_MOBILENET_CONFIG
from models.micro_vit import create_micro_vit, MODEL_CONFIG as MICRO_VIT_CONFIG
from models.airbubble_hybrid_net import create_airbubble_hybrid_net, MODEL_CONFIG as AIRBUBBLE_CONFIG
from models.coatnet import create_coatnet, MODEL_CONFIG as COATNET_CONFIG
from models.convnext_tiny import create_convnext_tiny, MODEL_CONFIG as CONVNEXT_CONFIG
from models.efficientnet import create_efficientnet_b0, create_efficientnet_b1
from models.resnet_improved import create_resnet18_improved, create_resnet34_improved, create_resnet50_improved
from models.vit_tiny import create_vit_tiny
from models.enhanced_airbubble_detector import create_enhanced_airbubble_detector

# 新增模型导入
from models.efficientnet_v2 import create_efficientnetv2_s, create_efficientnetv2_m
from models.mobilenet_v3 import create_mobilenetv3_large, create_mobilenetv3_small
from models.regnet import create_regnet_x_400mf, create_regnet_y_400mf
from models.densenet import create_densenet121, create_densenet169
from models.shufflenet_v2 import create_shufflenetv2_x0_5, create_shufflenetv2_x1_0
from models.ghostnet import create_ghostnet
from models.mnasnet import create_mnasnet_1_0

class SingleModelTrainer:
    """单个模型训练器"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2
        
        # 设置默认配置
        self.batch_size = config.get('batch_size', 64)
        self.epochs = config.get('epochs', 10)
        self.lr = config.get('lr', 0.001)
        
        print(f"🚀 初始化 {model_name} 训练器")
        print(f"📱 设备: {self.device}")
        print(f"📊 批次大小: {self.batch_size}")
        print(f"🔄 训练轮数: {self.epochs}")
        print(f"📈 学习率: {self.lr}")
        
        # 创建保存目录
        self.save_dir = Path(f'checkpoints/{model_name}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def create_model(self) -> nn.Module:
        """创建模型"""
        if self.model_name == 'vit_tiny':
            model = create_vit_tiny(num_classes=2, dropout_rate=0.1)
        elif self.model_name == 'micro_vit':
            model = create_micro_vit(num_classes=2)
        elif self.model_name == 'mic_mobilenetv3':
            model = create_mic_mobilenetv3(num_classes=2)
        elif self.model_name == 'airbubble_hybrid_net':
            model = create_airbubble_hybrid_net(num_classes=2)
        elif self.model_name == 'enhanced_airbubble_detector':
            model = create_enhanced_airbubble_detector(num_classes=2)
        elif self.model_name == 'efficientnet_b0':
            model = create_efficientnet_b0(num_classes=2)
        elif self.model_name == 'efficientnet_b1':
            model = create_efficientnet_b1(num_classes=2)
        elif self.model_name == 'resnet18_improved':
            model = create_resnet18_improved(num_classes=2)
        elif self.model_name == 'resnet34_improved':
            model = create_resnet34_improved(num_classes=2)
        elif self.model_name == 'resnet50_improved':
            model = create_resnet50_improved(num_classes=2)
        elif self.model_name == 'coatnet':
            model = create_coatnet(num_classes=2)
        elif self.model_name == 'convnext_tiny':
            model = create_convnext_tiny(num_classes=2)
        # 新增模型 - EfficientNet V2系列
        elif self.model_name == 'efficientnetv2_s':
            model = create_efficientnetv2_s(num_classes=2)
        elif self.model_name == 'efficientnetv2_m':
            model = create_efficientnetv2_m(num_classes=2)
        # 新增模型 - MobileNet V3系列
        elif self.model_name == 'mobilenetv3_large':
            model = create_mobilenetv3_large(num_classes=2)
        elif self.model_name == 'mobilenetv3_small':
            model = create_mobilenetv3_small(num_classes=2)
        # 新增模型 - RegNet系列
        elif self.model_name == 'regnet_x_400mf':
            model = create_regnet_x_400mf(num_classes=2)
        elif self.model_name == 'regnet_y_400mf':
            model = create_regnet_y_400mf(num_classes=2)
        # 新增模型 - DenseNet系列
        elif self.model_name == 'densenet121':
            model = create_densenet121(num_classes=2)
        elif self.model_name == 'densenet169':
            model = create_densenet169(num_classes=2)
        # 新增模型 - 轻量级模型
        elif self.model_name == 'shufflenetv2_x0_5':
            model = create_shufflenetv2_x0_5(num_classes=2)
        elif self.model_name == 'shufflenetv2_x1_0':
            model = create_shufflenetv2_x1_0(num_classes=2)
        elif self.model_name == 'ghostnet':
            model = create_ghostnet(num_classes=2)
        elif self.model_name == 'mnasnet_1_0':
            model = create_mnasnet_1_0(num_classes=2)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        return model.to(self.device)

    def create_data_loaders(self):
        """创建数据加载器"""
        # 数据集路径
        dataset_dir = Path('bioast_dataset')
        
        if not dataset_dir.exists():
            raise FileNotFoundError("数据集目录不存在，请确保 bioast_dataset 目录存在")
        
        # 数据变换
        from torchvision import transforms
        
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        train_dataset = BioastDataset(
            data_dir='bioast_dataset',
            split='train',
            transform=train_transform
        )
        
        val_dataset = BioastDataset(
            data_dir='bioast_dataset',
            split='val',
            transform=val_transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"📊 训练样本数: {len(train_dataset)}")
        print(f"📊 验证样本数: {len(val_dataset)}")
        
        return train_loader, val_loader

    def create_optimizer_and_scheduler(self, model):
        """创建优化器和调度器"""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        
        return optimizer, scheduler

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch: int) -> tuple:
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            # Handle models that return dict (like mic_mobilenetv3)
            if isinstance(output, dict):
                output = output['classification']
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 20 == 0:
                accuracy = 100. * correct / total
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {accuracy:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def validate_epoch(self, model, val_loader, criterion) -> tuple:
        """验证一个epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                # Handle models that return dict (like mic_mobilenetv3)
                if isinstance(output, dict):
                    output = output['classification']
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def save_checkpoint(self, model, optimizer, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"💾 保存最佳模型，验证准确率: {self.best_val_acc:.4f}")

    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    def train(self):
        """主训练循环"""
        print(f"\n🚀 开始训练 {self.model_name}")
        start_time = time.time()
        
        # 创建模型
        model = self.create_model()
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders()
        
        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_and_scheduler(model)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        for epoch in range(1, self.epochs + 1):
            print(f"\n📅 Epoch {epoch}/{self.epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # 更新调度器
            scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 检查是否是最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # 保存检查点
            self.save_checkpoint(model, optimizer, epoch, is_best)
            
            print(f"📊 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"📊 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"🏆 最佳验证准确率: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        
        # 保存训练历史
        self.save_training_history()
        
        training_time = time.time() - start_time
        print(f"\n✅ {self.model_name} 训练完成!")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"⏱️ 训练时间: {training_time:.1f}秒")
        
        return {
            'model': self.model_name,
            'params': self.config.get('estimated_params', 0),
            'best_acc': self.best_val_acc / 100.0,  # 转换为小数
            'training_time': training_time
        }

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """获取所有可用的模型配置"""
    return {
        # 原有模型
        'vit_tiny': {'estimated_params': 0.5, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'micro_vit': MICRO_VIT_CONFIG,
        'mic_mobilenetv3': MIC_MOBILENET_CONFIG,
        'airbubble_hybrid_net': AIRBUBBLE_CONFIG,
        'enhanced_airbubble_detector': {'estimated_params': 4.0, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'efficientnet_b0': {'estimated_params': 5.3, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'efficientnet_b1': {'estimated_params': 7.8, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'resnet18_improved': {'estimated_params': 11.2, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'resnet34_improved': {'estimated_params': 21.3, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'resnet50_improved': {'estimated_params': 23.5, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'coatnet': COATNET_CONFIG,
        'convnext_tiny': CONVNEXT_CONFIG,
        
        # 新增模型 - EfficientNet V2系列
        'efficientnetv2_s': {'estimated_params': 21.5, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'efficientnetv2_m': {'estimated_params': 54.1, 'lr': 0.001, 'batch_size': 32, 'epochs': 10},
        
        # 新增模型 - MobileNet V3系列
        'mobilenetv3_large': {'estimated_params': 5.4, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'mobilenetv3_small': {'estimated_params': 2.9, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        
        # 新增模型 - RegNet系列
        'regnet_x_400mf': {'estimated_params': 5.2, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'regnet_y_400mf': {'estimated_params': 4.3, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        
        # 新增模型 - DenseNet系列
        'densenet121': {'estimated_params': 8.0, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'densenet169': {'estimated_params': 14.1, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        
        # 新增模型 - 轻量级模型
        'shufflenetv2_x0_5': {'estimated_params': 1.4, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'shufflenetv2_x1_0': {'estimated_params': 2.3, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'ghostnet': {'estimated_params': 5.2, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
        'mnasnet_1_0': {'estimated_params': 4.4, 'lr': 0.001, 'batch_size': 64, 'epochs': 10},
    }

def main():
    parser = argparse.ArgumentParser(description='训练单个模型')
    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--list_models', action='store_true', help='列出所有可用模型')
    
    args = parser.parse_args()
    
    available_models = get_available_models()
    
    if args.list_models:
        print("\n📋 可用模型列表:")
        print("=" * 60)
        for model_name, config in available_models.items():
            params = config.get('estimated_params', 'Unknown')
            print(f"{model_name:<25} | {params:>6}M 参数")
        return
    
    if not args.model:
        parser.error("--model is required when not using --list_models")
    
    if args.model not in available_models:
        print(f"❌ 错误: 模型 '{args.model}' 不存在")
        print("\n📋 可用模型:")
        for model_name in available_models.keys():
            print(f"  - {model_name}")
        return
    
    # 获取模型配置
    config = available_models[args.model].copy()
    
    # 覆盖命令行参数
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    
    print(f"\n🎯 准备训练模型: {args.model}")
    print(f"📊 参数量: {config.get('estimated_params', 'Unknown')}M")
    print(f"🔄 训练轮数: {config.get('epochs', 10)}")
    print(f"📦 批次大小: {config.get('batch_size', 64)}")
    print(f"📈 学习率: {config.get('lr', 0.001)}")
    
    # 创建训练器并开始训练
    trainer = SingleModelTrainer(args.model, config)
    result = trainer.train()
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'single_model_result_{args.model}_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 结果已保存到: {result_file}")

if __name__ == "__main__":
    main()