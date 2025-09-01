#!/usr/bin/env python3
"""
全模型GPU训练脚本 - 支持所有9种模型的训练
按照模型参数量从小到大依次训练，用于GPU性能测试
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

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training.dataset import BioastDataset
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from training.visualizer import TrainingVisualizer

# Import all models
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

class UniversalModelTrainer:
    """通用模型训练器，支持所有9种模型"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练配置
        self.batch_size = config.get('batch_size', 16)
        self.epochs = config.get('epochs', 2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        # 创建实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = Path(f'experiments/experiment_{timestamp}/{model_name}')
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def create_model(self) -> nn.Module:
        """创建指定的模型"""
        print(f"🏗️ Creating {self.model_name} model...")
        
        if self.model_name == 'mic_mobilenetv3':
            model = create_mic_mobilenetv3(**MIC_MOBILENET_CONFIG['default_params'])
        elif self.model_name == 'micro_vit':
            model = create_micro_vit(**MICRO_VIT_CONFIG['default_params'])
        elif self.model_name == 'airbubble_hybrid_net':
            model = create_airbubble_hybrid_net(**AIRBUBBLE_CONFIG['default_params'])
        elif self.model_name == 'coatnet':
            model = create_coatnet(**COATNET_CONFIG['default_params'])
        elif self.model_name == 'convnext_tiny':
            model = create_convnext_tiny(**CONVNEXT_CONFIG['default_params'])
        elif self.model_name == 'efficientnet_b0':
            model = create_efficientnet_b0(num_classes=2)
        elif self.model_name == 'efficientnet_b1':
            model = create_efficientnet_b1(num_classes=2)
        elif self.model_name == 'enhanced_airbubble_detector':
            model = create_enhanced_airbubble_detector({'num_classes': 2})
        elif self.model_name == 'resnet18_improved':
            model = create_resnet18_improved(num_classes=2)
        elif self.model_name == 'resnet34_improved':
            model = create_resnet34_improved(num_classes=2)
        elif self.model_name == 'resnet50_improved':
            model = create_resnet50_improved(num_classes=2)
        elif self.model_name == 'vit_tiny':
            model = create_vit_tiny(num_classes=2, dropout_rate=0.1)
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
        
        # 获取模型信息
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            print(f"📊 Model parameters: {model_info['total_parameters']:,}")
        else:
            # 对于没有get_model_info方法的模型，手动计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 Model parameters: {total_params:,}")
        
        return model.to(self.device)
    
    def create_data_loaders(self):
        """创建数据加载器"""
        print("📂 Loading datasets...")
        
        # 数据变换
        from torchvision import transforms
        
        train_transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((70, 70)),
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
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"训练样本: {len(train_dataset)} 验证样本: {len(val_dataset)}")
    
    def create_optimizer_and_scheduler(self):
        """创建优化器和学习率调度器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6
        )
        
        print(f"学习率: {self.learning_rate} 权重衰减: {self.weight_decay}")
        print(f"调度器: cosine 早停耐心: 10")
    
    def train_epoch(self, epoch: int) -> tuple:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            
            # 处理不同模型的输出格式
            if isinstance(output, dict):
                # 多任务模型
                if self.model_name in ['mic_mobilenetv3', 'micro_vit', 'airbubble_hybrid_net', 'enhanced_airbubble_detector']:
                    logits = output['classification']
                else:
                    logits = output.get('logits', output.get('classification', list(output.values())[0]))
            else:
                logits = output
            
            # 计算损失
            loss = nn.CrossEntropyLoss()(logits, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 打印进度
            if (batch_idx + 1) % 20 == 0:
                acc = 100. * correct / total
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} - Loss: {loss.item():.4f} Acc: {acc:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> tuple:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                
                # 处理不同模型的输出格式
                if isinstance(output, dict):
                    # 多任务模型
                    if self.model_name in ['mic_mobilenetv3', 'micro_vit', 'airbubble_hybrid_net', 'enhanced_airbubble_detector']:
                        logits = output['classification']
                    else:
                        logits = output.get('logits', output.get('classification', list(output.values())[0]))
                else:
                    logits = output
                
                # 计算损失
                loss = nn.CrossEntropyLoss()(logits, target)
                
                # 统计
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.experiment_dir / 'latest_checkpoint.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.experiment_dir / 'best_model.pth')
    
    def save_training_history(self):
        """保存训练历史"""
        with open(self.experiment_dir / 'training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    def train(self):
        """主训练循环"""
        print("🚀 Starting training...")
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['learning_rates'].append(current_lr)
            
            # 检查是否为最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                print(f"🎯 New best validation accuracy: {val_acc:.4f}")
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 打印epoch结果
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s) - "
                  f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} - LR: {current_lr:.6f}")
        
        # 训练完成
        total_time = time.time() - start_time
        print(f"✅ Training completed in {total_time:.1f}s")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        
        # 保存训练历史
        self.save_training_history()
        
        return self.best_val_acc
    
    def run(self):
        """运行完整的训练流程"""
        try:
            # 创建模型
            self.model = self.create_model()
            
            # 创建数据加载器
            self.create_data_loaders()
            
            # 创建优化器和调度器
            self.create_optimizer_and_scheduler()
            
            # 开始训练
            best_acc = self.train()
            
            return best_acc
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0

def get_model_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """获取所有模型配置，按参数量排序"""
    models_info = [
        # 估计参数量（百万）
        ('vit_tiny', {'estimated_params': 0.5, 'batch_size': 16, 'epochs': 2}),
        ('micro_vit', {'estimated_params': 1.8, 'batch_size': 16, 'epochs': 2}),
        ('mic_mobilenetv3', {'estimated_params': 2.5, 'batch_size': 16, 'epochs': 2}),
        ('airbubble_hybrid_net', {'estimated_params': 3.2, 'batch_size': 16, 'epochs': 2}),
        ('enhanced_airbubble_detector', {'estimated_params': 4.0, 'batch_size': 16, 'epochs': 2}),
        ('efficientnet_b0', {'estimated_params': 5.3, 'batch_size': 16, 'epochs': 2}),
        ('efficientnet_b1', {'estimated_params': 7.8, 'batch_size': 16, 'epochs': 2}),
        ('resnet18_improved', {'estimated_params': 11.2, 'batch_size': 16, 'epochs': 2}),
        ('resnet34_improved', {'estimated_params': 21.3, 'batch_size': 16, 'epochs': 2}),
        ('resnet50_improved', {'estimated_params': 23.5, 'batch_size': 16, 'epochs': 2}),
        ('coatnet', {'estimated_params': 25.0, 'batch_size': 16, 'epochs': 2}),
        ('convnext_tiny', {'estimated_params': 28.6, 'batch_size': 16, 'epochs': 2}),
        
        # 新增模型 - EfficientNet V2系列
        ('efficientnetv2_s', {'estimated_params': 21.5, 'batch_size': 16, 'epochs': 2}),
        ('efficientnetv2_m', {'estimated_params': 54.1, 'batch_size': 16, 'epochs': 2}),
        
        # 新增模型 - MobileNet V3系列
        ('mobilenetv3_large', {'estimated_params': 5.4, 'batch_size': 16, 'epochs': 2}),
        ('mobilenetv3_small', {'estimated_params': 2.9, 'batch_size': 16, 'epochs': 2}),
        
        # 新增模型 - RegNet系列
        ('regnet_x_400mf', {'estimated_params': 5.2, 'batch_size': 16, 'epochs': 2}),
        ('regnet_y_400mf', {'estimated_params': 4.3, 'batch_size': 16, 'epochs': 2}),
        
        # 新增模型 - DenseNet系列
        ('densenet121', {'estimated_params': 8.0, 'batch_size': 16, 'epochs': 2}),
        ('densenet169', {'estimated_params': 14.1, 'batch_size': 16, 'epochs': 2}),
        
        # 新增模型 - 轻量级模型
        ('shufflenetv2_x0_5', {'estimated_params': 1.4, 'batch_size': 16, 'epochs': 2}),
        ('shufflenetv2_x1_0', {'estimated_params': 2.3, 'batch_size': 16, 'epochs': 2}),
        ('ghostnet', {'estimated_params': 5.2, 'batch_size': 16, 'epochs': 2}),
        ('mnasnet_1_0', {'estimated_params': 4.4, 'batch_size': 16, 'epochs': 2}),
    ]
    
    # 按参数量排序
    models_info.sort(key=lambda x: x[1]['estimated_params'])
    
    return models_info

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train all models for GPU performance testing')
    parser.add_argument('--model', type=str, help='Specific model to train (optional)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--skip_trained', action='store_true', help='Skip already trained models')
    
    args = parser.parse_args()
    
    # 获取模型配置
    models_info = get_model_configs()
    
    if args.model:
        # 训练指定模型
        model_found = False
        for model_name, config in models_info:
            if model_name == args.model:
                config.update({
                    'batch_size': args.batch_size,
                    'epochs': args.epochs
                })
                
                print(f"\n🚀 Training {model_name}")
                print("=" * 50)
                print(f"📱 Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
                print(f"📊 Batch size: {config['batch_size']}")
                print(f"🔄 Max epochs: {config['epochs']}")
                print(f"📈 Estimated parameters: {config['estimated_params']}M")
                
                trainer = UniversalModelTrainer(model_name, config)
                best_acc = trainer.run()
                
                print(f"\n🎉 Training completed for {model_name}")
                print(f"🏆 Best validation accuracy: {best_acc:.4f}")
                
                model_found = True
                break
        
        if not model_found:
            print(f"❌ Model '{args.model}' not found!")
            print("Available models:")
            for model_name, _ in models_info:
                print(f"  - {model_name}")
    else:
        # 训练所有模型
        print("🚀 Starting GPU Performance Testing for All Models")
        print("=" * 60)
        print(f"📱 Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        print(f"📊 Total models: {len(models_info)}")
        print(f"🔄 Epochs per model: {args.epochs}")
        
        results = []
        
        for i, (model_name, config) in enumerate(models_info, 1):
            config.update({
                'batch_size': args.batch_size,
                'epochs': args.epochs
            })
            
            print(f"\n[{i}/{len(models_info)}] 🚀 Training {model_name}")
            print("=" * 50)
            print(f"📈 Estimated parameters: {config['estimated_params']}M")
            print(f"📊 Batch size: {config['batch_size']}")
            
            start_time = time.time()
            trainer = UniversalModelTrainer(model_name, config)
            best_acc = trainer.run()
            training_time = time.time() - start_time
            
            results.append({
                'model': model_name,
                'params': config['estimated_params'],
                'best_acc': best_acc,
                'training_time': training_time
            })
            
            print(f"\n🎉 Completed {model_name}")
            print(f"🏆 Best accuracy: {best_acc:.4f}")
            print(f"⏱️ Training time: {training_time:.1f}s")
        
        # 打印总结
        print("\n" + "=" * 60)
        print("📊 GPU Performance Testing Results Summary")
        print("=" * 60)
        
        for result in results:
            print(f"{result['model']:<25} | {result['params']:>6.1f}M | {result['best_acc']:>6.4f} | {result['training_time']:>8.1f}s")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'gpu_performance_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()