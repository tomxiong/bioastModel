#!/usr/bin/env python3
"""
模型性能对比脚本
比较原始MultiLevel MobileNetV3和增强版模型的性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import sys

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from models.multilevel_mobilenetv3 import MultiLevelMobileNetV3
    from models.enhanced_multilevel_mobilenetv3 import EnhancedMultiLevelMobileNetV3, PoresSpecificAugmentation
except ImportError:
    print("警告：无法导入原始模型，将只测试增强版模型")
    MultiLevelMobileNetV3 = None

class ModelComparator:
    """模型性能对比器"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def create_test_data(self, num_samples: int = 1000) -> Tuple[DataLoader, DataLoader]:
        """创建测试数据"""
        print(f"📊 创建测试数据集，样本数: {num_samples}")
        
        # 生成模拟数据
        images = torch.randn(num_samples, 1, 70, 70)
        labels = {
            'growth_level': torch.randint(0, 2, (num_samples,)),
            'growth_pattern': torch.randint(0, 12, (num_samples,)),
            'interference_factors': torch.randint(0, 2, (num_samples, 4)).float()
        }
        
        # 创建数据集
        dataset = TensorDataset(
            images, 
            labels['growth_level'], 
            labels['growth_pattern'], 
            labels['interference_factors']
        )
        
        # 分割训练和测试集
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader
    
    def test_model_performance(self, model: nn.Module, model_name: str, 
                             train_loader: DataLoader, test_loader: DataLoader,
                             num_epochs: int = 5) -> Dict:
        """测试模型性能"""
        print(f"\n🔬 测试模型: {model_name}")
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 记录性能指标
        metrics = {
            'train_losses': [],
            'test_losses': [],
            'train_accuracies': [],
            'test_accuracies': [],
            'training_time': 0,
            'inference_time': 0,
            'model_size': self._get_model_size(model),
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        # 训练阶段
        start_time = time.time()
        for epoch in range(num_epochs):
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer)
            test_loss, test_acc = self._evaluate_model(model, test_loader)
            
            metrics['train_losses'].append(train_loss)
            metrics['test_losses'].append(test_loss)
            metrics['train_accuracies'].append(train_acc)
            metrics['test_accuracies'].append(test_acc)
            
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        metrics['training_time'] = time.time() - start_time
        
        # 推理速度测试
        metrics['inference_time'] = self._test_inference_speed(model, test_loader)
        
        return metrics
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, growth_level, growth_pattern, interference_factors) in enumerate(train_loader):
            images = images.to(self.device)
            growth_level = growth_level.to(self.device)
            growth_pattern = growth_pattern.to(self.device)
            interference_factors = interference_factors.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            if hasattr(model, 'compute_enhanced_loss'):
                # 增强版模型
                targets = {
                    'growth_level': growth_level,
                    'growth_pattern': growth_pattern,
                    'interference_factors': interference_factors,
                    'pores_detection': torch.randint(0, 2, (images.size(0),)).to(self.device)
                }
                loss_dict = model.compute_enhanced_loss(outputs, targets)
                loss = sum(loss_dict.values())
            else:
                # 原始模型
                loss = self._compute_original_loss(outputs, growth_level, growth_pattern, interference_factors)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率（使用growth_level作为主要指标）
            if 'growth_level' in outputs:
                pred = torch.argmax(outputs['growth_level'], dim=1)
                correct_predictions += (pred == growth_level).sum().item()
                total_samples += growth_level.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Tuple[float, float]:
        """评估模型"""
        model.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, growth_level, growth_pattern, interference_factors in test_loader:
                images = images.to(self.device)
                growth_level = growth_level.to(self.device)
                growth_pattern = growth_pattern.to(self.device)
                interference_factors = interference_factors.to(self.device)
                
                outputs = model(images)
                
                # 计算损失
                if hasattr(model, 'compute_enhanced_loss'):
                    targets = {
                        'growth_level': growth_level,
                        'growth_pattern': growth_pattern,
                        'interference_factors': interference_factors,
                        'pores_detection': torch.randint(0, 2, (images.size(0),)).to(self.device)
                    }
                    loss_dict = model.compute_enhanced_loss(outputs, targets)
                    loss = sum(loss_dict.values())
                else:
                    loss = self._compute_original_loss(outputs, growth_level, growth_pattern, interference_factors)
                
                total_loss += loss.item()
                
                # 计算准确率
                if 'growth_level' in outputs:
                    pred = torch.argmax(outputs['growth_level'], dim=1)
                    correct_predictions += (pred == growth_level).sum().item()
                    total_samples += growth_level.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return avg_loss, accuracy
    
    def _compute_original_loss(self, outputs: Dict, growth_level: torch.Tensor, 
                             growth_pattern: torch.Tensor, interference_factors: torch.Tensor) -> torch.Tensor:
        """计算原始模型损失"""
        criterion_ce = nn.CrossEntropyLoss()
        criterion_bce = nn.BCEWithLogitsLoss()
        
        loss = 0
        if 'growth_level' in outputs:
            loss += criterion_ce(outputs['growth_level'], growth_level)
        if 'growth_pattern' in outputs:
            loss += criterion_ce(outputs['growth_pattern'], growth_pattern)
        if 'interference_factors' in outputs:
            loss += criterion_bce(outputs['interference_factors'], interference_factors)
        
        return loss
    
    def _test_inference_speed(self, model: nn.Module, test_loader: DataLoader) -> float:
        """测试推理速度"""
        model.eval()
        total_time = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, _, _, _ in test_loader:
                images = images.to(self.device)
                
                start_time = time.time()
                _ = model(images)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                total_samples += images.size(0)
        
        avg_inference_time = total_time / total_samples * 1000  # ms per sample
        return avg_inference_time
    
    def _get_model_size(self, model: nn.Module) -> float:
        """获取模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def compare_models(self, num_samples: int = 1000, num_epochs: int = 5):
        """比较模型性能"""
        print("🚀 开始模型性能对比")
        
        # 创建测试数据
        train_loader, test_loader = self.create_test_data(num_samples)
        
        # 测试增强版模型
        print("\n" + "="*50)
        enhanced_model = EnhancedMultiLevelMobileNetV3(
            model_size='small',
            input_channels=1,
            use_pores_attention=True
        )
        self.results['Enhanced'] = self.test_model_performance(
            enhanced_model, "Enhanced MultiLevel MobileNetV3", 
            train_loader, test_loader, num_epochs
        )
        
        # 测试原始模型（如果可用）
        if MultiLevelMobileNetV3 is not None:
            print("\n" + "="*50)
            try:
                original_model = MultiLevelMobileNetV3(
                    model_size='small',
                    input_channels=1
                )
                self.results['Original'] = self.test_model_performance(
                    original_model, "Original MultiLevel MobileNetV3",
                    train_loader, test_loader, num_epochs
                )
            except Exception as e:
                print(f"⚠️ 无法测试原始模型: {e}")
                self.results['Original'] = None
        else:
            print("\n⚠️ 原始模型不可用，跳过对比")
            self.results['Original'] = None
    
    def print_comparison_results(self):
        """打印对比结果"""
        print("\n" + "="*60)
        print("📊 模型性能对比结果")
        print("="*60)
        
        if self.results['Original'] is not None:
            # 完整对比
            original = self.results['Original']
            enhanced = self.results['Enhanced']
            
            print(f"{'指标':<20} {'原始模型':<15} {'增强版模型':<15} {'改进':<10}")
            print("-" * 60)
            
            # 最终测试准确率
            orig_acc = original['test_accuracies'][-1] if original['test_accuracies'] else 0
            enh_acc = enhanced['test_accuracies'][-1] if enhanced['test_accuracies'] else 0
            acc_improvement = ((enh_acc - orig_acc) / orig_acc * 100) if orig_acc > 0 else 0
            print(f"{'测试准确率':<20} {orig_acc:<15.4f} {enh_acc:<15.4f} {acc_improvement:+.2f}%")
            
            # 最终测试损失
            orig_loss = original['test_losses'][-1] if original['test_losses'] else 0
            enh_loss = enhanced['test_losses'][-1] if enhanced['test_losses'] else 0
            loss_improvement = ((orig_loss - enh_loss) / orig_loss * 100) if orig_loss > 0 else 0
            print(f"{'测试损失':<20} {orig_loss:<15.4f} {enh_loss:<15.4f} {loss_improvement:+.2f}%")
            
            # 训练时间
            time_diff = enhanced['training_time'] - original['training_time']
            print(f"{'训练时间(s)':<20} {original['training_time']:<15.2f} {enhanced['training_time']:<15.2f} {time_diff:+.2f}s")
            
            # 推理时间
            inf_diff = enhanced['inference_time'] - original['inference_time']
            print(f"{'推理时间(ms)':<20} {original['inference_time']:<15.4f} {enhanced['inference_time']:<15.4f} {inf_diff:+.4f}ms")
            
            # 模型大小
            size_diff = enhanced['model_size'] - original['model_size']
            print(f"{'模型大小(MB)':<20} {original['model_size']:<15.2f} {enhanced['model_size']:<15.2f} {size_diff:+.2f}MB")
            
            # 参数数量
            param_diff = enhanced['parameters'] - original['parameters']
            print(f"{'参数数量':<20} {original['parameters']:<15,} {enhanced['parameters']:<15,} {param_diff:+,}")
            
        else:
            # 只显示增强版模型结果
            enhanced = self.results['Enhanced']
            print("增强版模型性能指标:")
            print(f"  最终测试准确率: {enhanced['test_accuracies'][-1]:.4f}")
            print(f"  最终测试损失: {enhanced['test_losses'][-1]:.4f}")
            print(f"  训练时间: {enhanced['training_time']:.2f}s")
            print(f"  推理时间: {enhanced['inference_time']:.4f}ms/sample")
            print(f"  模型大小: {enhanced['model_size']:.2f}MB")
            print(f"  参数数量: {enhanced['parameters']:,}")
        
        print("\n🎯 增强版模型特性:")
        print("  ✅ Growth Pattern类别权重平衡")
        print("  ✅ Focal Loss处理类别不平衡")
        print("  ✅ Pores特定注意力机制")
        print("  ✅ Pores特定损失函数")
        print("  ✅ 专门的数据增强策略")
    
    def save_results(self, filename: str = "model_comparison_results.txt"):
        """保存对比结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("模型性能对比结果\n")
            f.write("="*50 + "\n\n")
            
            for model_name, metrics in self.results.items():
                if metrics is not None:
                    f.write(f"{model_name} 模型:\n")
                    f.write(f"  最终测试准确率: {metrics['test_accuracies'][-1]:.4f}\n")
                    f.write(f"  最终测试损失: {metrics['test_losses'][-1]:.4f}\n")
                    f.write(f"  训练时间: {metrics['training_time']:.2f}s\n")
                    f.write(f"  推理时间: {metrics['inference_time']:.4f}ms/sample\n")
                    f.write(f"  模型大小: {metrics['model_size']:.2f}MB\n")
                    f.write(f"  参数数量: {metrics['parameters']:,}\n\n")
        
        print(f"📄 结果已保存到: {filename}")

def main():
    """主函数"""
    print("🔬 MultiLevel MobileNetV3 模型性能对比")
    
    # 创建对比器
    comparator = ModelComparator()
    
    # 运行对比
    comparator.compare_models(num_samples=1000, num_epochs=3)
    
    # 显示结果
    comparator.print_comparison_results()
    
    # 保存结果
    comparator.save_results()
    
    print("\n✅ 模型对比完成！")

if __name__ == "__main__":
    main()