#!/usr/bin/env python3
"""
精确复现训练脚本中的错误
使用与训练脚本完全相同的配置和流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_multilevel_mobilenetv3 import (
    EnhancedMultiLevelMobileNetV3, 
    PoresSpecificAugmentation
)
from utils.data_loader import create_synthetic_data_loaders

class DebugTrainer:
    """调试训练器，复制原始训练器的逻辑"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = self._create_model()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建数据增强
        self.pores_augmentation = PoresSpecificAugmentation()
        
        print(f"✅ 调试训练器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  模型: {type(self.model).__name__}")
    
    def _create_model(self) -> EnhancedMultiLevelMobileNetV3:
        """创建模型"""
        model = EnhancedMultiLevelMobileNetV3(
            model_size='small',
            input_channels=1,
            dropout_rate=0.3,
            use_hierarchical_loss=True,
            freeze_backbone=False,
            use_pores_attention=True
        )
        return model.to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
    
    def debug_train_step(self, train_loader):
        """调试训练步骤"""
        print("🔍 开始调试训练步骤...")
        
        self.model.train()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"\n📊 批次 {batch_idx}:")
            print(f"  原始数据类型: images={type(images)}, labels={type(labels)}")
            print(f"  图像形状: {images.shape}")
            
            # 检查输入数据
            if not isinstance(images, torch.Tensor):
                print(f"  ❌ 错误: images不是tensor，而是 {type(images)}")
                return False
            
            if not isinstance(labels, dict):
                print(f"  ❌ 错误: labels不是dict，而是 {type(labels)}")
                return False
            
            # 移动到设备
            images = images.to(self.device)
            print(f"  移动后图像形状: {images.shape}")
            
            # 转换标签格式
            batch_labels = {}
            for task_name in self.config['num_classes'].keys():
                if task_name in labels:
                    if task_name == 'interference_factors':
                        batch_labels[task_name] = labels[task_name].to(self.device).float()
                    else:
                        batch_labels[task_name] = labels[task_name].to(self.device)
            
            # 添加pores检测标签
            if 'growth_pattern' in batch_labels:
                pores_labels = (batch_labels['growth_pattern'] == 11).long()
                batch_labels['pores_detection'] = pores_labels
            
            print(f"  处理后标签:")
            for key, value in batch_labels.items():
                print(f"    {key}: 形状={value.shape}, 类型={value.dtype}")
            
            # 应用数据增强
            if np.random.random() < 0.3:
                print("  应用对比度增强...")
                images = self.pores_augmentation.enhance_pores_contrast(images)
            if np.random.random() < 0.2:
                print("  应用边缘增强...")
                images = self.pores_augmentation.pores_edge_enhancement(images)
            
            # 检查增强后的图像
            print(f"  增强后图像类型: {type(images)}")
            print(f"  增强后图像形状: {images.shape}")
            
            # 前向传播
            try:
                print("  🚀 前向传播...")
                self.optimizer.zero_grad()
                
                # 检查模型输入
                if not isinstance(images, torch.Tensor):
                    print(f"  ❌ 模型输入不是tensor: {type(images)}")
                    return False
                
                outputs = self.model(images)
                print(f"  ✅ 前向传播成功!")
                
                # 计算损失
                print("  📊 计算损失...")
                loss_dict = self.model.compute_enhanced_loss(outputs, batch_labels)
                total_batch_loss = loss_dict['total']
                print(f"  ✅ 损失计算成功: {total_batch_loss.item():.4f}")
                
                # 反向传播
                print("  🔄 反向传播...")
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                print("  ✅ 反向传播成功!")
                
            except Exception as e:
                print(f"  ❌ 训练步骤失败: {e}")
                print(f"  错误类型: {type(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            if batch_idx >= 1:  # 只测试前2个批次
                break
        
        return True

def main():
    """主函数"""
    print("🔍 开始精确复现训练脚本错误...")
    
    # 使用与原始训练脚本相同的配置
    config = {
        'num_classes': {
            'growth_level': 2,
            'growth_pattern': 12,
            'interference_factors': 4
        },
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 1
    }
    
    # 创建调试训练器
    trainer = DebugTrainer(config)
    
    # 创建数据加载器
    train_loader, _ = create_synthetic_data_loaders(
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    # 调试训练步骤
    success = trainer.debug_train_step(train_loader)
    
    if success:
        print("\n✅ 调试成功，未发现错误!")
    else:
        print("\n❌ 调试发现错误!")

if __name__ == "__main__":
    main()