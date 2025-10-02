#!/usr/bin/env python3
"""
简化的增强版模型训练测试脚本 v2
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_multilevel_mobilenetv3 import EnhancedMultiLevelMobileNetV3, PoresSpecificAugmentation

def test_enhanced_training():
    """测试增强版模型训练"""
    print("🚀 开始测试增强版模型训练...")
    
    # 配置
    config = {
        'num_classes': {
            'growth_level': 2,
            'growth_pattern': 12,
            'interference_factors': 4,
            'pores_detection': 2
        },
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 2
    }
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = EnhancedMultiLevelMobileNetV3(config['num_classes'])
    model = model.to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 创建Pores特定数据增强
    pores_augmentation = PoresSpecificAugmentation()
    
    # 创建模拟数据
    num_samples = 100
    images = torch.randn(num_samples, 1, 70, 70)
    labels = {
        'growth_level': torch.randint(0, 2, (num_samples,)),
        'growth_pattern': torch.randint(0, 12, (num_samples,)),
        'interference_factors': torch.randint(0, 2, (num_samples, 4)).float()  # 多标签二分类
    }
    
    # 创建数据集
    dataset = TensorDataset(images, labels['growth_level'], labels['growth_pattern'], labels['interference_factors'])
    
    # 分割训练和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 修改数据加载器以返回正确格式
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = {
            'growth_level': torch.stack([item[1] for item in batch]),
            'growth_pattern': torch.stack([item[2] for item in batch]),
            'interference_factors': torch.stack([item[3] for item in batch])
        }
        return images, labels
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print(f"📊 数据统计:")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  批次大小: {config['batch_size']}")
    
    # 训练循环
    for epoch in range(config['epochs']):
        print(f"\n📈 Epoch {epoch + 1}/{config['epochs']}")
        
        # 训练阶段
        model.train()
        train_losses = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 移动数据到设备
            images = images.to(device)
            
            # 转换标签格式
            batch_labels = {}
            for task_name in config['num_classes'].keys():
                if task_name in labels:
                    if task_name == 'interference_factors':
                        # interference_factors是多标签任务，确保维度正确
                        batch_labels[task_name] = labels[task_name].to(device).float()
                    else:
                        batch_labels[task_name] = labels[task_name].to(device)
            
            # 添加pores检测标签（基于growth_pattern）
            if 'growth_pattern' in batch_labels:
                # 假设类别11是pores相关的
                pores_labels = (batch_labels['growth_pattern'] == 11).long()
                batch_labels['pores_detection'] = pores_labels
            
            # 应用Pores特定数据增强
            if np.random.random() < 0.3:  # 30%概率应用对比度增强
                images = pores_augmentation.enhance_pores_contrast(images)
            if np.random.random() < 0.2:  # 20%概率应用边缘增强
                images = pores_augmentation.pores_edge_enhancement(images)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            
            # 计算损失
            loss_dict = model.compute_enhanced_loss(outputs, batch_labels)
            total_batch_loss = loss_dict['total']
            
            # 反向传播
            total_batch_loss.backward()
            optimizer.step()
            
            train_losses.append(total_batch_loss.item())
            
            if batch_idx % 2 == 0:  # 每2个批次打印一次
                print(f"  批次 {batch_idx}: 损失 = {total_batch_loss.item():.4f}")
        
        avg_train_loss = np.mean(train_losses)
        print(f"  平均训练损失: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                # 移动数据到设备
                images = images.to(device)
                
                # 转换标签格式
                batch_labels = {}
                for task_name in config['num_classes'].keys():
                    if task_name in labels:
                        if task_name == 'interference_factors':
                            # interference_factors是多标签任务，确保维度正确
                            batch_labels[task_name] = labels[task_name].to(device).float()
                        else:
                            batch_labels[task_name] = labels[task_name].to(device)
                
                # 添加pores检测标签（基于growth_pattern）
                if 'growth_pattern' in batch_labels:
                    # 假设类别11是pores相关的
                    pores_labels = (batch_labels['growth_pattern'] == 11).long()
                    batch_labels['pores_detection'] = pores_labels
                
                # 前向传播
                outputs = model(images)
                
                # 计算损失
                loss_dict = model.compute_enhanced_loss(outputs, batch_labels)
                total_batch_loss = loss_dict['total']
                
                val_losses.append(total_batch_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"  平均验证损失: {avg_val_loss:.4f}")
    
    print("\n✅ 训练测试完成!")
    print(f"最终训练损失: {avg_train_loss:.4f}")
    print(f"最终验证损失: {avg_val_loss:.4f}")

if __name__ == "__main__":
    test_enhanced_training()