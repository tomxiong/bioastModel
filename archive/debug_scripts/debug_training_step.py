#!/usr/bin/env python3
"""
调试完整训练步骤
模拟训练脚本中的完整流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_multilevel_mobilenetv3 import (
    EnhancedMultiLevelMobileNetV3, 
    PoresSpecificAugmentation
)
from utils.data_loader import create_synthetic_data_loaders

def debug_training_step():
    """调试完整训练步骤"""
    print("🔍 调试完整训练步骤...")
    
    # 配置
    config = {
        'num_classes': {
            'growth_level': 2,
            'growth_pattern': 12,
            'interference_factors': 4
        },
        'batch_size': 4,
        'learning_rate': 0.001
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = EnhancedMultiLevelMobileNetV3(
        model_size='small',
        input_channels=1,
        dropout_rate=0.3
    ).to(device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 创建数据增强
    pores_augmentation = PoresSpecificAugmentation()
    
    # 创建数据加载器
    train_loader, _ = create_synthetic_data_loaders(batch_size=config['batch_size'])
    
    # 模拟训练步骤
    model.train()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"\n📊 批次 {batch_idx}:")
        print(f"  原始图像形状: {images.shape}")
        print(f"  原始标签: {list(labels.keys())}")
        
        # 移动到设备
        images = images.to(device)
        print(f"  移动后图像形状: {images.shape}")
        
        # 转换标签格式（模拟训练脚本中的处理）
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
        
        print(f"  处理后标签:")
        for key, value in batch_labels.items():
            print(f"    {key}: 形状={value.shape}, 类型={value.dtype}")
        
        # 应用数据增强（模拟训练脚本中的处理）
        if np.random.random() < 0.3:  # 30%概率应用对比度增强
            print("  应用对比度增强...")
            images = pores_augmentation.enhance_pores_contrast(images)
        if np.random.random() < 0.2:  # 20%概率应用边缘增强
            print("  应用边缘增强...")
            images = pores_augmentation.pores_edge_enhancement(images)
        
        # 前向传播
        try:
            print("  🚀 前向传播...")
            optimizer.zero_grad()
            outputs = model(images)
            print(f"  ✅ 前向传播成功!")
            print(f"  模型输出:")
            for key, value in outputs.items():
                print(f"    {key}: 形状={value.shape}, 类型={value.dtype}")
            
            # 计算损失
            print("  📊 计算损失...")
            loss_dict = model.compute_enhanced_loss(outputs, batch_labels)
            total_batch_loss = loss_dict['total']
            print(f"  ✅ 损失计算成功!")
            print(f"  总损失: {total_batch_loss.item():.4f}")
            
            # 反向传播
            print("  🔄 反向传播...")
            total_batch_loss.backward()
            optimizer.step()
            print("  ✅ 反向传播成功!")
            
        except Exception as e:
            print(f"  ❌ 训练步骤失败: {e}")
            print(f"  错误类型: {type(e)}")
            import traceback
            traceback.print_exc()
            break
        
        if batch_idx >= 1:  # 只测试前2个批次
            break
    
    print("\n✅ 完整训练步骤调试完成")

if __name__ == "__main__":
    debug_training_step()