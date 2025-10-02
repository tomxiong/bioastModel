#!/usr/bin/env python3
"""
调试训练过程中标签处理的脚本
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_multilevel_mobilenetv3 import EnhancedMultiLevelMobileNetV3
from utils.data_loader import create_data_loaders

def debug_training_labels():
    """调试训练过程中的标签处理"""
    print("🔍 调试训练过程中的标签处理...")
    
    # 配置
    config = {
        'num_classes': {
            'growth_level': 2,
            'growth_pattern': 12,
            'interference_factors': 4,
            'pores_detection': 2
        }
    }
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders("test_data", batch_size=4)
    
    # 创建模型
    model = EnhancedMultiLevelMobileNetV3(config['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 获取一个批次的数据
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"\n=== 批次 {batch_idx} ===")
        
        # 移动数据到设备
        images = images.to(device)
        
        print("原始标签:")
        for task_name, task_labels in labels.items():
            print(f"  {task_name}: 形状={task_labels.shape}, 类型={task_labels.dtype}")
            if task_name == 'interference_factors':
                print(f"    前2个样本: {task_labels[:2]}")
        
        # 模拟训练脚本中的标签处理
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
        
        print("\n处理后的标签:")
        for task_name, task_labels in batch_labels.items():
            print(f"  {task_name}: 形状={task_labels.shape}, 类型={task_labels.dtype}")
            if task_name == 'interference_factors':
                print(f"    前2个样本: {task_labels[:2]}")
        
        # 前向传播
        outputs = model(images)
        
        print("\n模型输出:")
        for task_name, output in outputs.items():
            print(f"  {task_name}: 形状={output.shape}")
            if task_name == 'interference_factors':
                print(f"    前2个样本: {output[:2]}")
        
        # 尝试计算损失
        try:
            loss_dict = model.compute_enhanced_loss(outputs, batch_labels)
            print(f"\n✅ 损失计算成功:")
            for task_name, loss_value in loss_dict.items():
                print(f"  {task_name}: {loss_value:.4f}")
        except Exception as e:
            print(f"\n❌ 损失计算失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            
            # 详细检查每个任务的输出和标签
            print("\n详细检查:")
            for task_name in outputs.keys():
                if task_name in batch_labels:
                    output = outputs[task_name]
                    target = batch_labels[task_name]
                    print(f"  {task_name}:")
                    print(f"    输出形状: {output.shape}, 类型: {output.dtype}")
                    print(f"    标签形状: {target.shape}, 类型: {target.dtype}")
                    
                    if task_name == 'interference_factors':
                        print(f"    输出值范围: [{output.min():.4f}, {output.max():.4f}]")
                        print(f"    标签值范围: [{target.min():.4f}, {target.max():.4f}]")
                        print(f"    输出前2个样本: {output[:2]}")
                        print(f"    标签前2个样本: {target[:2]}")
        
        if batch_idx >= 1:  # 只测试前2个批次
            break
    
    print("\n✅ 调试完成")

if __name__ == "__main__":
    debug_training_labels()