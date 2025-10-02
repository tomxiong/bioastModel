#!/usr/bin/env python3
"""
调试标签形状和类型的脚本
"""

import torch
from utils.data_loader import create_data_loaders

def debug_labels():
    print("🔍 调试标签形状和类型...")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders("test_data", batch_size=4)
    
    # 获取一个批次
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"\n批次 {batch_idx}:")
        print(f"图像形状: {images.shape}")
        
        for task_name, task_labels in labels.items():
            print(f"{task_name}:")
            print(f"  形状: {task_labels.shape}")
            print(f"  类型: {task_labels.dtype}")
            print(f"  值范围: {task_labels.min().item():.3f} - {task_labels.max().item():.3f}")
            if task_name == 'interference_factors':
                print(f"  前3个样本: {task_labels[:3]}")
        
        if batch_idx >= 1:  # 只检查前2个批次
            break
    
    print("\n✅ 标签调试完成")

if __name__ == "__main__":
    debug_labels()