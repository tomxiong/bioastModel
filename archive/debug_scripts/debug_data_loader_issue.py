#!/usr/bin/env python3
"""
调试数据加载器问题
检查数据加载器返回的数据格式和类型
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import create_data_loaders

def debug_data_loader():
    """调试数据加载器返回的数据格式"""
    print("🔍 调试数据加载器...")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders("test_data", batch_size=4)
    
    # 检查第一个批次
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"\n📊 批次 {batch_idx}:")
        print(f"  批次数据类型: {type(batch_data)}")
        print(f"  批次数据长度: {len(batch_data)}")
        
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            images, labels = batch_data
            print(f"  图像类型: {type(images)}")
            print(f"  图像形状: {images.shape}")
            print(f"  标签类型: {type(labels)}")
            
            if isinstance(labels, dict):
                print(f"  标签键: {list(labels.keys())}")
                for key, value in labels.items():
                    print(f"    {key}: 形状={value.shape}, 类型={value.dtype}")
            else:
                print(f"  标签形状: {labels.shape}")
                print(f"  标签数据类型: {labels.dtype}")
        else:
            print(f"  意外的批次数据格式: {batch_data}")
        
        if batch_idx >= 1:  # 只检查前2个批次
            break
    
    print("\n✅ 数据加载器调试完成")

if __name__ == "__main__":
    debug_data_loader()