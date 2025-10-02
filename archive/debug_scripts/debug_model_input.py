#!/usr/bin/env python3
"""
调试模型输入的脚本
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_multilevel_mobilenetv3 import EnhancedMultiLevelMobileNetV3

def debug_model_input():
    """调试模型输入"""
    print("🔍 调试模型输入...")
    
    # 配置
    config = {
        'num_classes': {
            'growth_level': 2,
            'growth_pattern': 12,
            'interference_factors': 4,
            'pores_detection': 2
        }
    }
    
    # 创建模型
    model = EnhancedMultiLevelMobileNetV3(config['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建模拟数据
    num_samples = 8
    images = torch.randn(num_samples, 1, 70, 70)
    labels = {
        'growth_level': torch.randint(0, 2, (num_samples,)),
        'growth_pattern': torch.randint(0, 12, (num_samples,)),
        'interference_factors': torch.randint(0, 2, (num_samples, 4)).float()
    }
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(images, labels['growth_level'], labels['growth_pattern'], labels['interference_factors'])
    
    # 修改数据加载器以返回正确格式
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = {
            'growth_level': torch.stack([item[1] for item in batch]),
            'growth_pattern': torch.stack([item[2] for item in batch]),
            'interference_factors': torch.stack([item[3] for item in batch])
        }
        return images, labels
    
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # 获取一个批次的数据
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"\n=== 批次 {batch_idx} ===")
        
        print(f"输入图像:")
        print(f"  类型: {type(images)}")
        print(f"  形状: {images.shape}")
        print(f"  数据类型: {images.dtype}")
        
        # 移动数据到设备
        images = images.to(device)
        
        print(f"\n移动到设备后:")
        print(f"  类型: {type(images)}")
        print(f"  形状: {images.shape}")
        print(f"  设备: {images.device}")
        
        # 尝试前向传播
        try:
            print(f"\n🔄 尝试前向传播...")
            outputs = model(images)
            print(f"✅ 前向传播成功!")
            
            print(f"\n模型输出:")
            for task_name, output in outputs.items():
                print(f"  {task_name}: 形状={output.shape}, 类型={output.dtype}")
                
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            
            # 检查模型的第一层
            print(f"\n检查模型结构:")
            print(f"模型类型: {type(model)}")
            
            # 尝试手动调用模型的第一层
            try:
                print(f"\n尝试调用模型的backbone...")
                if hasattr(model, 'backbone'):
                    print(f"backbone类型: {type(model.backbone)}")
                    if hasattr(model.backbone, 'features'):
                        print(f"features类型: {type(model.backbone.features)}")
                        first_layer = model.backbone.features[0]
                        print(f"第一层类型: {type(first_layer)}")
                        print(f"第一层: {first_layer}")
                        
                        # 尝试调用第一层
                        output = first_layer(images)
                        print(f"第一层输出形状: {output.shape}")
                        
            except Exception as inner_e:
                print(f"调用第一层失败: {inner_e}")
        
        break  # 只测试第一个批次
    
    print("\n✅ 调试完成")

if __name__ == "__main__":
    debug_model_input()