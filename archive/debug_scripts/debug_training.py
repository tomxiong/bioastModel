#!/usr/bin/env python3
"""
调试训练过程中标签处理的脚本
"""

import torch
import numpy as np
from utils.data_loader import create_data_loaders
from models.enhanced_multilevel_mobilenetv3 import EnhancedMultiLevelMobileNetV3

def debug_training_labels():
    print("🔍 调试训练过程中的标签处理...")
    
    # 配置
    config = {
        'num_classes': {
            'growth_level': 2,
            'growth_pattern': 12,
            'interference_factors': 4
        }
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = EnhancedMultiLevelMobileNetV3(
        model_size='small',
        input_channels=1,
        dropout_rate=0.2,
        use_pores_attention=True
    ).to(device)
    
    # 创建数据加载器
    train_loader, _ = create_data_loaders("test_data", batch_size=4)
    
    # 获取一个批次并模拟训练过程
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"\n=== 批次 {batch_idx} ===")
        images = images.to(device)
        
        print("原始标签:")
        for task_name, task_labels in labels.items():
            print(f"  {task_name}: 形状={task_labels.shape}, 类型={task_labels.dtype}")
        
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
                print(f"    值: {task_labels}")
        
        # 前向传播
        outputs = model(images)
        print("\n模型输出:")
        for task_name, output in outputs.items():
            print(f"  {task_name}: 形状={output.shape}")
        
        # 计算损失
        try:
            losses = model.compute_enhanced_loss(outputs, batch_labels)
            print("\n损失计算成功:")
            for loss_name, loss_value in losses.items():
                print(f"  {loss_name}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"\n❌ 损失计算失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            
            # 详细检查interference_factors
            if 'interference_factors' in outputs and 'interference_factors' in batch_labels:
                print(f"\ninterference_factors详细信息:")
                print(f"  输出形状: {outputs['interference_factors'].shape}")
                print(f"  标签形状: {batch_labels['interference_factors'].shape}")
                print(f"  输出类型: {outputs['interference_factors'].dtype}")
                print(f"  标签类型: {batch_labels['interference_factors'].dtype}")
        
        break  # 只处理第一个批次
    
    print("\n✅ 调试完成")

if __name__ == "__main__":
    debug_training_labels()