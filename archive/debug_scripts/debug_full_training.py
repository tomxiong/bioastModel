#!/usr/bin/env python3
"""
完整的训练调试脚本
"""

import torch
import numpy as np
from utils.data_loader import create_data_loaders
from models.enhanced_multilevel_mobilenetv3 import (
    EnhancedMultiLevelMobileNetV3, 
    PoresSpecificAugmentation
)

def debug_full_training():
    print("🔍 完整训练过程调试...")
    
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
    
    # 创建数据增强
    pores_augmentation = PoresSpecificAugmentation()
    
    # 创建数据加载器
    train_loader, _ = create_data_loaders("test_data", batch_size=32)
    
    # 模拟训练过程
    model.train()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"\n=== 批次 {batch_idx} ===")
        images = images.to(device)
        
        print("原始标签:")
        for task_name, task_labels in labels.items():
            print(f"  {task_name}: 形状={task_labels.shape}, 类型={task_labels.dtype}")
        
        # 完全模拟训练脚本中的标签处理
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
        
        # 计算损失
        try:
            loss_dict = model.compute_enhanced_loss(outputs, batch_labels)
            print("\n✅ 损失计算成功:")
            for loss_name, loss_value in loss_dict.items():
                print(f"  {loss_name}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"\n❌ 损失计算失败: {e}")
            print(f"错误类型: {type(e).__name__}")
            
            # 详细检查每个任务的输出和标签
            for task_name in batch_labels.keys():
                if task_name in outputs:
                    print(f"\n{task_name}详细信息:")
                    print(f"  输出形状: {outputs[task_name].shape}")
                    print(f"  标签形状: {batch_labels[task_name].shape}")
                    print(f"  输出类型: {outputs[task_name].dtype}")
                    print(f"  标签类型: {batch_labels[task_name].dtype}")
                    print(f"  输出设备: {outputs[task_name].device}")
                    print(f"  标签设备: {batch_labels[task_name].device}")
            
            # 尝试单独计算每个任务的损失
            print("\n尝试单独计算损失:")
            
            # Growth level loss
            if 'growth_level' in outputs and 'growth_level' in batch_labels:
                try:
                    focal_loss = torch.nn.CrossEntropyLoss()
                    gl_loss = focal_loss(outputs['growth_level'], batch_labels['growth_level'])
                    print(f"  growth_level损失: {gl_loss.item():.4f}")
                except Exception as e:
                    print(f"  growth_level损失失败: {e}")
            
            # Growth pattern loss
            if 'growth_pattern' in outputs and 'growth_pattern' in batch_labels:
                try:
                    ce_loss = torch.nn.CrossEntropyLoss()
                    gp_loss = ce_loss(outputs['growth_pattern'], batch_labels['growth_pattern'])
                    print(f"  growth_pattern损失: {gp_loss.item():.4f}")
                except Exception as e:
                    print(f"  growth_pattern损失失败: {e}")
            
            # Interference factors loss
            if 'interference_factors' in outputs and 'interference_factors' in batch_labels:
                try:
                    bce_loss = torch.nn.BCEWithLogitsLoss()
                    if_loss = bce_loss(outputs['interference_factors'], batch_labels['interference_factors'])
                    print(f"  interference_factors损失: {if_loss.item():.4f}")
                except Exception as e:
                    print(f"  interference_factors损失失败: {e}")
                    print(f"    输出值范围: {outputs['interference_factors'].min().item():.4f} - {outputs['interference_factors'].max().item():.4f}")
                    print(f"    标签值范围: {batch_labels['interference_factors'].min().item():.4f} - {batch_labels['interference_factors'].max().item():.4f}")
            
            # Pores detection loss
            if 'pores_detection' in outputs and 'pores_detection' in batch_labels:
                try:
                    ce_loss = torch.nn.CrossEntropyLoss()
                    pd_loss = ce_loss(outputs['pores_detection'], batch_labels['pores_detection'])
                    print(f"  pores_detection损失: {pd_loss.item():.4f}")
                except Exception as e:
                    print(f"  pores_detection损失失败: {e}")
            
            break  # 出错时停止
        
        if batch_idx >= 1:  # 只处理前2个批次
            break
    
    print("\n✅ 调试完成")

if __name__ == "__main__":
    debug_full_training()