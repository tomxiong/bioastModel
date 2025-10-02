#!/usr/bin/env python3
"""
测试增强版多级MobileNetV3训练功能
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enhanced_multilevel_mobilenetv3 import EnhancedMultiLevelMobileNetV3, PoresSpecificAugmentation

def test_enhanced_training():
    """测试增强版训练功能"""
    print("=== 增强版多级MobileNetV3训练测试 ===")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # Growth Pattern类别权重
    growth_pattern_weights = torch.tensor([
        5.219, 2.156, 1.847, 1.234, 0.892, 0.654,
        0.456, 0.321, 0.234, 0.156, 0.089, 0.019
    ], device=device)
    
    # 创建模型
    model = EnhancedMultiLevelMobileNetV3(
        model_size='small',
        input_channels=1,
        dropout_rate=0.2,
        use_pores_attention=True,
        growth_pattern_weights=growth_pattern_weights.cpu().tolist()
    ).to(device)
    
    print(f"✅ 模型创建成功")
    print(f"📊 模型信息: {model.get_model_info()}")
    
    # 创建模拟数据
    batch_size = 8
    num_samples = 32
    
    images = torch.randn(num_samples, 1, 70, 70)
    labels = {
        'growth_level': torch.randint(0, 2, (num_samples,)),
        'growth_pattern': torch.randint(0, 12, (num_samples,)),
        'interference_factors': torch.randint(0, 2, (num_samples, 4)).float()  # 多标签格式
    }
    
    # 创建数据加载器
    def collate_fn(batch):
        batch_images = torch.stack([item[0] for item in batch])
        batch_labels = {
            'growth_level': torch.stack([item[1] for item in batch]),
            'growth_pattern': torch.stack([item[2] for item in batch]),
            'interference_factors': torch.stack([item[3] for item in batch])
        }
        return batch_images, batch_labels
    
    dataset = TensorDataset(images, labels['growth_level'], labels['growth_pattern'], labels['interference_factors'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"✅ 数据加载器创建成功，样本数: {len(dataset)}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 创建Pores专用数据增强
    pores_augmentation = PoresSpecificAugmentation()
    
    print("\n=== 开始训练测试 ===")
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (batch_images, batch_labels) in enumerate(dataloader):
        batch_images = batch_images.to(device)
        
        # 转换标签到设备
        for task_name in batch_labels:
            batch_labels[task_name] = batch_labels[task_name].to(device)
        
        # 添加pores检测标签
        pores_labels = (batch_labels['growth_pattern'] == 11).long()
        batch_labels['pores_detection'] = pores_labels
        
        # 应用数据增强（测试）
        if np.random.random() < 0.5:
            batch_images = pores_augmentation.enhance_pores_contrast(batch_images)
        if np.random.random() < 0.3:
            batch_images = pores_augmentation.pores_edge_enhancement(batch_images)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch_images)
        
        # 计算损失
        loss_dict = model.compute_enhanced_loss(outputs, batch_labels)
        total_batch_loss = loss_dict['total']
        
        # 反向传播
        total_batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        num_batches += 1
        
        print(f"Batch {batch_idx + 1}/{len(dataloader)}:")
        print(f"  总损失: {total_batch_loss.item():.4f}")
        for task_name, loss_value in loss_dict.items():
            if task_name != 'total':
                print(f"  {task_name}: {loss_value.item():.4f}")
        
        # 测试预测
        predictions = model.predict(batch_images)
        print(f"  预测形状: {[(task, pred.shape) for task, pred in predictions.items()]}")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    avg_loss = total_loss / num_batches
    print(f"\n✅ 训练测试完成")
    print(f"📊 平均损失: {avg_loss:.4f}")
    
    # 测试验证模式
    print("\n=== 验证模式测试 ===")
    model.eval()
    
    with torch.no_grad():
        batch_images, batch_labels = next(iter(dataloader))
        batch_images = batch_images.to(device)
        
        for task_name in batch_labels:
            batch_labels[task_name] = batch_labels[task_name].to(device)
        
        batch_labels['pores_detection'] = (batch_labels['growth_pattern'] == 11).long()
        
        outputs = model(batch_images)
        loss_dict = model.compute_enhanced_loss(outputs, batch_labels)
        predictions = model.predict(batch_images)
        
        print(f"验证损失: {loss_dict['total'].item():.4f}")
        
        # 计算准确率
        for task_name, pred in predictions.items():
            if task_name == 'interference_factors':
                # 多标签分类：使用阈值0.5
                pred_binary = (torch.sigmoid(pred) > 0.5).float()
                correct = (pred_binary == batch_labels[task_name]).all(dim=1).sum().item()
            elif task_name == 'pores_detection':
                # 二分类
                pred_class = torch.argmax(pred, dim=1)
                # 生成随机的pores_detection标签
                pores_labels = torch.randint(0, 2, (pred.size(0),)).to(pred.device)
                correct = (pred_class == pores_labels).sum().item()
            else:
                # 单标签分类
                pred_class = torch.argmax(pred, dim=1)
                correct = (pred_class == batch_labels[task_name]).sum().item()
            
            accuracy = correct / pred.size(0)
            print(f"{task_name} 准确率: {accuracy:.4f}")
    
    print("\n=== 特殊功能测试 ===")
    
    # 测试Growth Pattern权重
    print(f"Growth Pattern权重: {model.growth_pattern_weights}")
    
    # 测试Focal Loss参数
    print(f"Focal Loss参数 - Alpha: {model.focal_loss.alpha}, Gamma: {model.focal_loss.gamma}")
    
    # 测试Pores注意力
    if model.use_pores_attention:
        print("✅ Pores注意力模块已启用")
    
    # 测试数据增强
    test_image = torch.randn(1, 70, 70).to(device)
    enhanced_image = pores_augmentation.enhance_pores_contrast(test_image)
    edge_enhanced = pores_augmentation.pores_edge_enhancement(test_image)
    hist_eq = pores_augmentation.adaptive_histogram_equalization(test_image)
    
    print(f"数据增强测试:")
    print(f"  原始图像: {test_image.shape}")
    print(f"  对比度增强: {enhanced_image.shape}")
    print(f"  边缘增强: {edge_enhanced.shape}")
    print(f"  直方图均衡: {hist_eq.shape}")
    
    print("\n✅ 所有测试通过！增强版模型功能正常")

if __name__ == "__main__":
    test_enhanced_training()