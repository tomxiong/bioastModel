#!/usr/bin/env python3
"""
测试修复后的MobileNetV3模型
"""

import torch
import sys
import os
sys.path.append(os.path.abspath('.'))

from models.fixed_mobilenetv3_multitask import create_fixed_mobilenetv3_multitask

def test_model():
    """测试模型"""
    # 创建模型
    num_classes = {
        'growth_level': 2,
        'growth_pattern': 12,
        'interference_factors': 4,
        'microbe_type': 4
    }
    
    model = create_fixed_mobilenetv3_multitask(num_classes)
    model.eval()
    
    # 测试输入
    x = torch.randn(2, 1, 70, 70)
    print(f"输入形状: {x.shape}")
    
    try:
        with torch.no_grad():
            outputs = model(x)
        
        print("\n模型输出:")
        for task, output in outputs.items():
            print(f"  {task}: {output.shape}")
        
        # 测试损失计算
        targets = {
            'growth_level': torch.randint(0, 2, (2,)),
            'growth_pattern': torch.randint(0, 12, (2,)),
            'interference_factors': torch.randint(0, 2, (2, 4)).float(),
            'microbe_type': torch.randint(0, 4, (2,)),
            'confidence': torch.rand(2)
        }
        
        loss, individual_losses = model.compute_loss(outputs, targets)
        print(f"\n总损失: {loss.item():.4f}")
        print("各任务损失:")
        for task, loss_val in individual_losses.items():
            print(f"  {task}: {loss_val.item():.4f}")
        
        print("\n✅ 模型测试成功!")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()