#!/usr/bin/env python3
"""
测试多任务灰度菌落检测网络与现有系统的集成
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.multitask_models import create_multitask_model, get_multitask_model_config

def test_integration():
    """测试集成"""
    print("=== 测试多任务灰度菌落检测网络集成 ===")
    
    # 1. 测试获取模型配置
    print("\n1. 获取模型配置...")
    try:
        config = get_multitask_model_config('multitask_gray_colony')
        print(f"✓ 成功获取配置: {config['description']}")
        print(f"  模型类型: {config['model_type']}")
        print(f"  特征维度: {config['feature_dim']}")
    except Exception as e:
        print(f"✗ 获取配置失败: {e}")
        return
    
    # 2. 测试创建模型
    print("\n2. 创建模型...")
    try:
        model = create_multitask_model(
            model_type='multitask_gray',
            feature_dim=128,
            enable_background_filter=True,
            dropout_rate=0.2
        )
        print("✓ 成功创建模型")
        
        # 获取模型信息
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print(f"  模型名称: {info['model_name']}")
            print(f"  参数量: {info['total_parameters']:,}")
            print(f"  任务配置: {info['tasks']}")
    except Exception as e:
        print(f"✗ 创建模型失败: {e}")
        return
    
    # 3. 测试前向传播
    print("\n3. 测试前向传播...")
    try:
        model.eval()
        # 测试灰度输入
        dummy_input = torch.randn(2, 1, 70, 70)
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("✓ 前向传播成功")
        print(f"  输入形状: {dummy_input.shape}")
        
        # 检查输出
        expected_tasks = ['growth_level', 'growth_pattern', 'interference_mapping', 'fine_grained']
        for task in expected_tasks:
            if task in outputs:
                print(f"  {task}: {outputs[task].shape}")
            else:
                print(f"  ✗ 缺少任务输出: {task}")
        
        # 测试预测解析
        if hasattr(model, 'get_task_predictions'):
            predictions = model.get_task_predictions(outputs)
            print(f"\n  预测解析成功:")
            print(f"    生长级别: {predictions['growth_level']['class'][0]}")
            print(f"    精细分类: {predictions['fine_grained']['class'][0]}")
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 测试RGB输入自动转换
    print("\n4. 测试RGB输入自动转换...")
    try:
        rgb_input = torch.randn(2, 3, 70, 70)
        with torch.no_grad():
            rgb_outputs = model(rgb_input)
        print("✓ RGB输入自动转换成功")
        
    except Exception as e:
        print(f"✗ RGB输入测试失败: {e}")
    
    print("\n=== 集成测试完成 ===")


if __name__ == "__main__":
    test_integration()