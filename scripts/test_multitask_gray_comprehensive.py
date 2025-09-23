#!/usr/bin/env python3
"""
多任务灰度菌落检测网络完整测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from models.multitask_models import create_multitask_model, get_multitask_model_config

def test_comprehensive():
    """全面测试模型功能"""
    print("=== 多任务灰度菌落检测网络完整测试 ===")
    
    # 创建模型
    model = create_multitask_model(
        model_type='multitask_gray',
        feature_dim=128,
        enable_background_filter=True,
        dropout_rate=0.2
    )
    
    print(f"\n模型信息:")
    model_info = model.get_model_info()
    print(f"  名称: {model_info['model_name']}")
    print(f"  参数量: {model_info['total_parameters']:,}")
    print(f"  输入尺寸: {model_info['input_size']}")
    print(f"  架构: {model_info['architecture']}")
    
    # 测试不同类型的输入
    test_cases = [
        ("灰度输入", torch.randn(4, 1, 70, 70)),
        ("RGB输入", torch.randn(4, 3, 70, 70)),
        ("批量测试", torch.randn(8, 1, 70, 70))
    ]
    
    for case_name, input_tensor in test_cases:
        print(f"\n--- {case_name} ---")
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_tensor)
        
        print(f"输入形状: {input_tensor.shape}")
        print(f"输出任务:")
        for task_name, output in outputs.items():
            if task_name in ['growth_level', 'growth_pattern', 'interference_mapping', 'fine_grained', 'fine_grained_refined']:
                print(f"  {task_name}: {output.shape}")
        
        # 测试预测解析
        predictions = model.get_task_predictions(outputs)
        
        # 显示第一个样本的预测结果
        print(f"\n样本1预测结果:")
        print(f"  生长级别: {predictions['growth_level']['class'][0]}")
        print(f"  生长模式: {predictions['growth_pattern']['class'][0]}")
        print(f"  干扰因素: {predictions['interference_mapping']['labels'][0]}")
        print(f"  精细分类: {predictions['fine_grained']['class'][0]}")
        print(f"  气孔置信度: {predictions['auxiliary']['pore_confidence'][0].item():.3f}")
        print(f"  背景置信度: {predictions['auxiliary']['bg_confidence'][0].item():.3f}")
        
        # 检查是否有中空结构检测
        if 'hollow_score' in predictions['auxiliary']:
            hollow_score = predictions['auxiliary']['hollow_score'][0]
            if hollow_score.numel() == 1:
                print(f"  中空结构得分: {hollow_score.item():.3f}")
            else:
                print(f"  中空结构得分: {hollow_score.shape} (空间特征图)")
    
    # 测试任务配置验证
    print(f"\n--- 任务配置验证 ---")
    expected_tasks = {
        'growth_level': 3,
        'growth_pattern': 9,
        'interference_mapping': 4,
        'fine_grained': 15
    }
    
    for task_name, expected_classes in expected_tasks.items():
        actual_shape = outputs[task_name].shape
        assert actual_shape[1] == expected_classes, f"{task_name} 输出维度不匹配"
        print(f"✓ {task_name}: {actual_shape[1]} 类")
    
    # 测试多标签任务
    print(f"\n--- 多标签任务测试 ---")
    interference_probs = outputs['interference_mapping']
    interference_pred = (interference_probs > 0.5).cpu().numpy()
    
    print(f"干扰因素检测示例 (前2个样本):")
    for i in range(2):
        labels = predictions['interference_mapping']['labels'][i]
        print(f"  样本{i+1}: {labels if labels else '无干扰'}")
    
    # 测试精细分类的组合逻辑
    print(f"\n--- 精细分类组合验证 ---")
    fine_grained_names = predictions['fine_grained']['class']
    
    # 统计各类别数量
    from collections import Counter
    class_counts = Counter(fine_grained_names)
    print(f"精细分类分布:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # 验证特征融合
    print(f"\n--- 特征融合验证 ---")
    original_fg = outputs['fine_grained']
    refined_fg = outputs['fine_grained_refined']
    
    print(f"原始精细分类输出: {original_fg.shape}")
    print(f"融合后精细分类输出: {refined_fg.shape}")
    print(f"融合后置信度提升: {(refined_fg.std(dim=1) > original_fg.std(dim=1)).sum().item()} / {len(refined_fg)} 样本")
    
    # 测试辅助输出
    print(f"\n--- 辅助输出测试 ---")
    auxiliary_outputs = {
        'pore_confidence': outputs['pore_confidence'],
        'bg_confidence': outputs['bg_confidence']
    }
    
    if 'bg_strength' in outputs:
        auxiliary_outputs['bg_strength'] = outputs['bg_strength']
    
    for name, output in auxiliary_outputs.items():
        print(f"  {name}: {output.shape}, 均值={output.mean().item():.3f}, 标准差={output.std().item():.3f}")
    
    # 测试背景注意力机制
    if outputs['background_attention'] is not None:
        bg_attn = outputs['background_attention']
        print(f"\n背景注意力图: {bg_attn.shape}")
        print(f"  注意力强度范围: [{bg_attn.min().item():.3f}, {bg_attn.max().item():.3f}]")
    
    # 测试中空结构检测
    hollow_info = outputs['hollow_detection']
    print(f"\n中空结构检测:")
    for key, value in hollow_info.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, 均值={value.mean().item():.3f}")
    
    print(f"\n✓ 所有测试通过！多任务灰度菌落检测网络功能完整。")

if __name__ == "__main__":
    test_comprehensive()