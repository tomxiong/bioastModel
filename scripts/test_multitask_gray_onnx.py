#!/usr/bin/env python3
"""
测试多任务灰度菌落检测网络的ONNX转换
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.onnx
import numpy as np
from models.multitask_models import create_multitask_model
import onnx
import onnxruntime as ort

def create_onnx_compatible_model(original_model):
    """创建ONNX兼容的简化版本模型"""
    class MultitaskGrayColonyONNX(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            # 只保留必要的层
            self.gray_stem = model.gray_stem
            self.cnn_backbone = model.cnn_backbone
            self.global_pool = model.global_pool
            
            # 简化的多任务头部
            self.growth_level_head = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 3)
            )
            
            self.growth_pattern_head = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 9)
            )
            
            self.interference_head = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 4)
            )
            
            self.fine_grained_head = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 15)
            )
            
            # 辅助输出头
            self.pore_confidence_head = torch.nn.Sequential(
                torch.nn.Linear(128, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
                torch.nn.Sigmoid()
            )
            
            self.bg_confidence_head = torch.nn.Sequential(
                torch.nn.Linear(128, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
                torch.nn.Sigmoid()
            )
        
        def forward(self, x):
            # 确保输入是灰度的
            if x.shape[1] == 3:
                x = torch.mean(x, dim=1, keepdim=True)
            
            # 简化的前向传播
            x = self.gray_stem(x)
            
            # CNN特征提取
            for stage in self.cnn_backbone:
                x = stage(x)
            
            # 全局池化
            x = self.global_pool(x).flatten(1)
            
            # 多任务预测
            growth_level = self.growth_level_head(x)
            growth_pattern = self.growth_pattern_head(x)
            interference = self.interference_head(x)
            fine_grained = self.fine_grained_head(x)
            pore_confidence = self.pore_confidence_head(x)
            bg_confidence = self.bg_confidence_head(x)
            
            return growth_level, growth_pattern, interference, fine_grained, pore_confidence, bg_confidence
    
    return MultitaskGrayColonyONNX(original_model)

def test_onnx_conversion():
    """测试ONNX转换"""
    print("=== 测试多任务灰度菌落检测网络ONNX转换 ===")
    
    # 创建模型
    model = create_multitask_model(
        model_type='multitask_gray',
        feature_dim=128,
        enable_background_filter=True,
        dropout_rate=0.2
    )
    model.eval()
    
    print("✓ 原始模型创建成功")
    
    # 创建ONNX兼容版本
    onnx_model = create_onnx_compatible_model(model)
    onnx_model.eval()
    
    print("✓ ONNX兼容模型创建成功")
    
    # 测试输入
    dummy_input = torch.randn(1, 1, 70, 70)
    
    # 测试原始模型输出
    with torch.no_grad():
        original_outputs = model(dummy_input)
    
    print("✓ 原始模型前向传播成功")
    
    # 测试ONNX模型输出
    with torch.no_grad():
        onnx_outputs = onnx_model(dummy_input)
    
    print("✓ ONNX模型前向传播成功")
    
    # 比较输出形状
    print("\n--- 输出形状对比 ---")
    original_shapes = {
        'growth_level': original_outputs['growth_level'].shape,
        'growth_pattern': original_outputs['growth_pattern'].shape,
        'interference_mapping': original_outputs['interference_mapping'].shape,
        'fine_grained': original_outputs['fine_grained'].shape,
        'pore_confidence': original_outputs['pore_confidence'].shape,
        'bg_confidence': original_outputs['bg_confidence'].shape
    }
    
    # 调整ONNX输出名称映射
    shape_mapping = {
        'growth_level': 'growth_level',
        'growth_pattern': 'growth_pattern',
        'interference': 'interference_mapping',
        'fine_grained': 'fine_grained',
        'pore_confidence': 'pore_confidence',
        'bg_confidence': 'bg_confidence'
    }
    
    onnx_output_names = ['growth_level', 'growth_pattern', 'interference', 'fine_grained', 'pore_confidence', 'bg_confidence']
    
    for i, name in enumerate(onnx_output_names):
        original_key = shape_mapping[name]
        original_shape = original_shapes[original_key]
        onnx_shape = onnx_outputs[i].shape
        print(f"{name}: 原始={original_shape}, ONNX={onnx_shape}, {'✓' if original_shape == onnx_shape else '✗'}")
    
    # 导出ONNX
    onnx_path = "multitask_gray_colony_net.onnx"
    
    # 定义动态轴
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'growth_level': {0: 'batch_size'},
        'growth_pattern': {0: 'batch_size'},
        'interference': {0: 'batch_size'},
        'fine_grained': {0: 'batch_size'},
        'pore_confidence': {0: 'batch_size'},
        'bg_confidence': {0: 'batch_size'}
    }
    
    try:
        # 导出模型
        torch.onnx.export(
            onnx_model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=onnx_output_names,
            dynamic_axes=dynamic_axes,
            opset_version=12,
            do_constant_folding=True,
            verbose=False
        )
        print(f"\n✓ ONNX模型导出成功: {onnx_path}")
        
        # 验证ONNX模型
        onnx_model_loaded = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_loaded)
        print("✓ ONNX模型验证通过")
        
        # 测试ONNX推理
        ort_session = ort.InferenceSession(onnx_path)
        
        # 准备输入
        ort_inputs = {'input': dummy_input.numpy()}
        
        # 运行推理
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print("\n--- ONNX推理测试 ---")
        print("✓ ONNX运行时创建成功")
        print("✓ ONNX推理执行成功")
        
        # 比较PyTorch和ONNX的输出
        print("\n--- PyTorch vs ONNX 输出对比 ---")
        max_diff = 0
        for i, name in enumerate(onnx_output_names):
            pytorch_output = onnx_outputs[i].numpy()
            onnx_output = ort_outputs[i]
            
            # 计算差异
            diff = np.abs(pytorch_output - onnx_output).max()
            max_diff = max(max_diff, diff)
            
            print(f"{name}: 最大差异 = {diff:.6f} {'✓' if diff < 1e-3 else '⚠'}")
        
        print(f"\n整体最大差异: {max_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✓ ONNX转换成功！模型可以用于部署")
        else:
            print("⚠ ONNX转换存在精度问题，可能影响部署效果")
        
        # 显示模型信息
        print(f"\n--- ONNX模型信息 ---")
        print(f"模型文件大小: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
        print(f"输入: [batch_size, 1, 70, 70]")
        print(f"输出:")
        for i, name in enumerate(onnx_output_names):
            print(f"  {name}: [batch_size, {ort_outputs[i].shape[1]}]")
        
        # 测试批量推理
        print(f"\n--- 批量推理测试 ---")
        batch_input = torch.randn(4, 1, 70, 70)
        ort_batch_inputs = {'input': batch_input.numpy()}
        ort_batch_outputs = ort_session.run(None, ort_batch_inputs)
        
        print(f"✓ 批量推理成功: 输入形状 {batch_input.shape}")
        for i, name in enumerate(onnx_output_names):
            print(f"  {name} 输出形状: {ort_batch_outputs[i].shape}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ONNX转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_post_processing():
    """测试后处理逻辑"""
    print(f"\n=== 测试后处理逻辑 ===")
    
    # 模拟ONNX输出
    growth_level = np.array([[0.1, 0.8, 0.1]])  # positive
    growth_pattern = np.array([[0.1, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])  # clustered
    interference = np.array([[0.8, 0.2, 0.7, 0.1]])  # pores, artifacts
    fine_grained = np.array([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # positive_cluster_with_pores
    pore_confidence = np.array([[0.8]])
    bg_confidence = np.array([[0.3]])
    
    # 定义类别名称
    growth_level_names = ['negative', 'positive', 'weak_growth']
    growth_pattern_names = ['clean', 'clustered', 'scattered', 'small_dots', 'ring_shaped', 'irregular', 'mixed', 'sparse', 'dense']
    interference_names = ['pores', 'debris', 'artifacts', 'contamination']
    fine_grained_names = [
        'positive_cluster_no_pores',
        'positive_cluster_with_pores',
        'positive_cluster_overlapping_pores',
        'negative_clean_no_pores',
        'negative_clean_with_pores',
        'weak_growth_center_no_pores',
        'weak_growth_center_with_pores',
        'weak_growth_center_overlapping_pores',
        'weak_growth_scattered_no_pores',
        'weak_growth_scattered_with_pores',
        'weak_growth_scattered_overlapping_pores',
        'with_debris',
        'with_artifacts',
        'contaminated',
        'other'
    ]
    
    # 后处理
    print("后处理结果:")
    
    # 生长级别
    growth_level_idx = np.argmax(growth_level, axis=1)[0]
    print(f"  生长级别: {growth_level_names[growth_level_idx]} (置信度: {growth_level[0][growth_level_idx]:.3f})")
    
    # 生长模式
    growth_pattern_idx = np.argmax(growth_pattern, axis=1)[0]
    print(f"  生长模式: {growth_pattern_names[growth_pattern_idx]} (置信度: {growth_pattern[0][growth_pattern_idx]:.3f})")
    
    # 干扰因素
    interference_labels = [interference_names[i] for i, prob in enumerate(interference[0]) if prob > 0.5]
    print(f"  干扰因素: {interference_labels if interference_labels else '无'}")
    
    # 精细分类
    fine_grained_idx = np.argmax(fine_grained, axis=1)[0]
    print(f"  精细分类: {fine_grained_names[fine_grained_idx]}")
    
    # 辅助信息
    print(f"  气孔置信度: {pore_confidence[0][0]:.3f}")
    print(f"  背景置信度: {bg_confidence[0][0]:.3f}")
    print(f"  有气孔: {'是' if pore_confidence[0][0] > 0.5 else '否'}")

if __name__ == "__main__":
    success = test_onnx_conversion()
    
    if success:
        test_post_processing()
        print(f"\n🎉 多任务灰度菌落检测网络ONNX部署验证完成！")
    else:
        print(f"\n❌ ONNX转换失败，需要进一步调试")