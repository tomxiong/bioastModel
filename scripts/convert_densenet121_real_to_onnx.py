"""
DenseNet-121 真实数据训练模型 ONNX转换脚本
"""

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.densenet_wrapper import DenseNet121

def convert_densenet121_real_to_onnx():
    """将DenseNet-121真实数据训练模型转换为ONNX格式"""
    
    print("开始转换DenseNet-121真实数据模型到ONNX格式...")
    
    # 模型路径
    model_path = "experiments/experiment_20250804_184102/densenet121_real/best_model.pth"
    onnx_path = "onnx_models/densenet121_real.onnx"
    
    # 确保输出目录存在
    os.makedirs("onnx_models", exist_ok=True)
    
    # 创建模型
    model = DenseNet121(num_classes=2)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"已加载模型权重，验证准确率: {checkpoint['val_accuracy']:.2f}%")
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 70, 70)
    
    # 测试模型输出
    with torch.no_grad():
        output = model(dummy_input)
        print(f"模型输出形状: {output.shape}")
    
    # 转换为ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"ONNX模型已保存到: {onnx_path}")
    
    # 验证ONNX模型
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证通过!")
    except Exception as e:
        print(f"ONNX模型验证失败: {e}")
        return False
    
    # 测试ONNX推理精度
    print("测试ONNX推理精度...")
    
    # 创建ONNX Runtime会话
    ort_session = ort.InferenceSession(onnx_path)
    
    # 多次测试确保精度
    max_diff = 0.0
    test_cases = [
        torch.randn(1, 3, 70, 70),
        torch.zeros(1, 3, 70, 70),
        torch.ones(1, 3, 70, 70),
        torch.randn(1, 3, 70, 70) * 0.1
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        # PyTorch推理
        with torch.no_grad():
            pytorch_output = model(test_input).numpy()
        
        # ONNX推理
        onnx_output = ort_session.run(
            None, 
            {'input': test_input.numpy()}
        )[0]
        
        # 计算差异
        diff = np.abs(pytorch_output - onnx_output).max()
        max_diff = max(max_diff, diff)
        
        print(f"测试 {i}: 最大差异 = {diff:.8f}")
        if diff < 1e-4:
            print(f"  ✅ 测试 {i} 通过")
        else:
            print(f"  ❌ 测试 {i} 失败，差异过大")
    
    print(f"总体最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-4:
        print("✅ 修复版ONNX转换成功，精度验证通过!")
    else:
        print("⚠️ ONNX输出与PyTorch存在差异")
    
    # 输出模型信息
    print("\n模型信息:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数数量: {total_params:,}")
    print(f"  输入尺寸: {dummy_input.shape}")
    print(f"  输出尺寸: {output.shape}")
    print(f"  最佳验证准确率: {checkpoint['val_accuracy']:.2f}%")
    print(f"  数据类型: 真实数据")
    print(f"  训练样本: {checkpoint.get('config', {}).get('train_samples', 'N/A')}")
    
    return True

if __name__ == "__main__":
    convert_densenet121_real_to_onnx()