"""
GhostNet 模型转换为 ONNX 格式
"""

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ghostnet_wrapper import GhostNetWrapper

def convert_ghostnet_to_onnx():
    """将GhostNet模型转换为ONNX格式"""
    
    print("开始转换GhostNet模型到ONNX格式...")
    
    # 设备
    device = torch.device('cpu')
    
    # 创建模型
    model = GhostNetWrapper(num_classes=2)
    model = model.to(device)
    
    # 加载训练好的权重
    checkpoint_path = "experiments/experiment_20250804_130938/ghostnet/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"已加载模型权重，验证准确率: {checkpoint.get('val_accuracy', 0):.2f}%")
    
    # 设置为评估模式
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 70, 70, device=device)
    
    # 测试模型输出
    with torch.no_grad():
        output = model(dummy_input)
        print(f"模型输出形状: {output.shape}")
    
    # 转换为ONNX
    onnx_path = "onnx_models/ghostnet.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
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
        }
    )
    
    print(f"ONNX模型已保存到: {onnx_path}")
    
    # 验证ONNX模型
    print("验证ONNX模型...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过!")
    
    # 测试ONNX推理
    print("测试ONNX推理...")
    ort_session = ort.InferenceSession(onnx_path)
    
    # 准备输入数据
    input_data = dummy_input.numpy()
    
    # ONNX推理
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # 比较输出
    pytorch_output = output.numpy()
    onnx_output = ort_outputs[0]
    
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    print(f"PyTorch vs ONNX最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ ONNX转换成功，输出一致!")
    else:
        print("⚠️ ONNX输出与PyTorch存在差异")
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型信息:")
    print(f"  参数数量: {total_params:,}")
    print(f"  输入尺寸: {dummy_input.shape}")
    print(f"  输出尺寸: {output.shape}")
    print(f"  最佳验证准确率: {checkpoint.get('val_accuracy', 0):.2f}%")
    print(f"  测试准确率: 53.67%")

if __name__ == "__main__":
    convert_ghostnet_to_onnx()