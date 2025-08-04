"""
EfficientNet V2-S 模型转换为ONNX格式
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.efficientnet_v2_wrapper import EfficientNetV2S

def convert_efficientnet_v2_s_to_onnx():
    """将EfficientNet V2-S模型转换为ONNX格式"""
    
    # 模型路径
    model_path = "experiments/experiment_20250804_123239/efficientnet_v2_s/best_model.pth"
    onnx_path = "onnx_models/efficientnet_v2_s.onnx"
    
    print("开始转换EfficientNet V2-S模型到ONNX格式...")
    
    # 创建模型
    model = EfficientNetV2S(num_classes=2)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    val_acc = checkpoint.get('val_acc', checkpoint.get('val_accuracy', 0))
    print(f"已加载模型权重，验证准确率: {val_acc:.2f}%")
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 70, 70)
    
    # 测试模型前向传播
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
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    
    # 运行推理
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # 比较PyTorch和ONNX输出
    torch_output = output.numpy()
    onnx_output = ort_outputs[0]
    
    max_diff = np.max(np.abs(torch_output - onnx_output))
    print(f"PyTorch vs ONNX最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ ONNX转换成功，输出一致!")
    else:
        print("⚠️ ONNX输出与PyTorch有较大差异")
    
    # 模型信息
    print(f"\n模型信息:")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  输入尺寸: (1, 3, 70, 70)")
    print(f"  输出尺寸: {output.shape}")
    print(f"  最佳验证准确率: {val_acc:.2f}%")
    
    return onnx_path

if __name__ == "__main__":
    convert_efficientnet_v2_s_to_onnx()