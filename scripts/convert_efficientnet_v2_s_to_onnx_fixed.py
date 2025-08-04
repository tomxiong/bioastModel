"""
修复版EfficientNet V2-S ONNX转换脚本
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

from models.efficientnet_v2_wrapper import EfficientNetV2S

def convert_efficientnet_v2_s_to_onnx_fixed():
    """修复版EfficientNet V2-S ONNX转换"""
    
    print("开始修复版EfficientNet V2-S ONNX转换...")
    
    # 设备
    device = torch.device('cpu')
    
    # 创建模型
    model = EfficientNetV2S(num_classes=2)
    model = model.to(device)
    
    # 加载训练好的权重
    checkpoint_path = "experiments/experiment_20250804_123239/efficientnet_v2_s/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"已加载模型权重，验证准确率: {checkpoint.get('val_accuracy', checkpoint.get('val_acc', 0)):.2f}%")
    
    # 设置为评估模式
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 70, 70, device=device)
    
    # 测试模型输出
    with torch.no_grad():
        output = model(dummy_input)
        print(f"模型输出形状: {output.shape}")
    
    # 转换为ONNX - 使用更严格的参数
    onnx_path = "onnx_models/efficientnet_v2_s_fixed.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 使用更高精度的转换设置
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
        # 添加更严格的转换选项
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        keep_initializers_as_inputs=False
    )
    
    print(f"ONNX模型已保存到: {onnx_path}")
    
    # 验证ONNX模型
    print("验证ONNX模型...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过!")
    
    # 多次测试ONNX推理精度
    print("测试ONNX推理精度...")
    ort_session = ort.InferenceSession(onnx_path)
    
    test_cases = [
        torch.randn(1, 3, 70, 70),
        torch.zeros(1, 3, 70, 70),
        torch.ones(1, 3, 70, 70),
        torch.randn(1, 3, 70, 70) * 0.1
    ]
    
    max_diff = 0
    all_passed = True
    
    for i, test_input in enumerate(test_cases):
        # PyTorch推理
        with torch.no_grad():
            pytorch_output = model(test_input)
        
        # ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 计算差异
        diff = np.max(np.abs(pytorch_output.numpy() - onnx_output))
        max_diff = max(max_diff, diff)
        
        print(f"测试 {i+1}: 最大差异 = {diff:.8f}")
        
        if diff > 1e-4:
            all_passed = False
    
    print(f"总体最大差异: {max_diff:.8f}")
    
    if all_passed and max_diff < 1e-4:
        print("✅ 修复版ONNX转换成功，精度验证通过!")
        
        # 替换原始文件
        import shutil
        shutil.move(onnx_path, "onnx_models/efficientnet_v2_s.onnx")
        print("已替换原始ONNX文件")
        
    else:
        print("⚠️ 修复版ONNX转换仍有精度问题")
    
    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型信息:")
    print(f"  参数数量: {total_params:,}")
    print(f"  输入尺寸: {dummy_input.shape}")
    print(f"  输出尺寸: {output.shape}")
    print(f"  最佳验证准确率: {checkpoint.get('val_accuracy', checkpoint.get('val_acc', 0)):.2f}%")

if __name__ == "__main__":
    convert_efficientnet_v2_s_to_onnx_fixed()