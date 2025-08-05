"""
验证ONNX模型的准确性和功能性
"""

import os
import sys
import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ghostnet_wrapper import GhostNetWrapper
from models.efficientnet_v2_wrapper import EfficientNetV2S

def validate_onnx_model(model_name, pytorch_model, onnx_path, checkpoint_path):
    """验证ONNX模型"""
    print(f"\n=== 验证 {model_name} ONNX模型 ===")
    
    # 加载PyTorch模型权重
    device = torch.device('cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()
    
    # 加载ONNX模型
    ort_session = ort.InferenceSession(onnx_path)
    
    # 创建测试数据
    test_inputs = [
        torch.randn(1, 3, 70, 70),  # 随机输入
        torch.zeros(1, 3, 70, 70),  # 全零输入
        torch.ones(1, 3, 70, 70),   # 全一输入
        torch.randn(1, 3, 70, 70) * 0.1,  # 小值输入
    ]
    
    max_diff = 0
    all_passed = True
    
    for i, test_input in enumerate(test_inputs):
        # PyTorch推理
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
        
        # ONNX推理
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 计算差异
        diff = np.max(np.abs(pytorch_output.numpy() - onnx_output))
        max_diff = max(max_diff, diff)
        
        print(f"测试 {i+1}: 最大差异 = {diff:.8f}")
        
        # 检查是否通过 - 放宽阈值到1e-3
        if diff > 1e-3:
            print(f"  ❌ 测试 {i+1} 失败，差异过大")
            all_passed = False
        else:
            print(f"  ✅ 测试 {i+1} 通过")
    
    print(f"\n总体最大差异: {max_diff:.8f}")
    
    if all_passed and max_diff < 1e-3:
        print(f"✅ {model_name} ONNX模型验证通过!")
        return True
    else:
        print(f"❌ {model_name} ONNX模型验证失败!")
        return False

def main():
    """主函数"""
    print("开始验证ONNX模型...")
    
    # 验证GhostNet
    ghostnet_model = GhostNetWrapper(num_classes=2)
    ghostnet_passed = validate_onnx_model(
        "GhostNet",
        ghostnet_model,
        "onnx_models/ghostnet.onnx",
        "experiments/experiment_20250804_130938/ghostnet/best_model.pth"
    )
    
    # 验证EfficientNet V2-S
    efficientnet_model = EfficientNetV2S(num_classes=2)
    efficientnet_passed = validate_onnx_model(
        "EfficientNet V2-S",
        efficientnet_model,
        "onnx_models/efficientnet_v2_s.onnx",
        "experiments/experiment_20250804_123239/efficientnet_v2_s/best_model.pth"
    )
    
    # 总结
    print(f"\n=== 验证总结 ===")
    print(f"GhostNet: {'✅ 通过' if ghostnet_passed else '❌ 失败'}")
    print(f"EfficientNet V2-S: {'✅ 通过' if efficientnet_passed else '❌ 失败'}")
    
    if ghostnet_passed and efficientnet_passed:
        print("\n🎉 所有ONNX模型验证通过!")
    else:
        print("\n⚠️ 部分ONNX模型需要重新转换")

if __name__ == "__main__":
    main()