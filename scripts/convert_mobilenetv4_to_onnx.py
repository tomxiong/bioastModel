#!/usr/bin/env python3
"""
Convert MobileNetV4 v1.1 model to ONNX format
将MobileNetV4 v1.1模型转换为ONNX格式以便部署
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv4 import create_multilevel_mobilenetv4_small


def convert_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape: tuple = (1, 1, 70, 70),
    opset_version: int = 14,
    simplify: bool = True
):
    """
    将PyTorch模型转换为ONNX格式

    Args:
        checkpoint_path: PyTorch模型权重路径
        output_path: 输出ONNX模型路径
        input_shape: 输入形状 (batch_size, channels, height, width)
        opset_version: ONNX opset版本
        simplify: 是否简化ONNX模型
    """
    print("=" * 80)
    print("MobileNetV4 v1.1 ONNX转换工具")
    print("=" * 80)

    # 创建模型
    print("\n1. 创建模型...")
    model = create_multilevel_mobilenetv4_small(
        input_channels=1,
        dropout_rate=0.3
    )

    # 加载权重
    print(f"2. 加载权重: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   - 训练轮数: {checkpoint.get('epoch', 'N/A')}")
        print(f"   - 验证准确率: {checkpoint.get('best_val_accuracy', 'N/A'):.4f}"
              if checkpoint.get('best_val_accuracy') else "")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 准备输入
    print(f"\n3. 准备示例输入: {input_shape}")
    dummy_input = torch.randn(*input_shape)

    # 测试模型输出
    print("4. 测试模型输出...")
    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"   输出结构:")
    for task_name, output in outputs.items():
        print(f"   - {task_name}: {output.shape}")

    # 转换为ONNX
    print(f"\n5. 转换为ONNX (opset version {opset_version})...")

    # 定义输入和输出名称
    input_names = ['input']
    output_names = ['growth_level', 'growth_pattern', 'interference_factors']

    # 定义动态轴（支持batch size变化）
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'growth_level': {0: 'batch_size'},
        'growth_pattern': {0: 'batch_size'},
        'interference_factors': {0: 'batch_size'}
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False
    )

    print(f"   ✓ ONNX模型已保存: {output_path}")

    # 验证ONNX模型
    print("\n6. 验证ONNX模型...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX模型验证通过")

        # 显示模型信息
        print(f"\n   模型信息:")
        print(f"   - IR版本: {onnx_model.ir_version}")
        print(f"   - Opset版本: {opset_version}")
        print(f"   - 输入: {input_names}")
        print(f"   - 输出: {output_names}")

    except ImportError:
        print("   ⚠ 警告: 未安装onnx库，跳过验证")
    except Exception as e:
        print(f"   ⚠ 验证出错: {e}")

    # 简化ONNX模型
    if simplify:
        print("\n7. 简化ONNX模型...")
        try:
            import onnxsim
            simplified_model, check = onnxsim.simplify(output_path)
            if check:
                onnx.save(simplified_model, output_path)
                print("   ✓ ONNX模型简化完成")
            else:
                print("   ⚠ 简化失败，保留原始模型")
        except ImportError:
            print("   ⚠ 未安装onnx-simplifier，跳过简化")
        except Exception as e:
            print(f"   ⚠ 简化出错: {e}")

    # 测试ONNX推理
    print("\n8. 测试ONNX推理...")
    try:
        import onnxruntime as ort

        # 创建推理会话
        session = ort.InferenceSession(output_path)

        # 准备输入
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}

        # 运行推理
        ort_outputs = session.run(None, ort_inputs)

        print("   ✓ ONNX推理测试通过")
        print(f"\n   ONNX输出:")
        for i, output_name in enumerate(output_names):
            print(f"   - {output_name}: {ort_outputs[i].shape}")

        # 对比PyTorch和ONNX输出
        print("\n9. 对比PyTorch和ONNX输出...")
        with torch.no_grad():
            torch_outputs = model(dummy_input)

        max_diff = 0
        for i, task_name in enumerate(output_names):
            torch_out = torch_outputs[task_name].numpy()
            onnx_out = ort_outputs[i]
            diff = abs(torch_out - onnx_out).max()
            max_diff = max(max_diff, diff)
            print(f"   - {task_name}: 最大差异 = {diff:.6f}")

        if max_diff < 1e-4:
            print(f"\n   ✓ 转换精度优秀 (最大差异 < 1e-4)")
        elif max_diff < 1e-3:
            print(f"\n   ✓ 转换精度良好 (最大差异 < 1e-3)")
        else:
            print(f"\n   ⚠ 转换精度一般 (最大差异 = {max_diff:.6f})")

    except ImportError:
        print("   ⚠ 未安装onnxruntime，跳过推理测试")
    except Exception as e:
        print(f"   ⚠ 推理测试出错: {e}")

    # 获取文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"\n" + "=" * 80)
    print(f"转换完成！")
    print(f"ONNX模型: {output_path}")
    print(f"文件大小: {file_size:.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Convert MobileNetV4 to ONNX')
    parser.add_argument('--checkpoint', type=str,
                       default='experiments/mobilenetv4_v1.1/best_model.pth',
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str,
                       default='deployment/onnx_models/mobilenetv4_v1.1.onnx',
                       help='Output ONNX model path')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for dummy input')
    parser.add_argument('--opset_version', type=int, default=14,
                       help='ONNX opset version')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Do not simplify ONNX model')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 转换
    convert_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=(args.batch_size, 1, 70, 70),
        opset_version=args.opset_version,
        simplify=not args.no_simplify
    )


if __name__ == '__main__':
    main()
