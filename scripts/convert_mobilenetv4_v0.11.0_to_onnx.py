#!/usr/bin/env python3
"""
转换 MobileNetV4 v0.11.0 为 ONNX 格式
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv4 import create_multilevel_mobilenetv4_small


def convert_to_onnx():
    print('='*80)
    print('MobileNetV4 v0.11.0 ONNX 转换')
    print('='*80)

    # 创建输出目录
    output_dir = Path('deployment/onnx_models/mobilenetv4_v0.11.0')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 PyTorch 模型
    print('\n[1/5] 加载 PyTorch 模型...')
    model = create_multilevel_mobilenetv4_small(input_channels=1, dropout_rate=0.3)

    checkpoint_path = 'experiments/multilevel_mobilenetv4_v0.11.0/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f'  ✓ 模型加载成功')
    print(f'  ✓ 参数量: {num_params:,} ({num_params/1e6:.2f}M)')

    # 2. 准备示例输入
    print('\n[2/5] 准备示例输入...')
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, 70, 70)
    print(f'  ✓ 输入形状: {dummy_input.shape}')

    # 3. 执行 PyTorch 推理 (作为参考)
    print('\n[3/5] 执行 PyTorch 推理...')
    with torch.no_grad():
        pytorch_outputs = model(dummy_input)

    print(f'  ✓ PyTorch 输出:')
    for task, output in pytorch_outputs.items():
        print(f'    - {task}: {output.shape}')

    # 4. 转换为 ONNX
    print('\n[4/5] 转换为 ONNX 格式...')
    onnx_path = output_dir / 'model.onnx'

    # 定义输入输出名称
    input_names = ['input']
    output_names = ['growth_level', 'growth_pattern', 'interference_factors']

    # 动态轴配置
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'growth_level': {0: 'batch_size'},
        'growth_pattern': {0: 'batch_size'},
        'interference_factors': {0: 'batch_size'}
    }

    # 转换
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )

    print(f'  ✓ ONNX 模型已保存: {onnx_path}')

    # 验证 ONNX 模型
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f'  ✓ ONNX 模型验证通过')

    # 5. 验证 ONNX 推理
    print('\n[5/5] 验证 ONNX 推理精度...')

    # 创建 ONNX Runtime 会话
    ort_session = ort.InferenceSession(str(onnx_path))

    # 准备输入
    ort_inputs = {
        'input': dummy_input.numpy()
    }

    # 执行推理
    ort_outputs = ort_session.run(None, ort_inputs)

    # 对比精度
    print(f'\n  精度验证:')
    max_diff_list = []

    for i, task in enumerate(output_names):
        pytorch_output = pytorch_outputs[task].detach().numpy()
        onnx_output = ort_outputs[i]

        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

        max_diff_list.append(max_diff)

        print(f'    {task}:')
        print(f'      最大差异: {max_diff:.6e}')
        print(f'      平均差异: {mean_diff:.6e}')

    overall_max_diff = max(max_diff_list)

    if overall_max_diff < 1e-5:
        print(f'\n  ✅ 精度验证通过! (最大差异: {overall_max_diff:.6e} < 1e-5)')
    elif overall_max_diff < 1e-3:
        print(f'\n  ⚠️ 精度可接受 (最大差异: {overall_max_diff:.6e} < 1e-3)')
    else:
        print(f'\n  ❌ 精度差异较大! (最大差异: {overall_max_diff:.6e})')

    # 保存模型信息
    model_info = {
        'model_name': 'MobileNetV4 v0.11.0',
        'architecture': 'Universal Inverted Bottleneck + SE/ECA',
        'version': 'v0.11.0',
        'parameters': int(num_params),
        'input_shape': [1, 70, 70],
        'tasks': {
            'growth_level': {'num_classes': 2, 'type': 'classification'},
            'growth_pattern': {'num_classes': 10, 'type': 'classification'},
            'interference_factors': {'num_classes': 4, 'type': 'multilabel'}
        },
        'opset_version': 14,
        'max_precision_diff': float(overall_max_diff),
        'onnx_path': str(onnx_path.absolute())
    }

    info_path = output_dir / 'model_info.json'
    import json
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f'\n  ✓ 模型信息已保存: {info_path}')

    # 获取 ONNX 模型大小
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)

    print('\n' + '='*80)
    print('转换完成总结')
    print('='*80)
    print(f'\n模型信息:')
    print(f'  名称: MobileNetV4 v0.11.0')
    print(f'  参数量: {num_params:,} ({num_params/1e6:.2f}M)')
    print(f'  ONNX 大小: {onnx_size_mb:.2f} MB')
    print(f'\n输入:')
    print(f'  名称: input')
    print(f'  形状: [batch_size, 1, 70, 70]')
    print(f'  类型: float32')
    print(f'\n输出:')
    print(f'  1. growth_level: [batch_size, 2]')
    print(f'  2. growth_pattern: [batch_size, 10]')
    print(f'  3. interference_factors: [batch_size, 4]')
    print(f'\n精度:')
    print(f'  最大差异: {overall_max_diff:.6e}')
    print(f'  状态: {"✅ 通过" if overall_max_diff < 1e-5 else "⚠️ 可接受" if overall_max_diff < 1e-3 else "❌ 需检查"}')
    print(f'\n文件位置:')
    print(f'  ONNX 模型: {onnx_path}')
    print(f'  模型信息: {info_path}')
    print('='*80)
    print('\n✅ ONNX 转换成功完成!')

    return str(onnx_path), model_info


if __name__ == '__main__':
    convert_to_onnx()
