#!/usr/bin/env python3
"""
ONNX 推理速度基准测试
对比 PyTorch vs ONNX 推理性能
"""

import torch
import onnxruntime as ort
import numpy as np
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv4 import create_multilevel_mobilenetv4_small


def benchmark_pytorch(model, test_inputs, num_iterations=100, warmup=10):
    """PyTorch 推理基准测试"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(test_inputs)

    # 基准测试
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(test_inputs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def benchmark_onnx(onnx_path, test_inputs, num_iterations=100, warmup=10):
    """ONNX Runtime 推理基准测试"""
    ort_session = ort.InferenceSession(str(onnx_path))

    ort_inputs = {'input': test_inputs.numpy()}

    # Warmup
    for _ in range(warmup):
        _ = ort_session.run(None, ort_inputs)

    # 基准测试
    start_time = time.time()

    for _ in range(num_iterations):
        _ = ort_session.run(None, ort_inputs)

    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def main():
    print('='*80)
    print('ONNX 推理性能基准测试')
    print('='*80)

    # 1. 加载模型
    print('\n[1/4] 加载模型...')

    # PyTorch 模型
    pytorch_model = create_multilevel_mobilenetv4_small()
    checkpoint = torch.load(
        'experiments/multilevel_mobilenetv4_v0.11.0/best_model.pth',
        map_location='cpu',
        weights_only=False
    )
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()
    print('  ✓ PyTorch 模型加载完成')

    # ONNX 模型路径
    onnx_path = Path('deployment/onnx_models/mobilenetv4_v0.11.0/model.onnx')
    print(f'  ✓ ONNX 模型路径: {onnx_path}')

    # 2. 准备测试数据
    print('\n[2/4] 准备测试数据...')

    batch_sizes = [1, 4, 16, 32, 64]
    test_data = {}

    for bs in batch_sizes:
        test_data[bs] = torch.randn(bs, 1, 70, 70)
        print(f'  ✓ Batch size {bs}: {test_data[bs].shape}')

    # 3. PyTorch 基准测试
    print('\n[3/4] PyTorch 推理基准测试...')
    pytorch_results = {}

    for bs in batch_sizes:
        avg_time = benchmark_pytorch(pytorch_model, test_data[bs], num_iterations=100)
        pytorch_results[bs] = avg_time
        print(f'  Batch {bs:2d}: {avg_time*1000:.2f} ms/batch, {avg_time*1000/bs:.2f} ms/sample')

    # 4. ONNX Runtime 基准测试
    print('\n[4/4] ONNX Runtime 推理基准测试...')
    onnx_results = {}

    for bs in batch_sizes:
        avg_time = benchmark_onnx(onnx_path, test_data[bs], num_iterations=100)
        onnx_results[bs] = avg_time
        print(f'  Batch {bs:2d}: {avg_time*1000:.2f} ms/batch, {avg_time*1000/bs:.2f} ms/sample')

    # 5. 对比结果
    print('\n' + '='*80)
    print('性能对比总结')
    print('='*80)

    print(f"\n{'Batch':>6s} | {'PyTorch (ms)':>14s} | {'ONNX (ms)':>12s} | {'加速比':>8s} | {'提升':>8s}")
    print('-'*60)

    for bs in batch_sizes:
        pytorch_time = pytorch_results[bs] * 1000
        onnx_time = onnx_results[bs] * 1000
        speedup = pytorch_time / onnx_time
        improvement = (1 - onnx_time/pytorch_time) * 100

        print(f"{bs:6d} | {pytorch_time:14.2f} | {onnx_time:12.2f} | {speedup:8.2f}x | {improvement:+7.1f}%")

    # 平均加速比
    avg_speedup = sum(pytorch_results[bs] / onnx_results[bs] for bs in batch_sizes) / len(batch_sizes)
    avg_improvement = (1 - sum(onnx_results.values()) / sum(pytorch_results.values())) * 100

    print('-'*60)
    print(f"{'平均':>6s} | {'':>14s} | {'':>12s} | {avg_speedup:8.2f}x | {avg_improvement:+7.1f}%")

    # 模型大小对比
    print('\n模型大小对比:')
    pytorch_size = Path('experiments/multilevel_mobilenetv4_v0.11.0/best_model.pth').stat().st_size / (1024**2)
    onnx_size = onnx_path.stat().st_size / (1024**2)

    print(f'  PyTorch 模型: {pytorch_size:.2f} MB')
    print(f'  ONNX 模型:    {onnx_size:.2f} MB')
    print(f'  大小差异:     {((onnx_size - pytorch_size) / pytorch_size * 100):+.1f}%')

    # 推荐部署配置
    print('\n推荐部署配置:')

    # 找到性价比最高的 batch size (延迟 < 20ms/sample 且加速比最大)
    best_bs = None
    best_speedup = 0

    for bs in batch_sizes:
        per_sample_time = onnx_results[bs] * 1000 / bs
        speedup = pytorch_results[bs] / onnx_results[bs]

        if per_sample_time < 20 and speedup > best_speedup:
            best_speedup = speedup
            best_bs = bs

    if best_bs:
        print(f'  ✅ 推荐 Batch Size: {best_bs}')
        print(f'     - 单样本延迟: {onnx_results[best_bs]*1000/best_bs:.2f} ms')
        print(f'     - 加速比: {best_speedup:.2f}x')
        print(f'     - 吞吐量: {best_bs / onnx_results[best_bs]:.0f} samples/s')

    print('\n' + '='*80)
    print('✅ 基准测试完成!')


if __name__ == '__main__':
    main()
