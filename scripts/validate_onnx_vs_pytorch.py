#!/usr/bin/env python3
"""
ONNX vs PyTorch Model Validation Script
对比ONNX模型和PyTorch模型在验证集上的性能
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import onnxruntime as ort
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv4 import create_multilevel_mobilenetv4_small
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset


class ONNXValidator:
    """ONNX模型验证器"""

    def __init__(self, onnx_path: str, use_gpu: bool = True):
        """
        初始化ONNX验证器

        Args:
            onnx_path: ONNX模型路径
            use_gpu: 是否使用GPU
        """
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"ONNX Runtime initialized")
        print(f"  Providers: {self.session.get_providers()}")
        print(f"  Input: {self.input_name}")
        print(f"  Outputs: {self.output_names}")

    def predict(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """
        ONNX模型预测

        Args:
            images: 输入图像 [batch_size, 1, 70, 70]

        Returns:
            预测结果字典
        """
        outputs = self.session.run(self.output_names, {self.input_name: images})

        return {
            'growth_level': outputs[0],
            'growth_pattern': outputs[1],
            'interference_factors': outputs[2]
        }


class PyTorchValidator:
    """PyTorch模型验证器"""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        初始化PyTorch验证器

        Args:
            checkpoint_path: PyTorch检查点路径
            device: 计算设备
        """
        self.device = device

        # 创建模型
        self.model = create_multilevel_mobilenetv4_small(
            input_channels=1,
            dropout_rate=0.3
        )

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        print(f"PyTorch model loaded from {checkpoint_path}")
        print(f"  Device: {device}")

        # 尝试获取训练时的验证集指标
        if 'val_metrics' in checkpoint:
            print(f"  Training validation metrics:")
            metrics = checkpoint['val_metrics']
            if isinstance(metrics, dict):
                print(f"    Growth Level Acc: {metrics.get('growth_level_acc', 'N/A'):.2%}")
                print(f"    Growth Pattern Acc: {metrics.get('growth_pattern_acc', 'N/A'):.2%}")
                print(f"    Interference F1: {metrics.get('interference_f1', 'N/A'):.4f}")

    def predict(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        PyTorch模型预测

        Args:
            images: 输入图像 [batch_size, 1, 70, 70]

        Returns:
            预测结果字典
        """
        with torch.no_grad():
            outputs = self.model(images)

        return {
            'growth_level': outputs['growth_level'],
            'growth_pattern': outputs['growth_pattern'],
            'interference_factors': outputs['interference_factors']
        }


def compute_metrics(predictions: Dict[str, np.ndarray],
                   labels: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    计算评估指标

    Args:
        predictions: 预测结果
        labels: 真实标签

    Returns:
        指标字典
    """
    metrics = {}

    # Growth Level准确率
    growth_level_pred = np.argmax(predictions['growth_level'], axis=1)
    growth_level_true = labels['growth_level']
    metrics['growth_level_acc'] = np.mean(growth_level_pred == growth_level_true)

    # Growth Pattern准确率
    growth_pattern_pred = np.argmax(predictions['growth_pattern'], axis=1)
    growth_pattern_true = labels['growth_pattern']
    metrics['growth_pattern_acc'] = np.mean(growth_pattern_pred == growth_pattern_true)

    # Interference Factors F1 (多标签)
    interference_pred = (predictions['interference_factors'] > 0.5).astype(int)
    interference_true = labels['interference_factors']

    # 计算每个类别的F1
    f1_scores = []
    for i in range(interference_true.shape[1]):
        pred_i = interference_pred[:, i]
        true_i = interference_true[:, i]

        tp = np.sum((pred_i == 1) & (true_i == 1))
        fp = np.sum((pred_i == 1) & (true_i == 0))
        fn = np.sum((pred_i == 0) & (true_i == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)

    metrics['interference_f1'] = np.mean(f1_scores)
    metrics['interference_f1_per_class'] = f1_scores

    # 综合准确率
    metrics['overall_acc'] = (
        metrics['growth_level_acc'] * 0.4 +
        metrics['growth_pattern_acc'] * 0.4 +
        metrics['interference_f1'] * 0.2
    )

    return metrics


def validate_onnx(onnx_validator: ONNXValidator,
                  data_loader: DataLoader,
                  device: str = 'cuda') -> Tuple[Dict[str, float], float]:
    """
    验证ONNX模型

    Args:
        onnx_validator: ONNX验证器
        data_loader: 数据加载器
        device: 设备

    Returns:
        (指标字典, 推理时间)
    """
    all_predictions = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': []
    }
    all_labels = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': []
    }

    total_time = 0.0

    print("\n=== Validating ONNX Model ===")
    for images, targets in tqdm(data_loader, desc="ONNX Inference"):
        # 转换为numpy
        images_np = images.cpu().numpy()

        # 推理
        start_time = time.time()
        predictions = onnx_validator.predict(images_np)
        total_time += time.time() - start_time

        # 收集预测结果
        all_predictions['growth_level'].append(predictions['growth_level'])
        all_predictions['growth_pattern'].append(predictions['growth_pattern'])
        all_predictions['interference_factors'].append(predictions['interference_factors'])

        # 收集标签
        all_labels['growth_level'].append(targets['growth_level'].cpu().numpy())
        all_labels['growth_pattern'].append(targets['growth_pattern'].cpu().numpy())
        all_labels['interference_factors'].append(targets['interference_factors'].cpu().numpy())

    # 合并所有batch
    all_predictions = {
        k: np.concatenate(v, axis=0) for k, v in all_predictions.items()
    }
    all_labels = {
        k: np.concatenate(v, axis=0) for k, v in all_labels.items()
    }

    # 计算指标
    metrics = compute_metrics(all_predictions, all_labels)

    return metrics, total_time


def validate_pytorch(pytorch_validator: PyTorchValidator,
                    data_loader: DataLoader,
                    device: str = 'cuda') -> Tuple[Dict[str, float], float]:
    """
    验证PyTorch模型

    Args:
        pytorch_validator: PyTorch验证器
        data_loader: 数据加载器
        device: 设备

    Returns:
        (指标字典, 推理时间)
    """
    all_predictions = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': []
    }
    all_labels = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': []
    }

    total_time = 0.0

    print("\n=== Validating PyTorch Model ===")
    for images, targets in tqdm(data_loader, desc="PyTorch Inference"):
        images = images.to(device)

        # 推理
        start_time = time.time()
        predictions = pytorch_validator.predict(images)
        torch.cuda.synchronize() if device == 'cuda' else None
        total_time += time.time() - start_time

        # 收集预测结果 (转为numpy)
        all_predictions['growth_level'].append(predictions['growth_level'].cpu().numpy())
        all_predictions['growth_pattern'].append(predictions['growth_pattern'].cpu().numpy())
        all_predictions['interference_factors'].append(predictions['interference_factors'].cpu().numpy())

        # 收集标签
        all_labels['growth_level'].append(targets['growth_level'].cpu().numpy())
        all_labels['growth_pattern'].append(targets['growth_pattern'].cpu().numpy())
        all_labels['interference_factors'].append(targets['interference_factors'].cpu().numpy())

    # 合并所有batch
    all_predictions = {
        k: np.concatenate(v, axis=0) for k, v in all_predictions.items()
    }
    all_labels = {
        k: np.concatenate(v, axis=0) for k, v in all_labels.items()
    }

    # 计算指标
    metrics = compute_metrics(all_predictions, all_labels)

    return metrics, total_time


def compare_outputs(onnx_validator: ONNXValidator,
                   pytorch_validator: PyTorchValidator,
                   data_loader: DataLoader,
                   device: str = 'cuda',
                   num_samples: int = 100) -> Dict[str, float]:
    """
    对比ONNX和PyTorch的输出差异

    Args:
        onnx_validator: ONNX验证器
        pytorch_validator: PyTorch验证器
        data_loader: 数据加载器
        device: 设备
        num_samples: 对比样本数

    Returns:
        差异统计
    """
    differences = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': []
    }

    sample_count = 0
    print(f"\n=== Comparing ONNX vs PyTorch Outputs ({num_samples} samples) ===")

    for images, _ in data_loader:
        if sample_count >= num_samples:
            break

        # ONNX预测
        images_np = images.cpu().numpy()
        onnx_pred = onnx_validator.predict(images_np)

        # PyTorch预测
        images_torch = images.to(device)
        pytorch_pred = pytorch_validator.predict(images_torch)
        pytorch_pred = {k: v.cpu().numpy() for k, v in pytorch_pred.items()}

        # 计算差异
        for key in differences.keys():
            diff = np.abs(onnx_pred[key] - pytorch_pred[key])
            differences[key].append(diff)

        sample_count += images.shape[0]

    # 统计差异
    stats = {}
    for key, diffs in differences.items():
        diffs = np.concatenate(diffs, axis=0)
        stats[f'{key}_mean_diff'] = np.mean(diffs)
        stats[f'{key}_max_diff'] = np.max(diffs)
        stats[f'{key}_std_diff'] = np.std(diffs)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Validate ONNX vs PyTorch models')
    parser.add_argument('--onnx_path', type=str,
                       default='deployment/onnx_models/mobilenetv4_v1.1.onnx',
                       help='Path to ONNX model')
    parser.add_argument('--checkpoint_path', type=str,
                       default='experiments/mobilenetv4_v1.1/best_model.pth',
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--data_dir', type=str,
                       default='../bioastModel-trainers/data/m9e1n170',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for validation')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                       help='Use GPU for inference')
    parser.add_argument('--compare_samples', type=int, default=100,
                       help='Number of samples to compare outputs')

    args = parser.parse_args()

    # 设置设备
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 检查文件是否存在
    onnx_path = Path(args.onnx_path)
    checkpoint_path = Path(args.checkpoint_path)

    if not onnx_path.exists():
        print(f"Error: ONNX model not found at {onnx_path}")
        return

    if not checkpoint_path.exists():
        print(f"Error: PyTorch checkpoint not found at {checkpoint_path}")
        return

    print(f"\n{'='*80}")
    print(f"ONNX vs PyTorch Model Validation")
    print(f"{'='*80}")
    print(f"ONNX Model: {onnx_path}")
    print(f"PyTorch Checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.data_dir}")
    print(f"{'='*80}\n")

    # 加载验证集
    print("Loading validation dataset...")
    val_dataset = EnhancedMultitaskDataset(
        data_root=args.data_dir,
        split='val',
        transform=None  # 验证时不使用数据��强
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Validation set size: {len(val_dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(val_loader)}")

    # 初始化验证器
    print("\n" + "="*80)
    onnx_validator = ONNXValidator(str(onnx_path), use_gpu=args.use_gpu)
    print("="*80)

    print("\n" + "="*80)
    pytorch_validator = PyTorchValidator(str(checkpoint_path), device=device)
    print("="*80)

    # 1. 对比输出差异
    output_diffs = compare_outputs(
        onnx_validator,
        pytorch_validator,
        val_loader,
        device=device,
        num_samples=args.compare_samples
    )

    print("\nOutput Differences:")
    for key, value in output_diffs.items():
        print(f"  {key}: {value:.2e}")

    # 2. 验证ONNX模型
    onnx_metrics, onnx_time = validate_onnx(onnx_validator, val_loader, device)

    # 3. 验证PyTorch模型
    pytorch_metrics, pytorch_time = validate_pytorch(pytorch_validator, val_loader, device)

    # 4. 生成对比报告
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    # 性能对比
    print("\n📊 Performance Metrics:")
    print(f"\n{'Metric':<30} {'ONNX':<15} {'PyTorch':<15} {'Difference':<15}")
    print("-" * 75)

    metrics_to_compare = [
        ('Growth Level Accuracy', 'growth_level_acc'),
        ('Growth Pattern Accuracy', 'growth_pattern_acc'),
        ('Interference F1 Score', 'interference_f1'),
        ('Overall Accuracy', 'overall_acc')
    ]

    for name, key in metrics_to_compare:
        onnx_val = onnx_metrics[key]
        pytorch_val = pytorch_metrics[key]
        diff = onnx_val - pytorch_val

        onnx_str = f"{onnx_val:.4f}" if 'f1' in key else f"{onnx_val:.2%}"
        pytorch_str = f"{pytorch_val:.4f}" if 'f1' in key else f"{pytorch_val:.2%}"
        diff_str = f"{diff:+.4f}" if 'f1' in key else f"{diff:+.2%}"

        print(f"{name:<30} {onnx_str:<15} {pytorch_str:<15} {diff_str:<15}")

    # 推理时间对比
    num_samples = len(val_dataset)
    print(f"\n⏱️  Inference Time:")
    print(f"{'Model':<30} {'Total Time':<15} {'Avg per Sample':<20} {'Throughput':<15}")
    print("-" * 80)
    print(f"{'ONNX':<30} {onnx_time:.3f}s{'':<8} {onnx_time/num_samples*1000:.2f} ms{'':<9} {num_samples/onnx_time:.1f} FPS")
    print(f"{'PyTorch':<30} {pytorch_time:.3f}s{'':<8} {pytorch_time/num_samples*1000:.2f} ms{'':<9} {num_samples/pytorch_time:.1f} FPS")

    speedup = pytorch_time / onnx_time
    print(f"\nONNX Speedup: {speedup:.2f}x")

    # 验证结论
    print("\n" + "="*80)
    print("✅ VALIDATION SUMMARY")
    print("="*80)

    max_diff = max(abs(onnx_metrics[key] - pytorch_metrics[key])
                   for _, key in metrics_to_compare)

    if max_diff < 0.001:  # 0.1%差异
        print("✅ PASS: ONNX model matches PyTorch model (difference < 0.1%)")
        validation_status = "PASS"
    elif max_diff < 0.01:  # 1%差异
        print("⚠️  WARNING: ONNX model has minor differences (difference < 1%)")
        validation_status = "WARNING"
    else:
        print("❌ FAIL: ONNX model differs significantly from PyTorch model")
        validation_status = "FAIL"

    print(f"\nMaximum metric difference: {max_diff:.4f}")
    print(f"Output precision difference: {output_diffs['growth_level_mean_diff']:.2e}")

    # 保存结果
    results = {
        'validation_status': validation_status,
        'onnx_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in onnx_metrics.items()},
        'pytorch_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in pytorch_metrics.items()},
        'output_differences': {k: float(v) for k, v in output_diffs.items()},
        'inference_time': {
            'onnx': float(onnx_time),
            'pytorch': float(pytorch_time),
            'speedup': float(speedup)
        },
        'dataset_info': {
            'num_samples': num_samples,
            'batch_size': args.batch_size
        }
    }

    # 转换numpy类型为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results = convert_numpy(results)

    output_dir = Path('deployment/validation_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / 'onnx_validation_results.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 Results saved to: {result_file}")

    print("\n" + "="*80)

    return 0 if validation_status == "PASS" else 1


if __name__ == '__main__':
    exit(main())
