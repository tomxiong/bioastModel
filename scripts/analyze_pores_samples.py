#!/usr/bin/env python3
"""
深度分析 pores 样本问题

目标:
1. 对比验证集和测试集的 pores 样本
2. 可视化样本图片
3. 分析模型预测结果
4. 找出标注错误的具体样本
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3


def load_split_data(split_file):
    """加载数据集划分"""
    with open(split_file, 'r') as f:
        data = json.load(f)
    return data['splits']


def load_annotations(data_root, annotations_file="m9e1n170.json"):
    """加载标注数据"""
    ann_path = Path(data_root) / annotations_file
    with open(ann_path, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)

    return {ann['image_path']: ann for ann in annotations_data['annotations']}


def get_pores_samples(annotations, split_samples):
    """获取 pores 样本"""
    pores_samples = []

    for img_path in split_samples:
        ann = annotations[img_path]
        features = ann['features']
        interference = features.get('interference_factors', [])

        if 'pores' in interference:
            pores_samples.append({
                'image_path': img_path,
                'annotation': ann,
                'growth_level': features.get('growth_level', 'unknown'),
                'growth_pattern': features.get('growth_pattern', 'unknown'),
                'interference': interference
            })

    return pores_samples


def visualize_samples(data_root, samples, title, save_path, num_samples=20):
    """可视化样本"""
    # 随机选择样本
    selected = random.sample(samples, min(num_samples, len(samples)))

    # 创建图表
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    fig.suptitle(title, fontsize=20, fontweight='bold')

    for idx, sample in enumerate(selected):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # 读取图片
        img_path = Path(data_root) / sample['image_path']
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            ax.imshow(img, cmap='gray')

            # 添加标题信息
            growth_level = sample['growth_level']
            growth_pattern = sample['growth_pattern']
            interference = sample['interference']

            title_text = f"{growth_level}\n{growth_pattern}\n"
            title_text += f"IF: {', '.join(interference)}"

            ax.set_title(title_text, fontsize=8)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            ax.axis('off')

    # 隐藏多余的子图
    for idx in range(len(selected), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"可视化保存至: {save_path}")


def load_model_predictions(model_path, data_root, samples, device='cuda'):
    """加载模型并进行预测"""
    # 加载模型
    model = create_multilevel_mobilenetv3(model_size='small', input_channels=1)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for sample in samples:
            img_path = Path(data_root) / sample['image_path']
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # 预处理
            img = cv2.resize(img, (70, 70))
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5  # 标准化

            # 转换为张量
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

            # 预测
            outputs = model(img_tensor)

            # 解析输出
            growth_level_logits = outputs['growth_level']
            growth_pattern_logits = outputs['growth_pattern']
            interference_logits = outputs['interference_factors']

            # Growth level
            growth_level_pred = torch.argmax(growth_level_logits, dim=1).item()
            growth_level_prob = torch.softmax(growth_level_logits, dim=1).cpu().numpy()[0]

            # Growth pattern
            growth_pattern_pred = torch.argmax(growth_pattern_logits, dim=1).item()
            growth_pattern_prob = torch.softmax(growth_pattern_logits, dim=1).cpu().numpy()[0]

            # Interference factors (multi-label)
            interference_prob = torch.sigmoid(interference_logits).cpu().numpy()[0]

            predictions.append({
                'image_path': sample['image_path'],
                'growth_level_pred': growth_level_pred,
                'growth_level_prob': growth_level_prob.tolist(),
                'growth_pattern_pred': growth_pattern_pred,
                'growth_pattern_prob': growth_pattern_prob.tolist(),
                'interference_prob': {
                    'artifacts': interference_prob[0],
                    'contamination': interference_prob[1],
                    'debris': interference_prob[2],
                    'pores': interference_prob[3]
                },
                'true_interference': sample['interference']
            })

    return predictions


def analyze_predictions(predictions, threshold=0.5):
    """分析预测结果"""
    print("\n" + "="*80)
    print("模型预测分析")
    print("="*80)

    pores_probs = [p['interference_prob']['pores'] for p in predictions]

    print(f"\nPores 预测概率统计:")
    print(f"  样本数: {len(pores_probs)}")
    print(f"  最小值: {min(pores_probs):.4f}")
    print(f"  最大值: {max(pores_probs):.4f}")
    print(f"  平均值: {np.mean(pores_probs):.4f}")
    print(f"  中位数: {np.median(pores_probs):.4f}")
    print(f"  标准差: {np.std(pores_probs):.4f}")

    # 统计预测为 pores 的样本
    pred_positive = sum(1 for p in pores_probs if p >= threshold)
    print(f"\n预测为 pores 的样本 (threshold={threshold}): {pred_positive} / {len(pores_probs)}")

    # 概率分布
    print(f"\n概率分布:")
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

    for low, high in bins:
        count = sum(1 for p in pores_probs if low <= p < high)
        pct = count / len(pores_probs) * 100
        bar = '█' * int(pct / 2)
        print(f"  [{low:.1f}, {high:.1f}): {count:4d} ({pct:5.1f}%) {bar}")

    return pores_probs


def compare_val_test_predictions(val_preds, test_preds):
    """对比验证集和测试集的预测"""
    print("\n" + "="*80)
    print("验证集 vs 测试集预测对比")
    print("="*80)

    val_probs = [p['interference_prob']['pores'] for p in val_preds]
    test_probs = [p['interference_prob']['pores'] for p in test_preds]

    print(f"\n验证集 (n={len(val_probs)}):")
    print(f"  平均概率: {np.mean(val_probs):.4f}")
    print(f"  标准差: {np.std(val_probs):.4f}")
    print(f"  >0.5: {sum(1 for p in val_probs if p > 0.5)} ({sum(1 for p in val_probs if p > 0.5)/len(val_probs)*100:.1f}%)")
    print(f"  >0.7: {sum(1 for p in val_probs if p > 0.7)} ({sum(1 for p in val_probs if p > 0.7)/len(val_probs)*100:.1f}%)")

    print(f"\n测试集 (n={len(test_probs)}):")
    print(f"  平均概率: {np.mean(test_probs):.4f}")
    print(f"  标准差: {np.std(test_probs):.4f}")
    print(f"  >0.5: {sum(1 for p in test_probs if p > 0.5)} ({sum(1 for p in test_probs if p > 0.5)/len(test_probs)*100:.1f}%)")
    print(f"  >0.7: {sum(1 for p in test_probs if p > 0.7)} ({sum(1 for p in test_probs if p > 0.7)/len(test_probs)*100:.1f}%)")

    # t检验
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(val_probs, test_probs)
    print(f"\nt检验结果:")
    print(f"  t统计量: {t_stat:.4f}")
    print(f"  p值: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  ✅ 显著差异 (p < 0.05)")
        print(f"  结论: 验证集和测试集的预测分布存在显著差异")
    else:
        print(f"  ❌ 无显著差异 (p >= 0.05)")
        print(f"  结论: 验证集和测试集的预测分布相似")


def find_mislabeled_samples(predictions, data_root, threshold=0.5):
    """找出可能标注错误的样本"""
    print("\n" + "="*80)
    print("可能的标注错误样本")
    print("="*80)

    # 所有样本都标注为 pores,但预测概率很低
    low_prob_samples = [
        p for p in predictions
        if p['interference_prob']['pores'] < 0.3
    ]

    print(f"\n标注为 pores,但预测概率 < 0.3 的样本: {len(low_prob_samples)}")

    if low_prob_samples:
        # 按概率排序
        low_prob_samples.sort(key=lambda x: x['interference_prob']['pores'])

        print(f"\n概率最低的 10 个样本:")
        print(f"{'序号':<6} {'图片路径':<50} {'Pores概率':<12} {'其他干扰因素'}")
        print("-" * 100)

        for idx, sample in enumerate(low_prob_samples[:10], 1):
            img_path = sample['image_path']
            pores_prob = sample['interference_prob']['pores']

            # 获取其他高概率的干扰因素
            other_factors = []
            for factor, prob in sample['interference_prob'].items():
                if factor != 'pores' and prob > 0.3:
                    other_factors.append(f"{factor}({prob:.2f})")

            other_str = ', '.join(other_factors) if other_factors else 'None'

            print(f"{idx:<6} {img_path:<50} {pores_prob:<12.4f} {other_str}")

    return low_prob_samples


def visualize_probability_distribution(val_probs, test_probs, save_path):
    """可视化概率分布对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 验证集直方图
    axes[0, 0].hist(val_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[0, 0].axvline(x=np.mean(val_probs), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(val_probs):.3f}')
    axes[0, 0].set_xlabel('Pores Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Validation Set - Pores Probability Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 测试集直方图
    axes[0, 1].hist(test_probs, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[0, 1].axvline(x=np.mean(test_probs), color='green', linestyle='--', linewidth=2, label=f'Mean={np.mean(test_probs):.3f}')
    axes[0, 1].set_xlabel('Pores Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Test Set - Pores Probability Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 箱线图对比
    axes[1, 0].boxplot([val_probs, test_probs], labels=['Validation', 'Test'])
    axes[1, 0].axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[1, 0].set_ylabel('Pores Probability')
    axes[1, 0].set_title('Validation vs Test - Boxplot Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # CDF 对比
    val_sorted = np.sort(val_probs)
    test_sorted = np.sort(test_probs)
    val_cdf = np.arange(1, len(val_sorted) + 1) / len(val_sorted)
    test_cdf = np.arange(1, len(test_sorted) + 1) / len(test_sorted)

    axes[1, 1].plot(val_sorted, val_cdf, label='Validation', linewidth=2)
    axes[1, 1].plot(test_sorted, test_cdf, label='Test', linewidth=2)
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[1, 1].set_xlabel('Pores Probability')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution Function (CDF)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n概率分布对比图保存至: {save_path}")


def main():
    # 配置
    data_root = "ds/images"
    split_file = "ds/images/dataset_split_seed42.json"
    model_path = "experiments/multilevel_mobilenetv3_v0.9.6/best_model.pth"
    output_dir = Path("analysis/pores_diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Pores 样本深度分析")
    print("="*80)
    print(f"数据集: {split_file}")
    print(f"模型: {model_path}")
    print(f"输出目录: {output_dir}")

    # 加载数据
    print("\n加载数据...")
    splits = load_split_data(split_file)
    annotations = load_annotations(data_root)

    # 获取 pores 样本
    print("\n获取 pores 样本...")
    val_pores = get_pores_samples(annotations, splits['val'])
    test_pores = get_pores_samples(annotations, splits['test'])

    print(f"  验证集 pores 样本: {len(val_pores)}")
    print(f"  测试集 pores 样本: {len(test_pores)}")

    # 可视化样本
    print("\n可视化样本...")
    visualize_samples(
        data_root, val_pores,
        "Validation Set - Pores Samples (Random 20)",
        output_dir / "val_pores_samples.png"
    )

    visualize_samples(
        data_root, test_pores,
        "Test Set - Pores Samples (Random 20)",
        output_dir / "test_pores_samples.png"
    )

    # 加载模型并预测
    print("\n加载模型并预测...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  设备: {device}")

    val_predictions = load_model_predictions(model_path, data_root, val_pores, device)
    test_predictions = load_model_predictions(model_path, data_root, test_pores, device)

    print(f"  验证集预测: {len(val_predictions)} 样本")
    print(f"  测试集预测: {len(test_predictions)} 样本")

    # 分析预测结果
    print("\n" + "="*80)
    print("【验证集】预测分析")
    val_probs = analyze_predictions(val_predictions, threshold=0.7)

    print("\n" + "="*80)
    print("【测试集】预测分析")
    test_probs = analyze_predictions(test_predictions, threshold=0.5)

    # 对比验证集和测试集
    compare_val_test_predictions(val_predictions, test_predictions)

    # 可视化概率分布
    print("\n可视化概率分布...")
    visualize_probability_distribution(
        val_probs, test_probs,
        output_dir / "pores_probability_distribution.png"
    )

    # 找出可能标注错误的样本
    print("\n" + "="*80)
    print("【验证集】可能的标注错误")
    val_mislabeled = find_mislabeled_samples(val_predictions, data_root)

    print("\n" + "="*80)
    print("【测试集】可能的标注错误")
    test_mislabeled = find_mislabeled_samples(test_predictions, data_root)

    # 保存结果
    results = {
        'validation': {
            'total_samples': len(val_pores),
            'predictions': val_predictions,
            'mislabeled_candidates': val_mislabeled,
            'statistics': {
                'mean_prob': float(np.mean(val_probs)),
                'std_prob': float(np.std(val_probs)),
                'min_prob': float(np.min(val_probs)),
                'max_prob': float(np.max(val_probs))
            }
        },
        'test': {
            'total_samples': len(test_pores),
            'predictions': test_predictions,
            'mislabeled_candidates': test_mislabeled,
            'statistics': {
                'mean_prob': float(np.mean(test_probs)),
                'std_prob': float(np.std(test_probs)),
                'min_prob': float(np.min(test_probs)),
                'max_prob': float(np.max(test_probs))
            }
        }
    }

    output_file = output_dir / "pores_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n分析结果已保存至: {output_file}")
    print("\n✅ 分析完成!")


if __name__ == '__main__':
    main()
