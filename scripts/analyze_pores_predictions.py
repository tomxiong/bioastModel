"""
分析 v0.9.8 模型在 pores 检测上的预测行为
"""
import json
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset
import torchvision.transforms as transforms

def analyze_pores_predictions():
    """分析模型对 pores 的预测行为"""

    # 加载数据集信息
    with open('ds/images/m9e1n170_cleaned_round2.json', 'r') as f:
        dataset = json.load(f)

    with open('ds/images/dataset_split_seed44.json', 'r') as f:
        split_data = json.load(f)

    test_indices = split_data['splits']['test']

    # 统计测试集中 pores 样本
    pores_samples = []
    for idx in test_indices:
        annotation = dataset['annotations'][idx]
        has_pores = 'pores' in annotation['features']['interference_factors']

        if has_pores:
            pores_samples.append({
                'index': idx,
                'image_path': annotation['image_path'],
                'growth_level': annotation['features']['growth_level'],
                'growth_pattern': annotation['features']['growth_pattern'],
                'interference_factors': annotation['features']['interference_factors']
            })

    print(f"测试集中 pores 样本统计:")
    print(f"总测试样本: {len(test_indices)}")
    print(f"总 pores 样本: {len(pores_samples)}")

    # 按 growth_level 分组
    negative_pores = [s for s in pores_samples if s['growth_level'] == 'negative']
    positive_pores = [s for s in pores_samples if s['growth_level'] != 'negative']

    print(f"Negative + pores: {len(negative_pores)} ({len(negative_pores)/len(pores_samples)*100:.1f}%)")
    print(f"Positive + pores: {len(positive_pores)} ({len(positive_pores)/len(pores_samples)*100:.1f}%)")

    # 统计 Positive pores 的 growth_pattern
    print("\nPositive pores 的 growth_pattern 分布:")
    pattern_counts = {}
    for s in positive_pores:
        pattern = s['growth_pattern']
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count} ({count/len(positive_pores)*100:.1f}%)")

    # 加载模型
    print("\n加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_multilevel_mobilenetv3(size='small', input_channels=1)

    checkpoint_path = 'experiments/multilevel_mobilenetv3_v0.9.8/model_best.pth'
    if not Path(checkpoint_path).exists():
        print(f"错误: 找不到模型检查点 {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 创建数据加载器
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = EnhancedMultitaskDataset(
        data_root='ds/images',
        annotations_file='m9e1n170_cleaned_round2.json',
        split_file='ds/images/dataset_split_seed44.json',
        split='test',
        transform=test_transform
    )

    # 分析 pores 样本的预测
    print("\n分析 pores 样本预测...")
    pores_predictions = []

    with torch.no_grad():
        for i, sample_info in enumerate(pores_samples):
            # 从数据集获取样本（使用索引直接访问）
            try:
                sample = test_dataset[i] if i < len(test_dataset) else None
                if sample is None:
                    continue

                image = sample['image']
                image = image.unsqueeze(0).to(device)

                # 预测
                level_out, pattern_out, interference_out = model(image)

                # 获取 pores 预测概率 (interference_factors: pores, artifacts, debris, contamination)
                pores_prob = torch.sigmoid(interference_out[0, 0]).item()  # pores is index 0

                pores_predictions.append({
                    'image_path': sample_info['image_path'],
                    'growth_level': sample_info['growth_level'],
                    'growth_pattern': sample_info['growth_pattern'],
                    'pores_prob': pores_prob,
                    'predicted_pores': pores_prob > 0.5
                })

                if len(pores_predictions) >= 200:  # 分析前200个
                    break

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    # 统计预测结果
    pores_predictions.sort(key=lambda x: x['pores_prob'], reverse=True)

    print(f"\nPores 预测概率分布 (前100个样本):")
    prob_ranges = {
        '0.9-1.0': 0,
        '0.7-0.9': 0,
        '0.5-0.7': 0,
        '0.3-0.5': 0,
        '0.1-0.3': 0,
        '0.0-0.1': 0
    }

    for pred in pores_predictions:
        prob = pred['pores_prob']
        if prob >= 0.9:
            prob_ranges['0.9-1.0'] += 1
        elif prob >= 0.7:
            prob_ranges['0.7-0.9'] += 1
        elif prob >= 0.5:
            prob_ranges['0.5-0.7'] += 1
        elif prob >= 0.3:
            prob_ranges['0.3-0.5'] += 1
        elif prob >= 0.1:
            prob_ranges['0.1-0.3'] += 1
        else:
            prob_ranges['0.0-0.1'] += 1

    for range_name, count in prob_ranges.items():
        print(f"  {range_name}: {count} ({count/len(pores_predictions)*100:.1f}%)")

    print(f"\n预测为 pores (prob > 0.5): {sum(1 for p in pores_predictions if p['predicted_pores'])}")
    print(f"预测为非 pores (prob <= 0.5): {sum(1 for p in pores_predictions if not p['predicted_pores'])}")

    # 显示最高和最低概率的样本
    print("\n预测概率最高的 10 个样本:")
    for i, pred in enumerate(pores_predictions[:10], 1):
        print(f"{i}. {pred['image_path']}")
        print(f"   Growth: {pred['growth_level']}, Pattern: {pred['growth_pattern']}")
        print(f"   Pores prob: {pred['pores_prob']:.4f}")

    print("\n预测概率最低的 10 个样本:")
    for i, pred in enumerate(pores_predictions[-10:], 1):
        print(f"{i}. {pred['image_path']}")
        print(f"   Growth: {pred['growth_level']}, Pattern: {pred['growth_pattern']}")
        print(f"   Pores prob: {pred['pores_prob']:.4f}")

    # 保存详细结果
    output = {
        'total_pores_samples': len(pores_samples),
        'negative_pores': len(negative_pores),
        'positive_pores': len(positive_pores),
        'analyzed_samples': len(pores_predictions),
        'probability_distribution': prob_ranges,
        'predictions': pores_predictions
    }

    output_file = 'experiments/multilevel_mobilenetv3_v0.9.8/pores_prediction_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n详细分析结果已保存到: {output_file}")

if __name__ == '__main__':
    analyze_pores_predictions()
