#!/usr/bin/env python3
"""
评估 v0.9.3 模型 (补充脚本)
"""
import os
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset
from training.improved_multilevel_trainer import ImprovedMultiLevelTrainer

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiment_dir = Path('experiments/multilevel_mobilenetv3_v0.9.3')

    # 加载配置
    with open(experiment_dir / 'config.json', 'r') as f:
        config = json.load(f)

    # 加载数据集
    test_dataset = EnhancedMultitaskDataset(
        data_root=config['data_root'],
        split='test',
        split_file=config['split_file'],
        transform=None
    )

    val_dataset = EnhancedMultitaskDataset(
        data_root=config['data_root'],
        split='val',
        split_file=config['split_file'],
        transform=None
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 创建模型
    model = create_multilevel_mobilenetv3(
        model_size=config['model_size'],
        input_channels=config['input_channels'],
        dropout_rate=config['dropout_rate']
    )

    # 加载标签信息
    with open(experiment_dir / 'label_info.json', 'r') as f:
        label_info = json.load(f)

    # 创建训练器
    trainer = ImprovedMultiLevelTrainer(
        model=model,
        train_loader=test_loader,  # dummy
        val_loader=val_loader,
        test_loader=test_loader,
        label_info=label_info,
        device=device,
        experiment_dir=str(experiment_dir),
        task_weights=config['task_weights'],
        interference_class_weights=config['interference_weights'],
        optimize_thresholds=True
    )

    # 加载最佳模型
    checkpoint = torch.load(experiment_dir / 'best_model.pth', map_location=device, weights_only=False)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])

    print("="*80)
    print("标准评估 (阈值 0.5)")
    print("="*80)
    results = trainer.evaluate()

    with open(experiment_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n"+"="*80)
    print("使用最优阈值评估")
    print("="*80)

    # 重新优化阈值(因为之前中断了)
    optimal_thresholds, optimal_f1_scores = trainer.optimize_thresholds_on_validation()

    results_with_thresholds = trainer.evaluate_with_thresholds(use_optimal_thresholds=True)

    with open(experiment_dir / 'test_results_with_thresholds.json', 'w', encoding='utf-8') as f:
        json.dump(results_with_thresholds, f, indent=2, ensure_ascii=False)

    # 打印结果
    print("\n" + "="*80)
    print("📊 v0.9.3 性能对比:")
    print("="*80)

    v091_f1 = 0.2575
    v092_f1 = 0.3961
    v091_growth_pattern = 0.8310
    v092_growth_pattern = 0.7350

    if 'interference_factors' in results:
        current_f1 = results['interference_factors']['overall_f1']
        print(f"\nInterference F1 (默认阈值 0.5):")
        print(f"  v0.9.1: {v091_f1:.4f} (25.75%)")
        print(f"  v0.9.2: {v092_f1:.4f} (39.61%)")
        print(f"  v0.9.3: {current_f1:.4f} ({current_f1*100:.2f}%)")
        improvement_from_092 = (current_f1 - v092_f1) / v092_f1 * 100
        print(f"  v0.9.3 vs v0.9.2: {improvement_from_092:+.2f}%")

        if 'interference_factors' in results_with_thresholds:
            optimized_f1 = results_with_thresholds['interference_factors']['overall_f1']
            print(f"\nInterference F1 (最优阈值):")
            print(f"  v0.9.3 优化后: {optimized_f1:.4f} ({optimized_f1*100:.2f}%)")
            threshold_improvement = (optimized_f1 - current_f1) / current_f1 * 100
            print(f"  阈值优化提升: {threshold_improvement:+.2f}%")

            print(f"\n  最优阈值:")
            print(f"    pores: {optimal_thresholds[0]:.2f}")
            print(f"    artifacts: {optimal_thresholds[1]:.2f}")
            print(f"    debris: {optimal_thresholds[2]:.2f}")
            print(f"    contamination: {optimal_thresholds[3]:.2f}")

    if 'growth_pattern' in results:
        current_gp = results['growth_pattern']['accuracy']
        print(f"\nGrowth Pattern 准确率:")
        print(f"  v0.9.1: {v091_growth_pattern:.4f} (83.10%)")
        print(f"  v0.9.2: {v092_growth_pattern:.4f} (73.50%)")
        print(f"  v0.9.3: {current_gp:.4f} ({current_gp*100:.2f}%)")
        gp_improvement_from_092 = (current_gp - v092_growth_pattern) / v092_growth_pattern * 100
        print(f"  v0.9.3 vs v0.9.2: {gp_improvement_from_092:+.2f}%")

        if current_gp > 0.80:
            print(f"  🎯 达到目标 (80%+)！")
        elif current_gp > v092_growth_pattern:
            print(f"  ⬆️ 有所恢复")
        else:
            print(f"  ⚠️ 需要进一步优化")

    if 'growth_level' in results:
        current_gl = results['growth_level']['accuracy']
        print(f"\nGrowth Level 准确率:")
        print(f"  v0.9.3: {current_gl:.4f} ({current_gl*100:.2f}%)")

    print("="*80)
    print("\n✅ 评估完成！")

if __name__ == '__main__':
    main()
