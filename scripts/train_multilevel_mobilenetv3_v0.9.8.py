#!/usr/bin/env python3
"""
训练 Multilevel MobileNetV3 v0.9.8

版本说明:
- 基于 v0.9.6，使用第二轮清理后的数据集
- 重点解决: pores 标注冲突问题（通过数据清理提升 pores 检测性能）

改进点:
1. ✅ 继承 v0.9.6 的所有优化（任务权重 + 类别权重 + 阈值优化）
2. 🆕 **使用第二轮清理后的数据集** (m9e1n170_cleaned_round2.json)
   - 移除了 positive + strong_scattered + pores 等明显冲突（95 个）
   - 累计移除 1,726 个冲突标注（第一轮 1,631 + 第二轮 95）
   - Positive + Pores: 2,144 → 418（减少 80.5%）
   - Negative 占比: 71.2% → 92.7%（提升 21.5 百分点）
3. 🆕 使用固定数据集划分 (dataset_split_seed44.json)
4. 🆕 保持 v0.9.6 的训练参数（已验证有效）

核心优化:
- **数据集**: 第二轮清理后（m9e1n170_cleaned_round2.json，pores 纯净度 92.7%）
- **任务权重**: [1.0, 2.0, 0.8]（继承 v0.9.6）
- **类别权重**: [3.0, 5.0, 50.0, 1.0]（继承 v0.9.6）
- **阈值优化**: 启用
- **训练轮数**: 35 epochs
- **早停耐心**: 15

目标（基于数据清理）:
- **pores F1**: 0% → **65-75%**（数据清理后预期大幅提升）
- **pores 精确率**: **75-80%**
- **pores 召回率**: **65-75%**
- **Interference 整体 F1**: 49.91% → **68-72%**（整体提升 18-22 百分点）
- **Growth Pattern**: 保持 82.50%+
- **Growth Level**: 保持 98%+

数据清理成果 (参见 DATASET_CLEANING_SUMMARY.md):
- 第一轮: 移除 clustered + pores（1,631 个）
- 第二轮: 移除 strong_scattered + pores（91 个）+ scattered（1 个）+ heavy_growth（3 个）
- 剩余 Positive + Pores 都是合理边界案例: center_dots (254) + weak_scattered_pos (162)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset
from training.improved_multilevel_trainer import ImprovedMultiLevelTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Train Multilevel MobileNetV3 v0.9.8 (Cleaned Dataset Round 2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集参数
    parser.add_argument('--data-root', type=str, default='ds/images',
                       help='数据集根目录')
    parser.add_argument('--annotations-file', type=str,
                       default='m9e1n170_cleaned_round2.json',
                       help='标注文件名（第二轮清理后，相对于 data-root）')
    parser.add_argument('--split-file', type=str,
                       default='ds/images/dataset_split_seed44.json',
                       help='固定数据集划分文件')

    # 模型参数
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'large'],
                       help='模型大小')
    parser.add_argument('--input-channels', type=int, default=1,
                       help='输入通道数')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                       help='Dropout 比例')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批量大小')
    parser.add_argument('--num-epochs', type=int, default=35,
                       help='训练轮数 (v0.9.5: 50→35, 基于最佳epoch发现)')
    parser.add_argument('--learning-rate', type=float, default=0.002,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='学习率预热轮数')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值 (v0.9.5: 20→15)')

    # v0.9.2-v0.9.6: 类别权重参数
    parser.add_argument('--interference-weights', type=float, nargs=4,
                       default=[3.0, 5.0, 50.0, 1.0],
                       help='Interference类别权重 [artifacts, debris, contamination, pores] (v0.9.6: contamination 20→50)')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='使用类别权重')

    # v0.9.3-v0.9.5: 任务权重参数
    parser.add_argument('--task-weights', type=float, nargs=3,
                       default=[1.0, 2.0, 0.8],
                       help='任务权重 [growth_level, growth_pattern, interference] (v0.9.5: 1.5→2.0)')
    parser.add_argument('--optimize-thresholds', action='store_true', default=True,
                       help='启用阈值优化')

    # 系统参数
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='计算设备')

    # 实验参数
    parser.add_argument('--experiment-dir', type=str,
                       default='experiments/multilevel_mobilenetv3_v0.9.8',
                       help='实验目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--eval-only', action='store_true',
                       help='仅评估模式')

    args = parser.parse_args()

    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # 设置随机种子
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # 创建实验目录
    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config = vars(args)
    config['device'] = device
    config['version'] = 'v0.9.8'
    config['description'] = 'Multilevel MobileNetV3 with cleaned dataset round 2 (pores purity 92.7%, -1726 conflicts)'

    with open(experiment_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("="*80)
    print("Multilevel MobileNetV3 v0.9.8 训练")
    print("="*80)
    print(f"版本: v0.9.8 (第二轮数据清理)")
    print(f"实验目录: {experiment_dir}")
    print(f"设备: {device}")
    print(f"标注文件: {args.annotations_file}")
    print(f"固定划分: {args.split_file}")
    print(f"\n🆕 核心改进 (v0.9.8):")
    print(f"  📊 使用第二轮清理后的数据集:")
    print(f"     - 累计移除 1,726 个 pores 冲突标注")
    print(f"     - Positive + Pores: 2,144 → 418 (减少 80.5%)")
    print(f"     - Negative 占比: 71.2% → 92.7% (提升 21.5%)")
    print(f"     - 剩余 Positive+Pores 都是合理边界案例")
    print(f"\n继承 v0.9.6 优化:")
    print(f"  ✅ Interference 类别权重: {args.interference_weights}")
    print(f"  ✅ 任务权重: {args.task_weights}")
    print(f"  ✅ 阈值优化: 启用")
    print(f"  ✅ 训练轮数: {args.num_epochs} epochs")
    print(f"  ✅ 早停耐心: {args.patience} epochs")
    print(f"\n🎯 v0.9.8 目标 (基于数据清理):")
    print(f"  - pores F1: 0% → 65-75% (数据清理预期大幅提升)")
    print(f"  - pores 精确率: 75-80%")
    print(f"  - pores 召回率: 65-75%")
    print(f"  - Interference 整体 F1: 49.91% → 68-72% (提升 18-22%)")
    print(f"  - Growth Pattern: 保持 82.50%+")
    print(f"  - Growth Level: 保持 98%+")
    print("="*80)

    # 加载数据集（使用固定划分和清理后的标注）
    print(f"\n加载数据集...")
    print(f"  标注文件: {args.annotations_file}")
    print(f"  划分文件: {args.split_file}")

    train_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='train',
        split_file=args.split_file,
        annotations_file=args.annotations_file,
        transform=None  # 使用默认数据增强
    )

    val_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='val',
        split_file=args.split_file,
        annotations_file=args.annotations_file,
        transform=None
    )

    test_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='test',
        split_file=args.split_file,
        annotations_file=args.annotations_file,
        transform=None
    )

    print(f"\n数据集统计:")
    print(f"  Train: {len(train_dataset)} 样本")
    print(f"  Val:   {len(val_dataset)} 样本")
    print(f"  Test:  {len(test_dataset)} 样本")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    # 创建模型
    print(f"\n创建模型...")
    model = create_multilevel_mobilenetv3(
        model_size=args.model_size,
        input_channels=args.input_channels,
        dropout_rate=args.dropout_rate
    )

    # 保存模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info = {
        'model_name': 'Multilevel MobileNetV3 v0.9.5',
        'model_size': args.model_size,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_channels': args.input_channels,
        'dropout_rate': args.dropout_rate,
        'interference_weights': args.interference_weights if args.use_class_weights else None,
        'task_weights': args.task_weights
    }

    with open(experiment_dir / 'model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"  模型: Multilevel MobileNetV3 ({args.model_size})")
    print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  可训练: {trainable_params:,}")

    # 保存标签信息
    label_info = {
        'growth_level': train_dataset.label_mappings['growth_level'],
        'growth_pattern': train_dataset.label_mappings['growth_pattern'],
        'interference_factors': train_dataset.label_mappings['interference_factors']
    }

    with open(experiment_dir / 'label_info.json', 'w', encoding='utf-8') as f:
        json.dump(label_info, f, indent=2, ensure_ascii=False)

    # 创建训练器 (v0.9.5: 任务权重 [1.0, 2.0, 0.8])
    print(f"\n创建训练器...")

    # 准备类别权重参数
    interference_class_weights = args.interference_weights if args.use_class_weights else None

    trainer = ImprovedMultiLevelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_info=label_info,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        experiment_dir=str(experiment_dir),
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        interference_class_weights=interference_class_weights,
        task_weights=args.task_weights,
        optimize_thresholds=args.optimize_thresholds
    )

    # 恢复训练或仅评估
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)

    if args.eval_only:
        print(f"\n仅评估模式...")
        results = trainer.evaluate()
        print(f"\n评估结果:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # 开始训练
    print(f"\n开始训练...")
    print(f"  轮数: {args.num_epochs}")
    print(f"  批量大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  预热轮数: {args.warmup_epochs}")
    print(f"  早停耐心: {args.patience}")
    print(f"  任务权重: {args.task_weights}")
    if args.use_class_weights:
        print(f"  类别权重: {args.interference_weights}")
    print("="*80)

    history = trainer.train(
        num_epochs=args.num_epochs,
        save_best=True
    )

    # 保存训练历史
    with open(experiment_dir / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 阈值优化
    if args.optimize_thresholds:
        print(f"\n" + "="*80)
        print("阈值优化（验证集）")
        print("="*80)
        optimal_thresholds, optimal_f1_scores = trainer.optimize_thresholds_on_validation()

    # 最终评估
    print(f"\n" + "="*80)
    print("最终评估（测试集）")
    print("="*80)

    # 标准评估 (阈值 0.5)
    results = trainer.evaluate()

    # 保存测试结果
    with open(experiment_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 使用最优阈值评估
    if args.optimize_thresholds:
        print(f"\n" + "="*80)
        print("使用最优阈值评估（测试集）")
        print("="*80)
        results_with_thresholds = trainer.evaluate_with_thresholds(use_optimal_thresholds=True)

        # 保存阈值优化后的结果
        with open(experiment_dir / 'test_results_with_thresholds.json', 'w', encoding='utf-8') as f:
            json.dump(results_with_thresholds, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 训练完成！")
    print(f"\n结果保存在: {experiment_dir}")
    print(f"  - config.json: 训练配置")
    print(f"  - model_info.json: 模型信息")
    print(f"  - label_info.json: 标签映射")
    print(f"  - training_history.json: 训练历史")
    print(f"  - test_results.json: 测试结果 (默认阈值 0.5)")
    if args.optimize_thresholds:
        print(f"  - test_results_with_thresholds.json: 测试结果 (最优阈值)")
        print(f"  - optimal_thresholds.json: 最优阈值配置")
    print(f"  - best_model.pth: 最佳检查点")
    print("="*80)

    # v0.9.5: 性能对比
    print(f"\n📊 v0.9.5 性能对比:")
    print("="*80)

    # 对比数据
    v093_growth_pattern = 0.8770
    v094_growth_pattern = 0.8057
    v093_interference_optimized = 0.4818
    v094_interference_optimized = 0.5345

    if 'growth_pattern' in results:
        current_gp = results['growth_pattern']['accuracy']
        print(f"\nGrowth Pattern 准确率:")
        print(f"  v0.9.3: {v093_growth_pattern:.4f} (87.70%)")
        print(f"  v0.9.4: {v094_growth_pattern:.4f} (80.57%)")
        print(f"  v0.9.5: {current_gp:.4f} ({current_gp*100:.2f}%)")

        if current_gp >= 0.85:
            print(f"  🎯 达到目标 (85%+)！")
            gp_improvement_from_094 = (current_gp - v094_growth_pattern) / v094_growth_pattern * 100
            print(f"  v0.9.5 vs v0.9.4: {gp_improvement_from_094:+.2f}%")
        elif current_gp > v094_growth_pattern:
            print(f"  ⬆️ 有所恢复")
        else:
            print(f"  ⚠️ 仍需优化")

    if args.optimize_thresholds and 'interference_factors' in results_with_thresholds:
        optimized_f1 = results_with_thresholds['interference_factors']['overall_f1']
        print(f"\nInterference F1 (优化阈值):")
        print(f"  v0.9.3: {v093_interference_optimized:.4f} (48.18%)")
        print(f"  v0.9.4: {v094_interference_optimized:.4f} (53.45%)")
        print(f"  v0.9.5: {optimized_f1:.4f} ({optimized_f1*100:.2f}%)")

        if optimized_f1 >= 0.50:
            print(f"  ✅ 保持目标 (50%+)")
        else:
            print(f"  ⚠️ 低于目标")

    if 'growth_level' in results:
        current_gl = results['growth_level']['accuracy']
        print(f"\nGrowth Level 准确率:")
        print(f"  v0.9.3: 98.80%")
        print(f"  v0.9.4: 98.63%")
        print(f"  v0.9.5: {current_gl:.4f} ({current_gl*100:.2f}%)")

    print("="*80)


if __name__ == '__main__':
    main()
