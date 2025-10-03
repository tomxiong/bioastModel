#!/usr/bin/env python3
"""
训练 Multilevel MobileNetV3 v0.9.3

版本说明:
- 基于v0.9.2，添加任务权重调整 + 阈值优化
- 目标: 恢复 Growth Pattern 性能 + 进一步提升 Interference F1

改进点:
1. ✅ 继承 v0.9.2 的所有改进 (类别权重优化)
2. 🆕 调整任务权重 [1.0, 1.5, 0.8] - 增加 Growth Pattern 权重
3. 🆕 启用阈值优化 - 为每个 Interference 类别找最优预测阈值
4. 🆕 延长训练时间 - 30 epochs
5. 🆕 增加早停耐心值 - patience=15

目标:
- Growth Pattern: 73.50% → 80%+
- Interference F1: 39.61% → 42-45%
- 整体平衡: 三个任务性能均衡
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
        description='Train Multilevel MobileNetV3 v0.9.3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集参数
    parser.add_argument('--data-root', type=str, default='ds/images',
                       help='数据集根目录')
    parser.add_argument('--split-file', type=str,
                       default='ds/images/dataset_split_seed42.json',
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
    parser.add_argument('--num-epochs', type=int, default=30,
                       help='训练轮数 (v0.9.3: 20→30)')
    parser.add_argument('--learning-rate', type=float, default=0.002,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='学习率预热轮数')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值 (v0.9.3: 10→15)')

    # 🆕 v0.9.2-v0.9.3: 类别权重参数
    parser.add_argument('--interference-weights', type=float, nargs=4,
                       default=[3.0, 5.0, 20.0, 1.0],
                       help='Interference类别权重 [artifacts, debris, contamination, pores]')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='使用类别权重')

    # 🆕 v0.9.3 新增：任务权重参数
    parser.add_argument('--task-weights', type=float, nargs=3,
                       default=[1.0, 1.5, 0.8],
                       help='任务权重 [growth_level, growth_pattern, interference] (v0.9.3核心改进)')
    parser.add_argument('--optimize-thresholds', action='store_true', default=True,
                       help='启用阈值优化 (v0.9.3核心改进)')

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
                       default='experiments/multilevel_mobilenetv3_v0.9.3',
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
    config['version'] = 'v0.9.3'
    config['description'] = 'Multilevel MobileNetV3 with task weights + threshold optimization'

    with open(experiment_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("="*80)
    print("Multilevel MobileNetV3 v0.9.3 训练")
    print("="*80)
    print(f"版本: v0.9.3")
    print(f"实验目录: {experiment_dir}")
    print(f"设备: {device}")
    print(f"固定划分: {args.split_file}")
    print(f"\n改进点:")
    print(f"  ✅ 继承 v0.9.2: Interference类别权重优化")
    print(f"  🆕 v0.9.3 核心改进:")
    print(f"     1. 任务权重调整: {args.task_weights}")
    print(f"        - growth_level: {args.task_weights[0]}")
    print(f"        - growth_pattern: {args.task_weights[1]} (增加)")
    print(f"        - interference: {args.task_weights[2]} (适当降低)")
    if args.use_class_weights:
        print(f"     2. Interference类别权重:")
        print(f"        - artifacts: {args.interference_weights[0]}")
        print(f"        - debris: {args.interference_weights[1]}")
        print(f"        - contamination: {args.interference_weights[2]}")
        print(f"        - pores: {args.interference_weights[3]}")
    if args.optimize_thresholds:
        print(f"     3. 阈值优化: 启用")
    print(f"  📈 目标:")
    print(f"     - Growth Pattern: 73.50% → 80%+")
    print(f"     - Interference F1: 39.61% → 42-45%")
    print(f"     - 三个任务性能均衡")
    print("="*80)

    # 加载数据集（使用固定划分）
    print(f"\n加载数据集...")

    train_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='train',
        split_file=args.split_file,
        transform=None  # 使用默认数据增强
    )

    val_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='val',
        split_file=args.split_file,
        transform=None
    )

    test_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='test',
        split_file=args.split_file,
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
        'model_name': 'Multilevel MobileNetV3 v0.9.2',
        'model_size': args.model_size,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_channels': args.input_channels,
        'dropout_rate': args.dropout_rate,
        'interference_weights': args.interference_weights if args.use_class_weights else None
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

    # 创建训练器 (🆕 v0.9.3: 传入任务权重 + 阈值优化)
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
        interference_class_weights=interference_class_weights,  # v0.9.2
        task_weights=args.task_weights,  # 🆕 v0.9.3
        optimize_thresholds=args.optimize_thresholds  # 🆕 v0.9.3
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

    # 🆕 v0.9.3: 阈值优化
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

    # 🆕 v0.9.3: 使用最优阈值评估
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

    # 🆕 v0.9.3: 性能对比
    print(f"\n📊 v0.9.3 性能对比:")
    print("="*80)

    # 对比数据
    v091_f1 = 0.2575
    v092_f1 = 0.3961
    v091_growth_pattern = 0.8310
    v092_growth_pattern = 0.7350

    if 'interference_factors' in results:
        current_f1 = results['interference_factors']['overall_f1']
        print(f"\nInterference F1:")
        print(f"  v0.9.1: {v091_f1:.4f} (25.75%)")
        print(f"  v0.9.2: {v092_f1:.4f} (39.61%)")
        print(f"  v0.9.3: {current_f1:.4f} ({current_f1*100:.2f}%)")
        improvement_from_092 = (current_f1 - v092_f1) / v092_f1 * 100
        print(f"  v0.9.3 vs v0.9.2: {improvement_from_092:+.2f}%")

        if args.optimize_thresholds and 'interference_factors' in results_with_thresholds:
            optimized_f1 = results_with_thresholds['interference_factors']['overall_f1']
            print(f"\n  使用最优阈值: {optimized_f1:.4f} ({optimized_f1*100:.2f}%)")
            threshold_improvement = (optimized_f1 - current_f1) / current_f1 * 100
            print(f"  阈值优化提升: {threshold_improvement:+.2f}%")

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

    print("="*80)


if __name__ == '__main__':
    main()