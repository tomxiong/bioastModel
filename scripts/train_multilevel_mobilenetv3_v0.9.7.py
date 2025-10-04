#!/usr/bin/env python3
"""
训练 Multilevel MobileNetV3 v0.9.7

版本说明:
- 基于 v0.9.2,使用新的数据集划分 (seed=43)
- 验证不同随机划分对模型性能的影响
- 重点观察 pores 和 contamination 性能变化

改进点:
1. ✅ 继承 v0.9.2 的所有优化 (类别权重)
2. 🆕 使用新的数据集划分 (seed=43, 比例仍为 7:1.5:1.5)
3. 🆕 保持与 v0.9.2 完全相同的训练配置,仅改变数据划分

核心配置 (继承 v0.9.2):
- 类别权重: [3.0, 5.0, 20.0, 1.0] (artifacts, debris, contamination, pores)
- 训练轮数: 20 epochs
- 早停耐心: 10
- 无任务权重 (默认 [1.0, 1.0, 1.0])
- 无阈值优化

目标:
- 验证数据集划分的稳定性
- 对比 v0.9.2 (seed=42) vs v0.9.7 (seed=43) 的性能差异
- 确认 pores 问题是否与特定数据划分相关
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
        description='Train Multilevel MobileNetV3 v0.9.7',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集参数
    parser.add_argument('--data-root', type=str, default='ds/images',
                       help='数据集根目录')
    parser.add_argument('--split-file', type=str,
                       default='ds/images/dataset_split_seed43.json',
                       help='固定数据集划分文件 (v0.9.7: seed=43)')

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
    parser.add_argument('--num-epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.002,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='学习率预热轮数')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')

    # v0.9.2-v0.9.7: 类别权重参数
    parser.add_argument('--interference-weights', type=float, nargs=4,
                       default=[3.0, 5.0, 20.0, 1.0],
                       help='Interference类别权重 [artifacts, debris, contamination, pores]')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='使用类别权重 (v0.9.7: 继承 v0.9.2)')

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
                       default='experiments/multilevel_mobilenetv3_v0.9.7',
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
    config['version'] = 'v0.9.7'
    config['description'] = 'Multilevel MobileNetV3 v0.9.7 - Based on v0.9.2 with new dataset split (seed=43)'

    with open(experiment_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("="*80)
    print("Multilevel MobileNetV3 v0.9.7 训练")
    print("="*80)
    print(f"版本: v0.9.7")
    print(f"实验目录: {experiment_dir}")
    print(f"设备: {device}")
    print(f"固定划分: {args.split_file}")
    print(f"\n核心改进 (v0.9.7):")
    print(f"  🆕 使用新数据集划分 (seed=43)")
    print(f"     - 验证不同随机划分对性能的影响")
    print(f"     - pores 分布: train=37.26%, val=37.24%, test=37.30%")
    print(f"     - contamination: train=20, val=6, test=6")
    print(f"\n继承 v0.9.2 配置:")
    print(f"  ✅ Interference 类别权重优化")
    if args.use_class_weights:
        print(f"     - artifacts: {args.interference_weights[0]}")
        print(f"     - debris: {args.interference_weights[1]}")
        print(f"     - contamination: {args.interference_weights[2]}")
        print(f"     - pores: {args.interference_weights[3]}")
    print(f"\n📈 目标:")
    print(f"  - 对比 v0.9.2 (seed=42) vs v0.9.7 (seed=43)")
    print(f"  - 验证 pores 问题是否与特定数据划分相关")
    print(f"  - 确认模型性能的稳定性")
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

    # 创建训练器 (🆕 v0.9.2: 传入类别权重)
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
        interference_class_weights=interference_class_weights  # 🆕 新增参数
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

    # 最终评估
    print(f"\n" + "="*80)
    print("最终评估（测试集）")
    print("="*80)

    results = trainer.evaluate()

    # 保存测试结果
    with open(experiment_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 训练完成！")
    print(f"\n结果保存在: {experiment_dir}")
    print(f"  - config.json: 训练配置")
    print(f"  - model_info.json: 模型信息")
    print(f"  - label_info.json: 标签映射")
    print(f"  - training_history.json: 训练历史")
    print(f"  - test_results.json: 测试结果")
    print(f"  - best_model.pth: 最佳检查点")
    print("="*80)

    # 🆕 v0.9.2: 性能对比
    if 'interference_factors' in results:
        current_f1 = results['interference_factors']['overall_f1']
        baseline_f1 = 0.2575  # v0.9.1 的 F1 分数
        improvement = (current_f1 - baseline_f1) / baseline_f1 * 100

        print(f"\n📊 v0.9.2 性能改进:")
        print(f"  v0.9.1 Interference F1: {baseline_f1:.4f} (25.75%)")
        print(f"  v0.9.2 Interference F1: {current_f1:.4f} ({current_f1*100:.2f}%)")
        print(f"  改进幅度: {improvement:+.2f}%")

        if current_f1 > 0.36:  # 36% 目标
            print(f"  🎯 达到目标 (36%+)！")
        elif current_f1 > baseline_f1:
            print(f"  ⬆️ 有所改进，但未达到目标")
        else:
            print(f"  ⚠️ 性能下降，需要进一步优化")
        print("="*80)


if __name__ == '__main__':
    main()