#!/usr/bin/env python3
"""
训练 Multilevel MobileNetV4 v0.11.0

版本说明:
- 架构对比实验: MobileNetV4 vs MobileNetV3
- 集成 v0.10.0 的 Pattern-Conditional Pores Loss
- 测试更先进架构(UIB, SE/ECA注意力)的性能提升

架构创新 (MobileNetV4):
1. Universal Inverted Bottleneck (UIB)
   - 统一的可配置模块
   - Expansion + Depthwise + Attention + Projection
2. SE/ECA 注意力机制
   - 通道注意力增强特征表达
3. 轻量化设计
   - 为 70×70 小图像优化

继承 v0.10.0 核心创新:
1. Pattern-Conditional Pores Loss
   - Negative 样本: 权重 15.0
   - Positive 关键 pattern: 权重 15.0
   - 其他 Positive: 权重 0.1
2. 业务逻辑编码
   - 直接支持业务需求
   - 避免代理学习问题

对比目标:
- MobileNetV3 v0.10.0: 1.62M 参数, Pores F1 91.76%
- MobileNetV4 v0.11.0: ~2-3M 参数, 期望 Pores F1 92-95%
- 验证架构提升对 pores 检测的贡献
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

from models.multilevel_mobilenetv4 import create_multilevel_mobilenetv4_small
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset
from training.improved_multilevel_trainer import ImprovedMultiLevelTrainer
from training.pattern_conditional_loss import PatternConditionalInterferenceLoss


def main():
    parser = argparse.ArgumentParser(
        description='Train Multilevel MobileNetV4 v0.11.0 (Architecture Comparison)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集参数
    parser.add_argument('--data-root', type=str, default='ds/images',
                       help='数据集根目录')
    parser.add_argument('--annotations-file', type=str,
                       default='m9e1n170_cleaned_round2.json',
                       help='标注文件名')
    parser.add_argument('--split-file', type=str,
                       default='ds/images/dataset_split_seed44.json',
                       help='固定数据集划分文件')

    # 模型参数
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='模型大小')
    parser.add_argument('--input-channels', type=int, default=1,
                       help='输入通道数')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                       help='Dropout 比例')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批量大小')
    parser.add_argument('--num-epochs', type=int, default=40,
                       help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.002,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='学习率预热轮数')
    parser.add_argument('--patience', type=int, default=15,
                       help='早停耐心值')

    # v0.10.0: Pattern-Conditional Loss 参数 (继承)
    parser.add_argument('--use-pattern-conditional', action='store_true', default=True,
                       help='使用 Pattern-Conditional Pores Loss')
    parser.add_argument('--negative-pores-weight', type=float, default=15.0,
                       help='Negative 样本 pores 权重')
    parser.add_argument('--positive-critical-pores-weight', type=float, default=15.0,
                       help='Positive 关键 pattern pores 权重')
    parser.add_argument('--other-pores-weight', type=float, default=0.1,
                       help='其他 pattern pores 权重')

    # 类别权重参数 (继承 v0.10.0)
    parser.add_argument('--interference-weights', type=float, nargs=4,
                       default=[8.0, 3.0, 5.0, 10.0],
                       help='Interference 类别权重 [pores, artifacts, debris, contamination] (按数据集顺序)')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='使用类别权重')

    # 任务权重参数 (继承 v0.10.0)
    parser.add_argument('--task-weights', type=float, nargs=3,
                       default=[1.0, 2.0, 1.5],
                       help='任务权重 [growth_level, growth_pattern, interference]')
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
                       default='experiments/multilevel_mobilenetv4_v0.11.0',
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
    config['version'] = 'v0.11.0'
    config['description'] = 'MobileNetV4 架构对比实验 + Pattern-Conditional Pores Loss'

    with open(experiment_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("="*80)
    print("Multilevel MobileNetV4 v0.11.0 Training")
    print("="*80)
    print(f"Version: v0.11.0 (Architecture Comparison Experiment)")
    print(f"Experiment dir: {experiment_dir}")
    print(f"Device: {device}")
    print(f"\nArchitecture Innovation (MobileNetV4):")
    print(f"  🆕 Universal Inverted Bottleneck (UIB)")
    print(f"  🆕 SE/ECA Attention Mechanisms")
    print(f"  🆕 Optimized for 70×70 images")
    print(f"\nInherited from v0.10.0:")
    print(f"  ✅ Pattern-Conditional Pores Loss:")
    print(f"     - Negative samples: weight {args.negative_pores_weight}")
    print(f"     - Positive critical (center_dots, weak_scattered_pos): weight {args.positive_critical_pores_weight}")
    print(f"     - Other positive: weight {args.other_pores_weight}")
    print(f"\nComparison Target:")
    print(f"  - MobileNetV3 v0.10.0: 1.62M params, Pores F1 91.76%")
    print(f"  - MobileNetV4 v0.11.0: ~2-3M params, Expected Pores F1 92-95%")
    print("="*80)

    # 加载数据集
    print(f"\nLoading datasets...")

    train_dataset = EnhancedMultitaskDataset(
        data_root=args.data_root,
        split='train',
        split_file=args.split_file,
        annotations_file=args.annotations_file,
        transform=None
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

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    print(f"Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # 创建模型
    print(f"\nCreating model...")
    model = create_multilevel_mobilenetv4_small(
        input_channels=args.input_channels,
        dropout_rate=args.dropout_rate
    )

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: Multilevel MobileNetV4 ({args.model_size})")
    print(f"  Total Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # 创建 Pattern-Conditional Interference Loss
    pattern_conditional_loss = None
    if args.use_pattern_conditional:
        pattern_mapping = train_dataset.label_mappings['growth_pattern']

        # 基础权重: [pores, artifacts, debris, contamination] (按数据集顺序)
        base_weights = torch.tensor(args.interference_weights, device=device)

        pattern_conditional_loss = PatternConditionalInterferenceLoss(
            pattern_mapping=pattern_mapping,
            base_weights=base_weights,
            negative_pores_weight=args.negative_pores_weight,
            positive_critical_pores_weight=args.positive_critical_pores_weight,
            other_pores_weight=args.other_pores_weight,
            pores_index=0  # pores 在 interference_factors 中的索引
        )

    # 获取 label_info
    label_info = {
        'growth_level': train_dataset.label_mappings['growth_level'],
        'growth_pattern': train_dataset.label_mappings['growth_pattern'],
        'interference_factors': train_dataset.label_mappings['interference_factors']
    }

    # 创建训练器
    print(f"\nCreating trainer...")
    trainer = ImprovedMultiLevelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_info=label_info,
        device=device,
        experiment_dir=experiment_dir,
        task_weights=args.task_weights,
        interference_class_weights=args.interference_weights if args.use_class_weights else None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        optimize_thresholds=args.optimize_thresholds,
        pattern_conditional_pores_loss=pattern_conditional_loss  # 传入自定义 loss
    )

    # 训练
    if not args.eval_only:
        print(f"\nStarting training...")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Task weights: {args.task_weights}")
        print(f"  Interference weights: {args.interference_weights}")
        print(f"  Pattern-Conditional Loss: {'Enabled' if args.use_pattern_conditional else 'Disabled'}")
        print("="*80)

        trainer.train(num_epochs=args.num_epochs)

        print(f"\n✅ Training completed!")
        print(f"\nResults saved in: {experiment_dir}")
        print(f"  - config.json: training configuration")
        print(f"  - training_history.json: training history")
        print(f"  - test_results.json: test results (threshold 0.5)")
        print(f"  - test_results_with_thresholds.json: test results (optimized thresholds)")
        print(f"  - best_model.pth: best checkpoint")
        print("="*80)
    else:
        # 仅评估
        checkpoint_path = args.resume if args.resume else experiment_dir / 'best_model.pth'
        print(f"\nLoading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
        trainer.evaluate()


if __name__ == '__main__':
    main()
