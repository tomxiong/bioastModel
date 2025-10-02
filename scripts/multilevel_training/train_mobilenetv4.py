#!/usr/bin/env python3
"""
Training script for MultiLevelMobileNetV4
基于改进版 multilevel mobilenetv3 的成功经验训练 MobileNetV4

性能目标:
- 总体准确率: >92.65% (超越改进版 V3)
- Growth Level: >98%
- Growth Pattern: >86%
- Interference Factors: >92%
"""

import os
import sys
import json
import logging
import argparse
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.multilevel_mobilenetv4 import (
    create_multilevel_mobilenetv4_small,
    create_multilevel_mobilenetv4_medium,
    create_multilevel_mobilenetv4_large,
    MODEL_CONFIG
)
from training.improved_multilevel_trainer import ImprovedMultiLevelTrainer
from training.multilevel_dataset import create_multilevel_dataloaders


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Train MultiLevelMobileNetV4 Model'
    )

    # Data arguments
    parser.add_argument('--json_path', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='Path to dataset JSON file')
    parser.add_argument('--image_root', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images',
                       help='Root directory for images')

    # Model arguments
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Model size variant')
    parser.add_argument('--input_channels', type=int, default=1,
                       help='Number of input channels (1 for grayscale)')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')

    # Training arguments (基于改进版的最佳配置)
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')

    # Data split
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test data ratio')

    # Experiment
    parser.add_argument('--experiment_dir', type=str,
                       default='experiments/mobilenetv4',
                       help='Directory to save experiment results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate the model')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建实验目录
    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)

    # 设置日志
    setup_logging(experiment_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info(f"MultiLevelMobileNetV4 Training ({args.model_size.upper()})")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Arguments: {vars(args)}")

    # 保存配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 创建数据加载器
    logger.info("\n" + "=" * 80)
    logger.info("Creating data loaders...")
    logger.info("=" * 80)
    try:
        split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)

        train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
            json_path=args.json_path,
            image_root=args.image_root,
            batch_size=args.batch_size,
            split_ratio=split_ratio
        )

        logger.info(f"Data loaders created successfully")
        logger.info(f"Train: {len(train_loader)} batches ({len(train_loader.dataset)} samples)")
        logger.info(f"Val:   {len(val_loader)} batches ({len(val_loader.dataset)} samples)")
        logger.info(f"Test:  {len(test_loader)} batches ({len(test_loader.dataset)} samples)")

        # 保存标签信息
        label_info_path = os.path.join(experiment_dir, 'label_info.json')
        with open(label_info_path, 'w') as f:
            json.dump(label_info, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建模型
    logger.info("\n" + "=" * 80)
    logger.info("Creating model...")
    logger.info("=" * 80)
    try:
        # 选择模型大小
        if args.model_size == 'small':
            model = create_multilevel_mobilenetv4_small(
                input_channels=args.input_channels,
                dropout_rate=args.dropout_rate
            )
        elif args.model_size == 'medium':
            model = create_multilevel_mobilenetv4_medium(
                input_channels=args.input_channels,
                dropout_rate=args.dropout_rate
            )
        else:  # large
            model = create_multilevel_mobilenetv4_large(
                input_channels=args.input_channels,
                dropout_rate=args.dropout_rate
            )

        model_info = model.get_model_info()
        logger.info("Model created successfully")
        logger.info(f"Model: {model_info['model_name']}")
        logger.info(f"Total Parameters: {model_info['total_parameters']:,}")
        logger.info(f"Trainable Parameters: {model_info['trainable_parameters']:,}")
        logger.info(f"Task Weights: {model_info['task_weights']}")

        # 保存模型信息
        model_info_path = os.path.join(experiment_dir, 'model_info.json')
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 创建训练器
    logger.info("\n" + "=" * 80)
    logger.info("Creating trainer...")
    logger.info("=" * 80)
    try:
        trainer = ImprovedMultiLevelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            label_info=label_info,
            device=device,
            experiment_dir=experiment_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience
        )

        logger.info("Trainer created successfully")
        logger.info(f"Warmup epochs: {args.warmup_epochs}")
        logger.info(f"Patience: {args.patience}")
        logger.info(f"Initial learning rate: {args.learning_rate}")

    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # 恢复检查点（如果指定）
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
            trainer.main_scheduler.load_state_dict(checkpoint['main_scheduler_state_dict'])
            trainer.best_val_accuracy = checkpoint['best_val_accuracy']
            trainer.history = checkpoint['history']
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return

    # 仅评估模式
    if args.eval_only:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Mode")
        logger.info("=" * 80)

        logger.info("Running final evaluation on test set...")
        test_results = trainer.evaluate_final()

        logger.info("\n" + "=" * 80)
        logger.info("Test Results")
        logger.info("=" * 80)
        logger.info(json.dumps(test_results, indent=2))

        return

    # 训练模型
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    try:
        trainer.train(num_epochs=args.num_epochs)

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 最终评估
    logger.info("\n" + "=" * 80)
    logger.info("Running final evaluation on test set...")
    logger.info("=" * 80)
    try:
        test_results = trainer.evaluate_final()

        logger.info("\n" + "=" * 80)
        logger.info("Final Test Results")
        logger.info("=" * 80)
        logger.info(json.dumps(test_results, indent=2))

        # 保存训练总结
        summary_path = os.path.join(experiment_dir, 'training_summary.json')
        summary = {
            'best_val_accuracy': trainer.best_val_accuracy,
            'best_epoch': trainer.best_epoch,
            'total_epochs': args.num_epochs,
            'final_results': test_results,
            'config': vars(args)
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nTraining summary saved to: {summary_path}")
        logger.info(f"Best model saved to: {os.path.join(experiment_dir, 'best_model.pth')}")

    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
