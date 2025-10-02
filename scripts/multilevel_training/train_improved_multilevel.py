#!/usr/bin/env python3
"""
Improved Multi-level MobileNetV3 Training Script
改进版多级MobileNetV3训练脚本
"""

import os
import sys
import json
import logging
import argparse
import torch
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
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

def main():
    parser = argparse.ArgumentParser(description='Improved Multi-level MobileNetV3 Training')
    
    # 数据参数
    parser.add_argument('--json_path', type=str, 
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='Path to dataset JSON file')
    parser.add_argument('--image_root', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images',
                       help='Root directory for images')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'large'], help='MobileNetV3 model size')
    parser.add_argument('--input_channels', type=int, default=1,
                       help='Number of input channels')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (increased from 32)')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='Initial learning rate (increased from 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (reduced from 15)')
    
    # 实验参数
    parser.add_argument('--experiment_dir', type=str,
                       default='experiments/improved_multilevel',
                       help='Experiment directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate the model')
    
    # 数据分割参数
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test data ratio')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建实验目录
    experiment_dir = args.experiment_dir
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(experiment_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Improved Multi-level MobileNetV3 Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Arguments: {vars(args)}")
    
    # 保存配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    try:
        split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)
        
        train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
            json_path=args.json_path,
            image_root=args.image_root,
            batch_size=args.batch_size,
            split_ratio=split_ratio
        )
        
        logger.info(f"Data loaders created successfully")
        logger.info(f"Train batches: {len(train_loader)} (samples: {len(train_loader.dataset)})")
        logger.info(f"Val batches: {len(val_loader)} (samples: {len(val_loader.dataset)})")
        logger.info(f"Test batches: {len(test_loader)} (samples: {len(test_loader.dataset)})")
        
        # 保存标签信息
        label_info_path = os.path.join(experiment_dir, 'label_info.json')
        with open(label_info_path, 'w') as f:
            json.dump(label_info, f, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # 创建模型
    logger.info("Creating improved model...")
    try:
        model = create_multilevel_mobilenetv3(
            model_size=args.model_size,
            input_channels=args.input_channels,
            dropout_rate=args.dropout_rate
        )
        
        model_info = model.get_model_info()
        logger.info("Improved model created successfully")
        logger.info(f"Model info: {model_info}")
        logger.info(f"Task weights: {model.task_weights}")
        
        # 保存模型信息
        model_info_path = os.path.join(experiment_dir, 'model_info.json')
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return
    
    # 创建改进版训练器
    logger.info("Creating improved trainer...")
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
        
        logger.info("Improved trainer created successfully")
        logger.info(f"Warmup epochs: {args.warmup_epochs}")
        logger.info(f"Patience: {args.patience}")
        logger.info(f"Initial learning rate: {args.learning_rate}")
        
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        return
    
    # 恢复检查点（如果指定）
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
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
            return
    
    # 训练或评估
    if args.eval_only:
        logger.info("Evaluation mode")
        try:
            results = trainer.evaluate()
            logger.info("Evaluation completed successfully")
            logger.info(f"Test results: {results}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    else:
        logger.info("Training mode")
        try:
            # 训练模型
            history = trainer.train(num_epochs=args.num_epochs, save_best=True)
            
            logger.info("Training completed successfully")
            logger.info(f"Best validation accuracy: {trainer.best_val_accuracy:.4f} at epoch {trainer.best_epoch+1}")
            
            # 评估最佳模型
            logger.info("Evaluating best model...")
            
            # 加载最佳模型
            best_model_path = os.path.join(experiment_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Best model loaded for evaluation")
            
            results = trainer.evaluate()
            logger.info("Final evaluation completed")
            
            # 绘制训练曲线
            trainer.plot_training_curves()
            logger.info("Training curves saved")
            
            # 保存最终结果摘要
            summary = {
                'best_val_accuracy': trainer.best_val_accuracy,
                'best_epoch': trainer.best_epoch + 1,
                'total_epochs': len(history['train_loss']),
                'final_results': results,
                'config': vars(args)
            }
            
            summary_path = os.path.join(experiment_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Training summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()