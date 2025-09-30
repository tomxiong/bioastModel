#!/usr/bin/env python3
"""
Multi-level MobileNetV3 Training Script
多层分类MobileNetV3训练脚本

用于训练基于MobileNetV3的细菌图像四层分类模型：
1. microbe_type (目前只有bacteria，可跳过)
2. growth_level (positive/negative)
3. growth_pattern (12种模式)
4. interference_factors (5种干扰因子，多标签)
"""

import os
import sys
import argparse
import logging
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
from training.multilevel_dataset import create_multilevel_dataloaders
from training.multilevel_trainer import MultiLevelTrainer

def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multi-level MobileNetV3 Training')
    
    # 数据参数
    parser.add_argument('--json_path', type=str, 
                       default='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
                       help='Path to JSON annotation file')
    parser.add_argument('--image_root', type=str,
                       default='/home/aaa/ws/bioastModel/ds/images',
                       help='Root directory of images')
    
    # 模型参数
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'large'],
                       help='MobileNetV3 model size')
    parser.add_argument('--input_channels', type=int, default=1,
                       help='Number of input channels (1 for grayscale)')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # 数据分割参数
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio')
    
    # 实验参数
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Experiments root directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate the model')
    
    return parser.parse_args()

def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def get_device(device_arg: str) -> torch.device:
    """获取设备"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    return device

def create_experiment_dir(args) -> str:
    """创建实验目录"""
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"multilevel_mobilenetv3_{args.model_size}_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join(args.experiment_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir

def save_config(args, experiment_dir: str):
    """保存配置"""
    config = vars(args)
    config_path = os.path.join(experiment_dir, 'config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建实验目录
    experiment_dir = create_experiment_dir(args)
    
    # 设置日志
    logger = setup_logging(experiment_dir)
    logger.info("Starting multi-level MobileNetV3 training")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # 保存配置
    save_config(args, experiment_dir)
    
    # 设置设备
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        cudnn.benchmark = True
    
    # 检查数据文件
    if not os.path.exists(args.json_path):
        logger.error(f"JSON file not found: {args.json_path}")
        return
    
    if not os.path.exists(args.image_root):
        logger.error(f"Image root directory not found: {args.image_root}")
        return
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)
    
    try:
        train_loader, val_loader, test_loader, label_info = create_multilevel_dataloaders(
            json_path=args.json_path,
            image_root=args.image_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_ratio=split_ratio
        )
        
        logger.info(f"Data loaders created successfully")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")
        
        # 保存标签信息
        label_info_path = os.path.join(experiment_dir, 'label_info.json')
        with open(label_info_path, 'w') as f:
            json.dump(label_info, f, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        return
    
    # 创建模型
    logger.info("Creating model...")
    try:
        model = create_multilevel_mobilenetv3(
            model_size=args.model_size,
            input_channels=args.input_channels,
            dropout_rate=args.dropout_rate,
            freeze_backbone=args.freeze_backbone
        )
        
        model_info = model.get_model_info()
        logger.info("Model created successfully")
        logger.info(f"Model info: {model_info}")
        
        # 保存模型信息
        model_info_path = os.path.join(experiment_dir, 'model_info.json')
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return
    
    # 创建训练器
    logger.info("Creating trainer...")
    try:
        trainer = MultiLevelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            label_info=label_info,
            device=device,
            experiment_dir=experiment_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        logger.info("Trainer created successfully")
        
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
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
            history = trainer.train(num_epochs=args.num_epochs)
            logger.info("Training completed successfully")
            
            # 评估模型
            logger.info("Starting final evaluation...")
            results = trainer.evaluate()
            logger.info("Final evaluation completed")
            
            # 绘制训练曲线
            trainer.plot_training_curves()
            logger.info("Training curves saved")
            
            # 打印最终结果
            logger.info("=== Final Results ===")
            for task, task_results in results.items():
                logger.info(f"{task}: {task_results}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("Program completed")

if __name__ == "__main__":
    main()