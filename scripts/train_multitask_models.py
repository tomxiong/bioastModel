#!/usr/bin/env python3
"""
多任务训练和评估脚本
整合了多任务模型训练、评估和可视化的完整流程
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.multitask_dataset import create_multitask_dataloaders
from training.multitask_trainer import MultitaskTrainer
from models.multitask_models import create_multitask_model, get_multitask_model_config
from evaluation.multitask_evaluator import MultitaskEvaluator, compare_multitask_models
from core.config.model_configs import get_model_config


def setup_logging(log_dir: str):
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'multitask_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_multitask_model(config: dict, logger: logging.Logger):
    """训练多任务模型"""
    logger.info("开始多任务模型训练...")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    dataloaders = create_multitask_dataloaders(
        annotation_file=config['annotation_file'],
        image_root=config['image_root'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)
    )
    
    # 创建模型
    logger.info(f"创建模型: {config['model_name']}")
    model_config = get_multitask_model_config(config['model_name'])
    model = create_multitask_model(**model_config)
    
    # 打印模型信息
    model_info = model.get_task_info()
    logger.info(f"模型参数量: {model_info['total_params']:,}")
    
    # 创建训练器
    trainer = MultitaskTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        config=config
    )
    
    # 开始训练
    logger.info("开始训练...")
    training_results = trainer.train()
    
    # 保存最终模型
    final_model_path = f"checkpoints/multitask_{config['model_name']}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_results': training_results
    }, final_model_path)
    
    logger.info(f"模型已保存: {final_model_path}")
    
    return model, dataloaders, training_results


def evaluate_multitask_model(model, dataloaders, config: dict, logger: logging.Logger):
    """评估多任务模型"""
    logger.info("开始多任务模型评估...")
    
    # 获取任务信息
    dataset_info = dataloaders['dataset_info']
    
    # 创建评估器
    evaluator = MultitaskEvaluator(
        model=model,
        task_info=dataset_info,
        class_mappings=dataset_info['mappings'],
        save_dir=f"evaluation_results/multitask_{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 执行评估
    evaluation_results = evaluator.evaluate(dataloaders['test'])
    
    # 打印主要结果
    logger.info("\n=== 评估结果 ===")
    logger.info(f"综合得分: {evaluation_results['composite_score']:.4f}")
    
    for task_name in dataset_info['task_names']:
        if task_name in evaluation_results:
            result = evaluation_results[task_name]
            if result['task_type'] == 'single_label':
                logger.info(f"{task_name}: F1={result['f1_score']:.4f}, Acc={result['accuracy']:.4f}")
            else:
                logger.info(f"{task_name}: F1_micro={result['f1_micro']:.4f}")
    
    return evaluation_results


def create_experiment_config(args):
    """创建实验配置"""
    config = {
        # 数据配置
        'annotation_file': args.annotation_file,
        'image_root': args.image_root,
        
        # 模型配置
        'model_name': args.model_name,
        
        # 训练配置
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'use_amp': args.use_amp,
        'gradient_clip': args.gradient_clip,
        'num_workers': args.num_workers,
        
        # 任务权重
        'task_weights': {
            'growth_level': args.growth_level_weight,
            'growth_pattern': args.growth_pattern_weight,
            'interference_mapping': args.interference_weight,
            'fine_grained': args.fine_grained_weight
        },
        
        # 保存配置
        'log_dir': f"runs/multitask_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'checkpoint_dir': 'checkpoints'
    }
    
    return config


def main():
    parser = argparse.ArgumentParser(description='多任务生物图像分类训练和评估')
    
    # 数据参数
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='标注文件路径')
    parser.add_argument('--image_root', type=str, required=True,
                       help='图像根目录')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='multitask_airbubble_hybrid',
                       choices=['multitask_airbubble_hybrid', 'multitask_resnet18', 
                               'multitask_efficientnet', 'hierarchical_airbubble'],
                       help='模型名称')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='梯度裁剪阈值')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    
    # 任务权重
    parser.add_argument('--growth_level_weight', type=float, default=1.0,
                       help='生长级别任务权重')
    parser.add_argument('--growth_pattern_weight', type=float, default=1.0,
                       help='生长模式任务权重')
    parser.add_argument('--interference_weight', type=float, default=0.5,
                       help='干扰因素任务权重')
    parser.add_argument('--fine_grained_weight', type=float, default=1.0,
                       help='精细分类任务权重')
    
    # 运行模式
    parser.add_argument('--mode', choices=['train', 'eval', 'both'], default='both',
                       help='运行模式')
    parser.add_argument('--model_path', type=str,
                       help='预训练模型路径（评估模式使用）')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir / 'logs')
    
    # 创建配置
    config = create_experiment_config(args)
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("实验配置:")
    logger.info(json.dumps(config, indent=2))
    
    # 创建必要的目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    
    model = None
    dataloaders = None
    
    try:
        if args.mode in ['train', 'both']:
            # 训练模型
            model, dataloaders, training_results = train_multitask_model(config, logger)
        
        if args.mode in ['eval', 'both']:
            # 加载模型用于评估
            if args.mode == 'eval':
                logger.info(f"加载预训练模型: {args.model_path}")
                checkpoint = torch.load(args.model_path, map_location='cpu')
                model_config = get_multitask_model_config(config['model_name'])
                model = create_multitask_model(**model_config)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # 创建数据加载器
                dataloaders = create_multitask_dataloaders(
                    annotation_file=config['annotation_file'],
                    image_root=config['image_root'],
                    batch_size=config['batch_size'],
                    num_workers=config['num_workers']
                )
            
            # 评估模型
            evaluation_results = evaluate_multitask_model(model, dataloaders, config, logger)
            
            # 保存评估结果
            with open(output_dir / 'evaluation_results.json', 'w') as f:
                json.dump(evaluation_results, f, indent=2)
    
    except Exception as e:
        logger.error(f"运行出错: {e}", exc_info=True)
        return 1
    
    logger.info("完成!")
    return 0


def run_multitask_experiments():
    """运行多个多任务实验用于模型比较"""
    experiments = [
        {
            'name': 'airbubble_multitask',
            'model_name': 'multitask_airbubble_hybrid',
            'config': {
                'annotation_file': 'bioast_dataset/annotations/multitask_annotations.json',
                'image_root': 'bioast_dataset/images',
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'use_amp': True
            }
        },
        {
            'name': 'resnet_multitask',
            'model_name': 'multitask_resnet18',
            'config': {
                'annotation_file': 'bioast_dataset/annotations/multitask_annotations.json',
                'image_root': 'bioast_dataset/images',
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'use_amp': True
            }
        },
        {
            'name': 'efficientnet_multitask',
            'model_name': 'multitask_efficientnet',
            'config': {
                'annotation_file': 'bioast_dataset/annotations/multitask_annotations.json',
                'image_root': 'bioast_dataset/images',
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'use_amp': True
            }
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"运行实验: {exp['name']}")
        print(f"{'='*60}")
        
        # 设置输出目录
        output_dir = Path(f"experiments/{exp['name']}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        logger = setup_logging(output_dir / 'logs')
        
        try:
            # 训练和评估
            model, dataloaders, training_results = train_multitask_model(exp['config'], logger)
            evaluation_results = evaluate_multitask_model(model, dataloaders, exp['config'], logger)
            
            # 保存结果
            results[exp['name']] = {
                'training_results': training_results,
                'evaluation_results': evaluation_results
            }
            
            # 保存完整结果
            with open(output_dir / 'full_results.json', 'w') as f:
                json.dump(results[exp['name']], f, indent=2)
            
        except Exception as e:
            logger.error(f"实验 {exp['name']} 失败: {e}", exc_info=True)
            results[exp['name']] = {'error': str(e)}
    
    # 模型比较
    print(f"\n{'='*60}")
    print("模型比较")
    print(f"{'='*60}")
    
    comparison_results = {}
    for exp_name, exp_results in results.items():
        if 'error' not in exp_results:
            comparison_results[exp_name] = exp_results['evaluation_results']
    
    if comparison_results:
        comparison_df = compare_multitask_models(comparison_results, 
                                               save_dir='experiments/model_comparison')
        print("\n模型比较结果:")
        print(comparison_df.to_string(index=False))
    
    return results


if __name__ == "__main__":
    # 单次运行
    # exit(main())
    
    # 或者运行多个实验进行比较
    run_multitask_experiments()