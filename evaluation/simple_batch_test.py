import os
import sys
import yaml
import torch
import numpy as np
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.faster_vit import create_faster_vit
from data.dataset import create_dataloaders
from evaluation.metrics import evaluate_model, print_evaluation_summary, check_performance_requirements
from evaluation.visualizer import create_comprehensive_report, SampleVisualizer

def setup_logging(log_dir='results/logs'):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'batch_test_{timestamp}.log')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger

def load_model(model_path, config, device):
    """加载训练好的模型"""
    logger = logging.getLogger(__name__)
    model = create_faster_vit(config['model'])
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"成功加载模型: {model_path}")
    else:
        logger.error(f"模型文件不存在: {model_path}")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model.to(device)
    model.eval()
    return model

def standard_batch_test(config_path, model_path=None):
    """标准批量测试"""
    logger = logging.getLogger(__name__)
    logger.info("开始标准批量测试...")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['device']['device_id']}")
    else:
        device = torch.device('cpu')
    
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['dataset_path'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        use_weighted_sampling=config['data']['use_weighted_sampling']
    )
    
    # 加载模型
    if model_path is None:
        model_path = os.path.join(config['output']['checkpoints_dir'], 'best_model.pth')
    
    model = load_model(model_path, config, device)
    
    # 测试集评估
    logger.info("在测试集上评估模型...")
    test_metrics, test_calculator = evaluate_model(
        model, test_loader, device, ['Negative', 'Positive']
    )
    
    # 打印评估结果
    print_evaluation_summary(test_metrics)
    
    # 检查性能要求
    requirements = check_performance_requirements(test_metrics)
    
    # 创建可视化报告
    logger.info("生成可视化报告...")
    
    # 准备数据
    cm = test_calculator.get_confusion_matrix()
    fpr, tpr, _ = test_calculator.get_roc_data()
    roc_auc = test_metrics.get('roc_auc', 0)
    
    # 创建报告
    create_comprehensive_report(
        metrics=test_metrics,
        history=None,  # 批量测试时不需要历史数据
        confusion_matrix_data=(cm, ['Negative', 'Positive']),
        roc_data=(fpr, tpr, roc_auc),
        save_dir=config['output']['plots_dir']
    )
    
    # 可视化预测样本
    sample_viz = SampleVisualizer(config['output']['plots_dir'])
    sample_viz.visualize_predictions(
        model, test_loader, device, 
        num_samples=config.get('visualization', {}).get('num_sample_plots', 16),
        class_names=['Negative', 'Positive']
    )
    
    # 可视化错误案例
    sample_viz.visualize_error_cases(
        model, test_loader, device,
        class_names=['Negative', 'Positive'],
        max_samples=16
    )
    
    # 保存结果
    results = {
        'test_metrics': test_metrics,
        'performance_requirements': requirements,
        'config': config,
        'test_type': 'standard_batch_test'
    }
    
    results_path = os.path.join(config['output']['reports_dir'], 'simple_batch_test_results.yaml')
    with open(results_path, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"批量测试结果保存到: {results_path}")
    
    return test_metrics

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='微生物菌落Faster ViT模型简化批量测试')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型文件路径（可选）')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("开始简化批量测试...")
    
    try:
        standard_batch_test(args.config, args.model_path)
        logger.info("批量测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()