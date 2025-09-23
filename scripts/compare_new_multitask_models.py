#!/usr/bin/env python3
"""
新模型性能对比脚本
比较最新添加的MobileNetV3和EfficientNet模型在多任务学习中的性能
"""

import os
import sys
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_models import create_multitask_model, get_multitask_model_config
from training.multitask_dataset import create_multitask_dataloaders
from evaluation.multitask_evaluator import MultitaskEvaluator


def setup_logging(save_dir: str):
    """设置日志"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'model_comparison.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def benchmark_model_performance(model_name: str, 
                            dataloaders: Dict,
                            logger: logging.Logger,
                            device: torch.device) -> Dict[str, Any]:
    """基准测试模型性能"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试模型: {model_name}")
    logger.info(f"{'='*60}")
    
    # 获取模型配置
    config = get_multitask_model_config(model_name)
    logger.info(f"模型描述: {config['description']}")
    
    # 创建模型
    try:
        if config['model_type'] == 'enhanced':
            # 增强模型特殊处理
            from models.enhanced_multitask_mobilenetv3 import create_enhanced_multitask_mobilenetv3
            model = create_enhanced_multitask_mobilenetv3(
                growth_level_classes=3,
                growth_pattern_classes=9,
                interference_classes=3,
                fine_grained_classes=40
            )
        else:
            model = create_multitask_model(
                model_type=config['model_type'],
                backbone_name=config['backbone_name'],
                feature_dim=config['feature_dim'],
                dropout_rate=config['dropout_rate'],
                use_attention=config['use_attention'],
                task_configs={
                    'growth_level': {'num_classes': 3},
                    'growth_pattern': {'num_classes': 9},
                    'interference_mapping': {'num_classes': 3, 'multilabel': True},
                    'fine_grained': {'num_classes': 40}
                }
            )
    except Exception as e:
        logger.error(f"创建模型失败: {e}")
        return {'error': str(e)}
    
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    
    # 测试推理速度
    model.eval()
    dummy_input = torch.randn(1, 3, 70, 70).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 测量推理时间
    times = []
    with torch.no_grad():
        for _ in range(100):
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_inference_time = np.mean(times) * 1000  # 转换为毫秒
    std_inference_time = np.std(times) * 1000
    
    logger.info(f"平均推理时间: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
    
    # 模型大小估算
    model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
    logger.info(f"模型大小估算: {model_size_mb:.2f} MB")
    
    # 评估模型性能
    try:
        evaluator = MultitaskEvaluator(
            model=model,
            task_info=dataloaders['dataset_info'],
            class_mappings=dataloaders['dataset_info']['mappings'],
            save_dir=f"temp_eval_{model_name}"
        )
        
        evaluation_results = evaluator.evaluate(dataloaders['test'])
        
        # 清理临时目录
        import shutil
        shutil.rmtree(f"temp_eval_{model_name}", ignore_errors=True)
        
        results = {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'evaluation_results': evaluation_results
        }
        
        # 提取关键指标
        if 'composite_score' in evaluation_results:
            results['composite_score'] = evaluation_results['composite_score']
        
        # 提取各任务F1分数
        task_f1_scores = {}
        for task_name, task_result in evaluation_results.items():
            if isinstance(task_result, dict) and 'f1_score' in task_result:
                task_f1_scores[task_name] = task_result['f1_score']
            elif isinstance(task_result, dict) and 'f1_micro' in task_result:
                task_f1_scores[task_name] = task_result['f1_micro']
        
        results['task_f1_scores'] = task_f1_scores
        
        logger.info(f"综合得分: {results.get('composite_score', 0):.4f}")
        for task, f1 in task_f1_scores.items():
            logger.info(f"{task} F1分数: {f1:.4f}")
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        results = {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'error': f"Evaluation failed: {str(e)}"
        }
    
    return results


def create_comparison_report(results: List[Dict[str, Any]], output_dir: str):
    """创建对比报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    comparison_data = []
    for result in results:
        if 'error' not in result:
            row = {
                'Model': result['model_name'],
                'Parameters (M)': result['total_params'] / 1e6,
                'Model Size (MB)': result['model_size_mb'],
                'Inference Time (ms)': result['avg_inference_time_ms'],
                'Composite Score': result.get('composite_score', 0)
            }
            
            # 添加任务F1分数
            for task, f1 in result.get('task_f1_scores', {}).items():
                row[f'{task} F1'] = f1
            
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # 保存CSV
    csv_path = output_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n对比报告已保存: {csv_path}")
    
    # 打印表格
    print("\n" + "="*80)
    print("模型性能对比")
    print("="*80)
    print(df.to_string(index=False, float_format="%.3f"))
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # 1. 参数量vs推理时间
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='Parameters (M)', y='Inference Time (ms)', 
                    hue='Model', s=100, alpha=0.7)
    plt.title('Model Size vs Inference Time')
    plt.xlabel('Parameters (Millions)')
    plt.ylabel('Inference Time (ms)')
    
    # 2. 综合得分对比
    plt.subplot(2, 2, 2)
    sns.barplot(data=df, x='Model', y='Composite Score')
    plt.title('Composite Score Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # 3. 任务F1分数热力图
    plt.subplot(2, 2, 3)
    task_cols = [col for col in df.columns if 'F1' in col]
    if task_cols:
        task_df = df[['Model'] + task_cols].set_index('Model')
        sns.heatmap(task_df, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Task F1 Scores Heatmap')
        plt.xticks(rotation=45)
    
    # 4. 模型效率（得分/参数量）
    plt.subplot(2, 2, 4)
    df['Efficiency'] = df['Composite Score'] / df['Parameters (M)']
    sns.barplot(data=df, x='Model', y='Efficiency')
    plt.title('Model Efficiency (Score/Parameter)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存: {output_dir / 'model_comparison.png'}")
    
    # 生成详细报告
    report_path = output_dir / 'detailed_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 新模型性能对比报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 概述\n\n")
        f.write("本报告对比了最新添加的MobileNetV3和EfficientNet模型在多任务学习中的性能。\n\n")
        
        f.write("## 关键发现\n\n")
        
        # 最佳综合性能
        best_composite = df.loc[df['Composite Score'].idxmax()]
        f.write(f"- **最佳综合性能**: {best_composite['Model']} (得分: {best_composite['Composite Score']:.3f})\n")
        
        # 最快推理速度
        fastest = df.loc[df['Inference Time (ms)'].idxmin()]
        f.write(f"- **最快推理速度**: {fastest['Model']} ({fastest['Inference Time (ms)']:.2f} ms)\n")
        
        # 最小模型
        smallest = df.loc[df['Parameters (M)'].idxmin()]
        f.write(f"- **最小模型**: {smallest['Model']} ({smallest['Parameters (M)']:.2f}M 参数)\n")
        
        # 最高效率
        most_efficient = df.loc[df['Efficiency'].idxmax()]
        f.write(f"- **最高效率**: {most_efficient['Model']} (效率: {most_efficient['Efficiency']:.3f})\n\n")
        
        f.write("## 详细结果\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        
        f.write("\n\n## 建议\n\n")
        f.write("- **实时应用**: 推荐使用推理速度最快的模型\n")
        f.write("- **边缘部署**: 推荐使用参数量最小的模型\n")
        f.write("- **精度优先**: 推荐使用综合得分最高的模型\n")
        f.write("- **平衡选择**: 推荐使用效率最高的模型\n")
    
    print(f"详细报告已保存: {report_path}")


def main():
    """主函数"""
    print("=== 新模型性能对比 ===")
    
    # 数据路径
    annotation_file = "bioast_dataset/annotations/multitask_annotations.json"
    image_root = "bioast_dataset/images"
    
    # 检查数据是否存在
    if not Path(annotation_file).exists():
        print(f"错误: 标注文件不存在 {annotation_file}")
        print("请先运行数据转换脚本生成多任务标注数据")
        return
    
    # 创建输出目录
    output_dir = Path("model_comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    try:
        dataloaders = create_multitask_dataloaders(
            annotation_file=annotation_file,
            image_root=image_root,
            batch_size=16,
            num_workers=2
        )
    except Exception as e:
        logger.error(f"创建数据加载器失败: {e}")
        return
    
    # 要测试的模型列表
    models_to_test = [
        'multitask_airbubble_hybrid',  # 基准模型
        'multitask_mobilenetv3_large',
        'multitask_mobilenetv3_small',
        'multitask_efficientnet_v2_s',
        'multitask_efficientnet_v2_b0',
        'multitask_mic_mobilenetv3',
        'enhanced_multitask_mobilenetv3'
    ]
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 收集结果
    all_results = []
    
    for model_name in models_to_test:
        result = benchmark_model_performance(
            model_name=model_name,
            dataloaders=dataloaders,
            logger=logger,
            device=device
        )
        all_results.append(result)
    
    # 保存原始结果
    with open(output_dir / 'raw_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 创建对比报告
    create_comparison_report(all_results, output_dir)
    
    logger.info(f"\n所有结果已保存到: {output_dir}")
    print(f"\n对比完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()