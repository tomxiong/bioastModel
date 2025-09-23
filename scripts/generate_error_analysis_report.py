#!/usr/bin/env python3
"""
生成错误样本分析报告
分析多任务模型的错误模式和样本特征
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from models.fixed_efficientnet_b0_multitask import create_fixed_efficientnet_b0_multitask
from models.resnet34_multitask import create_resnet34_multitask
from models.fixed_mobilenetv3_multitask import create_fixed_mobilenetv3_multitask

# 导入数据集
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_checkpoint(model_type, checkpoint_path, num_classes):
    """加载模型和检查点"""
    if model_type == 'fixed_efficientnet_b0':
        model = create_fixed_efficientnet_b0_multitask(num_classes)
    elif model_type == 'resnet34':
        model = create_resnet34_multitask(num_classes)
    elif model_type == 'fixed_mobilenetv3':
        model = create_fixed_mobilenetv3_multitask(num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def predict_batch(model, dataloader, device='cpu'):
    """批量预测"""
    model.to(device)
    model.eval()
    
    all_predictions = {
        'growth_level': [],
        'growth_pattern': [], 
        'interference_factors': [],
        'microbe_type': []
    }
    all_targets = {
        'growth_level': [],
        'growth_pattern': [],
        'interference_factors': [],
        'microbe_type': []
    }
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # 获取预测
            predictions = model(images)
            
            # 转换为概率和预测类别
            for task in all_predictions.keys():
                if task in predictions:
                    pred = predictions[task]
                    if task == 'interference_factors':
                        # 多标签任务
                        pred_probs = torch.sigmoid(pred)
                        pred_classes = (pred_probs > 0.5).float()
                    else:
                        # 单标签分类任务
                        pred_probs = F.softmax(pred, dim=1)
                        pred_classes = torch.argmax(pred_probs, dim=1)
                    
                    all_predictions[task].extend(pred_classes.cpu().numpy())
                    
                    # 获取真实标签
                    if task in batch:
                        targets = batch[task]
                        if task == 'interference_factors':
                            all_targets[task].extend(targets.cpu().numpy())
                        else:
                            all_targets[task].extend(targets.cpu().numpy())
            
            # 记录样本索引
            batch_size = images.size(0)
            start_idx = batch_idx * dataloader.batch_size
            indices = list(range(start_idx, start_idx + batch_size))
            all_indices.extend(indices)
    
    return all_predictions, all_targets, all_indices

def analyze_errors_by_task(predictions, targets, task_name):
    """分析特定任务的错误"""
    preds = np.array(predictions[task_name])
    targs = np.array(targets[task_name])
    
    if len(preds) == 0 or len(targs) == 0:
        return {}
    
    if task_name == 'interference_factors':
        # 多标签任务分析
        correct = (preds == targs).all(axis=1)
        accuracy = correct.mean()
        
        # 每个标签的准确率
        label_accuracies = []
        for i in range(targs.shape[1]):
            label_acc = (preds[:, i] == targs[:, i]).mean()
            label_accuracies.append(label_acc)
        
        return {
            'task_name': task_name,
            'accuracy': accuracy,
            'label_accuracies': label_accuracies,
            'error_indices': np.where(~correct)[0].tolist(),
            'correct_indices': np.where(correct)[0].tolist(),
            'error_rate': 1 - accuracy
        }
    else:
        # 单标签分类任务分析
        correct = (preds == targs)
        accuracy = correct.mean()
        
        # 混淆矩阵数据
        unique_classes = np.unique(np.concatenate([preds, targs]))
        confusion_data = []
        
        for true_class in unique_classes:
            for pred_class in unique_classes:
                count = np.sum((targs == true_class) & (preds == pred_class))
                confusion_data.append({
                    'true_class': int(true_class),
                    'pred_class': int(pred_class), 
                    'count': int(count)
                })
        
        return {
            'task_name': task_name,
            'accuracy': accuracy,
            'confusion_data': confusion_data,
            'error_indices': np.where(~correct)[0].tolist(),
            'correct_indices': np.where(correct)[0].tolist(),
            'error_rate': 1 - accuracy,
            'num_classes': len(unique_classes)
        }

def create_confusion_matrix_plot(confusion_data, task_name, save_path):
    """创建混淆矩阵图"""
    if not confusion_data:
        return
    
    # 构建混淆矩阵
    df = pd.DataFrame(confusion_data)
    matrix = df.pivot(index='true_class', columns='pred_class', values='count').fillna(0)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=True, yticklabels=True)
    plt.title(f'{task_name} 混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_error_patterns(all_errors, sample_indices):
    """分析错误模式"""
    error_analysis = {}
    
    # 统计每个样本的错误任务数
    sample_error_count = {}
    for task, error_indices in all_errors.items():
        for idx in error_indices:
            if idx not in sample_error_count:
                sample_error_count[idx] = []
            sample_error_count[idx].append(task)
    
    # 分析错误分布
    error_distribution = {
        'single_task_errors': 0,
        'multi_task_errors': 0,
        'all_task_errors': 0,
        'error_combinations': {}
    }
    
    for idx, error_tasks in sample_error_count.items():
        num_errors = len(error_tasks)
        if num_errors == 1:
            error_distribution['single_task_errors'] += 1
        elif num_errors > 1:
            error_distribution['multi_task_errors'] += 1
            if num_errors == len(all_errors):
                error_distribution['all_task_errors'] += 1
            
            # 记录错误组合
            error_combo = tuple(sorted(error_tasks))
            if error_combo not in error_distribution['error_combinations']:
                error_distribution['error_combinations'][error_combo] = 0
            error_distribution['error_combinations'][error_combo] += 1
    
    return error_distribution, sample_error_count

def generate_error_analysis_report(model_results, save_dir):
    """生成错误分析报告"""
    report = []
    report.append("# 多任务模型错误样本分析报告")
    report.append("")
    report.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 总体概览
    report.append("## 📊 错误分析概览")
    report.append("")
    
    for model_name, results in model_results.items():
        if not results:
            continue
            
        report.append(f"### {model_name}")
        report.append("")
        
        total_samples = len(results.get('sample_indices', []))
        error_samples = len(results.get('error_distribution', {}).get('sample_error_count', {}))
        overall_error_rate = error_samples / total_samples if total_samples > 0 else 0
        
        report.append(f"- **总样本数:** {total_samples}")
        report.append(f"- **错误样本数:** {error_samples}")
        report.append(f"- **总体错误率:** {overall_error_rate:.2%}")
        report.append("")
        
        # 任务级别错误分析
        report.append("**各任务错误情况:**")
        report.append("")
        
        for task_result in results.get('task_results', []):
            task_name = task_result['task_name']
            accuracy = task_result['accuracy']
            error_rate = task_result['error_rate']
            error_count = len(task_result['error_indices'])
            
            report.append(f"- **{task_name}:**")
            report.append(f"  - 准确率: {accuracy:.2%}")
            report.append(f"  - 错误率: {error_rate:.2%}")
            report.append(f"  - 错误样本数: {error_count}")
            report.append("")
        
        # 错误模式分析
        error_dist = results.get('error_distribution', {})
        report.append("**错误模式分析:**")
        report.append("")
        report.append(f"- 单任务错误样本: {error_dist.get('single_task_errors', 0)}")
        report.append(f"- 多任务错误样本: {error_dist.get('multi_task_errors', 0)}")
        report.append(f"- 全任务错误样本: {error_dist.get('all_task_errors', 0)}")
        report.append("")
        
        # 错误组合分析
        error_combos = error_dist.get('error_combinations', {})
        if error_combos:
            report.append("**常见错误组合:**")
            report.append("")
            sorted_combos = sorted(error_combos.items(), key=lambda x: x[1], reverse=True)
            for combo, count in sorted_combos[:5]:  # 显示前5个最常见的组合
                tasks = " + ".join(combo)
                report.append(f"- {tasks}: {count} 个样本")
            report.append("")
    
    # 模型对比
    report.append("## 🔄 模型错误率对比")
    report.append("")
    
    report.append("| 模型 | 总体错误率 | Growth Level | Growth Pattern | Interference Factors | Microbe Type |")
    report.append("|------|------------|--------------|----------------|---------------------|--------------|")
    
    for model_name, results in model_results.items():
        if not results:
            continue
            
        total_samples = len(results.get('sample_indices', []))
        error_samples = len(results.get('error_distribution', {}).get('sample_error_count', {}))
        overall_error_rate = error_samples / total_samples if total_samples > 0 else 0
        
        task_errors = {}
        for task_result in results.get('task_results', []):
            task_errors[task_result['task_name']] = task_result['error_rate']
        
        report.append(f"| {model_name} | {overall_error_rate:.1%} | "
                     f"{task_errors.get('growth_level', 0):.1%} | "
                     f"{task_errors.get('growth_pattern', 0):.1%} | "
                     f"{task_errors.get('interference_factors', 0):.1%} | "
                     f"{task_errors.get('microbe_type', 0):.1%} |")
    
    report.append("")
    
    # 改进建议
    report.append("## 💡 改进建议")
    report.append("")
    report.append("基于错误分析结果，建议采取以下优化策略：")
    report.append("")
    report.append("### 1. 数据层面优化")
    report.append("- 针对高错误率任务增加相应类别的训练样本")
    report.append("- 实施数据增强策略，特别是对困难样本")
    report.append("- 考虑样本重采样来平衡类别分布")
    report.append("")
    
    report.append("### 2. 模型层面优化")
    report.append("- 调整损失函数权重，重点优化高错误率任务")
    report.append("- 实施任务特定的正则化策略")
    report.append("- 考虑使用注意力机制提高特征学习")
    report.append("")
    
    report.append("### 3. 训练策略优化")
    report.append("- 实施课程学习，从简单样本开始训练")
    report.append("- 使用困难样本挖掘技术")
    report.append("- 考虑多阶段训练策略")
    report.append("")
    
    # 保存报告
    report_path = os.path.join(save_dir, "error_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return report_path

def main():
    print("🔍 开始生成错误样本分析报告")
    print("=" * 60)
    
    # 数据集配置
    data_root = "ds/images"
    num_classes = {
        'growth_level': 2,
        'growth_pattern': 12, 
        'interference_factors': 4,
        'microbe_type': 4
    }
    
    # 创建测试数据集
    print("加载测试数据集...")
    test_dataset = EnhancedMultitaskDataset(
        data_root=data_root,
        split='test',
        split_ratio=(0.7, 0.15, 0.15)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"测试集样本数: {len(test_dataset)}")
    
    # 要分析的模型
    models_to_analyze = [
        {
            'name': 'MobileNetV3',
            'type': 'fixed_mobilenetv3',
            'checkpoint': 'experiments/fixed_mobilenetv3_multitask_20250919_025346/best.pth'
        },
        {
            'name': 'ResNet-34', 
            'type': 'resnet34',
            'checkpoint': 'experiments/resnet34_gpu_optimized_20250919_021208/best.pth'
        },
        {
            'name': 'EfficientNet-B0',
            'type': 'fixed_efficientnet_b0', 
            'checkpoint': 'experiments/fixed_efficientnet_b0_multitask_20250919_020142/best.pth'
        }
    ]
    
    # 创建输出目录
    os.makedirs("reports", exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model_results = {}
    
    for model_info in models_to_analyze:
        model_name = model_info['name']
        model_type = model_info['type']
        checkpoint_path = model_info['checkpoint']
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ 检查点不存在: {checkpoint_path}")
            model_results[model_name] = None
            continue
        
        print(f"\n分析模型: {model_name}")
        print(f"检查点: {checkpoint_path}")
        
        try:
            # 加载模型
            model = load_model_and_checkpoint(model_type, checkpoint_path, num_classes)
            
            # 预测
            print("执行预测...")
            predictions, targets, sample_indices = predict_batch(model, test_loader, device)
            
            # 分析各任务错误
            task_results = []
            all_errors = {}
            
            for task in ['growth_level', 'growth_pattern', 'interference_factors', 'microbe_type']:
                if task in predictions and task in targets:
                    task_analysis = analyze_errors_by_task(predictions, targets, task)
                    if task_analysis:
                        task_results.append(task_analysis)
                        all_errors[task] = task_analysis['error_indices']
                        
                        # 生成混淆矩阵图（单标签任务）
                        if 'confusion_data' in task_analysis:
                            confusion_plot_path = f"reports/{model_name}_{task}_confusion_matrix.png"
                            create_confusion_matrix_plot(
                                task_analysis['confusion_data'], 
                                task, 
                                confusion_plot_path
                            )
            
            # 分析错误模式
            error_distribution, sample_error_count = analyze_error_patterns(all_errors, sample_indices)
            
            model_results[model_name] = {
                'task_results': task_results,
                'error_distribution': {
                    **error_distribution,
                    'sample_error_count': sample_error_count
                },
                'sample_indices': sample_indices
            }
            
            print(f"✅ {model_name} 分析完成")
            
        except Exception as e:
            print(f"❌ {model_name} 分析失败: {e}")
            model_results[model_name] = None
    
    # 生成综合报告
    print("\n生成错误分析报告...")
    report_path = generate_error_analysis_report(model_results, "reports")
    
    # 保存详细数据
    data_path = "reports/error_analysis_data.json"
    with open(data_path, 'w', encoding='utf-8') as f:
        # 序列化numpy数组为列表
        serializable_results = {}
        for model_name, results in model_results.items():
            if results is None:
                serializable_results[model_name] = None
            else:
                # 处理error_combinations中的tuple键
                error_dist = results['error_distribution'].copy()
                if 'error_combinations' in error_dist:
                    # 将tuple键转换为字符串
                    error_combos = {}
                    for combo_tuple, count in error_dist['error_combinations'].items():
                        combo_str = " + ".join(combo_tuple) if isinstance(combo_tuple, tuple) else str(combo_tuple)
                        error_combos[combo_str] = count
                    error_dist['error_combinations'] = error_combos
                
                serializable_results[model_name] = {
                    'task_results': results['task_results'],
                    'error_distribution': error_dist,
                    'sample_indices': results['sample_indices']
                }
        
        json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
    
    print("🎉 错误分析报告生成完成!")
    print(f"📄 Markdown报告: {report_path}")
    print(f"📊 详细数据: {data_path}")
    
    # 显示混淆矩阵图文件
    confusion_plots = [f for f in os.listdir("reports") if f.endswith("_confusion_matrix.png")]
    if confusion_plots:
        print("🖼️  混淆矩阵图:")
        for plot in confusion_plots:
            print(f"   - reports/{plot}")

if __name__ == "__main__":
    main()