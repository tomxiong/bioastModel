#!/usr/bin/env python3
"""
EfficientNet-B0多任务模型测试和错误分析脚本
对训练好的EfficientNet-B0多任务模型进行全面的性能评估
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.multitask_efficientnet_b0 import create_multitask_efficientnet_b0_standard
from training.ni_multitask_dataset import NIMultitaskDataset
from torch.utils.data import DataLoader


def find_latest_experiment(pattern='ni_multitask_efficientnet_b0'):
    """查找最新的实验目录"""
    experiments_dir = Path(project_root) / 'experiments'
    matching_dirs = [d for d in experiments_dir.iterdir() 
                    if d.is_dir() and pattern in d.name]
    
    if not matching_dirs:
        raise FileNotFoundError(f"未找到匹配的实验目录：{pattern}")
    
    return sorted(matching_dirs, key=lambda x: x.name)[-1]


def load_model_and_data(experiment_dir, device='cuda'):
    """加载模型和测试数据"""
    print("加载模型和数据...")
    
    # 加载最佳模型
    model_path = experiment_dir / 'best_model.pth'
    if not model_path.exists():
        model_path = experiment_dir / 'final_model.pth'
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 创建和加载模型
    model = create_multitask_efficientnet_b0_standard(pretrained=False)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建测试数据加载器
    dataset_path = Path(project_root) / 'dataset_ni_multitask'
    test_dataset = NIMultitaskDataset(data_root=dataset_path, split='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"✓ 模型加载完成")
    print(f"✓ 测试集样本数: {len(test_dataset)}")
    
    return model, test_loader, checkpoint


def evaluate_model(model, test_loader, device='cuda'):
    """全面评估模型性能"""
    print("开始模型评估...")
    
    model.eval()
    all_predictions = defaultdict(list)
    all_targets = defaultdict(list)
    all_image_ids = []
    all_panoramic_ids = []
    sample_errors = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            
            # 提取张量类型的任务标签
            task_targets = {}
            for task, target in targets.items():
                if isinstance(target, torch.Tensor):
                    task_targets[task] = target.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 收集预测和真实值
            for task in task_targets.keys():
                if task == 'interference_factors':
                    # 多标签分类
                    pred = (torch.sigmoid(outputs[task]) > 0.5).cpu().numpy()
                    target = task_targets[task].cpu().numpy()
                else:
                    # 单标签分类
                    pred = outputs[task].argmax(dim=1).cpu().numpy()
                    target = task_targets[task].cpu().numpy()
                
                all_predictions[task].extend(pred)
                all_targets[task].extend(target)
            
            # 收集样本信息
            all_image_ids.extend(targets.get('image_id', []))
            all_panoramic_ids.extend(targets.get('panoramic_id', []))
            
            # 分析错误样本
            for i in range(len(images)):
                is_error = False
                error_tasks = []
                
                for task in task_targets.keys():
                    if task == 'interference_factors':
                        pred_sample = (torch.sigmoid(outputs[task][i]) > 0.5).cpu().numpy()
                        target_sample = task_targets[task][i].cpu().numpy()
                        if not np.array_equal(pred_sample, target_sample):
                            is_error = True
                            error_tasks.append(task)
                    else:
                        pred_sample = outputs[task][i].argmax(dim=0).item()
                        target_sample = task_targets[task][i].item()
                        if pred_sample != target_sample:
                            is_error = True
                            error_tasks.append(task)
                
                if is_error:
                    error_info = {
                        'sample_idx': batch_idx * test_loader.batch_size + i,
                        'image_id': targets.get('image_id', ['unknown'])[i],
                        'panoramic_id': targets.get('panoramic_id', ['unknown'])[i],
                        'error_tasks': error_tasks,
                        'predictions': {},
                        'targets': {}
                    }
                    
                    for task in task_targets.keys():
                        if task == 'interference_factors':
                            pred_sample = (torch.sigmoid(outputs[task][i]) > 0.5).cpu().numpy()
                            target_sample = task_targets[task][i].cpu().numpy()
                            error_info['predictions'][task] = pred_sample.tolist()
                            error_info['targets'][task] = target_sample.tolist()
                        else:
                            pred_sample = outputs[task][i].argmax(dim=0).item()
                            target_sample = task_targets[task][i].item()
                            error_info['predictions'][task] = pred_sample
                            error_info['targets'][task] = target_sample
                    
                    sample_errors.append(error_info)
    
    print(f"✓ 评估完成，发现 {len(sample_errors)} 个错误样本")
    return all_predictions, all_targets, sample_errors, all_image_ids, all_panoramic_ids


def compute_metrics(predictions, targets, task_names):
    """计算各任务的性能指标"""
    print("计算性能指标...")
    
    results = {}
    
    for task in task_names:
        if task == 'interference_factors':
            # 多标签分类指标
            pred_array = np.array(predictions[task])
            target_array = np.array(targets[task])
            
            # 计算每个标签的准确率
            per_label_acc = []
            for i in range(pred_array.shape[1]):
                acc = (pred_array[:, i] == target_array[:, i]).mean()
                per_label_acc.append(acc)
            
            avg_accuracy = np.mean(per_label_acc)
            
            results[task] = {
                'accuracy': avg_accuracy,
                'per_label_accuracy': per_label_acc,
                'type': 'multilabel'
            }
        else:
            # 单标签分类指标
            pred_array = np.array(predictions[task])
            target_array = np.array(targets[task])
            
            accuracy = (pred_array == target_array).mean()
            
            # 混淆矩阵和分类报告
            unique_labels = sorted(set(target_array))
            cm = confusion_matrix(target_array, pred_array, labels=unique_labels)
            
            results[task] = {
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'unique_labels': unique_labels,
                'type': 'multiclass'
            }
    
    # 计算整体准确率
    overall_accuracy = np.mean([results[task]['accuracy'] for task in task_names])
    results['overall'] = {'accuracy': overall_accuracy}
    
    return results


def analyze_error_patterns(sample_errors):
    """分析错误模式"""
    print("分析错误模式...")
    
    # 按任务统计错误
    task_error_count = defaultdict(int)
    for error in sample_errors:
        for task in error['error_tasks']:
            task_error_count[task] += 1
    
    # 按全景图统计错误
    panoramic_error_count = defaultdict(int)
    for error in sample_errors:
        panoramic_error_count[error['panoramic_id']] += 1
    
    # 分析任务错误模式
    task_error_patterns = defaultdict(lambda: defaultdict(int))
    for error in sample_errors:
        for task in error['error_tasks']:
            if task != 'interference_factors':  # 单标签任务
                target_val = error['targets'][task]
                pred_val = error['predictions'][task]
                pattern = f"{target_val} → {pred_val}"
                task_error_patterns[task][pattern] += 1
    
    # 错误同时出现的组合分析
    co_occurrence_errors = defaultdict(int)
    for error in sample_errors:
        if len(error['error_tasks']) > 1:
            sorted_tasks = sorted(error['error_tasks'])
            combination = '+'.join(sorted_tasks)
            co_occurrence_errors[combination] += 1
        elif len(error['error_tasks']) == 1:
            co_occurrence_errors[error['error_tasks'][0]] += 1
    
    return {
        'total_errors': len(sample_errors),
        'error_by_task': dict(task_error_count),
        'error_by_panoramic': dict(panoramic_error_count),
        'task_error_patterns': dict(task_error_patterns),
        'co_occurrence_errors': dict(co_occurrence_errors)
    }


def save_results(results, error_analysis, sample_errors, experiment_dir):
    """保存结果和分析报告"""
    print("保存结果...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存性能指标，处理numpy类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results_serializable = convert_numpy_types(results)
    metrics_path = experiment_dir / f'test_metrics_{timestamp}.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    # 保存错误分析
    error_analysis_serializable = convert_numpy_types(error_analysis)
    error_analysis_path = experiment_dir / f'error_pattern_analysis_{timestamp}.json'
    with open(error_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(error_analysis_serializable, f, indent=2, ensure_ascii=False)
    
    # 保存详细错误样本
    if sample_errors:
        errors_path = experiment_dir / f'detailed_errors_{timestamp}.json'
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(sample_errors, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    report_path = experiment_dir / f'test_report_{timestamp}.md'
    generate_markdown_report(results, error_analysis, report_path, len(sample_errors))
    
    print(f"✓ 结果保存完成")
    return metrics_path, error_analysis_path, report_path


def generate_markdown_report(results, error_analysis, report_path, total_samples):
    """生成Markdown格式的测试报告"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# EfficientNet-B0多任务模型测试报告\n\n")
        
        f.write("## 测试总览\n")
        f.write(f"- 测试样本总数: {total_samples}\n")
        f.write(f"- 错误样本数量: {error_analysis['total_errors']}\n")
        f.write(f"- 整体错误率: {error_analysis['total_errors']/total_samples*100:.2f}%\n")
        f.write(f"- 整体准确率: {results['overall']['accuracy']*100:.2f}%\n\n")
        
        f.write("## 各任务性能指标\n")
        for task, metrics in results.items():
            if task == 'overall':
                continue
            f.write(f"### {task.replace('_', ' ').title()}\n")
            f.write(f"- 准确率: {metrics['accuracy']:.4f}\n\n")
        
        f.write("## 错误分析\n\n")
        
        f.write("### 各任务错误分布\n")
        for task, count in error_analysis['error_by_task'].items():
            f.write(f"- {task}错误: {count}个\n")
        f.write("\n")
        
        f.write("### 全景图错误分布\n")
        f.write("最容易出错的全景图:\n")
        sorted_panoramic = sorted(error_analysis['error_by_panoramic'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]
        for panoramic_id, count in sorted_panoramic:
            f.write(f"- {panoramic_id}: {count}个错误\n")
        f.write("\n")
        
        # 任务错误模式
        for task, patterns in error_analysis['task_error_patterns'].items():
            f.write(f"### {task}常见错误模式\n")
            sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]
            for pattern, count in sorted_patterns:
                f.write(f"- {pattern}: {count}次\n")
            f.write("\n")
        
        f.write("### 多任务同时出错情况\n")
        sorted_cooccur = sorted(error_analysis['co_occurrence_errors'].items(), 
                              key=lambda x: x[1], reverse=True)[:10]
        for combination, count in sorted_cooccur:
            f.write(f"- {combination}: {count}个样本\n")
        f.write("\n")


def main():
    """主函数"""
    print("=== EfficientNet-B0多任务模型测试开始 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 查找最新实验
        experiment_dir = find_latest_experiment('ni_multitask_efficientnet_b0')
        print(f"实验目录: {experiment_dir}")
        
        # 2. 加载模型和数据
        model, test_loader, checkpoint = load_model_and_data(experiment_dir, device)
        
        # 3. 评估模型
        predictions, targets, sample_errors, image_ids, panoramic_ids = evaluate_model(
            model, test_loader, device
        )
        
        # 4. 计算性能指标
        task_names = ['growth_level', 'growth_pattern', 'interference_factors', 'fine_grained']
        results = compute_metrics(predictions, targets, task_names)
        
        # 5. 错误模式分析
        error_analysis = analyze_error_patterns(sample_errors)
        
        # 6. 保存结果
        metrics_path, error_path, report_path = save_results(
            results, error_analysis, sample_errors, experiment_dir
        )
        
        # 7. 打印总结
        print("\n=== 测试完成 ===")
        print(f"整体准确率: {results['overall']['accuracy']*100:.2f}%")
        print(f"错误样本数: {error_analysis['total_errors']}")
        
        print("\n各任务准确率:")
        for task in task_names:
            print(f"  {task}: {results[task]['accuracy']*100:.2f}%")
        
        print(f"\n详细报告: {report_path}")
        print("✓ EfficientNet-B0多任务模型测试完成！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())