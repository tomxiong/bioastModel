#!/usr/bin/env python3
"""
专门测试NI多任务GrayColonyNet模型并生成错误样本分析报告
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_gray_colony_net import create_multitask_gray_colony_net
from training.ni_multitask_dataset import create_ni_dataloaders
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score

def test_model_performance(model: nn.Module, 
                          test_loader, 
                          device: torch.device,
                          save_dir: str) -> Tuple[Dict[str, Any], List[Dict]]:
    """测试模型性能并生成详细报告"""
    model.eval()
    
    all_predictions = defaultdict(list)
    all_targets = defaultdict(list)
    error_samples = []
    
    print(f"开始测试，共{len(test_loader)}个批次...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"处理批次 {batch_idx}/{len(test_loader)}")
                
            images = images.to(device)
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    targets[key] = value.to(device)
            
            outputs = model(images)
            
            # 收集预测结果
            batch_predictions = {}
            
            # 生长级别
            growth_level_pred = outputs['growth_level'].argmax(dim=1)
            batch_predictions['growth_level'] = growth_level_pred.cpu().numpy()
            all_predictions['growth_level'].extend(batch_predictions['growth_level'])
            all_targets['growth_level'].extend(targets['growth_level'].cpu().numpy())
            
            # 生长模式
            growth_pattern_pred = outputs['growth_pattern'].argmax(dim=1)
            batch_predictions['growth_pattern'] = growth_pattern_pred.cpu().numpy()
            all_predictions['growth_pattern'].extend(batch_predictions['growth_pattern'])
            all_targets['growth_pattern'].extend(targets['growth_pattern'].cpu().numpy())
            
            # 精细分类
            fine_output = outputs.get('fine_grained_refined', outputs['fine_grained'])
            fine_grained_pred = fine_output.argmax(dim=1)
            batch_predictions['fine_grained'] = fine_grained_pred.cpu().numpy()
            all_predictions['fine_grained'].extend(batch_predictions['fine_grained'])
            all_targets['fine_grained'].extend(targets['fine_grained'].cpu().numpy())
            
            # 干扰因素 (多标签)
            interference_pred_probs = torch.sigmoid(outputs['interference_mapping']).detach().cpu().numpy()
            interference_pred = (interference_pred_probs > 0.5).astype(int)
            batch_predictions['interference_factors'] = interference_pred
            all_predictions['interference_factors'].extend(interference_pred)
            all_targets['interference_factors'].extend(targets['interference_factors'].cpu().numpy())
            
            # 记录错误样本
            for i in range(len(images)):
                sample_errors = {}
                sample_errors['image_id'] = targets['image_id'][i]
                sample_errors['panoramic_id'] = targets['panoramic_id'][i]
                
                # 获取真实标签
                sample_errors['true_labels'] = {
                    'growth_level': int(targets['growth_level'][i].cpu().item()),
                    'growth_pattern': int(targets['growth_pattern'][i].cpu().item()),
                    'fine_grained': int(targets['fine_grained'][i].cpu().item()),
                    'interference_factors': targets['interference_factors'][i].cpu().numpy().tolist()
                }
                
                # 获取预测标签
                sample_errors['predicted_labels'] = {
                    'growth_level': int(batch_predictions['growth_level'][i]),
                    'growth_pattern': int(batch_predictions['growth_pattern'][i]),
                    'fine_grained': int(batch_predictions['fine_grained'][i]),
                    'interference_factors': batch_predictions['interference_factors'][i].tolist()
                }
                
                # 检查各任务是否预测错误
                has_error = False
                
                if batch_predictions['growth_level'][i] != targets['growth_level'][i].cpu().item():
                    sample_errors['growth_level_error'] = {
                        'predicted': int(batch_predictions['growth_level'][i]),
                        'actual': int(targets['growth_level'][i].cpu().item())
                    }
                    has_error = True
                
                if batch_predictions['growth_pattern'][i] != targets['growth_pattern'][i].cpu().item():
                    sample_errors['growth_pattern_error'] = {
                        'predicted': int(batch_predictions['growth_pattern'][i]),
                        'actual': int(targets['growth_pattern'][i].cpu().item())
                    }
                    has_error = True
                
                if batch_predictions['fine_grained'][i] != targets['fine_grained'][i].cpu().item():
                    sample_errors['fine_grained_error'] = {
                        'predicted': int(batch_predictions['fine_grained'][i]),
                        'actual': int(targets['fine_grained'][i].cpu().item())
                    }
                    has_error = True
                
                # 干扰因素错误检查
                true_interference = targets['interference_factors'][i].cpu().numpy()
                pred_interference = batch_predictions['interference_factors'][i]
                if not np.array_equal(true_interference, pred_interference):
                    sample_errors['interference_factors_error'] = {
                        'predicted': pred_interference.tolist(),
                        'actual': true_interference.tolist()
                    }
                    has_error = True
                
                if has_error:
                    error_samples.append(sample_errors)
    
    print(f"测试完成，收集到 {len(error_samples)} 个错误样本")
    
    # 计算详细指标
    test_report = {}
    
    # 单标签分类指标
    for task in ['growth_level', 'growth_pattern', 'fine_grained']:
        y_true = np.array(all_targets[task])
        y_pred = np.array(all_predictions[task])
        
        # 准确率
        acc = accuracy_score(y_true, y_pred)
        test_report[f'{task}_accuracy'] = acc
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        test_report[f'{task}_confusion_matrix'] = cm.tolist()
        
        # 分类报告
        try:
            cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            test_report[f'{task}_classification_report'] = cls_report
        except:
            test_report[f'{task}_classification_report'] = {}
    
    # 多标签分类 (干扰因素)
    if 'interference_factors' in all_predictions:
        y_true = np.array(all_targets['interference_factors'])
        y_pred = np.array(all_predictions['interference_factors'])
        
        # 计算每个标签的准确率
        interference_accuracies = []
        for i in range(y_true.shape[1]):
            acc = accuracy_score(y_true[:, i], y_pred[:, i])
            interference_accuracies.append(acc)
        
        test_report['interference_factors_label_accuracies'] = interference_accuracies
        test_report['interference_factors_mean_accuracy'] = np.mean(interference_accuracies)
    
    # 总体指标
    test_report['overall_accuracy'] = np.mean([
        test_report['growth_level_accuracy'],
        test_report['growth_pattern_accuracy'], 
        test_report['fine_grained_accuracy']
    ])
    
    test_report['total_samples'] = len(all_targets['growth_level'])
    test_report['error_samples_count'] = len(error_samples)
    test_report['error_rate'] = len(error_samples) / len(all_targets['growth_level'])
    
    # 保存测试报告
    save_path = Path(save_dir)
    
    # 保存详细报告
    report_file = save_path / 'test_performance_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)
    
    # 保存错误样本
    error_file = save_path / 'error_samples_analysis.json'
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_samples, f, ensure_ascii=False, indent=2)
    
    return test_report, error_samples


def analyze_error_patterns(error_samples: List[Dict], test_report: Dict) -> Dict:
    """分析错误模式"""
    print(f"分析 {len(error_samples)} 个错误样本...")
    
    error_analysis = {
        'total_errors': len(error_samples),
        'error_by_task': {
            'growth_level_errors': 0,
            'growth_pattern_errors': 0,
            'fine_grained_errors': 0,
            'interference_factors_errors': 0
        },
        'error_by_panoramic': Counter(),
        'task_error_patterns': {
            'growth_level': defaultdict(int),
            'growth_pattern': defaultdict(int),
            'fine_grained': defaultdict(int)
        },
        'co_occurrence_errors': defaultdict(int)
    }
    
    # 分析错误模式
    for sample in error_samples:
        panoramic_id = sample.get('panoramic_id', 'unknown')
        error_analysis['error_by_panoramic'][panoramic_id] += 1
        
        # 记录各任务错误
        error_tasks = []
        
        if 'growth_level_error' in sample:
            error_analysis['error_by_task']['growth_level_errors'] += 1
            error = sample['growth_level_error']
            pattern = f"{error['actual']} → {error['predicted']}"
            error_analysis['task_error_patterns']['growth_level'][pattern] += 1
            error_tasks.append('growth_level')
        
        if 'growth_pattern_error' in sample:
            error_analysis['error_by_task']['growth_pattern_errors'] += 1
            error = sample['growth_pattern_error']
            pattern = f"{error['actual']} → {error['predicted']}"
            error_analysis['task_error_patterns']['growth_pattern'][pattern] += 1
            error_tasks.append('growth_pattern')
            
        if 'fine_grained_error' in sample:
            error_analysis['error_by_task']['fine_grained_errors'] += 1
            error = sample['fine_grained_error']
            pattern = f"{error['actual']} → {error['predicted']}"
            error_analysis['task_error_patterns']['fine_grained'][pattern] += 1
            error_tasks.append('fine_grained')
            
        if 'interference_factors_error' in sample:
            error_analysis['error_by_task']['interference_factors_errors'] += 1
            error_tasks.append('interference_factors')
        
        # 记录任务错误共现
        if len(error_tasks) > 1:
            co_error = '+'.join(sorted(error_tasks))
            error_analysis['co_occurrence_errors'][co_error] += 1
    
    return error_analysis


def generate_detailed_error_report(error_analysis: Dict, test_report: Dict, save_dir: Path):
    """生成详细的错误分析报告"""
    
    report_content = f"""# NI多任务GrayColonyNet错误样本分析报告

## 测试总览
- 测试样本总数: {test_report['total_samples']}
- 错误样本数量: {error_analysis['total_errors']}
- 整体错误率: {test_report['error_rate']:.2%}
- 整体准确率: {test_report['overall_accuracy']:.2%}

## 各任务性能指标
### 生长级别分类
- 准确率: {test_report['growth_level_accuracy']:.4f}

### 生长模式分类  
- 准确率: {test_report['growth_pattern_accuracy']:.4f}

### 精细分类
- 准确率: {test_report['fine_grained_accuracy']:.4f}

### 干扰因素检测（多标签）
- 平均准确率: {test_report.get('interference_factors_mean_accuracy', 0):.4f}

## 错误分析

### 各任务错误分布
- 生长级别错误: {error_analysis['error_by_task']['growth_level_errors']}个
- 生长模式错误: {error_analysis['error_by_task']['growth_pattern_errors']}个  
- 精细分类错误: {error_analysis['error_by_task']['fine_grained_errors']}个
- 干扰因素错误: {error_analysis['error_by_task']['interference_factors_errors']}个

### 全景图错误分布
"""
    
    # 全景图错误统计
    if error_analysis['error_by_panoramic']:
        report_content += "最容易出错的全景图:\n"
        for panoramic_id, count in error_analysis['error_by_panoramic'].most_common(10):
            report_content += f"- {panoramic_id}: {count}个错误\n"
    
    # 各任务错误模式
    for task, patterns in error_analysis['task_error_patterns'].items():
        if patterns:
            report_content += f"\n### {task}常见错误模式\n"
            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
                report_content += f"- {pattern}: {count}次\n"
    
    # 任务错误共现
    if error_analysis['co_occurrence_errors']:
        report_content += "\n### 多任务同时出错情况\n"
        for co_error, count in sorted(error_analysis['co_occurrence_errors'].items(), key=lambda x: x[1], reverse=True):
            report_content += f"- {co_error}: {count}个样本\n"
    
    # 保存报告
    report_file = save_dir / 'detailed_error_analysis_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ 详细错误分析报告已保存: {report_file}")


def main():
    """主函数"""
    print("=== NI多任务GrayColonyNet模型测试和错误分析 ===")
    
    # 配置
    data_root = '/home/aaa/ws/bioastModel/dataset_ni_multitask'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_dir = Path('experiments/ni_multitask_gray_colony_net_20250904_231849')
    
    print(f"使用设备: {device}")
    print(f"实验目录: {experiment_dir}")
    
    # 检查实验目录是否存在
    if not experiment_dir.exists():
        print(f"❌ 找不到实验目录: {experiment_dir}")
        return
    
    # 创建数据加载器
    print("创建数据加载器...")
    _, _, test_loader = create_ni_dataloaders(
        data_root=data_root,
        batch_size=16,
        num_workers=4
    )
    
    # 创建模型
    print("创建模型...")
    model = create_multitask_gray_colony_net(
        feature_dim=128,
        enable_background_filter=True,
        dropout_rate=0.2
    )
    
    # 加载最佳模型权重
    best_checkpoint_path = experiment_dir / 'best_checkpoint.pth'
    if best_checkpoint_path.exists():
        print(f"加载最佳模型权重: {best_checkpoint_path}")
        try:
            # 使用weights_only=False来避免安全限制
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 成功加载模型 (epoch {checkpoint['epoch']})")
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            print("使用未训练的模型进行测试...")
    else:
        print(f"❌ 找不到模型文件: {best_checkpoint_path}")
        return
    
    model.to(device)
    
    # 测试模型性能
    print("开始模型测试...")
    test_report, error_samples = test_model_performance(
        model, test_loader, device, str(experiment_dir)
    )
    
    print(f"\n=== 测试结果概览 ===")
    print(f"总体准确率: {test_report['overall_accuracy']:.4f}")
    print(f"生长级别准确率: {test_report['growth_level_accuracy']:.4f}")
    print(f"生长模式准确率: {test_report['growth_pattern_accuracy']:.4f}")
    print(f"精细分类准确率: {test_report['fine_grained_accuracy']:.4f}")
    print(f"错误样本数: {test_report['error_samples_count']}/{test_report['total_samples']} ({test_report['error_rate']:.2%})")
    
    # 错误样本分析
    print("\n分析错误样本模式...")
    error_analysis = analyze_error_patterns(error_samples, test_report)
    
    # 生成详细错误报告
    generate_detailed_error_report(error_analysis, test_report, experiment_dir)
    
    # 保存错误分析结果
    error_analysis_file = experiment_dir / 'error_pattern_analysis.json'
    with open(error_analysis_file, 'w', encoding='utf-8') as f:
        # 转换Counter对象为普通dict
        analysis_copy = error_analysis.copy()
        analysis_copy['error_by_panoramic'] = dict(analysis_copy['error_by_panoramic'])
        analysis_copy['co_occurrence_errors'] = dict(analysis_copy['co_occurrence_errors'])
        analysis_copy['task_error_patterns'] = {
            task: dict(patterns) for task, patterns in analysis_copy['task_error_patterns'].items()
        }
        json.dump(analysis_copy, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 测试和错误分析完成!")
    print(f"📁 测试报告: {experiment_dir}/test_performance_report.json")
    print(f"📁 错误样本: {experiment_dir}/error_samples_analysis.json")
    print(f"📁 错误分析: {experiment_dir}/error_pattern_analysis.json")
    print(f"📄 详细报告: {experiment_dir}/detailed_error_analysis_report.md")
    
    return experiment_dir


if __name__ == "__main__":
    main()