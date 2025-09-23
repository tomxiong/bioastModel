#!/usr/bin/env python3
"""
生成NI多任务GrayColonyNet训练分析报告
包括性能指标、错误样本分析、可视化图表等
"""

import os
import sys
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import defaultdict, Counter

import torch
import torch.nn as nn

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multitask_gray_colony_net import create_multitask_gray_colony_net
from training.ni_multitask_dataset import create_ni_dataloaders
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def find_latest_experiment() -> Optional[Path]:
    """查找最新的实验目录"""
    experiment_pattern = "experiments/ni_multitask_gray_colony_net_*"
    experiment_dirs = glob.glob(experiment_pattern)
    
    if not experiment_dirs:
        return None
    
    # 按修改时间排序，返回最新的
    latest_dir = max(experiment_dirs, key=lambda x: os.path.getmtime(x))
    return Path(latest_dir)

def load_training_history(experiment_dir: Path) -> Dict:
    """加载训练历史"""
    history_file = experiment_dir / 'training_history.json'
    
    if not history_file.exists():
        print(f"警告: 找不到训练历史文件 {history_file}")
        return {}
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    return history

def load_test_report(experiment_dir: Path) -> Tuple[Dict, List]:
    """加载测试报告和错误样本"""
    test_report_file = experiment_dir / 'test_performance_report.json'
    error_samples_file = experiment_dir / 'error_samples_analysis.json'
    
    test_report = {}
    error_samples = []
    
    if test_report_file.exists():
        with open(test_report_file, 'r', encoding='utf-8') as f:
            test_report = json.load(f)
    
    if error_samples_file.exists():
        with open(error_samples_file, 'r', encoding='utf-8') as f:
            error_samples = json.load(f)
    
    return test_report, error_samples

def plot_training_curves(history: Dict, save_dir: Path):
    """绘制训练曲线"""
    if not history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('训练过程分析', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('损失变化曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 验证准确率曲线
    if history['val_metrics']:
        val_accuracies = [m.get('overall_accuracy', 0) for m in history['val_metrics']]
        axes[0, 1].plot(epochs, val_accuracies, 'g-', label='整体准确率', linewidth=2)
        
        # 各任务准确率
        growth_level_acc = [m.get('growth_level_accuracy', 0) for m in history['val_metrics']]
        growth_pattern_acc = [m.get('growth_pattern_accuracy', 0) for m in history['val_metrics']]
        fine_grained_acc = [m.get('fine_grained_accuracy', 0) for m in history['val_metrics']]
        
        axes[0, 1].plot(epochs, growth_level_acc, '--', label='生长级别', alpha=0.7)
        axes[0, 1].plot(epochs, growth_pattern_acc, '--', label='生长模式', alpha=0.7)
        axes[0, 1].plot(epochs, fine_grained_acc, '--', label='精细分类', alpha=0.7)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('验证集准确率变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 干扰因素mAP曲线
    if history['val_metrics']:
        interference_map = [m.get('interference_factors_mAP', 0) for m in history['val_metrics']]
        axes[1, 0].plot(epochs, interference_map, 'purple', label='干扰因素mAP', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].set_title('干扰因素检测性能')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 各任务F1分数对比
    if history['val_metrics'] and len(history['val_metrics']) > 0:
        latest_metrics = history['val_metrics'][-1]
        tasks = ['growth_level', 'growth_pattern', 'fine_grained']
        f1_scores = [latest_metrics.get(f'{task}_f1', 0) for task in tasks]
        task_names = ['生长级别', '生长模式', '精细分类']
        
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        bars = axes[1, 1].bar(task_names, f1_scores, color=colors)
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('各任务F1分数（最终）')
        axes[1, 1].set_ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{f1:.3f}', ha='center', va='bottom')
        
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(test_report: Dict, save_dir: Path):
    """绘制混淆矩阵"""
    if not test_report:
        return
    
    tasks = ['growth_level', 'growth_pattern', 'fine_grained']
    task_names = ['生长级别', '生长模式', '精细分类']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('各任务混淆矩阵', fontsize=16, fontweight='bold')
    
    for i, (task, task_name) in enumerate(zip(tasks, task_names)):
        cm_key = f'{task}_confusion_matrix'
        if cm_key in test_report:
            cm = np.array(test_report[cm_key])
            
            # 归一化混淆矩阵
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # 绘制热力图
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       ax=axes[i], cbar_kws={'label': 'Normalized Count'})
            axes[i].set_title(f'{task_name}混淆矩阵')
            axes[i].set_xlabel('预测类别')
            axes[i].set_ylabel('实际类别')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_error_patterns(error_samples: List[Dict], save_dir: Path) -> Dict:
    """分析错误样本模式"""
    if not error_samples:
        return {}
    
    error_analysis = {
        'total_errors': len(error_samples),
        'growth_level_errors': 0,
        'fine_grained_errors': 0,
        'error_by_panoramic': Counter(),
        'growth_level_error_patterns': defaultdict(list),
        'fine_grained_error_patterns': defaultdict(list)
    }
    
    # 分析错误模式
    for sample in error_samples:
        panoramic_id = sample.get('panoramic_id', 'unknown')
        error_analysis['error_by_panoramic'][panoramic_id] += 1
        
        # 生长级别错误
        if 'growth_level_error' in sample:
            error_analysis['growth_level_errors'] += 1
            error = sample['growth_level_error']
            pattern = f"{error['actual']} → {error['predicted']}"
            error_analysis['growth_level_error_patterns'][pattern].append(sample['image_id'])
        
        # 精细分类错误
        if 'fine_grained_error' in sample:
            error_analysis['fine_grained_errors'] += 1
            error = sample['fine_grained_error']
            pattern = f"{error['actual']} → {error['predicted']}"
            error_analysis['fine_grained_error_patterns'][pattern].append(sample['image_id'])
    
    # 可视化错误分布
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 按全景图统计错误
    if error_analysis['error_by_panoramic']:
        panoramic_ids = list(error_analysis['error_by_panoramic'].keys())
        error_counts = list(error_analysis['error_by_panoramic'].values())
        
        axes[0].bar(range(len(panoramic_ids)), error_counts)
        axes[0].set_xlabel('全景图ID')
        axes[0].set_ylabel('错误数量')
        axes[0].set_title('各全景图错误分布')
        axes[0].set_xticks(range(len(panoramic_ids)))
        axes[0].set_xticklabels(panoramic_ids, rotation=45)
    
    # 错误类型分布
    error_types = ['生长级别错误', '精细分类错误']
    error_counts = [error_analysis['growth_level_errors'], error_analysis['fine_grained_errors']]
    
    axes[1].pie(error_counts, labels=error_types, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('错误类型分布')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细错误分析
    with open(save_dir / 'detailed_error_analysis.json', 'w', encoding='utf-8') as f:
        # 转换Counter为普通dict以便JSON序列化
        analysis_copy = error_analysis.copy()
        analysis_copy['error_by_panoramic'] = dict(analysis_copy['error_by_panoramic'])
        analysis_copy['growth_level_error_patterns'] = dict(analysis_copy['growth_level_error_patterns'])
        analysis_copy['fine_grained_error_patterns'] = dict(analysis_copy['fine_grained_error_patterns'])
        json.dump(analysis_copy, f, ensure_ascii=False, indent=2)
    
    return error_analysis

def generate_performance_summary(test_report: Dict, error_analysis: Dict) -> Dict:
    """生成性能总结"""
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'model': 'MultitaskGrayColonyNet',
            'dataset': 'NI_Multitask_Dataset',
            'total_parameters': 931493
        },
        'performance_metrics': {},
        'error_analysis_summary': {},
        'recommendations': []
    }
    
    # 性能指标
    if test_report:
        summary['performance_metrics'] = {
            'overall_accuracy': test_report.get('overall_accuracy', 0),
            'growth_level_accuracy': test_report.get('growth_level_accuracy', 0),
            'growth_pattern_accuracy': test_report.get('growth_pattern_accuracy', 0),
            'fine_grained_accuracy': test_report.get('fine_grained_accuracy', 0),
            'total_samples': test_report.get('total_samples', 0),
            'error_rate': test_report.get('error_rate', 0)
        }
    
    # 错误分析总结
    if error_analysis:
        summary['error_analysis_summary'] = {
            'total_errors': error_analysis.get('total_errors', 0),
            'growth_level_errors': error_analysis.get('growth_level_errors', 0),
            'fine_grained_errors': error_analysis.get('fine_grained_errors', 0),
            'most_problematic_panoramic': error_analysis['error_by_panoramic'].most_common(3) if error_analysis.get('error_by_panoramic') else []
        }
    
    # 生成建议
    if test_report:
        overall_acc = test_report.get('overall_accuracy', 0)
        fine_grained_acc = test_report.get('fine_grained_accuracy', 0)
        
        if overall_acc > 0.85:
            summary['recommendations'].append("✅ 模型整体性能优秀，可以考虑部署")
        elif overall_acc > 0.75:
            summary['recommendations'].append("⚠️ 模型性能良好，但还有提升空间")
        else:
            summary['recommendations'].append("❌ 模型性能需要进一步优化")
        
        if fine_grained_acc < 0.7:
            summary['recommendations'].append("🔧 精细分类性能较低，建议增加数据或调整网络结构")
        
        if error_analysis and error_analysis.get('total_errors', 0) > 0:
            top_problematic = error_analysis['error_by_panoramic'].most_common(1)
            if top_problematic:
                panoramic_id, error_count = top_problematic[0]
                summary['recommendations'].append(f"🔍 全景图{panoramic_id}错误较多({error_count}个)，建议重点检查数据质量")
    
    return summary

def create_html_report(experiment_dir: Path, summary: Dict, save_dir: Path):
    """创建HTML报告"""
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NI多任务GrayColonyNet训练报告</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; }}
            .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
            .metric-card {{ background: white; padding: 15px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
            .metric-label {{ color: #7f8c8d; font-size: 14px; }}
            .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px; padding: 15px; }}
            .recommendation {{ margin: 8px 0; }}
            .image {{ text-align: center; margin: 20px 0; }}
            .image img {{ max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            .error {{ color: #e74c3c; }}
            .success {{ color: #27ae60; }}
            .warning {{ color: #f39c12; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 NI多任务GrayColonyNet训练报告</h1>
                <p>生成时间: {summary['experiment_info']['timestamp']}</p>
                <p>实验目录: {experiment_dir.name}</p>
            </div>
            
            <div class="section">
                <h2>📊 模型性能概览</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{summary['performance_metrics'].get('overall_accuracy', 0):.1%}</div>
                        <div class="metric-label">整体准确率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['performance_metrics'].get('growth_level_accuracy', 0):.1%}</div>
                        <div class="metric-label">生长级别准确率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['performance_metrics'].get('growth_pattern_accuracy', 0):.1%}</div>
                        <div class="metric-label">生长模式准确率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['performance_metrics'].get('fine_grained_accuracy', 0):.1%}</div>
                        <div class="metric-label">精细分类准确率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['performance_metrics'].get('total_samples', 0)}</div>
                        <div class="metric-label">测试样本数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary['performance_metrics'].get('error_rate', 0):.1%}</div>
                        <div class="metric-label">错误率</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 训练过程分析</h2>
                <div class="image">
                    <img src="training_curves.png" alt="训练曲线">
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 混淆矩阵分析</h2>
                <div class="image">
                    <img src="confusion_matrices.png" alt="混淆矩阵">
                </div>
            </div>
            
            <div class="section">
                <h2>❌ 错误样本分析</h2>
                <div class="image">
                    <img src="error_analysis.png" alt="错误分析">
                </div>
                <table>
                    <tr>
                        <th>错误类型</th>
                        <th>错误数量</th>
                        <th>占比</th>
                    </tr>
                    <tr>
                        <td>生长级别错误</td>
                        <td>{summary['error_analysis_summary'].get('growth_level_errors', 0)}</td>
                        <td>{summary['error_analysis_summary'].get('growth_level_errors', 0) / max(summary['error_analysis_summary'].get('total_errors', 1), 1) * 100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>精细分类错误</td>
                        <td>{summary['error_analysis_summary'].get('fine_grained_errors', 0)}</td>
                        <td>{summary['error_analysis_summary'].get('fine_grained_errors', 0) / max(summary['error_analysis_summary'].get('total_errors', 1), 1) * 100:.1f}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="section recommendations">
                <h2>💡 优化建议</h2>
    """
    
    for rec in summary['recommendations']:
        html_template += f'<div class="recommendation">{rec}</div>'
    
    html_template += """
            </div>
            
            <div class="section">
                <h2>🔧 模型信息</h2>
                <table>
                    <tr>
                        <th>项目</th>
                        <th>值</th>
                    </tr>
    """
    
    model_info = summary['experiment_info']
    html_template += f"""
                    <tr><td>模型名称</td><td>{model_info['model']}</td></tr>
                    <tr><td>数据集</td><td>{model_info['dataset']}</td></tr>
                    <tr><td>参数数量</td><td>{model_info['total_parameters']:,}</td></tr>
                    <tr><td>生成时间</td><td>{model_info['timestamp']}</td></tr>
    """
    
    html_template += """
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(save_dir / 'training_report.html', 'w', encoding='utf-8') as f:
        f.write(html_template)

def main():
    """主函数"""
    print("=== NI多任务GrayColonyNet训练分析报告生成 ===")
    
    # 查找最新实验
    experiment_dir = find_latest_experiment()
    if not experiment_dir:
        print("❌ 找不到实验目录")
        return
    
    print(f"分析实验: {experiment_dir}")
    
    # 创建报告目录
    report_dir = experiment_dir / 'analysis_report'
    report_dir.mkdir(exist_ok=True)
    
    # 加载数据
    print("📊 加载训练历史...")
    history = load_training_history(experiment_dir)
    
    print("📋 加载测试报告...")
    test_report, error_samples = load_test_report(experiment_dir)
    
    # 生成图表
    print("📈 生成训练曲线...")
    plot_training_curves(history, report_dir)
    
    print("🔍 生成混淆矩阵...")
    plot_confusion_matrices(test_report, report_dir)
    
    print("❌ 分析错误样本...")
    error_analysis = analyze_error_patterns(error_samples, report_dir)
    
    # 生成性能总结
    print("📝 生成性能总结...")
    summary = generate_performance_summary(test_report, error_analysis)
    
    # 保存总结
    with open(report_dir / 'performance_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 创建HTML报告
    print("🌐 创建HTML报告...")
    create_html_report(experiment_dir, summary, report_dir)
    
    print(f"\n✅ 分析报告生成完成!")
    print(f"📂 报告目录: {report_dir}")
    print(f"🌐 HTML报告: {report_dir / 'training_report.html'}")
    
    # 打印关键指标
    if test_report:
        print(f"\n📊 关键性能指标:")
        print(f"   整体准确率: {test_report.get('overall_accuracy', 0):.1%}")
        print(f"   生长级别准确率: {test_report.get('growth_level_accuracy', 0):.1%}")
        print(f"   生长模式准确率: {test_report.get('growth_pattern_accuracy', 0):.1%}")
        print(f"   精细分类准确率: {test_report.get('fine_grained_accuracy', 0):.1%}")
        print(f"   错误样本数: {len(error_samples)}/{test_report.get('total_samples', 0)}")
    
    return report_dir

if __name__ == "__main__":
    main()