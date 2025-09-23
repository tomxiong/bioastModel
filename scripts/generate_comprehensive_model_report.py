#!/usr/bin/env python3
"""
生成完整的多任务模型对比报告
包含训练性能、ONNX转换结果和综合分析
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def find_all_experiments():
    """查找所有多任务训练实验"""
    experiments = []
    
    # 查找所有实验目录
    patterns = [
        "experiments/fixed_efficientnet_b0_multitask_*",
        "experiments/resnet34_gpu_optimized_*", 
        "experiments/fixed_mobilenetv3_multitask_*"
    ]
    
    for pattern in patterns:
        for exp_dir in glob.glob(pattern):
            if os.path.exists(os.path.join(exp_dir, 'best.pth')):
                # 读取训练历史 (支持两种文件名)
                history_file = os.path.join(exp_dir, 'train_history.json')
                if not os.path.exists(history_file):
                    history_file = os.path.join(exp_dir, 'training_history.json')
                
                config_file = os.path.join(exp_dir, 'config.json')
                
                if os.path.exists(history_file) and os.path.exists(config_file):
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # 推断模型类型
                    if 'efficientnet_b0' in exp_dir:
                        model_name = 'EfficientNet-B0'
                        model_type = 'fixed_efficientnet_b0'
                    elif 'resnet34' in exp_dir:
                        model_name = 'ResNet-34'
                        model_type = 'resnet34'
                    elif 'mobilenetv3' in exp_dir:
                        model_name = 'MobileNetV3'
                        model_type = 'fixed_mobilenetv3'
                    else:
                        continue
                    
                    experiments.append({
                        'model_name': model_name,
                        'model_type': model_type,
                        'exp_dir': exp_dir,
                        'history': history,
                        'config': config
                    })
    
    return experiments

def load_onnx_conversion_data():
    """加载ONNX转换结果"""
    conversion_file = "onnx_models/conversion_summary.json"
    if os.path.exists(conversion_file):
        with open(conversion_file, 'r') as f:
            return json.load(f)
    return None

def extract_best_metrics(history):
    """提取最佳指标"""
    metrics = {}
    
    # 处理两种格式的准确率字段
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_accuracies'
    train_acc_key = 'train_accuracy' if 'train_accuracy' in history else 'train_accuracies'
    val_loss_key = 'val_loss' if 'val_loss' in history else 'val_losses'
    train_loss_key = 'train_loss' if 'train_loss' in history else 'train_losses'
    
    if val_acc_key in history and history[val_acc_key]:
        max_val = max(history[val_acc_key])
        # 智能检测：如果值已经是百分比格式（>1），则不乘以100
        multiplier = 1 if max_val > 1 else 100
        metrics['best_val_accuracy'] = max_val * multiplier
        metrics['final_val_accuracy'] = history[val_acc_key][-1] * multiplier
    
    if train_acc_key in history and history[train_acc_key]:
        max_train = max(history[train_acc_key])
        # 智能检测：如果值已经是百分比格式（>1），则不乘以100
        multiplier = 1 if max_train > 1 else 100
        metrics['best_train_accuracy'] = max_train * multiplier
        metrics['final_train_accuracy'] = history[train_acc_key][-1] * multiplier
    
    if val_loss_key in history and history[val_loss_key]:
        metrics['best_val_loss'] = min(history[val_loss_key])
        metrics['final_val_loss'] = history[val_loss_key][-1]
    
    if train_loss_key in history and history[train_loss_key]:
        metrics['best_train_loss'] = min(history[train_loss_key])
        metrics['final_train_loss'] = history[train_loss_key][-1]
    
    # 任务特定指标
    task_metrics = {}
    for key in history.keys():
        if key.startswith('val_') and '_accuracy' in key:
            task_name = key.replace('val_', '').replace('_accuracy', '')
            if task_name not in ['accuracy']:  # 排除总体准确率
                task_metrics[task_name] = max(history[key])
    
    metrics['task_metrics'] = task_metrics
    # 使用正确的字段计算epoch数
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_accuracies'
    metrics['total_epochs'] = len(history.get(val_acc_key, []))
    
    return metrics

def create_training_curves_plot(experiments, save_path):
    """创建训练曲线对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('多任务模型训练对比', fontsize=16, fontweight='bold')
    
    # 颜色映射
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, exp in enumerate(experiments):
        history = exp['history']
        model_name = exp['model_name']
        color = colors[i % len(colors)]
        
        # 处理两种格式的字段名
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_accuracies'
        train_acc_key = 'train_accuracy' if 'train_accuracy' in history else 'train_accuracies'
        val_loss_key = 'val_loss' if 'val_loss' in history else 'val_losses'
        train_loss_key = 'train_loss' if 'train_loss' in history else 'train_losses'
        
        epochs = range(1, len(history.get(val_acc_key, [])) + 1)
        
        # 验证准确率
        if val_acc_key in history and history[val_acc_key]:
            max_val = max(history[val_acc_key])
            multiplier = 1 if max_val > 1 else 100  # 智能检测格式
            val_acc = [x * multiplier for x in history[val_acc_key]]
            axes[0, 0].plot(epochs, val_acc, 
                           label=f"{model_name}", color=color, linewidth=2)
        
        # 训练准确率
        if train_acc_key in history and history[train_acc_key]:
            max_train = max(history[train_acc_key])
            multiplier = 1 if max_train > 1 else 100  # 智能检测格式
            train_acc = [x * multiplier for x in history[train_acc_key]]
            axes[0, 1].plot(epochs, train_acc, 
                           label=f"{model_name}", color=color, linewidth=2)
        
        # 验证损失
        if val_loss_key in history and history[val_loss_key]:
            axes[1, 0].plot(epochs, history[val_loss_key], 
                           label=f"{model_name}", color=color, linewidth=2)
        
        # 训练损失
        if train_loss_key in history and history[train_loss_key]:
            axes[1, 1].plot(epochs, history[train_loss_key], 
                           label=f"{model_name}", color=color, linewidth=2)
    
    # 设置子图
    axes[0, 0].set_title('验证准确率', fontweight='bold')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('训练准确率', fontweight='bold')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('验证损失', fontweight='bold')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('训练损失', fontweight='bold')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_comparison_plot(experiments, save_path):
    """创建性能对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('模型性能对比', fontsize=16, fontweight='bold')
    
    # 提取数据
    model_names = []
    val_accuracies = []
    train_accuracies = []
    
    for exp in experiments:
        metrics = extract_best_metrics(exp['history'])
        model_names.append(exp['model_name'])
        val_accuracies.append(metrics.get('best_val_accuracy', 0))
        train_accuracies.append(metrics.get('best_train_accuracy', 0))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    # 验证准确率对比
    bars1 = axes[0].bar(x - width/2, val_accuracies, width, 
                       label='验证准确率', alpha=0.8, color='skyblue')
    bars2 = axes[0].bar(x + width/2, train_accuracies, width, 
                       label='训练准确率', alpha=0.8, color='lightcoral')
    
    axes[0].set_xlabel('模型')
    axes[0].set_ylabel('准确率 (%)')
    axes[0].set_title('训练vs验证准确率对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 在柱子上显示数值
    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # 任务特定性能对比
    task_data = {}
    for exp in experiments:
        metrics = extract_best_metrics(exp['history'])
        for task, acc in metrics.get('task_metrics', {}).items():
            if task not in task_data:
                task_data[task] = []
            task_data[task].append(acc)
    
    if task_data:
        task_names = list(task_data.keys())
        x2 = np.arange(len(task_names))
        width2 = 0.25
        
        for i, exp in enumerate(experiments):
            metrics = extract_best_metrics(exp['history'])
            task_accs = [metrics.get('task_metrics', {}).get(task, 0) for task in task_names]
            axes[1].bar(x2 + i * width2, task_accs, width2, 
                       label=exp['model_name'], alpha=0.8)
        
        axes[1].set_xlabel('任务')
        axes[1].set_ylabel('准确率 (%)')
        axes[1].set_title('各任务性能对比')
        axes[1].set_xticks(x2 + width2)
        axes[1].set_xticklabels(task_names, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_onnx_comparison_plot(conversion_data, save_path):
    """创建ONNX性能对比图"""
    if not conversion_data:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('ONNX模型性能对比', fontsize=16, fontweight='bold')
    
    # 提取数据
    model_names = []
    file_sizes = []
    pytorch_times = []
    onnx_times = []
    speedups = []
    
    for model in conversion_data['models']:
        if model['conversion_success']:
            model_names.append(model['model_type'])
            file_sizes.append(model.get('onnx_file_size_mb', 0))
            pytorch_times.append(model.get('inference_pytorch_ms', 0))
            onnx_times.append(model.get('inference_onnx_ms', 0))
            speedups.append(model.get('speedup', 0))
    
    # 文件大小对比
    axes[0].bar(model_names, file_sizes, alpha=0.8, color='lightgreen')
    axes[0].set_xlabel('模型')
    axes[0].set_ylabel('文件大小 (MB)')
    axes[0].set_title('ONNX模型文件大小')
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, size in enumerate(file_sizes):
        axes[0].text(i, size + 1, f'{size:.1f}MB', ha='center', va='bottom')
    
    # 推理时间对比
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, pytorch_times, width, 
                       label='PyTorch', alpha=0.8, color='orange')
    bars2 = axes[1].bar(x + width/2, onnx_times, width, 
                       label='ONNX', alpha=0.8, color='blue')
    
    axes[1].set_xlabel('模型')
    axes[1].set_ylabel('推理时间 (ms)')
    axes[1].set_title('推理时间对比')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45)
    axes[1].legend()
    
    # 加速比对比
    axes[2].bar(model_names, speedups, alpha=0.8, color='red')
    axes[2].set_xlabel('模型')
    axes[2].set_ylabel('加速比')
    axes[2].set_title('ONNX加速比')
    axes[2].tick_params(axis='x', rotation=45)
    
    for i, speedup in enumerate(speedups):
        axes[2].text(i, speedup + 0.1, f'{speedup:.1f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_markdown_report(experiments, conversion_data, save_path):
    """生成Markdown格式的综合报告"""
    report = []
    report.append("# 多任务模型训练与转换综合报告")
    report.append("")
    report.append(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 概览
    report.append("## 📊 项目概览")
    report.append("")
    report.append(f"- **训练模型数量:** {len(experiments)} 个")
    report.append(f"- **ONNX转换成功:** {conversion_data['successful_conversions'] if conversion_data else 0} 个")
    report.append(f"- **数据集:** 70×70 灰度图像，4个多任务学习目标")
    report.append("")
    
    # 任务定义
    report.append("## 🎯 多任务学习目标")
    report.append("")
    report.append("| 任务 | 类别数 | 描述 |")
    report.append("|------|--------|------|")
    report.append("| Growth Level | 2 | 生长水平分类 (阴性/阳性) |")
    report.append("| Growth Pattern | 12 | 生长模式识别 |")
    report.append("| Interference Factors | 4 | 干扰因素检测 (多标签) |")
    report.append("| Microbe Type | 4 | 微生物类型分类 |")
    report.append("")
    
    # 模型性能排名
    report.append("## 🏆 模型性能排名")
    report.append("")
    
    # 按验证准确率排序
    experiments_sorted = sorted(experiments, 
                               key=lambda x: extract_best_metrics(x['history']).get('best_val_accuracy', 0), 
                               reverse=True)
    
    report.append("### 验证准确率排名")
    report.append("")
    report.append("| 排名 | 模型 | 验证准确率 | 训练准确率 | 训练轮数 |")
    report.append("|------|------|------------|------------|----------|")
    
    for i, exp in enumerate(experiments_sorted, 1):
        metrics = extract_best_metrics(exp['history'])
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
        report.append(f"| {emoji} {i} | **{exp['model_name']}** | "
                     f"{metrics.get('best_val_accuracy', 0):.2f}% | "
                     f"{metrics.get('best_train_accuracy', 0):.2f}% | "
                     f"{metrics.get('total_epochs', 0)} |")
    
    report.append("")
    
    # 详细模型分析
    report.append("## 📋 详细模型分析")
    report.append("")
    
    for exp in experiments_sorted:
        metrics = extract_best_metrics(exp['history'])
        config = exp['config']
        
        report.append(f"### {exp['model_name']}")
        report.append("")
        
        # 训练配置
        report.append("**训练配置:**")
        report.append("")
        report.append(f"- 批次大小: {config.get('batch_size', 'N/A')}")
        report.append(f"- 学习率: {config.get('learning_rate', 'N/A')}")
        report.append(f"- 优化器: {config.get('optimizer', 'N/A')}")
        report.append(f"- 调度器: {config.get('scheduler', 'N/A')}")
        report.append(f"- 混合精度: {'是' if config.get('use_amp', False) else '否'}")
        report.append("")
        
        # 性能指标
        report.append("**性能指标:**")
        report.append("")
        report.append(f"- 最佳验证准确率: **{metrics.get('best_val_accuracy', 0):.2f}%**")
        report.append(f"- 最佳训练准确率: **{metrics.get('best_train_accuracy', 0):.2f}%**")
        report.append(f"- 最低验证损失: {metrics.get('best_val_loss', 0):.4f}")
        report.append(f"- 训练轮数: {metrics.get('total_epochs', 0)}")
        report.append("")
        
        # 任务特定性能
        if metrics.get('task_metrics'):
            report.append("**各任务性能:**")
            report.append("")
            for task, acc in metrics['task_metrics'].items():
                report.append(f"- {task}: {acc:.2f}%")
            report.append("")
    
    # ONNX转换结果
    if conversion_data:
        report.append("## 🔄 ONNX转换结果")
        report.append("")
        
        report.append("| 模型 | 转换状态 | 文件大小 | PyTorch推理 | ONNX推理 | 加速比 |")
        report.append("|------|----------|----------|-------------|----------|--------|")
        
        for model in conversion_data['models']:
            status = "✅ 成功" if model['conversion_success'] else "❌ 失败"
            size = f"{model.get('onnx_file_size_mb', 0):.1f} MB" if model['conversion_success'] else "N/A"
            pytorch_time = f"{model.get('inference_pytorch_ms', 0):.2f} ms" if model['conversion_success'] else "N/A"
            onnx_time = f"{model.get('inference_onnx_ms', 0):.2f} ms" if model['conversion_success'] else "N/A"
            speedup = f"{model.get('speedup', 0):.1f}x" if model['conversion_success'] else "N/A"
            
            report.append(f"| {model['model_type']} | {status} | {size} | {pytorch_time} | {onnx_time} | {speedup} |")
        
        report.append("")
    
    # 结论和建议
    report.append("## 💡 结论与建议")
    report.append("")
    
    best_model = experiments_sorted[0]
    best_acc = extract_best_metrics(best_model['history']).get('best_val_accuracy', 0)
    
    report.append(f"1. **最佳模型:** {best_model['model_name']} (验证准确率: {best_acc:.2f}%)")
    report.append("")
    
    if conversion_data:
        # 找出最小的ONNX模型
        smallest_model = min([m for m in conversion_data['models'] if m['conversion_success']], 
                           key=lambda x: x.get('onnx_file_size_mb', float('inf')))
        fastest_model = max([m for m in conversion_data['models'] if m['conversion_success']], 
                          key=lambda x: x.get('speedup', 0))
        
        report.append(f"2. **最小ONNX模型:** {smallest_model['model_type']} ({smallest_model.get('onnx_file_size_mb', 0):.1f} MB)")
        report.append(f"3. **最快推理模型:** {fastest_model['model_type']} ({fastest_model.get('speedup', 0):.1f}x 加速比)")
        report.append("")
    
    report.append("4. **后续优化建议:**")
    report.append("   - 实施模型微调策略以提高准确率")
    report.append("   - 探索更多架构（如Vision Transformer, ConvNeXt等）")
    report.append("   - 优化数据增强策略")
    report.append("   - 实施集成学习方法")
    report.append("")
    
    # 生成C#使用说明
    report.append("## 🔧 C# ONNX模型使用指南")
    report.append("")
    report.append("### 环境依赖")
    report.append("")
    report.append("```xml")
    report.append("<PackageReference Include=\"Microsoft.ML.OnnxRuntime\" Version=\"1.16.0\" />")
    report.append("<PackageReference Include=\"Microsoft.ML.OnnxRuntime.Gpu\" Version=\"1.16.0\" />")
    report.append("```")
    report.append("")
    
    report.append("### 推理示例代码")
    report.append("")
    report.append("```csharp")
    report.append("using Microsoft.ML.OnnxRuntime;")
    report.append("using Microsoft.ML.OnnxRuntime.Tensors;")
    report.append("")
    report.append("// 加载模型")
    report.append("var sessionOptions = new SessionOptions();")
    report.append("var session = new InferenceSession(\"path/to/model.onnx\", sessionOptions);")
    report.append("")
    report.append("// 准备输入数据 (1x1x70x70)")
    report.append("var inputTensor = new DenseTensor<float>(new[] { 1, 1, 70, 70 });")
    report.append("// 填充图像数据到inputTensor...")
    report.append("")
    report.append("// 执行推理")
    report.append("var inputs = new List<NamedOnnxValue> {")
    report.append("    NamedOnnxValue.CreateFromTensor(\"input\", inputTensor)")
    report.append("};")
    report.append("")
    report.append("var outputs = session.Run(inputs);")
    report.append("")
    report.append("// 解析多任务输出")
    report.append("var growthLevel = outputs.First(x => x.Name == \"growth_level\").AsTensor<float>();")
    report.append("var growthPattern = outputs.First(x => x.Name == \"growth_pattern\").AsTensor<float>();")
    report.append("var interferenceFactors = outputs.First(x => x.Name == \"interference_factors\").AsTensor<float>();")
    report.append("var microbeType = outputs.First(x => x.Name == \"microbe_type\").AsTensor<float>();")
    report.append("```")
    report.append("")
    
    # 保存报告
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def main():
    print("🚀 生成完整多任务模型对比报告")
    print("=" * 60)
    
    # 查找所有实验
    experiments = find_all_experiments()
    
    if not experiments:
        print("❌ 未找到训练实验数据!")
        return
    
    print(f"发现 {len(experiments)} 个训练实验:")
    for exp in experiments:
        metrics = extract_best_metrics(exp['history'])
        print(f"  - {exp['model_name']}: {metrics.get('best_val_accuracy', 0):.2f}% 验证准确率")
    
    # 加载ONNX转换数据
    conversion_data = load_onnx_conversion_data()
    
    # 创建报告目录
    os.makedirs("reports", exist_ok=True)
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    create_training_curves_plot(experiments, "reports/comprehensive_training_curves.png")
    print("✅ 训练曲线图已生成")
    
    create_performance_comparison_plot(experiments, "reports/comprehensive_performance_comparison.png")
    print("✅ 性能对比图已生成")
    
    if conversion_data:
        create_onnx_comparison_plot(conversion_data, "reports/onnx_performance_comparison.png")
        print("✅ ONNX性能对比图已生成")
    
    # 生成Markdown报告
    print("\n生成综合报告...")
    generate_markdown_report(experiments, conversion_data, "reports/comprehensive_multitask_report.md")
    
    # 生成JSON数据报告
    report_data = {
        'generation_time': datetime.now().isoformat(),
        'experiments': [],
        'conversion_data': conversion_data
    }
    
    for exp in experiments:
        metrics = extract_best_metrics(exp['history'])
        report_data['experiments'].append({
            'model_name': exp['model_name'],
            'model_type': exp['model_type'],
            'experiment_dir': exp['exp_dir'],
            'config': exp['config'],
            'metrics': metrics
        })
    
    with open("reports/comprehensive_report_data.json", 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print("🎉 完整报告生成完成!")
    print("📁 报告文件:")
    print("  - Markdown报告: reports/comprehensive_multitask_report.md")
    print("  - JSON数据: reports/comprehensive_report_data.json")
    print("  - 训练曲线: reports/comprehensive_training_curves.png")
    print("  - 性能对比: reports/comprehensive_performance_comparison.png")
    if conversion_data:
        print("  - ONNX对比: reports/onnx_performance_comparison.png")

if __name__ == "__main__":
    main()