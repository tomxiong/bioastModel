#!/usr/bin/env python3
"""
分析刚训练完成的模型，生成错误样本清单和性能分析报告
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_loader import MICDataLoader
from core.config.model_configs import get_model_config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def load_model(model_name, checkpoint_path):
    """加载训练好的模型"""
    try:
        # 获取模型配置
        config = get_model_config(model_name)
        
        # 动态导入模型
        module_path = config['module_path']
        class_name = config['class_name']
        
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)
        
        # 创建模型实例
        model = model_class(num_classes=config['num_classes'])
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, config
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None, None

def analyze_model_predictions(model, data_loader, model_name, device='cpu'):
    """分析模型预测结果"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_probs = []
    
    print(f"🔍 分析 {model_name} 的预测结果...")
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_probs = np.array(all_probs)
    
    # 计算准确率
    accuracy = np.mean(all_predictions == all_labels)
    print(f"✅ {model_name} 准确率: {accuracy:.4f}")
    
    # 找出错误样本
    error_indices = np.where(all_predictions != all_labels)[0]
    print(f"❌ 错误样本数量: {len(error_indices)}")
    
    # 生成错误样本清单
    error_analysis = []
    for idx in error_indices:
        error_type = "False Positive" if all_labels[idx] == 0 else "False Negative"
        error_analysis.append({
            'index': idx,
            'true_label': int(all_labels[idx]),
            'predicted_label': int(all_predictions[idx]),
            'confidence': float(all_confidences[idx]),
            'prob_negative': float(all_probs[idx][0]),
            'prob_positive': float(all_probs[idx][1]),
            'error_type': error_type
        })
    
    return {
        'accuracy': accuracy,
        'error_count': len(error_indices),
        'total_samples': len(all_labels),
        'error_analysis': error_analysis,
        'all_predictions': all_predictions.tolist(),
        'all_labels': all_labels.tolist(),
        'all_confidences': all_confidences.tolist()
    }

def generate_error_report(analysis_results, model_name, output_dir):
    """生成错误分析报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存错误样本清单
    error_df = pd.DataFrame(analysis_results['error_analysis'])
    error_csv_path = os.path.join(output_dir, f'{model_name}_error_samples.csv')
    error_df.to_csv(error_csv_path, index=False)
    print(f"📊 错误样本清单已保存: {error_csv_path}")
    
    # 生成统计报告
    report = {
        'model_name': model_name,
        'analysis_time': datetime.now().isoformat(),
        'output_dir': output_dir,
        'total_samples': analysis_results['total_samples'],
        'accuracy': analysis_results['accuracy'],
        'error_count': analysis_results['error_count'],
        'error_rate': analysis_results['error_count'] / analysis_results['total_samples'],
        'false_positives': len([e for e in analysis_results['error_analysis'] if e['error_type'] == 'False Positive']),
        'false_negatives': len([e for e in analysis_results['error_analysis'] if e['error_type'] == 'False Negative']),
        'confidence_stats': {
            'mean_confidence': np.mean(analysis_results['all_confidences']),
            'std_confidence': np.std(analysis_results['all_confidences']),
            'min_confidence': np.min(analysis_results['all_confidences']),
            'max_confidence': np.max(analysis_results['all_confidences'])
        }
    }
    
    # 保存JSON报告
    report_json_path = os.path.join(output_dir, f'{model_name}_analysis_report.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"📋 分析报告已保存: {report_json_path}")
    
    return error_df, report

def find_latest_trained_model():
    """自动找到最后一个训练的模型"""
    import glob
    
    # 查找所有best_model.pth文件，按修改时间排序
    model_files = glob.glob('experiments/*/best_model.pth') + glob.glob('experiments/*/*/best_model.pth')
    
    if not model_files:
        return None
    
    # 按修改时间排序，获取最新的
    latest_model = max(model_files, key=os.path.getmtime)
    
    # 从路径中提取模型信息
    path_parts = latest_model.split(os.sep)
    if len(path_parts) >= 3:
        model_name = path_parts[-2]  # 模型名称作为目录名
        experiment_dir = os.path.dirname(latest_model)
        
        return {
            'name': model_name,
            'checkpoint': latest_model,
            'experiment_dir': experiment_dir
        }
    
    return None

def analyze_recent_models(target_model=None):
    """分析指定模型或最近训练的模型"""
    
    if target_model:
        # 分析指定的模型
        model_configs = {
            'efficientnet_v2_s': {
                'name': 'efficientnet_v2_s',
                'checkpoint': 'experiments/experiment_20250805_220537/efficientnet_v2_s/best_model.pth',
                'experiment_dir': 'experiments/experiment_20250805_220537/efficientnet_v2_s'
            },
            'ghostnet': {
                'name': 'ghostnet',
                'checkpoint': 'experiments/experiment_20250805_221601/ghostnet/best_model.pth',
                'experiment_dir': 'experiments/experiment_20250805_221601/ghostnet'
            },
            'densenet121': {
                'name': 'densenet121',
                'checkpoint': 'experiments/experiment_20250805_222613/densenet121/best_model.pth',
                'experiment_dir': 'experiments/experiment_20250805_222613/densenet121'
            }
        }
        
        if target_model not in model_configs:
            print(f"❌ 不支持的模型: {target_model}")
            print(f"支持的模型: {list(model_configs.keys())}")
            return {}
        
        models_to_analyze = [model_configs[target_model]]
        print(f"🎯 分析指定模型: {target_model}")
        
    else:
        # 自动找到最后一个训练的模型
        latest_model = find_latest_trained_model()
        if latest_model:
            models_to_analyze = [latest_model]
            print(f"🔍 自动分析最后训练的模型: {latest_model['name']}")
        else:
            print("❌ 未找到训练好的模型")
            return {}
    
    # 准备数据
    data_loader = MICDataLoader(data_dir='bioast_dataset')
    train_images, train_labels = data_loader.get_train_data()
    val_images, val_labels = data_loader.get_val_data()
    test_images, test_labels = data_loader.get_test_data()
    
    # 创建测试数据加载器
    from core.data_loader import MICDataset
    test_dataset = MICDataset(test_images, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 分析每个模型
    all_results = {}
    
    for model_info in recent_models:
        model_name = model_info['name']
        checkpoint_path = model_info['checkpoint']
        experiment_dir = model_info['experiment_dir']
        
        print(f"\n{'='*60}")
        print(f"🔍 分析模型: {model_name}")
        print(f"{'='*60}")
        
        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"❌ 模型文件不存在: {checkpoint_path}")
            continue
        
        # 加载模型
        model, config = load_model(model_name, checkpoint_path)
        if model is None:
            continue
        
        # 分析预测结果
        analysis_results = analyze_model_predictions(model, test_dataloader, model_name)
        
        # 生成报告
        output_dir = os.path.join(experiment_dir, 'error_analysis')
        error_df, report = generate_error_report(analysis_results, model_name, output_dir)
        
        all_results[model_name] = {
            'analysis': analysis_results,
            'report': report,
            'error_df': error_df
        }
    
    # 生成综合比较报告
    generate_comparison_report(all_results)
    
    return all_results

def generate_comparison_report(all_results):
    """生成模型比较报告"""
    print(f"\n{'='*60}")
    print(f"📊 模型性能比较报告")
    print(f"{'='*60}")
    
    comparison_data = []
    for model_name, results in all_results.items():
        report = results['report']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{report['accuracy']:.4f}",
            'Error Rate': f"{report['error_rate']:.4f}",
            'Error Count': report['error_count'],
            'False Positives': report['false_positives'],
            'False Negatives': report['false_negatives'],
            'Mean Confidence': f"{report['confidence_stats']['mean_confidence']:.4f}"
        })
    
    # 创建比较表格
    comparison_df = pd.DataFrame(comparison_data)
    print("\n📋 模型性能比较:")
    print(comparison_df.to_string(index=False))
    
    # 保存比较报告
    comparison_path = 'recent_models_comparison_report.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n📊 比较报告已保存: {comparison_path}")
    
    # 显示错误样本类型分布
    print(f"\n🔍 错误样本类型分析:")
    for model_name, results in all_results.items():
        report = results['report']
        print(f"\n{model_name}:")
        print(f"  - 假阳性 (False Positive): {report['false_positives']} 个")
        print(f"  - 假阴性 (False Negative): {report['false_negatives']} 个")
        
        # 显示高置信度错误样本
        error_df = results['error_df']
        high_conf_errors = error_df[error_df['confidence'] > 0.8]
        if len(high_conf_errors) > 0:
            print(f"  - 高置信度错误 (>0.8): {len(high_conf_errors)} 个")
            print(f"    {high_conf_errors[['error_type', 'confidence']].head().to_string(index=False)}")

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='分析训练好的模型性能并生成错误样本清单')
    parser.add_argument('--model', type=str, default=None,
                        help='指定要分析的模型名称 (efficientnet_v2_s, ghostnet, densenet121)')
    parser.add_argument('--latest', action='store_true',
                        help='分析最后训练的模型 (默认行为)')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可分析的模型')
    
    args = parser.parse_args()
    
    if args.list:
        # 列出所有可分析的模型
        print("📋 可分析的模型:")
        available_models = ['efficientnet_v2_s', 'ghostnet', 'densenet121']
        for model in available_models:
            print(f"  - {model}")
        
        # 显示最新的模型
        latest = find_latest_trained_model()
        if latest:
            print(f"\n🔍 最后训练的模型: {latest['name']}")
        return
    
    print("🚀 开始分析模型...")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {device}")
    
    # 确定分析目标
    target_model = args.model if args.model else None
    
    # 分析模型
    all_results = analyze_recent_models(target_model)
    
    if not all_results:
        print("❌ 模型分析失败")
        return
    
    print(f"\n{'='*60}")
    print(f"✅ 模型分析完成!")
    print(f"{'='*60}")
    
    print(f"\n📁 生成的文件:")
    for model_name in all_results.keys():
        print(f"  - {all_results[model_name]['report'].get('output_dir', model_name + '/error_analysis')}/{model_name}_error_samples.csv")
        print(f"  - {all_results[model_name]['report'].get('output_dir', model_name + '/error_analysis')}/{model_name}_analysis_report.json")

if __name__ == "__main__":
    main()