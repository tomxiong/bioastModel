#!/usr/bin/env python3
"""
快速分析最后训练的模型，生成错误样本清单
简化版本，专门用于分析最新完成的训练
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_loader import MICDataLoader
from core.config.model_configs import get_model_config
import torch
from torch.utils.data import DataLoader

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

def quick_model_analysis(model_name, checkpoint_path, experiment_dir):
    """快速分析模型性能"""
    print(f"🔍 分析模型: {model_name}")
    print(f"📁 模型路径: {checkpoint_path}")
    
    # 加载数据
    data_loader = MICDataLoader(data_dir='bioast_dataset')
    test_images, test_labels = data_loader.get_test_data()
    
    # 创建测试数据加载器
    from core.data_loader import MICDataset
    test_dataset = MICDataset(test_images, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载模型
    model, config = load_model(model_name, checkpoint_path)
    if model is None:
        return None
    
    # 分析预测结果
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_probs = []
    
    print("📊 正在分析预测结果...")
    
    with torch.no_grad():
        for images, labels in test_dataloader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算准确率
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    accuracy = np.mean(all_predictions == all_labels)
    error_count = np.sum(all_predictions != all_labels)
    
    print(f"✅ 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"❌ 错误样本: {error_count} / {len(all_labels)}")
    
    # 找出错误样本
    error_indices = np.where(all_predictions != all_labels)[0]
    
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
    
    # 统计错误类型
    false_positives = len([e for e in error_analysis if e['error_type'] == 'False Positive'])
    false_negatives = len([e for e in error_analysis if e['error_type'] == 'False Negative'])
    
    print(f"📊 错误类型分布:")
    print(f"   - 假阳性 (False Positive): {false_positives}")
    print(f"   - 假阴性 (False Negative): {false_negatives}")
    
    # 高置信度错误
    high_conf_errors = [e for e in error_analysis if e['confidence'] > 0.8]
    if high_conf_errors:
        print(f"⚠️  高置信度错误 (>0.8): {len(high_conf_errors)} 个")
    
    # 保存结果
    output_dir = os.path.join(experiment_dir, 'error_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存错误样本清单
    error_df = pd.DataFrame(error_analysis)
    error_csv_path = os.path.join(output_dir, f'{model_name}_error_samples.csv')
    error_df.to_csv(error_csv_path, index=False)
    print(f"📁 错误样本清单: {error_csv_path}")
    
    # 保存分析报告
    report = {
        'model_name': model_name,
        'analysis_time': datetime.now().isoformat(),
        'total_samples': len(all_labels),
        'accuracy': float(accuracy),
        'error_count': int(error_count),
        'error_rate': float(error_count / len(all_labels)),
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'high_confidence_errors': len(high_conf_errors),
        'confidence_stats': {
            'mean': float(np.mean(all_confidences)),
            'std': float(np.std(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences))
        }
    }
    
    report_path = os.path.join(output_dir, f'{model_name}_quick_analysis.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"📋 分析报告: {report_path}")
    
    # 显示一些典型错误样本
    if error_analysis:
        print(f"\n🔍 典型错误样本:")
        # 按置信度排序
        sorted_errors = sorted(error_analysis, key=lambda x: x['confidence'], reverse=True)
        for i, error in enumerate(sorted_errors[:5]):
            print(f"   {i+1}. 样本{error['index']}: {error['error_type']}, 置信度={error['confidence']:.3f}")
    
    return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='快速分析最后训练的模型')
    parser.add_argument('--model', type=str, default=None,
                        help='指定要分析的模型名称')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='指定模型检查点路径')
    
    args = parser.parse_args()
    
    print("🚀 快速模型分析工具")
    print("=" * 50)
    
    if args.model and args.checkpoint:
        # 分析指定的模型
        report = quick_model_analysis(args.model, args.checkpoint, 
                                    os.path.dirname(args.checkpoint))
    else:
        # 自动找到最后一个训练的模型
        latest_model = find_latest_trained_model()
        if latest_model:
            report = quick_model_analysis(
                latest_model['name'], 
                latest_model['checkpoint'], 
                latest_model['experiment_dir']
            )
        else:
            print("❌ 未找到训练好的模型")
            return
    
    if report:
        print(f"\n✅ 分析完成!")
        print(f"📊 模型 {report['model_name']} 准确率: {report['accuracy']:.4f}")
        print(f"📁 结果保存在: {os.path.join(os.path.dirname(args.checkpoint) if args.checkpoint else 'experiments', 'error_analysis')}")

if __name__ == "__main__":
    main()