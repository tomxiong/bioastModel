"""
简化版气孔检测器验证脚本
生成详细的验证报告，包括正确和错误样本的分析
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import pandas as pd
import seaborn as sns
from datetime import datetime

from models.simplified_airbubble_detector import create_simplified_airbubble_detector
from training.dataset import create_data_loaders
from core.config import get_experiment_path, DATA_DIR, get_latest_experiment_path

def load_model(model_path):
    """加载模型"""
    print(f"🔍 加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = create_simplified_airbubble_detector()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ 成功加载模型")
        return model
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    print("🔍 评估模型性能...")
    
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_images = []
    all_file_paths = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_images.extend(images.cpu().numpy())
            
            # 如果数据集有文件路径属性
            if hasattr(data_loader.dataset, 'samples'):
                batch_indices = list(range(len(all_labels) - len(labels), len(all_labels)))
                batch_paths = [data_loader.dataset.samples[i][0] for i in batch_indices]
                all_file_paths.extend(batch_paths)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f"📊 准确率: {accuracy:.4f}")
    print(f"📊 精确率: {precision:.4f}")
    print(f"📊 召回率: {recall:.4f}")
    print(f"📊 F1分数: {f1:.4f}")
    
    # 返回结果
    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'confidences': np.array(all_confidences),
        'images': np.array(all_images),
        'file_paths': all_file_paths,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    }

def plot_confusion_matrix(labels, predictions, save_path):
    """绘制混淆矩阵"""
    print("🔍 绘制混淆矩阵...")
    
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['无气孔', '有气孔'],
                yticklabels=['无气孔', '有气孔'])
    
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ 混淆矩阵已保存到: {save_path}")

def plot_confidence_distribution(confidences, labels, predictions, save_path):
    """绘制置信度分布"""
    print("🔍 绘制置信度分布...")
    
    plt.figure(figsize=(12, 8))
    
    # 正确预测的置信度
    correct = confidences[predictions == labels]
    # 错误预测的置信度
    incorrect = confidences[predictions != labels]
    
    plt.hist(correct, bins=20, alpha=0.7, label='正确预测', color='green')
    plt.hist(incorrect, bins=20, alpha=0.7, label='错误预测', color='red')
    
    plt.title('预测置信度分布')
    plt.xlabel('置信度')
    plt.ylabel('样本数量')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ 置信度分布已保存到: {save_path}")

def visualize_samples(images, labels, predictions, confidences, save_dir, category, num_samples=10):
    """可视化样本"""
    print(f"🔍 可视化{category}样本...")
    
    if category == 'correct':
        indices = np.where(predictions == labels)[0]
    elif category == 'incorrect':
        indices = np.where(predictions != labels)[0]
    else:
        indices = np.arange(len(labels))
    
    if len(indices) == 0:
        print(f"⚠️ 没有{category}样本可供可视化")
        return
    
    # 选择样本
    if len(indices) > num_samples:
        indices = np.random.choice(indices, num_samples, replace=False)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 可视化每个样本
    for i, idx in enumerate(indices):
        image = images[idx].transpose(1, 2, 0)  # 转换为(H, W, C)格式
        
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        
        true_label = "有气孔" if labels[idx] == 1 else "无气孔"
        pred_label = "有气孔" if predictions[idx] == 1 else "无气孔"
        
        title = f"真实: {true_label}, 预测: {pred_label}\n置信度: {confidences[idx]:.4f}"
        plt.title(title)
        plt.axis('off')
        
        save_path = os.path.join(save_dir, f"{category}_sample_{i+1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    print(f"✅ {category}样本已保存到: {save_dir}")

def generate_error_list(file_paths, labels, predictions, confidences, save_path):
    """生成错误样本列表"""
    print("🔍 生成错误样本列表...")
    
    if len(file_paths) == 0:
        print("⚠️ 没有文件路径信息，无法生成错误样本列表")
        return
    
    # 找出错误预测的样本
    error_indices = np.where(predictions != labels)[0]
    
    if len(error_indices) == 0:
        print("✅ 没有错误预测的样本")
        return
    
    # 创建错误样本列表
    error_list = []
    for idx in error_indices:
        file_path = file_paths[idx]
        file_name = os.path.basename(file_path)
        true_label = "有气孔" if labels[idx] == 1 else "无气孔"
        pred_label = "有气孔" if predictions[idx] == 1 else "无气孔"
        
        error_list.append({
            'file_name': file_name,
            'file_path': file_path,
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': float(confidences[idx])
        })
    
    # 按置信度排序
    error_list.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 保存为JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(error_list, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 错误样本列表已保存到: {save_path}")
    print(f"   共{len(error_list)}个错误样本")

def generate_validation_report(results, save_path):
    """生成验证报告"""
    print("🔍 生成验证报告...")
    
    # 计算类别指标
    class_report = classification_report(
        results['labels'], 
        results['predictions'], 
        target_names=['无气孔', '有气孔'],
        output_dict=True
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    # 创建报告
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': results['metrics'],
        'class_report': class_report,
        'confusion_matrix': cm.tolist(),
        'sample_count': len(results['labels']),
        'correct_count': int(np.sum(results['predictions'] == results['labels'])),
        'error_count': int(np.sum(results['predictions'] != results['labels'])),
        'confidence_stats': {
            'mean': float(np.mean(results['confidences'])),
            'median': float(np.median(results['confidences'])),
            'min': float(np.min(results['confidences'])),
            'max': float(np.max(results['confidences'])),
            'correct_mean': float(np.mean(results['confidences'][results['predictions'] == results['labels']])),
            'incorrect_mean': float(np.mean(results['confidences'][results['predictions'] != results['labels']])) if np.any(results['predictions'] != results['labels']) else None
        }
    }
    
    # 保存为JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 验证报告已保存到: {save_path}")

def generate_markdown_report(results, save_path):
    """生成Markdown格式的验证报告"""
    print("🔍 生成Markdown格式的验证报告...")
    
    # 计算类别指标
    class_report = classification_report(
        results['labels'], 
        results['predictions'], 
        target_names=['无气孔', '有气孔'],
        output_dict=True
    )
    
    # 创建报告内容
    report = f"""# 简化版气孔检测器验证报告

## 验证时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 总体性能指标

| 指标 | 值 |
|------|-----|
| 准确率 | {results['metrics']['accuracy']:.4f} |
| 精确率 | {results['metrics']['precision']:.4f} |
| 召回率 | {results['metrics']['recall']:.4f} |
| F1分数 | {results['metrics']['f1']:.4f} |
| 样本总数 | {len(results['labels'])} |
| 正确预测数 | {int(np.sum(results['predictions'] == results['labels']))} |
| 错误预测数 | {int(np.sum(results['predictions'] != results['labels']))} |

## 类别性能指标

### 无气孔类别

| 指标 | 值 |
|------|-----|
| 精确率 | {class_report['无气孔']['precision']:.4f} |
| 召回率 | {class_report['无气孔']['recall']:.4f} |
| F1分数 | {class_report['无气孔']['f1-score']:.4f} |
| 支持度 | {class_report['无气孔']['support']} |

### 有气孔类别

| 指标 | 值 |
|------|-----|
| 精确率 | {class_report['有气孔']['precision']:.4f} |
| 召回率 | {class_report['有气孔']['recall']:.4f} |
| F1分数 | {class_report['有气孔']['f1-score']:.4f} |
| 支持度 | {class_report['有气孔']['support']} |

## 置信度分析

| 统计量 | 值 |
|--------|-----|
| 平均置信度 | {np.mean(results['confidences']):.4f} |
| 中位数置信度 | {np.median(results['confidences']):.4f} |
| 最小置信度 | {np.min(results['confidences']):.4f} |
| 最大置信度 | {np.max(results['confidences']):.4f} |
| 正确预测平均置信度 | {np.mean(results['confidences'][results['predictions'] == results['labels']]):.4f} |
"""

    # 如果有错误预测，添加错误预测的平均置信度
    if np.any(results['predictions'] != results['labels']):
        report += f"| 错误预测平均置信度 | {np.mean(results['confidences'][results['predictions'] != results['labels']]):.4f} |\n"
    
    # 添加混淆矩阵
    cm = confusion_matrix(results['labels'], results['predictions'])
    report += f"""
## 混淆矩阵

|  | 预测: 无气孔 | 预测: 有气孔 |
|-----------------|--------------|--------------|
| **真实: 无气孔** | {cm[0][0]} | {cm[0][1]} |
| **真实: 有气孔** | {cm[1][0]} | {cm[1][1]} |

## 错误分析

### 假阳性样本 (预测为有气孔，实际无气孔)
- 数量: {cm[0][1]}
- 可能原因: 图像中的噪声或光照变化被误识别为气孔

### 假阴性样本 (预测为无气孔，实际有气孔)
- 数量: {cm[1][0]}
- 可能原因: 气孔太小或对比度不足，导致特征不明显

## 结论与建议

- 模型总体表现: {"优秀" if results['metrics']['accuracy'] > 0.95 else "良好" if results['metrics']['accuracy'] > 0.85 else "一般"}
- 主要问题: {"假阳性较多" if cm[0][1] > cm[1][0] else "假阴性较多" if cm[1][0] > cm[0][1] else "假阳性和假阴性均衡"}
- 改进建议:
  - {"增强数据增强以减少假阳性" if cm[0][1] > cm[1][0] else "增强对小气孔的检测能力" if cm[1][0] > cm[0][1] else "平衡模型的精确率和召回率"}
  - 考虑调整模型结构或参数以提高性能
  - 增加更多的训练样本，特别是难以分类的边缘案例
"""
    
    # 保存为Markdown文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Markdown格式的验证报告已保存到: {save_path}")

def main():
    """主函数"""
    print("🔍 简化版气孔检测器验证")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 设备: {device}")
    
    # 获取最新实验路径
    model_name = 'simplified_airbubble_detector'
    experiment_path = get_latest_experiment_path(model_name)
    
    if experiment_path is None:
        print(f"❌ 未找到{model_name}的实验目录")
        return
    
    print(f"📁 实验路径: {experiment_path}")
    
    # 加载模型
    model_path = os.path.join(experiment_path, 'best_model.pth')
    model = load_model(model_path)
    
    if model is None:
        return
    
    # 创建数据加载器
    print("📂 加载数据集...")
    data_loaders = create_data_loaders(
        str(DATA_DIR),
        batch_size=32,
        num_workers=2
    )
    
    # 评估模型
    print("🧪 在测试集上评估模型...")
    results = evaluate_model(model, data_loaders['test'], device)
    
    # 创建验证目录
    validation_dir = os.path.join(experiment_path, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        os.path.join(validation_dir, 'confusion_matrix.png')
    )
    
    # 绘制置信度分布
    plot_confidence_distribution(
        results['confidences'],
        results['labels'],
        results['predictions'],
        os.path.join(validation_dir, 'confidence_distribution.png')
    )
    
    # 可视化正确样本
    visualize_samples(
        results['images'],
        results['labels'],
        results['predictions'],
        results['confidences'],
        os.path.join(validation_dir, 'correct_samples'),
        'correct',
        num_samples=10
    )
    
    # 可视化错误样本
    visualize_samples(
        results['images'],
        results['labels'],
        results['predictions'],
        results['confidences'],
        os.path.join(validation_dir, 'incorrect_samples'),
        'incorrect',
        num_samples=10
    )
    
    # 生成错误样本列表
    generate_error_list(
        results['file_paths'],
        results['labels'],
        results['predictions'],
        results['confidences'],
        os.path.join(validation_dir, 'error_list.json')
    )
    
    # 生成验证报告
    generate_validation_report(
        results,
        os.path.join(validation_dir, 'validation_report.json')
    )
    
    # 生成Markdown格式的验证报告
    generate_markdown_report(
        results,
        os.path.join(validation_dir, 'validation_report.md')
    )
    
    print("\n✅ 验证完成")
    print(f"📁 验证结果保存到: {validation_dir}")

if __name__ == "__main__":
    main()