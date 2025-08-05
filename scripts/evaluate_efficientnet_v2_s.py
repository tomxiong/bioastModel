"""
EfficientNet V2-S 模型评估脚本
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json
import sys
import os
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.efficientnet_v2_wrapper import EfficientNetV2S
from core.data_loader import MICDataLoader, create_data_loaders

def evaluate_efficientnet_v2_s():
    """评估EfficientNet V2-S模型"""
    
    print("开始评估EfficientNet V2-S模型...")
    
    # 模型路径
    model_path = "experiments/experiment_20250804_123239/efficientnet_v2_s/best_model.pth"
    report_dir = Path("reports/individual/efficientnet_v2_s")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建模型
    model = EfficientNetV2S(num_classes=2)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    val_acc = checkpoint.get('val_acc', checkpoint.get('val_accuracy', 0))
    print(f"已加载模型权重，验证准确率: {val_acc:.2f}%")
    
    # 创建数据加载器
    mic_data_loader = MICDataLoader(data_dir="bioast_dataset", image_size=(70, 70))
    train_loader, val_loader, test_loader = create_data_loaders(
        mic_data_loader, batch_size=32, num_workers=4
    )
    
    # 在测试集上评估
    print("在测试集上评估模型...")
    test_loss, test_acc, y_true, y_pred, y_proba = evaluate_on_dataset(model, test_loader)
    
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试损失: {test_loss:.4f}")
    
    # 生成详细报告
    generate_detailed_report(
        model, checkpoint, test_acc, test_loss, y_true, y_pred, y_proba, report_dir
    )
    
    print(f"评估报告已保存到: {report_dir}")
    
    return test_acc, test_loss

def evaluate_on_dataset(model, data_loader):
    """在数据集上评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    y_true = []
    y_pred = []
    y_proba = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, targets in data_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 收集预测结果
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(y_true), np.array(y_pred), np.array(y_proba)

def generate_detailed_report(model, checkpoint, test_acc, test_loss, y_true, y_pred, y_proba, report_dir):
    """生成详细评估报告"""
    
    # 基本信息
    model_info = {
        'model_name': 'EfficientNet V2-S',
        'parameters': sum(p.numel() for p in model.parameters()),
        'parameters_millions': sum(p.numel() for p in model.parameters()) / 1e6,
        'input_size': '70x70',
        'num_classes': 2,
        'best_val_accuracy': checkpoint.get('val_acc', checkpoint.get('val_accuracy', 0)),
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'architecture': 'EfficientNet V2-S with 70x70 adaptation'
    }
    
    # 保存模型信息
    with open(report_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # 分类报告
    class_names = ['Negative', 'Positive']
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    with open(report_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('EfficientNet V2-S - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(report_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC曲线
    if y_proba.shape[1] == 2:  # 二分类
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('EfficientNet V2-S - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(report_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 生成HTML报告
    generate_html_report(model_info, report, cm, report_dir)
    
    print("详细报告生成完成!")

def generate_html_report(model_info, classification_report, confusion_matrix, report_dir):
    """生成HTML格式的评估报告"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EfficientNet V2-S 评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .highlight {{ color: #2e8b57; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>EfficientNet V2-S 模型评估报告</h1>
            <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>模型基本信息</h2>
            <div class="metric">
                <strong>模型名称:</strong> {model_info['model_name']}<br>
                <strong>参数数量:</strong> {model_info['parameters']:,} ({model_info['parameters_millions']:.2f}M)<br>
                <strong>输入尺寸:</strong> {model_info['input_size']}<br>
                <strong>类别数量:</strong> {model_info['num_classes']}<br>
                <strong>架构:</strong> {model_info['architecture']}
            </div>
        </div>
        
        <div class="section">
            <h2>性能指标</h2>
            <div class="metric">
                <strong>最佳验证准确率:</strong> <span class="highlight">{model_info['best_val_accuracy']:.2f}%</span><br>
                <strong>测试准确率:</strong> <span class="highlight">{model_info['test_accuracy']:.2f}%</span><br>
                <strong>测试损失:</strong> {model_info['test_loss']:.4f}
            </div>
        </div>
        
        <div class="section">
            <h2>分类报告</h2>
            <table>
                <tr>
                    <th>类别</th>
                    <th>精确率</th>
                    <th>召回率</th>
                    <th>F1分数</th>
                    <th>支持数</th>
                </tr>
                <tr>
                    <td>Negative</td>
                    <td>{classification_report['Negative']['precision']:.3f}</td>
                    <td>{classification_report['Negative']['recall']:.3f}</td>
                    <td>{classification_report['Negative']['f1-score']:.3f}</td>
                    <td>{classification_report['Negative']['support']}</td>
                </tr>
                <tr>
                    <td>Positive</td>
                    <td>{classification_report['Positive']['precision']:.3f}</td>
                    <td>{classification_report['Positive']['recall']:.3f}</td>
                    <td>{classification_report['Positive']['f1-score']:.3f}</td>
                    <td>{classification_report['Positive']['support']}</td>
                </tr>
                <tr style="background-color: #f0f8ff;">
                    <td><strong>宏平均</strong></td>
                    <td><strong>{classification_report['macro avg']['precision']:.3f}</strong></td>
                    <td><strong>{classification_report['macro avg']['recall']:.3f}</strong></td>
                    <td><strong>{classification_report['macro avg']['f1-score']:.3f}</strong></td>
                    <td><strong>{classification_report['macro avg']['support']}</strong></td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>可视化结果</h2>
            <div style="text-align: center;">
                <img src="confusion_matrix.png" alt="混淆矩阵" style="max-width: 45%; margin: 10px;">
                <img src="roc_curve.png" alt="ROC曲线" style="max-width: 45%; margin: 10px;">
            </div>
        </div>
        
        <div class="section">
            <h2>模型特点</h2>
            <ul>
                <li><strong>架构优势:</strong> EfficientNet V2改进的训练效率和模型结构</li>
                <li><strong>参数规模:</strong> 20.83M参数，属于中等规模模型</li>
                <li><strong>适用场景:</strong> 需要平衡性能和效率的应用场景</li>
                <li><strong>训练特点:</strong> 使用余弦退火学习率调度器，早停机制防止过拟合</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # 需要导入pandas用于时间戳
    import pandas as pd
    
    with open(report_dir / 'evaluation_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == "__main__":
    evaluate_efficientnet_v2_s()