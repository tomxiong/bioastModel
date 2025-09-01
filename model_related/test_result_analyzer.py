#!/usr/bin/env python3
"""
测试结果分析器 - 详细分析模型测试结果，识别失败样本并生成改进建议
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
from PIL import Image
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from training_index_manager import TrainingIndexManager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import BioastDataset
from models.efficientnet import create_efficientnet_b0
from models.efficientnet_v2 import create_efficientnetv2_s
from models.convnext_tiny import ConvNextTiny
from models.vit_tiny import create_vit_tiny
from models.coatnet import CoAtNet
from models.mic_mobilenetv3 import MIC_MobileNetV3
from models.micro_vit import MicroViT
from models.airbubble_hybrid_net import AirBubbleHybridNet
from models.resnet_improved import create_resnet18_improved, create_resnet34_improved, create_resnet50_improved
from models.enhanced_airbubble_detector import EnhancedAirBubbleDetector
from models.mobilenet_v3 import create_mobilenetv3_small
from models.shufflenet_v2 import create_shufflenetv2_x0_5

class TestResultAnalyzer:
    def __init__(self, data_dir="bioast_dataset"):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['negative', 'positive']
        # 类别映射 - 生物抗菌素敏感性测试
        # positive: 有菌生长（黑色阴影/菌落聚集）
        # negative: 菌生长抑制/无菌（清亮，可能有气孔环形阴影）
        self.class_to_idx = {'negative': 0, 'positive': 1}
        self.idx_to_class = {0: 'negative', 1: 'positive'}
        self.class_descriptions = {
            0: '阴性 (菌生长抑制/无菌)',
            1: '阳性 (有菌生长)'
        }
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载测试数据集
        self.test_dataset = BioastDataset(
            data_dir=self.data_dir,
            split='test',
            transform=self.transform
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"测试数据集加载完成: {len(self.test_dataset)} 个样本")
    
    def _calculate_root_relative_path(self, save_dir):
        """计算从分析目录到项目根目录的相对路径"""
        # 计算分析目录相对于项目根目录的深度
        analysis_parts = Path(save_dir).parts
        # 过滤掉当前目录和父目录标记
        real_parts = [p for p in analysis_parts if p not in ['.', '..']]
        
        # 新的目录结构: checkpoints/model_name/train_xxxx/test_analysis/failed_samples_analysis/
        # 需要5层../才能到达项目根目录
        if 'checkpoints' in real_parts:
            # 找到checkpoints在路径中的位置
            checkpoints_index = real_parts.index('checkpoints')
            # 计算从当前目录到项目根目录的深度（checkpoints的上一级）
            depth_to_root = len(real_parts) - checkpoints_index
            return '../' * depth_to_root
        else:
            # 如果没有checkpoints，使用原来的逻辑
            depth = len(real_parts)
            return '../' * depth
    
    def _generate_html_failed_samples_report(self, df_failed, failed_analysis, html_path, root_relative_path):
        """生成HTML版本的失败样本报告"""
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>失败样本分析报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #007bff;
            border-left: 4px solid #007bff;
            padding-left: 15px;
            margin-top: 30px;
        }}
        h3 {{
            color: #495057;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #28a745;
        }}
        .stats {{
            background-color: #e9ecef;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .stats ul {{
            list-style-type: none;
            padding: 0;
        }}
        .stats li {{
            padding: 8px 0;
            border-bottom: 1px solid #dee2e6;
        }}
        .stats li:last-child {{
            border-bottom: none;
        }}
        .sample-card {{
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin: 20px 0;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .sample-info {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .info-item {{
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
        .info-label {{
            font-weight: bold;
            color: #495057;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .sample-image {{
            max-width: 100%;
            height: auto;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .false-positive {{
            border-left: 4px solid #dc3545;
        }}
        .false-negative {{
            border-left: 4px solid #ffc107;
        }}
        .timestamp {{
            text-align: center;
            color: #6c757d;
            font-style: italic;
            margin-bottom: 30px;
        }}
        .section-count {{
            background-color: #007bff;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>失败样本分析报告</h1>
        <div class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <h2>总体统计</h2>
        <div class="stats">
            <ul>
                <li><span class="info-label">总失败样本数:</span> {len(df_failed)}</li>
                <li><span class="info-label">假阳性数量:</span> {failed_analysis['error_types']['false_positives']}</li>
                 <li><span class="info-label">假阴性数量:</span> {failed_analysis['error_types']['false_negatives']}</li>
                 <li><span class="info-label">平均置信度:</span> {failed_analysis['avg_failed_confidence']:.4f}</li>
            </ul>
        </div>
"""
        
        # 按错误类型分组显示
        false_positives = df_failed[df_failed['true_label'] == 0]
        false_negatives = df_failed[df_failed['true_label'] == 1]
        
        if len(false_positives) > 0:
            html_content += f"""
        <h2>假阳性样本 (误判为有菌生长)<span class="section-count">{len(false_positives)}</span></h2>
"""
            
            for idx, row in false_positives.iterrows():
                 filename = Path(row['image_path']).name
                 relative_image_path = f"{root_relative_path}{row['image_path'].replace(chr(92), '/')}"
                 html_content += f"""
         <div class="sample-card false-positive">
             <h3>{filename}</h3>
            <div class="sample-info">
                <div class="info-item">
                    <div class="info-label">真实标签:</div>
                    <div>{row['true_class_name']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">预测标签:</div>
                    <div>{row['predicted_class_name']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">置信度:</div>
                    <div>{row['confidence']:.4f}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">错误类型:</div>
                    <div>{row['error_type']}</div>
                </div>
            </div>
            <div class="image-container">
                 <img src="{relative_image_path}" alt="{filename}" class="sample-image" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                 <div style="display:none; color:#dc3545; font-style:italic;">图片文件不存在: {filename}</div>
             </div>
        </div>
"""
        
        if len(false_negatives) > 0:
            html_content += f"""
        <h2>假阴性样本 (漏检菌生长)<span class="section-count">{len(false_negatives)}</span></h2>
"""
            
            for idx, row in false_negatives.iterrows():
                 filename = Path(row['image_path']).name
                 relative_image_path = f"{root_relative_path}{row['image_path'].replace(chr(92), '/')}"
                 html_content += f"""
         <div class="sample-card false-negative">
             <h3>{filename}</h3>
            <div class="sample-info">
                <div class="info-item">
                    <div class="info-label">真实标签:</div>
                    <div>{row['true_class_name']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">预测标签:</div>
                    <div>{row['predicted_class_name']}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">置信度:</div>
                    <div>{row['confidence']:.4f}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">错误类型:</div>
                    <div>{row['error_type']}</div>
                </div>
            </div>
            <div class="image-container">
                 <img src="{relative_image_path}" alt="{filename}" class="sample-image" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                 <div style="display:none; color:#dc3545; font-style:italic;">图片文件不存在: {filename}</div>
             </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # 写入HTML文件
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def load_model(self, model_name, model_path):
        """加载训练好的模型"""
        try:
            if model_name == 'efficientnet_b0':
                model = create_efficientnet_b0(num_classes=2)
            elif model_name == 'efficientnet_b1':
                model = create_efficientnetv2_s(num_classes=2)
            elif model_name == 'convnext_tiny':
                model = ConvNextTiny(num_classes=2)
            elif model_name == 'vit_tiny':
                model = create_vit_tiny(num_classes=2)
            elif model_name == 'coatnet':
                model = CoAtNet(num_classes=2)
            elif model_name == 'mic_mobilenetv3':
                model = MIC_MobileNetV3(num_classes=2)
            elif model_name == 'micro_vit':
                model = MicroViT(num_classes=2)
            elif model_name == 'airbubble_hybrid_net':
                model = AirBubbleHybridNet(num_classes=2)
            elif model_name == 'resnet18_improved':
                model = create_resnet18_improved(num_classes=2)
            elif model_name == 'resnet34_improved':
                model = create_resnet34_improved(num_classes=2)
            elif model_name == 'resnet50_improved':
                model = create_resnet50_improved(num_classes=2)
            elif model_name == 'enhanced_airbubble_detector':
                model = EnhancedAirBubbleDetector(num_classes=2)
            elif model_name == 'mobilenetv3_small':
                model = create_mobilenetv3_small(num_classes=2)
            elif model_name == 'shufflenetv2_x0_5':
                model = create_shufflenetv2_x0_5(num_classes=2)
            else:
                raise ValueError(f"未知的模型类型: {model_name}")
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return None
    
    def analyze_test_results(self, model, save_dir):
        """详细分析测试结果，识别失败样本"""
        model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_image_paths = []
        failed_samples = []
        
        print("开始测试模型...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 识别失败样本
                for i in range(len(predictions)):
                    sample_idx = batch_idx * self.test_loader.batch_size + i
                    if sample_idx < len(self.test_dataset):
                        image_path = self.test_dataset.samples[sample_idx][0]
                        all_image_paths.append(image_path)
                        
                        if predictions[i] != labels[i]:
                            failed_samples.append({
                                'image_path': image_path,
                                'true_label': labels[i].item(),
                                'predicted_label': predictions[i].item(),
                                'confidence': probabilities[i][predictions[i]].item(),
                                'true_class_prob': probabilities[i][labels[i]].item(),
                                'sample_idx': sample_idx
                            })
        
        # 计算基本指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # 计算AUC
        auc = roc_auc_score(all_labels, [prob[1] for prob in all_probabilities])
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 分类报告
        class_report = classification_report(
            all_labels, all_predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # 分析失败样本
        failed_analysis = self.analyze_failed_samples(failed_samples)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(all_labels),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'failed_samples_count': len(failed_samples),
            'failed_samples': failed_samples,
            'failed_analysis': failed_analysis
        }
        
        # 保存详细结果
        results_path = save_dir / 'detailed_test_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成可视化报告
        self.generate_visual_report(results, save_dir)
        
        # 生成失败样本报告
        self.generate_failed_samples_report(failed_samples, failed_analysis, save_dir)
        
        print(f"\n测试结果分析完成:")
        print(f"准确率: {accuracy:.4f}")
        print(f"失败样本数: {len(failed_samples)}/{len(all_labels)}")
        print(f"结果保存到: {save_dir}")
        
        return results
    
    def analyze_failed_samples(self, failed_samples):
        """分析失败样本的特征"""
        if not failed_samples:
            return {'message': '没有失败样本'}
        
        # 按置信度分组
        high_conf_errors = [s for s in failed_samples if s['confidence'] > 0.8]
        medium_conf_errors = [s for s in failed_samples if 0.5 < s['confidence'] <= 0.8]
        low_conf_errors = [s for s in failed_samples if s['confidence'] <= 0.5]
        
        # 按错误类型分组
        false_positives = [s for s in failed_samples if s['true_label'] == 0 and s['predicted_label'] == 1]  # 误判为有菌生长
        false_negatives = [s for s in failed_samples if s['true_label'] == 1 and s['predicted_label'] == 0]  # 漏检菌生长
        
        analysis = {
            'total_failed': len(failed_samples),
            'confidence_distribution': {
                'high_confidence_errors': len(high_conf_errors),
                'medium_confidence_errors': len(medium_conf_errors),
                'low_confidence_errors': len(low_conf_errors)
            },
            'error_types': {
                'false_positives': len(false_positives),
                'false_negatives': len(false_negatives)
            },
            'avg_failed_confidence': np.mean([s['confidence'] for s in failed_samples]),
            'avg_true_class_prob': np.mean([s['true_class_prob'] for s in failed_samples])
        }
        
        return analysis
    
    def generate_visual_report(self, results, save_dir):
        """生成可视化报告"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型测试结果详细分析', fontsize=16, fontweight='bold')
        
        # 1. 混淆矩阵
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   ax=axes[0,0])
        axes[0,0].set_title('混淆矩阵')
        axes[0,0].set_xlabel('预测标签')
        axes[0,0].set_ylabel('真实标签')
        
        # 2. 性能指标
        metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC']
        values = [results['accuracy'], results['precision'], 
                 results['recall'], results['f1_score'], results['auc']]
        
        bars = axes[0,1].bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[0,1].set_title('性能指标')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].set_ylabel('分数')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 3. 失败样本置信度分布
        if results['failed_samples']:
            confidences = [s['confidence'] for s in results['failed_samples']]
            axes[1,0].hist(confidences, bins=20, alpha=0.7, color='red', edgecolor='black')
            axes[1,0].set_title(f'失败样本置信度分布 (共{len(confidences)}个)')
            axes[1,0].set_xlabel('预测置信度')
            axes[1,0].set_ylabel('样本数量')
            axes[1,0].axvline(np.mean(confidences), color='darkred', linestyle='--', 
                             label=f'平均值: {np.mean(confidences):.3f}')
            axes[1,0].legend()
        else:
            axes[1,0].text(0.5, 0.5, '没有失败样本', ha='center', va='center', 
                          transform=axes[1,0].transAxes, fontsize=14)
            axes[1,0].set_title('失败样本置信度分布')
        
        # 4. 错误类型分析
        if results['failed_analysis']['error_types']:
            error_types = ['假阳性\n(误判为有菌生长)', '假阴性\n(漏检菌生长)']
            error_counts = [results['failed_analysis']['error_types']['false_positives'],
                           results['failed_analysis']['error_types']['false_negatives']]
            
            colors = ['#ff9999', '#66b3ff']
            bars = axes[1,1].bar(error_types, error_counts, color=colors)
            axes[1,1].set_title('错误类型分析')
            axes[1,1].set_ylabel('错误数量')
            
            # 添加数值标签
            for bar, count in zip(bars, error_counts):
                if count > 0:
                    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                  str(count), ha='center', va='bottom')
        else:
            axes[1,1].text(0.5, 0.5, '没有错误样本', ha='center', va='center', 
                          transform=axes[1,1].transAxes, fontsize=14)
            axes[1,1].set_title('错误类型分析')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'test_analysis_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化报告已生成: test_analysis_report.png")
    
    def generate_failed_samples_report(self, failed_samples, failed_analysis, save_dir):
        """生成失败样本详细报告（Markdown和HTML版本）"""
        if not failed_samples:
            return
        
        # 创建失败样本目录
        failed_dir = save_dir / 'failed_samples_analysis'
        failed_dir.mkdir(exist_ok=True)
        
        # 生成CSV报告
        df_failed = pd.DataFrame(failed_samples)
        df_failed['true_class_name'] = df_failed['true_label'].map({0: '阴性 (菌生长抑制)', 1: '阳性 (有菌生长)'})
        df_failed['predicted_class_name'] = df_failed['predicted_label'].map({0: '阴性 (菌生长抑制)', 1: '阳性 (有菌生长)'})
        df_failed['error_type'] = df_failed.apply(
            lambda x: '假阳性 (误判为有菌生长)' if x['true_label'] == 0 else '假阴性 (漏检菌生长)', axis=1
        )
        
        # 按置信度排序
        df_failed = df_failed.sort_values('confidence', ascending=False)
        
        csv_path = failed_dir / 'failed_samples_detail.csv'
        df_failed.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 计算相对路径（从failed_samples_analysis目录到项目根目录）
        root_relative_path = self._calculate_root_relative_path(failed_dir)
        
        # 生成Markdown报告
        md_content = f"""# 失败样本详细分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 数据集说明

本数据集用于生物抗菌素敏感性测试分析：
- **阳性 (positive)**: 有菌生长，表现为明显菌落聚集成黑色阴影或弱生长的小黑色阴影
- **阴性 (negative)**: 菌生长抑制或无菌，一般为清亮，但部分样本因气孔影响出现环形黑色阴影

## 总体统计

- **总失败样本数**: {len(failed_samples)}
- **平均预测置信度**: {failed_analysis['avg_failed_confidence']:.3f}
- **平均真实类别概率**: {failed_analysis['avg_true_class_prob']:.3f}

## 置信度分布

- **高置信度错误** (>0.8): {failed_analysis['confidence_distribution']['high_confidence_errors']} 个
- **中等置信度错误** (0.5-0.8): {failed_analysis['confidence_distribution']['medium_confidence_errors']} 个
- **低置信度错误** (≤0.5): {failed_analysis['confidence_distribution']['low_confidence_errors']} 个

## 错误类型分析

- **假阳性** (误判为有菌生长): {failed_analysis['error_types']['false_positives']} 个
- **假阴性** (漏检菌生长): {failed_analysis['error_types']['false_negatives']} 个

## 改进建议

### 基于错误分析的建议:

1. **高置信度错误处理**:
   - 检查这些样本是否存在标注错误
   - 考虑这些样本可能代表边界情况，需要更多类似样本进行训练

2. **假阳性问题** (误判为有菌生长):
   - 增加阴性样本的多样性（不同的清亮状态）
   - 调整决策阈值，提高判断标准
   - 关注可能与菌生长相似的干扰因素（如气孔环形阴影）

3. **假阴性问题** (漏检菌生长):
   - 增加轻微菌生长、模糊黑色阴影等难检测样本
   - 降低决策阈值，提高敏感性
   - 加强数据增强，提高模型鲁棒性

## 详细失败样本列表

| 序号 | 图片 | 图片路径 | 真实标签 | 预测标签 | 置信度 | 错误类型 |
|------|------|----------|----------|----------|--------|----------|
"""
        
        for i, sample in enumerate(df_failed.head(50).to_dict('records'), 1):
            # 生成图片的相对路径（使用计算出的相对路径）
            image_path = sample['image_path']
            relative_image_path = f"{root_relative_path}{image_path.replace(chr(92), '/')}"
            # 创建可点击的图片链接
            image_markdown = f"[![样本{i}]({relative_image_path})]({relative_image_path})"
            
            md_content += f"| {i} | {image_markdown} | {sample['image_path']} | {sample['true_class_name']} | {sample['predicted_class_name']} | {sample['confidence']:.3f} | {sample['error_type']} |\n"
        
        if len(failed_samples) > 50:
            md_content += f"\n*注: 仅显示前50个失败样本，完整列表请查看 failed_samples_detail.csv*\n"
        
        md_content += f"\n\n## 文件说明\n\n- `failed_samples_detail.csv`: 完整的失败样本列表\n- `test_analysis_report.png`: 可视化分析图表\n- `detailed_test_results.json`: 完整的测试结果数据\n\n---\n*此报告由测试结果分析器自动生成*\n"
        
        md_path = failed_dir / 'failed_samples_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # 生成HTML报告
        html_path = failed_dir / 'failed_samples_report.html'
        self._generate_html_failed_samples_report(df_failed, failed_analysis, html_path, root_relative_path)
        
        print(f"失败样本分析报告已生成:")
        print(f"- CSV文件: {csv_path}")
        print(f"- Markdown报告: {md_path}")
        print(f"- HTML报告: {html_path}")
    
    def test_model_from_experiment(self, experiment_path):
        """从实验路径测试模型"""
        experiment_path = Path(experiment_path)
        
        # 查找模型文件（在子目录中）
        model_files = list(experiment_path.rglob('*.pth'))
        if not model_files:
            print(f"在 {experiment_path} 中未找到模型文件")
            return None
        
        # 优先选择best.pth文件
        model_file = None
        for f in model_files:
            if 'best' in f.name.lower():
                model_file = f
                break
        
        if model_file is None:
            model_file = model_files[0]  # 使用第一个找到的模型文件
        
        # 从模型文件路径推断模型名称
        # 检查是否是新的训练ID目录结构 (checkpoints/model_name/train_xxxx/)
        path_parts = model_file.parts
        model_name = None
        
        # 查找checkpoints目录的位置
        if 'checkpoints' in path_parts:
            checkpoints_idx = path_parts.index('checkpoints')
            if checkpoints_idx + 1 < len(path_parts):
                # checkpoints后面的第一个目录应该是模型名称
                model_name = path_parts[checkpoints_idx + 1]
        
        # 如果没有找到checkpoints结构，使用父目录名
        if model_name is None:
            model_name = model_file.parent.name
        
        # 如果模型名称看起来像训练ID（train_xxxx），尝试从更高层级获取
        if model_name.startswith('train_'):
            # 尝试从父目录的父目录获取模型名称
            if len(model_file.parts) >= 3:
                model_name = model_file.parts[-3]  # 往上两级目录
        
        print(f"测试模型: {model_name}")
        print(f"模型文件: {model_file}")
        
        # 加载模型
        model = self.load_model(model_name, model_file)
        if model is None:
            return None
        
        # 创建结果保存目录（在模型目录下）
        save_dir = model_file.parent / 'test_analysis'
        save_dir.mkdir(exist_ok=True)
        
        # 分析测试结果
        results = self.analyze_test_results(model, save_dir)
        
        return results

def main():
    """主函数 - 批量测试所有需要补充测试的模型"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试结果分析器')
    parser.add_argument('--force', action='store_true', help='强制重新生成所有报告，即使已存在')
    parser.add_argument('--model', type=str, help='指定要测试的模型名称')
    parser.add_argument('--training-id', type=str, help='指定训练ID进行测试')
    parser.add_argument('--use-index', action='store_true', help='使用训练索引管理器')
    args = parser.parse_args()
    
    analyzer = TestResultAnalyzer()
    
    # 初始化训练索引管理器
    index_manager = TrainingIndexManager() if args.use_index else None
    
    tested_count = 0
    total_experiments = 0
    
    print("开始批量测试模型...\n")
    
    # 如果使用索引管理器和指定训练ID
    if args.use_index and args.training_id:
        training_info = index_manager.get_training_info(args.training_id)
        if not training_info:
            print(f"错误: 未找到训练记录 {args.training_id}")
            return
        
        model_path = Path(training_info['paths']['checkpoint_dir'])
        
        # 检查是否已有详细测试结果（除非使用--force参数）
        detailed_results_path = model_path / 'test_analysis' / 'detailed_test_results.json'
        if detailed_results_path.exists() and not args.force:
            print(f"⏭️  跳过训练 {args.training_id} (已有详细测试结果)")
            return
        
        print(f"\n{'='*60}")
        print(f"测试训练: {args.training_id}")
        print(f"模型: {training_info['model_name']}")
        print(f"{'='*60}")
        
        try:
            results = analyzer.test_model_from_experiment(model_path)
            if results:
                tested_count += 1
                print(f"✅ 测试完成")
                
                # 更新训练记录
                metrics = {
                    "test_accuracy": results['accuracy'],
                    "test_precision": results['precision'],
                    "test_recall": results['recall'],
                    "test_f1_score": results['f1_score'],
                    "test_auc": results['auc']
                }
                files = {
                    "test_report": str(detailed_results_path)
                }
                index_manager.update_training_status(
                    training_id=args.training_id,
                    status="tested",
                    metrics=metrics,
                    files=files
                )
            else:
                print(f"❌ 测试失败")
        except Exception as e:
            print(f"❌ 测试出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        total_experiments = 1
        
    elif args.use_index and args.model:
        # 使用索引管理器获取指定模型的最新训练
        latest_training = index_manager.get_latest_training(args.model)
        if not latest_training:
            print(f"错误: 未找到模型 {args.model} 的训练记录")
            return
        
        model_path = Path(latest_training['paths']['checkpoint_dir'])
        
        # 检查是否已有详细测试结果（除非使用--force参数）
        detailed_results_path = model_path / 'test_analysis' / 'detailed_test_results.json'
        if detailed_results_path.exists() and not args.force:
            print(f"⏭️  跳过模型 {args.model} 最新训练 (已有详细测试结果)")
            return
        
        print(f"\n{'='*60}")
        print(f"测试模型: {args.model} (最新训练)")
        print(f"训练ID: {latest_training['training_id']}")
        print(f"{'='*60}")
        
        try:
            results = analyzer.test_model_from_experiment(model_path)
            if results:
                tested_count += 1
                print(f"✅ 测试完成")
                
                # 更新训练记录
                metrics = {
                    "test_accuracy": results['accuracy'],
                    "test_precision": results['precision'],
                    "test_recall": results['recall'],
                    "test_f1_score": results['f1_score'],
                    "test_auc": results['auc']
                }
                files = {
                    "test_report": str(detailed_results_path)
                }
                index_manager.update_training_status(
                    training_id=latest_training['training_id'],
                    status="tested",
                    metrics=metrics,
                    files=files
                )
            else:
                print(f"❌ 测试失败")
        except Exception as e:
            print(f"❌ 测试出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        total_experiments = 1
        
    else:
        # 传统方式：测试experiments目录中的模型
        experiments_dir = Path('experiments')
        if experiments_dir.exists():
            for model_dir in experiments_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                for exp_dir in model_dir.iterdir():
                        if not exp_dir.is_dir():
                            continue
                        
                        # 查找模型子目录
                        model_subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
                        if not model_subdirs:
                            continue
                        
                        for model_subdir in model_subdirs:
                            total_experiments += 1
                            
                            # 检查是否已有详细测试结果（除非使用--force参数）
                            detailed_results_path = model_subdir / 'test_analysis' / 'detailed_test_results.json'
                            if detailed_results_path.exists() and not args.force:
                                print(f"⏭️  跳过 {model_dir.name}/{model_subdir.name} (已有详细测试结果)")
                                continue
                            
                            # 如果指定了特定模型，只测试该模型
                            if args.model and model_subdir.name != args.model:
                                continue
                            
                            print(f"\n{'='*60}")
                            print(f"测试实验: {model_dir.name}/{model_subdir.name}")
                            print(f"{'='*60}")
                            
                            try:
                                results = analyzer.test_model_from_experiment(model_subdir)
                                if results:
                                    tested_count += 1
                                    print(f"✅ 测试完成")
                                else:
                                    print(f"❌ 测试失败")
                            except Exception as e:
                                print(f"❌ 测试出错: {str(e)}")
                                import traceback
                                traceback.print_exc()
        
        # 测试checkpoints目录中的模型
        checkpoints_dir = Path('checkpoints')
        if checkpoints_dir.exists():
            for model_dir in checkpoints_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                total_experiments += 1
                
                # 检查是否已有详细测试结果（除非使用--force参数）
                detailed_results_path = model_dir / 'test_analysis' / 'detailed_test_results.json'
                if detailed_results_path.exists() and not args.force:
                    print(f"⏭️  跳过 checkpoints/{model_dir.name} (已有详细测试结果)")
                    continue
                
                # 如果指定了特定模型，只测试该模型
                if args.model and model_dir.name != args.model:
                    continue
                
                print(f"\n{'='*60}")
                print(f"测试模型: checkpoints/{model_dir.name}")
                print(f"{'='*60}")
                
                try:
                    results = analyzer.test_model_from_experiment(model_dir)
                    if results:
                        tested_count += 1
                        print(f"✅ 测试完成")
                    else:
                        print(f"❌ 测试失败")
                except Exception as e:
                    print(f"❌ 测试出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"批量测试完成!")
    print(f"总实验数: {total_experiments}")
    print(f"新测试数: {tested_count}")
    print(f"{'='*60}")
    
    # 如果使用索引管理器，显示训练摘要
    if args.use_index:
        print("\n训练索引摘要:")
        index_manager.print_training_summary()

if __name__ == "__main__":
    main()