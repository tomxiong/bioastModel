#!/usr/bin/env python3
"""
为所有模型生成错误样本分析
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
from PIL import Image
import shutil
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import BioastDataset
from models.efficientnet import EfficientNetCustom
from models.resnet_improved import ResNetImproved
from models.convnext_tiny import ConvNextTiny
from models.vit_tiny import VisionTransformerTiny
from models.coatnet import CoAtNet
from models.mic_mobilenetv3 import MIC_MobileNetV3
from models.micro_vit import MicroViT
from models.airbubble_hybrid_net import AirBubbleHybridNet

class ErrorAnalysisGenerator:
    def __init__(self, data_dir="bioast_dataset"):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((70, 70)),
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
            batch_size=1,  # 单个样本处理
            shuffle=False,
            num_workers=0
        )
        
        print(f"测试集大小: {len(self.test_dataset)}")
    
    def load_model(self, model_name, model_path):
        """加载模型"""
        print(f"加载模型: {model_name}")
        
        # 根据模型名称创建模型实例
        if model_name == 'efficientnet_b0':
            model = EfficientNetCustom(num_classes=2)
        elif model_name == 'resnet18_improved':
            from models.resnet_improved import ImprovedBasicBlock
            model = ResNetImproved(ImprovedBasicBlock, [2, 2, 2, 2], num_classes=2)
        elif model_name == 'convnext_tiny':
            model = ConvNextTiny(num_classes=2)
        elif model_name == 'vit_tiny':
            model = VisionTransformerTiny(num_classes=2)
        elif model_name == 'coatnet':
            model = CoAtNet(num_classes=2)
        elif model_name == 'mic_mobilenetv3':
            model = MIC_MobileNetV3(num_classes=2)
        elif model_name == 'micro_vit':
            model = MicroViT(num_classes=2)
        elif model_name == 'airbubble_hybrid_net':
            model = AirBubbleHybridNet(num_classes=2)
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 检查是否有base_model前缀并修复
        has_base_model_prefix = any(key.startswith('base_model.') for key in state_dict.keys())
        if has_base_model_prefix:
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('base_model.'):
                    new_key = key[11:]  # 移除 'base_model.' 前缀
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # 加载权重
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def analyze_model_errors(self, model, model_name, experiment_path):
        """分析模型的错误样本"""
        print(f"分析 {model_name} 的错误样本...")
        
        all_results = []
        error_samples = []
        correct_samples = []
        
        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = model(data)
                
                # 处理模型输出可能是字典的情况
                if isinstance(outputs, dict):
                    if 'logits' in outputs:
                        outputs = outputs['logits']
                    elif 'classification' in outputs:
                        outputs = outputs['classification']
                    elif 'output' in outputs:
                        outputs = outputs['output']
                    else:
                        # 取字典中第一个张量值
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor) and value.dim() == 2:
                                outputs = value
                                break
                
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)
                
                # 获取原始图像路径
                image_path = self.test_dataset.samples[idx][0]
                image_name = os.path.basename(image_path)
                
                result = {
                    'index': idx,
                    'image_name': image_name,
                    'image_path': image_path,
                    'true_label': target.item(),
                    'predicted_label': pred.item(),
                    'confidence': probs.max().item(),
                    'prob_negative': probs[0, 0].item(),
                    'prob_positive': probs[0, 1].item(),
                    'is_correct': pred.item() == target.item()
                }
                
                all_results.append(result)
                
                if not result['is_correct']:
                    error_samples.append(result)
                else:
                    correct_samples.append(result)
        
        # 创建样本分析目录
        analysis_dir = os.path.join(experiment_path, 'sample_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 保存详细结果到CSV
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(analysis_dir, 'detailed_predictions.csv')
        df.to_csv(csv_path, index=False)
        
        # 生成错误样本分析
        self.generate_error_sample_analysis(error_samples, correct_samples, analysis_dir, model_name)
        
        print(f"✅ {model_name} 错误分析完成")
        print(f"   总样本: {len(all_results)}")
        print(f"   错误样本: {len(error_samples)}")
        print(f"   正确样本: {len(correct_samples)}")
        print(f"   准确率: {len(correct_samples)/len(all_results)*100:.2f}%")
        
        return error_samples, correct_samples, all_results
    
    def generate_error_sample_analysis(self, error_samples, correct_samples, analysis_dir, model_name):
        """生成错误样本分析图表和文件"""
        
        # 1. 置信度分布分析
        self.plot_confidence_distribution(error_samples, correct_samples, analysis_dir, model_name)
        
        # 2. 错误类型分析
        self.plot_error_type_analysis(error_samples, analysis_dir, model_name)
        
        # 3. 生成样本图像网格
        self.generate_sample_grids(error_samples, correct_samples, analysis_dir)
        
        # 4. 生成错误样本CSV
        self.generate_error_csv(error_samples, analysis_dir)
    
    def plot_confidence_distribution(self, error_samples, correct_samples, analysis_dir, model_name):
        """绘制置信度分布图"""
        plt.figure(figsize=(12, 8))
        
        if error_samples:
            error_confidences = [s['confidence'] for s in error_samples]
            plt.hist(error_confidences, bins=20, alpha=0.7, label='Error Samples', color='red')
        
        if correct_samples:
            correct_confidences = [s['confidence'] for s in correct_samples]
            plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct Samples', color='green')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of Samples')
        plt.title(f'{model_name} - Confidence Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(analysis_dir, 'confidence_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_error_type_analysis(self, error_samples, analysis_dir, model_name):
        """绘制错误类型分析图"""
        if not error_samples:
            return
        
        # 统计错误类型
        false_positives = [s for s in error_samples if s['true_label'] == 0 and s['predicted_label'] == 1]
        false_negatives = [s for s in error_samples if s['true_label'] == 1 and s['predicted_label'] == 0]
        
        # 绘制错误类型分布
        plt.figure(figsize=(10, 6))
        error_types = ['False Positives', 'False Negatives']
        error_counts = [len(false_positives), len(false_negatives)]
        
        bars = plt.bar(error_types, error_counts, color=['orange', 'red'], alpha=0.7)
        plt.title(f'{model_name} - Error Type Analysis')
        plt.ylabel('Number of Errors')
        
        # 添加数值标签
        for bar, count in zip(bars, error_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(analysis_dir, 'error_type_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_sample_grids(self, error_samples, correct_samples, analysis_dir):
        """生成样本图像网格"""
        # 生成错误样本网格
        if error_samples:
            self.create_sample_grid(error_samples[:16], analysis_dir, 'error_samples_grid.png', 'Error Samples')
        
        # 生成高置信度正确样本网格
        high_conf_correct = [s for s in correct_samples if s['confidence'] > 0.9]
        if high_conf_correct:
            self.create_sample_grid(high_conf_correct[:16], analysis_dir, 'high_confidence_correct_grid.png', 'High Confidence Correct')
        
        # 生成低置信度正确样本网格
        low_conf_correct = [s for s in correct_samples if s['confidence'] < 0.7]
        if low_conf_correct:
            self.create_sample_grid(low_conf_correct[:16], analysis_dir, 'low_confidence_correct_grid.png', 'Low Confidence Correct')
    
    def create_sample_grid(self, samples, analysis_dir, filename, title):
        """创建样本图像网格"""
        if not samples:
            return
        
        n_samples = min(len(samples), 16)
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row = i // n_cols
            col = i % n_cols
            
            sample = samples[i]
            
            # 加载并显示图像
            try:
                image = Image.open(sample['image_path']).convert('RGB')
                axes[row, col].imshow(image)
                
                # 设置标题
                true_label = 'Pos' if sample['true_label'] == 1 else 'Neg'
                pred_label = 'Pos' if sample['predicted_label'] == 1 else 'Neg'
                conf = sample['confidence']
                
                title_text = f"True: {true_label}, Pred: {pred_label}\nConf: {conf:.3f}"
                axes[row, col].set_title(title_text, fontsize=10)
                axes[row, col].axis('off')
                
            except Exception as e:
                axes[row, col].text(0.5, 0.5, f'Error loading\n{sample["image_name"]}', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_error_csv(self, error_samples, analysis_dir):
        """生成错误样本CSV文件"""
        if not error_samples:
            return
        
        # 创建错误样本DataFrame
        error_df = pd.DataFrame(error_samples)
        
        # 添加错误类型列
        error_df['error_type'] = error_df.apply(
            lambda row: 'False Positive' if row['true_label'] == 0 and row['predicted_label'] == 1 
            else 'False Negative', axis=1
        )
        
        # 保存到CSV
        csv_path = os.path.join(analysis_dir, 'error_samples.csv')
        error_df.to_csv(csv_path, index=False)
        
        print(f"✅ 错误样本CSV已保存: {csv_path}")
    
    def generate_all_error_analyses(self):
        """为所有模型生成错误分析"""
        experiments = [
            ('experiments/experiment_20250802_140818/efficientnet_b0', 'efficientnet_b0'),
            ('experiments/experiment_20250802_164948/resnet18_improved', 'resnet18_improved'),
            ('experiments/experiment_20250802_231639/convnext_tiny', 'convnext_tiny'),
            ('experiments/experiment_20250803_020217/vit_tiny', 'vit_tiny'),
            ('experiments/experiment_20250803_032628/coatnet', 'coatnet'),
            ('experiments/experiment_20250803_101438/mic_mobilenetv3', 'mic_mobilenetv3'),
            ('experiments/experiment_20250803_102845/micro_vit', 'micro_vit'),
            ('experiments/experiment_20250803_115344/airbubble_hybrid_net', 'airbubble_hybrid_net')
        ]
        
        success_count = 0
        
        for experiment_path, model_name in experiments:
            print(f"\n{'='*60}")
            print(f"处理实验: {experiment_path}")
            print(f"模型: {model_name}")
            
            model_path = os.path.join(experiment_path, 'best_model.pth')
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                continue
            
            try:
                # 加载模型
                model = self.load_model(model_name, model_path)
                
                # 分析错误样本
                error_samples, correct_samples, all_results = self.analyze_model_errors(
                    model, model_name, experiment_path
                )
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ 处理失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"错误分析完成! 成功: {success_count}/{len(experiments)}")
        
        return success_count == len(experiments)

def main():
    """主函数"""
    print("🔍 开始生成所有模型的错误样本分析...")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    generator = ErrorAnalysisGenerator()
    success = generator.generate_all_error_analyses()
    
    if success:
        print("🎉 所有模型的错误分析都已完成!")
    else:
        print("⚠️ 部分模型的错误分析失败")

if __name__ == "__main__":
    main()