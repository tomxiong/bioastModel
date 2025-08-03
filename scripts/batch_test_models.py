#!/usr/bin/env python3
"""
批量测试模型脚本 - 为缺失test_results.json的模型重新生成测试结果
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import BioastDataset
from models.efficientnet import create_efficientnet_b0
from models.convnext_tiny import ConvNextTiny
from models.vit_tiny import create_vit_tiny
from models.coatnet import CoAtNet
from models.mic_mobilenetv3 import MIC_MobileNetV3
from models.micro_vit import MicroViT
from models.airbubble_hybrid_net import AirBubbleHybridNet

class ModelTester:
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
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        print(f"测试集大小: {len(self.test_dataset)}")
    
    def load_model(self, model_name, model_path, config_path=None):
        """加载指定的模型"""
        print(f"加载模型: {model_name}")
        
        # 根据模型名称创建模型实例
        if model_name == 'efficientnet_b0':
            model = create_efficientnet_b0(num_classes=2)
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
        else:
            raise ValueError(f"未知的模型名称: {model_name}")
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_model(self, model):
        """评估模型性能"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"处理批次: {batch_idx}/{len(self.test_loader)}")
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
        
        # AUC计算
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except:
            auc = 0.0
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        # 敏感性和特异性
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 分类报告
        class_report = classification_report(all_labels, all_preds, target_names=['negative', 'positive'])
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'confusion_matrix': cm.tolist(),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'classification_report': class_report,
            'f1': float(f1)  # 兼容性字段
        }
        
        return results
    
    def test_experiment(self, experiment_path, model_name):
        """测试单个实验"""
        print(f"\n{'='*50}")
        print(f"测试实验: {experiment_path}")
        print(f"模型: {model_name}")
        
        model_path = os.path.join(experiment_path, 'best_model.pth')
        config_path = os.path.join(experiment_path, 'config.json')
        results_path = os.path.join(experiment_path, 'test_results.json')
        
        # 检查是否已存在测试结果
        if os.path.exists(results_path):
            print(f"测试结果已存在: {results_path}")
            return True
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return False
        
        try:
            # 加载和测试模型
            model = self.load_model(model_name, model_path, config_path)
            results = self.evaluate_model(model)
            
            # 保存结果
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"测试完成! 准确率: {results['accuracy']:.4f}")
            print(f"结果已保存到: {results_path}")
            
            return True
            
        except Exception as e:
            print(f"测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    # 需要测试的实验配置
    experiments_to_test = [
        ('experiments/experiment_20250802_140818/efficientnet_b0', 'efficientnet_b0'),
        ('experiments/experiment_20250802_231639/convnext_tiny', 'convnext_tiny'),
        ('experiments/experiment_20250803_020217/vit_tiny', 'vit_tiny'),
        ('experiments/experiment_20250803_032628/coatnet', 'coatnet'),
        ('experiments/experiment_20250803_101438/mic_mobilenetv3', 'mic_mobilenetv3'),
        ('experiments/experiment_20250803_102845/micro_vit', 'micro_vit'),
        ('experiments/experiment_20250803_115344/airbubble_hybrid_net', 'airbubble_hybrid_net')
    ]
    
    # 创建测试器
    tester = ModelTester()
    
    # 统计结果
    success_count = 0
    total_count = len(experiments_to_test)
    
    print(f"开始批量测试 {total_count} 个模型...")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 逐个测试
    for experiment_path, model_name in experiments_to_test:
        if tester.test_experiment(experiment_path, model_name):
            success_count += 1
    
    # 输出总结
    print(f"\n{'='*50}")
    print(f"批量测试完成!")
    print(f"成功: {success_count}/{total_count}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("所有模型测试成功! 可以进行下一步分析。")
    else:
        print("部分模型测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()