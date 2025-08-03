#!/usr/bin/env python3
"""
修复模型加载问题 - 处理带有base_model前缀的权重文件
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
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import BioastDataset
from models.mic_mobilenetv3 import MIC_MobileNetV3
from models.micro_vit import MicroViT
from models.airbubble_hybrid_net import AirBubbleHybridNet

class FixedModelTester:
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
    
    def fix_state_dict_keys(self, state_dict, has_base_model_prefix=True):
        """修复状态字典的键名"""
        if has_base_model_prefix:
            # 移除 base_model. 前缀
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('base_model.'):
                    new_key = key[11:]  # 移除 'base_model.' 前缀
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict
        else:
            return state_dict
    
    def load_model_with_fix(self, model_name, model_path):
        """加载模型并修复权重键名问题"""
        print(f"加载模型: {model_name}")
        
        # 根据模型名称创建模型实例
        if model_name == 'mic_mobilenetv3':
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
        
        # 检查是否有base_model前缀
        has_base_model_prefix = any(key.startswith('base_model.') for key in state_dict.keys())
        
        if has_base_model_prefix:
            print(f"检测到base_model前缀，正在修复...")
            state_dict = self.fix_state_dict_keys(state_dict, True)
        
        # 尝试加载权重
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ 成功加载权重")
        except RuntimeError as e:
            print(f"❌ 严格模式加载失败: {e}")
            # 尝试非严格模式
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"⚠️ 非严格模式加载成功")
                if missing_keys:
                    print(f"缺失的键: {len(missing_keys)} 个")
                if unexpected_keys:
                    print(f"意外的键: {len(unexpected_keys)} 个")
            except Exception as e2:
                print(f"❌ 非严格模式也失败: {e2}")
                raise e2
        
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
                
                # 处理模型输出可能是字典的情况
                if isinstance(outputs, dict):
                    print(f"模型输出是字典，键: {list(outputs.keys())}")
                    # 如果是字典，尝试获取logits或主要输出
                    if 'logits' in outputs:
                        outputs = outputs['logits']
                    elif 'output' in outputs:
                        outputs = outputs['output']
                    elif 'classification' in outputs:
                        outputs = outputs['classification']
                    elif 'pred' in outputs:
                        outputs = outputs['pred']
                    else:
                        # 取字典中第一个张量值
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor) and value.dim() == 2:
                                outputs = value
                                print(f"使用键 '{key}' 的输出: {outputs.shape}")
                                break
                        else:
                            raise ValueError(f"无法从字典输出中找到合适的张量: {outputs.keys()}")
                
                # 确保outputs是张量
                if not isinstance(outputs, torch.Tensor):
                    raise TypeError(f"模型输出不是张量: {type(outputs)}")
                
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
    
    def test_failed_models(self):
        """测试之前失败的模型"""
        failed_experiments = [
            ('experiments/experiment_20250803_101438/mic_mobilenetv3', 'mic_mobilenetv3'),
            ('experiments/experiment_20250803_102845/micro_vit', 'micro_vit'),
            ('experiments/experiment_20250803_115344/airbubble_hybrid_net', 'airbubble_hybrid_net')
        ]
        
        success_count = 0
        
        for experiment_path, model_name in failed_experiments:
            print(f"\n{'='*60}")
            print(f"修复测试: {experiment_path}")
            print(f"模型: {model_name}")
            
            model_path = os.path.join(experiment_path, 'best_model.pth')
            results_path = os.path.join(experiment_path, 'test_results.json')
            
            # 检查是否已存在测试结果
            if os.path.exists(results_path):
                print(f"测试结果已存在: {results_path}")
                success_count += 1
                continue
            
            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                continue
            
            try:
                # 加载和测试模型
                model = self.load_model_with_fix(model_name, model_path)
                results = self.evaluate_model(model)
                
                # 保存结果
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ 测试完成! 准确率: {results['accuracy']:.4f}")
                print(f"结果已保存到: {results_path}")
                success_count += 1
                
            except Exception as e:
                print(f"❌ 测试失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"修复测试完成! 成功: {success_count}/{len(failed_experiments)}")
        
        return success_count == len(failed_experiments)

def main():
    """主函数"""
    print("🔧 开始修复失败的模型测试...")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = FixedModelTester()
    success = tester.test_failed_models()
    
    if success:
        print("🎉 所有失败的模型都已成功修复!")
    else:
        print("⚠️ 部分模型仍然存在问题")

if __name__ == "__main__":
    main()