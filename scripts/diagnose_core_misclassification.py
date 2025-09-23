"""
核心误判问题诊断脚本
专门分析两个关键问题：
1. 阴性带气孔误判为阳性
2. 阳性强中心点或弱分散误判为阴性
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter, defaultdict

# 添加项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.multitask_mic_mobilenetv3 import create_multitask_mic_mobilenetv3
from training.enhanced_multitask_dataset import create_multitask_dataloaders

class CoreMisclassificationDiagnostic:
    """核心误判问题诊断器"""
    
    def __init__(self, experiment_dir="experiments/multitask_grayscale_focused"):
        self.experiment_dir = Path(experiment_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataloaders = None
        
        print("🔍 核心误判问题诊断器初始化")
        print(f"   实验目录: {self.experiment_dir}")
        print(f"   设备: {self.device}")
    
    def load_model_and_data(self):
        """加载模型和数据"""
        print("\n📊 加载模型和数据...")
        
        # 加载最佳模型
        model_path = self.experiment_dir / "best_model.pth"
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 创建数据加载器
        self.dataloaders = create_multitask_dataloaders(
            data_root="/home/aaa/ws/bioastModel/ds/images",
            annotations_file="m9e1n170.json",
            batch_size=64,
            num_workers=4,
            seed=42
        )
        
        # 重建模型
        dataset = next(iter(self.dataloaders.values())).dataset
        self.model = create_multitask_mic_mobilenetv3(
            num_classes=2,
            num_growth_patterns=len(dataset.label_mappings['growth_pattern']),
            num_interference_factors=len(dataset.label_mappings['interference_factors']),
            width_mult=1.0
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型和数据加载完成 (Epoch {checkpoint['epoch']})")
        return True
    
    def collect_predictions(self, split='val'):
        """收集预测结果和详细信息"""
        print(f"\n🔍 收集 {split} 集预测结果...")
        
        dataloader = self.dataloaders[split]
        results = {
            'image_paths': [],
            'true_labels': [],
            'pred_labels': [],
            'pred_probs': [],
            'growth_patterns': [],
            'interference_factors': [],
            'growth_pattern_preds': [],
            'interference_preds': []
        }
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                
                # 获取图片路径
                batch_start = batch_idx * dataloader.batch_size
                batch_end = min(batch_start + images.size(0), len(dataloader.dataset))
                batch_paths = [dataloader.dataset.annotations[i]['image_path'] for i in range(batch_start, batch_end)]
                
                # 前向传播
                outputs = self.model(images)
                
                # 主分类预测
                main_probs = torch.softmax(outputs['classification'], dim=1)
                main_preds = torch.argmax(main_probs, dim=1)
                
                # 生长模式预测
                pattern_preds = torch.argmax(outputs['growth_pattern'], dim=1)
                
                # 干扰因素预测
                interference_preds = torch.sigmoid(outputs['interference_factors']) > 0.5
                
                # 收集结果
                results['image_paths'].extend(batch_paths)
                results['true_labels'].extend(targets['classification'].cpu().numpy())
                results['pred_labels'].extend(main_preds.cpu().numpy())
                results['pred_probs'].extend(main_probs[:, 1].cpu().numpy())  # 阳性概率
                results['growth_patterns'].extend(targets['growth_pattern'].cpu().numpy())
                results['interference_factors'].extend(targets['interference_factors'].cpu().numpy())
                results['growth_pattern_preds'].extend(pattern_preds.cpu().numpy())
                results['interference_preds'].extend(interference_preds.cpu().numpy())
        
        print(f"✅ 收集完成，共 {len(results['true_labels'])} 个样本")
        return results
    
    def analyze_negative_with_pores_misclassified(self, results):
        """分析阴性带气孔误判为阳性的情况"""
        print("\n🔍 分析问题1: 阴性带气孔误判为阳性")
        print("=" * 50)
        
        # 找出阴性样本
        true_labels = np.array(results['true_labels'])
        pred_labels = np.array(results['pred_labels'])
        pred_probs = np.array(results['pred_probs'])
        interference_factors = np.array(results['interference_factors'])
        
        negative_mask = (true_labels == 0)
        negative_indices = np.where(negative_mask)[0]
        
        print(f"阴性样本总数: {negative_mask.sum()}")
        
        # 阴性样本中的气孔情况
        negative_interference = interference_factors[negative_mask]
        pore_column = 0  # 气孔是第0列
        negative_with_pores = negative_interference[:, pore_column] == 1
        negative_pores_count = negative_with_pores.sum()
        
        print(f"阴性样本中带气孔的数量: {negative_pores_count} ({negative_pores_count/negative_mask.sum():.2%})")
        
        # 阴性带气孔被误判为阳性的情况
        negative_pore_indices = negative_indices[negative_with_pores]
        negative_pore_misclassified = pred_labels[negative_pore_indices] == 1
        misclassified_count = negative_pore_misclassified.sum()
        
        print(f"阴性带气孔误判为阳性: {misclassified_count} / {negative_pores_count} ({misclassified_count/negative_pores_count:.2%})")
        
        if misclassified_count > 0:
            # 分析误判样本的置信度
            misclassified_indices = negative_pore_indices[negative_pore_misclassified]
            misclassified_probs = pred_probs[misclassified_indices]
            
            print(f"误判样本阳性置信度统计:")
            print(f"   平均值: {misclassified_probs.mean():.4f}")
            print(f"   中位数: {np.median(misclassified_probs):.4f}")
            print(f"   标准差: {misclassified_probs.std():.4f}")
            print(f"   范围: {misclassified_probs.min():.4f} - {misclassified_probs.max():.4f}")
            
            # 列出最严重的误判样本
            worst_indices = misclassified_indices[np.argsort(misclassified_probs)[::-1]][:5]
            print(f"\n最严重的5个误判样本:")
            for i, idx in enumerate(worst_indices, 1):
                print(f"   {i}. {Path(results['image_paths'][idx]).name} (置信度: {pred_probs[idx]:.4f})")
        
        return {
            'total_negative': int(negative_mask.sum()),
            'negative_with_pores': int(negative_pores_count),
            'misclassified_count': int(misclassified_count),
            'misclassification_rate': float(misclassified_count / negative_pores_count if negative_pores_count > 0 else 0),
            'misclassified_indices': [int(x) for x in (negative_pore_indices[negative_pore_misclassified] if misclassified_count > 0 else [])]
        }
    
    def analyze_positive_weak_features_misclassified(self, results):
        """分析阳性强中心点或弱分散误判为阴性的情况"""
        print("\n🔍 分析问题2: 阳性强中心点或弱分散误判为阴性")
        print("=" * 50)
        
        # 找出阳性样本
        true_labels = np.array(results['true_labels'])
        pred_labels = np.array(results['pred_labels'])
        pred_probs = np.array(results['pred_probs'])
        growth_patterns = np.array(results['growth_patterns'])
        
        positive_mask = (true_labels == 1)
        positive_indices = np.where(positive_mask)[0]
        
        print(f"阳性样本总数: {positive_mask.sum()}")
        
        # 获取数据集的标签映射来理解生长模式
        dataset = next(iter(self.dataloaders.values())).dataset
        growth_pattern_mapping = dataset.label_mappings['growth_pattern']
        pattern_names = list(growth_pattern_mapping.keys())
        
        print(f"生长模式类型: {pattern_names}")
        
        # 阳性样本的生长模式分布
        positive_patterns = growth_patterns[positive_mask]
        pattern_counts = Counter(positive_patterns)
        
        print(f"阳性样本生长模式分布:")
        for pattern_id, count in pattern_counts.items():
            if pattern_id < len(pattern_names):
                print(f"   {pattern_names[pattern_id]}: {count} ({count/positive_mask.sum():.2%})")
        
        # 识别强中心点和弱分散模式
        # 根据生长模式名称推断(可能需要根据实际标签调整)
        weak_patterns = []
        strong_center_patterns = []
        
        for i, name in enumerate(pattern_names):
            name_lower = name.lower()
            if '弱' in name or 'weak' in name_lower or '分散' in name or 'scattered' in name_lower:
                weak_patterns.append(i)
            elif '强' in name or 'strong' in name_lower or '中心' in name or 'center' in name_lower:
                strong_center_patterns.append(i)
        
        # 分析这些特殊模式的误判情况
        target_patterns = weak_patterns + strong_center_patterns
        if target_patterns:
            target_mask = np.isin(positive_patterns, target_patterns)
            target_indices = positive_indices[target_mask]
            target_count = len(target_indices)
            
            print(f"阳性样本中强中心点/弱分散数量: {target_count} ({target_count/positive_mask.sum():.2%})")
            
            # 误判为阴性的情况
            target_misclassified = pred_labels[target_indices] == 0
            misclassified_count = target_misclassified.sum()
            
            print(f"强中心点/弱分散误判为阴性: {misclassified_count} / {target_count} ({misclassified_count/target_count:.2%})")
            
            if misclassified_count > 0:
                # 分析误判样本的置信度
                misclassified_indices = target_indices[target_misclassified]
                misclassified_probs = pred_probs[misclassified_indices]
                
                print(f"误判样本阳性置信度统计:")
                print(f"   平均值: {misclassified_probs.mean():.4f}")
                print(f"   中位数: {np.median(misclassified_probs):.4f}")
                print(f"   标准差: {misclassified_probs.std():.4f}")
                print(f"   范围: {misclassified_probs.min():.4f} - {misclassified_probs.max():.4f}")
                
                # 列出最严重的误判样本
                worst_indices = misclassified_indices[np.argsort(misclassified_probs)][:5]
                print(f"\n最严重的5个误判样本:")
                for i, idx in enumerate(worst_indices, 1):
                    pattern_id = growth_patterns[idx]
                    pattern_name = pattern_names[pattern_id] if pattern_id < len(pattern_names) else f"Unknown({pattern_id})"
                    print(f"   {i}. {Path(results['image_paths'][idx]).name} (置信度: {pred_probs[idx]:.4f}, 模式: {pattern_name})")
        else:
            print("⚠️ 无法识别强中心点/弱分散模式，请检查生长模式标签定义")
            target_count = 0
            misclassified_count = 0
        
        return {
            'total_positive': int(positive_mask.sum()),
            'target_patterns_count': int(target_count),
            'misclassified_count': int(misclassified_count),
            'misclassification_rate': float(misclassified_count / target_count if target_count > 0 else 0),
            'target_patterns': [int(x) for x in target_patterns],
            'pattern_names': pattern_names
        }
    
    def generate_optimization_recommendations(self, negative_analysis, positive_analysis):
        """生成针对性的优化建议"""
        print("\n📈 针对性优化建议")
        print("=" * 50)
        
        recommendations = []
        
        # 针对阴性带气孔误判为阳性的建议
        if negative_analysis['misclassification_rate'] > 0.05:  # 如果误判率超过5%
            print(f"🎯 问题1优化建议 (误判率: {negative_analysis['misclassification_rate']:.2%}):")
            print("   1. 增强气孔特征学习")
            print("      • 在损失函数中增加气孔识别的对抗性损失")
            print("      • 使用注意力机制专门关注气孔区域")
            print("      • 增加阴性带气孔样本的数据增强")
            
            print("   2. 调整分类阈值")
            print("      • 对于检测到气孔的样本，提高阳性分类阈值")
            print("      • 实施条件分类策略：if 气孔存在 then 降低阳性置信度")
            
            print("   3. 多阶段分类策略")
            print("      • 第一阶段：气孔检测")
            print("      • 第二阶段：基于气孔信息的调整分类")
            
            recommendations.extend([
                "pore_adversarial_loss",
                "conditional_classification_threshold",
                "multi_stage_classification"
            ])
        
        # 针对阳性弱特征误判为阴性的建议
        if positive_analysis['misclassification_rate'] > 0.05:  # 如果误判率超过5%
            print(f"\n🎯 问题2优化建议 (误判率: {positive_analysis['misclassification_rate']:.2%}):")
            print("   1. 增强边界样本学习")
            print("      • 使用Focal Loss加强困难样本学习")
            print("      • 增加强中心点/弱分散样本的权重")
            print("      • 实施困难负样本挖掘")
            
            print("   2. 多尺度特征融合")
            print("      • 增强多尺度特征提取网络")
            print("      • 使用金字塔池化捕获不同尺度的生长模式")
            print("      • 添加空间注意力机制")
            
            print("   3. 生长模式感知分类")
            print("      • 引入生长模式先验知识")
            print("      • 对不同生长模式使用不同的分类策略")
            print("      • 增加生长模式辅助监督")
            
            recommendations.extend([
                "focal_loss_hard_samples",
                "multi_scale_pyramid_pooling", 
                "growth_pattern_aware_classification"
            ])
        
        # 通用建议
        print(f"\n🔧 通用优化策略:")
        print("   1. 数据增强策略")
        print("      • 针对误判样本增加特定数据增强")
        print("      • 气孔样本的几何变换和噪声添加")
        print("      • 弱特征样本的对比度和亮度调整")
        
        print("   2. 损失函数优化")
        print("      • 实施加权损失，重点关注核心误判类别")
        print("      • 添加正则化项防止过拟合特定模式")
        print("      • 使用标签平滑减少过度自信")
        
        print("   3. 模型架构改进")
        print("      • 增加特征注意力模块")
        print("      • 使用残差连接增强梯度传播")
        print("      • 引入不确定性估计模块")
        
        return recommendations
    
    def run_diagnostic(self):
        """运行完整诊断流程"""
        print("🚀 开始核心误判问题诊断")
        print("=" * 60)
        
        # 加载模型和数据
        if not self.load_model_and_data():
            return None
        
        # 收集预测结果
        results = self.collect_predictions('val')
        
        # 分析两个核心问题
        negative_analysis = self.analyze_negative_with_pores_misclassified(results)
        positive_analysis = self.analyze_positive_weak_features_misclassified(results)
        
        # 生成优化建议
        recommendations = self.generate_optimization_recommendations(negative_analysis, positive_analysis)
        
        # 保存诊断结果
        diagnostic_result = {
            'negative_pore_misclassification': negative_analysis,
            'positive_weak_feature_misclassification': positive_analysis,
            'optimization_recommendations': recommendations,
            'total_samples': int(len(results['true_labels'])),
            'overall_accuracy': float((np.array(results['true_labels']) == np.array(results['pred_labels'])).mean())
        }
        
        # 保存到文件
        output_file = self.experiment_dir / "core_misclassification_diagnostic.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(diagnostic_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 诊断结果已保存: {output_file}")
        
        return diagnostic_result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Core Misclassification Diagnostic')
    parser.add_argument('--experiment_dir', default='experiments/multitask_grayscale_focused',
                       help='实验目录')
    
    args = parser.parse_args()
    
    # 创建诊断器并运行
    diagnostic = CoreMisclassificationDiagnostic(args.experiment_dir)
    result = diagnostic.run_diagnostic()
    
    if result:
        print(f"\n🎊 核心误判问题诊断完成！")
        print(f"   总体准确率: {result['overall_accuracy']:.4f}")
        print(f"   阴性带气孔误判率: {result['negative_pore_misclassification']['misclassification_rate']:.4f}")
        print(f"   阳性弱特征误判率: {result['positive_weak_feature_misclassification']['misclassification_rate']:.4f}")

if __name__ == "__main__":
    main()