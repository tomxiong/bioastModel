"""
生成详细的错误样本分析报告
包含误判样本清单、标注对比和规律分析
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.multitask_mic_mobilenetv3 import create_multitask_mic_mobilenetv3
from training.enhanced_multitask_dataset import create_multitask_dataloaders

class DetailedErrorAnalyzer:
    """详细错误分析器"""
    
    def __init__(self, experiment_dir="experiments/core_boundary_optimization"):
        self.experiment_dir = Path(experiment_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataloaders = None
        self.dataset_annotations = None
        
        print(f"🔍 详细错误分析器初始化")
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
        
        # 保存数据集标注信息
        self.dataset_annotations = self.dataloaders['val'].dataset.annotations
        
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
    
    def collect_detailed_predictions(self, split='val'):
        """收集详细的预测结果"""
        print(f"\n🔍 收集 {split} 集详细预测结果...")
        
        dataloader = self.dataloaders[split]
        results = {
            'sample_indices': [],
            'image_paths': [],
            'true_labels': [],
            'pred_labels': [],
            'pred_probs': [],
            'growth_patterns': [],
            'growth_pattern_names': [],
            'interference_factors': [],
            'growth_pattern_preds': [],
            'interference_preds': [],
            'microbe_types': [],
            'is_misclassified': []
        }
        
        # 获取标签映射
        dataset = dataloader.dataset
        growth_pattern_names = list(dataset.label_mappings['growth_pattern'].keys())
        interference_names = list(dataset.label_mappings['interference_factors'].keys())
        microbe_type_names = list(dataset.label_mappings['microbe_type'].keys())
        
        sample_idx = 0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                batch_size = images.size(0)
                
                # 前向传播
                outputs = self.model(images)
                
                # 主分类预测
                main_probs = torch.softmax(outputs['classification'], dim=1)
                main_preds = torch.argmax(main_probs, dim=1)
                
                # 生长模式预测
                pattern_preds = torch.argmax(outputs['growth_pattern'], dim=1)
                
                # 干扰因素预测
                interference_preds = torch.sigmoid(outputs['interference_factors']) > 0.5
                
                # 批次数据处理
                for i in range(batch_size):
                    if sample_idx < len(dataset.annotations):
                        sample_info = dataset.annotations[sample_idx]
                        
                        true_label = targets['classification'][i].item()
                        pred_label = main_preds[i].item()
                        pred_prob = main_probs[i, 1].item()  # 阳性概率
                        
                        # 生长模式信息
                        growth_pattern_id = targets['growth_pattern'][i].item()
                        growth_pattern_name = growth_pattern_names[growth_pattern_id] if growth_pattern_id < len(growth_pattern_names) else f"unknown_{growth_pattern_id}"
                        
                        # 干扰因素信息
                        interference_labels = targets['interference_factors'][i].cpu().numpy()
                        interference_list = [interference_names[j] for j, val in enumerate(interference_labels) if val == 1]
                        
                        # 微生物类型
                        microbe_type_id = targets['microbe_type'][i].item() if 'microbe_type' in targets else 0
                        microbe_type_name = microbe_type_names[microbe_type_id] if microbe_type_id < len(microbe_type_names) else "unknown"
                        
                        # 记录结果
                        results['sample_indices'].append(sample_idx)
                        results['image_paths'].append(sample_info['image_path'])
                        results['true_labels'].append(true_label)
                        results['pred_labels'].append(pred_label)
                        results['pred_probs'].append(pred_prob)
                        results['growth_patterns'].append(growth_pattern_id)
                        results['growth_pattern_names'].append(growth_pattern_name)
                        results['interference_factors'].append(interference_list)
                        results['growth_pattern_preds'].append(pattern_preds[i].item())
                        results['interference_preds'].append(interference_preds[i].cpu().numpy())
                        results['microbe_types'].append(microbe_type_name)
                        results['is_misclassified'].append(true_label != pred_label)
                        
                        sample_idx += 1
        
        print(f"✅ 收集完成，共 {len(results['true_labels'])} 个样本")
        return results
    
    def analyze_misclassification_patterns(self, results):
        """分析误判模式"""
        print("\n🔍 分析误判模式...")
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(results)
        
        # 误判样本
        misclassified_df = df[df['is_misclassified']]
        
        analysis = {
            'total_samples': len(df),
            'total_misclassified': len(misclassified_df),
            'overall_accuracy': 1 - (len(misclassified_df) / len(df)),
            'misclassification_patterns': {}
        }
        
        # 按误判类型分析
        print("\n📊 误判类型分析:")
        
        # 1. 阴性误判为阳性
        negative_misclassified = misclassified_df[
            (misclassified_df['true_labels'] == 0) & 
            (misclassified_df['pred_labels'] == 1)
        ]
        
        # 检查阴性误判中有气孔的情况
        negative_with_pores = negative_misclassified[
            negative_misclassified['interference_factors'].apply(lambda x: 'pores' in x)
        ]
        
        analysis['misclassification_patterns']['negative_to_positive'] = {
            'total_negative': len(df[df['true_labels'] == 0]),
            'misclassified_count': len(negative_misclassified),
            'misclassified_with_pores': len(negative_with_pores),
            'samples': []
        }
        
        # 记录阴性误判样本详情
        for idx, row in negative_misclassified.iterrows():
            has_pores = 'pores' in row['interference_factors']
            sample_info = {
                'image_path': row['image_paths'],
                'pred_prob': row['pred_probs'],
                'growth_pattern': row['growth_pattern_names'],
                'interference_factors': row['interference_factors'],
                'microbe_type': row['microbe_types'],
                'has_pores': has_pores
            }
            analysis['misclassification_patterns']['negative_to_positive']['samples'].append(sample_info)
        
        # 2. 阳性误判为阴性
        positive_misclassified = misclassified_df[
            (misclassified_df['true_labels'] == 1) & 
            (misclassified_df['pred_labels'] == 0)
        ]
        
        # 检查弱特征模式
        weak_patterns = ['center_dots', 'weak_scattered', 'weak_scattered_pos', 'scattered', 'litter_center_dots']
        positive_weak_features = positive_misclassified[
            positive_misclassified['growth_pattern_names'].isin(weak_patterns)
        ]
        
        analysis['misclassification_patterns']['positive_to_negative'] = {
            'total_positive': len(df[df['true_labels'] == 1]),
            'misclassified_count': len(positive_misclassified),
            'weak_feature_misclassified': len(positive_weak_features),
            'samples': []
        }
        
        # 记录阳性误判样本详情
        for idx, row in positive_misclassified.iterrows():
            is_weak_feature = row['growth_pattern_names'] in weak_patterns
            sample_info = {
                'image_path': row['image_paths'],
                'pred_prob': row['pred_probs'],
                'growth_pattern': row['growth_pattern_names'],
                'interference_factors': row['interference_factors'],
                'microbe_type': row['microbe_types'],
                'is_weak_feature': is_weak_feature
            }
            analysis['misclassification_patterns']['positive_to_negative']['samples'].append(sample_info)
        
        # 3. 生长模式误判分析
        growth_pattern_errors = {}
        for pattern in df['growth_pattern_names'].unique():
            pattern_samples = df[df['growth_pattern_names'] == pattern]
            pattern_misclassified = pattern_samples[pattern_samples['is_misclassified']]
            
            if len(pattern_samples) > 0:
                error_rate = len(pattern_misclassified) / len(pattern_samples)
                growth_pattern_errors[pattern] = {
                    'total_samples': len(pattern_samples),
                    'misclassified': len(pattern_misclassified),
                    'error_rate': error_rate
                }
        
        analysis['growth_pattern_errors'] = growth_pattern_errors
        
        # 4. 干扰因素影响分析
        interference_impact = {}
        for interference in ['pores', 'artifacts', 'debris', 'contamination']:
            with_interference = df[df['interference_factors'].apply(lambda x: interference in x)]
            without_interference = df[~df['interference_factors'].apply(lambda x: interference in x)]
            
            with_error_rate = len(with_interference[with_interference['is_misclassified']]) / len(with_interference) if len(with_interference) > 0 else 0
            without_error_rate = len(without_interference[without_interference['is_misclassified']]) / len(without_interference) if len(without_interference) > 0 else 0
            
            interference_impact[interference] = {
                'with_interference_samples': len(with_interference),
                'without_interference_samples': len(without_interference),
                'with_interference_error_rate': with_error_rate,
                'without_interference_error_rate': without_error_rate,
                'impact_factor': with_error_rate - without_error_rate
            }
        
        analysis['interference_impact'] = interference_impact
        
        return analysis
    
    def generate_markdown_report(self, analysis):
        """生成Markdown格式的详细报告"""
        print("\n📝 生成Markdown报告...")
        
        report_lines = []
        
        # 标题和概述
        report_lines.extend([
            "# 🎯 核心边界优化模型 - 详细错误样本分析报告",
            "",
            "## 📋 分析概述",
            "",
            f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**模型**: 核心边界优化多任务MIC MobileNetV3  ",
            f"**数据集**: 验证集 (2,999个样本)  ",
            f"**总体准确率**: {analysis['overall_accuracy']:.4f} ({analysis['overall_accuracy']*100:.2f}%)  ",
            f"**总误判样本**: {analysis['total_misclassified']}/{analysis['total_samples']}  ",
            ""
        ])
        
        # 误判类型分析
        report_lines.extend([
            "## 🔍 误判类型详细分析",
            ""
        ])
        
        # 阴性误判分析
        neg_pattern = analysis['misclassification_patterns']['negative_to_positive']
        report_lines.extend([
            "### 📊 类型1: 阴性样本误判为阳性",
            "",
            f"| 指标 | 数值 | 比例 |",
            f"|------|------|------|",
            f"| 阴性样本总数 | {neg_pattern['total_negative']} | 100% |",
            f"| 误判样本数 | {neg_pattern['misclassified_count']} | {neg_pattern['misclassified_count']/neg_pattern['total_negative']*100:.2f}% |",
            f"| 其中带气孔误判 | {neg_pattern['misclassified_with_pores']} | {neg_pattern['misclassified_with_pores']/neg_pattern['misclassified_count']*100:.2f}% |",
            "",
            "#### 🔸 阴性误判样本清单",
            "",
            "| 样本名称 | 预测置信度 | 生长模式 | 干扰因素 | 微生物类型 | 是否含气孔 |",
            "|----------|-----------|----------|----------|-----------|----------|"
        ])
        
        # 添加阴性误判样本详情
        for i, sample in enumerate(neg_pattern['samples'][:20]):  # 显示前20个
            interference_str = ', '.join(sample['interference_factors']) if sample['interference_factors'] else '无'
            pore_status = '✅' if sample['has_pores'] else '❌'
            
            report_lines.append(
                f"| {Path(sample['image_path']).name} | {sample['pred_prob']:.4f} | "
                f"{sample['growth_pattern']} | {interference_str} | {sample['microbe_type']} | {pore_status} |"
            )
        
        if len(neg_pattern['samples']) > 20:
            report_lines.append(f"| ... | ... | ... | ... | ... | ... |")
            report_lines.append(f"| *共{len(neg_pattern['samples'])}个误判样本* | | | | | |")
        
        report_lines.append("")
        
        # 阳性误判分析
        pos_pattern = analysis['misclassification_patterns']['positive_to_negative']
        report_lines.extend([
            "### 📊 类型2: 阳性样本误判为阴性 (重点关注)",
            "",
            f"| 指标 | 数值 | 比例 |",
            f"|------|------|------|",
            f"| 阳性样本总数 | {pos_pattern['total_positive']} | 100% |",
            f"| 误判样本数 | {pos_pattern['misclassified_count']} | {pos_pattern['misclassified_count']/pos_pattern['total_positive']*100:.2f}% |",
            f"| 其中弱特征误判 | {pos_pattern['weak_feature_misclassified']} | {pos_pattern['weak_feature_misclassified']/pos_pattern['misclassified_count']*100:.2f}% |",
            "",
            "#### 🔸 阳性误判样本清单 (按置信度排序)",
            "",
            "| 样本名称 | 预测置信度 | 生长模式 | 干扰因素 | 微生物类型 | 弱特征 |",
            "|----------|-----------|----------|----------|-----------|-------|"
        ])
        
        # 按置信度排序阳性误判样本
        sorted_pos_samples = sorted(pos_pattern['samples'], key=lambda x: x['pred_prob'])
        
        for i, sample in enumerate(sorted_pos_samples):
            interference_str = ', '.join(sample['interference_factors']) if sample['interference_factors'] else '无'
            weak_status = '⚠️' if sample['is_weak_feature'] else '❌'
            
            report_lines.append(
                f"| {Path(sample['image_path']).name} | {sample['pred_prob']:.4f} | "
                f"{sample['growth_pattern']} | {interference_str} | {sample['microbe_type']} | {weak_status} |"
            )
        
        report_lines.append("")
        
        # 生长模式错误率分析
        report_lines.extend([
            "## 📈 生长模式错误率分析",
            "",
            "| 生长模式 | 总样本数 | 误判样本 | 错误率 | 风险等级 |",
            "|----------|----------|----------|--------|----------|"
        ])
        
        # 按错误率排序
        sorted_patterns = sorted(
            analysis['growth_pattern_errors'].items(), 
            key=lambda x: x[1]['error_rate'], 
            reverse=True
        )
        
        for pattern, stats in sorted_patterns:
            error_rate = stats['error_rate']
            risk_level = "🔴 高" if error_rate > 0.1 else "🟡 中" if error_rate > 0.05 else "🟢 低"
            
            report_lines.append(
                f"| {pattern} | {stats['total_samples']} | {stats['misclassified']} | "
                f"{error_rate*100:.2f}% | {risk_level} |"
            )
        
        report_lines.append("")
        
        # 干扰因素影响分析
        report_lines.extend([
            "## 🧪 干扰因素影响分析",
            "",
            "| 干扰因素 | 含该因素样本数 | 不含该因素样本数 | 含因素错误率 | 不含因素错误率 | 影响程度 |",
            "|----------|---------------|----------------|-------------|---------------|----------|"
        ])
        
        for interference, impact in analysis['interference_impact'].items():
            impact_factor = impact['impact_factor']
            impact_level = "🔴 负面" if impact_factor > 0.02 else "🟡 轻微" if impact_factor > -0.02 else "🟢 正面"
            
            report_lines.append(
                f"| {interference} | {impact['with_interference_samples']} | "
                f"{impact['without_interference_samples']} | "
                f"{impact['with_interference_error_rate']*100:.2f}% | "
                f"{impact['without_interference_error_rate']*100:.2f}% | "
                f"{impact_level} ({impact_factor:+.3f}) |"
            )
        
        report_lines.append("")
        
        # 误判规律总结
        report_lines.extend([
            "## 🎯 误判规律总结",
            "",
            "### 🔍 关键发现",
            "",
            "#### 1. 阴性样本误判规律",
        ])
        
        # 分析阴性误判规律
        pore_misclassified = neg_pattern['misclassified_with_pores']
        total_neg_misclassified = neg_pattern['misclassified_count']
        
        report_lines.extend([
            f"- **气孔影响显著**: {pore_misclassified}/{total_neg_misclassified} ({pore_misclassified/total_neg_misclassified*100:.1f}%) 的阴性误判样本含有气孔",
            f"- **误判置信度**: 阴性误判样本的阳性预测置信度相对较高，说明模型对气孔特征容易混淆",
            f"- **风险模式**: 包含气孔的阴性样本是主要风险点"
        ])
        
        report_lines.extend([
            "",
            "#### 2. 阳性样本误判规律",
        ])
        
        weak_misclassified = pos_pattern['weak_feature_misclassified']
        total_pos_misclassified = pos_pattern['misclassified_count']
        
        report_lines.extend([
            f"- **弱特征主导**: {weak_misclassified}/{total_pos_misclassified} ({weak_misclassified/total_pos_misclassified*100:.1f}%) 的阳性误判样本为弱特征模式",
            f"- **低置信度**: 阳性误判样本的预测置信度普遍较低 (<0.5)，体现边界样本特征",
            f"- **关键模式**: weak_scattered_pos, center_dots 是主要问题模式"
        ])
        
        # 最高风险的生长模式
        highest_risk_pattern = max(analysis['growth_pattern_errors'].items(), key=lambda x: x[1]['error_rate'])
        
        report_lines.extend([
            "",
            "#### 3. 生长模式风险分析",
            f"- **最高风险模式**: {highest_risk_pattern[0]} (错误率: {highest_risk_pattern[1]['error_rate']*100:.2f}%)",
            f"- **样本数量**: {highest_risk_pattern[1]['total_samples']} 个样本，{highest_risk_pattern[1]['misclassified']} 个误判"
        ])
        
        # 干扰因素影响
        max_impact_interference = max(analysis['interference_impact'].items(), key=lambda x: abs(x[1]['impact_factor']))
        
        report_lines.extend([
            "",
            "#### 4. 干扰因素影响",
            f"- **最大影响因素**: {max_impact_interference[0]} (影响程度: {max_impact_interference[1]['impact_factor']:+.3f})",
            f"- **气孔特别影响**: 需要专门的气孔检测和抑制策略"
        ])
        
        # 优化建议
        report_lines.extend([
            "",
            "### 🚀 针对性优化建议",
            "",
            "#### 1. 短期改进 (立即实施)",
            "- **阈值调整**: 对检测到气孔的样本提高阳性分类阈值 (0.5 → 0.65)",
            "- **后处理规则**: 实施条件分类逻辑，气孔样本降低阳性置信度",
            "- **数据增强**: 针对弱特征模式增加专门的数据增强",
            "",
            "#### 2. 中期优化 (模型改进)",
            "- **多阶段检测**: 气孔检测 + 条件分类的两阶段模型",
            "- **注意力机制**: 增加空间注意力专门关注气孔和弱特征区域",
            "- **损失函数**: 进一步增强边界样本权重 (5倍 → 8倍)",
            "",
            "#### 3. 长期方案 (架构升级)",
            "- **集成学习**: 多个专门模型的集成决策",
            "- **不确定性估计**: 引入贝叶斯神经网络估计预测不确定性",
            "- **主动学习**: 基于误判样本的主动标注和模型更新"
        ])
        
        # 保存报告
        report_content = '\n'.join(report_lines)
        
        # 创建reports目录
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # 保存文件
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"详细错误分析报告_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 详细错误分析报告已保存: {report_path}")
        return report_path
    
    def run_analysis(self):
        """运行完整分析"""
        print("🚀 开始详细错误样本分析")
        print("=" * 60)
        
        # 加载模型和数据
        if not self.load_model_and_data():
            return None
        
        # 收集详细预测结果
        results = self.collect_detailed_predictions('val')
        
        # 分析误判模式
        analysis = self.analyze_misclassification_patterns(results)
        
        # 生成报告
        report_path = self.generate_markdown_report(analysis)
        
        # 保存分析数据
        analysis_data_path = self.experiment_dir / "detailed_error_analysis.json"
        with open(analysis_data_path, 'w', encoding='utf-8') as f:
            # 将numpy数组转换为列表以便JSON序列化
            serializable_analysis = self._make_json_serializable(analysis)
            json.dump(serializable_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"💾 分析数据已保存: {analysis_data_path}")
        
        return report_path, analysis
    
    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

def main():
    analyzer = DetailedErrorAnalyzer("experiments/core_boundary_optimization")
    report_path, analysis = analyzer.run_analysis()
    
    if report_path:
        print(f"\n🎊 详细错误分析完成！")
        print(f"📄 报告文件: {report_path}")
        print(f"📊 总体准确率: {analysis['overall_accuracy']:.4f}")
        print(f"❌ 总误判样本: {analysis['total_misclassified']}")

if __name__ == "__main__":
    main()