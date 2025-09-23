"""
更新详细的错误样本分析报告 - 包含完整图片路径
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

class EnhancedDetailedErrorAnalyzer:
    """增强的详细错误分析器 - 包含完整路径信息"""
    
    def __init__(self, experiment_dir="experiments/core_boundary_optimization"):
        self.experiment_dir = Path(experiment_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataloaders = None
        self.dataset_annotations = None
        self.data_root = "/home/aaa/ws/bioastModel/ds/images"
        
        print(f"🔍 增强详细错误分析器初始化")
        print(f"   实验目录: {self.experiment_dir}")
        print(f"   数据根目录: {self.data_root}")
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
            data_root=self.data_root,
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
    
    def get_full_image_path(self, relative_path):
        """获取完整的图片路径"""
        full_path = Path(self.data_root) / relative_path
        return str(full_path)
    
    def verify_image_exists(self, full_path):
        """验证图片是否存在"""
        return Path(full_path).exists()
    
    def collect_enhanced_predictions(self, split='val'):
        """收集增强的预测结果，包含完整路径信息"""
        print(f"\n🔍 收集 {split} 集增强预测结果...")
        
        dataloader = self.dataloaders[split]
        results = {
            'sample_indices': [],
            'image_relative_paths': [],
            'image_full_paths': [],
            'image_exists': [],
            'true_labels': [],
            'pred_labels': [],
            'pred_probs': [],
            'growth_patterns': [],
            'growth_pattern_names': [],
            'interference_factors': [],
            'growth_pattern_preds': [],
            'interference_preds': [],
            'microbe_types': [],
            'is_misclassified': [],
            'confidence_level': []  # 添加置信度等级
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
                        
                        # 路径信息
                        relative_path = sample_info['image_path']
                        full_path = self.get_full_image_path(relative_path)
                        image_exists = self.verify_image_exists(full_path)
                        
                        # 置信度等级
                        confidence_level = self.get_confidence_level(pred_prob)
                        
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
                        results['image_relative_paths'].append(relative_path)
                        results['image_full_paths'].append(full_path)
                        results['image_exists'].append(image_exists)
                        results['true_labels'].append(true_label)
                        results['pred_labels'].append(pred_label)
                        results['pred_probs'].append(pred_prob)
                        results['confidence_level'].append(confidence_level)
                        results['growth_patterns'].append(growth_pattern_id)
                        results['growth_pattern_names'].append(growth_pattern_name)
                        results['interference_factors'].append(interference_list)
                        results['growth_pattern_preds'].append(pattern_preds[i].item())
                        results['interference_preds'].append(interference_preds[i].cpu().numpy())
                        results['microbe_types'].append(microbe_type_name)
                        results['is_misclassified'].append(true_label != pred_label)
                        
                        sample_idx += 1
        
        # 统计路径验证结果
        total_samples = len(results['image_exists'])
        existing_samples = sum(results['image_exists'])
        missing_samples = total_samples - existing_samples
        
        print(f"✅ 收集完成，共 {total_samples} 个样本")
        print(f"   存在的图片: {existing_samples}")
        print(f"   缺失的图片: {missing_samples}")
        
        if missing_samples > 0:
            print(f"⚠️ 发现 {missing_samples} 个图片文件缺失")
        
        return results
    
    def get_confidence_level(self, pred_prob):
        """获取置信度等级"""
        if pred_prob >= 0.8:
            return "极高"
        elif pred_prob >= 0.65:
            return "高"
        elif pred_prob >= 0.35:
            return "中等"
        elif pred_prob >= 0.2:
            return "低"
        else:
            return "极低"
    
    def analyze_enhanced_misclassification_patterns(self, results):
        """分析增强的误判模式"""
        print("\n🔍 分析增强误判模式...")
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(results)
        
        # 误判样本
        misclassified_df = df[df['is_misclassified']]
        
        analysis = {
            'total_samples': len(df),
            'total_misclassified': len(misclassified_df),
            'overall_accuracy': 1 - (len(misclassified_df) / len(df)),
            'path_verification': {
                'total_samples': len(df),
                'existing_images': sum(results['image_exists']),
                'missing_images': len(df) - sum(results['image_exists'])
            },
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
        
        # 记录阴性误判样本详情（增强版）
        for idx, row in negative_misclassified.iterrows():
            has_pores = 'pores' in row['interference_factors']
            sample_info = {
                'image_relative_path': row['image_relative_paths'],
                'image_full_path': row['image_full_paths'],
                'image_exists': row['image_exists'],
                'pred_prob': row['pred_probs'],
                'confidence_level': row['confidence_level'],
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
        
        # 记录阳性误判样本详情（增强版）
        for idx, row in positive_misclassified.iterrows():
            is_weak_feature = row['growth_pattern_names'] in weak_patterns
            sample_info = {
                'image_relative_path': row['image_relative_paths'],
                'image_full_path': row['image_full_paths'],
                'image_exists': row['image_exists'],
                'pred_prob': row['pred_probs'],
                'confidence_level': row['confidence_level'],
                'growth_pattern': row['growth_pattern_names'],
                'interference_factors': row['interference_factors'],
                'microbe_type': row['microbe_types'],
                'is_weak_feature': is_weak_feature
            }
            analysis['misclassification_patterns']['positive_to_negative']['samples'].append(sample_info)
        
        return analysis
    
    def generate_enhanced_markdown_report(self, analysis):
        """生成增强的Markdown格式报告"""
        print("\n📝 生成增强Markdown报告...")
        
        report_lines = []
        
        # 标题和概述
        report_lines.extend([
            "# 🎯 核心边界优化模型 - 增强版详细错误样本分析报告",
            "",
            "## 📋 分析概述",
            "",
            f"**分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**模型**: 核心边界优化多任务MIC MobileNetV3  ",
            f"**数据集**: 验证集 ({analysis['total_samples']:,}个样本)  ",
            f"**数据根目录**: `/home/aaa/ws/bioastModel/ds/images`  ",
            f"**总体准确率**: {analysis['overall_accuracy']:.4f} ({analysis['overall_accuracy']*100:.2f}%)  ",
            f"**总误判样本**: {analysis['total_misclassified']}/{analysis['total_samples']}  ",
            "",
            "### 📁 路径验证状态",
            "",
            f"| 项目 | 数量 | 状态 |",
            f"|------|------|------|",
            f"| 总样本数 | {analysis['path_verification']['total_samples']:,} | - |",
            f"| 图片存在 | {analysis['path_verification']['existing_images']:,} | ✅ |",
            f"| 图片缺失 | {analysis['path_verification']['missing_images']:,} | {'⚠️' if analysis['path_verification']['missing_images'] > 0 else '✅'} |",
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
            f"| 阴性样本总数 | {neg_pattern['total_negative']:,} | 100% |",
            f"| 误判样本数 | {neg_pattern['misclassified_count']} | {neg_pattern['misclassified_count']/neg_pattern['total_negative']*100:.2f}% |",
            f"| 其中带气孔误判 | {neg_pattern['misclassified_with_pores']} | {neg_pattern['misclassified_with_pores']/neg_pattern['misclassified_count']*100:.2f}% |",
            "",
            "#### 🔸 阴性误判样本完整清单",
            "",
            "| 序号 | 样本名称 | 完整路径 | 存在状态 | 预测置信度 | 置信度等级 | 生长模式 | 干扰因素 | 微生物类型 | 含气孔 |",
            "|------|----------|----------|----------|-----------|-----------|----------|----------|-----------|-------|"
        ])
        
        # 添加阴性误判样本详情（按置信度排序）
        sorted_neg_samples = sorted(neg_pattern['samples'], key=lambda x: x['pred_prob'], reverse=True)
        
        for i, sample in enumerate(sorted_neg_samples, 1):
            interference_str = ', '.join(sample['interference_factors']) if sample['interference_factors'] else '无'
            pore_status = '✅' if sample['has_pores'] else '❌'
            exist_status = '✅' if sample['image_exists'] else '❌ 缺失'
            
            report_lines.append(
                f"| {i} | {Path(sample['image_relative_path']).name} | "
                f"`{sample['image_full_path']}` | {exist_status} | "
                f"{sample['pred_prob']:.4f} | {sample['confidence_level']} | "
                f"{sample['growth_pattern']} | {interference_str} | "
                f"{sample['microbe_type']} | {pore_status} |"
            )
        
        report_lines.append("")
        
        # 阳性误判分析
        pos_pattern = analysis['misclassification_patterns']['positive_to_negative']
        report_lines.extend([
            "### 📊 类型2: 阳性样本误判为阴性 (重点关注)",
            "",
            f"| 指标 | 数值 | 比例 |",
            f"|------|------|------|",
            f"| 阳性样本总数 | {pos_pattern['total_positive']:,} | 100% |",
            f"| 误判样本数 | {pos_pattern['misclassified_count']} | {pos_pattern['misclassified_count']/pos_pattern['total_positive']*100:.2f}% |",
            f"| 其中弱特征误判 | {pos_pattern['weak_feature_misclassified']} | {pos_pattern['weak_feature_misclassified']/pos_pattern['misclassified_count']*100:.2f}% |",
            "",
            "#### 🔸 阳性误判样本完整清单 (按置信度排序)",
            "",
            "| 序号 | 样本名称 | 完整路径 | 存在状态 | 预测置信度 | 置信度等级 | 生长模式 | 干扰因素 | 微生物类型 | 弱特征 |",
            "|------|----------|----------|----------|-----------|-----------|----------|----------|-----------|-------|"
        ])
        
        # 按置信度排序阳性误判样本
        sorted_pos_samples = sorted(pos_pattern['samples'], key=lambda x: x['pred_prob'])
        
        for i, sample in enumerate(sorted_pos_samples, 1):
            interference_str = ', '.join(sample['interference_factors']) if sample['interference_factors'] else '无'
            weak_status = '⚠️' if sample['is_weak_feature'] else '❌'
            exist_status = '✅' if sample['image_exists'] else '❌ 缺失'
            
            report_lines.append(
                f"| {i} | {Path(sample['image_relative_path']).name} | "
                f"`{sample['image_full_path']}` | {exist_status} | "
                f"{sample['pred_prob']:.4f} | {sample['confidence_level']} | "
                f"{sample['growth_pattern']} | {interference_str} | "
                f"{sample['microbe_type']} | {weak_status} |"
            )
        
        report_lines.extend([
            "",
            "## 🎯 路径信息总结",
            "",
            "### 📁 数据路径说明",
            "",
            "**数据根目录**: `/home/aaa/ws/bioastModel/ds/images`  ",
            "**路径结构**: `数据根目录/相对路径`  ",
            "**示例**:",
            "- 相对路径: `EB20000092/hole_61.png`",
            "- 完整路径: `/home/aaa/ws/bioastModel/ds/images/EB20000092/hole_61.png`",
            "",
            "### 🔍 误判样本路径快速访问",
            "",
            "#### 高风险阴性误判样本路径 (Top 10)",
            ""
        ])
        
        # 添加高风险样本的快速访问路径
        high_risk_neg = sorted(neg_pattern['samples'], key=lambda x: x['pred_prob'], reverse=True)[:10]
        for i, sample in enumerate(high_risk_neg, 1):
            report_lines.append(f"{i}. `{sample['image_full_path']}` (置信度: {sample['pred_prob']:.4f})")
        
        report_lines.extend([
            "",
            "#### 核心阳性误判样本路径 (全部)",
            ""
        ])
        
        # 添加阳性误判样本的路径
        for i, sample in enumerate(sorted_pos_samples, 1):
            report_lines.append(f"{i}. `{sample['image_full_path']}` (置信度: {sample['pred_prob']:.4f})")
        
        # 添加使用说明
        report_lines.extend([
            "",
            "## 💡 使用说明",
            "",
            "### 📷 查看误判样本",
            "",
            "```bash",
            "# 查看单个样本",
            "ls -la \"/home/aaa/ws/bioastModel/ds/images/EB20000092/hole_61.png\"",
            "",
            "# 批量检查阴性误判样本存在状态",
            "find /home/aaa/ws/bioastModel/ds/images -name \"hole_*.png\" | head -20",
            "",
            "# 复制高风险样本到分析目录",
            "mkdir -p /tmp/error_analysis/negative_misclassified",
            "mkdir -p /tmp/error_analysis/positive_misclassified",
            "```",
            "",
            "### 🔧 批量验证脚本",
            "",
            "```python",
            "import os",
            "from pathlib import Path",
            "",
            "# 验证所有误判样本是否存在",
            "error_samples = [",
            "    # 从上述清单中复制路径",
            "]",
            "",
            "missing_samples = []",
            "for sample_path in error_samples:",
            "    if not Path(sample_path).exists():",
            "        missing_samples.append(sample_path)",
            "",
            "print(f\"缺失样本数量: {len(missing_samples)}\")",
            "```"
        ])
        
        # 保存报告
        report_content = '\n'.join(report_lines)
        
        # 创建reports目录
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # 保存文件
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"增强版详细错误分析报告_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 增强版详细错误分析报告已保存: {report_path}")
        return report_path
    
    def run_enhanced_analysis(self):
        """运行增强版完整分析"""
        print("🚀 开始增强版详细错误样本分析")
        print("=" * 60)
        
        # 加载模型和数据
        if not self.load_model_and_data():
            return None
        
        # 收集增强预测结果
        results = self.collect_enhanced_predictions('val')
        
        # 分析误判模式
        analysis = self.analyze_enhanced_misclassification_patterns(results)
        
        # 生成增强报告
        report_path = self.generate_enhanced_markdown_report(analysis)
        
        # 保存增强分析数据
        analysis_data_path = self.experiment_dir / "enhanced_detailed_error_analysis.json"
        with open(analysis_data_path, 'w', encoding='utf-8') as f:
            # 将numpy数组转换为列表以便JSON序列化
            serializable_analysis = self._make_json_serializable(analysis)
            json.dump(serializable_analysis, f, indent=2, ensure_ascii=False)
        
        print(f"💾 增强分析数据已保存: {analysis_data_path}")
        
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
    analyzer = EnhancedDetailedErrorAnalyzer("experiments/core_boundary_optimization")
    report_path, analysis = analyzer.run_enhanced_analysis()
    
    if report_path:
        print(f"\n🎊 增强版详细错误分析完成！")
        print(f"📄 报告文件: {report_path}")
        print(f"📊 总体准确率: {analysis['overall_accuracy']:.4f}")
        print(f"❌ 总误判样本: {analysis['total_misclassified']}")
        print(f"📁 图片存在: {analysis['path_verification']['existing_images']}")
        print(f"⚠️ 图片缺失: {analysis['path_verification']['missing_images']}")

if __name__ == "__main__":
    main()