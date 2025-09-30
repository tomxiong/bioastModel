#!/usr/bin/env python3
"""
Growth Pattern任务详细错误分析
深入分析Growth Pattern分类任务中的错误模式和特征
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class GrowthPatternErrorAnalyzer:
    def __init__(self, experiment_path):
        self.experiment_path = Path(experiment_path)
        self.test_results = None
        self.label_info = None
        self.load_data()
        
    def load_data(self):
        """加载测试结果和标签信息"""
        with open(self.experiment_path / 'test_results.json', 'r') as f:
            self.test_results = json.load(f)
            
        with open(self.experiment_path / 'label_info.json', 'r') as f:
            self.label_info = json.load(f)
    
    def analyze_confusion_patterns(self):
        """分析混淆矩阵中的错误模式"""
        confusion_matrix = np.array(self.test_results['growth_pattern']['confusion_matrix'])
        labels = list(self.label_info['growth_pattern'].keys())
        
        # 计算每个类别的详细错误信息
        class_analysis = {}
        
        for i, true_label in enumerate(labels):
            true_count = np.sum(confusion_matrix[i, :])
            correct_count = confusion_matrix[i, i]
            
            if true_count > 0:
                # 找出所有错误预测
                error_predictions = []
                for j, pred_label in enumerate(labels):
                    if i != j and confusion_matrix[i, j] > 0:
                        error_predictions.append({
                            'predicted_as': pred_label,
                            'count': int(confusion_matrix[i, j]),
                            'percentage': confusion_matrix[i, j] / true_count * 100
                        })
                
                # 按错误数量排序
                error_predictions.sort(key=lambda x: x['count'], reverse=True)
                
                class_analysis[true_label] = {
                    'total_samples': int(true_count),
                    'correct_predictions': int(correct_count),
                    'accuracy': correct_count / true_count,
                    'error_count': int(true_count - correct_count),
                    'error_rate': (true_count - correct_count) / true_count,
                    'error_predictions': error_predictions
                }
        
        return class_analysis
    
    def identify_problematic_classes(self, class_analysis):
        """识别问题最严重的类别"""
        # 按错误率排序
        problematic_classes = []
        
        for class_name, analysis in class_analysis.items():
            if analysis['total_samples'] >= 5:  # 只考虑样本数量足够的类别
                problematic_classes.append({
                    'class_name': class_name,
                    'error_rate': analysis['error_rate'],
                    'error_count': analysis['error_count'],
                    'total_samples': analysis['total_samples'],
                    'main_confusion': analysis['error_predictions'][0] if analysis['error_predictions'] else None
                })
        
        # 按错误率排序
        problematic_classes.sort(key=lambda x: x['error_rate'], reverse=True)
        
        return problematic_classes
    
    def analyze_confusion_pairs(self):
        """分析最容易混淆的类别对"""
        confusion_matrix = np.array(self.test_results['growth_pattern']['confusion_matrix'])
        labels = list(self.label_info['growth_pattern'].keys())
        
        confusion_pairs = []
        
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and confusion_matrix[i, j] > 0:
                    # 计算双向混淆
                    mutual_confusion = confusion_matrix[i, j] + confusion_matrix[j, i]
                    
                    confusion_pairs.append({
                        'class_a': labels[i],
                        'class_b': labels[j],
                        'a_to_b': int(confusion_matrix[i, j]),
                        'b_to_a': int(confusion_matrix[j, i]),
                        'mutual_confusion': int(mutual_confusion),
                        'direction': 'bidirectional' if confusion_matrix[j, i] > 0 else 'unidirectional'
                    })
        
        # 按互相混淆程度排序
        confusion_pairs.sort(key=lambda x: x['mutual_confusion'], reverse=True)
        
        # 去除重复的双向对
        unique_pairs = []
        seen_pairs = set()
        
        for pair in confusion_pairs:
            pair_key = tuple(sorted([pair['class_a'], pair['class_b']]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                unique_pairs.append(pair)
        
        return unique_pairs[:10]  # 返回前10个最混淆的对
    
    def create_detailed_visualizations(self, class_analysis, problematic_classes, confusion_pairs):
        """创建详细的可视化分析"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 各类别错误率分布
        ax1 = plt.subplot(2, 3, 1)
        classes = [item['class_name'] for item in problematic_classes]
        error_rates = [item['error_rate'] * 100 for item in problematic_classes]
        
        bars = ax1.barh(classes, error_rates, color='lightcoral', alpha=0.7)
        ax1.set_xlabel('错误率 (%)')
        ax1.set_title('Growth Pattern 各类别错误率', fontweight='bold')
        
        # 添加数值标签
        for bar, rate in zip(bars, error_rates):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{rate:.1f}%', va='center', fontweight='bold')
        
        # 2. 样本数量 vs 错误率散点图
        ax2 = plt.subplot(2, 3, 2)
        sample_counts = [item['total_samples'] for item in problematic_classes]
        error_rates_scatter = [item['error_rate'] * 100 for item in problematic_classes]
        
        scatter = ax2.scatter(sample_counts, error_rates_scatter, 
                            c=error_rates_scatter, cmap='Reds', 
                            s=100, alpha=0.7, edgecolors='black')
        
        ax2.set_xlabel('样本数量')
        ax2.set_ylabel('错误率 (%)')
        ax2.set_title('样本数量 vs 错误率关系', fontweight='bold')
        
        # 添加类别标签
        for i, class_name in enumerate(classes):
            ax2.annotate(class_name.replace('_', '\n'), 
                        (sample_counts[i], error_rates_scatter[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.colorbar(scatter, ax=ax2, label='错误率 (%)')
        
        # 3. 最容易混淆的类别对
        ax3 = plt.subplot(2, 3, 3)
        top_pairs = confusion_pairs[:8]
        pair_labels = [f"{pair['class_a']}\n↔\n{pair['class_b']}" for pair in top_pairs]
        mutual_confusions = [pair['mutual_confusion'] for pair in top_pairs]
        
        bars = ax3.bar(range(len(pair_labels)), mutual_confusions, 
                      color='lightblue', alpha=0.7)
        ax3.set_xlabel('类别对')
        ax3.set_ylabel('互相混淆次数')
        ax3.set_title('最容易混淆的类别对', fontweight='bold')
        ax3.set_xticks(range(len(pair_labels)))
        ax3.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
        
        # 添加数值标签
        for bar, count in zip(bars, mutual_confusions):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. 错误预测分布热图
        ax4 = plt.subplot(2, 3, 4)
        confusion_matrix = np.array(self.test_results['growth_pattern']['confusion_matrix'])
        labels = list(self.label_info['growth_pattern'].keys())
        
        # 只显示错误预测（对角线设为0）
        error_matrix = confusion_matrix.copy()
        np.fill_diagonal(error_matrix, 0)
        
        sns.heatmap(error_matrix, annot=True, fmt='d', cmap='Reds',
                   xticklabels=[l.replace('_', '\n') for l in labels], 
                   yticklabels=[l.replace('_', '\n') for l in labels], 
                   ax=ax4, cbar_kws={'label': '错误预测次数'})
        ax4.set_title('错误预测分布热图', fontweight='bold')
        ax4.set_xlabel('预测标签')
        ax4.set_ylabel('真实标签')
        
        # 5. 类别准确率对比
        ax5 = plt.subplot(2, 3, 5)
        all_classes = list(class_analysis.keys())
        accuracies = [class_analysis[cls]['accuracy'] * 100 for cls in all_classes]
        
        bars = ax5.bar(range(len(all_classes)), accuracies, 
                      color=['red' if acc < 50 else 'orange' if acc < 80 else 'green' 
                            for acc in accuracies], alpha=0.7)
        ax5.set_xlabel('类别')
        ax5.set_ylabel('准确率 (%)')
        ax5.set_title('各类别准确率对比', fontweight='bold')
        ax5.set_xticks(range(len(all_classes)))
        ax5.set_xticklabels([cls.replace('_', '\n') for cls in all_classes], 
                           rotation=45, ha='right', fontsize=8)
        ax5.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80%基准线')
        ax5.legend()
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 6. 错误类型分析
        ax6 = plt.subplot(2, 3, 6)
        
        # 统计不同错误类型
        error_types = {
            '完全错误': 0,  # 错误率100%
            '高错误': 0,    # 错误率50-99%
            '中等错误': 0,  # 错误率20-49%
            '低错误': 0     # 错误率<20%
        }
        
        for cls_name, analysis in class_analysis.items():
            if analysis['total_samples'] >= 5:
                error_rate = analysis['error_rate']
                if error_rate >= 1.0:
                    error_types['完全错误'] += 1
                elif error_rate >= 0.5:
                    error_types['高错误'] += 1
                elif error_rate >= 0.2:
                    error_types['中等错误'] += 1
                else:
                    error_types['低错误'] += 1
        
        colors = ['darkred', 'red', 'orange', 'green']
        wedges, texts, autotexts = ax6.pie(error_types.values(), 
                                          labels=error_types.keys(),
                                          colors=colors, autopct='%1.0f%%',
                                          startangle=90)
        ax6.set_title('错误严重程度分布', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_detailed_report(self, class_analysis, problematic_classes, confusion_pairs):
        """生成详细的Growth Pattern错误分析报告"""
        report = []
        report.append("# Growth Pattern 任务详细错误分析报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 总体统计
        total_samples = sum(analysis['total_samples'] for analysis in class_analysis.values())
        total_errors = sum(analysis['error_count'] for analysis in class_analysis.values())
        overall_accuracy = self.test_results['growth_pattern']['accuracy']
        
        report.append(f"\n## 总体统计")
        report.append(f"- 总样本数: {total_samples:,}")
        report.append(f"- 总错误数: {total_errors:,}")
        report.append(f"- 总体准确率: {overall_accuracy:.2%}")
        report.append(f"- 总体错误率: {1-overall_accuracy:.2%}")
        
        # 问题最严重的类别
        report.append(f"\n## 问题最严重的类别")
        for i, cls_info in enumerate(problematic_classes[:5]):
            report.append(f"\n### {i+1}. {cls_info['class_name']}")
            report.append(f"- 样本数量: {cls_info['total_samples']}")
            report.append(f"- 错误数量: {cls_info['error_count']}")
            report.append(f"- 错误率: {cls_info['error_rate']:.2%}")
            
            if cls_info['main_confusion']:
                main_conf = cls_info['main_confusion']
                report.append(f"- 主要混淆: 被误分类为 '{main_conf['predicted_as']}' "
                             f"({main_conf['count']}次, {main_conf['percentage']:.1f}%)")
            
            # 详细错误分析
            cls_analysis = class_analysis[cls_info['class_name']]
            if len(cls_analysis['error_predictions']) > 1:
                report.append(f"- 所有错误预测:")
                for error_pred in cls_analysis['error_predictions']:
                    report.append(f"  - {error_pred['predicted_as']}: {error_pred['count']}次 "
                                 f"({error_pred['percentage']:.1f}%)")
        
        # 最容易混淆的类别对
        report.append(f"\n## 最容易混淆的类别对")
        for i, pair in enumerate(confusion_pairs[:5]):
            report.append(f"\n### {i+1}. {pair['class_a']} ↔ {pair['class_b']}")
            report.append(f"- {pair['class_a']} → {pair['class_b']}: {pair['a_to_b']}次")
            report.append(f"- {pair['class_b']} → {pair['class_a']}: {pair['b_to_a']}次")
            report.append(f"- 总混淆次数: {pair['mutual_confusion']}")
            report.append(f"- 混淆类型: {pair['direction']}")
        
        # 类别特征分析
        report.append(f"\n## 类别特征分析")
        
        # 零样本或极少样本类别
        rare_classes = [cls for cls, analysis in class_analysis.items() 
                       if analysis['total_samples'] < 10]
        if rare_classes:
            report.append(f"\n### 稀有类别 (样本数<10)")
            for cls in rare_classes:
                analysis = class_analysis[cls]
                report.append(f"- {cls}: {analysis['total_samples']}个样本, "
                             f"准确率{analysis['accuracy']:.2%}")
        
        # 高准确率类别
        high_acc_classes = [cls for cls, analysis in class_analysis.items() 
                           if analysis['accuracy'] > 0.9 and analysis['total_samples'] >= 10]
        if high_acc_classes:
            report.append(f"\n### 高准确率类别 (>90%)")
            for cls in high_acc_classes:
                analysis = class_analysis[cls]
                report.append(f"- {cls}: {analysis['accuracy']:.2%} "
                             f"({analysis['total_samples']}个样本)")
        
        # 改进建议
        report.append(f"\n## 改进建议")
        
        # 针对问题最严重的类别
        worst_classes = problematic_classes[:3]
        report.append(f"\n### 针对问题类别的建议")
        for cls_info in worst_classes:
            cls_name = cls_info['class_name']
            report.append(f"\n#### {cls_name} (错误率: {cls_info['error_rate']:.2%})")
            
            if cls_info['total_samples'] < 50:
                report.append("- **数据增强**: 样本数量不足，建议:")
                report.append("  - 收集更多该类别的标注样本")
                report.append("  - 使用数据增强技术增加样本多样性")
                report.append("  - 考虑合成数据生成")
            
            if cls_info['main_confusion']:
                main_conf_class = cls_info['main_confusion']['predicted_as']
                report.append(f"- **特征区分**: 经常与'{main_conf_class}'混淆，建议:")
                report.append("  - 分析两类别的视觉差异")
                report.append("  - 增强区分性特征的学习")
                report.append("  - 考虑使用对比学习方法")
            
            if cls_info['error_rate'] > 0.8:
                report.append("- **模型架构**: 错误率极高，建议:")
                report.append("  - 检查标签质量和一致性")
                report.append("  - 考虑使用更复杂的模型架构")
                report.append("  - 增加该类别的训练权重")
        
        # 通用改进建议
        report.append(f"\n### 通用改进策略")
        report.append("1. **数据质量优化**:")
        report.append("   - 重新审查稀有类别的标签质量")
        report.append("   - 统一标注标准，减少标注不一致")
        report.append("   - 增加边界案例的标注样本")
        
        report.append("2. **模型优化**:")
        report.append("   - 使用类别权重平衡训练")
        report.append("   - 实施focal loss处理类别不平衡")
        report.append("   - 考虑使用集成学习方法")
        
        report.append("3. **训练策略**:")
        report.append("   - 增加困难样本的训练频次")
        report.append("   - 使用渐进式学习策略")
        report.append("   - 实施交叉验证确保泛化能力")
        
        return '\n'.join(report)
    
    def run_detailed_analysis(self):
        """运行详细的Growth Pattern错误分析"""
        print("开始Growth Pattern详细错误分析...")
        
        # 分析混淆模式
        class_analysis = self.analyze_confusion_patterns()
        
        # 识别问题类别
        problematic_classes = self.identify_problematic_classes(class_analysis)
        
        # 分析混淆对
        confusion_pairs = self.analyze_confusion_pairs()
        
        # 创建可视化
        print("生成详细可视化图表...")
        fig = self.create_detailed_visualizations(class_analysis, problematic_classes, confusion_pairs)
        
        # 保存图表
        reports_dir = Path('/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        viz_path = reports_dir / 'growth_pattern_detailed_analysis.png'
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 生成详细报告
        print("生成详细分析报告...")
        report = self.generate_detailed_report(class_analysis, problematic_classes, confusion_pairs)
        
        report_path = reports_dir / 'growth_pattern_detailed_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存分析数据
        analysis_data = {
            'class_analysis': class_analysis,
            'problematic_classes': problematic_classes,
            'confusion_pairs': confusion_pairs
        }
        
        data_path = reports_dir / 'growth_pattern_analysis_data.json'
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"Growth Pattern详细分析完成!")
        print(f"- 详细可视化: {viz_path}")
        print(f"- 详细报告: {report_path}")
        print(f"- 分析数据: {data_path}")
        
        return analysis_data

def main():
    # 设置实验路径
    experiment_path = "/home/aaa/ws/bioastModel/experiments/multilevel_mobilenetv3_ds"
    
    # 创建分析器并运行分析
    analyzer = GrowthPatternErrorAnalyzer(experiment_path)
    analysis_data = analyzer.run_detailed_analysis()
    
    # 打印关键发现
    print("\n=== Growth Pattern 关键发现 ===")
    problematic_classes = analysis_data['problematic_classes']
    
    print("最严重的问题类别:")
    for i, cls_info in enumerate(problematic_classes[:3]):
        print(f"{i+1}. {cls_info['class_name']}: {cls_info['error_rate']:.2%} "
              f"({cls_info['error_count']}/{cls_info['total_samples']})")
    
    print("\n最容易混淆的类别对:")
    confusion_pairs = analysis_data['confusion_pairs']
    for i, pair in enumerate(confusion_pairs[:3]):
        print(f"{i+1}. {pair['class_a']} ↔ {pair['class_b']}: {pair['mutual_confusion']}次混淆")

if __name__ == "__main__":
    main()