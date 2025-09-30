#!/usr/bin/env python3
"""
Interference Factors任务详细错误分析
深入分析Interference Factors多标签分类任务中的错误模式和特征
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

class InterferenceFactorsErrorAnalyzer:
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
    
    def analyze_factor_performance(self):
        """分析各干扰因子的性能表现"""
        if_results = self.test_results['interference_factors']
        factor_analysis = {}
        
        for factor, result in if_results.items():
            if factor != 'overall_accuracy':
                accuracy = result['accuracy']
                error_rate = 1 - accuracy
                
                # 根据准确率分类性能等级
                if accuracy >= 0.95:
                    performance_level = 'excellent'
                elif accuracy >= 0.90:
                    performance_level = 'good'
                elif accuracy >= 0.80:
                    performance_level = 'fair'
                else:
                    performance_level = 'poor'
                
                factor_analysis[factor] = {
                    'accuracy': accuracy,
                    'error_rate': error_rate,
                    'performance_level': performance_level
                }
        
        # 按错误率排序
        sorted_factors = sorted(factor_analysis.items(), 
                               key=lambda x: x[1]['error_rate'], 
                               reverse=True)
        
        return factor_analysis, sorted_factors
    
    def analyze_error_patterns(self, factor_analysis):
        """分析错误模式和相关性"""
        # 计算错误严重程度分布
        error_distribution = {
            'critical': [],    # 错误率 > 15%
            'high': [],        # 错误率 10-15%
            'medium': [],      # 错误率 5-10%
            'low': []          # 错误率 < 5%
        }
        
        for factor, analysis in factor_analysis.items():
            error_rate = analysis['error_rate']
            if error_rate > 0.15:
                error_distribution['critical'].append(factor)
            elif error_rate > 0.10:
                error_distribution['high'].append(factor)
            elif error_rate > 0.05:
                error_distribution['medium'].append(factor)
            else:
                error_distribution['low'].append(factor)
        
        return error_distribution
    
    def estimate_sample_distribution(self):
        """估算各因子的样本分布（基于准确率推断）"""
        # 由于没有详细的混淆矩阵，我们基于准确率来估算问题严重程度
        if_results = self.test_results['interference_factors']
        total_samples = 3000  # 测试集总样本数
        
        estimated_distribution = {}
        
        for factor, result in if_results.items():
            if factor != 'overall_accuracy':
                accuracy = result['accuracy']
                
                # 估算错误样本数（假设每个因子独立评估）
                estimated_errors = int(total_samples * (1 - accuracy))
                estimated_correct = total_samples - estimated_errors
                
                estimated_distribution[factor] = {
                    'estimated_total_samples': total_samples,
                    'estimated_correct': estimated_correct,
                    'estimated_errors': estimated_errors,
                    'accuracy': accuracy
                }
        
        return estimated_distribution
    
    def create_interference_visualizations(self, factor_analysis, error_distribution, estimated_distribution):
        """创建Interference Factors的详细可视化分析"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. 各因子准确率对比
        ax1 = plt.subplot(2, 3, 1)
        factors = list(factor_analysis.keys())
        accuracies = [factor_analysis[f]['accuracy'] * 100 for f in factors]
        
        # 根据性能等级设置颜色
        colors = []
        for f in factors:
            level = factor_analysis[f]['performance_level']
            if level == 'excellent':
                colors.append('#2ecc71')  # 绿色
            elif level == 'good':
                colors.append('#f39c12')  # 橙色
            elif level == 'fair':
                colors.append('#e74c3c')  # 红色
            else:
                colors.append('#8e44ad')  # 紫色
        
        bars = ax1.bar(factors, accuracies, color=colors, alpha=0.7)
        ax1.set_ylabel('准确率 (%)')
        ax1.set_title('各干扰因子准确率对比', fontweight='bold')
        ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90%基准线')
        ax1.legend()
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. 错误率分布
        ax2 = plt.subplot(2, 3, 2)
        error_rates = [factor_analysis[f]['error_rate'] * 100 for f in factors]
        
        bars = ax2.bar(factors, error_rates, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('错误率 (%)')
        ax2.set_title('各干扰因子错误率', fontweight='bold')
        
        # 添加数值标签
        for bar, rate in zip(bars, error_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. 错误严重程度分布饼图
        ax3 = plt.subplot(2, 3, 3)
        severity_counts = [len(error_distribution[level]) for level in error_distribution.keys()]
        severity_labels = ['严重 (>15%)', '高 (10-15%)', '中等 (5-10%)', '低 (<5%)']
        colors_pie = ['darkred', 'red', 'orange', 'green']
        
        wedges, texts, autotexts = ax3.pie(severity_counts, labels=severity_labels,
                                          colors=colors_pie, autopct='%1.0f%%',
                                          startangle=90)
        ax3.set_title('错误严重程度分布', fontweight='bold')
        
        # 4. 估算错误样本数量
        ax4 = plt.subplot(2, 3, 4)
        estimated_errors = [estimated_distribution[f]['estimated_errors'] for f in factors]
        
        bars = ax4.bar(factors, estimated_errors, color='lightblue', alpha=0.7)
        ax4.set_ylabel('估算错误样本数')
        ax4.set_title('各因子估算错误样本数量', fontweight='bold')
        
        # 添加数值标签
        for bar, count in zip(bars, estimated_errors):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 5. 性能等级分布
        ax5 = plt.subplot(2, 3, 5)
        performance_levels = ['excellent', 'good', 'fair', 'poor']
        level_counts = []
        level_factors = []
        
        for level in performance_levels:
            count = sum(1 for f in factor_analysis.values() if f['performance_level'] == level)
            level_counts.append(count)
            factors_in_level = [f for f, analysis in factor_analysis.items() 
                               if analysis['performance_level'] == level]
            level_factors.append(factors_in_level)
        
        level_labels = ['优秀 (≥95%)', '良好 (90-95%)', '一般 (80-90%)', '差 (<80%)']
        colors_level = ['darkgreen', 'green', 'orange', 'red']
        
        bars = ax5.bar(level_labels, level_counts, color=colors_level, alpha=0.7)
        ax5.set_ylabel('因子数量')
        ax5.set_title('性能等级分布', fontweight='bold')
        ax5.set_xticklabels(level_labels, rotation=45, ha='right')
        
        # 添加数值标签和因子名称
        for bar, count, factors_list in zip(bars, level_counts, level_factors):
            if count > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        str(count), ha='center', va='bottom', fontweight='bold')
                # 在条形图下方显示因子名称
                factor_names = ', '.join(factors_list)
                ax5.text(bar.get_x() + bar.get_width()/2, -0.15,
                        factor_names, ha='center', va='top', fontsize=8,
                        rotation=0, wrap=True)
        
        # 6. 准确率vs错误率散点图
        ax6 = plt.subplot(2, 3, 6)
        
        x_values = list(range(len(factors)))
        y_accuracies = [factor_analysis[f]['accuracy'] * 100 for f in factors]
        y_errors = [factor_analysis[f]['error_rate'] * 100 for f in factors]
        
        # 绘制准确率和错误率的对比
        width = 0.35
        ax6.bar([x - width/2 for x in x_values], y_accuracies, width, 
               label='准确率', color='lightgreen', alpha=0.7)
        ax6.bar([x + width/2 for x in x_values], y_errors, width,
               label='错误率', color='lightcoral', alpha=0.7)
        
        ax6.set_xlabel('干扰因子')
        ax6.set_ylabel('百分比 (%)')
        ax6.set_title('准确率 vs 错误率对比', fontweight='bold')
        ax6.set_xticks(x_values)
        ax6.set_xticklabels(factors, rotation=45, ha='right')
        ax6.legend()
        
        plt.tight_layout()
        return fig
    
    def generate_interference_report(self, factor_analysis, error_distribution, estimated_distribution):
        """生成Interference Factors详细分析报告"""
        report = []
        report.append("# Interference Factors 任务详细错误分析报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 总体统计
        overall_accuracy = self.test_results['interference_factors']['overall_accuracy']
        total_factors = len(factor_analysis)
        
        report.append(f"\n## 总体统计")
        report.append(f"- 干扰因子总数: {total_factors}")
        report.append(f"- 总体准确率: {overall_accuracy:.2%}")
        report.append(f"- 总体错误率: {1-overall_accuracy:.2%}")
        
        # 各因子详细分析
        report.append(f"\n## 各干扰因子详细分析")
        
        sorted_factors = sorted(factor_analysis.items(), 
                               key=lambda x: x[1]['error_rate'], 
                               reverse=True)
        
        for i, (factor, analysis) in enumerate(sorted_factors):
            report.append(f"\n### {i+1}. {factor}")
            report.append(f"- 准确率: {analysis['accuracy']:.2%}")
            report.append(f"- 错误率: {analysis['error_rate']:.2%}")
            report.append(f"- 性能等级: {analysis['performance_level']}")
            
            # 估算的错误样本信息
            est_info = estimated_distribution[factor]
            report.append(f"- 估算错误样本数: {est_info['estimated_errors']:,}")
            
            # 性能评价
            if analysis['performance_level'] == 'excellent':
                report.append("- 评价: 表现优秀，检测准确率很高")
            elif analysis['performance_level'] == 'good':
                report.append("- 评价: 表现良好，有小幅改进空间")
            elif analysis['performance_level'] == 'fair':
                report.append("- 评价: 表现一般，需要重点改进")
            else:
                report.append("- 评价: 表现较差，急需优化")
        
        # 错误严重程度分析
        report.append(f"\n## 错误严重程度分析")
        
        for severity, factors_list in error_distribution.items():
            if factors_list:
                severity_names = {
                    'critical': '严重错误 (>15%)',
                    'high': '高错误率 (10-15%)',
                    'medium': '中等错误率 (5-10%)',
                    'low': '低错误率 (<5%)'
                }
                
                report.append(f"\n### {severity_names[severity]}")
                report.append(f"- 因子数量: {len(factors_list)}")
                report.append(f"- 涉及因子: {', '.join(factors_list)}")
                
                if severity == 'critical':
                    report.append("- 建议: 立即优化，考虑重新设计检测算法")
                elif severity == 'high':
                    report.append("- 建议: 优先改进，增加训练数据和调整模型参数")
                elif severity == 'medium':
                    report.append("- 建议: 适度改进，优化特征提取方法")
                else:
                    report.append("- 建议: 保持现状，定期监控性能")
        
        # 关键发现
        report.append(f"\n## 关键发现")
        
        # 找出最大问题
        worst_factor = max(factor_analysis.items(), key=lambda x: x[1]['error_rate'])
        best_factor = min(factor_analysis.items(), key=lambda x: x[1]['error_rate'])
        
        report.append(f"1. **最大问题因子**: {worst_factor[0]} (错误率: {worst_factor[1]['error_rate']:.2%})")
        report.append(f"2. **最佳表现因子**: {best_factor[0]} (错误率: {best_factor[1]['error_rate']:.2%})")
        
        # 性能差异分析
        error_rates = [analysis['error_rate'] for analysis in factor_analysis.values()]
        max_error = max(error_rates)
        min_error = min(error_rates)
        avg_error = sum(error_rates) / len(error_rates)
        
        report.append(f"3. **性能差异**: 最大错误率差异 {(max_error - min_error):.2%}")
        report.append(f"4. **平均错误率**: {avg_error:.2%}")
        
        # 改进建议
        report.append(f"\n## 改进建议")
        
        # 针对最差因子的建议
        if worst_factor[1]['error_rate'] > 0.15:
            report.append(f"\n### 针对 {worst_factor[0]} 的紧急改进建议")
            report.append("1. **数据质量检查**:")
            report.append("   - 重新审查该因子的标注质量")
            report.append("   - 检查标注一致性和准确性")
            report.append("   - 增加该因子的标注样本数量")
            
            report.append("2. **算法优化**:")
            report.append("   - 设计专门的检测算法")
            report.append("   - 调整检测阈值和参数")
            report.append("   - 考虑使用集成学习方法")
            
            report.append("3. **特征工程**:")
            report.append("   - 分析该因子的视觉特征")
            report.append("   - 设计针对性的特征提取器")
            report.append("   - 使用多尺度特征融合")
        
        # 通用改进策略
        report.append(f"\n### 通用改进策略")
        
        report.append("1. **多标签学习优化**:")
        report.append("   - 考虑因子间的相关性")
        report.append("   - 使用标签相关性建模")
        report.append("   - 实施层次化分类策略")
        
        report.append("2. **不平衡数据处理**:")
        report.append("   - 分析各因子的样本分布")
        report.append("   - 使用重采样技术平衡数据")
        report.append("   - 调整损失函数权重")
        
        report.append("3. **模型架构改进**:")
        report.append("   - 考虑使用注意力机制")
        report.append("   - 实施多任务学习框架")
        report.append("   - 增加模型的表达能力")
        
        report.append("4. **评估和监控**:")
        report.append("   - 建立持续监控系统")
        report.append("   - 定期评估各因子性能")
        report.append("   - 收集用户反馈进行改进")
        
        # 优先级建议
        critical_factors = error_distribution.get('critical', [])
        high_factors = error_distribution.get('high', [])
        
        if critical_factors or high_factors:
            report.append(f"\n## 改进优先级")
            
            if critical_factors:
                report.append(f"### 第一优先级 (立即处理)")
                for factor in critical_factors:
                    error_rate = factor_analysis[factor]['error_rate']
                    report.append(f"- {factor}: {error_rate:.2%} 错误率")
            
            if high_factors:
                report.append(f"### 第二优先级 (近期处理)")
                for factor in high_factors:
                    error_rate = factor_analysis[factor]['error_rate']
                    report.append(f"- {factor}: {error_rate:.2%} 错误率")
        
        return '\n'.join(report)
    
    def run_interference_analysis(self):
        """运行完整的Interference Factors错误分析"""
        print("开始Interference Factors详细错误分析...")
        
        # 分析各因子性能
        factor_analysis, sorted_factors = self.analyze_factor_performance()
        
        # 分析错误模式
        error_distribution = self.analyze_error_patterns(factor_analysis)
        
        # 估算样本分布
        estimated_distribution = self.estimate_sample_distribution()
        
        # 创建可视化
        print("生成Interference Factors可视化图表...")
        fig = self.create_interference_visualizations(factor_analysis, error_distribution, estimated_distribution)
        
        # 保存图表
        reports_dir = Path('/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        viz_path = reports_dir / 'interference_factors_detailed_analysis.png'
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 生成详细报告
        print("生成Interference Factors详细分析报告...")
        report = self.generate_interference_report(factor_analysis, error_distribution, estimated_distribution)
        
        report_path = reports_dir / 'interference_factors_detailed_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存分析数据
        analysis_data = {
            'factor_analysis': factor_analysis,
            'sorted_factors': sorted_factors,
            'error_distribution': error_distribution,
            'estimated_distribution': estimated_distribution
        }
        
        data_path = reports_dir / 'interference_factors_analysis_data.json'
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"Interference Factors详细分析完成!")
        print(f"- 详细可视化: {viz_path}")
        print(f"- 详细报告: {report_path}")
        print(f"- 分析数据: {data_path}")
        
        return analysis_data

def main():
    # 设置实验路径
    experiment_path = "/home/aaa/ws/bioastModel/experiments/multilevel_mobilenetv3_ds"
    
    # 创建分析器并运行分析
    analyzer = InterferenceFactorsErrorAnalyzer(experiment_path)
    analysis_data = analyzer.run_interference_analysis()
    
    # 打印关键发现
    print("\n=== Interference Factors 关键发现 ===")
    sorted_factors = analysis_data['sorted_factors']
    
    print("各因子错误率排序:")
    for i, (factor, analysis) in enumerate(sorted_factors):
        print(f"{i+1}. {factor}: {analysis['error_rate']:.2%} ({analysis['performance_level']})")
    
    print(f"\n错误严重程度分布:")
    error_distribution = analysis_data['error_distribution']
    for severity, factors_list in error_distribution.items():
        if factors_list:
            print(f"- {severity}: {len(factors_list)}个因子 ({', '.join(factors_list)})")

if __name__ == "__main__":
    main()