#!/usr/bin/env python3
"""
错误样本分析脚本
分析模型在各任务上的错误预测样本，识别错误模式和特征
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

class ErrorSampleAnalyzer:
    def __init__(self, experiment_path):
        self.experiment_path = Path(experiment_path)
        self.test_results = None
        self.label_info = None
        self.load_data()
        
    def load_data(self):
        """加载测试结果和标签信息"""
        # 加载测试结果
        with open(self.experiment_path / 'test_results.json', 'r') as f:
            self.test_results = json.load(f)
            
        # 加载标签信息
        with open(self.experiment_path / 'label_info.json', 'r') as f:
            self.label_info = json.load(f)
            
    def analyze_growth_level_errors(self):
        """分析Growth Level任务的错误样本"""
        confusion_matrix = np.array(self.test_results['growth_level']['confusion_matrix'])
        labels = list(self.label_info['growth_level'].keys())
        
        # 计算错误样本数量
        total_samples = np.sum(confusion_matrix)
        correct_predictions = np.trace(confusion_matrix)
        error_samples = total_samples - correct_predictions
        
        # 分析具体错误类型
        false_negatives = confusion_matrix[1, 0]  # positive预测为negative
        false_positives = confusion_matrix[0, 1]  # negative预测为positive
        
        analysis = {
            'task': 'Growth Level',
            'total_samples': int(total_samples),
            'error_samples': int(error_samples),
            'error_rate': error_samples / total_samples,
            'false_negatives': int(false_negatives),
            'false_positives': int(false_positives),
            'fn_rate': false_negatives / (false_negatives + confusion_matrix[1, 1]),
            'fp_rate': false_positives / (false_positives + confusion_matrix[0, 0]),
            'confusion_matrix': confusion_matrix.tolist(),
            'labels': labels
        }
        
        return analysis
        
    def analyze_growth_pattern_errors(self):
        """分析Growth Pattern任务的错误样本"""
        confusion_matrix = np.array(self.test_results['growth_pattern']['confusion_matrix'])
        labels = list(self.label_info['growth_pattern'].keys())
        
        # 计算每个类别的错误情况
        class_errors = {}
        total_samples = np.sum(confusion_matrix)
        
        for i, label in enumerate(labels):
            true_samples = np.sum(confusion_matrix[i, :])
            correct_predictions = confusion_matrix[i, i]
            errors = true_samples - correct_predictions
            
            if true_samples > 0:
                error_rate = errors / true_samples
                
                # 找出主要的错误预测类别
                error_predictions = confusion_matrix[i, :].copy()
                error_predictions[i] = 0  # 排除正确预测
                main_error_class = np.argmax(error_predictions)
                main_error_count = error_predictions[main_error_class]
                
                class_errors[label] = {
                    'true_samples': int(true_samples),
                    'correct_predictions': int(correct_predictions),
                    'errors': int(errors),
                    'error_rate': float(error_rate),
                    'main_error_class': labels[main_error_class] if main_error_count > 0 else None,
                    'main_error_count': int(main_error_count)
                }
        
        # 找出错误率最高的类别
        high_error_classes = sorted(class_errors.items(), 
                                  key=lambda x: x[1]['error_rate'], 
                                  reverse=True)[:3]
        
        analysis = {
            'task': 'Growth Pattern',
            'total_samples': int(total_samples),
            'overall_accuracy': self.test_results['growth_pattern']['accuracy'],
            'class_errors': class_errors,
            'high_error_classes': high_error_classes,
            'confusion_matrix': confusion_matrix.tolist(),
            'labels': labels
        }
        
        return analysis
        
    def analyze_interference_factors_errors(self):
        """分析Interference Factors任务的错误样本"""
        if_results = self.test_results['interference_factors']
        
        # 计算各因子的错误率
        factor_errors = {}
        for factor, result in if_results.items():
            if factor != 'overall_accuracy':
                accuracy = result['accuracy']
                error_rate = 1 - accuracy
                factor_errors[factor] = {
                    'accuracy': accuracy,
                    'error_rate': error_rate
                }
        
        # 找出错误率最高的因子
        high_error_factors = sorted(factor_errors.items(), 
                                  key=lambda x: x[1]['error_rate'], 
                                  reverse=True)
        
        analysis = {
            'task': 'Interference Factors',
            'overall_accuracy': if_results['overall_accuracy'],
            'factor_errors': factor_errors,
            'high_error_factors': high_error_factors,
            'worst_factor': high_error_factors[0] if high_error_factors else None
        }
        
        return analysis
        
    def create_error_visualizations(self, analyses):
        """创建错误分析可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型错误样本分析', fontsize=16, fontweight='bold')
        
        # 1. Growth Level错误分析
        ax1 = axes[0, 0]
        gl_analysis = analyses['growth_level']
        categories = ['False Negatives', 'False Positives']
        values = [gl_analysis['false_negatives'], gl_analysis['false_positives']]
        colors = ['#ff6b6b', '#4ecdc4']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_title('Growth Level 错误类型分布', fontweight='bold')
        ax1.set_ylabel('错误样本数量')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 2. Growth Pattern类别错误率
        ax2 = axes[0, 1]
        gp_analysis = analyses['growth_pattern']
        class_names = []
        error_rates = []
        
        for class_name, error_info in gp_analysis['class_errors'].items():
            if error_info['true_samples'] > 10:  # 只显示样本数量足够的类别
                class_names.append(class_name.replace('_', '\n'))
                error_rates.append(error_info['error_rate'] * 100)
        
        bars = ax2.bar(range(len(class_names)), error_rates, 
                      color='#ff9999', alpha=0.7)
        ax2.set_title('Growth Pattern 各类别错误率', fontweight='bold')
        ax2.set_ylabel('错误率 (%)')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, rate in zip(bars, error_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Interference Factors错误率
        ax3 = axes[1, 0]
        if_analysis = analyses['interference_factors']
        factors = list(if_analysis['factor_errors'].keys())
        if_error_rates = [if_analysis['factor_errors'][f]['error_rate'] * 100 
                         for f in factors]
        
        bars = ax3.bar(factors, if_error_rates, color='#ffb366', alpha=0.7)
        ax3.set_title('Interference Factors 各因子错误率', fontweight='bold')
        ax3.set_ylabel('错误率 (%)')
        ax3.set_xticklabels(factors, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, rate in zip(bars, if_error_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. 整体任务错误率对比
        ax4 = axes[1, 1]
        tasks = ['Growth Level', 'Growth Pattern', 'Interference Factors']
        overall_error_rates = [
            gl_analysis['error_rate'] * 100,
            (1 - gp_analysis['overall_accuracy']) * 100,
            (1 - if_analysis['overall_accuracy']) * 100
        ]
        
        colors = ['#66b3ff', '#99ff99', '#ffcc99']
        bars = ax4.bar(tasks, overall_error_rates, color=colors, alpha=0.7)
        ax4.set_title('各任务整体错误率对比', fontweight='bold')
        ax4.set_ylabel('错误率 (%)')
        ax4.set_xticklabels(tasks, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, rate in zip(bars, overall_error_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
        
    def create_confusion_matrix_heatmap(self, analysis):
        """创建Growth Pattern的混淆矩阵热图"""
        if analysis['task'] != 'Growth Pattern':
            return None
            
        confusion_matrix = np.array(analysis['confusion_matrix'])
        labels = analysis['labels']
        
        # 计算百分比矩阵
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        percentage_matrix = confusion_matrix / row_sums * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 绝对数量热图
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title('Growth Pattern 混淆矩阵 (绝对数量)', fontweight='bold')
        ax1.set_xlabel('预测标签')
        ax1.set_ylabel('真实标签')
        
        # 百分比热图
        sns.heatmap(percentage_matrix, annot=True, fmt='.1f', cmap='Reds',
                   xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title('Growth Pattern 混淆矩阵 (百分比)', fontweight='bold')
        ax2.set_xlabel('预测标签')
        ax2.set_ylabel('真实标签')
        
        plt.tight_layout()
        return fig
        
    def generate_error_analysis_report(self, analyses):
        """生成错误分析报告"""
        report = []
        report.append("# 模型错误样本分析报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## 执行摘要")
        
        # Growth Level分析
        gl_analysis = analyses['growth_level']
        report.append(f"\n### Growth Level 任务")
        report.append(f"- 总样本数: {gl_analysis['total_samples']:,}")
        report.append(f"- 错误样本数: {gl_analysis['error_samples']:,}")
        report.append(f"- 错误率: {gl_analysis['error_rate']:.2%}")
        report.append(f"- 假阴性 (FN): {gl_analysis['false_negatives']} (漏检率: {gl_analysis['fn_rate']:.2%})")
        report.append(f"- 假阳性 (FP): {gl_analysis['false_positives']} (误检率: {gl_analysis['fp_rate']:.2%})")
        
        # Growth Pattern分析
        gp_analysis = analyses['growth_pattern']
        report.append(f"\n### Growth Pattern 任务")
        report.append(f"- 总体准确率: {gp_analysis['overall_accuracy']:.2%}")
        report.append(f"- 错误率最高的类别:")
        
        for i, (class_name, error_info) in enumerate(gp_analysis['high_error_classes'][:3]):
            report.append(f"  {i+1}. {class_name}: {error_info['error_rate']:.2%} "
                         f"({error_info['errors']}/{error_info['true_samples']})")
            if error_info['main_error_class']:
                report.append(f"     主要误分类为: {error_info['main_error_class']} "
                             f"({error_info['main_error_count']}次)")
        
        # Interference Factors分析
        if_analysis = analyses['interference_factors']
        report.append(f"\n### Interference Factors 任务")
        report.append(f"- 总体准确率: {if_analysis['overall_accuracy']:.2%}")
        report.append(f"- 各因子错误率:")
        
        for factor, error_info in if_analysis['high_error_factors']:
            report.append(f"  - {factor}: {error_info['error_rate']:.2%}")
        
        # 关键发现
        report.append(f"\n## 关键发现")
        
        # 找出最大问题
        worst_task = max([
            ('Growth Level', gl_analysis['error_rate']),
            ('Growth Pattern', 1 - gp_analysis['overall_accuracy']),
            ('Interference Factors', 1 - if_analysis['overall_accuracy'])
        ], key=lambda x: x[1])
        
        report.append(f"1. **最需要改进的任务**: {worst_task[0]} (错误率: {worst_task[1]:.2%})")
        
        # Growth Pattern具体问题
        if gp_analysis['high_error_classes']:
            worst_class = gp_analysis['high_error_classes'][0]
            report.append(f"2. **Growth Pattern最大问题**: {worst_class[0]}类别错误率高达{worst_class[1]['error_rate']:.2%}")
        
        # Interference Factors问题
        if if_analysis['worst_factor']:
            worst_factor = if_analysis['worst_factor']
            report.append(f"3. **Interference Factors最大问题**: {worst_factor[0]}因子错误率{worst_factor[1]['error_rate']:.2%}")
        
        # 改进建议
        report.append(f"\n## 改进建议")
        
        if gl_analysis['fn_rate'] > gl_analysis['fp_rate']:
            report.append("1. **Growth Level**: 假阴性率较高，建议降低分类阈值或增加正样本的训练权重")
        else:
            report.append("1. **Growth Level**: 假阳性率较高，建议提高分类阈值或增强负样本特征学习")
        
        if gp_analysis['high_error_classes']:
            worst_gp_class = gp_analysis['high_error_classes'][0]
            report.append(f"2. **Growth Pattern**: 重点关注{worst_gp_class[0]}类别，考虑:")
            report.append("   - 增加该类别的训练样本")
            report.append("   - 使用数据增强技术")
            report.append("   - 调整类别权重")
        
        if if_analysis['worst_factor']:
            worst_if_factor = if_analysis['worst_factor']
            report.append(f"3. **Interference Factors**: {worst_if_factor[0]}因子识别困难，建议:")
            report.append("   - 收集更多该因子的标注样本")
            report.append("   - 优化特征提取方法")
            report.append("   - 考虑使用专门的检测模块")
        
        return '\n'.join(report)
        
    def run_complete_analysis(self):
        """运行完整的错误分析"""
        print("开始错误样本分析...")
        
        # 分析各任务的错误样本
        analyses = {
            'growth_level': self.analyze_growth_level_errors(),
            'growth_pattern': self.analyze_growth_pattern_errors(),
            'interference_factors': self.analyze_interference_factors_errors()
        }
        
        # 创建可视化图表
        print("生成错误分析图表...")
        error_viz_fig = self.create_error_visualizations(analyses)
        
        # 确保保存目录存在
        reports_dir = Path('/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        error_viz_path = reports_dir / 'error_analysis_visualization.png'
        error_viz_fig.savefig(error_viz_path, dpi=300, bbox_inches='tight')
        plt.close(error_viz_fig)
        
        # 创建混淆矩阵热图
        confusion_fig = self.create_confusion_matrix_heatmap(analyses['growth_pattern'])
        if confusion_fig:
            confusion_path = reports_dir / 'growth_pattern_confusion_matrix.png'
            confusion_fig.savefig(confusion_path, dpi=300, bbox_inches='tight')
            plt.close(confusion_fig)
        
        # 生成分析报告
        print("生成错误分析报告...")
        report = self.generate_error_analysis_report(analyses)
        report_path = reports_dir / 'error_sample_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存详细分析数据
        analysis_data_path = reports_dir / 'error_analysis_data.json'
        with open(analysis_data_path, 'w', encoding='utf-8') as f:
            json.dump(analyses, f, indent=2, ensure_ascii=False)
        
        print(f"错误分析完成!")
        print(f"- 可视化图表: {error_viz_path}")
        print(f"- 混淆矩阵: {confusion_path}")
        print(f"- 分析报告: {report_path}")
        print(f"- 详细数据: {analysis_data_path}")
        
        return analyses

def main():
    # 设置实验路径
    experiment_path = "/home/aaa/ws/bioastModel/experiments/multilevel_mobilenetv3_ds"
    
    # 创建分析器并运行分析
    analyzer = ErrorSampleAnalyzer(experiment_path)
    analyses = analyzer.run_complete_analysis()
    
    # 打印关键统计信息
    print("\n=== 错误分析关键统计 ===")
    print(f"Growth Level 错误率: {analyses['growth_level']['error_rate']:.2%}")
    print(f"Growth Pattern 错误率: {1-analyses['growth_pattern']['overall_accuracy']:.2%}")
    print(f"Interference Factors 错误率: {1-analyses['interference_factors']['overall_accuracy']:.2%}")

if __name__ == "__main__":
    main()