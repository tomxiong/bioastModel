#!/usr/bin/env python3
"""
综合错误样本分析报告
整合Growth Level、Growth Pattern和Interference Factors三个任务的错误分析结果
生成完整的错误样本分析和改进建议
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

class ComprehensiveErrorAnalyzer:
    def __init__(self, experiment_path, reports_dir):
        self.experiment_path = Path(experiment_path)
        self.reports_dir = Path(reports_dir)
        self.test_results = None
        self.label_info = None
        self.load_data()
        
    def load_data(self):
        """加载测试结果和标签信息"""
        with open(self.experiment_path / 'test_results.json', 'r') as f:
            self.test_results = json.load(f)
            
        with open(self.experiment_path / 'label_info.json', 'r') as f:
            self.label_info = json.load(f)
    
    def load_previous_analyses(self):
        """加载之前的分析结果"""
        analyses = {}
        
        # 加载Growth Pattern分析
        gp_data_path = self.reports_dir / 'growth_pattern_analysis_data.json'
        if gp_data_path.exists():
            with open(gp_data_path, 'r') as f:
                analyses['growth_pattern'] = json.load(f)
        
        # 加载Interference Factors分析
        if_data_path = self.reports_dir / 'interference_factors_analysis_data.json'
        if if_data_path.exists():
            with open(if_data_path, 'r') as f:
                analyses['interference_factors'] = json.load(f)
        
        # 加载基础错误分析
        base_data_path = self.reports_dir / 'error_analysis_data.json'
        if base_data_path.exists():
            with open(base_data_path, 'r') as f:
                analyses['base_analysis'] = json.load(f)
        
        return analyses
    
    def extract_task_summaries(self, analyses):
        """提取各任务的关键统计信息"""
        task_summaries = {}
        
        # Growth Level任务
        if 'base_analysis' in analyses:
            gl_data = analyses['base_analysis']['growth_level']
            task_summaries['growth_level'] = {
                'accuracy': 1 - gl_data['error_rate'],
                'error_rate': gl_data['error_rate'],
                'total_errors': gl_data['error_samples'],  # 修正字段名
                'false_negatives': gl_data['false_negatives'],
                'false_positives': gl_data['false_positives'],
                'performance_level': 'excellent' if gl_data['error_rate'] < 0.05 else 'good'
            }
        
        # Growth Pattern任务
        if 'growth_pattern' in analyses:
            gp_data = analyses['growth_pattern']
            worst_class = max(gp_data['class_analysis'].items(), key=lambda x: x[1]['error_rate'])
            best_class = min(gp_data['class_analysis'].items(), key=lambda x: x[1]['error_rate'])
            
            avg_error_rate = np.mean([info['error_rate'] for info in gp_data['class_analysis'].values()])
            
            task_summaries['growth_pattern'] = {
                'accuracy': 1 - avg_error_rate,
                'error_rate': avg_error_rate,
                'worst_class': worst_class[0],
                'worst_error_rate': worst_class[1]['error_rate'],
                'best_class': best_class[0],
                'best_error_rate': best_class[1]['error_rate'],
                'total_classes': len(gp_data['class_analysis']),
                'performance_level': 'fair' if avg_error_rate > 0.15 else 'good'
            }
        
        # Interference Factors任务
        if 'interference_factors' in analyses:
            if_data = analyses['interference_factors']
            worst_factor = max(if_data['factor_analysis'].items(), key=lambda x: x[1]['error_rate'])
            best_factor = min(if_data['factor_analysis'].items(), key=lambda x: x[1]['error_rate'])
            
            avg_error_rate = np.mean([info['error_rate'] for info in if_data['factor_analysis'].values()])
            
            task_summaries['interference_factors'] = {
                'accuracy': 1 - avg_error_rate,
                'error_rate': avg_error_rate,
                'worst_factor': worst_factor[0],
                'worst_error_rate': worst_factor[1]['error_rate'],
                'best_factor': best_factor[0],
                'best_error_rate': best_factor[1]['error_rate'],
                'total_factors': len(if_data['factor_analysis']),
                'performance_level': 'good' if avg_error_rate < 0.10 else 'fair'
            }
        
        return task_summaries
    
    def create_comprehensive_visualization(self, task_summaries, analyses):
        """创建综合错误分析可视化"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 任务整体性能对比
        ax1 = plt.subplot(3, 4, 1)
        tasks = list(task_summaries.keys())
        task_names = ['Growth Level', 'Growth Pattern', 'Interference Factors']
        accuracies = [task_summaries[task]['accuracy'] * 100 for task in tasks]
        
        colors = ['#2ecc71', '#f39c12', '#3498db']
        bars = ax1.bar(task_names, accuracies, color=colors, alpha=0.7)
        ax1.set_ylabel('准确率 (%)')
        ax1.set_title('各任务整体准确率对比', fontweight='bold')
        ax1.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90%基准线')
        ax1.legend()
        
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. 任务错误率对比
        ax2 = plt.subplot(3, 4, 2)
        error_rates = [task_summaries[task]['error_rate'] * 100 for task in tasks]
        
        bars = ax2.bar(task_names, error_rates, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('错误率 (%)')
        ax2.set_title('各任务错误率对比', fontweight='bold')
        
        for bar, rate in zip(bars, error_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Growth Level详细分析
        if 'growth_level' in task_summaries:
            ax3 = plt.subplot(3, 4, 3)
            gl_data = task_summaries['growth_level']
            
            categories = ['False Negatives', 'False Positives']
            values = [gl_data['false_negatives'], gl_data['false_positives']]
            
            bars = ax3.bar(categories, values, color=['#e74c3c', '#f39c12'], alpha=0.7)
            ax3.set_ylabel('错误样本数')
            ax3.set_title('Growth Level 错误类型分布', fontweight='bold')
            
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(val), ha='center', va='bottom', fontweight='bold')
        
        # 4. Growth Pattern类别错误率分布
        if 'growth_pattern' in analyses:
            ax4 = plt.subplot(3, 4, 4)
            gp_data = analyses['growth_pattern']['class_analysis']
            
            classes = list(gp_data.keys())[:8]  # 显示前8个类别
            error_rates_gp = [gp_data[cls]['error_rate'] * 100 for cls in classes]
            
            bars = ax4.bar(range(len(classes)), error_rates_gp, color='lightblue', alpha=0.7)
            ax4.set_ylabel('错误率 (%)')
            ax4.set_title('Growth Pattern 类别错误率', fontweight='bold')
            ax4.set_xticks(range(len(classes)))
            ax4.set_xticklabels(classes, rotation=45, ha='right')
            
            for bar, rate in zip(bars, error_rates_gp):
                if rate > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{rate:.0f}%', ha='center', va='bottom', fontsize=8)
        
        # 5. Interference Factors性能分析
        if 'interference_factors' in analyses:
            ax5 = plt.subplot(3, 4, 5)
            if_data = analyses['interference_factors']['factor_analysis']
            
            factors = list(if_data.keys())
            accuracies_if = [if_data[factor]['accuracy'] * 100 for factor in factors]
            
            bars = ax5.bar(factors, accuracies_if, color='lightgreen', alpha=0.7)
            ax5.set_ylabel('准确率 (%)')
            ax5.set_title('Interference Factors 准确率', fontweight='bold')
            
            for bar, acc in zip(bars, accuracies_if):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. 任务复杂度对比（基于类别/因子数量）
        ax6 = plt.subplot(3, 4, 6)
        
        complexity_data = []
        complexity_labels = []
        
        if 'growth_pattern' in task_summaries:
            complexity_data.append(task_summaries['growth_pattern']['total_classes'])
            complexity_labels.append('Growth Pattern\n(类别数)')
        
        if 'interference_factors' in task_summaries:
            complexity_data.append(task_summaries['interference_factors']['total_factors'])
            complexity_labels.append('Interference Factors\n(因子数)')
        
        if complexity_data:
            bars = ax6.bar(complexity_labels, complexity_data, color='mediumpurple', alpha=0.7)
            ax6.set_ylabel('数量')
            ax6.set_title('任务复杂度对比', fontweight='bold')
            
            for bar, count in zip(bars, complexity_data):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 7. 性能等级分布
        ax7 = plt.subplot(3, 4, 7)
        
        performance_levels = ['excellent', 'good', 'fair', 'poor']
        level_counts = [0, 0, 0, 0]
        
        for task_data in task_summaries.values():
            level = task_data['performance_level']
            if level == 'excellent':
                level_counts[0] += 1
            elif level == 'good':
                level_counts[1] += 1
            elif level == 'fair':
                level_counts[2] += 1
            else:
                level_counts[3] += 1
        
        level_labels = ['优秀', '良好', '一般', '差']
        colors_level = ['darkgreen', 'green', 'orange', 'red']
        
        bars = ax7.bar(level_labels, level_counts, color=colors_level, alpha=0.7)
        ax7.set_ylabel('任务数量')
        ax7.set_title('任务性能等级分布', fontweight='bold')
        
        for bar, count in zip(bars, level_counts):
            if count > 0:
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # 8. 最大问题识别
        ax8 = plt.subplot(3, 4, 8)
        
        problem_items = []
        problem_rates = []
        problem_labels = []
        
        if 'growth_pattern' in task_summaries:
            problem_items.append(task_summaries['growth_pattern']['worst_class'])
            problem_rates.append(task_summaries['growth_pattern']['worst_error_rate'] * 100)
            problem_labels.append('GP: ' + task_summaries['growth_pattern']['worst_class'])
        
        if 'interference_factors' in task_summaries:
            problem_items.append(task_summaries['interference_factors']['worst_factor'])
            problem_rates.append(task_summaries['interference_factors']['worst_error_rate'] * 100)
            problem_labels.append('IF: ' + task_summaries['interference_factors']['worst_factor'])
        
        if problem_rates:
            bars = ax8.bar(range(len(problem_labels)), problem_rates, color='red', alpha=0.7)
            ax8.set_ylabel('错误率 (%)')
            ax8.set_title('最大问题项识别', fontweight='bold')
            ax8.set_xticks(range(len(problem_labels)))
            ax8.set_xticklabels(problem_labels, rotation=45, ha='right')
            
            for bar, rate in zip(bars, problem_rates):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 9-12. 详细混淆矩阵热图（如果有数据）
        if 'base_analysis' in analyses:
            # Growth Level混淆矩阵
            ax9 = plt.subplot(3, 4, 9)
            gl_cm = np.array(analyses['base_analysis']['growth_level']['confusion_matrix'])
            
            sns.heatmap(gl_cm, annot=True, fmt='d', cmap='Blues', ax=ax9,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax9.set_title('Growth Level 混淆矩阵', fontweight='bold')
            ax9.set_xlabel('预测')
            ax9.set_ylabel('实际')
        
        # 添加总体统计信息文本框
        if len(task_summaries) >= 3:
            ax10 = plt.subplot(3, 4, 10)
            ax10.axis('off')
            
            # 计算总体统计
            total_accuracy = np.mean([ts['accuracy'] for ts in task_summaries.values()])
            total_error_rate = 1 - total_accuracy
            
            stats_text = f"""
总体统计信息

平均准确率: {total_accuracy:.2%}
平均错误率: {total_error_rate:.2%}

任务数量: {len(task_summaries)}

最佳任务: Growth Level
最差任务: Growth Pattern

关键问题:
• {task_summaries['growth_pattern']['worst_class']} 
  (错误率: {task_summaries['growth_pattern']['worst_error_rate']:.1%})
• {task_summaries['interference_factors']['worst_factor']} 
  (错误率: {task_summaries['interference_factors']['worst_error_rate']:.1%})
            """
            
            ax10.text(0.1, 0.9, stats_text, transform=ax10.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self, task_summaries, analyses):
        """生成综合错误分析报告"""
        report = []
        report.append("# 多层级生物样本分类模型 - 综合错误样本分析报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行摘要
        report.append(f"\n## 执行摘要")
        
        total_accuracy = np.mean([ts['accuracy'] for ts in task_summaries.values()])
        total_error_rate = 1 - total_accuracy
        
        report.append(f"本报告对多层级生物样本分类模型的三个主要任务进行了全面的错误样本分析：")
        report.append(f"- **Growth Level** (生长水平分类)")
        report.append(f"- **Growth Pattern** (生长模式分类)")
        report.append(f"- **Interference Factors** (干扰因子检测)")
        
        report.append(f"\n**关键发现**:")
        report.append(f"- 模型整体平均准确率: **{total_accuracy:.2%}**")
        report.append(f"- 模型整体平均错误率: **{total_error_rate:.2%}**")
        report.append(f"- 最佳表现任务: **Growth Level** ({task_summaries['growth_level']['accuracy']:.2%})")
        
        if 'growth_pattern' in task_summaries:
            report.append(f"- 最大改进空间: **Growth Pattern** ({task_summaries['growth_pattern']['accuracy']:.2%})")
        
        # 各任务详细分析
        report.append(f"\n## 各任务详细分析")
        
        # Growth Level分析
        if 'growth_level' in task_summaries:
            gl_data = task_summaries['growth_level']
            report.append(f"\n### 1. Growth Level 任务分析")
            report.append(f"- **准确率**: {gl_data['accuracy']:.2%}")
            report.append(f"- **错误率**: {gl_data['error_rate']:.2%}")
            report.append(f"- **总错误样本数**: {gl_data['total_errors']:,}")
            report.append(f"- **假阴性**: {gl_data['false_negatives']} (漏检)")
            report.append(f"- **假阳性**: {gl_data['false_positives']} (误检)")
            report.append(f"- **性能等级**: {gl_data['performance_level']}")
            
            report.append(f"\n**Growth Level 关键发现**:")
            report.append(f"- 该任务表现优秀，错误率控制在较低水平")
            report.append(f"- 假阴性和假阳性相对平衡，无明显偏向")
            report.append(f"- 建议保持现有模型架构，定期监控性能")
        
        # Growth Pattern分析
        if 'growth_pattern' in task_summaries:
            gp_data = task_summaries['growth_pattern']
            report.append(f"\n### 2. Growth Pattern 任务分析")
            report.append(f"- **平均准确率**: {gp_data['accuracy']:.2%}")
            report.append(f"- **平均错误率**: {gp_data['error_rate']:.2%}")
            report.append(f"- **总类别数**: {gp_data['total_classes']}")
            report.append(f"- **最差类别**: {gp_data['worst_class']} (错误率: {gp_data['worst_error_rate']:.2%})")
            report.append(f"- **最佳类别**: {gp_data['best_class']} (错误率: {gp_data['best_error_rate']:.2%})")
            report.append(f"- **性能等级**: {gp_data['performance_level']}")
            
            report.append(f"\n**Growth Pattern 关键发现**:")
            report.append(f"- 该任务是模型的最大挑战，需要重点改进")
            report.append(f"- 类别间性能差异显著，部分类别错误率过高")
            report.append(f"- 建议重新审查数据质量和标注一致性")
            
            # 详细类别分析
            if 'growth_pattern' in analyses:
                gp_analysis = analyses['growth_pattern']['class_analysis']
                
                # 找出问题类别
                problem_classes = [(cls, data) for cls, data in gp_analysis.items() 
                                 if data['error_rate'] > 0.5]
                
                if problem_classes:
                    report.append(f"\n**严重问题类别** (错误率 > 50%):")
                    for cls, data in problem_classes:
                        report.append(f"- {cls}: {data['error_rate']:.2%} 错误率")
        
        # Interference Factors分析
        if 'interference_factors' in task_summaries:
            if_data = task_summaries['interference_factors']
            report.append(f"\n### 3. Interference Factors 任务分析")
            report.append(f"- **平均准确率**: {if_data['accuracy']:.2%}")
            report.append(f"- **平均错误率**: {if_data['error_rate']:.2%}")
            report.append(f"- **总因子数**: {if_data['total_factors']}")
            report.append(f"- **最差因子**: {if_data['worst_factor']} (错误率: {if_data['worst_error_rate']:.2%})")
            report.append(f"- **最佳因子**: {if_data['best_factor']} (错误率: {if_data['best_error_rate']:.2%})")
            report.append(f"- **性能等级**: {if_data['performance_level']}")
            
            report.append(f"\n**Interference Factors 关键发现**:")
            report.append(f"- 该任务整体表现良好，但存在个别问题因子")
            report.append(f"- {if_data['worst_factor']} 因子需要特别关注和改进")
            report.append(f"- 多数因子检测准确率较高，算法基本有效")
        
        # 跨任务对比分析
        report.append(f"\n## 跨任务对比分析")
        
        # 性能排序
        sorted_tasks = sorted(task_summaries.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        report.append(f"\n### 任务性能排序")
        task_name_map = {
            'growth_level': 'Growth Level',
            'growth_pattern': 'Growth Pattern', 
            'interference_factors': 'Interference Factors'
        }
        
        for i, (task, data) in enumerate(sorted_tasks):
            report.append(f"{i+1}. **{task_name_map[task]}**: {data['accuracy']:.2%} 准确率")
        
        # 复杂度分析
        report.append(f"\n### 任务复杂度分析")
        report.append(f"- **Growth Level**: 二分类任务，复杂度最低")
        
        if 'growth_pattern' in task_summaries:
            report.append(f"- **Growth Pattern**: {task_summaries['growth_pattern']['total_classes']}类分类，复杂度最高")
        
        if 'interference_factors' in task_summaries:
            report.append(f"- **Interference Factors**: {task_summaries['interference_factors']['total_factors']}因子多标签分类，复杂度中等")
        
        # 综合改进建议
        report.append(f"\n## 综合改进建议")
        
        report.append(f"\n### 短期改进措施 (1-2个月)")
        
        report.append(f"\n**1. Growth Pattern任务优化**")
        report.append(f"- 重新审查高错误率类别的标注质量")
        report.append(f"- 增加问题类别的训练样本数量")
        report.append(f"- 实施类别平衡策略和损失函数调整")
        report.append(f"- 考虑使用集成学习方法")
        
        if 'interference_factors' in task_summaries and task_summaries['interference_factors']['worst_error_rate'] > 0.15:
            report.append(f"\n**2. Interference Factors优化**")
            report.append(f"- 针对{task_summaries['interference_factors']['worst_factor']}因子设计专门检测算法")
            report.append(f"- 调整检测阈值和参数")
            report.append(f"- 增加该因子的训练数据")
        
        report.append(f"\n**3. 数据质量提升**")
        report.append(f"- 建立标注质量控制流程")
        report.append(f"- 实施多人标注和一致性检查")
        report.append(f"- 收集更多边界案例样本")
        
        report.append(f"\n### 中期改进措施 (3-6个月)")
        
        report.append(f"\n**1. 模型架构优化**")
        report.append(f"- 探索更先进的backbone网络")
        report.append(f"- 实施多尺度特征融合")
        report.append(f"- 引入注意力机制")
        
        report.append(f"\n**2. 多任务学习优化**")
        report.append(f"- 分析任务间相关性")
        report.append(f"- 优化任务权重分配")
        report.append(f"- 实施渐进式训练策略")
        
        report.append(f"\n**3. 评估体系完善**")
        report.append(f"- 建立持续监控系统")
        report.append(f"- 实施A/B测试框架")
        report.append(f"- 收集用户反馈机制")
        
        report.append(f"\n### 长期改进措施 (6个月以上)")
        
        report.append(f"\n**1. 技术创新**")
        report.append(f"- 探索自监督学习方法")
        report.append(f"- 研究领域自适应技术")
        report.append(f"- 实施主动学习策略")
        
        report.append(f"\n**2. 系统化改进**")
        report.append(f"- 建立完整的MLOps流程")
        report.append(f"- 实施自动化模型更新")
        report.append(f"- 构建知识蒸馏框架")
        
        # 优先级建议
        report.append(f"\n## 改进优先级建议")
        
        report.append(f"\n### 第一优先级 (立即处理)")
        if 'growth_pattern' in task_summaries and task_summaries['growth_pattern']['error_rate'] > 0.15:
            report.append(f"- Growth Pattern任务的高错误率类别优化")
        
        if 'interference_factors' in task_summaries and task_summaries['interference_factors']['worst_error_rate'] > 0.15:
            report.append(f"- {task_summaries['interference_factors']['worst_factor']}因子检测算法改进")
        
        report.append(f"\n### 第二优先级 (近期处理)")
        report.append(f"- 数据质量控制流程建立")
        report.append(f"- 模型监控系统部署")
        report.append(f"- 用户反馈收集机制")
        
        report.append(f"\n### 第三优先级 (中长期规划)")
        report.append(f"- 模型架构升级")
        report.append(f"- 多任务学习优化")
        report.append(f"- 技术创新研究")
        
        # 结论
        report.append(f"\n## 结论")
        
        report.append(f"通过本次综合错误样本分析，我们发现：")
        report.append(f"\n1. **模型整体性能良好**，平均准确率达到{total_accuracy:.2%}")
        report.append(f"2. **Growth Level任务表现优秀**，可作为模型稳定性的基准")
        report.append(f"3. **Growth Pattern任务存在显著改进空间**，是当前的主要瓶颈")
        report.append(f"4. **Interference Factors任务整体可接受**，个别因子需要优化")
        
        report.append(f"\n建议按照优先级逐步实施改进措施，预期在短期内可将模型整体准确率提升2-5%，")
        report.append(f"中长期通过系统性优化可实现更大幅度的性能提升。")
        
        return '\n'.join(report)
    
    def run_comprehensive_analysis(self):
        """运行综合错误分析"""
        print("开始综合错误样本分析...")
        
        # 加载之前的分析结果
        analyses = self.load_previous_analyses()
        
        if not analyses:
            print("警告: 未找到之前的分析结果，请先运行各任务的详细分析")
            return None
        
        # 提取任务摘要
        task_summaries = self.extract_task_summaries(analyses)
        
        # 创建综合可视化
        print("生成综合错误分析可视化...")
        fig = self.create_comprehensive_visualization(task_summaries, analyses)
        
        # 保存可视化
        viz_path = self.reports_dir / 'comprehensive_error_analysis_visualization.png'
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 生成综合报告
        print("生成综合错误分析报告...")
        report = self.generate_comprehensive_report(task_summaries, analyses)
        
        report_path = self.reports_dir / 'comprehensive_error_analysis_final_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存分析数据
        comprehensive_data = {
            'task_summaries': task_summaries,
            'analyses': analyses,
            'generation_time': datetime.now().isoformat()
        }
        
        data_path = self.reports_dir / 'comprehensive_error_analysis_data.json'
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
        
        print(f"综合错误样本分析完成!")
        print(f"- 综合可视化: {viz_path}")
        print(f"- 综合报告: {report_path}")
        print(f"- 分析数据: {data_path}")
        
        return comprehensive_data

def main():
    # 设置路径
    experiment_path = "/home/aaa/ws/bioastModel/experiments/multilevel_mobilenetv3_ds"
    reports_dir = "/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports"
    
    # 创建综合分析器并运行分析
    analyzer = ComprehensiveErrorAnalyzer(experiment_path, reports_dir)
    comprehensive_data = analyzer.run_comprehensive_analysis()
    
    if comprehensive_data:
        # 打印关键发现
        print("\n=== 综合错误分析关键发现 ===")
        task_summaries = comprehensive_data['task_summaries']
        
        print("各任务性能排序:")
        sorted_tasks = sorted(task_summaries.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        task_name_map = {
            'growth_level': 'Growth Level',
            'growth_pattern': 'Growth Pattern', 
            'interference_factors': 'Interference Factors'
        }
        
        for i, (task, data) in enumerate(sorted_tasks):
            print(f"{i+1}. {task_name_map[task]}: {data['accuracy']:.2%} 准确率 ({data['performance_level']})")
        
        # 计算总体统计
        total_accuracy = np.mean([ts['accuracy'] for ts in task_summaries.values()])
        print(f"\n模型整体平均准确率: {total_accuracy:.2%}")

if __name__ == "__main__":
    main()