#!/usr/bin/env python3
"""
最终错误样本分析总结报告生成器
整合所有错误分析结果，生成完整的分析总结和建议
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

class FinalErrorAnalysisSummary:
    def __init__(self, reports_dir):
        self.reports_dir = Path(reports_dir)
        self.all_data = {}
        self.load_all_analysis_data()
        
    def load_all_analysis_data(self):
        """加载所有分析数据"""
        data_files = {
            'error_analysis': 'error_analysis_data.json',
            'growth_pattern': 'growth_pattern_analysis_data.json',
            'interference_factors': 'interference_factors_analysis_data.json',
            'comprehensive': 'comprehensive_error_analysis_data.json',
            'improvement_plan': 'targeted_improvement_plan_data.json'
        }
        
        for key, filename in data_files.items():
            file_path = self.reports_dir / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.all_data[key] = json.load(f)
                print(f"已加载: {filename}")
            else:
                print(f"警告: 未找到 {filename}")
    
    def generate_executive_summary(self):
        """生成执行摘要"""
        summary = []
        
        if 'comprehensive' in self.all_data:
            comp_data = self.all_data['comprehensive']
            task_summaries = comp_data['task_summaries']
            
            # 计算整体性能
            accuracies = [task_data['accuracy'] for task_data in task_summaries.values()]
            overall_accuracy = sum(accuracies) / len(accuracies)
            overall_error_rate = 1 - overall_accuracy
            
            summary.append("## 执行摘要")
            summary.append(f"\n本次错误样本分析针对多层级生物样本分类模型进行了全面的性能评估和问题诊断。")
            summary.append(f"分析涵盖了三个主要任务：Growth Level（生长水平）、Growth Pattern（生长模式）和Interference Factors（干扰因子）。")
            
            summary.append(f"\n### 整体性能表现")
            summary.append(f"- **模型整体准确率**: {overall_accuracy:.2%}")
            summary.append(f"- **模型整体错误率**: {overall_error_rate:.2%}")
            
            # 各任务性能
            summary.append(f"\n### 各任务性能概览")
            
            task_names = {
                'growth_level': 'Growth Level (生长水平)',
                'growth_pattern': 'Growth Pattern (生长模式)',
                'interference_factors': 'Interference Factors (干扰因子)'
            }
            
            for task_key, task_name in task_names.items():
                if task_key in task_summaries:
                    task_data = task_summaries[task_key]
                    accuracy = 1 - task_data['error_rate']
                    
                    if accuracy >= 0.95:
                        performance_level = "优秀"
                    elif accuracy >= 0.85:
                        performance_level = "良好"
                    elif accuracy >= 0.70:
                        performance_level = "一般"
                    else:
                        performance_level = "需要改进"
                    
                    summary.append(f"- **{task_name}**: {accuracy:.2%} ({performance_level})")
        
        return '\n'.join(summary)
    
    def generate_key_findings(self):
        """生成关键发现"""
        findings = []
        findings.append("## 关键发现")
        
        # Growth Level 发现
        if 'error_analysis' in self.all_data:
            gl_data = self.all_data['error_analysis']['growth_level']
            findings.append(f"\n### Growth Level 任务")
            findings.append(f"- 整体表现优秀，错误率仅为 {gl_data['error_rate']:.2%}")
            findings.append(f"- 假阴性率: {gl_data['fn_rate']:.2%}")
            findings.append(f"- 假阳性率: {gl_data['fp_rate']:.2%}")
            findings.append(f"- 主要问题: 模型在边界样本的判断上仍有改进空间")
        
        # Growth Pattern 发现
        if 'growth_pattern' in self.all_data:
            gp_data = self.all_data['growth_pattern']
            findings.append(f"\n### Growth Pattern 任务")
            findings.append(f"- 整体错误率较高，为主要改进目标")
            
            # 找出最严重的问题类别
            class_analysis = gp_data['class_analysis']
            critical_classes = []
            for class_name, class_data in class_analysis.items():
                if class_data['error_rate'] > 0.5:
                    critical_classes.append(f"{class_name} ({class_data['error_rate']:.1%})")
            
            if critical_classes:
                findings.append(f"- 严重问题类别: {', '.join(critical_classes)}")
            
            # 混淆对分析
            if 'confusion_pairs' in gp_data:
                top_confusion = gp_data['confusion_pairs'][0]
                findings.append(f"- 最主要混淆: {top_confusion['class_a']} ↔ {top_confusion['class_b']} ({top_confusion['mutual_confusion']}次)")
        
        # Interference Factors 发现
        if 'interference_factors' in self.all_data:
            if_data = self.all_data['interference_factors']
            findings.append(f"\n### Interference Factors 任务")
            findings.append(f"- 整体性能良好，但个别因子需要重点关注")
            
            # 找出问题因子
            factor_analysis = if_data['factor_analysis']
            problem_factors = []
            for factor_name, factor_data in factor_analysis.items():
                if factor_data['error_rate'] > 0.1:
                    problem_factors.append(f"{factor_name} ({factor_data['error_rate']:.1%})")
            
            if problem_factors:
                findings.append(f"- 问题因子: {', '.join(problem_factors)}")
        
        return '\n'.join(findings)
    
    def generate_improvement_priorities(self):
        """生成改进优先级"""
        priorities = []
        priorities.append("## 改进优先级")
        
        if 'improvement_plan' in self.all_data:
            plan_data = self.all_data['improvement_plan']
            priority_issues = plan_data['priority_issues']
            
            priority_names = {
                'critical': '🔴 严重问题 (立即处理)',
                'high': '🟠 高优先级 (2周内)',
                'medium': '🟡 中等优先级 (1个月内)',
                'low': '🟢 低优先级 (3个月内)'
            }
            
            for priority, issues in priority_issues.items():
                if issues:
                    priorities.append(f"\n### {priority_names[priority]}")
                    priorities.append(f"问题数量: **{len(issues)}**")
                    
                    for issue in issues[:5]:  # 显示前5个最重要的
                        priorities.append(f"- {issue['task']} - {issue['item']}: {issue['error_rate']:.1%}")
                    
                    if len(issues) > 5:
                        priorities.append(f"- ... 还有 {len(issues) - 5} 个问题")
        
        return '\n'.join(priorities)
    
    def generate_resource_requirements(self):
        """生成资源需求总结"""
        resources = []
        resources.append("## 资源需求总结")
        
        if 'improvement_plan' in self.all_data:
            plan_data = self.all_data['improvement_plan']
            
            # 统计所有解决方案的资源需求
            all_resources = []
            for priority_solutions in plan_data['solutions'].values():
                for solution in priority_solutions:
                    all_resources.extend(solution['resources_needed'])
            
            # 分类资源
            human_resources = set()
            compute_resources = set()
            data_resources = set()
            time_resources = set()
            
            for resource in all_resources:
                if '人' in resource or '专家' in resource or '工程师' in resource:
                    human_resources.add(resource)
                elif 'GPU' in resource or '计算' in resource:
                    compute_resources.add(resource)
                elif '标注' in resource or '数据' in resource or '样本' in resource:
                    data_resources.add(resource)
                elif '周' in resource or '月' in resource:
                    time_resources.add(resource)
            
            if human_resources:
                resources.append(f"\n### 人力资源")
                for resource in sorted(human_resources):
                    resources.append(f"- {resource}")
            
            if compute_resources:
                resources.append(f"\n### 计算资源")
                for resource in sorted(compute_resources):
                    resources.append(f"- {resource}")
            
            if data_resources:
                resources.append(f"\n### 数据资源")
                for resource in sorted(data_resources):
                    resources.append(f"- {resource}")
        
        return '\n'.join(resources)
    
    def generate_success_metrics(self):
        """生成成功指标"""
        metrics = []
        metrics.append("## 成功指标与里程碑")
        
        metrics.append(f"\n### 短期目标 (2个月内)")
        metrics.append(f"- 解决所有严重问题（错误率 > 50%）")
        metrics.append(f"- Growth Pattern 任务准确率提升至 > 70%")
        metrics.append(f"- Interference Factors 中 pores 因子准确率 > 90%")
        metrics.append(f"- 模型整体准确率提升至 > 85%")
        
        metrics.append(f"\n### 中期目标 (6个月内)")
        metrics.append(f"- 模型整体准确率 > 90%")
        metrics.append(f"- 所有任务准确率 > 85%")
        metrics.append(f"- Growth Pattern 任务准确率 > 80%")
        metrics.append(f"- 建立完整的性能监控体系")
        
        metrics.append(f"\n### 长期目标 (1年内)")
        metrics.append(f"- 模型整体准确率 > 95%")
        metrics.append(f"- 所有任务准确率 > 90%")
        metrics.append(f"- 建立持续改进和自动化监控机制")
        metrics.append(f"- 用户满意度 > 95%")
        
        return '\n'.join(metrics)
    
    def generate_recommendations(self):
        """生成建议和下一步行动"""
        recommendations = []
        recommendations.append("## 建议与下一步行动")
        
        recommendations.append(f"\n### 立即行动项")
        recommendations.append(f"1. **组建专项改进团队**: 包括领域专家、算法工程师、数据标注人员")
        recommendations.append(f"2. **启动严重问题修复**: 重点关注 Growth Pattern 中的 irregular、scattered 等类别")
        recommendations.append(f"3. **数据质量审查**: 重新审查和标注问题类别的训练数据")
        recommendations.append(f"4. **建立监控机制**: 实时跟踪改进进展和模型性能变化")
        
        recommendations.append(f"\n### 技术改进建议")
        recommendations.append(f"1. **数据层面**:")
        recommendations.append(f"   - 增加问题类别的高质量标注样本")
        recommendations.append(f"   - 实施数据增强策略提高样本多样性")
        recommendations.append(f"   - 建立标注质量控制流程")
        
        recommendations.append(f"2. **算法层面**:")
        recommendations.append(f"   - 优化损失函数，增加困难样本权重")
        recommendations.append(f"   - 实施集成学习和主动学习策略")
        recommendations.append(f"   - 优化特征提取和决策边界")
        
        recommendations.append(f"3. **系统层面**:")
        recommendations.append(f"   - 建立A/B测试框架验证改进效果")
        recommendations.append(f"   - 实施渐进式部署策略")
        recommendations.append(f"   - 建立用户反馈收集机制")
        
        recommendations.append(f"\n### 风险控制")
        recommendations.append(f"1. **设置改进检查点**: 每2周评估一次改进进展")
        recommendations.append(f"2. **准备回滚方案**: 确保改进过程中系统稳定性")
        recommendations.append(f"3. **资源预留**: 为突发问题预留20%的额外资源")
        recommendations.append(f"4. **专家咨询**: 建立外部专家咨询机制")
        
        return '\n'.join(recommendations)
    
    def generate_appendix(self):
        """生成附录信息"""
        appendix = []
        appendix.append("## 附录")
        
        appendix.append(f"\n### 分析文件清单")
        appendix.append(f"本次分析生成的所有文件:")
        
        # 列出所有生成的文件
        report_files = [
            "error_sample_analysis_report.md - 初始错误样本分析报告",
            "growth_pattern_detailed_report.md - Growth Pattern详细分析报告",
            "interference_factors_detailed_report.md - Interference Factors详细分析报告",
            "comprehensive_error_analysis_final_report.md - 综合错误分析报告",
            "targeted_improvement_plan.md - 针对性改进计划",
            "final_error_analysis_summary.md - 最终分析总结报告"
        ]
        
        for file_desc in report_files:
            appendix.append(f"- {file_desc}")
        
        appendix.append(f"\n### 数据文件清单")
        data_files = [
            "error_analysis_data.json - 基础错误分析数据",
            "growth_pattern_analysis_data.json - Growth Pattern分析数据",
            "interference_factors_analysis_data.json - Interference Factors分析数据",
            "comprehensive_error_analysis_data.json - 综合分析数据",
            "targeted_improvement_plan_data.json - 改进计划数据"
        ]
        
        for file_desc in data_files:
            appendix.append(f"- {file_desc}")
        
        appendix.append(f"\n### 可视化文件清单")
        viz_files = [
            "error_analysis_visualization.png - 基础错误分析可视化",
            "growth_pattern_detailed_analysis.png - Growth Pattern详细分析图",
            "interference_factors_detailed_analysis.png - Interference Factors分析图",
            "comprehensive_error_analysis_visualization.png - 综合分析可视化"
        ]
        
        for file_desc in viz_files:
            appendix.append(f"- {file_desc}")
        
        return '\n'.join(appendix)
    
    def generate_final_summary_report(self):
        """生成最终总结报告"""
        report = []
        
        # 标题和基本信息
        report.append("# 多层级生物样本分类模型 - 错误样本分析总结报告")
        report.append(f"\n**分析完成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**分析范围**: Growth Level, Growth Pattern, Interference Factors")
        report.append(f"**报告类型**: 综合错误样本分析与改进建议")
        
        # 各部分内容
        report.append(f"\n{self.generate_executive_summary()}")
        report.append(f"\n{self.generate_key_findings()}")
        report.append(f"\n{self.generate_improvement_priorities()}")
        report.append(f"\n{self.generate_resource_requirements()}")
        report.append(f"\n{self.generate_success_metrics()}")
        report.append(f"\n{self.generate_recommendations()}")
        report.append(f"\n{self.generate_appendix()}")
        
        # 结论
        report.append(f"\n## 结论")
        report.append(f"\n本次错误样本分析全面评估了多层级生物样本分类模型的性能，")
        report.append(f"识别了关键问题并制定了详细的改进计划。通过系统性的改进措施，")
        report.append(f"预期能够显著提升模型的整体性能，特别是在Growth Pattern任务上的表现。")
        
        report.append(f"\n建议立即启动改进计划的执行，优先解决严重问题，")
        report.append(f"并建立持续监控和改进机制，确保模型性能的长期稳定和提升。")
        
        return '\n'.join(report)
    
    def run_final_summary(self):
        """运行最终总结生成"""
        print("开始生成最终错误样本分析总结报告...")
        
        if not self.all_data:
            print("错误: 未找到分析数据，请先运行相关分析脚本")
            return None
        
        # 生成最终报告
        print("生成最终总结报告...")
        report = self.generate_final_summary_report()
        
        # 保存报告
        report_path = self.reports_dir / 'final_error_analysis_summary.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 生成总结数据
        summary_data = {
            'generation_time': datetime.now().isoformat(),
            'data_sources': list(self.all_data.keys()),
            'total_files_analyzed': len(self.all_data),
            'report_sections': [
                'executive_summary',
                'key_findings', 
                'improvement_priorities',
                'resource_requirements',
                'success_metrics',
                'recommendations',
                'appendix'
            ]
        }
        
        # 如果有综合数据，添加关键统计
        if 'comprehensive' in self.all_data:
            comp_data = self.all_data['comprehensive']
            task_summaries = comp_data['task_summaries']
            
            # 计算整体统计
            accuracies = [task_data['accuracy'] for task_data in task_summaries.values()]
            overall_accuracy = sum(accuracies) / len(accuracies)
            overall_error_rate = 1 - overall_accuracy
            
            summary_data['key_statistics'] = {
                'overall_accuracy': overall_accuracy,
                'overall_error_rate': overall_error_rate,
                'task_count': len(task_summaries)
            }
        
        # 如果有改进计划，添加问题统计
        if 'improvement_plan' in self.all_data:
            plan_data = self.all_data['improvement_plan']
            priority_issues = plan_data['priority_issues']
            summary_data['improvement_statistics'] = {
                'total_issues': sum(len(issues) for issues in priority_issues.values()),
                'critical_issues': len(priority_issues.get('critical', [])),
                'high_priority_issues': len(priority_issues.get('high', [])),
                'medium_priority_issues': len(priority_issues.get('medium', [])),
                'low_priority_issues': len(priority_issues.get('low', []))
            }
        
        # 保存总结数据
        data_path = self.reports_dir / 'final_error_analysis_summary_data.json'
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"最终错误样本分析总结报告生成完成!")
        print(f"- 总结报告: {report_path}")
        print(f"- 总结数据: {data_path}")
        
        return summary_data

def main():
    # 设置报告目录
    reports_dir = "/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports"
    
    # 创建最终总结生成器并运行
    summarizer = FinalErrorAnalysisSummary(reports_dir)
    summary_data = summarizer.run_final_summary()
    
    if summary_data:
        # 打印关键统计
        print("\n=== 最终分析总结统计 ===")
        print(f"数据源数量: {summary_data['total_files_analyzed']}")
        print(f"报告章节数: {len(summary_data['report_sections'])}")
        
        if 'key_statistics' in summary_data:
            stats = summary_data['key_statistics']
            print(f"模型整体准确率: {stats['overall_accuracy']:.2%}")
            print(f"模型整体错误率: {stats['overall_error_rate']:.2%}")
        
        if 'improvement_statistics' in summary_data:
            imp_stats = summary_data['improvement_statistics']
            print(f"识别问题总数: {imp_stats['total_issues']}")
            print(f"严重问题数: {imp_stats['critical_issues']}")
            print(f"高优先级问题数: {imp_stats['high_priority_issues']}")

if __name__ == "__main__":
    main()