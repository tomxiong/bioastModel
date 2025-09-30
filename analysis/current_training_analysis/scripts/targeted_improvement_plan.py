#!/usr/bin/env python3
"""
针对性改进计划生成器
基于综合错误分析结果，生成具体的、可执行的改进计划和解决方案
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

class TargetedImprovementPlanner:
    def __init__(self, reports_dir):
        self.reports_dir = Path(reports_dir)
        self.comprehensive_data = None
        self.load_comprehensive_data()
        
    def load_comprehensive_data(self):
        """加载综合分析数据"""
        data_path = self.reports_dir / 'comprehensive_error_analysis_data.json'
        if data_path.exists():
            with open(data_path, 'r') as f:
                self.comprehensive_data = json.load(f)
        else:
            print("警告: 未找到综合分析数据")
    
    def analyze_priority_issues(self):
        """分析优先级问题"""
        if not self.comprehensive_data:
            return {}
        
        task_summaries = self.comprehensive_data['task_summaries']
        analyses = self.comprehensive_data['analyses']
        
        priority_issues = {
            'critical': [],    # 错误率 > 50%
            'high': [],        # 错误率 20-50%
            'medium': [],      # 错误率 10-20%
            'low': []          # 错误率 < 10%
        }
        
        # Growth Pattern问题分析
        if 'growth_pattern' in analyses:
            gp_analysis = analyses['growth_pattern']['class_analysis']
            for class_name, class_data in gp_analysis.items():
                error_rate = class_data['error_rate']
                issue = {
                    'task': 'Growth Pattern',
                    'item': class_name,
                    'error_rate': error_rate,
                    'sample_count': class_data.get('sample_count', 0),
                    'type': 'classification_class'
                }
                
                if error_rate > 0.5:
                    priority_issues['critical'].append(issue)
                elif error_rate > 0.2:
                    priority_issues['high'].append(issue)
                elif error_rate > 0.1:
                    priority_issues['medium'].append(issue)
                else:
                    priority_issues['low'].append(issue)
        
        # Interference Factors问题分析
        if 'interference_factors' in analyses:
            if_analysis = analyses['interference_factors']['factor_analysis']
            for factor_name, factor_data in if_analysis.items():
                error_rate = factor_data['error_rate']
                issue = {
                    'task': 'Interference Factors',
                    'item': factor_name,
                    'error_rate': error_rate,
                    'type': 'detection_factor'
                }
                
                if error_rate > 0.5:
                    priority_issues['critical'].append(issue)
                elif error_rate > 0.2:
                    priority_issues['high'].append(issue)
                elif error_rate > 0.1:
                    priority_issues['medium'].append(issue)
                else:
                    priority_issues['low'].append(issue)
        
        # Growth Level问题分析
        if 'growth_level' in task_summaries:
            gl_data = task_summaries['growth_level']
            if gl_data['error_rate'] > 0.1:
                issue = {
                    'task': 'Growth Level',
                    'item': 'binary_classification',
                    'error_rate': gl_data['error_rate'],
                    'false_negatives': gl_data['false_negatives'],
                    'false_positives': gl_data['false_positives'],
                    'type': 'binary_classification'
                }
                priority_issues['medium'].append(issue)
        
        return priority_issues
    
    def generate_specific_solutions(self, priority_issues):
        """为每个优先级问题生成具体解决方案"""
        solutions = {}
        
        for priority, issues in priority_issues.items():
            solutions[priority] = []
            
            for issue in issues:
                solution = self.create_solution_for_issue(issue, priority)
                solutions[priority].append(solution)
        
        return solutions
    
    def create_solution_for_issue(self, issue, priority):
        """为特定问题创建解决方案"""
        task = issue['task']
        item = issue['item']
        error_rate = issue['error_rate']
        issue_type = issue['type']
        
        solution = {
            'issue': issue,
            'priority': priority,
            'solutions': [],
            'timeline': self.get_timeline_for_priority(priority),
            'resources_needed': [],
            'success_metrics': []
        }
        
        # 根据任务类型和优先级生成具体解决方案
        if task == 'Growth Pattern':
            solution.update(self.get_growth_pattern_solutions(issue, priority))
        elif task == 'Interference Factors':
            solution.update(self.get_interference_factors_solutions(issue, priority))
        elif task == 'Growth Level':
            solution.update(self.get_growth_level_solutions(issue, priority))
        
        return solution
    
    def get_growth_pattern_solutions(self, issue, priority):
        """获取Growth Pattern任务的解决方案"""
        class_name = issue['item']
        error_rate = issue['error_rate']
        
        solutions = []
        resources = []
        metrics = []
        
        if priority == 'critical':
            solutions.extend([
                f"立即重新审查 {class_name} 类别的所有标注数据",
                f"组织专家团队重新定义 {class_name} 类别的标注标准",
                f"收集更多 {class_name} 类别的高质量样本（目标：增加50%样本量）",
                f"设计专门针对 {class_name} 类别的数据增强策略",
                f"考虑将 {class_name} 合并到相似类别或重新分类",
                f"实施主动学习，重点标注 {class_name} 的边界案例"
            ])
            resources.extend([
                "领域专家 2-3人，2周时间",
                "数据标注团队 3-5人，1个月时间", 
                "GPU计算资源用于重新训练",
                "数据收集预算（如需要）"
            ])
            metrics.extend([
                f"{class_name} 类别错误率降低到 < 30%",
                f"{class_name} 类别F1-score > 0.7",
                "整体Growth Pattern任务准确率提升 > 10%"
            ])
            
        elif priority == 'high':
            solutions.extend([
                f"增加 {class_name} 类别的训练样本数量（目标：增加30%）",
                f"优化 {class_name} 类别的损失函数权重",
                f"实施针对 {class_name} 的困难样本挖掘",
                f"使用集成学习方法提升 {class_name} 的识别能力",
                f"分析 {class_name} 与其他类别的混淆模式，优化决策边界"
            ])
            resources.extend([
                "数据标注人员 2人，2周时间",
                "算法工程师 1人，1个月时间",
                "计算资源用于模型训练和验证"
            ])
            metrics.extend([
                f"{class_name} 类别错误率降低到 < 20%",
                f"{class_name} 类别精确率和召回率均 > 0.8"
            ])
            
        elif priority == 'medium':
            solutions.extend([
                f"调整 {class_name} 类别的分类阈值",
                f"增加 {class_name} 类别的数据增强多样性",
                f"优化特征提取器对 {class_name} 特征的敏感性",
                f"实施渐进式学习，逐步提升 {class_name} 的识别能力"
            ])
            resources.extend([
                "算法工程师 1人，2周时间",
                "少量计算资源用于参数调优"
            ])
            metrics.extend([
                f"{class_name} 类别错误率降低到 < 15%"
            ])
        
        return {
            'solutions': solutions,
            'resources_needed': resources,
            'success_metrics': metrics
        }
    
    def get_interference_factors_solutions(self, issue, priority):
        """获取Interference Factors任务的解决方案"""
        factor_name = issue['item']
        error_rate = issue['error_rate']
        
        solutions = []
        resources = []
        metrics = []
        
        if priority == 'critical' or priority == 'high':
            solutions.extend([
                f"重新设计 {factor_name} 因子的检测算法",
                f"分析 {factor_name} 因子的视觉特征，优化特征提取",
                f"收集更多包含 {factor_name} 因子的标注样本",
                f"调整 {factor_name} 因子的检测阈值和参数",
                f"实施多尺度检测策略针对 {factor_name} 因子",
                f"使用注意力机制增强对 {factor_name} 因子的关注"
            ])
            resources.extend([
                "计算机视觉专家 1人，3周时间",
                "数据标注团队，标注500-1000个样本",
                "GPU计算资源用于算法开发和测试"
            ])
            metrics.extend([
                f"{factor_name} 因子检测准确率 > 90%",
                f"{factor_name} 因子假阳性率 < 5%",
                f"{factor_name} 因子假阴性率 < 10%"
            ])
            
        elif priority == 'medium':
            solutions.extend([
                f"微调 {factor_name} 因子的检测参数",
                f"增加 {factor_name} 因子的训练数据多样性",
                f"优化 {factor_name} 因子的后处理逻辑"
            ])
            resources.extend([
                "算法工程师 1人，1周时间",
                "少量标注数据补充"
            ])
            metrics.extend([
                f"{factor_name} 因子检测准确率 > 85%"
            ])
        
        return {
            'solutions': solutions,
            'resources_needed': resources,
            'success_metrics': metrics
        }
    
    def get_growth_level_solutions(self, issue, priority):
        """获取Growth Level任务的解决方案"""
        solutions = []
        resources = []
        metrics = []
        
        fn_count = issue.get('false_negatives', 0)
        fp_count = issue.get('false_positives', 0)
        
        if fn_count > fp_count:
            # 假阴性较多，模型过于保守
            solutions.extend([
                "降低正类分类阈值，提高敏感性",
                "增加正类样本的训练权重",
                "分析假阴性样本的特征，优化特征提取",
                "实施困难负样本挖掘，提升模型判别能力"
            ])
        else:
            # 假阳性较多，模型过于激进
            solutions.extend([
                "提高正类分类阈值，增加特异性",
                "增加负类样本的训练权重",
                "分析假阳性样本的特征，优化决策边界",
                "实施困难正样本挖掘，提升模型精确性"
            ])
        
        resources.extend([
            "算法工程师 1人，1周时间",
            "少量计算资源用于参数调优"
        ])
        
        metrics.extend([
            "Growth Level任务准确率 > 98%",
            "假阴性率和假阳性率均 < 2%"
        ])
        
        return {
            'solutions': solutions,
            'resources_needed': resources,
            'success_metrics': metrics
        }
    
    def get_timeline_for_priority(self, priority):
        """根据优先级获取时间线"""
        base_date = datetime.now()
        
        if priority == 'critical':
            return {
                'start_immediately': True,
                'target_completion': (base_date + timedelta(weeks=2)).strftime('%Y-%m-%d'),
                'review_date': (base_date + timedelta(weeks=1)).strftime('%Y-%m-%d'),
                'urgency': 'immediate'
            }
        elif priority == 'high':
            return {
                'start_date': (base_date + timedelta(days=3)).strftime('%Y-%m-%d'),
                'target_completion': (base_date + timedelta(weeks=4)).strftime('%Y-%m-%d'),
                'review_date': (base_date + timedelta(weeks=2)).strftime('%Y-%m-%d'),
                'urgency': 'high'
            }
        elif priority == 'medium':
            return {
                'start_date': (base_date + timedelta(weeks=1)).strftime('%Y-%m-%d'),
                'target_completion': (base_date + timedelta(weeks=8)).strftime('%Y-%m-%d'),
                'review_date': (base_date + timedelta(weeks=4)).strftime('%Y-%m-%d'),
                'urgency': 'medium'
            }
        else:
            return {
                'start_date': (base_date + timedelta(weeks=4)).strftime('%Y-%m-%d'),
                'target_completion': (base_date + timedelta(weeks=12)).strftime('%Y-%m-%d'),
                'review_date': (base_date + timedelta(weeks=8)).strftime('%Y-%m-%d'),
                'urgency': 'low'
            }
    
    def generate_implementation_roadmap(self, solutions):
        """生成实施路线图"""
        roadmap = {
            'phase_1_immediate': [],  # 0-2周
            'phase_2_short_term': [], # 2-8周
            'phase_3_medium_term': [], # 2-6个月
            'phase_4_long_term': []   # 6个月以上
        }
        
        # 按优先级和时间线分配到不同阶段
        for priority in ['critical', 'high', 'medium', 'low']:
            if priority in solutions:
                for solution in solutions[priority]:
                    timeline = solution['timeline']
                    
                    if priority == 'critical':
                        roadmap['phase_1_immediate'].append(solution)
                    elif priority == 'high':
                        roadmap['phase_2_short_term'].append(solution)
                    elif priority == 'medium':
                        roadmap['phase_3_medium_term'].append(solution)
                    else:
                        roadmap['phase_4_long_term'].append(solution)
        
        return roadmap
    
    def generate_improvement_report(self, priority_issues, solutions, roadmap):
        """生成改进计划报告"""
        report = []
        report.append("# 多层级生物样本分类模型 - 针对性改进计划")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 执行摘要
        report.append(f"\n## 执行摘要")
        
        total_issues = sum(len(issues) for issues in priority_issues.values())
        critical_count = len(priority_issues.get('critical', []))
        high_count = len(priority_issues.get('high', []))
        
        report.append(f"基于综合错误样本分析，识别出 **{total_issues}** 个需要改进的问题项，")
        report.append(f"其中 **{critical_count}** 个严重问题需要立即处理，**{high_count}** 个高优先级问题需要近期解决。")
        
        report.append(f"\n**改进目标**:")
        report.append(f"- 短期目标（2个月内）：解决所有严重和高优先级问题")
        report.append(f"- 中期目标（6个月内）：模型整体准确率提升至 > 90%")
        report.append(f"- 长期目标（1年内）：建立持续改进和监控体系")
        
        # 问题优先级分析
        report.append(f"\n## 问题优先级分析")
        
        priority_names = {
            'critical': '严重问题 (错误率 > 50%)',
            'high': '高优先级问题 (错误率 20-50%)',
            'medium': '中等优先级问题 (错误率 10-20%)',
            'low': '低优先级问题 (错误率 < 10%)'
        }
        
        for priority, issues in priority_issues.items():
            if issues:
                report.append(f"\n### {priority_names[priority]}")
                report.append(f"问题数量: **{len(issues)}**")
                
                for issue in issues:
                    report.append(f"\n**{issue['task']} - {issue['item']}**")
                    report.append(f"- 错误率: {issue['error_rate']:.2%}")
                    report.append(f"- 问题类型: {issue['type']}")
                    
                    if 'sample_count' in issue:
                        report.append(f"- 样本数量: {issue['sample_count']}")
        
        # 详细解决方案
        report.append(f"\n## 详细解决方案")
        
        for priority in ['critical', 'high', 'medium', 'low']:
            if priority in solutions and solutions[priority]:
                report.append(f"\n### {priority_names[priority]}解决方案")
                
                for i, solution in enumerate(solutions[priority]):
                    issue = solution['issue']
                    report.append(f"\n#### {i+1}. {issue['task']} - {issue['item']}")
                    
                    report.append(f"\n**具体措施:**")
                    for j, sol in enumerate(solution['solutions']):
                        report.append(f"{j+1}. {sol}")
                    
                    report.append(f"\n**所需资源:**")
                    for resource in solution['resources_needed']:
                        report.append(f"- {resource}")
                    
                    report.append(f"\n**成功指标:**")
                    for metric in solution['success_metrics']:
                        report.append(f"- {metric}")
                    
                    timeline = solution['timeline']
                    if 'start_immediately' in timeline:
                        report.append(f"\n**时间安排:** 立即开始，目标完成时间: {timeline['target_completion']}")
                    else:
                        report.append(f"\n**时间安排:** {timeline['start_date']} 开始，目标完成时间: {timeline['target_completion']}")
                    
                    report.append(f"**进度检查:** {timeline['review_date']}")
        
        # 实施路线图
        report.append(f"\n## 实施路线图")
        
        phase_names = {
            'phase_1_immediate': '第一阶段：立即行动 (0-2周)',
            'phase_2_short_term': '第二阶段：短期改进 (2-8周)',
            'phase_3_medium_term': '第三阶段：中期优化 (2-6个月)',
            'phase_4_long_term': '第四阶段：长期完善 (6个月以上)'
        }
        
        for phase, phase_solutions in roadmap.items():
            if phase_solutions:
                report.append(f"\n### {phase_names[phase]}")
                
                for solution in phase_solutions:
                    issue = solution['issue']
                    report.append(f"- **{issue['task']} - {issue['item']}** (错误率: {issue['error_rate']:.2%})")
                
                # 阶段总结
                if phase == 'phase_1_immediate':
                    report.append(f"\n**阶段目标:** 解决最严重的问题，防止性能进一步恶化")
                    report.append(f"**预期效果:** 模型整体准确率提升 5-10%")
                elif phase == 'phase_2_short_term':
                    report.append(f"\n**阶段目标:** 解决高优先级问题，显著提升模型性能")
                    report.append(f"**预期效果:** 模型整体准确率提升至 85-90%")
                elif phase == 'phase_3_medium_term':
                    report.append(f"\n**阶段目标:** 全面优化模型，达到生产环境要求")
                    report.append(f"**预期效果:** 模型整体准确率提升至 90-95%")
                else:
                    report.append(f"\n**阶段目标:** 建立持续改进机制，保持模型先进性")
                    report.append(f"**预期效果:** 模型性能持续稳定，适应新场景")
        
        # 资源需求汇总
        report.append(f"\n## 资源需求汇总")
        
        all_resources = []
        for priority_solutions in solutions.values():
            for solution in priority_solutions:
                all_resources.extend(solution['resources_needed'])
        
        # 统计资源类型
        human_resources = [r for r in all_resources if '人' in r]
        compute_resources = [r for r in all_resources if 'GPU' in r or '计算' in r]
        data_resources = [r for r in all_resources if '标注' in r or '数据' in r]
        
        report.append(f"\n### 人力资源需求")
        for resource in set(human_resources):
            report.append(f"- {resource}")
        
        report.append(f"\n### 计算资源需求")
        for resource in set(compute_resources):
            report.append(f"- {resource}")
        
        report.append(f"\n### 数据资源需求")
        for resource in set(data_resources):
            report.append(f"- {resource}")
        
        # 风险评估
        report.append(f"\n## 风险评估与缓解措施")
        
        report.append(f"\n### 主要风险")
        report.append(f"1. **数据质量风险**: 标注不一致可能影响改进效果")
        report.append(f"   - 缓解措施: 建立标注质量控制流程，多人交叉验证")
        
        report.append(f"2. **资源不足风险**: 人力或计算资源可能不够")
        report.append(f"   - 缓解措施: 分阶段实施，优先解决关键问题")
        
        report.append(f"3. **技术风险**: 某些问题可能需要更复杂的解决方案")
        report.append(f"   - 缓解措施: 预留技术调研时间，准备备选方案")
        
        report.append(f"4. **时间风险**: 改进周期可能超出预期")
        report.append(f"   - 缓解措施: 设置里程碑检查点，及时调整计划")
        
        # 成功标准
        report.append(f"\n## 成功标准")
        
        report.append(f"\n### 短期成功标准 (2个月)")
        report.append(f"- 解决所有严重问题（错误率 > 50%的项目）")
        report.append(f"- Growth Pattern任务准确率提升至 > 70%")
        report.append(f"- Interference Factors中pores因子准确率 > 90%")
        
        report.append(f"\n### 中期成功标准 (6个月)")
        report.append(f"- 模型整体准确率 > 90%")
        report.append(f"- 所有任务准确率 > 85%")
        report.append(f"- 建立完整的监控和评估体系")
        
        report.append(f"\n### 长期成功标准 (1年)")
        report.append(f"- 模型性能稳定，适应性强")
        report.append(f"- 持续改进机制有效运行")
        report.append(f"- 用户满意度 > 95%")
        
        return '\n'.join(report)
    
    def run_improvement_planning(self):
        """运行改进计划生成"""
        print("开始生成针对性改进计划...")
        
        if not self.comprehensive_data:
            print("错误: 未找到综合分析数据，请先运行综合错误分析")
            return None
        
        # 分析优先级问题
        print("分析优先级问题...")
        priority_issues = self.analyze_priority_issues()
        
        # 生成具体解决方案
        print("生成具体解决方案...")
        solutions = self.generate_specific_solutions(priority_issues)
        
        # 生成实施路线图
        print("生成实施路线图...")
        roadmap = self.generate_implementation_roadmap(solutions)
        
        # 生成改进报告
        print("生成改进计划报告...")
        report = self.generate_improvement_report(priority_issues, solutions, roadmap)
        
        # 保存报告
        report_path = self.reports_dir / 'targeted_improvement_plan.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存计划数据
        plan_data = {
            'priority_issues': priority_issues,
            'solutions': solutions,
            'roadmap': roadmap,
            'generation_time': datetime.now().isoformat()
        }
        
        data_path = self.reports_dir / 'targeted_improvement_plan_data.json'
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)
        
        print(f"针对性改进计划生成完成!")
        print(f"- 改进计划报告: {report_path}")
        print(f"- 计划数据: {data_path}")
        
        return plan_data

def main():
    # 设置报告目录
    reports_dir = "/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports"
    
    # 创建改进计划生成器并运行
    planner = TargetedImprovementPlanner(reports_dir)
    plan_data = planner.run_improvement_planning()
    
    if plan_data:
        # 打印关键统计
        print("\n=== 改进计划关键统计 ===")
        priority_issues = plan_data['priority_issues']
        
        total_issues = sum(len(issues) for issues in priority_issues.values())
        print(f"总问题数: {total_issues}")
        
        for priority, issues in priority_issues.items():
            if issues:
                print(f"{priority}优先级: {len(issues)}个问题")
                for issue in issues[:3]:  # 显示前3个
                    print(f"  - {issue['task']} - {issue['item']}: {issue['error_rate']:.2%}")

if __name__ == "__main__":
    main()