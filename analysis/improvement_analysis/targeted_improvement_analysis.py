#!/usr/bin/env python3
"""
针对性改进分析脚本
基于真实训练问题和性能分析，提出具体的改进方向和方法
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_training_issues(history_path):
    """分析训练中存在的具体问题"""
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    issues = {}
    
    # 1. 分析各任务的学习曲线问题
    task_accuracies = history['task_accuracies']
    individual_losses = history['individual_losses']
    
    for task in task_accuracies.keys():
        task_issues = []
        
        accuracies = task_accuracies[task]
        losses = individual_losses.get(task, [])
        
        # 检查学习停滞
        if len(accuracies) >= 10:
            last_10_improvement = accuracies[-1] - accuracies[-10]
            if last_10_improvement < 0.01:  # 最后10轮改进不足1%
                task_issues.append("学习停滞：最后10轮准确率改进不足1%")
        
        # 检查过拟合
        if len(losses) >= 10:
            min_loss_idx = losses.index(min(losses))
            if min_loss_idx < len(losses) - 5:
                post_min_avg = np.mean(losses[min_loss_idx+1:])
                min_loss = losses[min_loss_idx]
                if post_min_avg > min_loss * 1.1:  # 损失增加超过10%
                    task_issues.append("过拟合风险：最小损失后损失显著增加")
        
        # 检查收敛速度
        if len(accuracies) >= 5:
            early_avg = np.mean(accuracies[:5])
            if accuracies[-1] - early_avg < 0.1:  # 总改进不足10%
                task_issues.append("收敛缓慢：整体准确率改进不足10%")
        
        # 检查稳定性
        if len(accuracies) >= 10:
            stability = np.std(accuracies[-10:])
            if stability > 0.05:  # 标准差大于5%
                task_issues.append(f"训练不稳定：后期准确率波动大(σ={stability:.3f})")
        
        issues[task] = task_issues
    
    return issues

def identify_data_related_problems():
    """识别数据相关的问题"""
    
    # 基于之前的数据分布分析结果
    data_problems = {
        'growth_pattern': {
            'class_imbalance': {
                'severity': 'critical',
                'details': {
                    'major_classes': ['clean(5590)', 'clustered(5335)', 'weak_scattered(3314)'],
                    'minor_classes': ['scattered(36)', 'irregular(35)', 'default_positive(16)'],
                    'imbalance_ratio': 349.38,
                    'impact': '极小类别几乎无法学习，导致整体准确率下降'
                }
            },
            'boundary_confusion': {
                'severity': 'high',
                'details': {
                    'confused_pairs': [
                        ('center_dots', 'litter_center_dots'),
                        ('weak_scattered', 'weak_scattered_pos'),
                        ('scattered', 'strong_scattered')
                    ],
                    'impact': '相似类别间界限模糊，增加分类难度'
                }
            }
        },
        'interference_factors': {
            'pores_dominance': {
                'severity': 'high',
                'details': {
                    'pores_ratio': 0.7546,
                    'pores_accuracy': 0.7477,
                    'problem': 'Pores样本占75%但准确率仅74.77%',
                    'impact': '模型过度依赖Pores特征但效果不佳'
                }
            },
            'rare_class_overfitting': {
                'severity': 'medium',
                'details': {
                    'contamination_samples': 32,
                    'contamination_accuracy': 0.9983,
                    'problem': '极少样本的contamination显示99.83%准确率可能是过拟合',
                    'impact': '实际泛化能力可能很弱'
                }
            }
        },
        'microbe_type': {
            'perfect_accuracy_concern': {
                'severity': 'medium',
                'details': {
                    'accuracy': 1.0,
                    'concern': '100%准确率可能表明任务过于简单或存在数据泄露',
                    'impact': '需要验证是否存在过拟合或数据问题'
                }
            }
        }
    }
    
    return data_problems

def generate_targeted_improvements():
    """生成针对性的改进方案"""
    
    improvements = {
        'immediate_actions': {
            'task_weight_rebalancing': {
                'priority': 'critical',
                'description': '重新平衡多任务学习权重',
                'specific_actions': [
                    '降低microbe_type任务权重（从当前权重降至0.1-0.2）',
                    '提高growth_pattern任务权重（提升至1.5-2.0）',
                    '适度提高interference_factors任务权重（提升至1.2-1.5）',
                    '保持growth_level任务权重不变（表现良好）'
                ],
                'expected_impact': 'growth_pattern准确率提升5-8%，整体平衡性改善'
            },
            'loss_function_optimization': {
                'priority': 'critical',
                'description': '针对类别不平衡优化损失函数',
                'specific_actions': [
                    '对growth_pattern使用Focal Loss (α=0.25, γ=2.0)',
                    '对极小类别(scattered, irregular, default_positive)使用Class-Balanced Loss',
                    '对interference_factors中的pores类别降低权重至0.8',
                    '引入Label Smoothing (ε=0.1)减少过拟合'
                ],
                'expected_impact': '小类别准确率提升15-25%，整体鲁棒性增强'
            }
        },
        'data_level_optimizations': {
            'targeted_data_augmentation': {
                'priority': 'high',
                'description': '针对性数据增强',
                'specific_actions': [
                    '对极小类别(scattered, irregular, default_positive)进行10x过采样',
                    '使用MixUp/CutMix技术生成边界样本',
                    '对center_dots和litter_center_dots进行对比增强',
                    '为pores类别生成更多变化样本'
                ],
                'expected_impact': '小类别样本增加10倍，边界区分能力提升'
            },
            'negative_pores_reweighting': {
                'priority': 'high',
                'description': '重新权衡Negative+Pores样本',
                'specific_actions': [
                    '将Negative+Pores样本权重提升至1.5',
                    '增加Negative+Pores的困难样本挖掘',
                    '使用对比学习区分Pores和其他干扰因素',
                    '引入注意力机制突出Pores特征'
                ],
                'expected_impact': 'Pores检测准确率从74.77%提升至82-85%'
            }
        },
        'model_architecture_enhancements': {
            'task_specific_attention': {
                'priority': 'medium',
                'description': '引入任务特定注意力机制',
                'specific_actions': [
                    '为growth_pattern任务添加空间注意力模块',
                    '为interference_factors任务添加通道注意力',
                    '使用多尺度特征融合处理不同大小的目标',
                    '引入残差连接增强梯度流动'
                ],
                'expected_impact': '特征表达能力提升，任务特异性增强'
            },
            'contrastive_learning': {
                'priority': 'medium',
                'description': '引入对比学习机制',
                'specific_actions': [
                    '对相似类别(center_dots vs litter_center_dots)使用对比损失',
                    '构建困难负样本对提升区分能力',
                    '使用原型网络学习类别中心表示',
                    '引入三元组损失增强类间距离'
                ],
                'expected_impact': '相似类别区分能力提升10-15%'
            }
        },
        'training_strategy_improvements': {
            'curriculum_learning': {
                'priority': 'medium',
                'description': '课程学习策略',
                'specific_actions': [
                    '先训练简单类别(clean, clustered)建立基础特征',
                    '逐步引入困难类别(scattered, irregular)',
                    '使用渐进式难度增加策略',
                    '动态调整样本权重'
                ],
                'expected_impact': '学习效率提升，收敛更稳定'
            },
            'ensemble_methods': {
                'priority': 'low',
                'description': '集成学习方法',
                'specific_actions': [
                    '训练多个专门化模型处理不同任务',
                    '使用投票机制融合预测结果',
                    '对困难样本使用专门的分类器',
                    '引入不确定性估计'
                ],
                'expected_impact': '整体准确率提升3-5%，鲁棒性增强'
            }
        }
    }
    
    return improvements

def calculate_improvement_priorities():
    """计算改进方案的优先级和预期效果"""
    
    priority_analysis = {
        'critical_path': [
            {
                'action': 'task_weight_rebalancing',
                'effort': 'low',
                'impact': 'high',
                'timeline': '1-2 days',
                'expected_improvement': 'growth_pattern: 78.99% → 85-88%'
            },
            {
                'action': 'loss_function_optimization', 
                'effort': 'medium',
                'impact': 'high',
                'timeline': '3-5 days',
                'expected_improvement': '小类别准确率提升15-25%'
            },
            {
                'action': 'targeted_data_augmentation',
                'effort': 'medium',
                'impact': 'high', 
                'timeline': '5-7 days',
                'expected_improvement': '极小类别从39%提升至60-70%'
            }
        ],
        'secondary_improvements': [
            {
                'action': 'negative_pores_reweighting',
                'effort': 'medium',
                'impact': 'medium',
                'timeline': '3-4 days',
                'expected_improvement': 'Pores: 74.77% → 82-85%'
            },
            {
                'action': 'task_specific_attention',
                'effort': 'high',
                'impact': 'medium',
                'timeline': '7-10 days',
                'expected_improvement': '整体准确率提升2-4%'
            }
        ],
        'long_term_optimizations': [
            {
                'action': 'contrastive_learning',
                'effort': 'high',
                'impact': 'medium',
                'timeline': '10-14 days',
                'expected_improvement': '相似类别区分提升10-15%'
            },
            {
                'action': 'curriculum_learning',
                'effort': 'high',
                'impact': 'low',
                'timeline': '7-10 days',
                'expected_improvement': '训练稳定性提升'
            }
        ]
    }
    
    return priority_analysis

def generate_improvement_report():
    """生成改进方案报告"""
    
    # 分析训练问题
    history_path = "/home/aaa/ws/bioastModel/experiments/resnet34_gpu_optimized_20250919_021208/training_history.json"
    training_issues = analyze_training_issues(history_path)
    
    # 识别数据问题
    data_problems = identify_data_related_problems()
    
    # 生成改进方案
    improvements = generate_targeted_improvements()
    
    # 计算优先级
    priorities = calculate_improvement_priorities()
    
    report = []
    report.append("# 针对性改进方案分析报告\n")
    
    # 1. 训练问题诊断
    report.append("## 1. 训练问题诊断\n")
    for task, issues in training_issues.items():
        if issues:
            report.append(f"### {task.upper()}任务问题")
            for issue in issues:
                report.append(f"- {issue}")
            report.append("")
    
    # 2. 数据层面问题分析
    report.append("## 2. 数据层面问题分析\n")
    for task, problems in data_problems.items():
        report.append(f"### {task.upper()}任务数据问题")
        for problem_type, details in problems.items():
            report.append(f"#### {problem_type} (严重程度: {details['severity']})")
            if 'details' in details:
                for key, value in details['details'].items():
                    if isinstance(value, list):
                        report.append(f"- **{key}**: {', '.join(map(str, value))}")
                    else:
                        report.append(f"- **{key}**: {value}")
            report.append("")
    
    # 3. 改进方案详述
    report.append("## 3. 针对性改进方案\n")
    for category, methods in improvements.items():
        report.append(f"### {category.replace('_', ' ').title()}")
        for method, details in methods.items():
            report.append(f"#### {method.replace('_', ' ').title()} (优先级: {details['priority']})")
            report.append(f"**描述**: {details['description']}")
            report.append("**具体行动**:")
            for action in details['specific_actions']:
                report.append(f"- {action}")
            report.append(f"**预期效果**: {details['expected_impact']}\n")
    
    # 4. 实施优先级和时间线
    report.append("## 4. 实施优先级和时间线\n")
    
    report.append("### 关键路径 (立即实施)")
    for item in priorities['critical_path']:
        report.append(f"#### {item['action'].replace('_', ' ').title()}")
        report.append(f"- **工作量**: {item['effort']}")
        report.append(f"- **影响程度**: {item['impact']}")
        report.append(f"- **预计时间**: {item['timeline']}")
        report.append(f"- **预期改进**: {item['expected_improvement']}\n")
    
    report.append("### 次要改进 (后续实施)")
    for item in priorities['secondary_improvements']:
        report.append(f"#### {item['action'].replace('_', ' ').title()}")
        report.append(f"- **工作量**: {item['effort']}")
        report.append(f"- **影响程度**: {item['impact']}")
        report.append(f"- **预计时间**: {item['timeline']}")
        report.append(f"- **预期改进**: {item['expected_improvement']}\n")
    
    report.append("### 长期优化 (可选实施)")
    for item in priorities['long_term_optimizations']:
        report.append(f"#### {item['action'].replace('_', ' ').title()}")
        report.append(f"- **工作量**: {item['effort']}")
        report.append(f"- **影响程度**: {item['impact']}")
        report.append(f"- **预计时间**: {item['timeline']}")
        report.append(f"- **预期改进**: {item['expected_improvement']}\n")
    
    # 5. 总体预期效果
    report.append("## 5. 总体预期效果\n")
    report.append("### 短期改进 (1-2周内)")
    report.append("- **Growth Pattern准确率**: 78.99% → 85-88% (+6-9%)")
    report.append("- **Interference Factors准确率**: 77.43% → 82-85% (+5-8%)")
    report.append("- **极小类别准确率**: 39% → 60-70% (+21-31%)")
    report.append("- **整体多任务准确率**: 当前 → 提升8-12%\n")
    
    report.append("### 中长期改进 (1个月内)")
    report.append("- **Growth Pattern准确率**: 可达90-92%")
    report.append("- **Pores检测准确率**: 74.77% → 85-88%")
    report.append("- **模型稳定性**: 显著提升，波动减少50%")
    report.append("- **泛化能力**: 在新数据上表现提升10-15%\n")
    
    report.append("### 核心结论")
    report.append("通过系统性的改进方案，预计可以实现：")
    report.append("1. **主要瓶颈任务准确率提升8-12%**")
    report.append("2. **极小类别分类能力显著改善**")
    report.append("3. **多任务学习平衡性大幅提升**")
    report.append("4. **模型整体鲁棒性和泛化能力增强**")
    
    return '\n'.join(report)

def main():
    # 生成改进方案报告
    report = generate_improvement_report()
    
    # 保存报告
    output_dir = Path("performance_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "targeted_improvement_analysis.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("✅ 针对性改进方案分析完成！")
    print(f"📋 报告已保存到: {output_dir / 'targeted_improvement_analysis.md'}")
    
    # 输出关键改进建议
    print("\n🎯 关键改进建议:")
    print("1. 立即调整任务权重：降低microbe_type，提高growth_pattern")
    print("2. 使用Focal Loss处理类别不平衡问题")
    print("3. 对极小类别进行10x过采样")
    print("4. 重新权衡Negative+Pores样本")
    print("5. 引入任务特定注意力机制")
    
    print("\n📈 预期效果:")
    print("- Growth Pattern: 78.99% → 85-88%")
    print("- Interference Factors: 77.43% → 82-85%")
    print("- 极小类别: 39% → 60-70%")
    print("- 整体提升: 8-12%")

if __name__ == "__main__":
    main()