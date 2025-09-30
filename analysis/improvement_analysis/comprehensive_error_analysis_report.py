#!/usr/bin/env python3
"""
综合错误分析报告生成脚本
整合Growth Pattern和Pores检测的错误分析结果和改进建议
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_analysis_results():
    """加载分析结果"""
    
    base_dir = Path('/home/aaa/ws/bioastModel')
    
    # 加载测试结果
    test_results_path = base_dir / 'experiments/gpu_training_optimized_run/test_results.json'
    with open(test_results_path, 'r') as f:
        test_results = json.load(f)
    
    # 加载改进配置
    reports_dir = base_dir / 'analysis/improvement_analysis/reports'
    
    with open(reports_dir / 'growth_pattern_improvement_config.json', 'r') as f:
        growth_pattern_config = json.load(f)
    
    with open(reports_dir / 'pores_detection_improvement_config.json', 'r') as f:
        pores_config = json.load(f)
    
    return test_results, growth_pattern_config, pores_config

def analyze_current_performance():
    """分析当前模型性能"""
    
    performance_summary = {
        'growth_level': {
            'accuracy': 98.07,
            'precision': 98.07,
            'recall': 98.07,
            'f1_score': 98.07,
            'status': 'excellent',
            'improvement_needed': False
        },
        'growth_pattern': {
            'accuracy': 76.67,
            'precision': 71.59,
            'recall': 76.67,
            'f1_score': 73.16,
            'status': 'needs_improvement',
            'improvement_needed': True,
            'priority': 'high'
        },
        'interference_factors': {
            'overall_accuracy': 93.47,
            'individual_performance': {
                'artifacts': 92.53,
                'contamination': 99.83,
                'debris': 95.30,
                'pores': 86.20
            },
            'status': 'good_with_exceptions',
            'improvement_needed': True,
            'focus_area': 'pores'
        }
    }
    
    return performance_summary

def identify_critical_issues():
    """识别关键问题"""
    
    critical_issues = {
        'growth_pattern_issues': {
            'class_imbalance': {
                'severity': 'critical',
                'description': '严重的类别不平衡问题',
                'details': [
                    'default_positive仅有3个样本',
                    'irregular仅有5个样本',
                    'scattered仅有6个样本',
                    'clean和clustered样本数量过多(800+)'
                ],
                'impact': '导致小样本类别性能极差'
            },
            'confusion_patterns': {
                'severity': 'high',
                'description': '特定类别间的高混淆率',
                'details': [
                    'focal -> clustered: 94.6%混淆率',
                    'weak_scattered_pos识别率仅43.2%',
                    'strong_scattered经常被误分类'
                ],
                'impact': '整体准确率被拉低至76.67%'
            }
        },
        
        'pores_detection_issues': {
            'visual_similarity': {
                'severity': 'high',
                'description': 'Pores与其他干扰因子视觉相似',
                'details': [
                    '与debris的纹理相似性',
                    '与artifacts的形状相似性',
                    '尺度变化范围大'
                ],
                'impact': '检测准确率仅86.20%'
            },
            'feature_extraction': {
                'severity': 'medium',
                'description': '当前特征提取不足以区分pores',
                'details': [
                    '单一尺度特征提取',
                    '缺乏专门的pores特征',
                    '上下文信息利用不足'
                ],
                'impact': '在干扰因子中性能排名第3'
            }
        }
    }
    
    return critical_issues

def generate_improvement_roadmap():
    """生成改进路线图"""
    
    roadmap = {
        'phase_1_immediate': {
            'duration': '1-2周',
            'priority': 'critical',
            'actions': [
                {
                    'task': '实施Growth Pattern类别权重',
                    'target': 'growth_pattern',
                    'expected_improvement': '+5-8%',
                    'effort': 'low'
                },
                {
                    'task': '添加Focal Loss',
                    'target': 'both',
                    'expected_improvement': '+3-5%',
                    'effort': 'low'
                },
                {
                    'task': '增强数据增强策略',
                    'target': 'both',
                    'expected_improvement': '+2-4%',
                    'effort': 'medium'
                },
                {
                    'task': 'Pores专用损失函数',
                    'target': 'pores',
                    'expected_improvement': '+2-3%',
                    'effort': 'medium'
                }
            ],
            'expected_results': {
                'growth_pattern': '84-88%',
                'pores': '90-94%'
            }
        },
        
        'phase_2_medium_term': {
            'duration': '3-4周',
            'priority': 'high',
            'actions': [
                {
                    'task': '实施注意力机制',
                    'target': 'both',
                    'expected_improvement': '+4-6%',
                    'effort': 'high'
                },
                {
                    'task': '渐进式训练策略',
                    'target': 'growth_pattern',
                    'expected_improvement': '+3-5%',
                    'effort': 'medium'
                },
                {
                    'task': '多尺度特征提取',
                    'target': 'pores',
                    'expected_improvement': '+3-5%',
                    'effort': 'high'
                },
                {
                    'task': '困难负样本挖掘',
                    'target': 'pores',
                    'expected_improvement': '+2-4%',
                    'effort': 'medium'
                }
            ],
            'expected_results': {
                'growth_pattern': '90-95%',
                'pores': '95-97%'
            }
        },
        
        'phase_3_advanced': {
            'duration': '2-3个月',
            'priority': 'medium',
            'actions': [
                {
                    'task': '集成学习策略',
                    'target': 'both',
                    'expected_improvement': '+3-5%',
                    'effort': 'high'
                },
                {
                    'task': '数据集扩充',
                    'target': 'growth_pattern',
                    'expected_improvement': '+5-8%',
                    'effort': 'very_high'
                },
                {
                    'task': '架构搜索优化',
                    'target': 'both',
                    'expected_improvement': '+5-10%',
                    'effort': 'very_high'
                }
            ],
            'expected_results': {
                'growth_pattern': '95%+',
                'pores': '97%+'
            }
        }
    }
    
    return roadmap

def create_performance_visualization():
    """创建性能可视化图表"""
    
    # 当前性能对比
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 整体性能对比
    categories = ['Growth Level', 'Growth Pattern', 'Interference Factors']
    current_scores = [98.07, 76.67, 93.47]
    target_scores = [98.07, 90.0, 96.0]  # 目标分数
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, current_scores, width, label='Current', color='lightcoral', alpha=0.8)
    ax1.bar(x + width/2, target_scores, width, label='Target', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('Task Categories')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Current vs Target Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Growth Pattern类别分布
    gp_labels = ['center_dots', 'clean', 'clustered', 'default_pos', 'focal', 'heavy_growth', 
                 'irregular', 'litter_center', 'scattered', 'strong_scattered', 'weak_scattered', 'weak_scattered_pos']
    gp_counts = [96, 820, 790, 3, 249, 243, 5, 140, 6, 108, 503, 37]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(gp_labels)))
    ax2.pie(gp_counts, labels=gp_labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Growth Pattern Class Distribution')
    
    # 3. 干扰因子性能对比
    if_labels = ['Artifacts', 'Contamination', 'Debris', 'Pores']
    if_scores = [92.53, 99.83, 95.30, 86.20]
    if_targets = [95.0, 99.83, 97.0, 94.0]
    
    x_if = np.arange(len(if_labels))
    ax3.bar(x_if - width/2, if_scores, width, label='Current', color='lightblue', alpha=0.8)
    ax3.bar(x_if + width/2, if_targets, width, label='Target', color='orange', alpha=0.8)
    
    ax3.set_xlabel('Interference Factors')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Interference Factors Performance')
    ax3.set_xticks(x_if)
    ax3.set_xticklabels(if_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 改进时间线
    phases = ['Current', 'Phase 1\n(2 weeks)', 'Phase 2\n(1 month)', 'Phase 3\n(3 months)']
    gp_timeline = [76.67, 84, 92, 95]
    pores_timeline = [86.20, 91, 96, 97]
    
    ax4.plot(phases, gp_timeline, marker='o', linewidth=2, label='Growth Pattern', color='red')
    ax4.plot(phases, pores_timeline, marker='s', linewidth=2, label='Pores Detection', color='blue')
    
    ax4.set_xlabel('Timeline')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Improvement Timeline')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(70, 100)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports')
    plt.savefig(output_dir / 'comprehensive_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_dir / 'comprehensive_performance_analysis.png')

def generate_detailed_report():
    """生成详细的分析报告"""
    
    test_results, growth_pattern_config, pores_config = load_analysis_results()
    performance = analyze_current_performance()
    issues = identify_critical_issues()
    roadmap = generate_improvement_roadmap()
    
    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'model_version': 'MultiLevel-MobileNetV3-small',
            'training_run': 'gpu_training_optimized_run',
            'analysis_scope': ['growth_pattern', 'pores_detection']
        },
        
        'executive_summary': {
            'current_status': {
                'growth_level': 'Excellent (98.07%)',
                'growth_pattern': 'Needs Improvement (76.67%)',
                'interference_factors': 'Good with exceptions (93.47%)',
                'pores_detection': 'Needs Improvement (86.20%)'
            },
            'key_findings': [
                'Growth Pattern分类受严重类别不平衡影响',
                'Pores检测面临视觉相似性挑战',
                '小样本类别性能极差需要专门处理',
                '现有特征提取对复杂模式识别不足'
            ],
            'improvement_potential': {
                'growth_pattern': '预期可提升至90-95%',
                'pores_detection': '预期可提升至94-97%',
                'timeline': '3-6个月内达到目标'
            }
        },
        
        'detailed_analysis': {
            'performance_breakdown': performance,
            'critical_issues': issues,
            'root_cause_analysis': {
                'data_issues': [
                    '类别分布极度不均衡',
                    '小样本类别标注质量不一致',
                    '视觉相似类别缺乏区分性特征'
                ],
                'model_issues': [
                    '单一尺度特征提取限制',
                    '缺乏针对性的损失函数',
                    '注意力机制缺失'
                ],
                'training_issues': [
                    '未考虑类别权重',
                    '数据增强策略不够针对性',
                    '缺乏困难样本挖掘'
                ]
            }
        },
        
        'improvement_strategy': {
            'roadmap': roadmap,
            'technical_solutions': {
                'growth_pattern': {
                    'class_weighting': '权重范围0.019-5.219',
                    'focal_loss': 'alpha=1.0, gamma=2.0',
                    'progressive_training': '4阶段训练策略',
                    'attention_mechanism': 'Channel-Spatial Attention',
                    'data_augmentation': '12种增强技术'
                },
                'pores_detection': {
                    'multi_scale_features': '3x3, 5x5, 7x7卷积核',
                    'specialized_loss': 'Pores权重2.0',
                    'hard_negative_mining': '30%困难样本比例',
                    'two_stage_training': '通用+专用训练',
                    'ensemble_strategy': '3模型集成'
                }
            }
        },
        
        'implementation_plan': {
            'immediate_actions': roadmap['phase_1_immediate']['actions'],
            'resource_requirements': {
                'computational': 'GPU训练时间约100-150小时',
                'human_resources': '1-2名工程师，3-6个月',
                'data_requirements': '可能需要额外收集500-1000个样本'
            },
            'risk_assessment': {
                'technical_risks': [
                    '复杂模型可能过拟合',
                    '训练时间显著增加',
                    '内存需求增大'
                ],
                'mitigation_strategies': [
                    '渐进式实施策略',
                    '充分的验证和测试',
                    '性能监控和早停机制'
                ]
            }
        },
        
        'expected_outcomes': {
            'performance_targets': {
                'growth_pattern': {
                    'short_term': '84-88% (2周内)',
                    'medium_term': '90-95% (2个月内)',
                    'long_term': '95%+ (6个月内)'
                },
                'pores_detection': {
                    'short_term': '90-94% (2周内)',
                    'medium_term': '95-97% (2个月内)',
                    'long_term': '97%+ (6个月内)'
                }
            },
            'business_impact': {
                'accuracy_improvement': '整体模型准确率提升5-10%',
                'reliability_enhancement': '减少误诊率和漏诊率',
                'deployment_readiness': '达到生产环境部署标准'
            }
        }
    }
    
    return report

def save_comprehensive_report():
    """保存综合报告"""
    
    # 生成报告
    report = generate_detailed_report()
    
    # 创建可视化
    chart_path = create_performance_visualization()
    
    # 保存JSON报告
    output_dir = Path('/home/aaa/ws/bioastModel/analysis/improvement_analysis/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'comprehensive_error_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成Markdown报告
    markdown_report = generate_markdown_report(report)
    with open(output_dir / 'comprehensive_error_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    return output_dir

def generate_markdown_report(report_data):
    """生成Markdown格式的报告"""
    
    markdown = f"""# 生物样本分析模型错误分析与改进策略报告

## 报告概要

**生成时间**: {report_data['report_metadata']['generated_at']}  
**模型版本**: {report_data['report_metadata']['model_version']}  
**训练运行**: {report_data['report_metadata']['training_run']}  

## 执行摘要

### 当前性能状态
- **Growth Level**: {report_data['executive_summary']['current_status']['growth_level']}
- **Growth Pattern**: {report_data['executive_summary']['current_status']['growth_pattern']}
- **Interference Factors**: {report_data['executive_summary']['current_status']['interference_factors']}
- **Pores Detection**: {report_data['executive_summary']['current_status']['pores_detection']}

### 关键发现
"""
    
    for finding in report_data['executive_summary']['key_findings']:
        markdown += f"- {finding}\n"
    
    markdown += f"""
### 改进潜力
- **Growth Pattern**: {report_data['executive_summary']['improvement_potential']['growth_pattern']}
- **Pores Detection**: {report_data['executive_summary']['improvement_potential']['pores_detection']}
- **时间线**: {report_data['executive_summary']['improvement_potential']['timeline']}

## 详细分析

### 关键问题识别

#### Growth Pattern分类问题
1. **类别不平衡** (严重程度: 关键)
   - default_positive仅有3个样本
   - irregular仅有5个样本  
   - scattered仅有6个样本
   - 导致小样本类别性能极差

2. **混淆模式** (严重程度: 高)
   - focal -> clustered: 94.6%混淆率
   - weak_scattered_pos识别率仅43.2%
   - 整体准确率被拉低至76.67%

#### Pores检测问题
1. **视觉相似性** (严重程度: 高)
   - 与debris的纹理相似性
   - 与artifacts的形状相似性
   - 检测准确率仅86.20%

2. **特征提取不足** (严重程度: 中等)
   - 单一尺度特征提取
   - 缺乏专门的pores特征
   - 在干扰因子中性能排名第3

## 改进策略

### 第一阶段 (1-2周) - 立即实施
**目标**: Growth Pattern 84-88%, Pores 90-94%

1. **实施类别权重** - Growth Pattern (+5-8%)
2. **添加Focal Loss** - 两个任务 (+3-5%)
3. **增强数据增强** - 两个任务 (+2-4%)
4. **Pores专用损失函数** - Pores (+2-3%)

### 第二阶段 (3-4周) - 中期实施
**目标**: Growth Pattern 90-95%, Pores 95-97%

1. **注意力机制** - 两个任务 (+4-6%)
2. **渐进式训练** - Growth Pattern (+3-5%)
3. **多尺度特征提取** - Pores (+3-5%)
4. **困难负样本挖掘** - Pores (+2-4%)

### 第三阶段 (2-3个月) - 高级优化
**目标**: Growth Pattern 95%+, Pores 97%+

1. **集成学习策略** - 两个任务 (+3-5%)
2. **数据集扩充** - Growth Pattern (+5-8%)
3. **架构搜索优化** - 两个任务 (+5-10%)

## 技术实施细节

### Growth Pattern改进
- **类别权重范围**: 0.019-5.219
- **Focal Loss参数**: alpha=1.0, gamma=2.0
- **渐进式训练**: 4阶段训练策略
- **注意力机制**: Channel-Spatial Attention
- **数据增强**: 12种增强技术

### Pores检测改进
- **多尺度特征**: 3x3, 5x5, 7x7卷积核
- **专用损失**: Pores权重2.0
- **困难样本挖掘**: 30%困难样本比例
- **两阶段训练**: 通用+专用训练
- **集成策略**: 3模型集成

## 资源需求与风险评估

### 资源需求
- **计算资源**: GPU训练时间约100-150小时
- **人力资源**: 1-2名工程师，3-6个月
- **数据需求**: 可能需要额外收集500-1000个样本

### 风险缓解
- 渐进式实施策略
- 充分的验证和测试
- 性能监控和早停机制

## 预期成果

### 性能目标
- **Growth Pattern**: 短期84-88% → 中期90-95% → 长期95%+
- **Pores Detection**: 短期90-94% → 中期95-97% → 长期97%+

### 业务影响
- 整体模型准确率提升5-10%
- 减少误诊率和漏诊率
- 达到生产环境部署标准

## 结论与建议

1. **立即开始第一阶段改进**，预期2周内看到显著提升
2. **优先处理类别不平衡问题**，这是Growth Pattern性能的主要瓶颈
3. **为Pores检测实施专门的特征提取**，解决视觉相似性问题
4. **采用渐进式实施策略**，降低风险并确保稳定改进

通过系统性的改进策略，预期在3-6个月内将模型性能提升至生产就绪水平。
"""
    
    return markdown

def main():
    """主函数"""
    
    print("=== 生成综合错误分析报告 ===\n")
    
    print("1. 加载分析结果...")
    test_results, growth_pattern_config, pores_config = load_analysis_results()
    
    print("2. 分析当前性能...")
    performance = analyze_current_performance()
    
    print("3. 识别关键问题...")
    issues = identify_critical_issues()
    
    print("4. 生成改进路线图...")
    roadmap = generate_improvement_roadmap()
    
    print("5. 创建性能可视化...")
    chart_path = create_performance_visualization()
    print(f"   可视化图表已保存: {chart_path}")
    
    print("6. 生成详细报告...")
    report = generate_detailed_report()
    
    print("7. 保存综合报告...")
    output_dir = save_comprehensive_report()
    
    print(f"\n=== 报告生成完成 ===")
    print(f"输出目录: {output_dir}")
    print(f"文件列表:")
    print(f"  - comprehensive_error_analysis_report.json")
    print(f"  - comprehensive_error_analysis_report.md")
    print(f"  - comprehensive_performance_analysis.png")
    print(f"  - growth_pattern_improvement_config.json")
    print(f"  - pores_detection_improvement_config.json")
    
    print(f"\n=== 关键建议 ===")
    print("1. 立即实施第一阶段改进 (预期2周内完成)")
    print("2. Growth Pattern: 重点解决类别不平衡问题")
    print("3. Pores Detection: 实施多尺度特征提取")
    print("4. 预期整体性能提升: 5-10%")

if __name__ == "__main__":
    main()