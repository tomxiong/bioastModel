#!/usr/bin/env python3
"""
正确GPU训练改进建议生成脚本
基于瓶颈分析结果提供具体的、可执行的改进方案
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

class ImprovementSuggestionGenerator:
    def __init__(self, bottleneck_file, metrics_file):
        self.bottleneck_file = Path(bottleneck_file)
        self.metrics_file = Path(metrics_file)
        self.bottleneck_data = {}
        self.metrics_data = {}
        self.improvement_plan = {}
        
    def load_analysis_data(self):
        """加载分析数据"""
        print("📊 加载分析数据...")
        
        with open(self.bottleneck_file, 'r', encoding='utf-8') as f:
            self.bottleneck_data = json.load(f)
        
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            self.metrics_data = json.load(f)
        
        print("✅ 分析数据加载完成")
    
    def generate_comprehensive_improvements(self):
        """生成全面的改进建议"""
        print("\n🚀 生成全面改进建议...")
        
        # 基于关键瓶颈生成针对性改进
        self._generate_critical_issue_solutions()
        
        # 生成数据层面改进
        self._generate_data_improvements()
        
        # 生成模型层面改进
        self._generate_model_improvements()
        
        # 生成训练策略改进
        self._generate_training_improvements()
        
        # 生成系统层面改进
        self._generate_system_improvements()
        
        # 优先级排序和实施计划
        self._create_implementation_plan()
        
        print("✅ 改进建议生成完成")
    
    def _generate_critical_issue_solutions(self):
        """生成关键问题解决方案"""
        print("  🎯 生成关键问题解决方案...")
        
        critical_bottlenecks = self.bottleneck_data.get('critical_bottlenecks', [])
        critical_solutions = {}
        
        for bottleneck in critical_bottlenecks[:3]:  # 处理前3个最严重的瓶颈
            bottleneck_name = bottleneck['name']
            bottleneck_type = bottleneck['type']
            severity = bottleneck['severity']
            issues = bottleneck['issues']
            
            solutions = []
            
            if 'interference_factors' in bottleneck_name:
                if bottleneck_type == 'task':
                    # interference_factors任务完全失败的解决方案
                    solutions.extend([
                        {
                            'action': '检查标签映射',
                            'description': '验证interference_factors标签是否正确映射到模型输出',
                            'priority': 'critical',
                            'effort': 'low',
                            'timeline': '立即',
                            'expected_impact': '修复0准确率问题',
                            'implementation': [
                                '检查数据加载器中interference_factors标签的处理',
                                '验证标签编码是否与模型输出维度匹配',
                                '确认标签值范围是否正确'
                            ]
                        },
                        {
                            'action': '修复损失函数配置',
                            'description': '确保interference_factors任务使用正确的损失函数',
                            'priority': 'critical',
                            'effort': 'low',
                            'timeline': '立即',
                            'expected_impact': '恢复正常训练',
                            'implementation': [
                                '检查多任务损失函数中interference_factors的配置',
                                '验证损失权重是否设置正确',
                                '确认损失函数类型（分类vs回归）'
                            ]
                        },
                        {
                            'action': '验证数据预处理',
                            'description': '检查interference_factors数据预处理流程',
                            'priority': 'critical',
                            'effort': 'medium',
                            'timeline': '1-2天',
                            'expected_impact': '确保数据质量',
                            'implementation': [
                                '检查interference_factors特征提取是否正确',
                                '验证数据归一化和标准化',
                                '确认数据增强不会破坏标签'
                            ]
                        }
                    ])
                
                elif bottleneck_type == 'data':
                    # 数据不平衡解决方案
                    imbalance_ratio = bottleneck['details']['imbalance_ratio']
                    solutions.extend([
                        {
                            'action': '实施Focal Loss',
                            'description': f'使用Focal Loss处理{imbalance_ratio:.1f}:1的严重不平衡',
                            'priority': 'high',
                            'effort': 'medium',
                            'timeline': '2-3天',
                            'expected_impact': '显著改善少数类识别',
                            'implementation': [
                                '为interference_factors任务实施Focal Loss',
                                '调整alpha和gamma参数',
                                '监控各类别的损失贡献'
                            ]
                        },
                        {
                            'action': '类别权重调整',
                            'description': '根据类别分布调整损失权重',
                            'priority': 'high',
                            'effort': 'low',
                            'timeline': '1天',
                            'expected_impact': '平衡各类别学习',
                            'implementation': [
                                '计算逆频率权重',
                                '应用到损失函数',
                                '监控训练平衡性'
                            ]
                        },
                        {
                            'action': '数据重采样',
                            'description': '对少数类进行过采样',
                            'priority': 'medium',
                            'effort': 'medium',
                            'timeline': '3-5天',
                            'expected_impact': '增加少数类样本',
                            'implementation': [
                                '实施SMOTE或ADASYN过采样',
                                '保持验证集原始分布',
                                '监控过拟合风险'
                            ]
                        }
                    ])
            
            elif 'growth_pattern' in bottleneck_name:
                # growth_pattern类别混淆解决方案
                solutions.extend([
                    {
                        'action': '分析混淆类别',
                        'description': '深入分析类别4和类别2的混淆原因',
                        'priority': 'high',
                        'effort': 'medium',
                        'timeline': '2-3天',
                        'expected_impact': '理解混淆根因',
                        'implementation': [
                            '可视化混淆样本',
                            '分析特征相似性',
                            '检查标签质量'
                        ]
                    },
                    {
                        'action': '特征增强',
                        'description': '增强区分性特征提取',
                        'priority': 'medium',
                        'effort': 'high',
                        'timeline': '1-2周',
                        'expected_impact': '提高类别区分度',
                        'implementation': [
                            '添加注意力机制',
                            '使用更深的特征提取',
                            '引入空间注意力'
                        ]
                    }
                ])
            
            critical_solutions[bottleneck_name] = {
                'bottleneck_info': bottleneck,
                'solutions': solutions,
                'total_solutions': len(solutions)
            }
        
        self.improvement_plan['critical_solutions'] = critical_solutions
    
    def _generate_data_improvements(self):
        """生成数据层面改进"""
        print("  📊 生成数据层面改进...")
        
        data_bottlenecks = self.bottleneck_data.get('detailed_analysis', {}).get('data_level', {})
        data_improvements = {}
        
        for task, data_info in data_bottlenecks.items():
            imbalance_ratio = data_info.get('imbalance_ratio', 1)
            distribution = data_info.get('distribution', {})
            
            task_improvements = []
            
            # 基于不平衡程度的改进
            if imbalance_ratio > 50:
                task_improvements.extend([
                    {
                        'strategy': '极端不平衡处理',
                        'techniques': [
                            'Focal Loss (alpha=0.25, gamma=2.0)',
                            'Class-balanced Loss',
                            'SMOTE + Tomek Links',
                            '集成学习方法'
                        ],
                        'priority': 'critical',
                        'expected_improvement': '30-50%准确率提升'
                    }
                ])
            elif imbalance_ratio > 10:
                task_improvements.extend([
                    {
                        'strategy': '严重不平衡处理',
                        'techniques': [
                            'Focal Loss',
                            '类别权重调整',
                            'SMOTE过采样',
                            '数据增强'
                        ],
                        'priority': 'high',
                        'expected_improvement': '15-25%准确率提升'
                    }
                ])
            elif imbalance_ratio > 3:
                task_improvements.extend([
                    {
                        'strategy': '中等不平衡处理',
                        'techniques': [
                            '类别权重调整',
                            '轻度过采样',
                            '数据增强'
                        ],
                        'priority': 'medium',
                        'expected_improvement': '5-15%准确率提升'
                    }
                ])
            
            # 基于具体分布的改进
            if distribution:
                min_samples = min(distribution.values())
                max_samples = max(distribution.values())
                
                # 识别极小类别
                small_classes = [k for k, v in distribution.items() if v < 50]
                if small_classes:
                    task_improvements.append({
                        'strategy': '极小类别增强',
                        'target_classes': small_classes,
                        'techniques': [
                            f'对{len(small_classes)}个小类别进行10x过采样',
                            '使用生成对抗网络(GAN)生成样本',
                            '迁移学习从相似类别',
                            '半监督学习'
                        ],
                        'priority': 'high',
                        'expected_improvement': f'小类别准确率提升40-60%'
                    })
                
                # 数据收集建议
                if min_samples < 100:
                    task_improvements.append({
                        'strategy': '数据收集计划',
                        'techniques': [
                            f'为少数类收集至少{max(100, min_samples * 3)}个样本',
                            '主动学习选择有价值样本',
                            '众包标注',
                            '合成数据生成'
                        ],
                        'priority': 'long_term',
                        'expected_improvement': '长期稳定性提升'
                    })
            
            data_improvements[task] = task_improvements
        
        self.improvement_plan['data_improvements'] = data_improvements
    
    def _generate_model_improvements(self):
        """生成模型层面改进"""
        print("  🏗️ 生成模型层面改进...")
        
        task_performance = self.metrics_data.get('analysis', {}).get('task_performance_analysis', {})
        model_improvements = {}
        
        # 多任务学习改进
        task_accuracies = {}
        for task, perf in task_performance.items():
            if task == 'interference_factors':
                acc = perf.get('final_metrics', {}).get('overall', {}).get('accuracy', 0)
            else:
                acc = perf.get('final_metrics', {}).get('accuracy', 0)
            task_accuracies[task] = acc
        
        if task_accuracies:
            max_acc = max(task_accuracies.values())
            min_acc = min(task_accuracies.values())
            acc_gap = max_acc - min_acc
            
            multitask_improvements = []
            
            if acc_gap > 0.3:
                multitask_improvements.extend([
                    {
                        'approach': '任务权重动态调整',
                        'description': '根据任务难度动态调整损失权重',
                        'implementation': [
                            '实施Uncertainty Weighting',
                            '使用GradNorm算法',
                            '监控任务间梯度冲突'
                        ],
                        'priority': 'high',
                        'expected_impact': '平衡各任务学习'
                    },
                    {
                        'approach': '任务特定适配器',
                        'description': '为每个任务添加专门的适配器层',
                        'implementation': [
                            '在共享特征后添加任务特定层',
                            '使用Adapter或LoRA技术',
                            '保持共享特征的通用性'
                        ],
                        'priority': 'medium',
                        'expected_impact': '提高任务特异性'
                    }
                ])
            
            # 针对失败任务的特殊处理
            failed_tasks = [task for task, acc in task_accuracies.items() if acc < 0.1]
            if failed_tasks:
                multitask_improvements.append({
                    'approach': '失败任务重构',
                    'description': f'重新设计{", ".join(failed_tasks)}任务的架构',
                    'implementation': [
                        '检查任务定义和标签',
                        '使用单独的分支处理',
                        '考虑任务分解'
                    ],
                    'priority': 'critical',
                    'expected_impact': '修复任务失败问题'
                })
            
            model_improvements['multitask_learning'] = multitask_improvements
        
        # 架构改进
        architecture_improvements = [
            {
                'approach': '注意力机制增强',
                'description': '添加多尺度注意力机制',
                'techniques': [
                    'Spatial Attention Module (SAM)',
                    'Channel Attention Module (CAM)',
                    'Self-Attention for long-range dependencies'
                ],
                'priority': 'medium',
                'expected_impact': '5-10%准确率提升'
            },
            {
                'approach': '特征融合改进',
                'description': '改进多层特征融合策略',
                'techniques': [
                    'Feature Pyramid Network (FPN)',
                    'Path Aggregation Network (PANet)',
                    'Bi-directional Feature Pyramid'
                ],
                'priority': 'medium',
                'expected_impact': '3-8%准确率提升'
            },
            {
                'approach': '正则化增强',
                'description': '添加更强的正则化技术',
                'techniques': [
                    'DropBlock for spatial regularization',
                    'Mixup/CutMix data augmentation',
                    'Label Smoothing',
                    'Stochastic Depth'
                ],
                'priority': 'low',
                'expected_impact': '2-5%准确率提升，提高泛化'
            }
        ]
        
        model_improvements['architecture'] = architecture_improvements
        
        # 模型容量调整
        convergence_analysis = self.metrics_data.get('analysis', {}).get('convergence_analysis', {})
        convergence_status = convergence_analysis.get('convergence_status', 'unknown')
        
        capacity_improvements = []
        
        if convergence_status == 'converged' and min(task_accuracies.values()) < 0.85:
            capacity_improvements.append({
                'approach': '增加模型容量',
                'description': '当前模型可能容量不足',
                'options': [
                    '使用MobileNetV3-Large替代Small',
                    '增加额外的特征提取层',
                    '使用EfficientNet-B1或B2'
                ],
                'priority': 'medium',
                'expected_impact': '5-15%准确率提升'
            })
        elif convergence_status == 'still_changing':
            capacity_improvements.append({
                'approach': '优化训练效率',
                'description': '模型收敛缓慢，需要优化',
                'options': [
                    '使用更好的初始化策略',
                    '调整学习率调度',
                    '使用预训练权重'
                ],
                'priority': 'high',
                'expected_impact': '加速收敛，稳定训练'
            })
        
        if capacity_improvements:
            model_improvements['capacity'] = capacity_improvements
        
        self.improvement_plan['model_improvements'] = model_improvements
    
    def _generate_training_improvements(self):
        """生成训练策略改进"""
        print("  🎓 生成训练策略改进...")
        
        convergence_analysis = self.metrics_data.get('analysis', {}).get('convergence_analysis', {})
        stability_analysis = self.metrics_data.get('analysis', {}).get('training_stability', {})
        
        training_improvements = {}
        
        # 优化器和学习率改进
        optimizer_improvements = [
            {
                'strategy': '自适应学习率',
                'description': '使用更先进的学习率调度策略',
                'techniques': [
                    'Cosine Annealing with Warm Restarts',
                    'OneCycleLR调度器',
                    'ReduceLROnPlateau with patience=3'
                ],
                'priority': 'high',
                'expected_impact': '更稳定的收敛'
            },
            {
                'strategy': '优化器升级',
                'description': '使用更适合多任务学习的优化器',
                'techniques': [
                    'AdamW with weight decay=0.01',
                    'RAdam (Rectified Adam)',
                    'Lookahead optimizer'
                ],
                'priority': 'medium',
                'expected_impact': '更好的泛化性能'
            }
        ]
        
        training_improvements['optimization'] = optimizer_improvements
        
        # 训练策略改进
        strategy_improvements = []
        
        # 基于收敛分析的改进
        overfitting_signs = convergence_analysis.get('overfitting_signs', [])
        if overfitting_signs:
            strategy_improvements.extend([
                {
                    'strategy': '过拟合控制',
                    'description': '实施更强的过拟合控制措施',
                    'techniques': [
                        'Early Stopping (patience=5)',
                        'Dropout rate增加到0.3',
                        'L2正则化权重增加',
                        '数据增强强度提升'
                    ],
                    'priority': 'high',
                    'expected_impact': '提高泛化能力'
                }
            ])
        
        # 基于稳定性分析的改进
        if stability_analysis:
            batch_stability = stability_analysis.get('batch_level', {})
            if batch_stability.get('stability_assessment') == 'unstable':
                strategy_improvements.append({
                    'strategy': '训练稳定性提升',
                    'description': '改善训练过程的稳定性',
                    'techniques': [
                        '梯度裁剪 (max_norm=1.0)',
                        '批次大小调整',
                        '学习率降低50%',
                        '使用EMA (Exponential Moving Average)'
                    ],
                    'priority': 'high',
                    'expected_impact': '更稳定的训练过程'
                })
        
        # 多任务学习策略
        multitask_strategies = [
            {
                'strategy': '渐进式多任务学习',
                'description': '逐步引入任务，避免任务冲突',
                'implementation': [
                    '先训练最简单的任务(growth_level)',
                    '逐步添加其他任务',
                    '使用任务课程学习'
                ],
                'priority': 'medium',
                'expected_impact': '减少任务间干扰'
            },
            {
                'strategy': '任务平衡采样',
                'description': '确保各任务得到平衡的训练',
                'implementation': [
                    '按任务难度调整采样权重',
                    '使用任务特定的batch组成',
                    '监控各任务的梯度贡献'
                ],
                'priority': 'high',
                'expected_impact': '平衡多任务学习'
            }
        ]
        
        strategy_improvements.extend(multitask_strategies)
        training_improvements['strategies'] = strategy_improvements
        
        # 数据增强改进
        augmentation_improvements = [
            {
                'strategy': '任务特定数据增强',
                'description': '为不同任务设计专门的数据增强',
                'techniques': {
                    'growth_level': ['轻度旋转', '亮度调整'],
                    'growth_pattern': ['几何变换', '纹理增强'],
                    'interference_factors': ['噪声添加', '对比度调整']
                },
                'priority': 'medium',
                'expected_impact': '提高各任务的鲁棒性'
            },
            {
                'strategy': '高级数据增强',
                'description': '使用更先进的数据增强技术',
                'techniques': [
                    'AutoAugment策略搜索',
                    'MixUp/CutMix混合',
                    'AugMax自适应增强'
                ],
                'priority': 'low',
                'expected_impact': '进一步提升泛化能力'
            }
        ]
        
        training_improvements['augmentation'] = augmentation_improvements
        
        self.improvement_plan['training_improvements'] = training_improvements
    
    def _generate_system_improvements(self):
        """生成系统层面改进"""
        print("  ⚙️ 生成系统层面改进...")
        
        system_improvements = {
            'monitoring': [
                {
                    'improvement': '增强训练监控',
                    'description': '实施更全面的训练监控系统',
                    'components': [
                        'TensorBoard详细日志记录',
                        'Weights & Biases实验跟踪',
                        '实时性能指标监控',
                        '梯度和权重分布可视化'
                    ],
                    'priority': 'medium',
                    'benefit': '更好的训练洞察和调试能力'
                }
            ],
            'evaluation': [
                {
                    'improvement': '评估体系完善',
                    'description': '建立更全面的模型评估体系',
                    'components': [
                        '类别级别详细分析',
                        '混淆矩阵热图可视化',
                        '错误样本分析',
                        '模型解释性分析'
                    ],
                    'priority': 'medium',
                    'benefit': '更深入的性能理解'
                }
            ],
            'deployment': [
                {
                    'improvement': '模型部署优化',
                    'description': '为生产环境优化模型',
                    'components': [
                        '模型量化和压缩',
                        'ONNX格式转换',
                        '推理速度优化',
                        'A/B测试框架'
                    ],
                    'priority': 'low',
                    'benefit': '生产环境性能提升'
                }
            ]
        }
        
        self.improvement_plan['system_improvements'] = system_improvements
    
    def _create_implementation_plan(self):
        """创建实施计划"""
        print("  📋 创建实施计划...")
        
        all_improvements = []
        
        # 收集所有改进建议
        for category, improvements in self.improvement_plan.items():
            if category == 'critical_solutions':
                for task, solution_data in improvements.items():
                    for solution in solution_data['solutions']:
                        all_improvements.append({
                            'category': 'critical',
                            'task': task,
                            'improvement': solution,
                            'priority_score': self._calculate_improvement_priority(solution, 'critical')
                        })
            elif category == 'data_improvements':
                for task, task_improvements in improvements.items():
                    for improvement in task_improvements:
                        all_improvements.append({
                            'category': 'data',
                            'task': task,
                            'improvement': improvement,
                            'priority_score': self._calculate_improvement_priority(improvement, 'data')
                        })
            # 可以继续添加其他类别...
        
        # 按优先级排序
        all_improvements.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 创建分阶段实施计划
        implementation_phases = {
            'phase_1_immediate': {
                'timeline': '1-3天',
                'description': '立即执行的关键修复',
                'improvements': [imp for imp in all_improvements[:5] 
                               if imp['improvement'].get('timeline') in ['立即', '1天']]
            },
            'phase_2_short_term': {
                'timeline': '1-2周',
                'description': '短期改进措施',
                'improvements': [imp for imp in all_improvements 
                               if imp['improvement'].get('priority') in ['high', 'critical'] 
                               and imp not in all_improvements[:5]][:8]
            },
            'phase_3_medium_term': {
                'timeline': '2-4周',
                'description': '中期优化措施',
                'improvements': [imp for imp in all_improvements 
                               if imp['improvement'].get('priority') == 'medium'][:6]
            },
            'phase_4_long_term': {
                'timeline': '1-3个月',
                'description': '长期战略改进',
                'improvements': [imp for imp in all_improvements 
                               if imp['improvement'].get('priority') in ['low', 'long_term']][:4]
            }
        }
        
        self.improvement_plan['implementation_phases'] = implementation_phases
        
        # 计算预期总体改进
        self._calculate_expected_improvements()
    
    def _calculate_improvement_priority(self, improvement, category):
        """计算改进优先级分数"""
        priority_weights = {
            'critical': 10,
            'high': 8,
            'medium': 5,
            'low': 2,
            'long_term': 1
        }
        
        category_weights = {
            'critical': 1.0,
            'data': 0.8,
            'model': 0.6,
            'training': 0.4,
            'system': 0.2
        }
        
        priority = improvement.get('priority', 'medium')
        effort = improvement.get('effort', 'medium')
        
        priority_score = priority_weights.get(priority, 5)
        category_weight = category_weights.get(category, 0.5)
        
        # 努力程度反向权重（努力越小，优先级越高）
        effort_weights = {'low': 1.0, 'medium': 0.7, 'high': 0.4}
        effort_weight = effort_weights.get(effort, 0.7)
        
        return priority_score * category_weight * effort_weight
    
    def _calculate_expected_improvements(self):
        """计算预期改进效果"""
        print("  📊 计算预期改进效果...")
        
        current_performance = {}
        task_analysis = self.metrics_data.get('analysis', {}).get('task_performance_analysis', {})
        
        for task, perf in task_analysis.items():
            if task == 'interference_factors':
                current_acc = perf.get('final_metrics', {}).get('overall', {}).get('accuracy', 0)
            else:
                current_acc = perf.get('final_metrics', {}).get('accuracy', 0)
            current_performance[task] = current_acc
        
        # 预期改进计算
        expected_improvements = {}
        
        for task, current_acc in current_performance.items():
            task_improvements = []
            
            # 关键问题修复的改进
            if task == 'interference_factors' and current_acc == 0:
                task_improvements.append({
                    'source': '修复关键问题',
                    'improvement': 0.6,  # 从0提升到60%
                    'confidence': 'high'
                })
            
            # 数据不平衡处理的改进
            data_bottlenecks = self.bottleneck_data.get('detailed_analysis', {}).get('data_level', {})
            if task in data_bottlenecks:
                imbalance_ratio = data_bottlenecks[task].get('imbalance_ratio', 1)
                if imbalance_ratio > 10:
                    improvement = min(0.3, (imbalance_ratio - 1) * 0.01)  # 最多30%改进
                    task_improvements.append({
                        'source': '数据不平衡处理',
                        'improvement': improvement,
                        'confidence': 'medium'
                    })
            
            # 模型改进的效果
            if current_acc > 0.5:  # 只有在有基础性能时才考虑模型改进
                task_improvements.append({
                    'source': '模型架构优化',
                    'improvement': 0.05,  # 5%改进
                    'confidence': 'medium'
                })
            
            # 训练策略改进
            task_improvements.append({
                'source': '训练策略优化',
                'improvement': 0.03,  # 3%改进
                'confidence': 'low'
            })
            
            # 计算总预期改进（非线性叠加）
            total_improvement = 0
            for imp in task_improvements:
                # 使用递减效应模拟改进叠加
                total_improvement += imp['improvement'] * (1 - total_improvement)
            
            expected_final = min(0.95, current_acc + total_improvement)  # 最高95%
            
            expected_improvements[task] = {
                'current_accuracy': current_acc,
                'expected_accuracy': expected_final,
                'absolute_improvement': expected_final - current_acc,
                'relative_improvement': ((expected_final - current_acc) / max(current_acc, 0.01)) * 100,
                'improvement_breakdown': task_improvements
            }
        
        # 计算整体改进
        current_weighted_avg = np.mean(list(current_performance.values()))
        expected_weighted_avg = np.mean([exp['expected_accuracy'] for exp in expected_improvements.values()])
        
        overall_improvement = {
            'current_weighted_accuracy': current_weighted_avg,
            'expected_weighted_accuracy': expected_weighted_avg,
            'absolute_improvement': expected_weighted_avg - current_weighted_avg,
            'relative_improvement': ((expected_weighted_avg - current_weighted_avg) / current_weighted_avg) * 100
        }
        
        self.improvement_plan['expected_improvements'] = {
            'task_level': expected_improvements,
            'overall': overall_improvement
        }
    
    def generate_improvement_report(self):
        """生成改进建议报告"""
        print("\n📝 生成改进建议报告...")
        
        report = {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'source_bottleneck_analysis': str(self.bottleneck_file),
                'source_metrics': str(self.metrics_file),
                'total_improvement_suggestions': self._count_total_suggestions()
            },
            'executive_summary': self._generate_executive_summary(),
            'critical_fixes': self.improvement_plan.get('critical_solutions', {}),
            'improvement_categories': {
                'data_improvements': self.improvement_plan.get('data_improvements', {}),
                'model_improvements': self.improvement_plan.get('model_improvements', {}),
                'training_improvements': self.improvement_plan.get('training_improvements', {}),
                'system_improvements': self.improvement_plan.get('system_improvements', {})
            },
            'implementation_plan': self.improvement_plan.get('implementation_phases', {}),
            'expected_outcomes': self.improvement_plan.get('expected_improvements', {}),
            'risk_assessment': self._generate_risk_assessment(),
            'success_metrics': self._define_success_metrics()
        }
        
        return report
    
    def _count_total_suggestions(self):
        """统计总建议数量"""
        total = 0
        for category, improvements in self.improvement_plan.items():
            if category == 'critical_solutions':
                for task, data in improvements.items():
                    total += len(data.get('solutions', []))
            elif category in ['data_improvements', 'model_improvements', 'training_improvements']:
                if isinstance(improvements, dict):
                    for task, task_improvements in improvements.items():
                        if isinstance(task_improvements, list):
                            total += len(task_improvements)
                        elif isinstance(task_improvements, dict):
                            total += len(task_improvements)
        return total
    
    def _generate_executive_summary(self):
        """生成执行摘要"""
        critical_bottlenecks = self.bottleneck_data.get('critical_bottlenecks', [])
        expected_improvements = self.improvement_plan.get('expected_improvements', {})
        
        summary = {
            'key_findings': [
                f"识别出{len(critical_bottlenecks)}个关键性能瓶颈",
                "interference_factors任务完全失败（0%准确率）需要立即修复",
                "严重的数据不平衡问题影响模型性能",
                "多任务学习存在任务间冲突"
            ],
            'priority_actions': [
                "立即修复interference_factors任务的标签和损失函数问题",
                "实施Focal Loss处理严重的类别不平衡",
                "调整多任务学习的权重平衡",
                "增强训练监控和稳定性"
            ],
            'expected_outcomes': {
                'overall_improvement': expected_improvements.get('overall', {}).get('relative_improvement', 0),
                'timeline': '2-4周内完成主要改进',
                'confidence_level': 'high'
            }
        }
        
        return summary
    
    def _generate_risk_assessment(self):
        """生成风险评估"""
        return {
            'implementation_risks': [
                {
                    'risk': '过度调整导致其他任务性能下降',
                    'probability': 'medium',
                    'impact': 'medium',
                    'mitigation': '逐步实施改进，持续监控所有任务性能'
                },
                {
                    'risk': '数据增强可能引入噪声',
                    'probability': 'low',
                    'impact': 'low',
                    'mitigation': '仔细验证增强策略，使用验证集监控'
                },
                {
                    'risk': '模型复杂度增加导致过拟合',
                    'probability': 'medium',
                    'impact': 'medium',
                    'mitigation': '增强正则化，使用早停策略'
                }
            ],
            'overall_risk_level': 'low_to_medium'
        }
    
    def _define_success_metrics(self):
        """定义成功指标"""
        return {
            'primary_metrics': [
                'interference_factors准确率 > 60%',
                '整体加权准确率 > 92%',
                'growth_pattern准确率 > 85%'
            ],
            'secondary_metrics': [
                '训练稳定性改善（损失波动 < 5%）',
                '收敛速度提升（5-8轮内达到最佳性能）',
                '各任务性能差距 < 20%'
            ],
            'monitoring_frequency': '每轮训练后评估',
            'success_threshold': '达到80%以上的主要指标'
        }
    
    def save_improvement_suggestions(self, output_path):
        """保存改进建议"""
        print(f"\n💾 保存改进建议: {output_path}")
        
        report = self.generate_improvement_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("✅ 改进建议保存完成")
        
        return report

def main():
    """主函数"""
    print("🚀 正确GPU训练改进建议生成")
    print("=" * 50)
    
    # 文件路径
    bottleneck_file = "/home/aaa/ws/bioastModel/correct_training_bottleneck_analysis.json"
    metrics_file = "/home/aaa/ws/bioastModel/correct_training_detailed_metrics.json"
    
    # 创建生成器
    generator = ImprovementSuggestionGenerator(bottleneck_file, metrics_file)
    
    # 加载分析数据
    generator.load_analysis_data()
    
    # 生成全面改进建议
    generator.generate_comprehensive_improvements()
    
    # 保存改进建议
    output_path = "/home/aaa/ws/bioastModel/correct_training_improvement_suggestions.json"
    report = generator.save_improvement_suggestions(output_path)
    
    # 打印关键建议
    print("\n🎯 关键改进建议:")
    print("-" * 30)
    
    executive_summary = report['executive_summary']
    
    print("优先行动:")
    for i, action in enumerate(executive_summary['priority_actions'][:3], 1):
        print(f"{i}. {action}")
    
    print(f"\n预期改进:")
    expected_outcomes = executive_summary['expected_outcomes']
    print(f"• 整体性能提升: {expected_outcomes['overall_improvement']:.1f}%")
    print(f"• 完成时间: {expected_outcomes['timeline']}")
    print(f"• 信心水平: {expected_outcomes['confidence_level']}")
    
    # 打印实施阶段
    implementation_phases = report['implementation_plan']
    print(f"\n📋 实施计划:")
    for phase, details in list(implementation_phases.items())[:2]:
        print(f"• {details['description']} ({details['timeline']})")
    
    print(f"\n📊 详细建议已保存至: {output_path}")

if __name__ == "__main__":
    main()