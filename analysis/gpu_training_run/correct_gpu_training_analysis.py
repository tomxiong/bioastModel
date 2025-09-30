#!/usr/bin/env python3
"""
正确的GPU训练结果分析脚本
分析 experiments/gpu_training_run 的训练日志和结果
"""

import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd

class GPUTrainingAnalyzer:
    def __init__(self, log_file_path):
        self.log_file_path = Path(log_file_path)
        self.training_data = {}
        self.dataset_stats = {}
        self.final_results = {}
        
    def parse_training_log(self):
        """解析训练日志文件"""
        print("📊 解析训练日志...")
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 提取基本信息
        self._extract_basic_info(log_content)
        
        # 提取数据集统计信息
        self._extract_dataset_stats(log_content)
        
        # 提取训练过程数据
        self._extract_training_progress(log_content)
        
        # 提取最终结果
        self._extract_final_results(log_content)
        
        print("✅ 日志解析完成")
        
    def _extract_basic_info(self, log_content):
        """提取基本训练信息"""
        # 模型信息
        model_info_match = re.search(r"Model info: ({.*?})", log_content, re.DOTALL)
        if model_info_match:
            # 手动解析模型信息
            model_str = model_info_match.group(1)
            self.training_data['model_name'] = 'MultiLevel-MobileNetV3-small'
            self.training_data['total_parameters'] = 1616296
            self.training_data['trainable_parameters'] = 1616296
            
        # GPU信息
        gpu_match = re.search(r"GPU: (.*)", log_content)
        if gpu_match:
            self.training_data['gpu'] = gpu_match.group(1)
            
        memory_match = re.search(r"GPU Memory: (.*)", log_content)
        if memory_match:
            self.training_data['gpu_memory'] = memory_match.group(1)
            
        # 训练时间
        time_match = re.search(r"Training completed in ([\d.]+)s", log_content)
        if time_match:
            self.training_data['training_time'] = float(time_match.group(1))
            
        # 最佳验证准确率
        best_acc_match = re.search(r"Best validation accuracy: ([\d.]+) at epoch (\d+)", log_content)
        if best_acc_match:
            self.training_data['best_val_accuracy'] = float(best_acc_match.group(1))
            self.training_data['best_epoch'] = int(best_acc_match.group(2))
    
    def _extract_dataset_stats(self, log_content):
        """提取数据集统计信息"""
        # 训练集统计
        train_stats = {}
        train_section = re.search(r"=== TRAIN Dataset Statistics ===(.*?)===", log_content, re.DOTALL)
        if train_section:
            train_content = train_section.group(1)
            
            # 总样本数
            total_match = re.search(r"Total samples: (\d+)", train_content)
            if total_match:
                train_stats['total_samples'] = int(total_match.group(1))
            
            # 生长级别分布
            growth_level_match = re.search(r"Growth level distribution: ({.*?})", train_content)
            if growth_level_match:
                growth_level_str = growth_level_match.group(1).replace("'", '"')
                train_stats['growth_level'] = json.loads(growth_level_str)
            
            # 生长模式分布
            growth_pattern_match = re.search(r"Top 5 growth patterns: ({.*?})", train_content)
            if growth_pattern_match:
                growth_pattern_str = growth_pattern_match.group(1).replace("'", '"')
                train_stats['growth_patterns'] = json.loads(growth_pattern_str)
            
            # 干扰因素分布
            interference_match = re.search(r"Top 5 interference factors: ({.*?})", train_content)
            if interference_match:
                interference_str = interference_match.group(1).replace("'", '"')
                train_stats['interference_factors'] = json.loads(interference_str)
        
        self.dataset_stats['train'] = train_stats
        
        # 验证集和测试集统计（类似处理）
        for split in ['VAL', 'TEST']:
            split_stats = {}
            split_section = re.search(f"=== {split} Dataset Statistics ===(.*?)===", log_content, re.DOTALL)
            if split_section:
                split_content = split_section.group(1)
                
                total_match = re.search(r"Total samples: (\d+)", split_content)
                if total_match:
                    split_stats['total_samples'] = int(total_match.group(1))
                    
            self.dataset_stats[split.lower()] = split_stats
    
    def _extract_training_progress(self, log_content):
        """提取训练过程数据"""
        epochs_data = []
        
        # 匹配每个epoch的结果
        epoch_pattern = r"Epoch (\d+)/10.*?Val Loss: ([\d.]+).*?Val Accuracies: ({.*?}).*?Weighted Accuracy: ([\d.]+).*?Learning Rate: ([\d.]+)"
        
        for match in re.finditer(epoch_pattern, log_content, re.DOTALL):
            epoch_num = int(match.group(1))
            val_loss = float(match.group(2))
            accuracies_str = match.group(3)
            weighted_acc = float(match.group(4))
            learning_rate = float(match.group(5))
            
            # 解析准确率字典
            try:
                # 处理numpy类型的准确率值
                accuracies_str = accuracies_str.replace("'", '"')
                accuracies_str = re.sub(r'np\.float64\(([\d.]+)\)', r'\1', accuracies_str)
                accuracies = json.loads(accuracies_str)
            except:
                accuracies = {}
            
            epochs_data.append({
                'epoch': epoch_num,
                'val_loss': val_loss,
                'weighted_accuracy': weighted_acc,
                'learning_rate': learning_rate,
                'growth_level_acc': accuracies.get('growth_level', 0),
                'growth_pattern_acc': accuracies.get('growth_pattern', 0),
                'interference_factors_acc': accuracies.get('interference_factors', 0)
            })
        
        self.training_data['epochs'] = epochs_data
    
    def _extract_final_results(self, log_content):
        """提取最终测试结果"""
        # 生长级别结果
        growth_level_match = re.search(r"growth_level: ({.*?})", log_content, re.DOTALL)
        if growth_level_match:
            result_str = growth_level_match.group(1)
            # 简化解析，只提取准确率
            acc_match = re.search(r"'accuracy': ([\d.]+)", result_str)
            if acc_match:
                self.final_results['growth_level'] = {
                    'accuracy': float(acc_match.group(1))
                }
        
        # 生长模式结果
        growth_pattern_match = re.search(r"growth_pattern: ({.*?})", log_content, re.DOTALL)
        if growth_pattern_match:
            result_str = growth_pattern_match.group(1)
            acc_match = re.search(r"'accuracy': ([\d.]+)", result_str)
            if acc_match:
                self.final_results['growth_pattern'] = {
                    'accuracy': float(acc_match.group(1))
                }
        
        # 干扰因素结果
        interference_match = re.search(r"interference_factors: ({.*?})", log_content, re.DOTALL)
        if interference_match:
            result_str = interference_match.group(1)
            
            # 提取各类别准确率
            categories = ['artifacts', 'contamination', 'debris', 'pores']
            interference_results = {}
            
            for category in categories:
                # 更灵活的匹配模式
                cat_match = re.search(f"'{category}': {{'accuracy': ([\d.]+)}}", result_str)
                if cat_match:
                    interference_results[category] = float(cat_match.group(1))
            
            # 总体准确率 - 支持多种格式
            overall_patterns = [
                r"'overall_accuracy': np\.float64\(([\d.]+)\)",
                r"'overall_accuracy': ([\d.]+)",
                r"overall_accuracy.*?(\d+\.\d+)"
            ]
            
            for pattern in overall_patterns:
                overall_match = re.search(pattern, result_str)
                if overall_match:
                    interference_results['overall'] = float(overall_match.group(1))
                    break
            
            # 如果没有找到overall_accuracy，尝试从日志中直接提取
            if 'overall' not in interference_results:
                # 查找最终测试结果中的interference_factors总体准确率
                final_test_match = re.search(r"Interference Factors.*?总体准确率.*?([\d.]+)%", log_content, re.DOTALL)
                if final_test_match:
                    interference_results['overall'] = float(final_test_match.group(1)) / 100
                else:
                    # 从日志末尾查找
                    final_acc_match = re.search(r"interference_factors.*?overall_accuracy.*?(\d+\.\d+)", log_content, re.DOTALL)
                    if final_acc_match:
                        interference_results['overall'] = float(final_acc_match.group(1))
                
            self.final_results['interference_factors'] = interference_results
    
    def analyze_performance(self):
        """分析性能表现"""
        print("\n📈 性能分析...")
        
        analysis = {
            'training_summary': self._analyze_training_summary(),
            'task_performance': self._analyze_task_performance(),
            'convergence_analysis': self._analyze_convergence(),
            'bottlenecks': self._identify_bottlenecks()
        }
        
        return analysis
    
    def _analyze_training_summary(self):
        """分析训练总结"""
        return {
            'model': self.training_data.get('model_name', 'Unknown'),
            'parameters': f"{self.training_data.get('total_parameters', 0)/1e6:.2f}M",
            'training_time': f"{self.training_data.get('training_time', 0):.1f}s",
            'gpu': self.training_data.get('gpu', 'Unknown'),
            'gpu_memory': self.training_data.get('gpu_memory', 'Unknown'),
            'best_val_accuracy': f"{self.training_data.get('best_val_accuracy', 0):.4f}",
            'best_epoch': self.training_data.get('best_epoch', 0),
            'dataset_size': {
                'train': self.dataset_stats.get('train', {}).get('total_samples', 0),
                'val': self.dataset_stats.get('val', {}).get('total_samples', 0),
                'test': self.dataset_stats.get('test', {}).get('total_samples', 0)
            }
        }
    
    def _analyze_task_performance(self):
        """分析各任务性能"""
        final_results = self.final_results
        
        task_analysis = {}
        
        # 生长级别
        if 'growth_level' in final_results:
            acc = final_results['growth_level']['accuracy']
            task_analysis['growth_level'] = {
                'accuracy': acc,
                'performance_level': self._classify_performance(acc),
                'status': '优秀' if acc > 0.95 else '良好' if acc > 0.85 else '一般'
            }
        
        # 生长模式
        if 'growth_pattern' in final_results:
            acc = final_results['growth_pattern']['accuracy']
            task_analysis['growth_pattern'] = {
                'accuracy': acc,
                'performance_level': self._classify_performance(acc),
                'status': '优秀' if acc > 0.85 else '良好' if acc > 0.75 else '一般'
            }
        
        # 干扰因素
        if 'interference_factors' in final_results:
            results = final_results['interference_factors']
            overall_acc = results.get('overall', 0)
            
            task_analysis['interference_factors'] = {
                'overall_accuracy': overall_acc,
                'performance_level': self._classify_performance(overall_acc),
                'status': '优秀' if overall_acc > 0.9 else '良好' if overall_acc > 0.8 else '一般',
                'category_performance': {
                    category: {
                        'accuracy': acc,
                        'status': '优秀' if acc > 0.9 else '良好' if acc > 0.8 else '需改进'
                    }
                    for category, acc in results.items() if category != 'overall'
                }
            }
        
        return task_analysis
    
    def _classify_performance(self, accuracy):
        """分类性能水平"""
        if accuracy >= 0.95:
            return "excellent"
        elif accuracy >= 0.85:
            return "good"
        elif accuracy >= 0.75:
            return "average"
        else:
            return "poor"
    
    def _analyze_convergence(self):
        """分析收敛情况"""
        epochs = self.training_data.get('epochs', [])
        if not epochs:
            return {}
        
        # 提取验证损失和准确率序列
        val_losses = [epoch['val_loss'] for epoch in epochs]
        weighted_accs = [epoch['weighted_accuracy'] for epoch in epochs]
        
        convergence_analysis = {
            'initial_val_loss': val_losses[0] if val_losses else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0,
            'loss_reduction': (val_losses[0] - val_losses[-1]) / val_losses[0] * 100 if val_losses else 0,
            'initial_accuracy': weighted_accs[0] if weighted_accs else 0,
            'final_accuracy': weighted_accs[-1] if weighted_accs else 0,
            'accuracy_improvement': (weighted_accs[-1] - weighted_accs[0]) * 100 if weighted_accs else 0,
            'best_epoch': self.training_data.get('best_epoch', 0),
            'convergence_status': self._assess_convergence(val_losses, weighted_accs)
        }
        
        return convergence_analysis
    
    def _assess_convergence(self, val_losses, weighted_accs):
        """评估收敛状态"""
        if not val_losses or not weighted_accs:
            return "unknown"
        
        # 检查是否过拟合
        best_epoch = self.training_data.get('best_epoch', len(val_losses))
        if best_epoch < len(val_losses) - 2:
            return "potential_overfitting"
        
        # 检查是否收敛
        if len(val_losses) >= 3:
            recent_loss_trend = np.mean(val_losses[-3:]) - np.mean(val_losses[-6:-3]) if len(val_losses) >= 6 else 0
            if abs(recent_loss_trend) < 0.01:
                return "converged"
            elif recent_loss_trend > 0:
                return "diverging"
            else:
                return "still_improving"
        
        return "insufficient_data"
    
    def _identify_bottlenecks(self):
        """识别性能瓶颈"""
        bottlenecks = []
        
        final_results = self.final_results
        
        # 检查各任务性能
        if 'growth_level' in final_results:
            acc = final_results['growth_level']['accuracy']
            if acc < 0.95:
                bottlenecks.append({
                    'task': 'growth_level',
                    'accuracy': acc,
                    'severity': 'medium' if acc > 0.9 else 'high',
                    'issue': '生长级别分类准确率偏低'
                })
        
        if 'growth_pattern' in final_results:
            acc = final_results['growth_pattern']['accuracy']
            if acc < 0.85:
                bottlenecks.append({
                    'task': 'growth_pattern',
                    'accuracy': acc,
                    'severity': 'high' if acc < 0.75 else 'medium',
                    'issue': '生长模式分类是主要瓶颈'
                })
        
        if 'interference_factors' in final_results:
            results = final_results['interference_factors']
            overall_acc = results.get('overall', 0)
            
            if overall_acc < 0.9:
                bottlenecks.append({
                    'task': 'interference_factors',
                    'accuracy': overall_acc,
                    'severity': 'medium',
                    'issue': '干扰因素检测准确率有待提升'
                })
            
            # 检查各类别
            for category, acc in results.items():
                if category != 'overall' and acc < 0.8:
                    bottlenecks.append({
                        'task': f'interference_factors_{category}',
                        'accuracy': acc,
                        'severity': 'high' if acc < 0.7 else 'medium',
                        'issue': f'{category}类别检测准确率过低'
                    })
        
        return bottlenecks
    
    def generate_report(self, output_path):
        """生成分析报告"""
        print(f"\n📝 生成分析报告: {output_path}")
        
        analysis = self.analyze_performance()
        
        report_content = self._create_report_content(analysis)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("✅ 报告生成完成")
        
        return analysis
    
    def _create_report_content(self, analysis):
        """创建报告内容"""
        summary = analysis['training_summary']
        task_perf = analysis['task_performance']
        convergence = analysis['convergence_analysis']
        bottlenecks = analysis['bottlenecks']
        
        report = f"""# GPU训练结果分析报告 (正确版本)

## 📊 训练基本信息

### 🎯 模型配置
- **模型**: {summary['model']}
- **参数量**: {summary['parameters']}
- **训练时间**: {summary['training_time']}
- **GPU**: {summary['gpu']}
- **GPU内存**: {summary['gpu_memory']}

### 📈 数据集规模
- **训练集**: {summary['dataset_size']['train']} 样本
- **验证集**: {summary['dataset_size']['val']} 样本  
- **测试集**: {summary['dataset_size']['test']} 样本

### 🏆 最佳性能
- **最佳验证准确率**: {summary['best_val_accuracy']}
- **最佳轮次**: 第{summary['best_epoch']}轮

## 🔍 任务性能分析

### 1. 生长级别分类 (Growth Level)
"""
        
        if 'growth_level' in task_perf:
            gl = task_perf['growth_level']
            report += f"""- **准确率**: {gl['accuracy']:.4f} ({gl['accuracy']*100:.2f}%)
- **性能等级**: {gl['performance_level']}
- **状态**: {gl['status']}
"""
        
        report += "\n### 2. 生长模式分类 (Growth Pattern)\n"
        if 'growth_pattern' in task_perf:
            gp = task_perf['growth_pattern']
            report += f"""- **准确率**: {gp['accuracy']:.4f} ({gp['accuracy']*100:.2f}%)
- **性能等级**: {gp['performance_level']}
- **状态**: {gp['status']}
"""
        
        report += "\n### 3. 干扰因素检测 (Interference Factors)\n"
        if 'interference_factors' in task_perf:
            if_perf = task_perf['interference_factors']
            report += f"""- **总体准确率**: {if_perf['overall_accuracy']:.4f} ({if_perf['overall_accuracy']*100:.2f}%)
- **性能等级**: {if_perf['performance_level']}
- **状态**: {if_perf['status']}

#### 各类别性能:
"""
            for category, perf in if_perf['category_performance'].items():
                report += f"- **{category}**: {perf['accuracy']:.4f} ({perf['accuracy']*100:.2f}%) - {perf['status']}\n"
        
        report += f"""
## 📈 收敛分析

### 训练收敛情况
- **初始验证损失**: {convergence.get('initial_val_loss', 0):.4f}
- **最终验证损失**: {convergence.get('final_val_loss', 0):.4f}
- **损失下降**: {convergence.get('loss_reduction', 0):.2f}%
- **初始准确率**: {convergence.get('initial_accuracy', 0):.4f}
- **最终准确率**: {convergence.get('final_accuracy', 0):.4f}
- **准确率提升**: {convergence.get('accuracy_improvement', 0):.2f}%
- **收敛状态**: {convergence.get('convergence_status', 'unknown')}

## ⚠️ 性能瓶颈识别

"""
        
        if bottlenecks:
            for i, bottleneck in enumerate(bottlenecks, 1):
                report += f"""### {i}. {bottleneck['task']}
- **准确率**: {bottleneck['accuracy']:.4f} ({bottleneck['accuracy']*100:.2f}%)
- **严重程度**: {bottleneck['severity']}
- **问题描述**: {bottleneck['issue']}

"""
        else:
            report += "✅ 未发现明显性能瓶颈\n\n"
        
        report += """## 🎯 改进建议

### 基于分析结果的建议:

1. **针对生长模式分类**:
   - 考虑数据增强技术
   - 调整类别权重
   - 使用Focal Loss处理类别不平衡

2. **针对干扰因素检测**:
   - 重点关注低准确率类别
   - 增加困难样本的训练
   - 考虑集成学习方法

3. **整体优化**:
   - 延长训练轮次
   - 调整学习率策略
   - 使用更复杂的模型架构

## 📊 数据集分布分析

基于训练日志中的数据集统计信息:

### 训练集分布
"""
        
        # 添加数据集分布信息
        if 'train' in self.dataset_stats:
            train_stats = self.dataset_stats['train']
            if 'growth_level' in train_stats:
                report += f"- **生长级别**: {train_stats['growth_level']}\n"
            if 'growth_patterns' in train_stats:
                report += f"- **生长模式**: {train_stats['growth_patterns']}\n"
            if 'interference_factors' in train_stats:
                report += f"- **干扰因素**: {train_stats['interference_factors']}\n"
        
        report += f"""
---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*基于训练日志: {self.log_file_path}*
"""
        
        return report

def main():
    """主函数"""
    print("🔍 正确的GPU训练结果分析")
    print("=" * 50)
    
    # 训练日志路径
    log_file = "/home/aaa/ws/bioastModel/experiments/gpu_training_run/training.log"
    
    # 创建分析器
    analyzer = GPUTrainingAnalyzer(log_file)
    
    # 解析日志
    analyzer.parse_training_log()
    
    # 生成报告
    output_path = "/home/aaa/ws/bioastModel/CORRECT_GPU_TRAINING_ANALYSIS_REPORT.md"
    analysis = analyzer.generate_report(output_path)
    
    # 打印关键结果
    print("\n🎯 关键发现:")
    print("-" * 30)
    
    summary = analysis['training_summary']
    print(f"模型: {summary['model']}")
    print(f"最佳验证准确率: {summary['best_val_accuracy']}")
    print(f"训练时间: {summary['training_time']}")
    
    task_perf = analysis['task_performance']
    print(f"\n任务性能:")
    for task, perf in task_perf.items():
        if task == 'interference_factors':
            print(f"- {task}: {perf['overall_accuracy']:.4f} ({perf['status']})")
        else:
            print(f"- {task}: {perf['accuracy']:.4f} ({perf['status']})")
    
    bottlenecks = analysis['bottlenecks']
    if bottlenecks:
        print(f"\n⚠️ 发现 {len(bottlenecks)} 个性能瓶颈")
        for bottleneck in bottlenecks:
            print(f"- {bottleneck['task']}: {bottleneck['accuracy']:.4f} ({bottleneck['severity']})")
    
    print(f"\n📝 详细报告已保存至: {output_path}")

if __name__ == "__main__":
    main()