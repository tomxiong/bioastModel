#!/usr/bin/env python3
"""
任务特定性能分析
分析各个任务的性能表现和失败原因
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

class TaskSpecificAnalyzer:
    def __init__(self):
        self.task_names = {
            'growth_level': '生长水平分类',
            'growth_pattern': '生长模式识别', 
            'interference_factors': '干扰因素检测'
        }
        
        self.analysis_results = {}
        
    def load_performance_data(self):
        """加载性能数据"""
        with open('/home/aaa/ws/bioastModel/analysis/performance_reports/detailed_analysis.json', 'r') as f:
            return json.load(f)
    
    def analyze_task_performance(self, data):
        """分析各任务性能"""
        print("📊 任务特定性能分析")
        print("="*60)
        
        performance_data = data['performance']
        
        # 创建任务性能对比表
        task_comparison = {}
        
        for model_name, perf in performance_data.items():
            if 'final_accuracies' in perf:
                for task, accuracy in perf['final_accuracies'].items():
                    if task not in task_comparison:
                        task_comparison[task] = {}
                    task_comparison[task][model_name] = accuracy
        
        # 分析每个任务
        for task, task_data in task_comparison.items():
            print(f"\n🎯 {self.task_names.get(task, task)} 任务分析:")
            print("-" * 40)
            
            # 排序模型性能
            sorted_models = sorted(task_data.items(), key=lambda x: x[1], reverse=True)
            
            best_model, best_acc = sorted_models[0]
            worst_model, worst_acc = sorted_models[-1]
            
            print(f"   最佳模型: {best_model.upper()} ({best_acc:.4f})")
            print(f"   最差模型: {worst_model.upper()} ({worst_acc:.4f})")
            print(f"   性能差距: {best_acc - worst_acc:.4f}")
            
            # 性能分析
            if best_acc > 0.9:
                print("   ✅ 任务表现: 优秀")
            elif best_acc > 0.8:
                print("   ⚠️  任务表现: 良好")
            elif best_acc > 0.7:
                print("   ⚠️  任务表现: 一般")
            else:
                print("   ❌ 任务表现: 需要改进")
            
            # 存储分析结果
            self.analysis_results[task] = {
                'best_model': best_model,
                'best_accuracy': best_acc,
                'worst_model': worst_model,
                'worst_accuracy': worst_acc,
                'performance_gap': best_acc - worst_acc,
                'all_results': task_data
            }
        
        return task_comparison
    
    def identify_failure_patterns(self, task_comparison):
        """识别失败模式"""
        print(f"\n🔍 失败模式分析")
        print("-" * 40)
        
        failure_analysis = {}
        
        for task, results in task_comparison.items():
            print(f"\n📋 {self.task_names.get(task, task)}:")
            
            # 找出表现最差的模型
            worst_performance = min(results.items(), key=lambda x: x[1])
            model_name, accuracy = worst_performance
            
            if accuracy < 0.5:
                failure_type = "严重失败"
                print(f"   ❌ {model_name}: {failure_type} ({accuracy:.4f})")
                
                # 分析可能原因
                if model_name == 'optimized':
                    reasons = [
                        "复杂损失函数导致训练不稳定",
                        "自适应权重机制过于激进",
                        "多种优化技术相互干扰",
                        "过拟合到训练数据"
                    ]
                else:
                    reasons = ["数据不平衡", "特征提取不充分", "模型容量不足"]
                    
            elif accuracy < 0.7:
                failure_type = "性能不佳"
                print(f"   ⚠️  {model_name}: {failure_type} ({accuracy:.4f})")
                reasons = ["需要更多训练数据", "特征工程优化", "超参数调整"]
            else:
                failure_type = "表现正常"
                print(f"   ✅ {model_name}: {failure_type} ({accuracy:.4f})")
                reasons = []
            
            failure_analysis[task] = {
                'worst_model': model_name,
                'failure_type': failure_type,
                'accuracy': accuracy,
                'possible_reasons': reasons
            }
            
            if reasons:
                print("      可能原因:")
                for reason in reasons:
                    print(f"        - {reason}")
        
        return failure_analysis
    
    def analyze_model_strengths_weaknesses(self, task_comparison):
        """分析各模型的优势和劣势"""
        print(f"\n💪 模型优势劣势分析")
        print("-" * 40)
        
        model_analysis = {}
        
        # 收集每个模型在各任务上的表现
        for model in ['original', 'optimized', 'simple_enhanced']:
            if model in ['original', 'optimized']:  # 只分析有完整任务数据的模型
                model_performance = {}
                for task, results in task_comparison.items():
                    if model in results:
                        model_performance[task] = results[model]
                
                if model_performance:
                    # 找出最强和最弱的任务
                    best_task = max(model_performance.items(), key=lambda x: x[1])
                    worst_task = min(model_performance.items(), key=lambda x: x[1])
                    avg_performance = np.mean(list(model_performance.values()))
                    
                    print(f"\n🤖 {model.upper()} 模型:")
                    print(f"   平均性能: {avg_performance:.4f}")
                    print(f"   最强任务: {self.task_names.get(best_task[0], best_task[0])} ({best_task[1]:.4f})")
                    print(f"   最弱任务: {self.task_names.get(worst_task[0], worst_task[0])} ({worst_task[1]:.4f})")
                    
                    # 分析优势和劣势
                    strengths = []
                    weaknesses = []
                    
                    for task, acc in model_performance.items():
                        if acc > 0.9:
                            strengths.append(f"{self.task_names.get(task, task)} (优秀)")
                        elif acc < 0.7:
                            weaknesses.append(f"{self.task_names.get(task, task)} (需改进)")
                    
                    if strengths:
                        print("   优势:")
                        for strength in strengths:
                            print(f"     ✅ {strength}")
                    
                    if weaknesses:
                        print("   劣势:")
                        for weakness in weaknesses:
                            print(f"     ❌ {weakness}")
                    
                    model_analysis[model] = {
                        'avg_performance': avg_performance,
                        'best_task': best_task,
                        'worst_task': worst_task,
                        'strengths': strengths,
                        'weaknesses': weaknesses
                    }
        
        return model_analysis
    
    def generate_improvement_suggestions(self, failure_analysis, model_analysis):
        """生成改进建议"""
        print(f"\n🚀 改进建议")
        print("-" * 40)
        
        suggestions = {}
        
        # 基于任务失败分析的建议
        print("\n📈 任务特定改进建议:")
        
        for task, analysis in failure_analysis.items():
            task_name = self.task_names.get(task, task)
            print(f"\n🎯 {task_name}:")
            
            if analysis['failure_type'] == "严重失败":
                suggestions[task] = [
                    "重新设计损失函数，使用更稳定的交叉熵损失",
                    "增加该任务的训练数据",
                    "使用数据增强技术",
                    "调整类别权重处理数据不平衡",
                    "简化模型架构避免过拟合"
                ]
            elif analysis['failure_type'] == "性能不佳":
                suggestions[task] = [
                    "增加特征工程",
                    "调整学习率和优化器",
                    "使用预训练权重",
                    "增加正则化"
                ]
            else:
                suggestions[task] = [
                    "微调超参数进一步优化",
                    "尝试集成学习方法"
                ]
            
            for suggestion in suggestions[task]:
                print(f"     - {suggestion}")
        
        # 基于模型分析的建议
        print(f"\n🤖 模型特定改进建议:")
        
        for model, analysis in model_analysis.items():
            print(f"\n{model.upper()} 模型:")
            
            if model == 'optimized' and analysis['avg_performance'] < 0.7:
                print("     - 简化优化策略，避免过度工程化")
                print("     - 使用单一稳定的损失函数")
                print("     - 移除自适应权重机制")
                print("     - 增加训练轮数确保收敛")
            
            elif model == 'original':
                print("     - 保持当前稳定的训练策略")
                print("     - 可以尝试轻微的特征增强")
                print("     - 考虑增加模型容量")
            
            # 针对弱势任务的建议
            if analysis['weaknesses']:
                print("     针对弱势任务:")
                for weakness in analysis['weaknesses']:
                    if '生长模式' in weakness:
                        print("       - 增加生长模式相关的特征提取")
                        print("       - 使用注意力机制关注关键区域")
                    elif '干扰因素' in weakness:
                        print("       - 改进多标签分类策略")
                        print("       - 使用焦点损失处理类别不平衡")
        
        return suggestions
    
    def create_visualization(self, task_comparison):
        """创建可视化图表"""
        print(f"\n📊 生成可视化图表...")
        
        # 创建输出目录
        output_dir = Path('/home/aaa/ws/bioastModel/analysis/performance_reports')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        df_data = []
        for task, results in task_comparison.items():
            for model, accuracy in results.items():
                df_data.append({
                    'Task': self.task_names.get(task, task),
                    'Model': model.upper(),
                    'Accuracy': accuracy
                })
        
        df = pd.DataFrame(df_data)
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 子图1: 任务性能对比
        plt.subplot(2, 2, 1)
        pivot_df = df.pivot(index='Task', columns='Model', values='Accuracy')
        pivot_df.plot(kind='bar', ax=plt.gca())
        plt.title('各任务性能对比')
        plt.ylabel('准确率')
        plt.xticks(rotation=45)
        plt.legend(title='模型')
        
        # 子图2: 模型整体性能
        plt.subplot(2, 2, 2)
        model_avg = df.groupby('Model')['Accuracy'].mean()
        model_avg.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('模型平均性能')
        plt.ylabel('平均准确率')
        plt.xticks(rotation=45)
        
        # 子图3: 性能热图
        plt.subplot(2, 2, 3)
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r')
        plt.title('性能热图')
        
        # 子图4: 任务难度分析
        plt.subplot(2, 2, 4)
        task_difficulty = df.groupby('Task')['Accuracy'].agg(['mean', 'std'])
        task_difficulty['mean'].plot(kind='bar', yerr=task_difficulty['std'], 
                                   color='orange', alpha=0.7)
        plt.title('任务难度分析')
        plt.ylabel('准确率 (均值±标准差)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'task_specific_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 图表已保存: {output_dir / 'task_specific_analysis.png'}")
    
    def run_analysis(self):
        """运行完整的任务特定分析"""
        print("🚀 开始任务特定分析...")
        
        # 加载数据
        data = self.load_performance_data()
        
        # 分析任务性能
        task_comparison = self.analyze_task_performance(data)
        
        # 识别失败模式
        failure_analysis = self.identify_failure_patterns(task_comparison)
        
        # 分析模型优势劣势
        model_analysis = self.analyze_model_strengths_weaknesses(task_comparison)
        
        # 生成改进建议
        suggestions = self.generate_improvement_suggestions(failure_analysis, model_analysis)
        
        # 创建可视化
        self.create_visualization(task_comparison)
        
        # 保存分析结果
        analysis_results = {
            'task_comparison': task_comparison,
            'failure_analysis': failure_analysis,
            'model_analysis': model_analysis,
            'improvement_suggestions': suggestions
        }
        
        output_path = Path('/home/aaa/ws/bioastModel/analysis/performance_reports/task_specific_analysis.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 任务分析结果已保存: {output_path}")
        print("✅ 任务特定分析完成!")
        
        return analysis_results

if __name__ == "__main__":
    analyzer = TaskSpecificAnalyzer()
    results = analyzer.run_analysis()