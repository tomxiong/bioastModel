#!/usr/bin/env python3
"""
性能分析可视化总结
创建综合的可视化图表展示模型性能对比和优化建议
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class PerformanceSummaryVisualizer:
    def __init__(self):
        self.base_path = Path('/home/aaa/ws/bioastModel/analysis/performance_reports')
        self.colors = {
            'original': '#3498db',      # 蓝色
            'optimized': '#e74c3c',     # 红色  
            'simple_enhanced': '#2ecc71' # 绿色
        }
        
    def load_data(self):
        """加载分析数据"""
        with open(self.base_path / 'detailed_analysis.json', 'r') as f:
            self.detailed_data = json.load(f)
            
        with open(self.base_path / 'task_specific_analysis.json', 'r') as f:
            self.task_data = json.load(f)
    
    def create_performance_dashboard(self):
        """创建性能仪表板"""
        fig = plt.figure(figsize=(20, 16))
        
        # 创建网格布局
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. 整体准确率对比
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_overall_accuracy(ax1)
        
        # 2. 模型复杂度对比
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_model_complexity(ax2)
        
        # 3. 任务特定性能
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_task_performance(ax3)
        
        # 4. 推理时间对比
        ax4 = fig.add_subplot(gs[1, 2:])
        self.plot_inference_time(ax4)
        
        # 5. 训练收敛性
        ax5 = fig.add_subplot(gs[2, :2])
        self.plot_training_convergence(ax5)
        
        # 6. 效率分析
        ax6 = fig.add_subplot(gs[2, 2:])
        self.plot_efficiency_analysis(ax6)
        
        # 7. 优化建议总结
        ax7 = fig.add_subplot(gs[3, :])
        self.plot_optimization_summary(ax7)
        
        plt.suptitle('模型性能分析与优化建议总结', fontsize=20, fontweight='bold', y=0.98)
        
        # 保存图表
        output_path = self.base_path / 'performance_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 性能仪表板已保存: {output_path}")
        
        return fig
    
    def plot_overall_accuracy(self, ax):
        """绘制整体准确率对比"""
        models = list(self.detailed_data['performance'].keys())
        accuracies = [self.detailed_data['performance'][model]['overall_accuracy'] 
                     for model in models]
        
        bars = ax.bar(models, accuracies, color=[self.colors[model] for model in models])
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('整体准确率对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('准确率')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # 添加最佳性能标记
        best_idx = np.argmax(accuracies)
        ax.annotate('最佳性能', xy=(best_idx, accuracies[best_idx]), 
                   xytext=(best_idx, accuracies[best_idx] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, ha='center', color='red', fontweight='bold')
    
    def plot_model_complexity(self, ax):
        """绘制模型复杂度对比"""
        models = list(self.detailed_data['complexity'].keys())
        params = [self.detailed_data['complexity'][model]['params_millions'] 
                 for model in models]
        sizes = [self.detailed_data['complexity'][model]['model_size_mb'] 
                for model in models]
        
        # 创建双轴图
        ax2 = ax.twinx()
        
        bars1 = ax.bar([i-0.2 for i in range(len(models))], params, 0.4, 
                      color=[self.colors[model] for model in models], alpha=0.7, label='参数量(M)')
        bars2 = ax2.bar([i+0.2 for i in range(len(models))], sizes, 0.4, 
                       color=[self.colors[model] for model in models], alpha=0.5, label='模型大小(MB)')
        
        ax.set_title('模型复杂度对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('模型')
        ax.set_ylabel('参数量 (百万)', color='blue')
        ax2.set_ylabel('模型大小 (MB)', color='orange')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        
        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def plot_task_performance(self, ax):
        """绘制任务特定性能"""
        tasks = ['growth_level', 'growth_pattern', 'interference_factors']
        task_names = ['生长水平', '生长模式', '干扰因子']
        models = list(self.detailed_data['performance'].keys())
        
        x = np.arange(len(task_names))
        width = 0.25
        
        for i, model in enumerate(models):
            values = []
            for task in tasks:
                acc_key = f'{task}_accuracy'
                if acc_key in self.detailed_data['performance'][model]:
                    values.append(self.detailed_data['performance'][model][acc_key])
                else:
                    values.append(0)
            
            bars = ax.bar(x + i*width, values, width, label=model, 
                         color=self.colors[model], alpha=0.8)
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title('各任务性能对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('准确率')
        ax.set_xlabel('任务类型')
        ax.set_xticks(x + width)
        ax.set_xticklabels(task_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_inference_time(self, ax):
        """绘制推理时间对比"""
        models = list(self.detailed_data['complexity'].keys())
        times = [self.detailed_data['complexity'][model]['avg_inference_time_ms'] 
                for model in models]
        
        bars = ax.bar(models, times, color=[self.colors[model] for model in models])
        
        # 添加数值标签
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('推理时间对比', fontsize=14, fontweight='bold')
        ax.set_ylabel('推理时间 (毫秒)')
        ax.grid(True, alpha=0.3)
        
        # 标记最快的模型
        best_idx = np.argmin(times)
        ax.annotate('最快', xy=(best_idx, times[best_idx]), 
                   xytext=(best_idx, times[best_idx] + 2),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=12, ha='center', color='green', fontweight='bold')
    
    def plot_training_convergence(self, ax):
        """绘制训练收敛性"""
        models = list(self.detailed_data['performance'].keys())
        epochs = [self.detailed_data['performance'][model]['epochs_trained'] 
                 for model in models]
        final_loss = [self.detailed_data['performance'][model]['final_val_loss'] 
                     for model in models]
        
        # 创建散点图
        for i, model in enumerate(models):
            ax.scatter(epochs[i], final_loss[i], s=200, color=self.colors[model], 
                      label=model, alpha=0.8)
            ax.annotate(f'{model}\n({epochs[i]}轮, {final_loss[i]:.3f})', 
                       xy=(epochs[i], final_loss[i]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, ha='left')
        
        ax.set_title('训练收敛性分析', fontsize=14, fontweight='bold')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel('最终验证损失')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_efficiency_analysis(self, ax):
        """绘制效率分析"""
        models = list(self.detailed_data['performance'].keys())
        
        # 计算效率指标：准确率/参数量
        efficiency_scores = []
        for model in models:
            accuracy = self.detailed_data['performance'][model]['overall_accuracy']
            params = self.detailed_data['complexity'][model]['params_millions']
            efficiency = accuracy / params if params > 0 else 0
            efficiency_scores.append(efficiency)
        
        bars = ax.bar(models, efficiency_scores, color=[self.colors[model] for model in models])
        
        # 添加数值标签
        for bar, score in zip(bars, efficiency_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('效率分析 (准确率/参数量)', fontsize=14, fontweight='bold')
        ax.set_ylabel('效率分数')
        ax.grid(True, alpha=0.3)
        
        # 标记最高效率
        best_idx = np.argmax(efficiency_scores)
        ax.annotate('最高效率', xy=(best_idx, efficiency_scores[best_idx]), 
                   xytext=(best_idx, efficiency_scores[best_idx] + 0.05),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                   fontsize=12, ha='center', color='purple', fontweight='bold')
    
    def plot_optimization_summary(self, ax):
        """绘制优化建议总结"""
        ax.axis('off')
        
        # 创建建议框
        recommendations = [
            "🏆 生产部署: Simple Enhanced模型 (准确率: 91.61%)",
            "⚡ 边缘计算: Original模型 (最佳参数效率)",
            "🔧 立即行动: 停用复杂优化版本，建立监控系统",
            "📈 短期改进: 数据增强，模型量化，A/B测试",
            "🎯 长期策略: 专门模型，知识蒸馏，多模态融合",
            "⚠️ 关键风险: 生长模式识别准确率低，需要专门优化"
        ]
        
        # 创建彩色背景框
        colors = ['#e8f5e8', '#fff3cd', '#f8d7da', '#d4edda', '#cce5ff', '#ffe6cc']
        
        y_pos = 0.9
        for i, (rec, color) in enumerate(zip(recommendations, colors)):
            # 添加背景框
            rect = Rectangle((0.02, y_pos-0.08), 0.96, 0.12, 
                           facecolor=color, alpha=0.7, transform=ax.transAxes)
            ax.add_patch(rect)
            
            # 添加文本
            ax.text(0.05, y_pos-0.02, rec, transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', va='center')
            
            y_pos -= 0.15
        
        ax.set_title('优化建议总结', fontsize=16, fontweight='bold', pad=20)
    
    def create_model_comparison_radar(self):
        """创建模型对比雷达图"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # 定义评估维度
        categories = ['整体准确率', '参数效率', '推理速度', '训练稳定性', '生长水平', '生长模式', '干扰因子']
        N = len(categories)
        
        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形
        
        models = list(self.detailed_data['performance'].keys())
        
        for model in models:
            # 计算各维度分数 (归一化到0-1)
            perf = self.detailed_data['performance'][model]
            comp = self.detailed_data['complexity'][model]
            
            scores = [
                perf['overall_accuracy'],  # 整体准确率
                1 / comp['params_millions'] * 10,  # 参数效率 (倒数)
                1 / comp['avg_inference_time_ms'] * 100,  # 推理速度 (倒数)
                1 - perf['final_val_loss'],  # 训练稳定性
                perf.get('growth_level_accuracy', 0),  # 生长水平
                perf.get('growth_pattern_accuracy', 0),  # 生长模式
                perf.get('interference_factors_accuracy', 0)  # 干扰因子
            ]
            
            # 归一化分数
            scores = [min(max(score, 0), 1) for score in scores]
            scores += scores[:1]  # 闭合图形
            
            # 绘制雷达图
            ax.plot(angles, scores, 'o-', linewidth=2, label=model, 
                   color=self.colors[model])
            ax.fill(angles, scores, alpha=0.25, color=self.colors[model])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('模型综合性能对比雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        # 保存雷达图
        output_path = self.base_path / 'model_comparison_radar.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 雷达图已保存: {output_path}")
        
        return fig
    
    def run_visualization(self):
        """运行可视化分析"""
        print("🎨 开始创建性能分析可视化...")
        
        # 加载数据
        self.load_data()
        
        # 创建仪表板
        dashboard_fig = self.create_performance_dashboard()
        
        # 创建雷达图
        radar_fig = self.create_model_comparison_radar()
        
        print("✅ 可视化分析完成!")
        
        return dashboard_fig, radar_fig

if __name__ == "__main__":
    visualizer = PerformanceSummaryVisualizer()
    dashboard, radar = visualizer.run_visualization()
    plt.show()