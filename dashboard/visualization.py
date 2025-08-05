"""可视化组件

提供各种图表和可视化功能。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class Visualizer:
    """可视化器"""
    
    def __init__(self, output_dir: str = "reports/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色主题
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6C5CE7',
            'light': '#DDD6FE',
            'dark': '#2D3748'
        }
        
        self.color_palette = list(self.colors.values())
    
    def plot_training_curves(self, 
                           experiment_data: Dict[str, Any],
                           save_path: Optional[str] = None) -> str:
        """绘制训练曲线"""
        metrics = experiment_data.get('metrics', {})
        train_losses = metrics.get('train_losses', [])
        val_losses = metrics.get('val_losses', [])
        train_accuracies = metrics.get('train_accuracies', [])
        val_accuracies = metrics.get('val_accuracies', [])
        
        if not any([train_losses, val_losses, train_accuracies, val_accuracies]):
            return ""
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"训练曲线 - {experiment_data.get('name', 'Unknown')}", fontsize=16)
        
        epochs = range(1, len(train_losses) + 1) if train_losses else []
        
        # 损失曲线
        if train_losses and val_losses:
            ax1.plot(epochs, train_losses, label='训练损失', color=self.colors['primary'], linewidth=2)
            ax1.plot(epochs, val_losses, label='验证损失', color=self.colors['secondary'], linewidth=2)
            ax1.set_title('损失曲线')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        if train_accuracies and val_accuracies:
            ax2.plot(epochs, train_accuracies, label='训练准确率', color=self.colors['success'], linewidth=2)
            ax2.plot(epochs, val_accuracies, label='验证准确率', color=self.colors['warning'], linewidth=2)
            ax2.set_title('准确率曲线')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 学习率曲线
        learning_rates = metrics.get('learning_rates', [])
        if learning_rates:
            ax3.plot(epochs, learning_rates, color=self.colors['info'], linewidth=2)
            ax3.set_title('学习率变化')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # 训练时间
        epoch_times = metrics.get('epoch_times', [])
        if epoch_times:
            ax4.bar(epochs, epoch_times, color=self.colors['light'], alpha=0.7)
            ax4.set_title('每轮训练时间')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Time (seconds)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / f"training_curves_{experiment_data.get('experiment_id', 'unknown')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_model_comparison(self, 
                            experiments: List[Dict[str, Any]],
                            metric: str = 'best_val_accuracy',
                            save_path: Optional[str] = None) -> str:
        """绘制模型对比图"""
        if not experiments:
            return ""
        
        # 准备数据
        model_names = [exp.get('model_name', 'Unknown') for exp in experiments]
        values = [exp.get(metric, 0) for exp in experiments]
        experiment_names = [exp.get('name', f"Exp-{i+1}") for i, exp in enumerate(experiments)]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 柱状图
        bars = ax1.bar(range(len(values)), values, color=self.color_palette[:len(values)])
        ax1.set_title(f'模型性能对比 ({metric})')
        ax1.set_xlabel('实验')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_xticks(range(len(experiment_names)))
        ax1.set_xticklabels(experiment_names, rotation=45, ha='right')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom')
        
        ax1.grid(True, alpha=0.3)
        
        # 按模型分组的箱线图
        model_data = {}
        for exp in experiments:
            model = exp.get('model_name', 'Unknown')
            if model not in model_data:
                model_data[model] = []
            model_data[model].append(exp.get(metric, 0))
        
        if len(model_data) > 1:
            ax2.boxplot(model_data.values(), labels=model_data.keys())
            ax2.set_title(f'模型性能分布 ({metric})')
            ax2.set_ylabel(metric.replace('_', ' ').title())
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '需要多个模型进行对比', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('模型性能分布')
        
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / f"model_comparison_{metric}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_performance_timeline(self, 
                                experiments: List[Dict[str, Any]],
                                save_path: Optional[str] = None) -> str:
        """绘制性能时间线"""
        if not experiments:
            return ""
        
        # 准备数据
        df_data = []
        for exp in experiments:
            if exp.get('completed_at'):
                df_data.append({
                    'date': datetime.fromisoformat(exp['completed_at']),
                    'accuracy': exp.get('best_val_accuracy', 0),
                    'loss': exp.get('best_val_loss', 0),
                    'model': exp.get('model_name', 'Unknown'),
                    'experiment': exp.get('name', 'Unknown')
                })
        
        if not df_data:
            return ""
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('date')
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 准确率时间线
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax1.plot(model_data['date'], model_data['accuracy'], 
                    marker='o', label=model, linewidth=2, markersize=6)
        
        ax1.set_title('模型准确率时间线')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('验证准确率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 损失时间线
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax2.plot(model_data['date'], model_data['loss'], 
                    marker='s', label=model, linewidth=2, markersize=6)
        
        ax2.set_title('模型损失时间线')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('验证损失')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / "performance_timeline.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_interactive_dashboard(self, 
                                   experiments: List[Dict[str, Any]],
                                   save_path: Optional[str] = None) -> str:
        """创建交互式仪表板"""
        if not experiments:
            return ""
        
        # 准备数据
        df_data = []
        for exp in experiments:
            df_data.append({
                'experiment_id': exp.get('experiment_id', ''),
                'name': exp.get('name', 'Unknown'),
                'model': exp.get('model_name', 'Unknown'),
                'status': exp.get('status', 'unknown'),
                'accuracy': exp.get('best_val_accuracy', 0),
                'loss': exp.get('best_val_loss', 0),
                'epochs': exp.get('total_epochs', 0),
                'duration': exp.get('duration_seconds', 0),
                'created_at': exp.get('created_at', ''),
                'batch_size': exp.get('batch_size', 0),
                'learning_rate': exp.get('learning_rate', 0)
            })
        
        df = pd.DataFrame(df_data)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('模型准确率分布', '训练时长 vs 准确率', '模型状态分布', '超参数关系'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. 模型准确率分布
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data.index,
                    y=model_data['accuracy'],
                    mode='markers+lines',
                    name=f'{model} 准确率',
                    text=model_data['name'],
                    hovertemplate='<b>%{text}</b><br>准确率: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. 训练时长 vs 准确率
        fig.add_trace(
            go.Scatter(
                x=df['duration'],
                y=df['accuracy'],
                mode='markers',
                text=df['name'],
                marker=dict(
                    size=df['epochs'],
                    sizemode='diameter',
                    sizeref=2.*max(df['epochs'])/(40.**2),
                    sizemin=4,
                    color=df['loss'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="验证损失")
                ),
                hovertemplate='<b>%{text}</b><br>时长: %{x:.1f}s<br>准确率: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. 模型状态分布
        status_counts = df['status'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                name="状态分布"
            ),
            row=2, col=1
        )
        
        # 4. 超参数关系 (学习率 vs 准确率)
        fig.add_trace(
            go.Scatter(
                x=df['learning_rate'],
                y=df['accuracy'],
                mode='markers',
                text=df['name'],
                marker=dict(
                    size=df['batch_size']/4,
                    color=df['model'],
                    showscale=False
                ),
                hovertemplate='<b>%{text}</b><br>学习率: %{x}<br>准确率: %{y:.4f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text="实验仪表板",
            showlegend=True,
            height=800
        )
        
        # 更新坐标轴标签
        fig.update_xaxes(title_text="实验序号", row=1, col=1)
        fig.update_yaxes(title_text="准确率", row=1, col=1)
        fig.update_xaxes(title_text="训练时长 (秒)", row=1, col=2)
        fig.update_yaxes(title_text="准确率", row=1, col=2)
        fig.update_xaxes(title_text="学习率", row=2, col=2)
        fig.update_yaxes(title_text="准确率", row=2, col=2)
        
        if not save_path:
            save_path = self.output_dir / "interactive_dashboard.html"
        
        fig.write_html(save_path)
        
        return str(save_path)
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            class_names: List[str],
                            experiment_name: str = "Unknown",
                            save_path: Optional[str] = None) -> str:
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        
        # 计算百分比
        cm_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # 创建热力图
        sns.heatmap(cm_percent, 
                   annot=True, 
                   fmt='.2%',
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': '预测准确率'})
        
        plt.title(f'混淆矩阵 - {experiment_name}')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / f"confusion_matrix_{experiment_name.replace(' ', '_')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_resource_usage(self, 
                          resource_data: Dict[str, List[float]],
                          experiment_name: str = "Unknown",
                          save_path: Optional[str] = None) -> str:
        """绘制资源使用情况"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'资源使用情况 - {experiment_name}', fontsize=16)
        
        # CPU使用率
        if 'cpu_percent' in resource_data:
            axes[0, 0].plot(resource_data['cpu_percent'], color=self.colors['primary'], linewidth=2)
            axes[0, 0].set_title('CPU使用率')
            axes[0, 0].set_ylabel('使用率 (%)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 内存使用
        if 'memory_mb' in resource_data:
            axes[0, 1].plot(resource_data['memory_mb'], color=self.colors['secondary'], linewidth=2)
            axes[0, 1].set_title('内存使用')
            axes[0, 1].set_ylabel('内存 (MB)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # GPU使用率
        if 'gpu_percent' in resource_data:
            axes[1, 0].plot(resource_data['gpu_percent'], color=self.colors['success'], linewidth=2)
            axes[1, 0].set_title('GPU使用率')
            axes[1, 0].set_ylabel('使用率 (%)')
            axes[1, 0].set_xlabel('时间点')
            axes[1, 0].grid(True, alpha=0.3)
        
        # GPU内存
        if 'gpu_memory_mb' in resource_data:
            axes[1, 1].plot(resource_data['gpu_memory_mb'], color=self.colors['warning'], linewidth=2)
            axes[1, 1].set_title('GPU内存使用')
            axes[1, 1].set_ylabel('内存 (MB)')
            axes[1, 1].set_xlabel('时间点')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if not save_path:
            save_path = self.output_dir / f"resource_usage_{experiment_name.replace(' ', '_')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_model_architecture_diagram(self, 
                                          model_info: Dict[str, Any],
                                          save_path: Optional[str] = None) -> str:
        """生成模型架构图"""
        try:
            import torchviz
            import torch
            
            # 这里需要根据实际模型结构生成
            # 暂时创建一个简单的架构描述图
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 模型信息
            model_name = model_info.get('name', 'Unknown Model')
            total_params = model_info.get('total_params', 0)
            trainable_params = model_info.get('trainable_params', 0)
            model_size = model_info.get('model_size_mb', 0)
            
            # 创建文本描述
            info_text = f"""
模型名称: {model_name}
总参数量: {total_params:,}
可训练参数: {trainable_params:,}
模型大小: {model_size:.2f} MB
            """
            
            ax.text(0.5, 0.5, info_text, 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            ax.set_title(f'模型架构信息 - {model_name}', fontsize=16)
            ax.axis('off')
            
            if not save_path:
                save_path = self.output_dir / f"model_architecture_{model_name.replace(' ', '_')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(save_path)
            
        except ImportError:
            # 如果没有torchviz，创建简单的信息图
            return self.generate_model_architecture_diagram(model_info, save_path)
    
    def create_summary_report(self, 
                            experiments: List[Dict[str, Any]],
                            save_path: Optional[str] = None) -> str:
        """创建汇总报告"""
        if not experiments:
            return ""
        
        # 创建多页面报告
        from matplotlib.backends.backend_pdf import PdfPages
        
        if not save_path:
            save_path = self.output_dir / "summary_report.pdf"
        
        with PdfPages(save_path) as pdf:
            # 第一页：总体统计
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('实验汇总报告', fontsize=20)
            
            # 统计信息
            total_experiments = len(experiments)
            completed_experiments = len([e for e in experiments if e.get('status') == 'completed'])
            avg_accuracy = np.mean([e.get('best_val_accuracy', 0) for e in experiments if e.get('best_val_accuracy')])
            
            # 模型分布
            model_counts = {}
            for exp in experiments:
                model = exp.get('model_name', 'Unknown')
                model_counts[model] = model_counts.get(model, 0) + 1
            
            ax1.pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.1f%%')
            ax1.set_title('模型分布')
            
            # 状态分布
            status_counts = {}
            for exp in experiments:
                status = exp.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            ax2.bar(status_counts.keys(), status_counts.values(), color=self.color_palette)
            ax2.set_title('实验状态分布')
            ax2.tick_params(axis='x', rotation=45)
            
            # 准确率分布
            accuracies = [e.get('best_val_accuracy', 0) for e in experiments if e.get('best_val_accuracy')]
            if accuracies:
                ax3.hist(accuracies, bins=20, color=self.colors['primary'], alpha=0.7)
                ax3.set_title('准确率分布')
                ax3.set_xlabel('准确率')
                ax3.set_ylabel('实验数量')
            
            # 训练时长分布
            durations = [e.get('duration_seconds', 0) for e in experiments if e.get('duration_seconds')]
            if durations:
                ax4.hist(durations, bins=20, color=self.colors['secondary'], alpha=0.7)
                ax4.set_title('训练时长分布')
                ax4.set_xlabel('时长 (秒)')
                ax4.set_ylabel('实验数量')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # 第二页：性能对比
            if len(experiments) > 1:
                self.plot_model_comparison(experiments, save_path=None)
                pdf.savefig(plt.gcf(), bbox_inches='tight')
                plt.close()
            
            # 第三页：时间线
            if len(experiments) > 1:
                self.plot_performance_timeline(experiments, save_path=None)
                pdf.savefig(plt.gcf(), bbox_inches='tight')
                plt.close()
        
        return str(save_path)