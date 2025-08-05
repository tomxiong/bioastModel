"""报告生成器

生成标准化的模型分析报告。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from jinja2 import Template

from .visualization import Visualizer


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = Visualizer(str(self.output_dir / "visualizations"))
        
        # 创建子目录
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "json").mkdir(exist_ok=True)
        (self.output_dir / "pdf").mkdir(exist_ok=True)
    
    def generate_experiment_report(self, 
                                 experiment_data: Dict[str, Any],
                                 include_visualizations: bool = True) -> Dict[str, str]:
        """生成单个实验报告"""
        experiment_id = experiment_data.get('experiment_id', 'unknown')
        experiment_name = experiment_data.get('name', 'Unknown Experiment')
        
        # 生成可视化图表
        chart_paths = {}
        if include_visualizations:
            chart_paths['training_curves'] = self.visualizer.plot_training_curves(experiment_data)
            
            # 如果有资源使用数据
            if 'resource_usage' in experiment_data:
                chart_paths['resource_usage'] = self.visualizer.plot_resource_usage(
                    experiment_data['resource_usage'], experiment_name
                )
        
        # 生成各种格式的报告
        report_paths = {
            'json': self._generate_json_report(experiment_data),
            'markdown': self._generate_markdown_report(experiment_data, chart_paths),
            'html': self._generate_html_report(experiment_data, chart_paths)
        }
        
        return report_paths
    
    def generate_comparison_report(self, 
                                 experiments: List[Dict[str, Any]],
                                 title: str = "模型对比报告") -> Dict[str, str]:
        """生成模型对比报告"""
        if not experiments:
            return {}
        
        # 生成对比图表
        chart_paths = {
            'comparison': self.visualizer.plot_model_comparison(experiments),
            'timeline': self.visualizer.plot_performance_timeline(experiments),
            'dashboard': self.visualizer.create_interactive_dashboard(experiments),
            'summary': self.visualizer.create_summary_report(experiments)
        }
        
        # 生成报告
        report_data = {
            'title': title,
            'generated_at': datetime.now().isoformat(),
            'experiments': experiments,
            'summary': self._calculate_comparison_summary(experiments),
            'charts': chart_paths
        }
        
        report_paths = {
            'json': self._generate_comparison_json_report(report_data),
            'markdown': self._generate_comparison_markdown_report(report_data),
            'html': self._generate_comparison_html_report(report_data)
        }
        
        return report_paths
    
    def generate_model_registry_report(self, 
                                     registry_data: Dict[str, Any]) -> Dict[str, str]:
        """生成模型注册表报告"""
        report_data = {
            'title': '模型注册表报告',
            'generated_at': datetime.now().isoformat(),
            'registry': registry_data,
            'summary': self._calculate_registry_summary(registry_data)
        }
        
        report_paths = {
            'json': self._generate_registry_json_report(report_data),
            'markdown': self._generate_registry_markdown_report(report_data),
            'html': self._generate_registry_html_report(report_data)
        }
        
        return report_paths
    
    def _generate_json_report(self, experiment_data: Dict[str, Any]) -> str:
        """生成JSON格式报告"""
        experiment_id = experiment_data.get('experiment_id', 'unknown')
        output_path = self.output_dir / "json" / f"experiment_{experiment_id}.json"
        
        # 添加报告元数据
        report_data = {
            'report_type': 'experiment',
            'generated_at': datetime.now().isoformat(),
            'experiment': experiment_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def _generate_markdown_report(self, 
                                experiment_data: Dict[str, Any],
                                chart_paths: Dict[str, str]) -> str:
        """生成Markdown格式报告"""
        experiment_id = experiment_data.get('experiment_id', 'unknown')
        output_path = self.output_dir / "markdown" / f"experiment_{experiment_id}.md"
        
        # Markdown模板
        template_str = """
# 实验报告: {{ experiment.name }}

## 基本信息

- **实验ID**: {{ experiment.experiment_id }}
- **模型名称**: {{ experiment.model_name }}
- **数据集**: {{ experiment.dataset_name }}
- **状态**: {{ experiment.status }}
- **创建时间**: {{ experiment.created_at }}
- **完成时间**: {{ experiment.completed_at }}

## 配置参数

- **批次大小**: {{ experiment.batch_size }}
- **学习率**: {{ experiment.learning_rate }}
- **优化器**: {{ experiment.optimizer }}
- **调度器**: {{ experiment.scheduler }}
- **总轮数**: {{ experiment.total_epochs }}

## 性能指标

- **最佳验证准确率**: {{ "%.4f" | format(experiment.best_val_accuracy) }}
- **最佳验证损失**: {{ "%.4f" | format(experiment.best_val_loss) }}
- **最佳轮数**: {{ experiment.best_epoch }}
- **训练时长**: {{ "%.2f" | format(experiment.duration_seconds) }} 秒

{% if experiment.notes %}
## 备注

{{ experiment.notes }}
{% endif %}

{% if experiment.error_message %}
## 错误信息

```
{{ experiment.error_message }}
```
{% endif %}

## 可视化图表

{% for chart_name, chart_path in charts.items() %}
### {{ chart_name | title }}

![{{ chart_name }}]({{ chart_path }})

{% endfor %}

## 详细指标

{% if experiment.metrics %}
### 训练过程

| Epoch | 训练损失 | 训练准确率 | 验证损失 | 验证准确率 | 学习率 | 时间(秒) |
|-------|----------|------------|----------|------------|--------|----------|
{% for i in range(experiment.metrics.train_losses|length) %}
| {{ i + 1 }} | {{ "%.4f" | format(experiment.metrics.train_losses[i]) }} | {{ "%.4f" | format(experiment.metrics.train_accuracies[i]) }} | {{ "%.4f" | format(experiment.metrics.val_losses[i]) }} | {{ "%.4f" | format(experiment.metrics.val_accuracies[i]) }} | {{ "%.6f" | format(experiment.metrics.learning_rates[i]) }} | {{ "%.2f" | format(experiment.metrics.epoch_times[i]) }} |
{% endfor %}
{% endif %}

---

*报告生成时间: {{ generated_at }}*
        """
        
        template = Template(template_str)
        content = template.render(
            experiment=experiment_data,
            charts=chart_paths,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)
    
    def _generate_html_report(self, 
                            experiment_data: Dict[str, Any],
                            chart_paths: Dict[str, str]) -> str:
        """生成HTML格式报告"""
        experiment_id = experiment_data.get('experiment_id', 'unknown')
        output_path = self.output_dir / "html" / f"experiment_{experiment_id}.html"
        
        # HTML模板
        template_str = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实验报告: {{ experiment.name }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2E86AB;
            margin: 0;
        }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .info-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2E86AB;
        }
        .info-card h3 {
            margin-top: 0;
            color: #2E86AB;
        }
        .metric-highlight {
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 20px 0;
        }
        .metric-highlight h2 {
            margin: 0;
            font-size: 2.5em;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #2E86AB;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .status {
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status.completed {
            background-color: #d4edda;
            color: #155724;
        }
        .status.running {
            background-color: #fff3cd;
            color: #856404;
        }
        .status.failed {
            background-color: #f8d7da;
            color: #721c24;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ experiment.name }}</h1>
            <p>实验ID: {{ experiment.experiment_id }}</p>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>基本信息</h3>
                <p><strong>模型:</strong> {{ experiment.model_name }}</p>
                <p><strong>数据集:</strong> {{ experiment.dataset_name }}</p>
                <p><strong>状态:</strong> <span class="status {{ experiment.status }}">{{ experiment.status }}</span></p>
                <p><strong>创建时间:</strong> {{ experiment.created_at }}</p>
                <p><strong>完成时间:</strong> {{ experiment.completed_at }}</p>
            </div>
            
            <div class="info-card">
                <h3>训练配置</h3>
                <p><strong>批次大小:</strong> {{ experiment.batch_size }}</p>
                <p><strong>学习率:</strong> {{ experiment.learning_rate }}</p>
                <p><strong>优化器:</strong> {{ experiment.optimizer }}</p>
                <p><strong>调度器:</strong> {{ experiment.scheduler }}</p>
                <p><strong>总轮数:</strong> {{ experiment.total_epochs }}</p>
            </div>
        </div>
        
        <div class="metric-highlight">
            <h2>{{ "%.2f" | format(experiment.best_val_accuracy * 100) }}%</h2>
            <p>最佳验证准确率</p>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <h3>性能指标</h3>
                <p><strong>最佳验证损失:</strong> {{ "%.4f" | format(experiment.best_val_loss) }}</p>
                <p><strong>最佳轮数:</strong> {{ experiment.best_epoch }}</p>
                <p><strong>训练时长:</strong> {{ "%.2f" | format(experiment.duration_seconds) }} 秒</p>
            </div>
        </div>
        
        {% for chart_name, chart_path in charts.items() %}
        <div class="chart-container">
            <h3>{{ chart_name | title }}</h3>
            <img src="{{ chart_path }}" alt="{{ chart_name }}">
        </div>
        {% endfor %}
        
        {% if experiment.notes %}
        <div class="info-card">
            <h3>备注</h3>
            <p>{{ experiment.notes }}</p>
        </div>
        {% endif %}
        
        <div class="footer">
            <p>报告生成时间: {{ generated_at }}</p>
        </div>
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        content = template.render(
            experiment=experiment_data,
            charts=chart_paths,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)
    
    def _generate_comparison_json_report(self, report_data: Dict[str, Any]) -> str:
        """生成对比报告JSON"""
        output_path = self.output_dir / "json" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def _generate_comparison_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """生成对比报告Markdown"""
        output_path = self.output_dir / "markdown" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        template_str = """
# {{ title }}

## 概览

- **报告生成时间**: {{ generated_at }}
- **对比实验数量**: {{ experiments|length }}
- **平均准确率**: {{ "%.4f" | format(summary.avg_accuracy) }}
- **最佳准确率**: {{ "%.4f" | format(summary.best_accuracy) }}
- **最差准确率**: {{ "%.4f" | format(summary.worst_accuracy) }}

## 实验对比表

| 实验名称 | 模型 | 准确率 | 损失 | 训练时长 | 状态 |
|----------|------|--------|------|----------|------|
{% for exp in experiments %}
| {{ exp.name }} | {{ exp.model_name }} | {{ "%.4f" | format(exp.best_val_accuracy) }} | {{ "%.4f" | format(exp.best_val_loss) }} | {{ "%.2f" | format(exp.duration_seconds) }}s | {{ exp.status }} |
{% endfor %}

## 模型性能排名

{% for exp in summary.ranked_experiments %}
{{ loop.index }}. **{{ exp.name }}** ({{ exp.model_name }}) - 准确率: {{ "%.4f" | format(exp.best_val_accuracy) }}
{% endfor %}

## 可视化图表

{% for chart_name, chart_path in charts.items() %}
### {{ chart_name | title }}

![{{ chart_name }}]({{ chart_path }})

{% endfor %}

## 分析总结

### 最佳模型
- **名称**: {{ summary.best_experiment.name }}
- **模型**: {{ summary.best_experiment.model_name }}
- **准确率**: {{ "%.4f" | format(summary.best_experiment.best_val_accuracy) }}
- **训练时长**: {{ "%.2f" | format(summary.best_experiment.duration_seconds) }} 秒

### 性能分布
- **准确率标准差**: {{ "%.4f" | format(summary.accuracy_std) }}
- **平均训练时长**: {{ "%.2f" | format(summary.avg_duration) }} 秒
- **完成率**: {{ "%.1f" | format(summary.completion_rate * 100) }}%

---

*报告生成时间: {{ generated_at }}*
        """
        
        template = Template(template_str)
        content = template.render(**report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)
    
    def _generate_comparison_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成对比报告HTML"""
        output_path = self.output_dir / "html" / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # 这里可以创建更复杂的HTML模板
        # 为了简化，先使用基本模板
        template_str = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background-color: #f2f2f2; }
        .chart { margin: 20px 0; text-align: center; }
        .chart img { max-width: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p>生成时间: {{ generated_at }}</p>
        
        <h2>实验对比</h2>
        <table>
            <tr>
                <th>实验名称</th>
                <th>模型</th>
                <th>准确率</th>
                <th>损失</th>
                <th>训练时长</th>
                <th>状态</th>
            </tr>
            {% for exp in experiments %}
            <tr>
                <td>{{ exp.name }}</td>
                <td>{{ exp.model_name }}</td>
                <td>{{ "%.4f" | format(exp.best_val_accuracy) }}</td>
                <td>{{ "%.4f" | format(exp.best_val_loss) }}</td>
                <td>{{ "%.2f" | format(exp.duration_seconds) }}s</td>
                <td>{{ exp.status }}</td>
            </tr>
            {% endfor %}
        </table>
        
        {% for chart_name, chart_path in charts.items() %}
        <div class="chart">
            <h3>{{ chart_name | title }}</h3>
            <img src="{{ chart_path }}" alt="{{ chart_name }}">
        </div>
        {% endfor %}
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        content = template.render(**report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)
    
    def _generate_registry_json_report(self, report_data: Dict[str, Any]) -> str:
        """生成注册表JSON报告"""
        output_path = self.output_dir / "json" / f"registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def _generate_registry_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """生成注册表Markdown报告"""
        output_path = self.output_dir / "markdown" / f"registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        template_str = """
# {{ title }}

## 注册表统计

- **总模型数**: {{ summary.total_models }}
- **活跃模型数**: {{ summary.active_models }}
- **平均准确率**: {{ "%.4f" | format(summary.avg_accuracy) }}
- **最新更新**: {{ summary.last_updated }}

## 模型列表

{% for model_id, model in registry.items() %}
### {{ model.name }}

- **ID**: {{ model_id }}
- **版本**: {{ model.version }}
- **架构**: {{ model.architecture }}
- **状态**: {{ model.status }}
- **准确率**: {{ "%.4f" | format(model.performance.accuracy) }}
- **模型大小**: {{ "%.2f" | format(model.model_size_mb) }} MB
- **创建时间**: {{ model.created_at }}

{% endfor %}

---

*报告生成时间: {{ generated_at }}*
        """
        
        template = Template(template_str)
        content = template.render(**report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)
    
    def _generate_registry_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成注册表HTML报告"""
        output_path = self.output_dir / "html" / f"registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # 简化的HTML模板
        template_str = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .model-card { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .model-header { font-size: 1.2em; font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p>生成时间: {{ generated_at }}</p>
        
        <h2>统计信息</h2>
        <ul>
            <li>总模型数: {{ summary.total_models }}</li>
            <li>活跃模型数: {{ summary.active_models }}</li>
            <li>平均准确率: {{ "%.4f" | format(summary.avg_accuracy) }}</li>
        </ul>
        
        <h2>模型列表</h2>
        {% for model_id, model in registry.items() %}
        <div class="model-card">
            <div class="model-header">{{ model.name }}</div>
            <p><strong>ID:</strong> {{ model_id }}</p>
            <p><strong>版本:</strong> {{ model.version }}</p>
            <p><strong>架构:</strong> {{ model.architecture }}</p>
            <p><strong>准确率:</strong> {{ "%.4f" | format(model.performance.accuracy) }}</p>
        </div>
        {% endfor %}
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        content = template.render(**report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)
    
    def _calculate_comparison_summary(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算对比摘要"""
        if not experiments:
            return {}
        
        accuracies = [exp.get('best_val_accuracy', 0) for exp in experiments if exp.get('best_val_accuracy')]
        durations = [exp.get('duration_seconds', 0) for exp in experiments if exp.get('duration_seconds')]
        completed = [exp for exp in experiments if exp.get('status') == 'completed']
        
        # 按准确率排序
        ranked_experiments = sorted(experiments, 
                                  key=lambda x: x.get('best_val_accuracy', 0), 
                                  reverse=True)
        
        return {
            'total_experiments': len(experiments),
            'completed_experiments': len(completed),
            'avg_accuracy': np.mean(accuracies) if accuracies else 0,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'worst_accuracy': min(accuracies) if accuracies else 0,
            'accuracy_std': np.std(accuracies) if accuracies else 0,
            'avg_duration': np.mean(durations) if durations else 0,
            'completion_rate': len(completed) / len(experiments) if experiments else 0,
            'best_experiment': ranked_experiments[0] if ranked_experiments else None,
            'ranked_experiments': ranked_experiments
        }
    
    def _calculate_registry_summary(self, registry_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算注册表摘要"""
        models = registry_data.get('models', {})
        
        if not models:
            return {
                'total_models': 0,
                'active_models': 0,
                'avg_accuracy': 0,
                'last_updated': 'N/A'
            }
        
        active_models = [m for m in models.values() if m.get('status') == 'active']
        accuracies = [m.get('performance', {}).get('accuracy', 0) for m in models.values()]
        
        return {
            'total_models': len(models),
            'active_models': len(active_models),
            'avg_accuracy': np.mean(accuracies) if accuracies else 0,
            'last_updated': max([m.get('updated_at', '') for m in models.values()]) if models else 'N/A'
        }