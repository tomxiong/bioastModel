"""主仪表板

整合模型管理、实验监控和报告生成功能。
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, jsonify, request, send_file
import threading
import webbrowser
from werkzeug.serving import make_server

from .visualization import Visualizer
from .report_generator import ReportGenerator
from ..model_registry import ModelRegistry
from ..experiment_manager import ExperimentTracker, ExperimentDatabase


class Dashboard:
    """主仪表板"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5000,
                 debug: bool = False):
        self.host = host
        self.port = port
        self.debug = debug
        
        # 初始化组件
        self.visualizer = Visualizer()
        self.report_generator = ReportGenerator()
        self.model_registry = ModelRegistry()
        self.experiment_tracker = ExperimentTracker()
        self.experiment_db = ExperimentDatabase()
        
        # Flask应用
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / "templates"),
                        static_folder=str(Path(__file__).parent / "static"))
        self.server = None
        self.server_thread = None
        
        self._setup_routes()
        self._create_templates()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('index.html')
        
        @self.app.route('/api/experiments')
        def get_experiments():
            """获取实验列表"""
            status = request.args.get('status')
            model_name = request.args.get('model_name')
            limit = int(request.args.get('limit', 50))
            offset = int(request.args.get('offset', 0))
            
            experiments = self.experiment_db.list_experiments(
                status=status, model_name=model_name, limit=limit, offset=offset
            )
            
            return jsonify({
                'experiments': experiments,
                'total': len(experiments)
            })
        
        @self.app.route('/api/experiments/<experiment_id>')
        def get_experiment(experiment_id):
            """获取单个实验"""
            experiment = self.experiment_db.get_experiment(experiment_id)
            if experiment:
                # 获取详细指标
                metrics = self.experiment_db.get_experiment_metrics(experiment_id)
                experiment['detailed_metrics'] = metrics
                return jsonify(experiment)
            return jsonify({'error': 'Experiment not found'}), 404
        
        @self.app.route('/api/models')
        def get_models():
            """获取模型列表"""
            models = self.model_registry.list_models()
            return jsonify({'models': models})
        
        @self.app.route('/api/models/<model_id>')
        def get_model(model_id):
            """获取单个模型"""
            model = self.model_registry.get_model(model_id)
            if model:
                return jsonify(model.to_dict())
            return jsonify({'error': 'Model not found'}), 404
        
        @self.app.route('/api/statistics')
        def get_statistics():
            """获取统计信息"""
            exp_stats = self.experiment_db.get_statistics()
            model_stats = self.model_registry.get_statistics()
            
            return jsonify({
                'experiments': exp_stats,
                'models': model_stats
            })
        
        @self.app.route('/api/reports/experiment/<experiment_id>')
        def generate_experiment_report(experiment_id):
            """生成实验报告"""
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                return jsonify({'error': 'Experiment not found'}), 404
            
            # 获取详细指标
            metrics = self.experiment_db.get_experiment_metrics(experiment_id)
            experiment['detailed_metrics'] = metrics
            
            # 生成报告
            report_paths = self.report_generator.generate_experiment_report(experiment)
            
            return jsonify({
                'report_paths': report_paths,
                'experiment_id': experiment_id
            })
        
        @self.app.route('/api/reports/comparison')
        def generate_comparison_report():
            """生成对比报告"""
            experiment_ids = request.args.getlist('experiment_ids')
            if not experiment_ids:
                return jsonify({'error': 'No experiments specified'}), 400
            
            experiments = []
            for exp_id in experiment_ids:
                exp = self.experiment_db.get_experiment(exp_id)
                if exp:
                    experiments.append(exp)
            
            if not experiments:
                return jsonify({'error': 'No valid experiments found'}), 404
            
            # 生成对比报告
            report_paths = self.report_generator.generate_comparison_report(experiments)
            
            return jsonify({
                'report_paths': report_paths,
                'experiment_count': len(experiments)
            })
        
        @self.app.route('/api/visualizations/training_curves/<experiment_id>')
        def get_training_curves(experiment_id):
            """获取训练曲线"""
            experiment = self.experiment_db.get_experiment(experiment_id)
            if not experiment:
                return jsonify({'error': 'Experiment not found'}), 404
            
            # 获取详细指标
            metrics = self.experiment_db.get_experiment_metrics(experiment_id)
            experiment['metrics'] = {
                'train_losses': [m['train_loss'] for m in metrics],
                'val_losses': [m['val_loss'] for m in metrics],
                'train_accuracies': [m['train_accuracy'] for m in metrics],
                'val_accuracies': [m['val_accuracy'] for m in metrics],
                'learning_rates': [m['learning_rate'] for m in metrics],
                'epoch_times': [m['epoch_time'] for m in metrics]
            }
            
            chart_path = self.visualizer.plot_training_curves(experiment)
            
            return jsonify({
                'chart_path': chart_path,
                'experiment_id': experiment_id
            })
        
        @self.app.route('/api/visualizations/model_comparison')
        def get_model_comparison():
            """获取模型对比图"""
            experiment_ids = request.args.getlist('experiment_ids')
            metric = request.args.get('metric', 'best_val_accuracy')
            
            experiments = []
            for exp_id in experiment_ids:
                exp = self.experiment_db.get_experiment(exp_id)
                if exp:
                    experiments.append(exp)
            
            if not experiments:
                return jsonify({'error': 'No valid experiments found'}), 404
            
            chart_path = self.visualizer.plot_model_comparison(experiments, metric)
            
            return jsonify({
                'chart_path': chart_path,
                'experiment_count': len(experiments)
            })
        
        @self.app.route('/api/visualizations/interactive_dashboard')
        def get_interactive_dashboard():
            """获取交互式仪表板"""
            experiments = self.experiment_db.list_experiments(limit=100)
            
            dashboard_path = self.visualizer.create_interactive_dashboard(experiments)
            
            return jsonify({
                'dashboard_path': dashboard_path,
                'experiment_count': len(experiments)
            })
        
        @self.app.route('/files/<path:filename>')
        def serve_file(filename):
            """提供文件服务"""
            try:
                return send_file(filename)
            except FileNotFoundError:
                return jsonify({'error': 'File not found'}), 404
        
        @self.app.route('/api/search')
        def search():
            """搜索功能"""
            query = request.args.get('q', '')
            search_type = request.args.get('type', 'all')  # all, experiments, models
            
            results = {'experiments': [], 'models': []}
            
            if search_type in ['all', 'experiments']:
                results['experiments'] = self.experiment_db.search_experiments(query)
            
            if search_type in ['all', 'models']:
                results['models'] = self.model_registry.search_models(query)
            
            return jsonify(results)
    
    def _create_templates(self):
        """创建模板文件"""
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        
        # 创建主页模板
        index_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型管理仪表板</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .nav {
            background: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .nav-buttons {
            display: flex;
            gap: 1rem;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background-color: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #5a6fd8;
        }
        
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background-color: #5a6268;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card h3 {
            color: #667eea;
            margin-bottom: 1rem;
            font-size: 1.2rem;
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: bold;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }
        
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        
        .error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .search-container {
            margin-bottom: 2rem;
        }
        
        .search-input {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 1rem;
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .table th,
        .table td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        .table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        
        .table tr:hover {
            background-color: #f8f9fa;
        }
        
        .status {
            padding: 0.25rem 0.5rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
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
    </style>
</head>
<body>
    <div class="header">
        <h1>模型管理仪表板</h1>
        <p>深度学习模型训练与管理平台</p>
    </div>
    
    <div class="nav">
        <div class="nav-buttons">
            <button class="btn btn-primary" onclick="loadDashboard()">仪表板</button>
            <button class="btn btn-secondary" onclick="loadExperiments()">实验管理</button>
            <button class="btn btn-secondary" onclick="loadModels()">模型注册表</button>
            <button class="btn btn-secondary" onclick="loadReports()">报告中心</button>
            <button class="btn btn-secondary" onclick="loadVisualizations()">可视化</button>
        </div>
    </div>
    
    <div class="container">
        <div id="content">
            <div class="loading">正在加载数据...</div>
        </div>
    </div>
    
    <script>
        // 全局变量
        let currentView = 'dashboard';
        let experimentsData = [];
        let modelsData = [];
        let statisticsData = {};
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboard();
        });
        
        // 加载仪表板
        async function loadDashboard() {
            currentView = 'dashboard';
            updateNavButtons();
            
            try {
                const response = await fetch('/api/statistics');
                statisticsData = await response.json();
                
                renderDashboard();
            } catch (error) {
                showError('加载统计数据失败: ' + error.message);
            }
        }
        
        // 加载实验列表
        async function loadExperiments() {
            currentView = 'experiments';
            updateNavButtons();
            
            try {
                const response = await fetch('/api/experiments');
                const data = await response.json();
                experimentsData = data.experiments;
                
                renderExperiments();
            } catch (error) {
                showError('加载实验数据失败: ' + error.message);
            }
        }
        
        // 加载模型列表
        async function loadModels() {
            currentView = 'models';
            updateNavButtons();
            
            try {
                const response = await fetch('/api/models');
                const data = await response.json();
                modelsData = data.models;
                
                renderModels();
            } catch (error) {
                showError('加载模型数据失败: ' + error.message);
            }
        }
        
        // 渲染仪表板
        function renderDashboard() {
            const content = document.getElementById('content');
            const expStats = statisticsData.experiments || {};
            const modelStats = statisticsData.models || {};
            
            content.innerHTML = `
                <div class="dashboard-grid">
                    <div class="card">
                        <h3>实验统计</h3>
                        <div class="stat-number">${expStats.total_experiments || 0}</div>
                        <div class="stat-label">总实验数</div>
                    </div>
                    
                    <div class="card">
                        <h3>模型统计</h3>
                        <div class="stat-number">${modelStats.total_models || 0}</div>
                        <div class="stat-label">注册模型数</div>
                    </div>
                    
                    <div class="card">
                        <h3>平均准确率</h3>
                        <div class="stat-number">${(expStats.average_performance?.accuracy * 100 || 0).toFixed(1)}%</div>
                        <div class="stat-label">验证准确率</div>
                    </div>
                    
                    <div class="card">
                        <h3>最近活动</h3>
                        <div class="stat-number">${expStats.recent_experiments_7days || 0}</div>
                        <div class="stat-label">7天内实验</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>快速操作</h3>
                    <div class="nav-buttons">
                        <button class="btn btn-primary" onclick="generateInteractiveDashboard()">生成交互式仪表板</button>
                        <button class="btn btn-secondary" onclick="loadExperiments()">查看所有实验</button>
                        <button class="btn btn-secondary" onclick="loadModels()">查看模型注册表</button>
                    </div>
                </div>
            `;
        }
        
        // 渲染实验列表
        function renderExperiments() {
            const content = document.getElementById('content');
            
            let tableRows = '';
            experimentsData.forEach(exp => {
                tableRows += `
                    <tr>
                        <td>${exp.name || 'Unknown'}</td>
                        <td>${exp.model_name || 'Unknown'}</td>
                        <td><span class="status ${exp.status}">${exp.status}</span></td>
                        <td>${(exp.best_val_accuracy * 100 || 0).toFixed(2)}%</td>
                        <td>${(exp.duration_seconds || 0).toFixed(1)}s</td>
                        <td>${exp.created_at ? new Date(exp.created_at).toLocaleDateString() : 'Unknown'}</td>
                        <td>
                            <button class="btn btn-primary" onclick="viewExperiment('${exp.experiment_id}')">查看</button>
                            <button class="btn btn-secondary" onclick="generateExperimentReport('${exp.experiment_id}')">报告</button>
                        </td>
                    </tr>
                `;
            });
            
            content.innerHTML = `
                <div class="search-container">
                    <input type="text" class="search-input" placeholder="搜索实验..." onkeyup="searchExperiments(this.value)">
                </div>
                
                <table class="table">
                    <thead>
                        <tr>
                            <th>实验名称</th>
                            <th>模型</th>
                            <th>状态</th>
                            <th>准确率</th>
                            <th>训练时长</th>
                            <th>创建时间</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tableRows}
                    </tbody>
                </table>
            `;
        }
        
        // 渲染模型列表
        function renderModels() {
            const content = document.getElementById('content');
            
            let tableRows = '';
            modelsData.forEach(model => {
                tableRows += `
                    <tr>
                        <td>${model.name || 'Unknown'}</td>
                        <td>${model.architecture || 'Unknown'}</td>
                        <td>${model.version || '1.0.0'}</td>
                        <td><span class="status ${model.status}">${model.status}</span></td>
                        <td>${(model.model_size_mb || 0).toFixed(2)} MB</td>
                        <td>${model.created_at ? new Date(model.created_at).toLocaleDateString() : 'Unknown'}</td>
                        <td>
                            <button class="btn btn-primary" onclick="viewModel('${model.model_id}')">查看</button>
                        </td>
                    </tr>
                `;
            });
            
            content.innerHTML = `
                <div class="search-container">
                    <input type="text" class="search-input" placeholder="搜索模型..." onkeyup="searchModels(this.value)">
                </div>
                
                <table class="table">
                    <thead>
                        <tr>
                            <th>模型名称</th>
                            <th>架构</th>
                            <th>版本</th>
                            <th>状态</th>
                            <th>大小</th>
                            <th>创建时间</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${tableRows}
                    </tbody>
                </table>
            `;
        }
        
        // 更新导航按钮状态
        function updateNavButtons() {
            const buttons = document.querySelectorAll('.nav-buttons .btn');
            buttons.forEach(btn => {
                btn.className = 'btn btn-secondary';
            });
            
            // 高亮当前视图按钮
            const currentButton = document.querySelector(`[onclick="load${currentView.charAt(0).toUpperCase() + currentView.slice(1)}()"]`);
            if (currentButton) {
                currentButton.className = 'btn btn-primary';
            }
        }
        
        // 显示错误信息
        function showError(message) {
            const content = document.getElementById('content');
            content.innerHTML = `<div class="error">${message}</div>`;
        }
        
        // 生成交互式仪表板
        async function generateInteractiveDashboard() {
            try {
                const response = await fetch('/api/visualizations/interactive_dashboard');
                const data = await response.json();
                
                if (data.dashboard_path) {
                    window.open(data.dashboard_path, '_blank');
                }
            } catch (error) {
                showError('生成仪表板失败: ' + error.message);
            }
        }
        
        // 生成实验报告
        async function generateExperimentReport(experimentId) {
            try {
                const response = await fetch(`/api/reports/experiment/${experimentId}`);
                const data = await response.json();
                
                if (data.report_paths) {
                    alert('报告生成成功！\n' + Object.values(data.report_paths).join('\n'));
                }
            } catch (error) {
                showError('生成报告失败: ' + error.message);
            }
        }
        
        // 其他功能函数
        function loadReports() {
            alert('报告中心功能开发中...');
        }
        
        function loadVisualizations() {
            alert('可视化功能开发中...');
        }
        
        function viewExperiment(experimentId) {
            alert('查看实验详情功能开发中...');
        }
        
        function viewModel(modelId) {
            alert('查看模型详情功能开发中...');
        }
        
        function searchExperiments(query) {
            // 实现搜索功能
            console.log('搜索实验:', query);
        }
        
        function searchModels(query) {
            // 实现搜索功能
            console.log('搜索模型:', query);
        }
    </script>
</body>
</html>
        """
        
        with open(template_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(index_template)
    
    def start_server(self, open_browser: bool = True):
        """启动服务器"""
        if self.server_thread and self.server_thread.is_alive():
            print(f"服务器已在运行: http://{self.host}:{self.port}")
            return
        
        self.server = make_server(self.host, self.port, self.app, threaded=True)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        url = f"http://{self.host}:{self.port}"
        print(f"仪表板已启动: {url}")
        
        if open_browser:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    
    def stop_server(self):
        """停止服务器"""
        if self.server:
            self.server.shutdown()
            print("服务器已停止")
    
    def generate_full_report(self, 
                           output_dir: Optional[str] = None) -> Dict[str, str]:
        """生成完整报告"""
        if not output_dir:
            output_dir = f"reports/full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有数据
        experiments = self.experiment_db.list_experiments(limit=1000)
        models = self.model_registry.list_models()
        
        # 生成各种报告
        report_paths = {}
        
        # 实验对比报告
        if experiments:
            comparison_paths = self.report_generator.generate_comparison_report(
                experiments, "完整实验对比报告"
            )
            report_paths.update({f"comparison_{k}": v for k, v in comparison_paths.items()})
        
        # 模型注册表报告
        if models:
            registry_data = {'models': {m['model_id']: m for m in models}}
            registry_paths = self.report_generator.generate_model_registry_report(registry_data)
            report_paths.update({f"registry_{k}": v for k, v in registry_paths.items()})
        
        # 生成汇总可视化
        if experiments:
            summary_chart = self.visualizer.create_summary_report(experiments)
            report_paths['summary_chart'] = summary_chart
            
            interactive_dashboard = self.visualizer.create_interactive_dashboard(experiments)
            report_paths['interactive_dashboard'] = interactive_dashboard
        
        # 创建索引文件
        index_content = self._create_report_index(report_paths)
        index_path = output_path / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        report_paths['index'] = str(index_path)
        
        return report_paths
    
    def _create_report_index(self, report_paths: Dict[str, str]) -> str:
        """创建报告索引页面"""
        template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>完整报告索引</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .report-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .report-link { display: inline-block; margin: 5px; padding: 10px 15px; background: #007bff; color: white; text-decoration: none; border-radius: 3px; }
        .report-link:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>模型管理完整报告</h1>
        <p>生成时间: {{ generated_at }}</p>
        
        <div class="report-section">
            <h2>实验对比报告</h2>
            {% for name, path in comparison_reports.items() %}
            <a href="{{ path }}" class="report-link">{{ name }}</a>
            {% endfor %}
        </div>
        
        <div class="report-section">
            <h2>模型注册表报告</h2>
            {% for name, path in registry_reports.items() %}
            <a href="{{ path }}" class="report-link">{{ name }}</a>
            {% endfor %}
        </div>
        
        <div class="report-section">
            <h2>可视化图表</h2>
            {% for name, path in visualizations.items() %}
            <a href="{{ path }}" class="report-link">{{ name }}</a>
            {% endfor %}
        </div>
    </div>
</body>
</html>
        """
        
        # 分类报告路径
        comparison_reports = {k: v for k, v in report_paths.items() if k.startswith('comparison_')}
        registry_reports = {k: v for k, v in report_paths.items() if k.startswith('registry_')}
        visualizations = {k: v for k, v in report_paths.items() if k in ['summary_chart', 'interactive_dashboard']}
        
        from jinja2 import Template
        template_obj = Template(template)
        
        return template_obj.render(
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            comparison_reports=comparison_reports,
            registry_reports=registry_reports,
            visualizations=visualizations
        )
    
    def get_status(self) -> Dict[str, Any]:
        """获取仪表板状态"""
        return {
            'server_running': self.server_thread and self.server_thread.is_alive(),
            'host': self.host,
            'port': self.port,
            'url': f"http://{self.host}:{self.port}",
            'components': {
                'visualizer': bool(self.visualizer),
                'report_generator': bool(self.report_generator),
                'model_registry': bool(self.model_registry),
                'experiment_tracker': bool(self.experiment_tracker),
                'experiment_db': bool(self.experiment_db)
            }
        }