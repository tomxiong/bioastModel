"""
FUA Web Interface Module

提供Web界面用于监控和管理FUA系统
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# 可选依赖检查
WEB_AVAILABLE = True
try:
    import plotly
    import plotly.graph_objs as go
    import pandas as pd
    import mlflow
    from fua import create_mlflow_integration, FUAMLflowIntegration
    from fua.production import create_distributed_monitor
    from fua.experiment_tracking.mlflow_integration import FUAExperimentTracker, FUAModelRegistry
except ImportError as e:
    logger.warning(f"Web dependencies not available: {e}")
    WEB_AVAILABLE = False


class FUAWebInterface:
    """FUA Web界面主类"""
    
    def __init__(self, 
                 host: str = "127.0.0.1",
                 port: int = 8080,
                 debug: bool = False,
                 mlflow_tracking_uri: str = "mlruns",
                 mlflow_registry_uri: str = "mlruns"):
        """
        初始化Web界面
        
        Args:
            host: 服务器主机
            port: 服务器端口
            debug: 调试模式
            mlflow_tracking_uri: MLflow跟踪URI
            mlflow_registry_uri: MLflow注册表URI
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_registry_uri = mlflow_registry_uri
        
        # 创建Flask应用
        self.app = Flask(__name__,
                        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                        static_folder=os.path.join(os.path.dirname(__file__), 'static'))
        
        # 启用CORS
        CORS(self.app)
        
        # 初始化组件
        self.mlflow_integration = None
        self.distributed_monitor = None
        self.experiment_tracker = None
        self.model_registry = None
        
        # 立即初始化后端组件
        self.initialize_components()
        
        # 注册路由
        self._register_routes()
        
        logger.info(f"FUA Web Interface initialized on {host}:{port}")
    
    def _register_routes(self):
        """注册所有路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            return render_template('index.html')
        
        @self.app.route('/dashboard')
        def dashboard():
            """仪表板页面"""
            return render_template('dashboard.html')
        
        @self.app.route('/experiments')
        def experiments():
            """实验管理页面"""
            return render_template('experiments.html')
        
        @self.app.route('/models')
        def models():
            """模型管理页面"""
            return render_template('models.html')
        
        @self.app.route('/monitoring')
        def monitoring():
            """监控页面"""
            return render_template('monitoring.html')
        
        # API路由
        @self.app.route('/api/status')
        def api_status():
            """获取系统状态"""
            status = {
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "mlflow": self.mlflow_integration is not None,
                    "distributed_monitor": self.distributed_monitor is not None,
                    "experiment_tracker": self.experiment_tracker is not None,
                    "model_registry": self.model_registry is not None
                },
                "version": "1.0.0"
            }
            return jsonify(status)
        
        @self.app.route('/api/experiments')
        def api_experiments():
            """获取实验列表"""
            if not self.experiment_tracker:
                return jsonify({"error": "Experiment tracker not available"}), 500
            
            try:
                runs = self.experiment_tracker.search_runs()
                experiments = []
                
                for run in runs:
                    experiment = {
                        "run_id": run.info.run_id,
                        "run_name": run.info.run_name,
                        "status": run.info.status,
                        "start_time": run.info.start_time.isoformat() if run.info.start_time else None,
                        "end_time": run.info.end_time.isoformat() if run.info.end_time else None,
                        "metrics": run.data.metrics,
                        "params": run.data.params
                    }
                    experiments.append(experiment)
                
                return jsonify({"experiments": experiments})
            except Exception as e:
                logger.error(f"Failed to get experiments: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/models')
        def api_models():
            """获取模型列表"""
            if not self.model_registry:
                return jsonify({"error": "Model registry not available"}), 500
            
            try:
                models = self.model_registry.list_models()
                model_list = []
                
                for model in models:
                    # 处理时间戳
                    creation_ts = model.creation_timestamp
                    if hasattr(creation_ts, 'isoformat'):
                        creation_ts = creation_ts.isoformat()
                    elif isinstance(creation_ts, (int, float)):
                        creation_ts = datetime.fromtimestamp(creation_ts / 1000 if creation_ts > 1e12 else creation_ts).isoformat()
                    
                    updated_ts = model.last_updated_timestamp
                    if hasattr(updated_ts, 'isoformat'):
                        updated_ts = updated_ts.isoformat()
                    elif isinstance(updated_ts, (int, float)):
                        updated_ts = datetime.fromtimestamp(updated_ts / 1000 if updated_ts > 1e12 else updated_ts).isoformat()
                    
                    model_info = {
                        "name": model.name,
                        "creation_timestamp": creation_ts,
                        "last_updated_timestamp": updated_ts,
                        "description": model.description,
                        "versions": []
                    }
                    
                    # 获取版本信息
                    try:
                        latest_version = self.model_registry.get_latest_model_version(model.name)
                        if latest_version:
                            model_info["latest_version"] = {
                                "version": latest_version.version,
                                "stage": latest_version.current_stage,
                                "status": latest_version.status,
                                "run_id": latest_version.run_id
                            }
                    except:
                        pass
                    
                    model_list.append(model_info)
                
                return jsonify({"models": model_list})
            except Exception as e:
                logger.error(f"Failed to get models: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/monitoring/metrics')
        def api_monitoring_metrics():
            """获取监控指标"""
            if not self.distributed_monitor:
                return jsonify({"error": "Distributed monitor not available"}), 500
            
            try:
                model_id = request.args.get('model_id')
                version_id = request.args.get('version_id')
                
                # 获取系统级指标（不传递model_id和version_id）
                if not model_id and not version_id:
                    # 收集系统级别的监控指标
                    metrics = {
                        "timestamp": datetime.now().isoformat(),
                        "system_status": "healthy",
                        "active_nodes": 1,
                        "running_tasks": 0,
                        "cpu_usage": 25.5,  # 模拟数据
                        "memory_usage": 45.2,  # 模拟数据
                        "nodes": [{
                            "id": self.distributed_monitor.cluster_manager.node_id,
                            "host": "127.0.0.1",
                            "port": 8773,
                            "role": "monitor",
                            "status": "active",
                            "cpu_usage": 25.5,
                            "memory_usage": 45.2,
                            "last_heartbeat": datetime.now().isoformat()
                        }],
                        "alerts": []
                    }
                else:
                    # 获取特定模型的监控指标
                    metrics = self.distributed_monitor.collect_distributed_metrics(
                        model_id=model_id,
                        version_id=version_id
                    )
                
                return jsonify(metrics)
            except Exception as e:
                logger.error(f"Failed to get monitoring metrics: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/metrics/chart')
        def api_metrics_chart():
            """生成指标图表"""
            if not self.experiment_tracker:
                return jsonify({"error": "Experiment tracker not available"}), 500
            
            try:
                run_id = request.args.get('run_id')
                metric_name = request.args.get('metric_name', 'train_loss')
                
                if not run_id:
                    return jsonify({"error": "run_id required"}), 400
                
                run = self.experiment_tracker.get_run(run_id)
                if not run:
                    return jsonify({"error": "Run not found"}), 404
                
                # 提取指标历史
                metrics_history = []
                for key, value in run.data.metrics.items():
                    if key.startswith(metric_name):
                        metrics_history.append({
                            "step": int(key.split('_')[-1]) if '_' in key else 0,
                            "value": value
                        })
                
                # 创建图表
                if WEB_AVAILABLE and metrics_history:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[m["step"] for m in metrics_history],
                        y=[m["value"] for m in metrics_history],
                        mode='lines+markers',
                        name=metric_name
                    ))
                    fig.update_layout(
                        title=f"{metric_name} over time",
                        xaxis_title="Step",
                        yaxis_title=metric_name
                    )
                    
                    chart_json = fig.to_json()
                    return jsonify({"chart": chart_json})
                else:
                    return jsonify({"error": "No data available"}), 404
                    
            except Exception as e:
                logger.error(f"Failed to generate chart: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/experiment/summary')
        def api_experiment_summary():
            """获取实验摘要"""
            if not self.mlflow_integration:
                return jsonify({"error": "MLflow integration not available"}), 500
            
            try:
                summary = self.mlflow_integration.get_experiment_summary()
                return jsonify(summary)
            except Exception as e:
                logger.error(f"Failed to get experiment summary: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/health')
        def health_check():
            """健康检查"""
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
        
        @self.app.route('/api/dataset')
        def api_dataset():
            """获取数据集信息"""
            try:
                dataset_info = {
                    "path": "bioast_dataset",
                    "structure": {
                        "train": {
                            "negative": "train/negative/",
                            "positive": "train/positive/"
                        },
                        "val": {
                            "negative": "val/negative/",
                            "positive": "val/positive/"
                        },
                        "test": {
                            "negative": "test/negative/",
                            "positive": "test/positive/"
                        }
                    },
                    "stats": self._get_dataset_stats(),
                    "last_updated": datetime.now().isoformat()
                }
                return jsonify(dataset_info)
            except Exception as e:
                logger.error(f"Failed to get dataset info: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/dataset')
        def dataset_page():
            """数据集管理页面"""
            return render_template('dataset.html')
        
        @self.app.route('/help')
        def help_page():
            """使用指南页面"""
            return render_template('help.html')
        
        @self.app.route('/api/dataset/upload', methods=['POST'])
        def api_dataset_upload():
            """上传数据集文件"""
            try:
                if 'files' not in request.files:
                    return jsonify({"error": "No files provided"}), 400
                
                files = request.files.getlist('files')
                dataset_type = request.form.get('type', 'train')
                category = request.form.get('category', 'positive')
                
                # 创建目标目录
                target_dir = Path(f"bioast_dataset/{dataset_type}/{category}")
                target_dir.mkdir(parents=True, exist_ok=True)
                
                uploaded_files = []
                for file in files:
                    if file.filename:
                        # 保存文件
                        file_path = target_dir / file.filename
                        file.save(str(file_path))
                        uploaded_files.append(file.filename)
                
                return jsonify({
                    "message": f"Successfully uploaded {len(uploaded_files)} files",
                    "files": uploaded_files,
                    "target_dir": str(target_dir)
                })
                
            except Exception as e:
                logger.error(f"Failed to upload dataset files: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/experiments', methods=['POST'])
        def api_create_experiment():
            """创建新实验"""
            if not self.experiment_tracker:
                return jsonify({"error": "Experiment tracker not available"}), 500
            
            try:
                data = request.get_json()
                
                # 使用MLflow创建实验
                with mlflow.start_run(run_name=data.get('name', 'Unnamed Experiment')) as run:
                    # 记录参数
                    mlflow.log_param("model_type", data.get('model_type', 'unknown'))
                    mlflow.log_param("learning_rate", data.get('learning_rate', 0.001))
                    mlflow.log_param("batch_size", data.get('batch_size', 32))
                    mlflow.log_param("epochs", data.get('epochs', 50))
                    mlflow.log_param("description", data.get('description', ''))
                    
                    # 记录标签
                    mlflow.set_tag("user_created", "true")
                    mlflow.set_tag("created_from", "web_interface")
                
                return jsonify({
                    "message": "Experiment created successfully",
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name
                })
                
            except Exception as e:
                logger.error(f"Failed to create experiment: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/experiments/<run_id>', methods=['GET'])
        def api_get_experiment(run_id):
            """获取单个实验详情"""
            if not self.experiment_tracker:
                return jsonify({"error": "Experiment tracker not available"}), 500
            
            try:
                run = self.experiment_tracker.get_run(run_id)
                if not run:
                    return jsonify({"error": "Experiment not found"}), 404
                
                experiment = {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time.isoformat() if run.info.start_time else None,
                    "end_time": run.info.end_time.isoformat() if run.info.end_time else None,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                
                return jsonify(experiment)
                
            except Exception as e:
                logger.error(f"Failed to get experiment: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/experiments/<run_id>', methods=['DELETE'])
        def api_delete_experiment(run_id):
            """删除实验"""
            if not self.experiment_tracker:
                return jsonify({"error": "Experiment tracker not available"}), 500
            
            try:
                # MLflow doesn't support deleting runs directly through API
                # This is a placeholder implementation
                # In a real implementation, you would use MLflow's client API
                
                return jsonify({
                    "message": f"Experiment {run_id} marked for deletion",
                    "note": "Actual deletion would be handled by MLflow client"
                })
                
            except Exception as e:
                logger.error(f"Failed to delete experiment: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/experiments/compare', methods=['POST'])
        def api_compare_experiments():
            """比较多个实验"""
            if not self.experiment_tracker:
                return jsonify({"error": "Experiment tracker not available"}), 500
            
            try:
                data = request.get_json()
                run_ids = data.get('run_ids', [])
                
                if len(run_ids) < 2:
                    return jsonify({"error": "At least 2 experiments required for comparison"}), 400
                
                experiments = []
                for run_id in run_ids:
                    run = self.experiment_tracker.get_run(run_id)
                    if run:
                        experiments.append({
                            "run_id": run.info.run_id,
                            "run_name": run.info.run_name,
                            "metrics": run.data.metrics,
                            "params": run.data.params
                        })
                
                # 生成比较数据
                comparison = {
                    "experiments": experiments,
                    "metrics_comparison": {},
                    "params_comparison": {}
                }
                
                # 比较指标
                all_metrics = set()
                for exp in experiments:
                    all_metrics.update(exp['metrics'].keys())
                
                for metric in all_metrics:
                    comparison['metrics_comparison'][metric] = {
                        exp['run_id']: exp['metrics'].get(metric, None)
                        for exp in experiments
                    }
                
                return jsonify(comparison)
                
            except Exception as e:
                logger.error(f"Failed to compare experiments: {e}")
                return jsonify({"error": str(e)}), 500
    
    def initialize_components(self):
        """初始化后端组件"""
        if not WEB_AVAILABLE:
            logger.warning("Web dependencies not available, skipping component initialization")
            return
        
        try:
            # 初始化MLflow集成
            self.mlflow_integration = create_mlflow_integration(
                tracking_uri=self.mlflow_tracking_uri,
                registry_uri=self.mlflow_registry_uri,
                experiment_name="fua_web_interface"
            )
            
            # 初始化分布式监控
            self.distributed_monitor = create_distributed_monitor(
                node_id="web_interface",
                config={
                    "node_role": "monitor",
                    "region": "web-region",
                    "consul_enabled": False,
                    "redis_enabled": False,
                    "websocket_port": 8773,
                    "api_port": 8088
                }
            )
            
            # 获取组件引用
            self.experiment_tracker = self.mlflow_integration.tracker
            self.model_registry = self.mlflow_integration.registry
            
            logger.info("Backend components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
    
    def start(self):
        """启动Web服务器"""
        logger.info(f"Starting FUA Web Interface on http://{self.host}:{self.port}")
        
        # 初始化组件
        self.initialize_components()
        
        # 启动服务器
        self.app.run(host=self.host, port=self.port, debug=self.debug)
    
    def _get_dataset_stats(self):
        """获取数据集统计信息"""
        import os
        from pathlib import Path
        
        dataset_path = Path("bioast_dataset")
        stats = {
            "train": {"negative": 0, "positive": 0},
            "val": {"negative": 0, "positive": 0},
            "test": {"negative": 0, "positive": 0},
            "total": 0
        }
        
        if dataset_path.exists():
            for split in ["train", "val", "test"]:
                split_path = dataset_path / split
                if split_path.exists():
                    for category in ["negative", "positive"]:
                        category_path = split_path / category
                        if category_path.exists():
                            count = len(list(category_path.glob("*.jpg"))) + \
                                   len(list(category_path.glob("*.png"))) + \
                                   len(list(category_path.glob("*.jpeg")))
                            stats[split][category] = count
                            stats["total"] += count
        
        return stats
    
    def stop(self):
        """停止Web服务器"""
        if self.distributed_monitor:
            self.distributed_monitor.stop()
        logger.info("FUA Web Interface stopped")


# 便捷函数
def create_web_interface(host: str = "127.0.0.1",
                        port: int = 8080,
                        debug: bool = False,
                        **kwargs) -> FUAWebInterface:
    """创建FUA Web界面实例"""
    return FUAWebInterface(host=host, port=port, debug=debug, **kwargs)


def start_web_ui(host: str = "127.0.0.1", port: int = 8080, debug: bool = False):
    """启动Web UI"""
    web_interface = create_web_interface(host=host, port=port, debug=debug)
    web_interface.start()


if __name__ == "__main__":
    # 启动Web UI
    start_web_ui()