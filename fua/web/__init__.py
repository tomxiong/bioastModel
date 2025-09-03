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


def start_web_ui(host: str = "127.0.0.1", port: int = 8080):
    """启动Web UI"""
    web_interface = create_web_interface(host=host, port=port)
    web_interface.start()


if __name__ == "__main__":
    # 启动Web UI
    start_web_ui()