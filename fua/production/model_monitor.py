"""
模型监控系统

提供实时模型性能监控、异常检测、告警通知和
性能指标收集功能，确保生产环境中模型的稳定运行
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    MimeText = None
    MimeMultipart = None
import requests
import sqlite3
import pandas as pd
from collections import deque, defaultdict
from scipy import stats
import asyncio
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    PREDICTION_DRIFT = "prediction_drift"
    DATA_DRIFT = "data_drift"


class AlertChannel(Enum):
    """告警渠道"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    SMS = "sms"


@dataclass
class MetricThreshold:
    """指标阈值配置"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    operator: str = "less_than"  # less_than, greater_than, equal
    
    def check_threshold(self, value: float) -> Optional[AlertSeverity]:
        """检查是否超过阈值"""
        if self.operator == "less_than":
            if value <= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value <= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.operator == "greater_than":
            if value >= self.critical_threshold:
                return AlertSeverity.CRITICAL
            elif value >= self.warning_threshold:
                return AlertSeverity.WARNING
        elif self.operator == "equal":
            if abs(value - self.critical_threshold) < 1e-6:
                return AlertSeverity.CRITICAL
            elif abs(value - self.warning_threshold) < 1e-6:
                return AlertSeverity.WARNING
        return None


@dataclass
class Alert:
    """告警信息"""
    id: str
    model_id: str
    version_id: str
    metric_type: MetricType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """从字典创建"""
        data['metric_type'] = MetricType(data['metric_type'])
        data['severity'] = AlertSeverity(data['severity'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ModelMetrics:
    """模型指标数据"""
    model_id: str
    version_id: str
    timestamp: datetime
    accuracy: float = 0.0
    latency_ms: float = 0.0
    throughput_qps: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    prediction_drift: float = 0.0
    data_drift: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """从字典创建"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, buffer_size: int = 1000):
        """
        初始化指标收集器
        
        Args:
            buffer_size: 缓冲区大小
        """
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.custom_collectors: Dict[str, Callable] = {}
        
    def add_custom_collector(self, name: str, collector: Callable):
        """添加自定义指标收集器"""
        self.custom_collectors[name] = collector
        
    def collect_metrics(self, model: nn.Module, model_id: str, version_id: str,
                       input_data: torch.Tensor, predictions: torch.Tensor = None,
                       labels: torch.Tensor = None) -> ModelMetrics:
        """收集模型指标"""
        device = next(model.parameters()).device
        model.eval()
        
        # 基础指标
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_data.to(device))
        latency_ms = (time.time() - start_time) * 1000 / len(input_data)
        
        # 准确率
        accuracy = 0.0
        if labels is not None and predictions is not None:
            accuracy = (predictions == labels.to(device)).float().mean().item()
        
        # 内存使用
        memory_usage_mb = 0.0
        if torch.cuda.is_available():
            memory_usage_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        # 吞吐量
        throughput_qps = 1000 / latency_ms if latency_ms > 0 else 0
        
        # 错误率
        error_rate = 1.0 - accuracy
        
        # GPU使用率
        gpu_usage_percent = 0.0
        if torch.cuda.is_available():
            gpu_usage_percent = torch.cuda.utilization()
        
        # 创建指标对象
        metrics = ModelMetrics(
            model_id=model_id,
            version_id=version_id,
            timestamp=datetime.now(),
            accuracy=accuracy,
            latency_ms=latency_ms,
            throughput_qps=throughput_qps,
            error_rate=error_rate,
            memory_usage_mb=memory_usage_mb,
            gpu_usage_percent=gpu_usage_percent
        )
        
        # 收集自定义指标
        for name, collector in self.custom_collectors.items():
            try:
                value = collector(model, input_data, outputs)
                metrics.custom_metrics[name] = value
            except Exception as e:
                logger.error(f"Failed to collect custom metric {name}: {e}")
        
        # 添加到缓冲区
        self.metrics_buffer.append(metrics)
        
        return metrics


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        """
        初始化异常检测器
        
        Args:
            window_size: 滑动窗口大小
            sensitivity: 灵敏度（标准差倍数）
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
    def detect_anomaly(self, metrics: ModelMetrics) -> List[Tuple[str, float, float]]:
        """
        检测异常
        
        Returns:
            异常列表 [(metric_name, value, z_score)]
        """
        anomalies = []
        
        # 检查每个指标
        metric_fields = [
            ('accuracy', metrics.accuracy),
            ('latency_ms', metrics.latency_ms),
            ('throughput_qps', metrics.throughput_qps),
            ('error_rate', metrics.error_rate),
            ('memory_usage_mb', metrics.memory_usage_mb),
            ('gpu_usage_percent', metrics.gpu_usage_percent)
        ]
        
        for field_name, value in metric_fields:
            history = self.metric_history[field_name]
            history.append(value)
            
            if len(history) >= 10:  # 需要足够的历史数据
                # 计算Z分数
                mean = np.mean(history)
                std = np.std(history)
                
                if std > 0:
                    z_score = abs(value - mean) / std
                    
                    if z_score > self.sensitivity:
                        anomalies.append((field_name, value, z_score))
        
        # 检查自定义指标
        for name, value in metrics.custom_metrics.items():
            history = self.metric_history[f"custom_{name}"]
            history.append(value)
            
            if len(history) >= 10:
                mean = np.mean(history)
                std = np.std(history)
                
                if std > 0:
                    z_score = abs(value - mean) / std
                    
                    if z_score > self.sensitivity:
                        anomalies.append((f"custom_{name}", value, z_score))
        
        return anomalies


class AlertManager:
    """告警管理器"""
    
    def __init__(self, db_path: str = "alerts.db"):
        """
        初始化告警管理器
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.alert_channels: Dict[AlertChannel, Callable] = {}
        self.thresholds: List[MetricThreshold] = []
        self.alert_queue = queue.Queue()
        self.active = False
        self.worker_thread = None
        
        # 初始化数据库
        self._init_db()
        
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version_id TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL NOT NULL,
                timestamp TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0,
                resolved INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_threshold(self, threshold: MetricThreshold):
        """添加阈值配置"""
        self.thresholds.append(threshold)
        
    def add_alert_channel(self, channel: AlertChannel, handler: Callable):
        """添加告警渠道"""
        self.alert_channels[channel] = handler
        
    def start(self):
        """启动告警管理器"""
        if not self.active:
            self.active = True
            self.worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
            self.worker_thread.start()
            logger.info("Alert manager started")
            
    def stop(self):
        """停止告警管理器"""
        self.active = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Alert manager stopped")
        
    def _process_alerts(self):
        """处理告警队列"""
        while self.active:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._handle_alert(alert)
                self.alert_queue.task_done()
            except queue.Empty:
                continue
                
    def _handle_alert(self, alert: Alert):
        """处理单个告警"""
        # 保存到数据库
        self._save_alert(alert)
        
        # 发送告警
        for channel, handler in self.alert_channels.items():
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
                
    def _save_alert(self, alert: Alert):
        """保存告警到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO alerts 
            (id, model_id, version_id, metric_type, severity, message, 
             value, threshold, timestamp, acknowledged, resolved, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.id,
            alert.model_id,
            alert.version_id,
            alert.metric_type.value,
            alert.severity.value,
            alert.message,
            alert.value,
            alert.threshold,
            alert.timestamp.isoformat(),
            alert.acknowledged,
            alert.resolved,
            json.dumps(alert.metadata)
        ))
        
        conn.commit()
        conn.close()
        
    def check_thresholds(self, metrics: ModelMetrics) -> List[Alert]:
        """检查阈值并生成告警"""
        alerts = []
        
        for threshold in self.thresholds:
            # 获取指标值
            if threshold.metric_type == MetricType.ACCURACY:
                value = metrics.accuracy
            elif threshold.metric_type == MetricType.LATENCY:
                value = metrics.latency_ms
            elif threshold.metric_type == MetricType.THROUGHPUT:
                value = metrics.throughput_qps
            elif threshold.metric_type == MetricType.ERROR_RATE:
                value = metrics.error_rate
            elif threshold.metric_type == MetricType.MEMORY_USAGE:
                value = metrics.memory_usage_mb
            elif threshold.metric_type == MetricType.CPU_USAGE:
                value = metrics.cpu_usage_percent
            elif threshold.metric_type == MetricType.GPU_USAGE:
                value = metrics.gpu_usage_percent
            else:
                continue
                
            # 检查阈值
            severity = threshold.check_threshold(value)
            if severity:
                alert = Alert(
                    id=f"{metrics.model_id}_{threshold.metric_type.value}_{int(time.time())}",
                    model_id=metrics.model_id,
                    version_id=metrics.version_id,
                    metric_type=threshold.metric_type,
                    severity=severity,
                    message=f"{threshold.metric_type.value} {severity.value}: {value:.4f}",
                    value=value,
                    threshold=threshold.warning_threshold if severity == AlertSeverity.WARNING else threshold.critical_threshold,
                    metadata={
                        'metric_type': threshold.metric_type.value,
                        'operator': threshold.operator
                    }
                )
                
                alerts.append(alert)
                self.alert_queue.put(alert)
                
        return alerts


class EmailNotifier:
    """邮件通知器"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, recipients: List[str]):
        """
        初始化邮件通知器
        
        Args:
            smtp_server: SMTP服务器
            smtp_port: SMTP端口
            username: 用户名
            password: 密码
            recipients: 收件人列表
        """
        if not EMAIL_AVAILABLE:
            logger.warning("Email functionality not available")
            
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        
    def __call__(self, alert: Alert):
        """发送邮件告警"""
        if not EMAIL_AVAILABLE:
            logger.warning(f"Cannot send email alert for {alert.id}: Email not available")
            return
            
        msg = MimeMultipart()
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = f"[{alert.severity.value.upper()}] Model Alert: {alert.model_id}"
        
        body = f"""
Model Alert Notification

Model ID: {alert.model_id}
Version ID: {alert.version_id}
Severity: {alert.severity.value}
Metric: {alert.metric_type.value}
Value: {alert.value:.4f}
Threshold: {alert.threshold:.4f}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.message}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            logger.info(f"Email alert sent for {alert.id}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


class SlackNotifier:
    """Slack通知器"""
    
    def __init__(self, webhook_url: str, channel: str = "#alerts"):
        """
        初始化Slack通知器
        
        Args:
            webhook_url: Webhook URL
            channel: 频道名称
        """
        self.webhook_url = webhook_url
        self.channel = channel
        
    def __call__(self, alert: Alert):
        """发送Slack告警"""
        color = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }.get(alert.severity, "warning")
        
        payload = {
            "channel": self.channel,
            "attachments": [
                {
                    "color": color,
                    "title": f"Model Alert: {alert.model_id}",
                    "text": alert.message,
                    "fields": [
                        {
                            "title": "Metric",
                            "value": alert.metric_type.value,
                            "short": True
                        },
                        {
                            "title": "Value",
                            "value": f"{alert.value:.4f}",
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": f"{alert.threshold:.4f}",
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.value,
                            "short": True
                        }
                    ],
                    "footer": f"Version: {alert.version_id}",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            if response.status_code == 200:
                logger.info(f"Slack alert sent for {alert.id}")
            else:
                logger.error(f"Failed to send Slack alert: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")


class WebhookNotifier:
    """Webhook通知器"""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None):
        """
        初始化Webhook通知器
        
        Args:
            webhook_url: Webhook URL
            headers: 请求头
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}
        
    def __call__(self, alert: Alert):
        """发送Webhook告警"""
        try:
            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                headers=self.headers
            )
            if response.status_code == 200:
                logger.info(f"Webhook alert sent for {alert.id}")
            else:
                logger.error(f"Failed to send webhook alert: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")


class ModelMonitor:
    """模型监控器主类"""
    
    def __init__(self, 
                 db_path: str = "monitoring.db",
                 metrics_buffer_size: int = 1000,
                 anomaly_window_size: int = 100,
                 anomaly_sensitivity: float = 2.0):
        """
        初始化模型监控器
        
        Args:
            db_path: 数据库路径
            metrics_buffer_size: 指标缓冲区大小
            anomaly_window_size: 异常检测窗口大小
            anomaly_sensitivity: 异常检测灵敏度
        """
        self.db_path = db_path
        self.metrics_collector = MetricsCollector(buffer_size=metrics_buffer_size)
        self.anomaly_detector = AnomalyDetector(
            window_size=anomaly_window_size,
            sensitivity=anomaly_sensitivity
        )
        self.alert_manager = AlertManager(db_path=db_path)
        
        # 监控的模型
        self.monitored_models: Dict[str, Dict] = {}
        
        # 监控状态
        self.monitoring_active = False
        self.monitor_thread = None
        
        # WebSocket服务器
        self.websocket_server = None
        self.websocket_port = 8765
        
        # 初始化数据库
        self._init_db()
        
        logger.info("ModelMonitor initialized")
        
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                accuracy REAL,
                latency_ms REAL,
                throughput_qps REAL,
                error_rate REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                gpu_usage_percent REAL,
                prediction_drift REAL,
                data_drift REAL,
                custom_metrics TEXT
            )
        ''')
        
        # 模型配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_configs (
                model_id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                config TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_model(self, model_id: str, version_id: str, model: nn.Module,
                  config: Dict[str, Any] = None):
        """添加要监控的模型"""
        self.monitored_models[model_id] = {
            'version_id': version_id,
            'model': model,
            'config': config or {}
        }
        
        # 保存配置
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO model_configs (model_id, version_id, config, created_at)
            VALUES (?, ?, ?, ?)
        ''', (
            model_id,
            version_id,
            json.dumps(config or {}),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added model to monitoring: {model_id}")
        
    def remove_model(self, model_id: str):
        """移除监控的模型"""
        if model_id in self.monitored_models:
            del self.monitored_models[model_id]
            logger.info(f"Removed model from monitoring: {model_id}")
            
    def start_monitoring(self, interval: int = 60):
        """
        开始监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_models,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            
            # 启动告警管理器
            self.alert_manager.start()
            
            # 启动WebSocket服务器
            self._start_websocket_server()
            
            logger.info(f"Model monitoring started with {interval}s interval")
            
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
        # 停止告警管理器
        self.alert_manager.stop()
        
        # 停止WebSocket服务器
        self._stop_websocket_server()
        
        logger.info("Model monitoring stopped")
        
    def _monitor_models(self, interval: int):
        """监控模型主循环"""
        while self.monitoring_active:
            try:
                # 为每个模型生成测试数据
                for model_id, model_info in self.monitored_models.items():
                    model = model_info['model']
                    version_id = model_info['version_id']
                    
                    # 生成随机测试数据
                    if hasattr(model, 'config') and 'input_size' in model.config:
                        input_size = model.config['input_size']
                    else:
                        input_size = (1, 3, 224, 224)
                    
                    test_data = torch.randn(input_size)
                    
                    # 收集指标
                    metrics = self.metrics_collector.collect_metrics(
                        model, model_id, version_id, test_data
                    )
                    
                    # 保存指标
                    self._save_metrics(metrics)
                    
                    # 异常检测
                    anomalies = self.anomaly_detector.detect_anomaly(metrics)
                    
                    if anomalies:
                        logger.warning(f"Anomalies detected for {model_id}: {anomalies}")
                        
                        # 生成告警
                        for field_name, value, z_score in anomalies:
                            alert = Alert(
                                id=f"{model_id}_{field_name}_{int(time.time())}",
                                model_id=model_id,
                                version_id=version_id,
                                metric_type=MetricType.PREDICTION_DRIFT,
                                severity=AlertSeverity.WARNING if z_score < 3 else AlertSeverity.ERROR,
                                message=f"Anomaly detected in {field_name}: {value:.4f} (z-score: {z_score:.2f})",
                                value=value,
                                threshold=0.0,
                                metadata={
                                    'field_name': field_name,
                                    'z_score': z_score
                                }
                            )
                            
                            self.alert_manager.alert_queue.put(alert)
                    
                    # 检查阈值
                    alerts = self.alert_manager.check_thresholds(metrics)
                    if alerts:
                        logger.warning(f"Threshold alerts for {model_id}: {len(alerts)} alerts")
                
                # 等待下次监控
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
                
    def _save_metrics(self, metrics: ModelMetrics):
        """保存指标到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (
                model_id, version_id, timestamp, accuracy, latency_ms,
                throughput_qps, error_rate, memory_usage_mb, cpu_usage_percent,
                gpu_usage_percent, prediction_drift, data_drift, custom_metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.model_id,
            metrics.version_id,
            metrics.timestamp.isoformat(),
            metrics.accuracy,
            metrics.latency_ms,
            metrics.throughput_qps,
            metrics.error_rate,
            metrics.memory_usage_mb,
            metrics.cpu_usage_percent,
            metrics.gpu_usage_percent,
            metrics.prediction_drift,
            metrics.data_drift,
            json.dumps(metrics.custom_metrics)
        ))
        
        conn.commit()
        conn.close()
        
    def _start_websocket_server(self):
        """启动WebSocket服务器"""
        async def handle_client(websocket, path):
            """处理WebSocket客户端"""
            try:
                while True:
                    # 发送最新指标
                    latest_metrics = self.get_latest_metrics()
                    await websocket.send(json.dumps(latest_metrics))
                    await asyncio.sleep(1)
            except websockets.exceptions.ConnectionClosed:
                pass
                
        async def server():
            """WebSocket服务器"""
            self.websocket_server = await websockets.serve(
                handle_client,
                "localhost",
                self.websocket_port
            )
            logger.info(f"WebSocket server started on port {self.websocket_port}")
            await self.websocket_server.wait_closed()
            
        # 在新线程中运行服务器
        def run_server():
            asyncio.run(server())
            
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
    def _stop_websocket_server(self):
        """停止WebSocket服务器"""
        if self.websocket_server:
            self.websocket_server.close()
            logger.info("WebSocket server stopped")
            
    def get_metrics(self, model_id: str, version_id: str = None,
                   start_time: datetime = None, end_time: datetime = None) -> List[ModelMetrics]:
        """获取指标历史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM metrics WHERE model_id = ?"
        params = [model_id]
        
        if version_id:
            query += " AND version_id = ?"
            params.append(version_id)
            
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
            
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        # 转换为ModelMetrics对象
        metrics_list = []
        for row in rows:
            metrics = ModelMetrics(
                model_id=row[1],
                version_id=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                accuracy=row[4] or 0.0,
                latency_ms=row[5] or 0.0,
                throughput_qps=row[6] or 0.0,
                error_rate=row[7] or 0.0,
                memory_usage_mb=row[8] or 0.0,
                cpu_usage_percent=row[9] or 0.0,
                gpu_usage_percent=row[10] or 0.0,
                prediction_drift=row[11] or 0.0,
                data_drift=row[12] or 0.0,
                custom_metrics=json.loads(row[13]) if row[13] else {}
            )
            metrics_list.append(metrics)
            
        return metrics_list
        
    def get_latest_metrics(self, model_id: str = None) -> Dict[str, Any]:
        """获取最新指标"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if model_id:
            cursor.execute('''
                SELECT * FROM metrics 
                WHERE model_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (model_id,))
        else:
            cursor.execute('''
                SELECT model_id, version_id, timestamp, accuracy, latency_ms,
                       throughput_qps, error_rate, memory_usage_mb
                FROM metrics 
                WHERE timestamp >= datetime('now', '-1 hour')
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            
        rows = cursor.fetchall()
        conn.close()
        
        if model_id and rows:
            row = rows[0]
            return {
                'model_id': row[1],
                'version_id': row[2],
                'timestamp': row[3],
                'accuracy': row[4],
                'latency_ms': row[5],
                'throughput_qps': row[6],
                'error_rate': row[7],
                'memory_usage_mb': row[8]
            }
        elif not model_id:
            return [{
                'model_id': row[0],
                'version_id': row[1],
                'timestamp': row[2],
                'accuracy': row[3],
                'latency_ms': row[4],
                'throughput_qps': row[5],
                'error_rate': row[6],
                'memory_usage_mb': row[7]
            } for row in rows]
        else:
            return {}
            
    def get_alerts(self, model_id: str = None, severity: AlertSeverity = None,
                   start_time: datetime = None, end_time: datetime = None,
                   resolved: bool = None) -> List[Alert]:
        """获取告警历史"""
        conn = sqlite3.connect(self.alert_manager.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM alerts WHERE 1=1"
        params = []
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
            
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
            
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
            
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(int(resolved))
            
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        # 转换为Alert对象
        alerts = []
        for row in rows:
            alert = Alert(
                id=row[0],
                model_id=row[1],
                version_id=row[2],
                metric_type=MetricType(row[3]),
                severity=AlertSeverity(row[4]),
                message=row[5],
                value=row[6],
                threshold=row[7],
                timestamp=datetime.fromisoformat(row[8]),
                acknowledged=bool(row[9]),
                resolved=bool(row[10]),
                metadata=json.loads(row[11]) if row[11] else {}
            )
            alerts.append(alert)
            
        return alerts
        
    def acknowledge_alert(self, alert_id: str, user: str = "system"):
        """确认告警"""
        conn = sqlite3.connect(self.alert_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts 
            SET acknowledged = 1, metadata = json_set(
                json(metadata), 
                '$.acknowledged_by', ?,
                '$.acknowledged_at', ?
            )
            WHERE id = ?
        ''', (user, datetime.now().isoformat(), alert_id))
        
        conn.commit()
        conn.close()
        
    def resolve_alert(self, alert_id: str, user: str = "system", 
                     resolution: str = "Resolved"):
        """解决告警"""
        conn = sqlite3.connect(self.alert_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts 
            SET resolved = 1, metadata = json_set(
                json(metadata), 
                '$.resolved_by', ?,
                '$.resolved_at', ?,
                '$.resolution', ?
            )
            WHERE id = ?
        ''', (user, datetime.now().isoformat(), resolution, alert_id))
        
        conn.commit()
        conn.close()
        
    def generate_report(self, model_id: str = None, 
                       output_path: str = None) -> str:
        """生成监控报告"""
        if output_path is None:
            output_path = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 获取统计信息
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 模型统计
        if model_id:
            cursor.execute('''
                SELECT COUNT(*) as total_metrics,
                       AVG(accuracy) as avg_accuracy,
                       AVG(latency_ms) as avg_latency,
                       AVG(error_rate) as avg_error_rate
                FROM metrics 
                WHERE model_id = ?
            ''', (model_id,))
        else:
            cursor.execute('''
                SELECT COUNT(*) as total_metrics,
                       AVG(accuracy) as avg_accuracy,
                       AVG(latency_ms) as avg_latency,
                       AVG(error_rate) as avg_error_rate
                FROM metrics 
                WHERE timestamp >= datetime('now', '-24 hours')
            ''')
            
        stats = cursor.fetchone()
        
        # 告警统计
        alert_conn = sqlite3.connect(self.alert_manager.db_path)
        alert_cursor = alert_conn.cursor()
        
        if model_id:
            alert_cursor.execute('''
                SELECT severity, COUNT(*) 
                FROM alerts 
                WHERE model_id = ? AND timestamp >= datetime('now', '-24 hours')
                GROUP BY severity
            ''', (model_id,))
        else:
            alert_cursor.execute('''
                SELECT severity, COUNT(*) 
                FROM alerts 
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY severity
            ''')
            
        alert_stats = dict(alert_cursor.fetchall())
        
        conn.close()
        alert_conn.close()
        
        # 生成报告
        report = f"# Model Monitoring Report\\n\\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
        
        if model_id:
            report += f"## Model: {model_id}\\n\\n"
        
        report += "## Summary\\n\\n"
        report += f"- Total metrics: {stats[0] or 0}\\n"
        report += f"- Average accuracy: {stats[1] or 0:.4f}\\n"
        report += f"- Average latency: {stats[2] or 0:.2f}ms\\n"
        report += f"- Average error rate: {stats[3] or 0:.4f}\\n\\n"
        
        report += "## Alerts (24h)\\n\\n"
        for severity in ['critical', 'error', 'warning', 'info']:
            count = alert_stats.get(severity, 0)
            if count > 0:
                report += f"- {severity.title()}: {count}\\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Monitoring report saved to: {output_path}")
        return output_path


def create_model_monitor(db_path: str = "monitoring.db",
                        metrics_buffer_size: int = 1000,
                        anomaly_window_size: int = 100,
                        anomaly_sensitivity: float = 2.0) -> ModelMonitor:
    """创建模型监控器实例"""
    return ModelMonitor(
        db_path=db_path,
        metrics_buffer_size=metrics_buffer_size,
        anomaly_window_size=anomaly_window_size,
        anomaly_sensitivity=anomaly_sensitivity
    )