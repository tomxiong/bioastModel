"""
分布式监控系统

支持多节点监控、数据聚合、负载均衡和故障转移的分布式架构
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
import asyncio
import aiohttp
import websockets
import sqlite3
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    import consul
    CONSUL_AVAILABLE = True
except ImportError:
    CONSUL_AVAILABLE = False
    consul = None
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import uuid
import socket
import psutil
from collections import defaultdict, deque
import pandas as pd
from scipy import stats

# 导入现有监控组件
from .model_monitor import (
    ModelMonitor, MetricsCollector, AnomalyDetector, AlertManager,
    Alert, AlertSeverity, MetricType, AlertChannel, MetricThreshold,
    ModelMetrics, EmailNotifier, SlackNotifier, WebhookNotifier
)

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """节点状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class NodeRole(Enum):
    """节点角色"""
    MONITOR = "monitor"          # 监控节点
    AGGREGATOR = "aggregator"    # 聚合节点
    COORDINATOR = "coordinator"  # 协调节点
    STORAGE = "storage"         # 存储节点


@dataclass
class ClusterNode:
    """集群节点信息"""
    id: str
    host: str
    port: int
    role: NodeRole
    status: NodeStatus
    region: str = "default"
    zone: str = "default"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    capacity: Dict[str, float] = field(default_factory=dict)
    load: float = 0.0
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['role'] = self.role.value
        data['status'] = self.status.value
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterNode':
        """从字典创建"""
        data['role'] = NodeRole(data['role'])
        data['status'] = NodeStatus(data['status'])
        data['last_heartbeat'] = datetime.fromisoformat(data['last_heartbeat'])
        return cls(**data)


@dataclass
class MonitoringTask:
    """监控任务"""
    id: str
    model_id: str
    version_id: str
    node_id: str
    schedule: str  # cron expression or interval
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        if self.last_run:
            data['last_run'] = self.last_run.isoformat()
        if self.next_run:
            data['next_run'] = self.next_run.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return data


class DistributedMetricsCollector:
    """分布式指标收集器"""
    
    def __init__(self, buffer_size: int = 10000):
        """
        初始化分布式指标收集器
        
        Args:
            buffer_size: 缓冲区大小
        """
        self.buffer_size = buffer_size
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.node_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated_metrics: Dict[str, Dict] = {}
        
        # Redis连接
        self.redis_client = None
        self.redis_enabled = False
        
        # Consul连接
        self.consul_client = None
        self.consul_enabled = False
        
    def enable_redis(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """启用Redis支持"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, skipping Redis setup")
            return
            
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db)
            self.redis_client.ping()
            self.redis_enabled = True
            logger.info(f"Redis enabled: {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to enable Redis: {e}")
            
    def enable_consul(self, host: str = "localhost", port: int = 8500):
        """启用Consul支持"""
        if not CONSUL_AVAILABLE:
            logger.warning("Consul not available, skipping Consul setup")
            return
            
        try:
            self.consul_client = consul.Consul(host=host, port=port)
            self.consul_client.agent.self()
            self.consul_enabled = True
            logger.info(f"Consul enabled: {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to enable Consul: {e}")
    
    def collect_distributed_metrics(self, nodes: List[ClusterNode], 
                                   model_id: str, version_id: str) -> Dict[str, Any]:
        """收集分布式指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "version_id": version_id,
            "nodes": len(nodes),
            "active_nodes": sum(1 for n in nodes if n.status == NodeStatus.ACTIVE),
            "aggregated": {}
        }
        
        # 收集各节点指标
        node_metrics = []
        for node in nodes:
            if node.status == NodeStatus.ACTIVE:
                node_metric = self._collect_node_metrics(node)
                node_metrics.append(node_metric)
                
                # 存储到Redis
                if self.redis_enabled:
                    key = f"metrics:{node.id}:{model_id}:{int(time.time())}"
                    self.redis_client.setex(key, 3600, json.dumps(node_metric))
        
        if node_metrics:
            # 聚合指标
            metrics["aggregated"] = self._aggregate_metrics(node_metrics)
            
        return metrics
    
    def _collect_node_metrics(self, node: ClusterNode) -> Dict[str, Any]:
        """收集单个节点指标"""
        try:
            # 模拟从远程节点收集指标
            # 实际实现中应该通过HTTP或gRPC调用
            metrics = {
                "node_id": node.id,
                "host": node.host,
                "region": node.region,
                "zone": node.zone,
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent,
                    "network_io": psutil.net_io_counters()._asdict()
                },
                "model": {
                    "inference_count": np.random.randint(100, 1000),
                    "avg_latency": np.random.uniform(10, 100),
                    "error_rate": np.random.uniform(0.01, 0.1),
                    "throughput": np.random.uniform(10, 100)
                }
            }
            return metrics
        except Exception as e:
            logger.error(f"Failed to collect metrics from node {node.id}: {e}")
            return {}
    
    def _aggregate_metrics(self, node_metrics: List[Dict]) -> Dict[str, Any]:
        """聚合节点指标"""
        if not node_metrics:
            return {}
            
        aggregated = {}
        
        # 系统指标聚合
        cpu_values = [m["system"]["cpu_usage"] for m in node_metrics if "system" in m]
        memory_values = [m["system"]["memory_usage"] for m in node_metrics if "system" in m]
        
        if cpu_values:
            aggregated["cpu"] = {
                "avg": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std": np.std(cpu_values)
            }
            
        if memory_values:
            aggregated["memory"] = {
                "avg": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
                "std": np.std(memory_values)
            }
        
        # 模型指标聚合
        latency_values = [m["model"]["avg_latency"] for m in node_metrics if "model" in m]
        throughput_values = [m["model"]["throughput"] for m in node_metrics if "model" in m]
        
        if latency_values:
            aggregated["latency"] = {
                "avg": np.mean(latency_values),
                "p95": np.percentile(latency_values, 95),
                "p99": np.percentile(latency_values, 99)
            }
            
        if throughput_values:
            aggregated["throughput"] = {
                "total": sum(throughput_values),
                "avg": np.mean(throughput_values),
                "max": np.max(throughput_values)
            }
            
        return aggregated


class ClusterManager:
    """集群管理器"""
    
    def __init__(self, node_id: str = None):
        """
        初始化集群管理器
        
        Args:
            node_id: 节点ID，如果为None则自动生成
        """
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.nodes: Dict[str, ClusterNode] = {}
        self.local_node: Optional[ClusterNode] = None
        
        # 服务发现
        self.consul_client = None
        self.consul_enabled = False
        
        # 健康检查
        self.health_check_interval = 30
        self.health_check_thread = None
        self.health_check_active = False
        
        # 负载均衡
        self.load_balancer = RoundRobinLoadBalancer()
        
        # 故障转移
        self.failover_enabled = True
        self.failover_threshold = 3  # 连续失败次数
        
    def initialize_local_node(self, role: NodeRole, host: str = None, 
                            port: int = 8765, region: str = "default"):
        """初始化本地节点"""
        if not host:
            host = socket.gethostbyname(socket.gethostname())
            
        self.local_node = ClusterNode(
            id=self.node_id,
            host=host,
            port=port,
            role=role,
            status=NodeStatus.ACTIVE,
            region=region
        )
        
        # 注册到集群
        self.nodes[self.node_id] = self.local_node
        
        logger.info(f"Local node initialized: {self.local_node}")
        
    def enable_consul(self, host: str = "localhost", port: int = 8500):
        """启用Consul服务发现"""
        if not CONSUL_AVAILABLE:
            logger.warning("Consul not available, skipping Consul setup")
            return
            
        try:
            self.consul_client = consul.Consul(host=host, port=port)
            self.consul_enabled = True
            
            # 注册服务
            if self.local_node:
                self.consul_client.agent.service.register(
                    name="fua-monitor",
                    service_id=self.local_node.id,
                    address=self.local_node.host,
                    port=self.local_node.port,
                    tags=[self.local_node.role.value, self.local_node.region]
                )
                
            logger.info(f"Consul enabled: {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to enable Consul: {e}")
    
    def discover_nodes(self):
        """发现集群中的其他节点"""
        if not self.consul_enabled:
            return
            
        try:
            # 获取所有fua-monitor服务
            services = self.consul_client.health.service("fua-monitor", passing=True)[1]
            
            for service in services:
                node_info = service['Service']
                node_id = node_info['ID']
                
                if node_id != self.node_id and node_id not in self.nodes:
                    # 创建新节点
                    node = ClusterNode(
                        id=node_id,
                        host=node_info['Address'],
                        port=node_info['Port'],
                        role=NodeRole(node_info['Tags'][0]) if node_info['Tags'] else NodeRole.MONITOR,
                        status=NodeStatus.ACTIVE,
                        region=node_info['Tags'][1] if len(node_info['Tags']) > 1 else "default"
                    )
                    
                    self.nodes[node_id] = node
                    logger.info(f"Discovered new node: {node}")
                    
        except Exception as e:
            logger.error(f"Failed to discover nodes: {e}")
    
    def start_health_check(self):
        """启动健康检查"""
        if not self.health_check_active:
            self.health_check_active = True
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self.health_check_thread.start()
            logger.info("Health check started")
    
    def stop_health_check(self):
        """停止健康检查"""
        self.health_check_active = False
        if self.health_check_thread:
            self.health_check_thread.join()
        logger.info("Health check stopped")
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self.health_check_active:
            try:
                # 检查所有节点
                for node_id, node in list(self.nodes.items()):
                    if node_id == self.node_id:
                        continue
                        
                    # 检查节点健康状态
                    is_healthy = self._check_node_health(node)
                    
                    if not is_healthy:
                        # 更新节点状态
                        node.status = NodeStatus.FAILED
                        node.last_heartbeat = datetime.now()
                        
                        # 触发故障转移
                        if self.failover_enabled:
                            self._handle_failover(node)
                    else:
                        if node.status == NodeStatus.FAILED:
                            node.status = NodeStatus.ACTIVE
                            
                # 更新本地节点状态
                if self.local_node:
                    self.local_node.load = self._get_local_load()
                    self.local_node.last_heartbeat = datetime.now()
                    
                # 服务发现
                self.discover_nodes()
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_node_health(self, node: ClusterNode) -> bool:
        """检查节点健康状态"""
        try:
            # 简单的HTTP健康检查
            url = f"http://{node.host}:{node.port}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _get_local_load(self) -> float:
        """获取本地节点负载"""
        try:
            # 计算综合负载
            cpu_load = psutil.cpu_percent() / 100
            memory_load = psutil.virtual_memory().percent / 100
            
            # 加权平均
            load = 0.6 * cpu_load + 0.4 * memory_load
            return min(load, 1.0)
        except:
            return 0.5
    
    def _handle_failover(self, failed_node: ClusterNode):
        """处理故障转移"""
        logger.warning(f"Handling failover for node: {failed_node.id}")
        
        # 重新分配该节点的任务
        # 这里应该根据具体的任务类型进行处理
        # 实际实现中需要更复杂的逻辑
        
    def select_node(self, role: NodeRole = None, region: str = None) -> Optional[ClusterNode]:
        """选择节点（负载均衡）"""
        candidates = []
        
        for node in self.nodes.values():
            if node.status != NodeStatus.ACTIVE:
                continue
                
            if role and node.role != role:
                continue
                
            if region and node.region != region:
                continue
                
            candidates.append(node)
        
        if not candidates:
            return None
            
        # 使用负载均衡器选择节点
        return self.load_balancer.select(candidates)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        status = {
            "total_nodes": len(self.nodes),
            "active_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.ACTIVE),
            "failed_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.FAILED),
            "regions": {},
            "roles": {}
        }
        
        # 按区域统计
        for node in self.nodes.values():
            if node.region not in status["regions"]:
                status["regions"][node.region] = {"total": 0, "active": 0}
            status["regions"][node.region]["total"] += 1
            if node.status == NodeStatus.ACTIVE:
                status["regions"][node.region]["active"] += 1
                
        # 按角色统计
        for node in self.nodes.values():
            if node.role.value not in status["roles"]:
                status["roles"][node.role.value] = {"total": 0, "active": 0}
            status["roles"][node.role.value]["total"] += 1
            if node.status == NodeStatus.ACTIVE:
                status["roles"][node.role.value]["active"] += 1
                
        return status


class RoundRobinLoadBalancer:
    """轮询负载均衡器"""
    
    def __init__(self):
        self.current_index = 0
        self.lock = threading.Lock()
    
    def select(self, nodes: List[ClusterNode]) -> ClusterNode:
        """选择节点"""
        if not nodes:
            raise ValueError("No nodes available")
            
        with self.lock:
            # 按负载排序
            sorted_nodes = sorted(nodes, key=lambda n: n.load)
            node = sorted_nodes[self.current_index % len(sorted_nodes)]
            self.current_index += 1
            return node


class DistributedModelMonitor:
    """分布式模型监控器"""
    
    def __init__(self, node_id: str = None, config: Dict[str, Any] = None):
        """
        初始化分布式模型监控器
        
        Args:
            node_id: 节点ID
            config: 配置参数
        """
        self.config = config or {}
        
        # 集群管理
        self.cluster_manager = ClusterManager(node_id)
        
        # 分布式指标收集器
        self.metrics_collector = DistributedMetricsCollector()
        
        # 本地监控器
        self.local_monitor = None
        
        # 任务调度器
        self.task_scheduler = TaskScheduler()
        
        # 数据聚合器
        self.data_aggregator = DataAggregator()
        
        # 告警管理器
        self.alert_manager = DistributedAlertManager()
        
        # 存储后端
        self.storage_backend = self._init_storage_backend()
        
        # WebSocket服务器
        self.websocket_server = None
        self.websocket_port = self.config.get("websocket_port", 8765)
        
        # API服务器
        self.api_server = None
        self.api_port = self.config.get("api_port", 8080)
        
        # 初始化
        self._initialize()
        
    def _initialize(self):
        """初始化组件"""
        # 初始化本地节点
        role = NodeRole(self.config.get("node_role", "monitor"))
        region = self.config.get("region", "default")
        self.cluster_manager.initialize_local_node(
            role=role,
            port=self.websocket_port,
            region=region
        )
        
        # 启用服务发现
        if self.config.get("consul_enabled", False):
            self.cluster_manager.enable_consul(
                host=self.config.get("consul_host", "localhost"),
                port=self.config.get("consul_port", 8500)
            )
        
        # 启用Redis
        if self.config.get("redis_enabled", False):
            self.metrics_collector.enable_redis(
                host=self.config.get("redis_host", "localhost"),
                port=self.config.get("redis_port", 6379)
            )
        
        # 启动健康检查
        self.cluster_manager.start_health_check()
        
        # 初始化本地监控器
        self.local_monitor = ModelMonitor(
            db_path=f"monitoring_{self.cluster_manager.node_id}.db"
        )
        
        logger.info("DistributedModelMonitor initialized")
    
    def _init_storage_backend(self) -> str:
        """初始化存储后端"""
        storage_type = self.config.get("storage_backend", "sqlite")
        
        if storage_type == "redis":
            if self.metrics_collector.redis_enabled:
                return "redis"
            else:
                logger.warning("Redis not available, falling back to SQLite")
                return "sqlite"
        else:
            return "sqlite"
    
    def start(self):
        """启动分布式监控"""
        logger.info("Starting distributed model monitoring...")
        
        # 启动本地监控
        self.local_monitor.start_monitoring(interval=60)
        
        # 启动任务调度器
        self.task_scheduler.start()
        
        # 启动数据聚合器
        self.data_aggregator.start()
        
        # 启动WebSocket服务器
        self._start_websocket_server()
        
        # 启动API服务器
        self._start_api_server()
        
        logger.info("Distributed model monitoring started")
    
    def stop(self):
        """停止分布式监控"""
        logger.info("Stopping distributed model monitoring...")
        
        # 停止API服务器
        self._stop_api_server()
        
        # 停止WebSocket服务器
        self._stop_websocket_server()
        
        # 停止数据聚合器
        self.data_aggregator.stop()
        
        # 停止任务调度器
        self.task_scheduler.stop()
        
        # 停止本地监控
        self.local_monitor.stop_monitoring()
        
        # 停止健康检查
        self.cluster_manager.stop_health_check()
        
        logger.info("Distributed model monitoring stopped")
    
    def _start_websocket_server(self):
        """启动WebSocket服务器"""
        async def handle_client(websocket, path):
            """处理WebSocket客户端"""
            try:
                while True:
                    # 发送集群状态
                    cluster_status = self.cluster_manager.get_cluster_status()
                    await websocket.send(json.dumps({
                        "type": "cluster_status",
                        "data": cluster_status
                    }))
                    
                    # 发送聚合指标
                    aggregated_metrics = self.data_aggregator.get_latest_metrics()
                    await websocket.send(json.dumps({
                        "type": "aggregated_metrics",
                        "data": aggregated_metrics
                    }))
                    
                    await asyncio.sleep(5)
            except websockets.exceptions.ConnectionClosed:
                pass
        
        async def server():
            """WebSocket服务器"""
            self.websocket_server = await websockets.serve(
                handle_client,
                "0.0.0.0",
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
    
    def _start_api_server(self):
        """启动API服务器"""
        # 这里应该实现一个HTTP API服务器
        # 简化版本，只启动一个基本的服务器
        api_thread = threading.Thread(
            target=self._run_api_server,
            daemon=True
        )
        api_thread.start()
    
    def _run_api_server(self):
        """运行API服务器"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class APIHandler(BaseHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.monitor = self
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "healthy"}).encode())
                elif self.path == "/cluster/status":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    status = self.monitor.cluster_manager.get_cluster_status()
                    self.wfile.write(json.dumps(status).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
        
        # 绑定monitor实例
        APIHandler.monitor = self
        
        server = HTTPServer(("0.0.0.0", self.api_port), APIHandler)
        logger.info(f"API server started on port {self.api_port}")
        server.serve_forever()
    
    def _stop_api_server(self):
        """停止API服务器"""
        # 简化版本，实际实现需要优雅停止
        logger.info("API server stopped")
    
    def add_model(self, model_id: str, version_id: str, model: nn.Module,
                  config: Dict[str, Any] = None):
        """添加要监控的模型"""
        # 添加到本地监控器
        self.local_monitor.add_model(model_id, version_id, model, config)
        
        # 创建监控任务
        task = MonitoringTask(
            id=f"monitor_{model_id}_{version_id}",
            model_id=model_id,
            version_id=version_id,
            node_id=self.cluster_manager.node_id,
            schedule="*/1 * * * *",  # 每分钟执行
            config=config or {}
        )
        
        self.task_scheduler.add_task(task)
        
        logger.info(f"Added model to distributed monitoring: {model_id}")
    
    def collect_distributed_metrics(self, model_id: str, version_id: str) -> Dict[str, Any]:
        """收集分布式指标"""
        # 获取活跃节点
        active_nodes = [
            node for node in self.cluster_manager.nodes.values()
            if node.status == NodeStatus.ACTIVE
        ]
        
        # 收集指标
        metrics = self.metrics_collector.collect_distributed_metrics(
            active_nodes, model_id, version_id
        )
        
        # 存储聚合指标
        self.data_aggregator.store_aggregated_metrics(metrics)
        
        return metrics
    
    def get_cluster_metrics(self, model_id: str = None, 
                           start_time: datetime = None,
                           end_time: datetime = None) -> Dict[str, Any]:
        """获取集群指标"""
        return self.data_aggregator.get_aggregated_metrics(
            model_id, start_time, end_time
        )
    
    def generate_cluster_report(self, output_path: str = None) -> str:
        """生成集群报告"""
        if output_path is None:
            output_path = f"cluster_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 获取集群状态
        cluster_status = self.cluster_manager.get_cluster_status()
        
        # 获取聚合指标
        aggregated_metrics = self.data_aggregator.get_latest_metrics()
        
        # 生成报告
        report = f"# Distributed Model Monitoring Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Cluster Status\n\n"
        report += f"- Total Nodes: {cluster_status['total_nodes']}\n"
        report += f"- Active Nodes: {cluster_status['active_nodes']}\n"
        report += f"- Failed Nodes: {cluster_status['failed_nodes']}\n\n"
        
        report += "### Regions\n\n"
        report += "| Region | Total | Active |\n"
        report += "|--------|-------|--------|\n"
        for region, stats in cluster_status['regions'].items():
            report += f"| {region} | {stats['total']} | {stats['active']} |\n"
        
        report += "\n### Roles\n\n"
        report += "| Role | Total | Active |\n"
        report += "|------|-------|--------|\n"
        for role, stats in cluster_status['roles'].items():
            report += f"| {role} | {stats['total']} | {stats['active']} |\n"
        
        if aggregated_metrics:
            report += "\n## Aggregated Metrics\n\n"
            report += "```json\n"
            report += json.dumps(aggregated_metrics, indent=2)
            report += "\n```\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Cluster report saved to: {output_path}")
        return output_path


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self):
        self.tasks: Dict[str, MonitoringTask] = {}
        self.scheduler_thread = None
        self.scheduler_active = False
        
    def start(self):
        """启动调度器"""
        if not self.scheduler_active:
            self.scheduler_active = True
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True
            )
            self.scheduler_thread.start()
            logger.info("Task scheduler started")
    
    def stop(self):
        """停止调度器"""
        self.scheduler_active = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
        logger.info("Task scheduler stopped")
    
    def add_task(self, task: MonitoringTask):
        """添加任务"""
        self.tasks[task.id] = task
        logger.info(f"Added task: {task.id}")
    
    def remove_task(self, task_id: str):
        """移除任务"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            logger.info(f"Removed task: {task_id}")
    
    def _scheduler_loop(self):
        """调度器循环"""
        while self.scheduler_active:
            try:
                now = datetime.now()
                
                # 检查并执行到期任务
                for task in list(self.tasks.values()):
                    if task.next_run and now >= task.next_run:
                        self._execute_task(task)
                
                # 计算下次运行时间
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(10)
    
    def _execute_task(self, task: MonitoringTask):
        """执行任务"""
        try:
            task.status = "running"
            task.last_run = datetime.now()
            
            # 执行监控任务
            # 这里应该根据任务类型执行具体的监控逻辑
            logger.info(f"Executing task: {task.id}")
            
            task.status = "completed"
            
            # 计算下次运行时间（简化版本）
            task.next_run = task.last_run + timedelta(minutes=1)
            
        except Exception as e:
            task.status = "failed"
            logger.error(f"Failed to execute task {task.id}: {e}")


class DataAggregator:
    """数据聚合器"""
    
    def __init__(self):
        self.aggregated_data: Dict[str, List] = defaultdict(list)
        self.aggregator_thread = None
        self.aggregator_active = False
        
    def start(self):
        """启动聚合器"""
        if not self.aggregator_active:
            self.aggregator_active = True
            self.aggregator_thread = threading.Thread(
                target=self._aggregator_loop,
                daemon=True
            )
            self.aggregator_thread.start()
            logger.info("Data aggregator started")
    
    def stop(self):
        """停止聚合器"""
        self.aggregator_active = False
        if self.aggregator_thread:
            self.aggregator_thread.join()
        logger.info("Data aggregator stopped")
    
    def store_aggregated_metrics(self, metrics: Dict[str, Any]):
        """存储聚合指标"""
        model_id = metrics.get("model_id", "default")
        self.aggregated_data[model_id].append(metrics)
        
        # 保持最近1000条记录
        if len(self.aggregated_data[model_id]) > 1000:
            self.aggregated_data[model_id] = self.aggregated_data[model_id][-1000:]
    
    def get_aggregated_metrics(self, model_id: str = None,
                              start_time: datetime = None,
                              end_time: datetime = None) -> Dict[str, Any]:
        """获取聚合指标"""
        if model_id:
            data = self.aggregated_data.get(model_id, [])
        else:
            # 获取所有模型的最新数据
            data = []
            for model_data in self.aggregated_data.values():
                if model_data:
                    data.append(model_data[-1])
        
        # 时间过滤
        if start_time or end_time:
            filtered_data = []
            for item in data:
                timestamp = datetime.fromisoformat(item["timestamp"])
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                filtered_data.append(item)
            data = filtered_data
        
        return {
            "count": len(data),
            "data": data
        }
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """获取最新指标"""
        latest = {}
        for model_id, data in self.aggregated_data.items():
            if data:
                latest[model_id] = data[-1]
        return latest
    
    def _aggregator_loop(self):
        """聚合器循环"""
        while self.aggregator_active:
            try:
                # 执行数据聚合逻辑
                # 例如：计算滑动平均、统计指标等
                time.sleep(60)
            except Exception as e:
                logger.error(f"Aggregator error: {e}")
                time.sleep(60)


class DistributedAlertManager:
    """分布式告警管理器"""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []
        
    def add_alert_rule(self, rule_id: str, rule: Dict[str, Any]):
        """添加告警规则"""
        self.alert_rules[rule_id] = rule
        logger.info(f"Added alert rule: {rule_id}")
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """检查告警"""
        alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            # 检查规则条件
            if self._evaluate_rule(rule, metrics):
                alert = {
                    "id": f"alert_{uuid.uuid4().hex}",
                    "rule_id": rule_id,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics
                }
                alerts.append(alert)
                self.alert_history.append(alert)
        
        # 发送告警
        for alert in alerts:
            self._send_alert(alert)
    
    def _evaluate_rule(self, rule: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """评估告警规则"""
        # 简化版本，实际实现应该支持复杂的规则表达式
        condition = rule.get("condition", {})
        
        for metric_path, threshold in condition.items():
            # 解析指标路径，例如 "aggregated.cpu.avg"
            value = self._get_metric_value(metrics, metric_path)
            
            if value is None:
                continue
                
            operator = rule.get("operator", ">")
            
            if operator == ">" and value > threshold:
                return True
            elif operator == "<" and value < threshold:
                return True
            elif operator == "==" and value == threshold:
                return True
                
        return False
    
    def _get_metric_value(self, metrics: Dict[str, Any], path: str) -> Optional[float]:
        """获取指标值"""
        keys = path.split(".")
        value = metrics
        
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                else:
                    return None
            return float(value)
        except:
            return None
    
    def _send_alert(self, alert: Dict[str, Any]):
        """发送告警"""
        # 简化版本，实际实现应该支持多种通知渠道
        logger.warning(f"Alert triggered: {alert['rule_id']}")
        
        # 这里可以集成邮件、Slack、Webhook等通知方式


def create_distributed_monitor(node_id: str = None, 
                              config: Dict[str, Any] = None) -> DistributedModelMonitor:
    """创建分布式监控器实例"""
    return DistributedModelMonitor(node_id=node_id, config=config)