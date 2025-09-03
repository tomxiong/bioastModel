"""
FUA Production Module

提供生产环境所需的监控、告警、A/B测试和自动化功能，
确保模型在生产环境中的稳定运行和持续优化
"""

from .model_monitor import (
    ModelMonitor, MetricsCollector, AnomalyDetector, AlertManager,
    Alert, AlertSeverity, MetricType, AlertChannel, MetricThreshold,
    ModelMetrics, EmailNotifier, SlackNotifier, WebhookNotifier,
    create_model_monitor
)

from .ab_test_framework import (
    ABTestManager, ABTest, ABTestConfig, TestVariant, TestMetric,
    TestResult, TrafficAllocationStrategy, TestStatus, MetricType as ABMetricType,
    StatisticalTest, TrafficAllocator, StatisticalAnalyzer,
    create_ab_test_manager
)

from .performance_degradation import (
    DegradationAnalyzer, PerformanceBaseline, DegradationEvent, DetectionConfig,
    DegradationType, SeverityLevel, DetectionMethod, PerformanceProfiler,
    StatisticalDetector, create_degradation_analyzer
)

from .auto_rollback import (
    AutoRollbackManager, ModelVersionManager, RollbackExecutor,
    RollbackTrigger, RollbackStatus, ModelHealthStatus,
    ModelVersion, RollbackPlan, RollbackConfig,
    create_auto_rollback_manager
)

from .distributed_monitor import (
    DistributedModelMonitor, ClusterManager, DistributedMetricsCollector,
    ClusterNode, NodeStatus, NodeRole, MonitoringTask, TaskScheduler,
    DataAggregator, DistributedAlertManager, RoundRobinLoadBalancer,
    create_distributed_monitor
)

__all__ = [
    # Model Monitor
    'ModelMonitor',
    'MetricsCollector', 
    'AnomalyDetector',
    'AlertManager',
    'Alert',
    'AlertSeverity',
    'MetricType',
    'AlertChannel',
    'MetricThreshold',
    'ModelMetrics',
    'EmailNotifier',
    'SlackNotifier',
    'WebhookNotifier',
    'create_model_monitor',
    
    # A/B Testing Framework
    'ABTestManager',
    'ABTest',
    'ABTestConfig',
    'TestVariant',
    'TestMetric',
    'TestResult',
    'TrafficAllocationStrategy',
    'TestStatus',
    'ABMetricType',
    'StatisticalTest',
    'TrafficAllocator',
    'StatisticalAnalyzer',
    'create_ab_test_manager',
    
    # Performance Degradation Detection
    'DegradationAnalyzer',
    'PerformanceProfiler',
    'StatisticalDetector',
    'DegradationType',
    'SeverityLevel',
    'DetectionMethod',
    'PerformanceBaseline',
    'DegradationEvent',
    'DetectionConfig',
    'create_degradation_analyzer',
    
    # Auto Rollback System
    'AutoRollbackManager',
    'ModelVersionManager',
    'RollbackExecutor',
    'RollbackTrigger',
    'RollbackStatus',
    'ModelHealthStatus',
    'ModelVersion',
    'RollbackPlan',
    'RollbackConfig',
    'create_auto_rollback_manager',
    
    # Distributed Monitoring
    'DistributedModelMonitor',
    'ClusterManager',
    'DistributedMetricsCollector',
    'ClusterNode',
    'NodeStatus',
    'NodeRole',
    'MonitoringTask',
    'TaskScheduler',
    'DataAggregator',
    'DistributedAlertManager',
    'RoundRobinLoadBalancer',
    'create_distributed_monitor'
]

# 模块版本
__version__ = "1.0.0"