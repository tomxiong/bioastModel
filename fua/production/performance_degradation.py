"""
模型性能降级检测系统

提供实时模型性能监控、降级检测、根本原因分析和
自动化恢复建议，确保模型在生产环境中的持续高性能
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
import sqlite3
import pandas as pd
from scipy import stats
from collections import deque, defaultdict
import pickle
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DegradationType(Enum):
    """降级类型"""
    ACCURACY_DROP = "accuracy_drop"
    LATENCY_INCREASE = "latency_increase"
    ERROR_RATE_SPIKE = "error_rate_spike"
    MEMORY_LEAK = "memory_leak"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    MODEL_STALITY = "model_staleness"
    INFERENCE_CONSISTENCY = "inference_consistency"


class SeverityLevel(Enum):
    """严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(Enum):
    """检测方法"""
    STATISTICAL = "statistical"  # 统计检验
    THRESHOLD_BASED = "threshold_based"  # 基于阈值
    TREND_ANALYSIS = "trend_analysis"  # 趋势分析
    ANOMALY_DETECTION = "anomaly_detection"  # 异常检测
    COMPARATIVE = "comparative"  # 对比分析


@dataclass
class PerformanceBaseline:
    """性能基线"""
    model_id: str
    version_id: str
    metric_name: str
    mean: float
    std: float
    min_value: float
    max_value: float
    percentile_5: float
    percentile_95: float
    created_at: datetime = field(default_factory=datetime.now)
    sample_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """从字典创建"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class DegradationEvent:
    """降级事件"""
    id: str
    model_id: str
    version_id: str
    degradation_type: DegradationType
    severity: SeverityLevel
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_score: float
    detection_method: DetectionMethod
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    root_causes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_action: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['degradation_type'] = self.degradation_type.value
        data['severity'] = self.severity.value
        data['detection_method'] = self.detection_method.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DegradationEvent':
        """从字典创建"""
        data['degradation_type'] = DegradationType(data['degradation_type'])
        data['severity'] = SeverityLevel(data['severity'])
        data['detection_method'] = DetectionMethod(data['detection_method'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('resolved_at'):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        return cls(**data)


@dataclass
class DetectionConfig:
    """检测配置"""
    metric_name: str
    degradation_type: DegradationType
    detection_method: DetectionMethod
    threshold: float = 0.1  # 降级阈值（10%）
    min_samples: int = 100  # 最小样本数
    window_size: int = 100  # 滑动窗口大小
    statistical_test: str = "t_test"  # 统计检验方法
    sensitivity: float = 2.0  # 灵敏度（标准差倍数）
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['degradation_type'] = self.degradation_type.value
        data['detection_method'] = self.detection_method.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionConfig':
        """从字典创建"""
        data['degradation_type'] = DegradationType(data['degradation_type'])
        data['detection_method'] = DetectionMethod(data['detection_method'])
        return cls(**data)


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, model: nn.Module):
        """
        初始化性能分析器
        
        Args:
            model: 要分析的模型
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    def profile_inference(self, input_data: torch.Tensor, 
                         warmup_runs: int = 10,
                         profile_runs: int = 100) -> Dict[str, Any]:
        """
        分析推理性能
        
        Args:
            input_data: 输入数据
            warmup_runs: 预热运行次数
            profile_runs: 分析运行次数
            
        Returns:
            性能分析结果
        """
        self.model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_data.to(self.device))
        
        # GPU同步（如果使用GPU）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 性能分析
        latencies = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(profile_runs):
                # 记录开始时间
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                # 推理
                output = self.model(input_data.to(self.device))
                
                # 记录结束时间
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                # 记录内存使用
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)  # MB
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
            'peak_memory_mb': np.max(memory_usage) if memory_usage else 0,
            'throughput_qps': 1000 / np.mean(latencies),
            'raw_latencies': latencies
        }


class StatisticalDetector:
    """统计检测器"""
    
    @staticmethod
    def detect_drift(current_values: List[float], 
                    baseline: PerformanceBaseline,
                    method: str = "t_test",
                    threshold: float = 0.05) -> Tuple[bool, float, str]:
        """
        检测性能漂移
        
        Args:
            current_values: 当前值列表
            baseline: 性能基线
            method: 检测方法
            threshold: 显著性阈值
            
        Returns:
            (是否检测到漂移, p值, 描述)
        """
        if len(current_values) < 10:
            return False, 1.0, "Insufficient data"
        
        current_mean = np.mean(current_values)
        
        if method == "t_test":
            # 生成基线分布（假设正态分布）
            baseline_samples = np.random.normal(
                baseline.mean, 
                baseline.std, 
                len(current_values)
            )
            
            # 执行t检验
            _, p_value = stats.ttest_ind(current_values, baseline_samples)
            
            # 计算相对变化
            relative_change = abs(current_mean - baseline.mean) / baseline.mean
            
            is_drift = p_value < threshold and relative_change > 0.05  # 5%最小变化
            
            return is_drift, p_value, f"T-test: p={p_value:.4f}, change={relative_change:.2%}"
        
        elif method == "z_score":
            # Z分数检测
            z_score = abs(current_mean - baseline.mean) / baseline.std
            
            is_drift = z_score > 2.0  # 2个标准差
            
            return is_drift, z_score, f"Z-score: {z_score:.2f}"
        
        elif method == "ks_test":
            # Kolmogorov-Smirnov检验
            baseline_samples = np.random.normal(
                baseline.mean, 
                baseline.std, 
                len(current_values)
            )
            
            _, p_value = stats.ks_2samp(current_values, baseline_samples)
            
            is_drift = p_value < threshold
            
            return is_drift, p_value, f"KS-test: p={p_value:.4f}"
        
        return False, 1.0, "Unknown method"
    
    @staticmethod
    def detect_trend(values: List[float], 
                     window_size: int = 20,
                     threshold: float = 0.1) -> Tuple[bool, float, str]:
        """
        检测趋势变化
        
        Args:
            values: 值列表
            window_size: 窗口大小
            threshold: 趋势阈值
            
        Returns:
            (是否检测到趋势, 趋势斜率, 描述)
        """
        if len(values) < window_size * 2:
            return False, 0.0, "Insufficient data for trend detection"
        
        # 计算最近窗口和之前窗口的趋势
        recent_window = values[-window_size:]
        previous_window = values[-2*window_size:-window_size]
        
        # 线性回归计算趋势
        x_recent = np.arange(len(recent_window))
        x_previous = np.arange(len(previous_window))
        
        slope_recent, _, _, _, _ = stats.linregress(x_recent, recent_window)
        slope_previous, _, _, _, _ = stats.linregress(x_previous, previous_window)
        
        # 计算趋势变化
        trend_change = abs(slope_recent - slope_previous)
        
        # 检查趋势方向
        if abs(slope_recent) > threshold:
            direction = "increasing" if slope_recent > 0 else "decreasing"
            return True, slope_recent, f"Strong {direction} trend: slope={slope_recent:.4f}"
        
        return False, slope_recent, f"No significant trend: slope={slope_recent:.4f}"


class DegradationAnalyzer:
    """降级分析器"""
    
    def __init__(self, db_path: str = "degradation.db"):
        """
        初始化降级分析器
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.baselines: Dict[str, Dict[str, PerformanceBaseline]] = {}
        self.detection_configs: List[DetectionConfig] = []
        self.active_events: Dict[str, DegradationEvent] = {}
        
        # 性能历史数据
        self.performance_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )
        
        # 统计检测器
        self.statistical_detector = StatisticalDetector()
        
        # 初始化数据库
        self._init_db()
        
        # 加载数据
        self._load_baselines()
        self._load_configs()
        
        logger.info("DegradationAnalyzer initialized")
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基线表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                mean REAL NOT NULL,
                std REAL NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                percentile_5 REAL NOT NULL,
                percentile_95 REAL NOT NULL,
                created_at TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                UNIQUE(model_id, version_id, metric_name)
            )
        ''')
        
        # 检测配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                degradation_type TEXT NOT NULL,
                detection_method TEXT NOT NULL,
                threshold REAL NOT NULL,
                min_samples INTEGER NOT NULL,
                window_size INTEGER NOT NULL,
                statistical_test TEXT NOT NULL,
                sensitivity REAL NOT NULL,
                enabled INTEGER NOT NULL
            )
        ''')
        
        # 降级事件表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS degradation_events (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version_id TEXT NOT NULL,
                degradation_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                current_value REAL NOT NULL,
                baseline_value REAL NOT NULL,
                degradation_score REAL NOT NULL,
                detection_method TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                description TEXT,
                root_causes TEXT,
                recommendations TEXT,
                resolved INTEGER NOT NULL,
                resolved_at TEXT,
                resolution_action TEXT
            )
        ''')
        
        # 性能数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_baselines(self):
        """加载性能基线"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM baselines')
        rows = cursor.fetchall()
        
        for row in rows:
            baseline = PerformanceBaseline(
                model_id=row[1],
                version_id=row[2],
                metric_name=row[3],
                mean=row[4],
                std=row[5],
                min_value=row[6],
                max_value=row[7],
                percentile_5=row[8],
                percentile_95=row[9],
                created_at=datetime.fromisoformat(row[10]),
                sample_size=row[11]
            )
            
            key = f"{baseline.model_id}_{baseline.version_id}"
            if key not in self.baselines:
                self.baselines[key] = {}
            self.baselines[key][baseline.metric_name] = baseline
        
        conn.close()
        logger.info(f"Loaded {len(rows)} performance baselines")
    
    def _load_configs(self):
        """加载检测配置"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM detection_configs')
        rows = cursor.fetchall()
        
        for row in rows:
            config = DetectionConfig(
                metric_name=row[1],
                degradation_type=DegradationType(row[2]),
                detection_method=DetectionMethod(row[3]),
                threshold=row[4],
                min_samples=row[5],
                window_size=row[6],
                statistical_test=row[7],
                sensitivity=row[8],
                enabled=bool(row[9])
            )
            self.detection_configs.append(config)
        
        conn.close()
        
        # 如果没有配置，添加默认配置
        if not self.detection_configs:
            self._add_default_configs()
        
        logger.info(f"Loaded {len(self.detection_configs)} detection configs")
    
    def _add_default_configs(self):
        """添加默认检测配置"""
        default_configs = [
            DetectionConfig(
                metric_name="accuracy",
                degradation_type=DegradationType.ACCURACY_DROP,
                detection_method=DetectionMethod.STATISTICAL,
                threshold=0.05,  # 5%下降
                min_samples=100,
                window_size=100,
                statistical_test="t_test",
                sensitivity=2.0
            ),
            DetectionConfig(
                metric_name="latency",
                degradation_type=DegradationType.LATENCY_INCREASE,
                detection_method=DetectionMethod.THRESHOLD_BASED,
                threshold=0.2,  # 20%增加
                min_samples=50,
                window_size=50,
                statistical_test="z_score",
                sensitivity=2.0
            ),
            DetectionConfig(
                metric_name="error_rate",
                degradation_type=DegradationType.ERROR_RATE_SPIKE,
                detection_method=DetectionMethod.STATISTICAL,
                threshold=0.1,  # 10%增加
                min_samples=100,
                window_size=100,
                statistical_test="t_test",
                sensitivity=2.0
            ),
            DetectionConfig(
                metric_name="memory_usage",
                degradation_type=DegradationType.MEMORY_LEAK,
                detection_method=DetectionMethod.TREND_ANALYSIS,
                threshold=0.1,
                min_samples=200,
                window_size=200,
                statistical_test="trend",
                sensitivity=1.5
            )
        ]
        
        for config in default_configs:
            self.add_detection_config(config)
    
    def establish_baseline(self, model_id: str, version_id: str,
                         metric_data: Dict[str, List[float]],
                         min_samples: int = 100) -> Dict[str, PerformanceBaseline]:
        """
        建立性能基线
        
        Args:
            model_id: 模型ID
            version_id: 版本ID
            metric_data: 指标数据字典
            min_samples: 最小样本数
            
        Returns:
            建立的基线字典
        """
        baselines = {}
        
        for metric_name, values in metric_data.items():
            if len(values) < min_samples:
                logger.warning(f"Insufficient data for {metric_name}: {len(values)} < {min_samples}")
                continue
            
            # 计算统计量
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            p5 = np.percentile(values, 5)
            p95 = np.percentile(values, 95)
            
            # 创建基线
            baseline = PerformanceBaseline(
                model_id=model_id,
                version_id=version_id,
                metric_name=metric_name,
                mean=mean,
                std=std,
                min_value=min_val,
                max_value=max_val,
                percentile_5=p5,
                percentile_95=p95,
                sample_size=len(values)
            )
            
            baselines[metric_name] = baseline
            
            # 保存到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO baselines 
                (model_id, version_id, metric_name, mean, std, min_value, max_value,
                 percentile_5, percentile_95, created_at, sample_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id, version_id, metric_name, mean, std, min_val, max_val,
                p5, p95, baseline.created_at.isoformat(), len(values)
            ))
            
            conn.commit()
            conn.close()
        
        # 更新内存中的基线
        key = f"{model_id}_{version_id}"
        self.baselines[key] = baselines
        
        logger.info(f"Established baseline for {model_id}:{version_id} with {len(baselines)} metrics")
        return baselines
    
    def add_detection_config(self, config: DetectionConfig):
        """添加检测配置"""
        self.detection_configs.append(config)
        
        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detection_configs 
            (metric_name, degradation_type, detection_method, threshold, min_samples,
             window_size, statistical_test, sensitivity, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            config.metric_name,
            config.degradation_type.value,
            config.detection_method.value,
            config.threshold,
            config.min_samples,
            config.window_size,
            config.statistical_test,
            config.sensitivity,
            int(config.enabled)
        ))
        
        conn.commit()
        conn.close()
    
    def record_performance(self, model_id: str, version_id: str,
                          metric_name: str, value: float,
                          context: Dict[str, Any] = None):
        """记录性能数据"""
        # 保存到内存
        key = f"{model_id}_{version_id}"
        self.performance_history[key][metric_name].append(value)
        
        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_data (model_id, version_id, metric_name, value, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_id, version_id, metric_name, value,
            datetime.now().isoformat(),
            json.dumps(context) if context else None
        ))
        
        conn.commit()
        conn.close()
    
    def detect_degradation(self, model_id: str, version_id: str) -> List[DegradationEvent]:
        """
        检测性能降级
        
        Args:
            model_id: 模型ID
            version_id: 版本ID
            
        Returns:
            检测到的降级事件列表
        """
        key = f"{model_id}_{version_id}"
        if key not in self.baselines:
            logger.warning(f"No baseline found for {model_id}:{version_id}")
            return []
        
        events = []
        baselines = self.baselines[key]
        
        for config in self.detection_configs:
            if not config.enabled:
                continue
            
            if config.metric_name not in baselines:
                continue
            
            # 获取当前数据
            current_data = list(self.performance_history[key][config.metric_name])
            if len(current_data) < config.min_samples:
                continue
            
            baseline = baselines[config.metric_name]
            current_values = current_data[-config.window_size:]
            
            # 根据检测方法进行检测
            if config.detection_method == DetectionMethod.STATISTICAL:
                is_degraded, p_value, description = self.statistical_detector.detect_drift(
                    current_values, baseline, config.statistical_test, config.threshold
                )
                
                if is_degraded:
                    current_mean = np.mean(current_values)
                    degradation_score = abs(current_mean - baseline.mean) / baseline.mean
                    
                    event = DegradationEvent(
                        id=f"{model_id}_{version_id}_{config.metric_name}_{int(time.time())}",
                        model_id=model_id,
                        version_id=version_id,
                        degradation_type=config.degradation_type,
                        severity=self._calculate_severity(degradation_score),
                        metric_name=config.metric_name,
                        current_value=current_mean,
                        baseline_value=baseline.mean,
                        degradation_score=degradation_score,
                        detection_method=config.detection_method,
                        description=description
                    )
                    
                    # 分析根本原因
                    event.root_causes = self._analyze_root_causes(event, current_values, baseline)
                    
                    # 生成建议
                    event.recommendations = self._generate_recommendations(event)
                    
                    events.append(event)
            
            elif config.detection_method == DetectionMethod.TREND_ANALYSIS:
                is_trending, slope, description = self.statistical_detector.detect_trend(
                    current_values, config.window_size, config.threshold
                )
                
                if is_trending:
                    current_mean = np.mean(current_values)
                    degradation_score = abs(slope) * 100  # 将斜率转换为百分比
                    
                    event = DegradationEvent(
                        id=f"{model_id}_{version_id}_{config.metric_name}_{int(time.time())}",
                        model_id=model_id,
                        version_id=version_id,
                        degradation_type=config.degradation_type,
                        severity=self._calculate_severity(degradation_score),
                        metric_name=config.metric_name,
                        current_value=current_mean,
                        baseline_value=baseline.mean,
                        degradation_score=degradation_score,
                        detection_method=config.detection_method,
                        description=description
                    )
                    
                    events.append(event)
            
            elif config.detection_method == DetectionMethod.THRESHOLD_BASED:
                current_mean = np.mean(current_values)
                relative_change = abs(current_mean - baseline.mean) / baseline.mean
                
                if relative_change > config.threshold:
                    degradation_score = relative_change
                    
                    event = DegradationEvent(
                        id=f"{model_id}_{version_id}_{config.metric_name}_{int(time.time())}",
                        model_id=model_id,
                        version_id=version_id,
                        degradation_type=config.degradation_type,
                        severity=self._calculate_severity(degradation_score),
                        metric_name=config.metric_name,
                        current_value=current_mean,
                        baseline_value=baseline.mean,
                        degradation_score=degradation_score,
                        detection_method=config.detection_method,
                        description=f"Threshold exceeded: {relative_change:.2%} > {config.threshold:.2%}"
                    )
                    
                    events.append(event)
        
        # 保存事件
        for event in events:
            self._save_degradation_event(event)
            self.active_events[event.id] = event
        
        return events
    
    def _calculate_severity(self, degradation_score: float) -> SeverityLevel:
        """计算严重程度"""
        if degradation_score >= 0.3:  # 30%
            return SeverityLevel.CRITICAL
        elif degradation_score >= 0.2:  # 20%
            return SeverityLevel.HIGH
        elif degradation_score >= 0.1:  # 10%
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _analyze_root_causes(self, event: DegradationEvent,
                            current_values: List[float],
                            baseline: PerformanceBaseline) -> List[str]:
        """分析根本原因"""
        causes = []
        
        if event.degradation_type == DegradationType.ACCURACY_DROP:
            causes.append("Potential data distribution shift")
            causes.append("Model may need retraining")
            causes.append("Input quality degradation")
        
        elif event.degradation_type == DegradationType.LATENCY_INCREASE:
            causes.append("Increased computational load")
            causes.append("Resource contention")
            causes.append("Network latency issues")
            causes.append("Hardware degradation")
        
        elif event.degradation_type == DegradationType.ERROR_RATE_SPIKE:
            causes.append("Input data anomalies")
            causes.append("Model overfitting")
            causes.append("Concept drift")
        
        elif event.degradation_type == DegradationType.MEMORY_LEAK:
            causes.append("Memory not properly released")
            causes.append("Accumulating intermediate results")
            causes.append("Framework or library issues")
        
        # 分析统计特征
        current_std = np.std(current_values)
        if current_std > baseline.std * 2:
            causes.append("Increased variability in performance")
        
        return causes[:3]  # 返回前3个最可能的原因
    
    def _generate_recommendations(self, event: DegradationEvent) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if event.degradation_type == DegradationType.ACCURACY_DROP:
            recommendations.append("Consider retraining with recent data")
            recommendations.append("Validate input data quality")
            recommendations.append("Check for data drift")
            recommendations.append("Consider model ensemble")
        
        elif event.degradation_type == DegradationType.LATENCY_INCREASE:
            recommendations.append("Optimize model architecture")
            recommendations.append("Consider model quantization")
            recommendations.append("Scale resources horizontally")
            recommendations.append("Implement caching mechanisms")
        
        elif event.degradation_type == DegradationType.ERROR_RATE_SPIKE:
            recommendations.append("Implement input validation")
            recommendations.append("Add fallback mechanisms")
            recommendations.append("Monitor data quality")
            recommendations.append("Consider model version rollback")
        
        elif event.degradation_type == DegradationType.MEMORY_LEAK:
            recommendations.append("Restart model service")
            recommendations.append("Investigate memory management")
            recommendations.append("Update framework version")
            recommendations.append("Implement memory monitoring")
        
        return recommendations
    
    def _save_degradation_event(self, event: DegradationEvent):
        """保存降级事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO degradation_events 
            (id, model_id, version_id, degradation_type, severity, metric_name,
             current_value, baseline_value, degradation_score, detection_method,
             timestamp, description, root_causes, recommendations, resolved,
             resolved_at, resolution_action)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.id,
            event.model_id,
            event.version_id,
            event.degradation_type.value,
            event.severity.value,
            event.metric_name,
            event.current_value,
            event.baseline_value,
            event.degradation_score,
            event.detection_method.value,
            event.timestamp.isoformat(),
            event.description,
            json.dumps(event.root_causes),
            json.dumps(event.recommendations),
            int(event.resolved),
            event.resolved_at.isoformat() if event.resolved_at else None,
            event.resolution_action
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_degradations(self, model_id: str = None) -> List[DegradationEvent]:
        """获取活跃的降级事件"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if model_id:
            cursor.execute('''
                SELECT * FROM degradation_events 
                WHERE model_id = ? AND resolved = 0
                ORDER BY timestamp DESC
            ''', (model_id,))
        else:
            cursor.execute('''
                SELECT * FROM degradation_events 
                WHERE resolved = 0
                ORDER BY timestamp DESC
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            event = DegradationEvent(
                id=row[0],
                model_id=row[1],
                version_id=row[2],
                degradation_type=DegradationType(row[3]),
                severity=SeverityLevel(row[4]),
                metric_name=row[5],
                current_value=row[6],
                baseline_value=row[7],
                degradation_score=row[8],
                detection_method=DetectionMethod(row[9]),
                timestamp=datetime.fromisoformat(row[10]),
                description=row[11] or "",
                root_causes=json.loads(row[12]) if row[12] else [],
                recommendations=json.loads(row[13]) if row[13] else [],
                resolved=bool(row[14]),
                resolved_at=datetime.fromisoformat(row[15]) if row[15] else None,
                resolution_action=row[16] or ""
            )
            events.append(event)
        
        return events
    
    def resolve_degradation(self, event_id: str, action: str):
        """解决降级事件"""
        if event_id in self.active_events:
            event = self.active_events[event_id]
            event.resolved = True
            event.resolved_at = datetime.now()
            event.resolution_action = action
            
            self._save_degradation_event(event)
            del self.active_events[event_id]
            
            logger.info(f"Resolved degradation event: {event_id}")
    
    def generate_degradation_report(self, model_id: str = None,
                                  output_path: str = None) -> str:
        """生成降级报告"""
        if output_path is None:
            output_path = f"degradation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 获取统计信息
        active_events = self.get_active_degradations(model_id)
        
        # 生成报告
        report = f"# Performance Degradation Report\\n\\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
        
        if model_id:
            report += f"**Model ID:** {model_id}\\n\\n"
        
        report += "## Summary\\n\\n"
        report += f"- Active degradation events: {len(active_events)}\\n\\n"
        
        # 按严重程度统计
        severity_counts = defaultdict(int)
        for event in active_events:
            severity_counts[event.severity.value] += 1
        
        if severity_counts:
            report += "### Severity Distribution\\n\\n"
            for severity in ['critical', 'high', 'medium', 'low']:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    report += f"- {severity.title()}: {count}\\n"
            report += "\\n"
        
        # 活跃事件详情
        if active_events:
            report += "## Active Degradation Events\\n\\n"
            report += "| ID | Model | Metric | Severity | Score | Detected |\\n"
            report += "|----|-------|--------|----------|--------|-----------|\\n"
            
            for event in sorted(active_events, key=lambda x: x.timestamp, reverse=True)[:10]:
                report += f"| {event.id[:8]}... | {event.model_id} | {event.metric_name} | "
                report += f"{event.severity.value} | {event.degradation_score:.2%} | {event.timestamp.strftime('%m-%d %H:%M')} |\\n"
            
            report += "\\n"
            
            # 事件详情
            report += "## Event Details\\n\\n"
            for event in sorted(active_events, key=lambda x: x.severity.value, reverse=True)[:5]:
                report += f"### {event.metric_name} ({event.severity.value.title()})\\n\\n"
                report += f"- **Event ID:** {event.id}\\n"
                report += f"- **Current Value:** {event.current_value:.4f}\\n"
                report += f"- **Baseline Value:** {event.baseline_value:.4f}\\n"
                report += f"- **Degradation Score:** {event.degradation_score:.2%}\\n"
                report += f"- **Detection Method:** {event.detection_method.value}\\n"
                report += f"- **Description:** {event.description}\\n\\n"
                
                if event.root_causes:
                    report += "**Potential Root Causes:**\\n"
                    for cause in event.root_causes:
                        report += f"- {cause}\\n"
                    report += "\\n"
                
                if event.recommendations:
                    report += "**Recommendations:**\\n"
                    for rec in event.recommendations:
                        report += f"- {rec}\\n"
                    report += "\\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Degradation report saved to: {output_path}")
        return output_path


def create_degradation_analyzer(db_path: str = "degradation.db") -> DegradationAnalyzer:
    """创建降级分析器实例"""
    return DegradationAnalyzer(db_path)