"""
模型A/B测试框架

提供模型A/B测试功能，包括流量分配、统计显著性检验、
结果分析和自动化决策，帮助用户科学地评估和选择最佳模型
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
from collections import defaultdict, deque
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class TrafficAllocationStrategy(Enum):
    """流量分配策略"""
    EQUAL = "equal"  # 平均分配
    WEIGHTED = "weighted"  # 加权分配
    BANDIT = "bandit"  # 多臂老虎机
    GRADUAL = "gradual"  # 渐进式分配


class TestStatus(Enum):
    """测试状态"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class MetricType(Enum):
    """指标类型"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CONVERSION_RATE = "conversion_rate"
    REVENUE_PER_USER = "revenue_per_user"
    CUSTOM = "custom"


class StatisticalTest(Enum):
    """统计检验方法"""
    T_TEST = "t_test"  # t检验
    Z_TEST = "z_test"  # z检验
    CHI_SQUARE = "chi_square"  # 卡方检验
    MANN_WHITNEY = "mann_whitney"  # Mann-Whitney U检验
    BOOTSTRAP = "bootstrap"  # 自助法


@dataclass
class TestVariant:
    """测试变体"""
    id: str
    name: str
    model_id: str
    version_id: str
    weight: float = 0.5
    is_control: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestVariant':
        """从字典创建"""
        return cls(**data)


@dataclass
class TestMetric:
    """测试指标"""
    name: str
    type: MetricType
    primary: bool = False
    improvement_direction: str = "higher"  # higher or lower
    min_detectable_effect: float = 0.01
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    custom_aggregator: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['type'] = self.type.value
        data['statistical_test'] = self.statistical_test.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMetric':
        """从字典创建"""
        data['type'] = MetricType(data['type'])
        data['statistical_test'] = StatisticalTest(data['statistical_test'])
        return cls(**data)


@dataclass
class ABTestConfig:
    """A/B测试配置"""
    name: str
    description: str = ""
    traffic_allocation_strategy: TrafficAllocationStrategy = TrafficAllocationStrategy.EQUAL
    duration_days: int = 7
    sample_size_per_variant: Optional[int] = None
    significance_level: float = 0.05
    power: float = 0.8
    variants: List[TestVariant] = field(default_factory=list)
    metrics: List[TestMetric] = field(default_factory=list)
    targeting_rules: Dict[str, Any] = field(default_factory=dict)
    min_runtime_hours: int = 24
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['traffic_allocation_strategy'] = self.traffic_allocation_strategy.value
        data['variants'] = [v.to_dict() for v in self.variants]
        data['metrics'] = [m.to_dict() for m in self.metrics]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTestConfig':
        """从字典创建"""
        data['traffic_allocation_strategy'] = TrafficAllocationStrategy(data['traffic_allocation_strategy'])
        data['variants'] = [TestVariant.from_dict(v) for v in data['variants']]
        data['metrics'] = [TestMetric.from_dict(m) for m in data['metrics']]
        return cls(**data)


@dataclass
class TestResult:
    """测试结果"""
    variant_id: str
    metric_name: str
    value: float
    count: int
    variance: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    p_value: float = 1.0
    effect_size: float = 0.0
    is_significant: bool = False
    is_winner: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestResult':
        """从字典创建"""
        return cls(**data)


@dataclass
class ABTest:
    """A/B测试"""
    id: str
    config: ABTestConfig
    status: TestStatus = TestStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    results: Dict[str, List[TestResult]] = field(default_factory=dict)
    winner_variant_id: Optional[str] = None
    decision_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['config'] = self.config.to_dict()
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.ended_at:
            data['ended_at'] = self.ended_at.isoformat()
        
        # 转换结果
        converted_results = {}
        for variant_id, results in self.results.items():
            converted_results[variant_id] = [r.to_dict() for r in results]
        data['results'] = converted_results
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ABTest':
        """从字典创建"""
        data['config'] = ABTestConfig.from_dict(data['config'])
        data['status'] = TestStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('ended_at'):
            data['ended_at'] = datetime.fromisoformat(data['ended_at'])
        
        # 转换结果
        converted_results = {}
        for variant_id, results in data.get('results', {}).items():
            converted_results[variant_id] = [TestResult.from_dict(r) for r in results]
        data['results'] = converted_results
        
        return cls(**data)


class TrafficAllocator:
    """流量分配器"""
    
    def __init__(self, strategy: TrafficAllocationStrategy = TrafficAllocationStrategy.EQUAL):
        """
        初始化流量分配器
        
        Args:
            strategy: 分配策略
        """
        self.strategy = strategy
        self.variant_stats: Dict[str, Dict] = defaultdict(lambda: {
            'impressions': 0,
            'conversions': 0,
            'revenue': 0.0,
            'performance_score': 0.5
        })
        
    def allocate_variant(self, variants: List[TestVariant], user_id: str = None,
                        context: Dict[str, Any] = None) -> TestVariant:
        """
        分配变体
        
        Args:
            variants: 变体列表
            user_id: 用户ID（用于一致性哈希）
            context: 上下文信息
            
        Returns:
            分配的变体
        """
        if not variants:
            raise ValueError("No variants provided")
        
        if self.strategy == TrafficAllocationStrategy.EQUAL:
            return self._equal_allocation(variants, user_id)
        elif self.strategy == TrafficAllocationStrategy.WEIGHTED:
            return self._weighted_allocation(variants, user_id)
        elif self.strategy == TrafficAllocationStrategy.BANDIT:
            return self._bandit_allocation(variants, user_id)
        elif self.strategy == TrafficAllocationStrategy.GRADUAL:
            return self._gradual_allocation(variants, user_id, context)
        else:
            return variants[0]
    
    def _equal_allocation(self, variants: List[TestVariant], user_id: str) -> TestVariant:
        """平均分配"""
        if user_id:
            # 使用一致性哈希确保同一用户总是分配到同一变体
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            index = hash_value % len(variants)
            return variants[index]
        else:
            # 随机分配
            return np.random.choice(variants)
    
    def _weighted_allocation(self, variants: List[TestVariant], user_id: str) -> TestVariant:
        """加权分配"""
        weights = [v.weight for v in variants]
        weights = np.array(weights) / sum(weights)
        
        if user_id:
            # 使用一致性哈希确定分配
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            rand_val = (hash_value % 1000) / 1000.0
        else:
            rand_val = np.random.random()
        
        # 根据权重分配
        cumulative = 0.0
        for variant, weight in zip(variants, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return variant
        
        return variants[-1]
    
    def _bandit_allocation(self, variants: List[TestVariant], user_id: str) -> TestVariant:
        """多臂老虎机分配（基于性能）"""
        # 计算每个变体的得分
        scores = []
        for variant in variants:
            stats = self.variant_stats[variant.id]
            if stats['impressions'] > 0:
                # 使用UCB1算法
                exploration = np.sqrt(2 * np.log(sum(v['impressions'] for v in self.variant_stats.values())) / stats['impressions'])
                score = stats['performance_score'] + exploration
            else:
                score = float('inf')  # 鼓励探索新变体
            scores.append(score)
        
        # 选择得分最高的变体
        best_index = np.argmax(scores)
        return variants[best_index]
    
    def _gradual_allocation(self, variants: List[TestVariant], user_id: str,
                           context: Dict[str, Any]) -> TestVariant:
        """渐进式分配"""
        # 根据时间或其他因素逐渐调整分配比例
        if context and 'day_of_test' in context:
            day = context['day_of_test']
            # 随着时间推移，逐渐向表现好的变体倾斜
            if day > 3:  # 3天后开始调整
                return self._bandit_allocation(variants, user_id)
        
        return self._equal_allocation(variants, user_id)
    
    def update_stats(self, variant_id: str, metric_name: str, value: float):
        """更新变体统计信息"""
        stats = self.variant_stats[variant_id]
        stats['impressions'] += 1
        
        if metric_name == 'conversion':
            stats['conversions'] += 1
            # 更新性能得分
            stats['performance_score'] = stats['conversions'] / stats['impressions']
        elif metric_name == 'revenue':
            stats['revenue'] += value
            # 更新性能得分
            stats['performance_score'] = stats['revenue'] / stats['impressions']


class StatisticalAnalyzer:
    """统计分析器"""
    
    @staticmethod
    def t_test(control_values: List[float], treatment_values: List[float]) -> Dict[str, float]:
        """执行t检验"""
        if len(control_values) < 2 or len(treatment_values) < 2:
            return {'p_value': 1.0, 'effect_size': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
        
        # 执行t检验
        t_stat, p_value = stats.ttest_ind(control_values, treatment_values, equal_var=False)
        
        # 计算效应量（Cohen's d）
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        control_std = np.std(control_values, ddof=1)
        treatment_std = np.std(treatment_values, ddof=1)
        
        pooled_std = np.sqrt(((len(control_values) - 1) * control_std**2 + 
                             (len(treatment_values) - 1) * treatment_std**2) / 
                            (len(control_values) + len(treatment_values) - 2))
        
        effect_size = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # 计算置信区间
        se = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
        ci_lower = (treatment_mean - control_mean) - stats.t.ppf(0.975, len(control_values) + len(treatment_values) - 2) * se
        ci_upper = (treatment_mean - control_mean) + stats.t.ppf(0.975, len(control_values) + len(treatment_values) - 2) * se
        
        return {
            'p_value': p_value,
            'effect_size': effect_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    @staticmethod
    def z_test(control_success: int, control_total: int,
               treatment_success: int, treatment_total: int) -> Dict[str, float]:
        """执行z检验（用于比例）"""
        if control_total == 0 or treatment_total == 0:
            return {'p_value': 1.0, 'effect_size': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
        
        # 计算比例
        p1 = control_success / control_total
        p2 = treatment_success / treatment_total
        p_pooled = (control_success + treatment_success) / (control_total + treatment_total)
        
        # 计算z统计量
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        z_stat = (p2 - p1) / se if se > 0 else 0
        
        # 计算p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # 计算效应量
        effect_size = p2 - p1
        
        # 计算置信区间
        se_diff = np.sqrt(p1 * (1 - p1) / control_total + p2 * (1 - p2) / treatment_total)
        ci_lower = effect_size - stats.norm.ppf(0.975) * se_diff
        ci_upper = effect_size + stats.norm.ppf(0.975) * se_diff
        
        return {
            'p_value': p_value,
            'effect_size': effect_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    @staticmethod
    def bootstrap_test(control_values: List[float], treatment_values: List[float],
                      n_iterations: int = 10000) -> Dict[str, float]:
        """自助法检验"""
        if len(control_values) == 0 or len(treatment_values) == 0:
            return {'p_value': 1.0, 'effect_size': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
        
        # 计算观测差异
        observed_diff = np.mean(treatment_values) - np.mean(control_values)
        
        # 自助采样
        bootstrap_diffs = []
        for _ in range(n_iterations):
            # 重采样
            control_sample = np.random.choice(control_values, size=len(control_values), replace=True)
            treatment_sample = np.random.choice(treatment_values, size=len(treatment_values), replace=True)
            
            # 计算差异
            diff = np.mean(treatment_sample) - np.mean(control_sample)
            bootstrap_diffs.append(diff)
        
        # 计算p值
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # 计算置信区间
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        return {
            'p_value': p_value,
            'effect_size': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }


class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self, db_path: str = "ab_tests.db"):
        """
        初始化A/B测试管理器
        
        Args:
            db_path: 数据库路径
        """
        self.db_path = db_path
        self.tests: Dict[str, ABTest] = {}
        self.traffic_allocator = TrafficAllocator()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # 数据存储
        self.metric_data: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # 初始化数据库
        self._init_db()
        
        # 加载现有测试
        self._load_tests()
        
        logger.info("ABTestManager initialized")
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 测试表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_tests (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                results TEXT,
                winner_variant_id TEXT,
                decision_reason TEXT
            )
        ''')
        
        # 指标数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metric_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                context TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_tests(self):
        """加载现有测试"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM ab_tests')
        rows = cursor.fetchall()
        
        for row in rows:
            test_data = {
                'id': row[0],
                'name': row[1],
                'config': json.loads(row[2]),
                'status': row[3],
                'created_at': row[4],
                'started_at': row[5],
                'ended_at': row[6],
                'results': json.loads(row[7]) if row[7] else {},
                'winner_variant_id': row[8],
                'decision_reason': row[9]
            }
            
            test = ABTest.from_dict(test_data)
            self.tests[test.id] = test
            
            # 加载指标数据
            self._load_metric_data(test.id)
        
        conn.close()
        logger.info(f"Loaded {len(self.tests)} tests from database")
    
    def _load_metric_data(self, test_id: str):
        """加载指标数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT variant_id, metric_name, value
            FROM metric_data 
            WHERE test_id = ?
            ORDER BY timestamp
        ''', (test_id,))
        
        for row in cursor.fetchall():
            variant_id, metric_name, value = row
            self.metric_data[test_id][variant_id][metric_name].append(value)
        
        conn.close()
    
    def create_test(self, config: ABTestConfig) -> ABTest:
        """创建A/B测试"""
        test_id = str(uuid.uuid4())
        test = ABTest(id=test_id, config=config)
        
        # 保存到数据库
        self._save_test(test)
        
        # 添加到内存
        self.tests[test_id] = test
        
        logger.info(f"Created AB test: {config.name} ({test_id})")
        return test
    
    def start_test(self, test_id: str):
        """启动测试"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        if test.status != TestStatus.DRAFT:
            raise ValueError(f"Test {test_id} is not in DRAFT status")
        
        test.status = TestStatus.RUNNING
        test.started_at = datetime.now()
        
        self._save_test(test)
        
        logger.info(f"Started AB test: {test.config.name}")
    
    def pause_test(self, test_id: str):
        """暂停测试"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        if test.status != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")
        
        test.status = TestStatus.PAUSED
        
        self._save_test(test)
        
        logger.info(f"Paused AB test: {test.config.name}")
    
    def stop_test(self, test_id: str, reason: str = ""):
        """停止测试"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        if test.status not in [TestStatus.RUNNING, TestStatus.PAUSED]:
            raise ValueError(f"Test {test_id} is not running or paused")
        
        test.status = TestStatus.STOPPED
        test.ended_at = datetime.now()
        test.decision_reason = reason
        
        # 分析结果
        self._analyze_test_results(test)
        
        self._save_test(test)
        
        logger.info(f"Stopped AB test: {test.config.name}")
    
    def _save_test(self, test: ABTest):
        """保存测试到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO ab_tests 
            (id, name, config, status, created_at, started_at, ended_at, 
             results, winner_variant_id, decision_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            test.id,
            test.config.name,
            json.dumps(test.config.to_dict()),
            test.status.value,
            test.created_at.isoformat(),
            test.started_at.isoformat() if test.started_at else None,
            test.ended_at.isoformat() if test.ended_at else None,
            json.dumps({k: [r.to_dict() for r in v] for k, v in test.results.items()}, default=str),
            test.winner_variant_id,
            test.decision_reason
        ))
        
        conn.commit()
        conn.close()
    
    def record_metric(self, test_id: str, variant_id: str, metric_name: str,
                     value: float, user_id: str = None, context: Dict[str, Any] = None):
        """记录指标"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        if test.status != TestStatus.RUNNING:
            logger.warning(f"Test {test_id} is not running, ignoring metric")
            return
        
        # 保存到内存
        self.metric_data[test_id][variant_id][metric_name].append(value)
        
        # 保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metric_data (test_id, variant_id, metric_name, value, 
                                   timestamp, user_id, context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            test_id,
            variant_id,
            metric_name,
            value,
            datetime.now().isoformat(),
            user_id,
            json.dumps(context) if context else None
        ))
        
        conn.commit()
        conn.close()
        
        # 更新流量分配器统计
        self.traffic_allocator.update_stats(variant_id, metric_name, value)
    
    def allocate_variant(self, test_id: str, user_id: str = None,
                        context: Dict[str, Any] = None) -> TestVariant:
        """为用户分配变体"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        if test.status != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")
        
        # 使用流量分配器
        variant = self.traffic_allocator.allocate_variant(
            test.config.variants, user_id, context
        )
        
        return variant
    
    def get_test_results(self, test_id: str) -> Dict[str, List[TestResult]]:
        """获取测试结果"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        
        # 实时分析结果
        if test.status == TestStatus.RUNNING:
            self._analyze_test_results(test)
        
        return test.results
    
    def _analyze_test_results(self, test: ABTest):
        """分析测试结果"""
        test.results = {}
        
        # 找到对照组
        control_variant = None
        for variant in test.config.variants:
            if variant.is_control:
                control_variant = variant
                break
        
        if not control_variant:
            logger.warning("No control variant found")
            return
        
        # 对每个指标进行分析
        for metric in test.config.metrics:
            results = []
            
            # 获取对照组数据
            control_data = self.metric_data[test.id][control_variant.id].get(metric.name, [])
            
            # 分析每个变体
            for variant in test.config.variants:
                if variant.id == control_variant.id:
                    continue
                
                # 获取变体数据
                variant_data = self.metric_data[test.id][variant.id].get(metric.name, [])
                
                if not control_data or not variant_data:
                    continue
                
                # 执行统计检验
                if metric.statistical_test == StatisticalTest.T_TEST:
                    if len(control_data) >= 2 and len(variant_data) >= 2:
                        stats_result = self.statistical_analyzer.t_test(control_data, variant_data)
                    else:
                        stats_result = {'p_value': 1.0, 'effect_size': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
                
                elif metric.statistical_test == StatisticalTest.BOOTSTRAP:
                    stats_result = self.statistical_analyzer.bootstrap_test(control_data, variant_data)
                
                else:
                    stats_result = {'p_value': 1.0, 'effect_size': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
                
                # 创建结果对象
                result = TestResult(
                    variant_id=variant.id,
                    metric_name=metric.name,
                    value=np.mean(variant_data) if variant_data else 0.0,
                    count=len(variant_data),
                    variance=np.var(variant_data) if variant_data else 0.0,
                    confidence_interval=(stats_result['ci_lower'], stats_result['ci_upper']),
                    p_value=stats_result['p_value'],
                    effect_size=stats_result['effect_size'],
                    is_significant=stats_result['p_value'] < test.config.significance_level
                )
                
                results.append(result)
            
            test.results[metric.name] = results
        
        # 确定获胜者
        self._determine_winner(test)
    
    def _determine_winner(self, test: ABTest):
        """确定获胜变体"""
        # 基于主要指标确定获胜者
        primary_metrics = [m for m in test.config.metrics if m.primary]
        
        if not primary_metrics:
            logger.warning("No primary metric specified")
            return
        
        primary_metric = primary_metrics[0]
        metric_results = test.results.get(primary_metric.name, [])
        
        if not metric_results:
            return
        
        # 根据改进方向确定获胜者
        if primary_metric.improvement_direction == "higher":
            # 值越高越好
            best_result = max(metric_results, key=lambda r: r.value)
        else:
            # 值越低越好
            best_result = min(metric_results, key=lambda r: r.value)
        
        # 检查是否显著
        if best_result.is_significant:
            test.winner_variant_id = best_result.variant_id
            best_result.is_winner = True
            
            # 找到变体信息
            for variant in test.config.variants:
                if variant.id == best_result.variant_id:
                    test.decision_reason = f"{variant.name} won with significant improvement in {primary_metric.name}"
                    break
    
    def calculate_sample_size(self, baseline_rate: float, min_detectable_effect: float,
                            significance_level: float = 0.05, power: float = 0.8) -> int:
        """计算所需样本量"""
        # 对于比例检验的样本量计算
        effect_size = min_detectable_effect / baseline_rate
        
        from statsmodels.stats.power import TTestIndPower
        power_analysis = TTestIndPower()
        
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            alpha=significance_level,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(sample_size))
    
    def list_tests(self, status: TestStatus = None) -> List[ABTest]:
        """列出测试"""
        tests = list(self.tests.values())
        
        if status:
            tests = [t for t in tests if t.status == status]
        
        return sorted(tests, key=lambda x: x.created_at, reverse=True)
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """获取测试"""
        return self.tests.get(test_id)
    
    def delete_test(self, test_id: str):
        """删除测试"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        if test.status == TestStatus.RUNNING:
            raise ValueError("Cannot delete running test")
        
        # 从数据库删除
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM ab_tests WHERE id = ?', (test_id,))
        cursor.execute('DELETE FROM metric_data WHERE test_id = ?', (test_id,))
        
        conn.commit()
        conn.close()
        
        # 从内存删除
        del self.tests[test_id]
        if test_id in self.metric_data:
            del self.metric_data[test_id]
        
        logger.info(f"Deleted AB test: {test.config.name}")
    
    def generate_report(self, test_id: str, output_path: str = None) -> str:
        """生成测试报告"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        
        if output_path is None:
            output_path = f"ab_test_report_{test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 获取最新结果
        if test.status in [TestStatus.RUNNING, TestStatus.COMPLETED, TestStatus.STOPPED]:
            self._analyze_test_results(test)
        
        # 生成报告
        report = f"# A/B Test Report: {test.config.name}\\n\\n"
        report += f"**Test ID:** {test.id}\\n\\n"
        report += f"**Status:** {test.status.value}\\n\\n"
        report += f"**Created:** {test.created_at.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
        
        if test.started_at:
            report += f"**Started:** {test.started_at.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
        
        if test.ended_at:
            duration = test.ended_at - test.started_at if test.started_at else timedelta(0)
            report += f"**Ended:** {test.ended_at.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
            report += f"**Duration:** {duration.days} days\\n\\n"
        
        report += f"**Description:** {test.config.description}\\n\\n"
        
        # 变体信息
        report += "## Test Variants\\n\\n"
        report += "| Variant | Model | Version | Weight | Control |\\n"
        report += "|---------|-------|---------|--------|---------|\\n"
        
        for variant in test.config.variants:
            control_mark = "✓" if variant.is_control else ""
            report += f"| {variant.name} | {variant.model_id} | {variant.version_id} | {variant.weight} | {control_mark} |\\n"
        
        report += "\\n"
        
        # 指标信息
        report += "## Metrics\\n\\n"
        report += "| Metric | Type | Primary | Improvement | Statistical Test |\\n"
        report += "|--------|------|---------|-------------|------------------|\\n"
        
        for metric in test.config.metrics:
            primary_mark = "✓" if metric.primary else ""
            report += f"| {metric.name} | {metric.type.value} | {primary_mark} | {metric.improvement_direction} | {metric.statistical_test.value} |\\n"
        
        report += "\\n"
        
        # 结果
        if test.results:
            report += "## Results\\n\\n"
            
            for metric_name, results in test.results.items():
                report += f"### {metric_name}\\n\\n"
                report += "| Variant | Value | Count | P-value | Effect Size | Significant | Winner |\\n"
                report += "|---------|-------|-------|---------|-------------|------------|--------|\\n"
                
                for result in results:
                    significant_mark = "✓" if result.is_significant else ""
                    winner_mark = "🏆" if result.is_winner else ""
                    report += f"| {result.variant_id} | {result.value:.4f} | {result.count} | {result.p_value:.4f} | {result.effect_size:.4f} | {significant_mark} | {winner_mark} |\\n"
                
                report += "\\n"
        
        # 结论
        if test.winner_variant_id:
            report += f"## Conclusion\\n\\n"
            report += f"**Winner:** {test.winner_variant_id}\\n\\n"
            report += f"**Reason:** {test.decision_reason}\\n\\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"AB test report saved to: {output_path}")
        return output_path


def create_ab_test_manager(db_path: str = "ab_tests.db") -> ABTestManager:
    """创建A/B测试管理器实例"""
    return ABTestManager(db_path)