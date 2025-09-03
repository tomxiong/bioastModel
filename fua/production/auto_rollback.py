"""
自动模型回滚机制

提供自动化的模型版本管理和回滚功能，当检测到性能降级时，
自动切换到稳定的模型版本，确保服务的连续性和可靠性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import json
import time
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import sqlite3
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
import os

from .model_monitor import ModelMonitor, Alert, AlertSeverity
from .performance_degradation import DegradationAnalyzer, DegradationEvent, SeverityLevel

logger = logging.getLogger(__name__)


class RollbackTrigger(Enum):
    """回滚触发条件"""
    DEGRADATION_DETECTED = "degradation_detected"  # 检测到性能降级
    ERROR_RATE_SPIKE = "error_rate_spike"  # 错误率激增
    LATENCY_THRESHOLD = "latency_threshold"  # 延迟超过阈值
    MANUAL_TRIGGER = "manual_trigger"  # 手动触发
    HEALTH_CHECK_FAILED = "health_check_failed"  # 健康检查失败


class RollbackStatus(Enum):
    """回滚状态"""
    PENDING = "pending"  # 等待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消


class ModelHealthStatus(Enum):
    """模型健康状态"""
    HEALTHY = "healthy"  # 健康
    DEGRADED = "degraded"  # 降级
    UNHEALTHY = "unhealthy"  # 不健康
    UNKNOWN = "unknown"  # 未知


@dataclass
class ModelVersion:
    """模型版本信息"""
    id: str
    model_id: str
    version: str
    path: str
    created_at: datetime = field(default_factory=datetime.now)
    is_stable: bool = False
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0
    rollback_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """从字典创建"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class RollbackPlan:
    """回滚计划"""
    id: str
    model_id: str
    from_version: str
    to_version: str
    trigger: RollbackTrigger
    reason: str
    status: RollbackStatus = RollbackStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    rollback_strategy: str = "immediate"  # immediate, gradual, canary
    canary_percentage: float = 0.1  # 金丝雀发布比例
    validation_required: bool = True
    rollback_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['trigger'] = self.trigger.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.executed_at:
            data['executed_at'] = self.executed_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackPlan':
        """从字典创建"""
        data['trigger'] = RollbackTrigger(data['trigger'])
        data['status'] = RollbackStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('executed_at'):
            data['executed_at'] = datetime.fromisoformat(data['executed_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class RollbackConfig:
    """回滚配置"""
    auto_rollback_enabled: bool = True
    degradation_threshold: float = 0.1  # 10%降级触发回滚
    error_rate_threshold: float = 0.3  # 30%错误率触发回滚
    latency_threshold: float = 2.0  # 延迟超过基线2倍触发回滚
    max_rollback_versions: int = 5  # 最多保留5个版本用于回滚
    health_check_interval: int = 60  # 健康检查间隔（秒）
    rollback_timeout: int = 300  # 回滚超时时间（秒）
    stabilization_period: int = 3600  # 稳定化周期（秒）
    canary_rollback_enabled: bool = True  # 启用金丝雀回滚
    notification_channels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RollbackConfig':
        """从字典创建"""
        return cls(**data)


class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, storage_path: str = "model_versions"):
        """
        初始化模型版本管理器
        
        Args:
            storage_path: 模型存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self.current_versions: Dict[str, str] = {}  # model_id -> version_id
        
        # 初始化数据库
        self._init_db()
        self._load_versions()
        
        logger.info(f"ModelVersionManager initialized with storage at {storage_path}")
    
    def _init_db(self):
        """初始化数据库"""
        db_path = self.storage_path / "versions.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 模型版本表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_stable INTEGER NOT NULL,
                performance_metrics TEXT,
                health_score REAL NOT NULL,
                rollback_count INTEGER NOT NULL,
                metadata TEXT
            )
        ''')
        
        # 当前版本表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS current_versions (
                model_id TEXT PRIMARY KEY,
                version_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_versions(self):
        """加载模型版本"""
        db_path = self.storage_path / "versions.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 加载所有版本
        cursor.execute('SELECT * FROM model_versions ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        for row in rows:
            version = ModelVersion(
                id=row[0],
                model_id=row[1],
                version=row[2],
                path=row[3],
                created_at=datetime.fromisoformat(row[4]),
                is_stable=bool(row[5]),
                performance_metrics=json.loads(row[6]) if row[6] else {},
                health_score=row[7],
                rollback_count=row[8],
                metadata=json.loads(row[9]) if row[9] else {}
            )
            self.versions[version.id] = version
        
        # 加载当前版本
        cursor.execute('SELECT * FROM current_versions')
        rows = cursor.fetchall()
        for row in rows:
            self.current_versions[row[0]] = row[1]
        
        conn.close()
        logger.info(f"Loaded {len(self.versions)} model versions")
    
    def save_version(self, model_id: str, version: str, 
                    model: nn.Module, metadata: Dict[str, Any] = None) -> ModelVersion:
        """
        保存模型版本
        
        Args:
            model_id: 模型ID
            version: 版本号
            model: 模型实例
            metadata: 元数据
            
        Returns:
            保存的版本信息
        """
        version_id = f"{model_id}_{version}_{int(time.time())}"
        version_path = self.storage_path / version_id
        
        # 创建版本目录
        version_path.mkdir(exist_ok=True)
        
        # 保存模型
        model_path = version_path / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # 保存模型配置
        if metadata:
            config_path = version_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # 创建版本记录
        model_version = ModelVersion(
            id=version_id,
            model_id=model_id,
            version=version,
            path=str(version_path),
            metadata=metadata or {}
        )
        
        # 保存到数据库
        db_path = self.storage_path / "versions.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_versions 
            (id, model_id, version, path, created_at, is_stable, 
             performance_metrics, health_score, rollback_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_version.id,
            model_version.model_id,
            model_version.version,
            model_version.path,
            model_version.created_at.isoformat(),
            int(model_version.is_stable),
            json.dumps(model_version.performance_metrics),
            model_version.health_score,
            model_version.rollback_count,
            json.dumps(model_version.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        # 更新内存
        self.versions[version_id] = model_version
        
        # 如果是第一个版本，设为当前版本
        if model_id not in self.current_versions:
            self.set_current_version(model_id, version_id)
        
        logger.info(f"Saved model version: {model_id}:{version}")
        return model_version
    
    def load_version(self, version_id: str, model_class: type) -> nn.Module:
        """
        加载模型版本
        
        Args:
            version_id: 版本ID
            model_class: 模型类
            
        Returns:
            加载的模型实例
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        model_path = Path(version.path) / "model.pth"
        
        # 创建模型实例
        model = model_class()
        
        # 加载权重
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model
    
    def set_current_version(self, model_id: str, version_id: str):
        """设置当前版本"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        self.current_versions[model_id] = version_id
        
        # 更新数据库
        db_path = self.storage_path / "versions.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO current_versions (model_id, version_id, updated_at)
            VALUES (?, ?, ?)
        ''', (model_id, version_id, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Set current version for {model_id}: {version_id}")
    
    def get_current_version(self, model_id: str) -> Optional[ModelVersion]:
        """获取当前版本"""
        version_id = self.current_versions.get(model_id)
        if version_id:
            return self.versions.get(version_id)
        return None
    
    def get_stable_versions(self, model_id: str) -> List[ModelVersion]:
        """获取稳定版本列表"""
        stable_versions = []
        for version in self.versions.values():
            if version.model_id == model_id and version.is_stable:
                stable_versions.append(version)
        
        # 按健康分数排序
        stable_versions.sort(key=lambda v: v.health_score, reverse=True)
        return stable_versions
    
    def mark_as_stable(self, version_id: str, is_stable: bool = True):
        """标记版本为稳定/不稳定"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        self.versions[version_id].is_stable = is_stable
        
        # 更新数据库
        db_path = self.storage_path / "versions.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE model_versions SET is_stable = ? WHERE id = ?
        ''', (int(is_stable), version_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Marked version {version_id} as {'stable' if is_stable else 'unstable'}")
    
    def update_performance_metrics(self, version_id: str, metrics: Dict[str, float]):
        """更新性能指标"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        version.performance_metrics.update(metrics)
        
        # 计算健康分数
        health_score = self._calculate_health_score(version)
        version.health_score = health_score
        
        # 更新数据库
        db_path = self.storage_path / "versions.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE model_versions 
            SET performance_metrics = ?, health_score = ?
            WHERE id = ?
        ''', (
            json.dumps(version.performance_metrics),
            health_score,
            version_id
        ))
        
        conn.commit()
        conn.close()
    
    def _calculate_health_score(self, version: ModelVersion) -> float:
        """计算健康分数"""
        if not version.performance_metrics:
            return 0.5
        
        # 简单的健康分数计算
        accuracy = version.performance_metrics.get('accuracy', 0.5)
        error_rate = version.performance_metrics.get('error_rate', 0.5)
        latency = version.performance_metrics.get('latency', 1000)
        
        # 归一化分数
        accuracy_score = accuracy
        error_score = 1 - error_rate
        latency_score = max(0, 1 - latency / 1000)  # 假设1000ms为基准
        
        # 加权平均
        health_score = 0.5 * accuracy_score + 0.3 * error_score + 0.2 * latency_score
        return health_score


class RollbackExecutor:
    """回滚执行器"""
    
    def __init__(self, version_manager: ModelVersionManager):
        """
        初始化回滚执行器
        
        Args:
            version_manager: 版本管理器
        """
        self.version_manager = version_manager
        self.active_rollbacks: Dict[str, RollbackPlan] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("RollbackExecutor initialized")
    
    def execute_rollback(self, plan: RollbackPlan) -> bool:
        """
        执行回滚计划
        
        Args:
            plan: 回滚计划
            
        Returns:
            是否成功
        """
        plan.status = RollbackStatus.IN_PROGRESS
        plan.executed_at = datetime.now()
        
        logger.info(f"Executing rollback plan: {plan.id}")
        
        try:
            # 根据策略执行回滚
            if plan.rollback_strategy == "immediate":
                success = self._execute_immediate_rollback(plan)
            elif plan.rollback_strategy == "gradual":
                success = self._execute_gradual_rollback(plan)
            elif plan.rollback_strategy == "canary":
                success = self._execute_canary_rollback(plan)
            else:
                raise ValueError(f"Unknown rollback strategy: {plan.rollback_strategy}")
            
            if success:
                plan.status = RollbackStatus.COMPLETED
                plan.completed_at = datetime.now()
                
                # 更新版本统计
                to_version = self.version_manager.versions.get(plan.to_version)
                if to_version:
                    to_version.rollback_count += 1
                
                logger.info(f"Rollback completed successfully: {plan.id}")
            else:
                plan.status = RollbackStatus.FAILED
                logger.error(f"Rollback failed: {plan.id}")
            
            return success
            
        except Exception as e:
            plan.status = RollbackStatus.FAILED
            logger.error(f"Rollback execution error: {e}")
            return False
    
    def _execute_immediate_rollback(self, plan: RollbackPlan) -> bool:
        """立即回滚"""
        # 直接切换到目标版本
        self.version_manager.set_current_version(plan.model_id, plan.to_version)
        
        # 记录回滚指标
        plan.rollback_metrics = {
            'strategy': 'immediate',
            'switch_time': time.time(),
            'success': True
        }
        
        return True
    
    def _execute_gradual_rollback(self, plan: RollbackPlan) -> bool:
        """渐进式回滚"""
        # 分阶段逐步切换流量
        steps = [0.5, 1.0]  # 流量比例（测试用）
        
        for step in steps:
            logger.info(f"Gradual rollback step: {int(step * 100)}% traffic")
            
            # 模拟流量切换
            time.sleep(1)  # 等待稳定（测试用）
            
            # 这里应该有健康检查
            if not self._health_check(plan.model_id, plan.to_version):
                logger.warning("Health check failed during gradual rollback")
                return False
        
        # 完全切换
        self.version_manager.set_current_version(plan.model_id, plan.to_version)
        
        plan.rollback_metrics = {
            'strategy': 'gradual',
            'steps': len(steps),
            'success': True
        }
        
        return True
    
    def _execute_canary_rollback(self, plan: RollbackPlan) -> bool:
        """金丝雀回滚"""
        canary_percentage = plan.canary_percentage or 0.1
        
        # 首先部署到小比例流量
        logger.info(f"Canary rollback: {int(canary_percentage * 100)}% traffic")
        
        # 监控金丝雀版本
        monitoring_time = 5  # 监控5秒（测试用）
        start_time = time.time()
        
        while time.time() - start_time < monitoring_time:
            if not self._health_check(plan.model_id, plan.to_version):
                logger.warning("Canary health check failed")
                return False
            time.sleep(10)
        
        # 金丝雀成功，完全切换
        self.version_manager.set_current_version(plan.model_id, plan.to_version)
        
        plan.rollback_metrics = {
            'strategy': 'canary',
            'canary_percentage': canary_percentage,
            'monitoring_time': monitoring_time,
            'success': True
        }
        
        return True
    
    def _health_check(self, model_id: str, version_id: str) -> bool:
        """健康检查"""
        version = self.version_manager.versions.get(version_id)
        if not version:
            return False
        
        # 简单的健康检查：检查健康分数
        return version.health_score > 0.7


class AutoRollbackManager:
    """自动回滚管理器"""
    
    def __init__(self, version_manager: ModelVersionManager,
                 degradation_analyzer: DegradationAnalyzer = None,
                 model_monitor: ModelMonitor = None,
                 config: RollbackConfig = None):
        """
        初始化自动回滚管理器
        
        Args:
            version_manager: 版本管理器
            degradation_analyzer: 降级分析器
            model_monitor: 模型监控器
            config: 回滚配置
        """
        self.version_manager = version_manager
        self.degradation_analyzer = degradation_analyzer
        self.model_monitor = model_monitor
        self.config = config or RollbackConfig()
        
        self.rollback_executor = RollbackExecutor(version_manager)
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        
        # 初始化数据库
        self._init_db()
        self._load_plans()
        
        # 回滚历史
        self.rollback_history: List[RollbackPlan] = []
        
        # 事件处理器
        self.event_handlers: Dict[str, Callable] = {}
        
        logger.info("AutoRollbackManager initialized")
    
    def _init_db(self):
        """初始化数据库"""
        db_path = self.version_manager.storage_path / "rollback.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 回滚计划表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rollback_plans (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                from_version TEXT NOT NULL,
                to_version TEXT NOT NULL,
                trigger TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                executed_at TEXT,
                completed_at TEXT,
                rollback_strategy TEXT NOT NULL,
                canary_percentage REAL NOT NULL,
                validation_required INTEGER NOT NULL,
                rollback_metrics TEXT
            )
        ''')
        
        # 回滚配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rollback_configs (
                model_id TEXT PRIMARY KEY,
                config TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_plans(self):
        """加载回滚计划"""
        db_path = self.version_manager.storage_path / "rollback.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM rollback_plans ORDER BY created_at DESC')
        rows = cursor.fetchall()
        
        for row in rows:
            plan = RollbackPlan(
                id=row[0],
                model_id=row[1],
                from_version=row[2],
                to_version=row[3],
                trigger=RollbackTrigger(row[4]),
                reason=row[5],
                status=RollbackStatus(row[6]),
                created_at=datetime.fromisoformat(row[7]),
                executed_at=datetime.fromisoformat(row[8]) if row[8] else None,
                completed_at=datetime.fromisoformat(row[9]) if row[9] else None,
                rollback_strategy=row[10],
                canary_percentage=row[11],
                validation_required=bool(row[12]),
                rollback_metrics=json.loads(row[13]) if row[13] else {}
            )
            self.rollback_plans[plan.id] = plan
            
            if plan.status == RollbackStatus.COMPLETED:
                self.rollback_history.append(plan)
        
        conn.close()
        logger.info(f"Loaded {len(self.rollback_plans)} rollback plans")
    
    def handle_degradation_alert(self, event: DegradationEvent):
        """处理性能降级告警"""
        if not self.config.auto_rollback_enabled:
            logger.info("Auto rollback is disabled")
            return
        
        # 检查是否需要回滚
        if self._should_rollback(event):
            # 创建回滚计划
            # Get the trigger enum value programmatically to avoid typing issues
            trigger_value = getattr(RollbackTrigger, 'DEGRADATION_DETECTED')
            plan = self._create_rollback_plan(
                model_id=event.model_id,
                version_id=event.version_id,
                trigger=trigger_value,
                reason=f"Performance degradation detected: {event.description}"
            )
            
            if plan:
                # 执行回滚
                self._execute_rollback_plan(plan)
    
    def handle_monitoring_alert(self, alert: Alert):
        """处理监控告警"""
        if not self.config.auto_rollback_enabled:
            return
        
        # 根据告警类型处理
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
            # 获取当前模型版本
            current_version = self.version_manager.get_current_version(alert.model_id)
            if not current_version:
                return
            
            # 创建回滚计划
            plan = self._create_rollback_plan(
                model_id=alert.model_id,
                version_id=current_version.id,
                trigger=RollbackTrigger.ERROR_RATE_SPIKE,
                reason=f"Critical alert: {alert.message}"
            )
            
            if plan:
                self._execute_rollback_plan(plan)
    
    def _should_rollback(self, event: DegradationEvent) -> bool:
        """判断是否应该回滚"""
        # 检查严重程度
        if event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            return True
        
        # 检查降级分数
        if event.degradation_score >= self.config.degradation_threshold:
            return True
        
        # 检查指标类型
        if event.degradation_type.value in ["error_rate_spike", "latency_increase"]:
            return True
        
        return False
    
    def _create_rollback_plan(self, model_id: str, version_id: str,
                            trigger: RollbackTrigger, reason: str) -> Optional[RollbackPlan]:
        """创建回滚计划"""
        # 获取稳定版本
        stable_versions = self.version_manager.get_stable_versions(model_id)
        
        # 过滤掉当前版本
        stable_versions = [v for v in stable_versions if v.id != version_id]
        
        if not stable_versions:
            logger.warning(f"No stable versions available for rollback: {model_id}")
            return None
        
        # 选择最佳版本（健康分数最高）
        best_version = stable_versions[0]
        
        # 创建回滚计划
        plan = RollbackPlan(
            id=f"rollback_{model_id}_{int(time.time())}",
            model_id=model_id,
            from_version=version_id,
            to_version=best_version.id,
            trigger=trigger,
            reason=reason,
            rollback_strategy="canary" if self.config.canary_rollback_enabled else "immediate",
            canary_percentage=0.1,
            validation_required=True
        )
        
        # 保存计划
        self.rollback_plans[plan.id] = plan
        self._save_plan(plan)
        
        logger.info(f"Created rollback plan: {plan.id}")
        return plan
    
    def _execute_rollback_plan(self, plan: RollbackPlan):
        """执行回滚计划"""
        # 通知事件处理器
        self._notify_handlers("rollback_started", plan)
        
        # 执行回滚
        success = self.rollback_executor.execute_rollback(plan)
        
        if success:
            # 更新失败版本的状态
            failed_version = self.version_manager.versions.get(plan.from_version)
            if failed_version:
                self.version_manager.mark_as_stable(failed_version.id, False)
            
            # 通知成功
            self._notify_handlers("rollback_completed", plan)
        else:
            # 通知失败
            self._notify_handlers("rollback_failed", plan)
        
        # 保存计划状态
        self._save_plan(plan)
    
    def _save_plan(self, plan: RollbackPlan):
        """保存回滚计划"""
        db_path = self.version_manager.storage_path / "rollback.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO rollback_plans 
            (id, model_id, from_version, to_version, trigger, reason, status,
             created_at, executed_at, completed_at, rollback_strategy,
             canary_percentage, validation_required, rollback_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            plan.id,
            plan.model_id,
            plan.from_version,
            plan.to_version,
            plan.trigger.value,
            plan.reason,
            plan.status.value,
            plan.created_at.isoformat(),
            plan.executed_at.isoformat() if plan.executed_at else None,
            plan.completed_at.isoformat() if plan.completed_at else None,
            plan.rollback_strategy,
            plan.canary_percentage,
            int(plan.validation_required),
            json.dumps(plan.rollback_metrics)
        ))
        
        conn.commit()
        conn.close()
    
    def manual_rollback(self, model_id: str, target_version: str = None,
                      strategy: str = "immediate") -> bool:
        """手动触发回滚"""
        # 获取当前版本
        current_version = self.version_manager.get_current_version(model_id)
        if not current_version:
            logger.error(f"No current version found for model: {model_id}")
            return False
        
        # 如果没有指定目标版本，选择最佳稳定版本
        if not target_version:
            stable_versions = self.version_manager.get_stable_versions(model_id)
            stable_versions = [v for v in stable_versions if v.id != current_version.id]
            
            if not stable_versions:
                logger.error("No stable versions available for rollback")
                return False
            
            target_version = stable_versions[0].id
        
        # 创建回滚计划
        plan = RollbackPlan(
            id=f"manual_rollback_{model_id}_{int(time.time())}",
            model_id=model_id,
            from_version=current_version.id,
            to_version=target_version,
            trigger=RollbackTrigger.MANUAL_TRIGGER,
            reason="Manual rollback triggered",
            rollback_strategy=strategy
        )
        
        # 执行回滚
        success = self.rollback_executor.execute_rollback(plan)
        
        if success:
            self.rollback_plans[plan.id] = plan
            self._save_plan(plan)
            logger.info(f"Manual rollback completed: {model_id} -> {target_version}")
        
        return success
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """添加事件处理器"""
        self.event_handlers[event_type] = handler
    
    def _notify_handlers(self, event_type: str, plan: RollbackPlan):
        """通知事件处理器"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type](plan)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
    
    def get_rollback_history(self, model_id: str = None) -> List[RollbackPlan]:
        """获取回滚历史"""
        history = self.rollback_history
        
        if model_id:
            history = [p for p in history if p.model_id == model_id]
        
        return sorted(history, key=lambda x: x.created_at, reverse=True)
    
    def generate_rollback_report(self, output_path: str = None) -> str:
        """生成回滚报告"""
        if output_path is None:
            output_path = f"rollback_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 生成报告
        report = f"# Model Rollback Report\\\\n\\\\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\\\n\\\\n"
        
        # 统计信息
        total_rollbacks = len(self.rollback_history)
        successful_rollbacks = len([p for p in self.rollback_history if p.status == RollbackStatus.COMPLETED])
        
        report += "## Summary\\\\n\\\\n"
        report += f"- Total rollbacks: {total_rollbacks}\\\\n"
        report += f"- Successful rollbacks: {successful_rollbacks}\\\\n"
        report += f"- Success rate: {successful_rollbacks/total_rollbacks*100:.1f}%\\\\n\\\\n" if total_rollbacks > 0 else "Success rate: N/A\\\\n\\\\n"
        
        # 按模型统计
        model_stats = {}
        for plan in self.rollback_history:
            if plan.model_id not in model_stats:
                model_stats[plan.model_id] = {'total': 0, 'successful': 0}
            model_stats[plan.model_id]['total'] += 1
            if plan.status == RollbackStatus.COMPLETED:
                model_stats[plan.model_id]['successful'] += 1
        
        if model_stats:
            report += "### Model Statistics\\\\n\\\\n"
            report += "| Model | Total Rollbacks | Successful | Success Rate |\\\\n"
            report += "|-------|----------------|------------|--------------|\\\\n"
            
            for model_id, stats in model_stats.items():
                success_rate = stats['successful'] / stats['total'] * 100
                report += f"| {model_id} | {stats['total']} | {stats['successful']} | {success_rate:.1f}% |\\\\n"
            
            report += "\\\\n"
        
        # 最近的回滚
        if self.rollback_history:
            report += "## Recent Rollbacks\\\\n\\\\n"
            report += "| ID | Model | From | To | Trigger | Status | Time |\\\\n"
            report += "|----|-------|------|----|---------|--------|------|\\\\n"
            
            for plan in sorted(self.rollback_history, key=lambda x: x.created_at, reverse=True)[:10]:
                report += f"| {plan.id[:12]}... | {plan.model_id} | {plan.from_version[-8:]} | {plan.to_version[-8:]} | "
                report += f"{plan.trigger.value} | {plan.status.value} | {plan.created_at.strftime('%m-%d %H:%M')} |\\\\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Rollback report saved to: {output_path}")
        return output_path


def create_auto_rollback_manager(storage_path: str = "model_versions",
                               degradation_analyzer: DegradationAnalyzer = None,
                               model_monitor: ModelMonitor = None,
                               config: RollbackConfig = None) -> AutoRollbackManager:
    """创建自动回滚管理器实例"""
    version_manager = ModelVersionManager(storage_path)
    return AutoRollbackManager(version_manager, degradation_analyzer, model_monitor, config)