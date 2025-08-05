"""实验类定义

定义实验的数据结构和基本操作。
"""

import json
import os
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import torch


@dataclass
class ExperimentConfig:
    """实验配置类"""
    
    # 模型配置
    model_name: str
    model_version: str = "latest"
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # 数据配置
    dataset_name: str = "bioast_dataset"
    dataset_version: str = "1.0"
    batch_size: int = 64
    num_workers: int = 4
    
    # 训练配置
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    
    # 验证配置
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    save_best_only: bool = True
    
    # 硬件配置
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    
    # 其他配置
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class ExperimentMetrics:
    """实验指标类"""
    
    # 训练指标
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    
    # 验证指标
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_precision: List[float] = field(default_factory=list)
    val_recall: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    
    # 学习率
    learning_rates: List[float] = field(default_factory=list)
    
    # 时间指标
    epoch_times: List[float] = field(default_factory=list)
    
    # 最佳指标
    best_val_accuracy: float = 0.0
    best_val_loss: float = float('inf')
    best_epoch: int = 0
    
    # 自定义指标
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_epoch_metrics(self, 
                         train_loss: float,
                         train_acc: float,
                         val_loss: float,
                         val_acc: float,
                         val_precision: float = 0.0,
                         val_recall: float = 0.0,
                         val_f1: float = 0.0,
                         lr: float = 0.0,
                         epoch_time: float = 0.0,
                         **custom_metrics):
        """添加一个epoch的指标"""
        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_accuracy.append(val_acc)
        self.val_precision.append(val_precision)
        self.val_recall.append(val_recall)
        self.val_f1.append(val_f1)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
        # 更新最佳指标
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.best_epoch = len(self.val_accuracy) - 1
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        # 添加自定义指标
        for metric_name, value in custom_metrics.items():
            if metric_name not in self.custom_metrics:
                self.custom_metrics[metric_name] = []
            self.custom_metrics[metric_name].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.val_accuracy:
            return {}
        
        return {
            "total_epochs": len(self.val_accuracy),
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "final_train_accuracy": self.train_accuracy[-1] if self.train_accuracy else 0.0,
            "final_val_accuracy": self.val_accuracy[-1] if self.val_accuracy else 0.0,
            "total_training_time": sum(self.epoch_times),
            "average_epoch_time": sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0.0,
            "convergence_epoch": self.best_epoch,
            "overfitting_score": self._calculate_overfitting_score()
        }
    
    def _calculate_overfitting_score(self) -> float:
        """计算过拟合分数"""
        if len(self.train_accuracy) < 5 or len(self.val_accuracy) < 5:
            return 0.0
        
        # 计算最后几个epoch的训练和验证准确率差异
        recent_epochs = min(5, len(self.train_accuracy))
        train_avg = sum(self.train_accuracy[-recent_epochs:]) / recent_epochs
        val_avg = sum(self.val_accuracy[-recent_epochs:]) / recent_epochs
        
        return max(0.0, train_avg - val_avg)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentMetrics':
        """从字典创建实例"""
        return cls(**data)


class Experiment:
    """实验类"""
    
    def __init__(self, 
                 name: str,
                 config: ExperimentConfig,
                 experiment_id: Optional[str] = None,
                 base_dir: str = "experiments"):
        
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.name = name
        self.config = config
        self.metrics = ExperimentMetrics()
        
        # 状态信息
        self.status = "created"  # created, running, completed, failed, stopped
        self.created_at = datetime.now().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        
        # 文件路径
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志和输出
        self.logs: List[str] = []
        self.artifacts: Dict[str, str] = {}
        self.notes: str = ""
        
        # 错误信息
        self.error_message: Optional[str] = None
        self.error_traceback: Optional[str] = None
        
        # 资源使用
        self.resource_usage: Dict[str, Any] = {}
        
        # 保存初始状态
        self.save()
    
    def _generate_experiment_id(self) -> str:
        """生成实验ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"exp_{timestamp}_{unique_id}"
    
    def start(self):
        """开始实验"""
        self.status = "running"
        self.started_at = datetime.now().isoformat()
        self.log(f"实验开始: {self.name}")
        self.save()
    
    def complete(self, success: bool = True):
        """完成实验"""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.now().isoformat()
        
        if success:
            self.log(f"实验成功完成: {self.name}")
        else:
            self.log(f"实验失败: {self.name}")
        
        self.save()
    
    def stop(self):
        """停止实验"""
        self.status = "stopped"
        self.completed_at = datetime.now().isoformat()
        self.log(f"实验被停止: {self.name}")
        self.save()
    
    def fail(self, error_message: str, traceback: Optional[str] = None):
        """标记实验失败"""
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()
        self.error_message = error_message
        self.error_traceback = traceback
        self.log(f"实验失败: {error_message}")
        self.save()
    
    def log(self, message: str, level: str = "INFO"):
        """添加日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        
        # 写入日志文件
        log_file = self.experiment_dir / "experiment.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
    
    def add_artifact(self, name: str, file_path: str):
        """添加实验产物"""
        self.artifacts[name] = file_path
        self.log(f"添加产物: {name} -> {file_path}")
        self.save()
    
    def update_resource_usage(self, **usage):
        """更新资源使用情况"""
        self.resource_usage.update(usage)
        self.save()
    
    def save(self):
        """保存实验状态"""
        experiment_file = self.experiment_dir / "experiment.json"
        
        data = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "resource_usage": self.resource_usage,
            "logs_count": len(self.logs)
        }
        
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, experiment_id: str, base_dir: str = "experiments") -> 'Experiment':
        """加载实验"""
        experiment_dir = Path(base_dir) / experiment_id
        experiment_file = experiment_dir / "experiment.json"
        
        if not experiment_file.exists():
            raise FileNotFoundError(f"实验文件不存在: {experiment_file}")
        
        with open(experiment_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建实验实例
        config = ExperimentConfig.from_dict(data["config"])
        experiment = cls(
            name=data["name"],
            config=config,
            experiment_id=data["experiment_id"],
            base_dir=base_dir
        )
        
        # 恢复状态
        experiment.metrics = ExperimentMetrics.from_dict(data["metrics"])
        experiment.status = data["status"]
        experiment.created_at = data["created_at"]
        experiment.started_at = data.get("started_at")
        experiment.completed_at = data.get("completed_at")
        experiment.artifacts = data.get("artifacts", {})
        experiment.notes = data.get("notes", "")
        experiment.error_message = data.get("error_message")
        experiment.error_traceback = data.get("error_traceback")
        experiment.resource_usage = data.get("resource_usage", {})
        
        # 加载日志
        log_file = experiment_dir / "experiment.log"
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                experiment.logs = f.read().strip().split('\n')
        
        return experiment
    
    def get_summary(self) -> Dict[str, Any]:
        """获取实验摘要"""
        duration = None
        if self.started_at and self.completed_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            duration = (end - start).total_seconds()
        
        summary = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": duration,
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "dataset_name": self.config.dataset_name,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "metrics_summary": self.metrics.get_summary(),
            "artifacts_count": len(self.artifacts),
            "logs_count": len(self.logs),
            "has_error": self.error_message is not None
        }
        
        return summary
    
    def to_markdown(self) -> str:
        """生成Markdown格式的实验报告"""
        md = f"# 实验报告: {self.name}\n\n"
        
        # 基本信息
        md += "## 基本信息\n\n"
        md += f"- **实验ID**: `{self.experiment_id}`\n"
        md += f"- **状态**: {self.status}\n"
        md += f"- **创建时间**: {self.created_at}\n"
        if self.started_at:
            md += f"- **开始时间**: {self.started_at}\n"
        if self.completed_at:
            md += f"- **完成时间**: {self.completed_at}\n"
        md += "\n"
        
        # 配置信息
        md += "## 实验配置\n\n"
        md += f"- **模型**: {self.config.model_name} (v{self.config.model_version})\n"
        md += f"- **数据集**: {self.config.dataset_name} (v{self.config.dataset_version})\n"
        md += f"- **训练轮数**: {self.config.epochs}\n"
        md += f"- **批次大小**: {self.config.batch_size}\n"
        md += f"- **学习率**: {self.config.learning_rate}\n"
        md += f"- **优化器**: {self.config.optimizer}\n"
        md += f"- **调度器**: {self.config.scheduler}\n"
        md += "\n"
        
        # 结果摘要
        metrics_summary = self.metrics.get_summary()
        if metrics_summary:
            md += "## 训练结果\n\n"
            md += f"- **最佳验证准确率**: {metrics_summary['best_val_accuracy']:.4f}\n"
            md += f"- **最佳验证损失**: {metrics_summary['best_val_loss']:.4f}\n"
            md += f"- **最佳轮次**: {metrics_summary['best_epoch']}\n"
            md += f"- **总训练时间**: {metrics_summary['total_training_time']:.2f}秒\n"
            md += f"- **平均轮次时间**: {metrics_summary['average_epoch_time']:.2f}秒\n"
            md += f"- **过拟合分数**: {metrics_summary['overfitting_score']:.4f}\n"
            md += "\n"
        
        # 产物列表
        if self.artifacts:
            md += "## 实验产物\n\n"
            for name, path in self.artifacts.items():
                md += f"- **{name}**: `{path}`\n"
            md += "\n"
        
        # 错误信息
        if self.error_message:
            md += "## 错误信息\n\n"
            md += f"```\n{self.error_message}\n```\n\n"
        
        # 备注
        if self.notes:
            md += "## 备注\n\n"
            md += f"{self.notes}\n\n"
        
        return md
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Experiment(id={self.experiment_id}, name={self.name}, status={self.status})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()