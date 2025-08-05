"""实验跟踪器

负责实验的创建、监控、管理和分析。
"""

import json
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import psutil
import torch

from .experiment import Experiment, ExperimentConfig
from .database import ExperimentDatabase


class ExperimentTracker:
    """实验跟踪器"""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.database = ExperimentDatabase()
        self.active_experiments: Dict[str, Experiment] = {}
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_monitoring: Dict[str, threading.Event] = {}
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            "on_experiment_start": [],
            "on_experiment_complete": [],
            "on_experiment_fail": [],
            "on_epoch_end": [],
            "on_metric_update": []
        }
    
    def create_experiment(self, 
                         name: str,
                         config: ExperimentConfig,
                         notes: str = "") -> Experiment:
        """创建新实验"""
        experiment = Experiment(name, config, base_dir=str(self.base_dir))
        experiment.notes = notes
        
        # 保存到数据库
        self.database.save_experiment(experiment)
        
        # 添加到活跃实验列表
        self.active_experiments[experiment.experiment_id] = experiment
        
        experiment.log(f"实验已创建: {name}")
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """开始实验"""
        if experiment_id not in self.active_experiments:
            experiment = self.load_experiment(experiment_id)
            if experiment:
                self.active_experiments[experiment_id] = experiment
            else:
                return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.start()
        
        # 开始监控
        self._start_monitoring(experiment_id)
        
        # 触发回调
        self._trigger_callbacks("on_experiment_start", experiment)
        
        # 更新数据库
        self.database.update_experiment(experiment)
        
        return True
    
    def complete_experiment(self, experiment_id: str, success: bool = True):
        """完成实验"""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.complete(success)
        
        # 停止监控
        self._stop_monitoring(experiment_id)
        
        # 触发回调
        if success:
            self._trigger_callbacks("on_experiment_complete", experiment)
        else:
            self._trigger_callbacks("on_experiment_fail", experiment)
        
        # 更新数据库
        self.database.update_experiment(experiment)
        
        # 从活跃列表移除
        del self.active_experiments[experiment_id]
        
        return True
    
    def fail_experiment(self, 
                       experiment_id: str, 
                       error_message: str, 
                       traceback: Optional[str] = None):
        """标记实验失败"""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.fail(error_message, traceback)
        
        # 停止监控
        self._stop_monitoring(experiment_id)
        
        # 触发回调
        self._trigger_callbacks("on_experiment_fail", experiment)
        
        # 更新数据库
        self.database.update_experiment(experiment)
        
        # 从活跃列表移除
        del self.active_experiments[experiment_id]
        
        return True
    
    def stop_experiment(self, experiment_id: str):
        """停止实验"""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.stop()
        
        # 停止监控
        self._stop_monitoring(experiment_id)
        
        # 更新数据库
        self.database.update_experiment(experiment)
        
        # 从活跃列表移除
        del self.active_experiments[experiment_id]
        
        return True
    
    def log_epoch(self, 
                 experiment_id: str,
                 epoch: int,
                 train_loss: float,
                 train_acc: float,
                 val_loss: float,
                 val_acc: float,
                 **kwargs):
        """记录epoch结果"""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        
        # 添加指标
        experiment.metrics.add_epoch_metrics(
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            **kwargs
        )
        
        # 记录日志
        experiment.log(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        
        # 保存实验状态
        experiment.save()
        
        # 触发回调
        self._trigger_callbacks("on_epoch_end", experiment, epoch)
        self._trigger_callbacks("on_metric_update", experiment)
        
        return True
    
    def add_artifact(self, experiment_id: str, name: str, file_path: str):
        """添加实验产物"""
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        experiment.add_artifact(name, file_path)
        
        return True
    
    def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """加载实验"""
        try:
            return Experiment.load(experiment_id, str(self.base_dir))
        except Exception as e:
            print(f"加载实验失败: {e}")
            return None
    
    def list_experiments(self, 
                        status: Optional[str] = None,
                        model_name: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """列出实验"""
        return self.database.list_experiments(status, model_name, limit)
    
    def search_experiments(self, query: str) -> List[Dict[str, Any]]:
        """搜索实验"""
        return self.database.search_experiments(query)
    
    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """获取实验摘要"""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            experiment = self.load_experiment(experiment_id)
        
        if experiment:
            return experiment.get_summary()
        return None
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """比较实验"""
        experiments = []
        for exp_id in experiment_ids:
            exp = self.active_experiments.get(exp_id)
            if not exp:
                exp = self.load_experiment(exp_id)
            if exp:
                experiments.append(exp)
        
        if len(experiments) < 2:
            return {"error": "需要至少2个实验进行比较"}
        
        comparison = {
            "experiment_count": len(experiments),
            "experiments": [],
            "performance_comparison": {},
            "config_comparison": {},
            "best_experiment": None
        }
        
        best_accuracy = 0.0
        best_exp_id = None
        
        for exp in experiments:
            exp_summary = exp.get_summary()
            comparison["experiments"].append(exp_summary)
            
            # 找出最佳实验
            metrics_summary = exp_summary.get("metrics_summary", {})
            val_acc = metrics_summary.get("best_val_accuracy", 0.0)
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_exp_id = exp.experiment_id
        
        comparison["best_experiment"] = best_exp_id
        
        # 性能比较
        metrics_to_compare = ["best_val_accuracy", "best_val_loss", "total_training_time"]
        for metric in metrics_to_compare:
            values = []
            for exp in experiments:
                metrics_summary = exp.get_summary().get("metrics_summary", {})
                values.append(metrics_summary.get(metric, 0.0))
            
            comparison["performance_comparison"][metric] = {
                "values": values,
                "best_value": max(values) if "accuracy" in metric else min(values),
                "worst_value": min(values) if "accuracy" in metric else max(values),
                "average": sum(values) / len(values),
                "std_dev": self._calculate_std_dev(values)
            }
        
        # 配置比较
        config_fields = ["learning_rate", "batch_size", "epochs", "optimizer"]
        for field in config_fields:
            values = [getattr(exp.config, field, None) for exp in experiments]
            comparison["config_comparison"][field] = {
                "values": values,
                "unique_values": list(set(str(v) for v in values if v is not None))
            }
        
        return comparison
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _start_monitoring(self, experiment_id: str):
        """开始监控实验"""
        if experiment_id in self.monitoring_threads:
            return
        
        stop_event = threading.Event()
        self.stop_monitoring[experiment_id] = stop_event
        
        monitor_thread = threading.Thread(
            target=self._monitor_experiment,
            args=(experiment_id, stop_event)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        self.monitoring_threads[experiment_id] = monitor_thread
    
    def _stop_monitoring(self, experiment_id: str):
        """停止监控实验"""
        if experiment_id in self.stop_monitoring:
            self.stop_monitoring[experiment_id].set()
            del self.stop_monitoring[experiment_id]
        
        if experiment_id in self.monitoring_threads:
            del self.monitoring_threads[experiment_id]
    
    def _monitor_experiment(self, experiment_id: str, stop_event: threading.Event):
        """监控实验资源使用"""
        experiment = self.active_experiments.get(experiment_id)
        if not experiment:
            return
        
        while not stop_event.is_set():
            try:
                # 获取系统资源使用情况
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                resource_usage = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_available_gb": memory.available / (1024**3)
                }
                
                # 如果有GPU，获取GPU使用情况
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
                    resource_usage.update({
                        "gpu_memory_allocated_gb": gpu_memory,
                        "gpu_memory_cached_gb": gpu_memory_cached
                    })
                
                # 更新实验资源使用情况
                if "resource_history" not in experiment.resource_usage:
                    experiment.resource_usage["resource_history"] = []
                
                experiment.resource_usage["resource_history"].append(resource_usage)
                
                # 只保留最近的100个记录
                if len(experiment.resource_usage["resource_history"]) > 100:
                    experiment.resource_usage["resource_history"] = \
                        experiment.resource_usage["resource_history"][-100:]
                
                # 更新当前资源使用情况
                experiment.resource_usage["current"] = resource_usage
                
                # 每30秒保存一次
                if len(experiment.resource_usage["resource_history"]) % 30 == 0:
                    experiment.save()
                
            except Exception as e:
                experiment.log(f"监控资源使用时出错: {e}", "WARNING")
            
            # 等待5秒
            stop_event.wait(5)
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """移除回调函数"""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """触发回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"回调函数执行失败: {e}")
    
    def get_active_experiments(self) -> List[str]:
        """获取活跃实验列表"""
        return list(self.active_experiments.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.database.get_statistics()
    
    def cleanup_old_experiments(self, days: int = 30):
        """清理旧实验"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 获取所有实验
        experiments = self.list_experiments()
        
        cleaned_count = 0
        for exp_summary in experiments:
            created_at = datetime.fromisoformat(exp_summary["created_at"])
            if created_at < cutoff_date and exp_summary["status"] in ["completed", "failed"]:
                exp_id = exp_summary["experiment_id"]
                
                # 删除实验目录
                exp_dir = self.base_dir / exp_id
                if exp_dir.exists():
                    import shutil
                    shutil.rmtree(exp_dir)
                
                # 从数据库删除
                self.database.delete_experiment(exp_id)
                
                cleaned_count += 1
        
        return cleaned_count
    
    def export_experiments(self, 
                          experiment_ids: List[str], 
                          output_file: str,
                          include_artifacts: bool = False):
        """导出实验数据"""
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "experiment_count": len(experiment_ids),
            "experiments": []
        }
        
        for exp_id in experiment_ids:
            exp = self.load_experiment(exp_id)
            if exp:
                exp_data = exp.get_summary()
                exp_data["full_config"] = exp.config.to_dict()
                exp_data["full_metrics"] = exp.metrics.to_dict()
                
                if include_artifacts:
                    exp_data["artifacts"] = exp.artifacts
                
                export_data["experiments"].append(exp_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def __del__(self):
        """析构函数，停止所有监控线程"""
        for exp_id in list(self.stop_monitoring.keys()):
            self._stop_monitoring(exp_id)