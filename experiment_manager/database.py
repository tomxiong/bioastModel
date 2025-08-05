"""实验数据库

提供实验数据的持久化存储和查询功能。
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .experiment import Experiment


class ExperimentDatabase:
    """实验数据库"""
    
    def __init__(self, db_path: str = "data/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建实验表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_name TEXT,
                    model_version TEXT,
                    dataset_name TEXT,
                    dataset_version TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_seconds REAL,
                    best_val_accuracy REAL,
                    best_val_loss REAL,
                    best_epoch INTEGER,
                    total_epochs INTEGER,
                    batch_size INTEGER,
                    learning_rate REAL,
                    optimizer TEXT,
                    scheduler TEXT,
                    notes TEXT,
                    error_message TEXT,
                    config_json TEXT,
                    metrics_json TEXT,
                    artifacts_json TEXT,
                    resource_usage_json TEXT
                )
            """)
            
            # 创建指标表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL,
                    train_accuracy REAL,
                    val_loss REAL,
                    val_accuracy REAL,
                    val_precision REAL,
                    val_recall REAL,
                    val_f1 REAL,
                    learning_rate REAL,
                    epoch_time REAL,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            # 创建标签表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id),
                    UNIQUE(experiment_id, tag)
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_model ON experiments(model_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON experiment_metrics(experiment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_experiment ON experiment_tags(experiment_id)")
            
            conn.commit()
    
    def save_experiment(self, experiment: Experiment):
        """保存实验到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 计算持续时间
            duration = None
            if experiment.started_at and experiment.completed_at:
                start = datetime.fromisoformat(experiment.started_at)
                end = datetime.fromisoformat(experiment.completed_at)
                duration = (end - start).total_seconds()
            
            # 获取最佳指标
            metrics_summary = experiment.metrics.get_summary()
            best_val_accuracy = metrics_summary.get("best_val_accuracy", 0.0)
            best_val_loss = metrics_summary.get("best_val_loss", 0.0)
            best_epoch = metrics_summary.get("best_epoch", 0)
            total_epochs = metrics_summary.get("total_epochs", 0)
            
            # 插入或更新实验记录
            cursor.execute("""
                INSERT OR REPLACE INTO experiments (
                    experiment_id, name, status, model_name, model_version,
                    dataset_name, dataset_version, created_at, started_at, completed_at,
                    duration_seconds, best_val_accuracy, best_val_loss, best_epoch,
                    total_epochs, batch_size, learning_rate, optimizer, scheduler,
                    notes, error_message, config_json, metrics_json, artifacts_json,
                    resource_usage_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.experiment_id,
                experiment.name,
                experiment.status,
                experiment.config.model_name,
                experiment.config.model_version,
                experiment.config.dataset_name,
                experiment.config.dataset_version,
                experiment.created_at,
                experiment.started_at,
                experiment.completed_at,
                duration,
                best_val_accuracy,
                best_val_loss,
                best_epoch,
                total_epochs,
                experiment.config.batch_size,
                experiment.config.learning_rate,
                experiment.config.optimizer,
                experiment.config.scheduler,
                experiment.notes,
                experiment.error_message,
                json.dumps(experiment.config.to_dict()),
                json.dumps(experiment.metrics.to_dict()),
                json.dumps(experiment.artifacts),
                json.dumps(experiment.resource_usage)
            ))
            
            conn.commit()
    
    def update_experiment(self, experiment: Experiment):
        """更新实验记录"""
        self.save_experiment(experiment)  # 使用INSERT OR REPLACE
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """获取实验记录"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def list_experiments(self, 
                        status: Optional[str] = None,
                        model_name: Optional[str] = None,
                        limit: int = 50,
                        offset: int = 0) -> List[Dict[str, Any]]:
        """列出实验"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM experiments"
            params = []
            conditions = []
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def search_experiments(self, query: str) -> List[Dict[str, Any]]:
        """搜索实验"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            search_query = """
                SELECT * FROM experiments 
                WHERE name LIKE ? OR notes LIKE ? OR model_name LIKE ?
                ORDER BY created_at DESC
            """
            
            search_term = f"%{query}%"
            cursor.execute(search_query, (search_term, search_term, search_term))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 删除相关的指标记录
            cursor.execute(
                "DELETE FROM experiment_metrics WHERE experiment_id = ?",
                (experiment_id,)
            )
            
            # 删除相关的标签记录
            cursor.execute(
                "DELETE FROM experiment_tags WHERE experiment_id = ?",
                (experiment_id,)
            )
            
            # 删除实验记录
            cursor.execute(
                "DELETE FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            
            conn.commit()
            return cursor.rowcount > 0
    
    def save_epoch_metrics(self, 
                          experiment_id: str,
                          epoch: int,
                          metrics: Dict[str, float]):
        """保存epoch指标"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO experiment_metrics (
                    experiment_id, epoch, train_loss, train_accuracy,
                    val_loss, val_accuracy, val_precision, val_recall,
                    val_f1, learning_rate, epoch_time, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                epoch,
                metrics.get("train_loss", 0.0),
                metrics.get("train_accuracy", 0.0),
                metrics.get("val_loss", 0.0),
                metrics.get("val_accuracy", 0.0),
                metrics.get("val_precision", 0.0),
                metrics.get("val_recall", 0.0),
                metrics.get("val_f1", 0.0),
                metrics.get("learning_rate", 0.0),
                metrics.get("epoch_time", 0.0),
                datetime.now().isoformat()
            ))
            
            conn.commit()
    
    def get_experiment_metrics(self, experiment_id: str) -> List[Dict[str, Any]]:
        """获取实验的所有指标"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM experiment_metrics WHERE experiment_id = ? ORDER BY epoch",
                (experiment_id,)
            )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def add_experiment_tag(self, experiment_id: str, tag: str):
        """添加实验标签"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT OR IGNORE INTO experiment_tags (experiment_id, tag) VALUES (?, ?)",
                (experiment_id, tag)
            )
            
            conn.commit()
    
    def remove_experiment_tag(self, experiment_id: str, tag: str):
        """移除实验标签"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM experiment_tags WHERE experiment_id = ? AND tag = ?",
                (experiment_id, tag)
            )
            
            conn.commit()
    
    def get_experiment_tags(self, experiment_id: str) -> List[str]:
        """获取实验标签"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT tag FROM experiment_tags WHERE experiment_id = ?",
                (experiment_id,)
            )
            
            return [row[0] for row in cursor.fetchall()]
    
    def get_experiments_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """根据标签获取实验"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT e.* FROM experiments e
                JOIN experiment_tags t ON e.experiment_id = t.experiment_id
                WHERE t.tag = ?
                ORDER BY e.created_at DESC
            """, (tag,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_model_performance_history(self, model_name: str) -> List[Dict[str, Any]]:
        """获取模型性能历史"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT experiment_id, name, model_version, created_at, completed_at,
                       best_val_accuracy, best_val_loss, total_epochs, batch_size,
                       learning_rate, optimizer, scheduler
                FROM experiments
                WHERE model_name = ? AND status = 'completed'
                ORDER BY created_at DESC
            """, (model_name,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_best_experiments(self, 
                           model_name: Optional[str] = None,
                           metric: str = "best_val_accuracy",
                           limit: int = 10) -> List[Dict[str, Any]]:
        """获取最佳实验"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = f"""
                SELECT * FROM experiments
                WHERE status = 'completed' AND {metric} IS NOT NULL
            """
            params = []
            
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            
            # 根据指标类型决定排序方向
            if "loss" in metric.lower():
                query += f" ORDER BY {metric} ASC LIMIT ?"
            else:
                query += f" ORDER BY {metric} DESC LIMIT ?"
            
            params.append(limit)
            
            cursor.execute(query, params)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 总实验数
            cursor.execute("SELECT COUNT(*) FROM experiments")
            total_experiments = cursor.fetchone()[0]
            
            # 按状态统计
            cursor.execute("""
                SELECT status, COUNT(*) FROM experiments GROUP BY status
            """)
            status_stats = dict(cursor.fetchall())
            
            # 按模型统计
            cursor.execute("""
                SELECT model_name, COUNT(*) FROM experiments 
                WHERE model_name IS NOT NULL 
                GROUP BY model_name 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """)
            model_stats = dict(cursor.fetchall())
            
            # 平均性能
            cursor.execute("""
                SELECT AVG(best_val_accuracy), AVG(best_val_loss), AVG(duration_seconds)
                FROM experiments 
                WHERE status = 'completed' AND best_val_accuracy IS NOT NULL
            """)
            avg_performance = cursor.fetchone()
            
            # 最近活动
            cursor.execute("""
                SELECT COUNT(*) FROM experiments 
                WHERE created_at > datetime('now', '-7 days')
            """)
            recent_experiments = cursor.fetchone()[0]
            
            return {
                "total_experiments": total_experiments,
                "status_distribution": status_stats,
                "model_distribution": model_stats,
                "average_performance": {
                    "accuracy": avg_performance[0] if avg_performance[0] else 0.0,
                    "loss": avg_performance[1] if avg_performance[1] else 0.0,
                    "duration_seconds": avg_performance[2] if avg_performance[2] else 0.0
                },
                "recent_experiments_7days": recent_experiments,
                "database_path": str(self.db_path)
            }
    
    def export_to_csv(self, output_file: str, experiment_ids: Optional[List[str]] = None):
        """导出实验数据到CSV"""
        import csv
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if experiment_ids:
                placeholders = ','.join('?' * len(experiment_ids))
                query = f"SELECT * FROM experiments WHERE experiment_id IN ({placeholders})"
                cursor.execute(query, experiment_ids)
            else:
                cursor.execute("SELECT * FROM experiments")
            
            rows = cursor.fetchall()
            
            if rows:
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(dict(row))
    
    def backup_database(self, backup_path: str):
        """备份数据库"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
    
    def restore_database(self, backup_path: str):
        """恢复数据库"""
        import shutil
        shutil.copy2(backup_path, self.db_path)
    
    def vacuum_database(self):
        """清理数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")
            conn.commit()