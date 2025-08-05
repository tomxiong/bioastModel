"""任务调度器

实现定时任务和批量处理功能。
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import schedule
from concurrent.futures import ThreadPoolExecutor, Future
import uuid


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SCHEDULED = "scheduled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class ScheduleType(Enum):
    """调度类型"""
    ONCE = "once"
    INTERVAL = "interval"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CRON = "cron"


@dataclass
class ScheduledTask:
    """调度任务"""
    name: str
    description: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    schedule_type: ScheduleType = ScheduleType.ONCE
    schedule_config: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    
    # 运行时状态
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    result: Any = None
    
    # 执行统计
    execution_count: int = 0
    total_runtime: float = 0.0
    last_execution_time: Optional[datetime] = None
    
    def reset(self):
        """重置任务状态"""
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.retry_count = 0
        self.error_message = None
        self.result = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'schedule_type': self.schedule_type.value,
            'schedule_config': self.schedule_config,
            'priority': self.priority.value,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'dependencies': self.dependencies,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count,
            'error_message': self.error_message,
            'execution_count': self.execution_count,
            'total_runtime': self.total_runtime,
            'last_execution_time': self.last_execution_time.isoformat() if self.last_execution_time else None,
            'average_runtime': self.total_runtime / max(self.execution_count, 1)
        }


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 storage_dir: str = "scheduler",
                 log_level: str = "INFO"):
        self.max_workers = max_workers
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # 任务存储
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_queue: List[ScheduledTask] = []
        self.running_tasks: Dict[str, Future] = {}
        self.completed_tasks: List[ScheduledTask] = []
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 调度器状态
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'task_scheduled': [],
            'task_started': [],
            'task_completed': [],
            'task_failed': [],
            'task_cancelled': []
        }
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        
        # 加载持久化任务
        self._load_tasks()
    
    def add_task(self, task: ScheduledTask) -> str:
        """添加任务"""
        self.tasks[task.task_id] = task
        
        # 根据调度类型设置调度时间
        self._schedule_task(task)
        
        self._trigger_callbacks('task_scheduled', task)
        self.logger.info(f"任务已添加: {task.name} ({task.task_id})")
        
        self._save_tasks()
        return task.task_id
    
    def create_task(self, 
                   name: str,
                   function: Callable,
                   description: str = "",
                   args: tuple = (),
                   kwargs: Optional[Dict[str, Any]] = None,
                   schedule_type: ScheduleType = ScheduleType.ONCE,
                   schedule_config: Optional[Dict[str, Any]] = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   max_retries: int = 3,
                   timeout: Optional[int] = None,
                   dependencies: Optional[List[str]] = None) -> str:
        """创建任务"""
        task = ScheduledTask(
            name=name,
            description=description,
            function=function,
            args=args,
            kwargs=kwargs or {},
            schedule_type=schedule_type,
            schedule_config=schedule_config or {},
            priority=priority,
            max_retries=max_retries,
            timeout=timeout,
            dependencies=dependencies or []
        )
        
        return self.add_task(task)
    
    def _schedule_task(self, task: ScheduledTask):
        """设置任务调度"""
        now = datetime.now()
        
        if task.schedule_type == ScheduleType.ONCE:
            # 立即执行或指定时间执行
            if 'at' in task.schedule_config:
                task.scheduled_at = datetime.fromisoformat(task.schedule_config['at'])
            else:
                task.scheduled_at = now
        
        elif task.schedule_type == ScheduleType.INTERVAL:
            # 间隔执行
            interval = task.schedule_config.get('seconds', 60)
            task.scheduled_at = now + timedelta(seconds=interval)
        
        elif task.schedule_type == ScheduleType.DAILY:
            # 每日执行
            time_str = task.schedule_config.get('at', '00:00')
            hour, minute = map(int, time_str.split(':'))
            
            scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if scheduled_time <= now:
                scheduled_time += timedelta(days=1)
            
            task.scheduled_at = scheduled_time
        
        elif task.schedule_type == ScheduleType.WEEKLY:
            # 每周执行
            weekday = task.schedule_config.get('weekday', 0)  # 0=Monday
            time_str = task.schedule_config.get('at', '00:00')
            hour, minute = map(int, time_str.split(':'))
            
            days_ahead = weekday - now.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            
            scheduled_time = now + timedelta(days=days_ahead)
            scheduled_time = scheduled_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            task.scheduled_at = scheduled_time
        
        elif task.schedule_type == ScheduleType.MONTHLY:
            # 每月执行
            day = task.schedule_config.get('day', 1)
            time_str = task.schedule_config.get('at', '00:00')
            hour, minute = map(int, time_str.split(':'))
            
            # 计算下个月的指定日期
            if now.month == 12:
                next_month = now.replace(year=now.year + 1, month=1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
            else:
                next_month = now.replace(month=now.month + 1, day=day, hour=hour, minute=minute, second=0, microsecond=0)
            
            task.scheduled_at = next_month
        
        task.status = TaskStatus.SCHEDULED
    
    def start_scheduler(self):
        """启动调度器"""
        if self.is_running:
            self.logger.warning("调度器已在运行")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("任务调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # 取消所有运行中的任务
        for task_id, future in self.running_tasks.items():
            future.cancel()
            task = self.tasks.get(task_id)
            if task:
                task.status = TaskStatus.CANCELLED
        
        self.running_tasks.clear()
        self.executor.shutdown(wait=True)
        
        self.logger.info("任务调度器已停止")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.is_running:
            try:
                self._check_scheduled_tasks()
                self._check_running_tasks()
                self._execute_ready_tasks()
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                self.logger.error(f"调度器循环错误: {e}")
                time.sleep(5)
    
    def _check_scheduled_tasks(self):
        """检查需要执行的调度任务"""
        now = datetime.now()
        
        for task in self.tasks.values():
            if (task.status == TaskStatus.SCHEDULED and 
                task.scheduled_at and 
                task.scheduled_at <= now):
                
                # 检查依赖
                if self._check_dependencies(task):
                    task.status = TaskStatus.PENDING
                    self.task_queue.append(task)
                    self.logger.debug(f"任务已加入队列: {task.name}")
    
    def _check_dependencies(self, task: ScheduledTask) -> bool:
        """检查任务依赖"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _check_running_tasks(self):
        """检查运行中的任务"""
        completed_tasks = []
        
        for task_id, future in self.running_tasks.items():
            if future.done():
                completed_tasks.append(task_id)
                task = self.tasks[task_id]
                
                try:
                    result = future.result()
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    
                    # 更新统计信息
                    task.execution_count += 1
                    if task.started_at:
                        runtime = (task.completed_at - task.started_at).total_seconds()
                        task.total_runtime += runtime
                    task.last_execution_time = task.completed_at
                    
                    self.completed_tasks.append(task)
                    self._trigger_callbacks('task_completed', task)
                    self.logger.info(f"任务完成: {task.name}")
                    
                    # 如果是重复任务，重新调度
                    if task.schedule_type != ScheduleType.ONCE:
                        self._reschedule_task(task)
                    
                except Exception as e:
                    task.error_message = str(e)
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    
                    self._trigger_callbacks('task_failed', task)
                    self.logger.error(f"任务失败: {task.name} - {e}")
                    
                    # 重试逻辑
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        self.task_queue.append(task)
                        self.logger.info(f"任务重试: {task.name} ({task.retry_count}/{task.max_retries})")
        
        # 清理已完成的任务
        for task_id in completed_tasks:
            del self.running_tasks[task_id]
    
    def _execute_ready_tasks(self):
        """执行准备好的任务"""
        # 按优先级排序
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        while (self.task_queue and 
               len(self.running_tasks) < self.max_workers):
            
            task = self.task_queue.pop(0)
            
            # 再次检查依赖
            if not self._check_dependencies(task):
                continue
            
            # 提交任务执行
            future = self.executor.submit(self._execute_task, task)
            self.running_tasks[task.task_id] = future
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            self._trigger_callbacks('task_started', task)
            self.logger.info(f"开始执行任务: {task.name}")
    
    def _execute_task(self, task: ScheduledTask) -> Any:
        """执行单个任务"""
        try:
            if task.timeout:
                # 带超时执行
                return self._execute_with_timeout(task)
            else:
                return task.function(*task.args, **task.kwargs)
        
        except Exception as e:
            self.logger.error(f"任务执行异常: {task.name} - {e}")
            raise e
    
    def _execute_with_timeout(self, task: ScheduledTask) -> Any:
        """带超时执行任务"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"任务执行超时: {task.timeout}秒")
        
        # 设置超时信号（仅在Unix系统上有效）
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(task.timeout)
            
            try:
                result = task.function(*task.args, **task.kwargs)
                signal.alarm(0)  # 取消超时
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Windows系统使用线程超时
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = task.function(*task.args, **task.kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(task.timeout)
            
            if thread.is_alive():
                raise TimeoutError(f"任务执行超时: {task.timeout}秒")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
    
    def _reschedule_task(self, task: ScheduledTask):
        """重新调度重复任务"""
        task.reset()
        self._schedule_task(task)
        self.logger.debug(f"任务已重新调度: {task.name}")
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # 如果任务正在运行，取消执行
        if task_id in self.running_tasks:
            future = self.running_tasks[task_id]
            future.cancel()
            del self.running_tasks[task_id]
        
        # 从队列中移除
        self.task_queue = [t for t in self.task_queue if t.task_id != task_id]
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        
        self._trigger_callbacks('task_cancelled', task)
        self.logger.info(f"任务已取消: {task.name}")
        
        return True
    
    def remove_task(self, task_id: str) -> bool:
        """移除任务"""
        if task_id in self.tasks:
            self.cancel_task(task_id)
            del self.tasks[task_id]
            self._save_tasks()
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def list_tasks(self, 
                  status: Optional[TaskStatus] = None,
                  priority: Optional[TaskPriority] = None) -> List[ScheduledTask]:
        """列出任务"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        if task:
            return task.to_dict()
        return None
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, task: ScheduledTask):
        """触发回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(task)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tasks = len(self.tasks)
        running_tasks = len(self.running_tasks)
        pending_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
        scheduled_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.SCHEDULED])
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        
        return {
            'total_tasks': total_tasks,
            'running_tasks': running_tasks,
            'pending_tasks': pending_tasks,
            'scheduled_tasks': scheduled_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': completed_tasks / max(total_tasks, 1),
            'queue_length': len(self.task_queue),
            'worker_utilization': running_tasks / self.max_workers
        }
    
    def _save_tasks(self):
        """保存任务到文件"""
        tasks_file = self.storage_dir / "tasks.json"
        
        # 序列化任务（排除function对象）
        serializable_tasks = {}
        for task_id, task in self.tasks.items():
            task_dict = task.to_dict()
            # 保存函数名而不是函数对象
            task_dict['function_name'] = getattr(task.function, '__name__', 'unknown')
            serializable_tasks[task_id] = task_dict
        
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_tasks, f, indent=2, ensure_ascii=False)
    
    def _load_tasks(self):
        """从文件加载任务"""
        tasks_file = self.storage_dir / "tasks.json"
        
        if not tasks_file.exists():
            return
        
        try:
            with open(tasks_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
            
            # 注意：这里只能加载任务元数据，函数对象需要重新注册
            for task_id, task_dict in tasks_data.items():
                # 创建任务对象（但没有函数）
                task = ScheduledTask(
                    name=task_dict['name'],
                    description=task_dict['description'],
                    function=lambda: None,  # 占位符函数
                    schedule_type=ScheduleType(task_dict['schedule_type']),
                    schedule_config=task_dict['schedule_config'],
                    priority=TaskPriority(task_dict['priority']),
                    max_retries=task_dict['max_retries'],
                    timeout=task_dict['timeout'],
                    dependencies=task_dict['dependencies']
                )
                
                # 恢复状态
                task.task_id = task_id
                task.status = TaskStatus(task_dict['status'])
                task.created_at = datetime.fromisoformat(task_dict['created_at'])
                task.scheduled_at = datetime.fromisoformat(task_dict['scheduled_at']) if task_dict['scheduled_at'] else None
                task.retry_count = task_dict['retry_count']
                task.execution_count = task_dict['execution_count']
                task.total_runtime = task_dict['total_runtime']
                
                # 只加载未完成的任务
                if task.status in [TaskStatus.PENDING, TaskStatus.SCHEDULED]:
                    self.tasks[task_id] = task
            
            self.logger.info(f"已加载 {len(self.tasks)} 个任务")
            
        except Exception as e:
            self.logger.error(f"加载任务失败: {e}")
    
    def create_model_training_task(self, 
                                 model_name: str,
                                 dataset_path: str,
                                 config: Dict[str, Any],
                                 schedule_config: Optional[Dict[str, Any]] = None) -> str:
        """创建模型训练任务"""
        def training_function():
            # 这里应该调用实际的训练逻辑
            from ..workflow.automation import WorkflowAutomation
            
            automation = WorkflowAutomation()
            workflow_config = automation.create_model_training_workflow(
                model_name, dataset_path, config
            )
            
            return automation.execute_workflow(workflow_config)
        
        return self.create_task(
            name=f"train_{model_name}",
            description=f"训练模型 {model_name}",
            function=training_function,
            schedule_type=ScheduleType.ONCE,
            schedule_config=schedule_config or {},
            priority=TaskPriority.HIGH,
            timeout=config.get('training_timeout', 3600)
        )
    
    def create_model_evaluation_task(self, 
                                   model_ids: List[str],
                                   schedule_config: Optional[Dict[str, Any]] = None) -> str:
        """创建模型评估任务"""
        def evaluation_function():
            # 这里应该调用实际的评估逻辑
            results = {}
            for model_id in model_ids:
                # 模拟评估
                results[model_id] = {
                    'accuracy': 0.95,
                    'precision': 0.94,
                    'recall': 0.96
                }
            return results
        
        return self.create_task(
            name=f"evaluate_models",
            description=f"评估 {len(model_ids)} 个模型",
            function=evaluation_function,
            schedule_type=ScheduleType.ONCE,
            schedule_config=schedule_config or {},
            priority=TaskPriority.NORMAL
        )
    
    def create_cleanup_task(self, 
                          days_to_keep: int = 30,
                          schedule_config: Optional[Dict[str, Any]] = None) -> str:
        """创建清理任务"""
        def cleanup_function():
            # 清理旧的实验数据、日志等
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # 这里应该实现实际的清理逻辑
            cleaned_files = 0
            freed_space = 0
            
            return {
                'cleaned_files': cleaned_files,
                'freed_space_mb': freed_space
            }
        
        return self.create_task(
            name="cleanup_old_data",
            description=f"清理 {days_to_keep} 天前的数据",
            function=cleanup_function,
            schedule_type=ScheduleType.WEEKLY,
            schedule_config=schedule_config or {'weekday': 6, 'at': '02:00'},  # 周日凌晨2点
            priority=TaskPriority.LOW
        )
    
    def create_backup_task(self, 
                         backup_path: str,
                         schedule_config: Optional[Dict[str, Any]] = None) -> str:
        """创建备份任务"""
        def backup_function():
            # 备份重要数据
            import shutil
            
            backup_dir = Path(backup_path) / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 这里应该实现实际的备份逻辑
            backed_up_files = 0
            backup_size = 0
            
            return {
                'backup_path': str(backup_dir),
                'backed_up_files': backed_up_files,
                'backup_size_mb': backup_size
            }
        
        return self.create_task(
            name="backup_data",
            description="备份重要数据",
            function=backup_function,
            schedule_type=ScheduleType.DAILY,
            schedule_config=schedule_config or {'at': '01:00'},  # 每天凌晨1点
            priority=TaskPriority.NORMAL
        )
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_scheduler()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_scheduler()