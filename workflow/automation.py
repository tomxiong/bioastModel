"""工作流自动化

实现模型训练和管理的自动化流程。
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

from ..model_registry import ModelRegistry
from ..experiment_manager import ExperimentTracker
from ..dashboard import Dashboard


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(Enum):
    """工作流状态"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """工作流步骤"""
    name: str
    description: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Any = None
    
    def reset(self):
        """重置步骤状态"""
        self.status = StepStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error_message = None
        self.result = None
        self.retry_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'dependencies': self.dependencies,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
        }


@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str
    description: str
    steps: List[WorkflowStep]
    parallel_execution: bool = False
    max_parallel_steps: int = 3
    auto_retry: bool = True
    notification_enabled: bool = True
    save_artifacts: bool = True
    cleanup_on_failure: bool = False
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        # 检查步骤名称唯一性
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            errors.append("步骤名称必须唯一")
        
        # 检查依赖关系
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(f"步骤 '{step.name}' 的依赖 '{dep}' 不存在")
        
        # 检查循环依赖
        if self._has_circular_dependency():
            errors.append("存在循环依赖")
        
        return errors
    
    def _has_circular_dependency(self) -> bool:
        """检查循环依赖"""
        visited = set()
        rec_stack = set()
        
        def dfs(step_name: str) -> bool:
            visited.add(step_name)
            rec_stack.add(step_name)
            
            step = next((s for s in self.steps if s.name == step_name), None)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(step_name)
            return False
        
        for step in self.steps:
            if step.name not in visited:
                if dfs(step.name):
                    return True
        
        return False
    
    def get_execution_order(self) -> List[List[str]]:
        """获取执行顺序（按层级）"""
        # 拓扑排序
        in_degree = {step.name: len(step.dependencies) for step in self.steps}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current_level = queue.copy()
            queue.clear()
            result.append(current_level)
            
            for step_name in current_level:
                step = next(s for s in self.steps if s.name == step_name)
                # 更新依赖此步骤的其他步骤
                for other_step in self.steps:
                    if step_name in other_step.dependencies:
                        in_degree[other_step.name] -= 1
                        if in_degree[other_step.name] == 0:
                            queue.append(other_step.name)
        
        return result


class WorkflowAutomation:
    """工作流自动化管理器"""
    
    def __init__(self, 
                 storage_dir: str = "workflows",
                 log_level: str = "INFO"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))
        
        # 当前运行的工作流
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'workflow_started': [],
            'workflow_completed': [],
            'workflow_failed': [],
            'step_started': [],
            'step_completed': [],
            'step_failed': []
        }
        
        # 组件引用
        self.model_registry = None
        self.experiment_tracker = None
        self.dashboard = None
        
        self._load_history()
    
    def set_components(self, 
                      model_registry: ModelRegistry,
                      experiment_tracker: ExperimentTracker,
                      dashboard: Dashboard):
        """设置组件引用"""
        self.model_registry = model_registry
        self.experiment_tracker = experiment_tracker
        self.dashboard = dashboard
    
    def create_model_training_workflow(self, 
                                     model_name: str,
                                     dataset_path: str,
                                     config: Dict[str, Any]) -> WorkflowConfig:
        """创建模型训练工作流"""
        steps = [
            WorkflowStep(
                name="validate_inputs",
                description="验证输入参数和数据",
                function=lambda: self._validate_training_inputs(model_name, dataset_path, config)
            ),
            WorkflowStep(
                name="prepare_data",
                description="准备训练数据",
                function=lambda: self._prepare_training_data(dataset_path, config),
                dependencies=["validate_inputs"]
            ),
            WorkflowStep(
                name="create_model",
                description="创建模型实例",
                function=lambda: self._create_model_instance(model_name, config),
                dependencies=["validate_inputs"]
            ),
            WorkflowStep(
                name="train_model",
                description="训练模型",
                function=lambda: self._train_model(model_name, dataset_path, config),
                dependencies=["prepare_data", "create_model"],
                timeout=config.get('training_timeout', 3600)
            ),
            WorkflowStep(
                name="evaluate_model",
                description="评估模型性能",
                function=lambda: self._evaluate_model(model_name, config),
                dependencies=["train_model"]
            ),
            WorkflowStep(
                name="register_model",
                description="注册模型到注册表",
                function=lambda: self._register_trained_model(model_name, config),
                dependencies=["evaluate_model"]
            ),
            WorkflowStep(
                name="generate_report",
                description="生成训练报告",
                function=lambda: self._generate_training_report(model_name, config),
                dependencies=["register_model"]
            )
        ]
        
        return WorkflowConfig(
            name=f"train_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"训练模型 {model_name}",
            steps=steps,
            parallel_execution=config.get('parallel_execution', False),
            auto_retry=config.get('auto_retry', True),
            notification_enabled=config.get('notification_enabled', True)
        )
    
    def create_model_comparison_workflow(self, 
                                       experiment_ids: List[str]) -> WorkflowConfig:
        """创建模型对比工作流"""
        steps = [
            WorkflowStep(
                name="load_experiments",
                description="加载实验数据",
                function=lambda: self._load_experiments_for_comparison(experiment_ids)
            ),
            WorkflowStep(
                name="analyze_performance",
                description="分析性能指标",
                function=lambda: self._analyze_model_performance(experiment_ids),
                dependencies=["load_experiments"]
            ),
            WorkflowStep(
                name="generate_comparison_charts",
                description="生成对比图表",
                function=lambda: self._generate_comparison_charts(experiment_ids),
                dependencies=["analyze_performance"]
            ),
            WorkflowStep(
                name="create_comparison_report",
                description="创建对比报告",
                function=lambda: self._create_comparison_report(experiment_ids),
                dependencies=["generate_comparison_charts"]
            )
        ]
        
        return WorkflowConfig(
            name=f"compare_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"对比 {len(experiment_ids)} 个模型",
            steps=steps,
            parallel_execution=True
        )
    
    def execute_workflow(self, 
                        workflow_config: WorkflowConfig,
                        context: Optional[Dict[str, Any]] = None) -> str:
        """执行工作流"""
        # 验证配置
        errors = workflow_config.validate()
        if errors:
            raise ValueError(f"工作流配置错误: {', '.join(errors)}")
        
        workflow_id = f"{workflow_config.name}_{int(time.time())}"
        
        # 创建工作流实例
        workflow_instance = {
            'id': workflow_id,
            'config': workflow_config,
            'context': context or {},
            'status': WorkflowStatus.CREATED,
            'start_time': datetime.now(),
            'end_time': None,
            'current_step': None,
            'completed_steps': [],
            'failed_steps': [],
            'artifacts': {},
            'logs': []
        }
        
        self.active_workflows[workflow_id] = workflow_instance
        
        # 在后台线程中执行
        thread = threading.Thread(
            target=self._execute_workflow_thread,
            args=(workflow_id,),
            daemon=True
        )
        thread.start()
        
        return workflow_id
    
    def _execute_workflow_thread(self, workflow_id: str):
        """在线程中执行工作流"""
        workflow = self.active_workflows[workflow_id]
        config = workflow['config']
        
        try:
            workflow['status'] = WorkflowStatus.RUNNING
            self._trigger_callbacks('workflow_started', workflow_id, workflow)
            
            # 获取执行顺序
            execution_order = config.get_execution_order()
            
            for level in execution_order:
                if config.parallel_execution and len(level) > 1:
                    # 并行执行
                    self._execute_steps_parallel(workflow_id, level)
                else:
                    # 串行执行
                    for step_name in level:
                        self._execute_step(workflow_id, step_name)
            
            # 工作流完成
            workflow['status'] = WorkflowStatus.COMPLETED
            workflow['end_time'] = datetime.now()
            
            self._trigger_callbacks('workflow_completed', workflow_id, workflow)
            self.logger.info(f"工作流 {workflow_id} 执行完成")
            
        except Exception as e:
            workflow['status'] = WorkflowStatus.FAILED
            workflow['end_time'] = datetime.now()
            workflow['error'] = str(e)
            
            self._trigger_callbacks('workflow_failed', workflow_id, workflow)
            self.logger.error(f"工作流 {workflow_id} 执行失败: {e}")
        
        finally:
            # 移动到历史记录
            self.workflow_history.append(workflow.copy())
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            self._save_history()
    
    def _execute_step(self, workflow_id: str, step_name: str):
        """执行单个步骤"""
        workflow = self.active_workflows[workflow_id]
        config = workflow['config']
        step = next(s for s in config.steps if s.name == step_name)
        
        workflow['current_step'] = step_name
        step.start_time = datetime.now()
        step.status = StepStatus.RUNNING
        
        self._trigger_callbacks('step_started', workflow_id, step)
        self.logger.info(f"开始执行步骤: {step_name}")
        
        try:
            # 执行步骤函数
            if step.timeout:
                # 带超时执行
                result = self._execute_with_timeout(step.function, step.timeout)
            else:
                result = step.function()
            
            step.result = result
            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now()
            
            workflow['completed_steps'].append(step_name)
            workflow['artifacts'][step_name] = result
            
            self._trigger_callbacks('step_completed', workflow_id, step)
            self.logger.info(f"步骤 {step_name} 执行完成")
            
        except Exception as e:
            step.error_message = str(e)
            step.status = StepStatus.FAILED
            step.end_time = datetime.now()
            
            # 重试逻辑
            if config.auto_retry and step.retry_count < step.max_retries:
                step.retry_count += 1
                self.logger.warning(f"步骤 {step_name} 失败，正在重试 ({step.retry_count}/{step.max_retries})")
                time.sleep(2 ** step.retry_count)  # 指数退避
                self._execute_step(workflow_id, step_name)
                return
            
            workflow['failed_steps'].append(step_name)
            self._trigger_callbacks('step_failed', workflow_id, step)
            self.logger.error(f"步骤 {step_name} 执行失败: {e}")
            
            raise e
    
    def _execute_steps_parallel(self, workflow_id: str, step_names: List[str]):
        """并行执行多个步骤"""
        threads = []
        exceptions = []
        
        def execute_step_wrapper(step_name):
            try:
                self._execute_step(workflow_id, step_name)
            except Exception as e:
                exceptions.append((step_name, e))
        
        # 创建并启动线程
        for step_name in step_names:
            thread = threading.Thread(target=execute_step_wrapper, args=(step_name,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查是否有异常
        if exceptions:
            error_msg = "; ".join([f"{name}: {str(e)}" for name, e in exceptions])
            raise Exception(f"并行步骤执行失败: {error_msg}")
    
    def _execute_with_timeout(self, func: Callable, timeout: int) -> Any:
        """带超时执行函数"""
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # 超时了，但无法强制终止线程
            raise TimeoutError(f"步骤执行超时 ({timeout}秒)")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    # 工作流步骤实现
    def _validate_training_inputs(self, model_name: str, dataset_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证训练输入"""
        validation_result = {
            'model_name_valid': bool(model_name and model_name.strip()),
            'dataset_exists': os.path.exists(dataset_path),
            'config_valid': isinstance(config, dict) and len(config) > 0
        }
        
        if not all(validation_result.values()):
            raise ValueError(f"输入验证失败: {validation_result}")
        
        return validation_result
    
    def _prepare_training_data(self, dataset_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """准备训练数据"""
        # 这里应该实现数据预处理逻辑
        return {
            'dataset_path': dataset_path,
            'preprocessing_config': config.get('preprocessing', {}),
            'data_ready': True
        }
    
    def _create_model_instance(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建模型实例"""
        # 这里应该实现模型创建逻辑
        return {
            'model_name': model_name,
            'model_config': config.get('model', {}),
            'model_created': True
        }
    
    def _train_model(self, model_name: str, dataset_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """训练模型"""
        # 这里应该调用实际的训练逻辑
        # 可以集成现有的 train_single_model.py
        return {
            'model_name': model_name,
            'training_completed': True,
            'best_accuracy': 0.95,  # 示例值
            'training_time': 300    # 示例值
        }
    
    def _evaluate_model(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """评估模型"""
        # 这里应该实现模型评估逻辑
        return {
            'model_name': model_name,
            'evaluation_metrics': {
                'accuracy': 0.95,
                'precision': 0.94,
                'recall': 0.96,
                'f1_score': 0.95
            }
        }
    
    def _register_trained_model(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """注册训练好的模型"""
        if self.model_registry:
            # 使用模型注册表注册模型
            pass
        
        return {
            'model_name': model_name,
            'registered': True,
            'model_id': f"{model_name}_{int(time.time())}"
        }
    
    def _generate_training_report(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成训练报告"""
        if self.dashboard:
            # 使用仪表板生成报告
            pass
        
        return {
            'model_name': model_name,
            'report_generated': True,
            'report_path': f"reports/{model_name}_training_report.html"
        }
    
    def _load_experiments_for_comparison(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """加载用于对比的实验"""
        return {
            'experiment_ids': experiment_ids,
            'experiments_loaded': len(experiment_ids)
        }
    
    def _analyze_model_performance(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """分析模型性能"""
        return {
            'experiment_ids': experiment_ids,
            'performance_analysis': {
                'best_model': experiment_ids[0] if experiment_ids else None,
                'average_accuracy': 0.92
            }
        }
    
    def _generate_comparison_charts(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """生成对比图表"""
        return {
            'experiment_ids': experiment_ids,
            'charts_generated': True,
            'chart_paths': [f"charts/comparison_{i}.png" for i in range(3)]
        }
    
    def _create_comparison_report(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """创建对比报告"""
        return {
            'experiment_ids': experiment_ids,
            'report_created': True,
            'report_path': f"reports/model_comparison_{int(time.time())}.html"
        }
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args):
        """触发回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流状态"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                'id': workflow_id,
                'status': workflow['status'].value,
                'current_step': workflow['current_step'],
                'completed_steps': workflow['completed_steps'],
                'failed_steps': workflow['failed_steps'],
                'start_time': workflow['start_time'].isoformat(),
                'progress': len(workflow['completed_steps']) / len(workflow['config'].steps)
            }
        
        # 查找历史记录
        for workflow in self.workflow_history:
            if workflow['id'] == workflow_id:
                return {
                    'id': workflow_id,
                    'status': workflow['status'].value,
                    'completed_steps': workflow['completed_steps'],
                    'failed_steps': workflow['failed_steps'],
                    'start_time': workflow['start_time'].isoformat(),
                    'end_time': workflow['end_time'].isoformat() if workflow['end_time'] else None
                }
        
        return None
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow['status'] = WorkflowStatus.CANCELLED
            workflow['end_time'] = datetime.now()
            
            # 移动到历史记录
            self.workflow_history.append(workflow.copy())
            del self.active_workflows[workflow_id]
            
            self.logger.info(f"工作流 {workflow_id} 已取消")
            return True
        
        return False
    
    def list_workflows(self, 
                      status: Optional[str] = None,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """列出工作流"""
        all_workflows = []
        
        # 添加活跃工作流
        for workflow in self.active_workflows.values():
            all_workflows.append({
                'id': workflow['id'],
                'name': workflow['config'].name,
                'status': workflow['status'].value,
                'start_time': workflow['start_time'].isoformat(),
                'current_step': workflow['current_step'],
                'progress': len(workflow['completed_steps']) / len(workflow['config'].steps)
            })
        
        # 添加历史工作流
        for workflow in self.workflow_history:
            all_workflows.append({
                'id': workflow['id'],
                'name': workflow['config'].name,
                'status': workflow['status'].value,
                'start_time': workflow['start_time'].isoformat(),
                'end_time': workflow['end_time'].isoformat() if workflow['end_time'] else None,
                'duration': (workflow['end_time'] - workflow['start_time']).total_seconds() if workflow['end_time'] else None
            })
        
        # 过滤状态
        if status:
            all_workflows = [w for w in all_workflows if w['status'] == status]
        
        # 按开始时间排序
        all_workflows.sort(key=lambda x: x['start_time'], reverse=True)
        
        return all_workflows[:limit]
    
    def _save_history(self):
        """保存历史记录"""
        history_file = self.storage_dir / "workflow_history.json"
        
        # 序列化历史记录
        serializable_history = []
        for workflow in self.workflow_history:
            serializable_workflow = workflow.copy()
            # 转换日期时间
            if 'start_time' in serializable_workflow:
                serializable_workflow['start_time'] = serializable_workflow['start_time'].isoformat()
            if 'end_time' in serializable_workflow and serializable_workflow['end_time']:
                serializable_workflow['end_time'] = serializable_workflow['end_time'].isoformat()
            # 移除不可序列化的对象
            if 'config' in serializable_workflow:
                del serializable_workflow['config']
            
            serializable_history.append(serializable_workflow)
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
    
    def _load_history(self):
        """加载历史记录"""
        history_file = self.storage_dir / "workflow_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                # 反序列化历史记录
                for workflow_data in history_data:
                    if 'start_time' in workflow_data:
                        workflow_data['start_time'] = datetime.fromisoformat(workflow_data['start_time'])
                    if 'end_time' in workflow_data and workflow_data['end_time']:
                        workflow_data['end_time'] = datetime.fromisoformat(workflow_data['end_time'])
                    if 'status' in workflow_data:
                        workflow_data['status'] = WorkflowStatus(workflow_data['status'])
                
                self.workflow_history = history_data
                
            except Exception as e:
                self.logger.error(f"加载工作流历史失败: {e}")
                self.workflow_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_workflows = len(self.workflow_history) + len(self.active_workflows)
        completed_workflows = len([w for w in self.workflow_history if w['status'] == WorkflowStatus.COMPLETED])
        failed_workflows = len([w for w in self.workflow_history if w['status'] == WorkflowStatus.FAILED])
        
        return {
            'total_workflows': total_workflows,
            'active_workflows': len(self.active_workflows),
            'completed_workflows': completed_workflows,
            'failed_workflows': failed_workflows,
            'success_rate': completed_workflows / max(len(self.workflow_history), 1),
            'average_duration': self._calculate_average_duration()
        }
    
    def _calculate_average_duration(self) -> float:
        """计算平均执行时间"""
        completed_workflows = [w for w in self.workflow_history 
                             if w['status'] == WorkflowStatus.COMPLETED and w['end_time']]
        
        if not completed_workflows:
            return 0.0
        
        total_duration = sum([
            (w['end_time'] - w['start_time']).total_seconds()
            for w in completed_workflows
        ])
        
        return total_duration / len(completed_workflows)