"""模型管道

实现模型训练的标准化流程管理。
"""

import os
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging


class PipelineStageStatus(Enum):
    """管道阶段状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStatus(Enum):
    """管道状态"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStage:
    """管道阶段"""
    name: str
    description: str
    handler: Callable
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: PipelineStageStatus = PipelineStageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def reset(self):
        """重置阶段状态"""
        self.status = PipelineStageStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error_message = None
        self.artifacts = {}
        self.metrics = {}
        self.retry_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'dependencies': self.dependencies,
            'optional': self.optional,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            'artifacts': self.artifacts,
            'metrics': self.metrics
        }


class ModelPipeline:
    """模型管道"""
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 workspace_dir: str = "pipeline_workspace"):
        self.name = name
        self.description = description
        self.workspace_dir = Path(workspace_dir) / name
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # 管道配置
        self.stages: List[PipelineStage] = []
        self.global_config: Dict[str, Any] = {}
        self.environment: Dict[str, Any] = {}
        
        # 执行状态
        self.status = PipelineStatus.CREATED
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_stage: Optional[str] = None
        self.completed_stages: List[str] = []
        self.failed_stages: List[str] = []
        
        # 数据存储
        self.stage_outputs: Dict[str, Any] = {}
        self.pipeline_artifacts: Dict[str, Any] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'pipeline_started': [],
            'pipeline_completed': [],
            'pipeline_failed': [],
            'stage_started': [],
            'stage_completed': [],
            'stage_failed': []
        }
        
        # 设置日志
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.logger.setLevel(logging.INFO)
    
    def add_stage(self, stage: PipelineStage) -> 'ModelPipeline':
        """添加管道阶段"""
        self.stages.append(stage)
        return self
    
    def set_config(self, config: Dict[str, Any]) -> 'ModelPipeline':
        """设置全局配置"""
        self.global_config.update(config)
        return self
    
    def set_environment(self, env: Dict[str, Any]) -> 'ModelPipeline':
        """设置环境变量"""
        self.environment.update(env)
        return self
    
    def add_callback(self, event: str, callback: Callable) -> 'ModelPipeline':
        """添加回调函数"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        return self
    
    def validate(self) -> List[str]:
        """验证管道配置"""
        errors = []
        
        # 检查阶段名称唯一性
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            errors.append("阶段名称必须唯一")
        
        # 检查依赖关系
        for stage in self.stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    errors.append(f"阶段 '{stage.name}' 的依赖 '{dep}' 不存在")
        
        # 检查输入输出关系
        all_outputs = set()
        for stage in self.stages:
            all_outputs.update(stage.outputs)
        
        for stage in self.stages:
            for input_name in stage.inputs:
                if input_name not in all_outputs and input_name not in self.environment:
                    errors.append(f"阶段 '{stage.name}' 的输入 '{input_name}' 没有对应的输出")
        
        # 检查循环依赖
        if self._has_circular_dependency():
            errors.append("存在循环依赖")
        
        return errors
    
    def _has_circular_dependency(self) -> bool:
        """检查循环依赖"""
        visited = set()
        rec_stack = set()
        
        def dfs(stage_name: str) -> bool:
            visited.add(stage_name)
            rec_stack.add(stage_name)
            
            stage = next((s for s in self.stages if s.name == stage_name), None)
            if stage:
                for dep in stage.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(stage_name)
            return False
        
        for stage in self.stages:
            if stage.name not in visited:
                if dfs(stage.name):
                    return True
        
        return False
    
    def get_execution_order(self) -> List[str]:
        """获取执行顺序"""
        # 拓扑排序
        in_degree = {stage.name: len(stage.dependencies) for stage in self.stages}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            stage_name = queue.pop(0)
            result.append(stage_name)
            
            stage = next(s for s in self.stages if s.name == stage_name)
            # 更新依赖此阶段的其他阶段
            for other_stage in self.stages:
                if stage_name in other_stage.dependencies:
                    in_degree[other_stage.name] -= 1
                    if in_degree[other_stage.name] == 0:
                        queue.append(other_stage.name)
        
        return result
    
    def execute(self, 
               inputs: Optional[Dict[str, Any]] = None,
               resume_from: Optional[str] = None) -> Dict[str, Any]:
        """执行管道"""
        # 验证配置
        errors = self.validate()
        if errors:
            raise ValueError(f"管道配置错误: {', '.join(errors)}")
        
        # 初始化
        if inputs:
            self.stage_outputs.update(inputs)
        
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now()
        self.current_stage = None
        
        if not resume_from:
            self.completed_stages = []
            self.failed_stages = []
            self.execution_log = []
        
        self._trigger_callbacks('pipeline_started')
        self.logger.info(f"开始执行管道: {self.name}")
        
        try:
            # 获取执行顺序
            execution_order = self.get_execution_order()
            
            # 如果是恢复执行，跳过已完成的阶段
            if resume_from:
                try:
                    start_index = execution_order.index(resume_from)
                    execution_order = execution_order[start_index:]
                except ValueError:
                    raise ValueError(f"恢复点 '{resume_from}' 不存在")
            
            # 执行各个阶段
            for stage_name in execution_order:
                if stage_name in self.completed_stages:
                    continue
                
                self._execute_stage(stage_name)
            
            # 管道完成
            self.status = PipelineStatus.COMPLETED
            self.end_time = datetime.now()
            
            self._trigger_callbacks('pipeline_completed')
            self.logger.info(f"管道 {self.name} 执行完成")
            
            return self.pipeline_artifacts
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.end_time = datetime.now()
            
            self._trigger_callbacks('pipeline_failed')
            self.logger.error(f"管道 {self.name} 执行失败: {e}")
            
            raise e
        
        finally:
            self._save_execution_state()
    
    def _execute_stage(self, stage_name: str):
        """执行单个阶段"""
        stage = next(s for s in self.stages if s.name == stage_name)
        
        # 检查依赖是否完成
        for dep in stage.dependencies:
            if dep not in self.completed_stages:
                if not stage.optional:
                    raise RuntimeError(f"阶段 '{stage_name}' 的依赖 '{dep}' 未完成")
                else:
                    stage.status = PipelineStageStatus.SKIPPED
                    self.logger.info(f"跳过可选阶段: {stage_name} (依赖未满足)")
                    return
        
        self.current_stage = stage_name
        stage.start_time = datetime.now()
        stage.status = PipelineStageStatus.RUNNING
        
        self._trigger_callbacks('stage_started', stage)
        self.logger.info(f"开始执行阶段: {stage_name}")
        
        try:
            # 准备输入数据
            stage_inputs = self._prepare_stage_inputs(stage)
            
            # 创建阶段工作目录
            stage_workspace = self.workspace_dir / stage_name
            stage_workspace.mkdir(exist_ok=True)
            
            # 执行阶段处理函数
            context = {
                'stage_name': stage_name,
                'workspace': str(stage_workspace),
                'global_config': self.global_config,
                'environment': self.environment,
                'inputs': stage_inputs,
                'pipeline_artifacts': self.pipeline_artifacts
            }
            
            if stage.timeout:
                # 带超时执行
                result = self._execute_with_timeout(stage.handler, context, stage.timeout)
            else:
                result = stage.handler(context)
            
            # 处理结果
            if isinstance(result, dict):
                stage.artifacts.update(result.get('artifacts', {}))
                stage.metrics.update(result.get('metrics', {}))
                
                # 更新输出数据
                outputs = result.get('outputs', {})
                for output_name in stage.outputs:
                    if output_name in outputs:
                        self.stage_outputs[output_name] = outputs[output_name]
                
                # 更新管道级别的工件
                if 'pipeline_artifacts' in result:
                    self.pipeline_artifacts.update(result['pipeline_artifacts'])
            
            stage.status = PipelineStageStatus.COMPLETED
            stage.end_time = datetime.now()
            
            self.completed_stages.append(stage_name)
            
            # 记录执行日志
            self.execution_log.append({
                'stage': stage_name,
                'status': 'completed',
                'start_time': stage.start_time.isoformat(),
                'end_time': stage.end_time.isoformat(),
                'duration': (stage.end_time - stage.start_time).total_seconds(),
                'metrics': stage.metrics
            })
            
            self._trigger_callbacks('stage_completed', stage)
            self.logger.info(f"阶段 {stage_name} 执行完成")
            
        except Exception as e:
            stage.error_message = str(e)
            stage.status = PipelineStageStatus.FAILED
            stage.end_time = datetime.now()
            
            # 重试逻辑
            if stage.retry_count < stage.max_retries:
                stage.retry_count += 1
                self.logger.warning(f"阶段 {stage_name} 失败，正在重试 ({stage.retry_count}/{stage.max_retries})")
                time.sleep(2 ** stage.retry_count)  # 指数退避
                stage.reset()
                self._execute_stage(stage_name)
                return
            
            self.failed_stages.append(stage_name)
            
            # 记录失败日志
            self.execution_log.append({
                'stage': stage_name,
                'status': 'failed',
                'start_time': stage.start_time.isoformat(),
                'end_time': stage.end_time.isoformat(),
                'error': str(e)
            })
            
            self._trigger_callbacks('stage_failed', stage)
            self.logger.error(f"阶段 {stage_name} 执行失败: {e}")
            
            # 如果不是可选阶段，则终止管道
            if not stage.optional:
                raise e
            else:
                self.logger.warning(f"可选阶段 {stage_name} 失败，继续执行")
    
    def _prepare_stage_inputs(self, stage: PipelineStage) -> Dict[str, Any]:
        """准备阶段输入数据"""
        inputs = {}
        
        for input_name in stage.inputs:
            if input_name in self.stage_outputs:
                inputs[input_name] = self.stage_outputs[input_name]
            elif input_name in self.environment:
                inputs[input_name] = self.environment[input_name]
            else:
                raise ValueError(f"阶段 '{stage.name}' 的输入 '{input_name}' 不可用")
        
        return inputs
    
    def _execute_with_timeout(self, handler: Callable, context: Dict[str, Any], timeout: int) -> Any:
        """带超时执行处理函数"""
        import threading
        
        result = [None]
        exception = [None]
        
        def target():
            try:
                result[0] = handler(context)
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"阶段执行超时 ({timeout}秒)")
        
        if exception[0]:
            raise exception[0]
        
        return result[0]
    
    def _trigger_callbacks(self, event: str, *args):
        """触发回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(self, *args)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")
    
    def _save_execution_state(self):
        """保存执行状态"""
        state_file = self.workspace_dir / "pipeline_state.json"
        
        state = {
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'current_stage': self.current_stage,
            'completed_stages': self.completed_stages,
            'failed_stages': self.failed_stages,
            'stage_outputs': self.stage_outputs,
            'pipeline_artifacts': self.pipeline_artifacts,
            'execution_log': self.execution_log,
            'stages': [stage.to_dict() for stage in self.stages]
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    
    def load_execution_state(self) -> bool:
        """加载执行状态"""
        state_file = self.workspace_dir / "pipeline_state.json"
        
        if not state_file.exists():
            return False
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.status = PipelineStatus(state['status'])
            self.start_time = datetime.fromisoformat(state['start_time']) if state['start_time'] else None
            self.end_time = datetime.fromisoformat(state['end_time']) if state['end_time'] else None
            self.current_stage = state['current_stage']
            self.completed_stages = state['completed_stages']
            self.failed_stages = state['failed_stages']
            self.stage_outputs = state['stage_outputs']
            self.pipeline_artifacts = state['pipeline_artifacts']
            self.execution_log = state['execution_log']
            
            # 恢复阶段状态
            stage_states = {s['name']: s for s in state['stages']}
            for stage in self.stages:
                if stage.name in stage_states:
                    stage_state = stage_states[stage.name]
                    stage.status = PipelineStageStatus(stage_state['status'])
                    stage.start_time = datetime.fromisoformat(stage_state['start_time']) if stage_state['start_time'] else None
                    stage.end_time = datetime.fromisoformat(stage_state['end_time']) if stage_state['end_time'] else None
                    stage.error_message = stage_state['error_message']
                    stage.artifacts = stage_state['artifacts']
                    stage.metrics = stage_state['metrics']
                    stage.retry_count = stage_state['retry_count']
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载执行状态失败: {e}")
            return False
    
    def resume(self, from_stage: Optional[str] = None) -> Dict[str, Any]:
        """恢复执行"""
        if not self.load_execution_state():
            raise RuntimeError("无法加载执行状态")
        
        if self.status not in [PipelineStatus.FAILED, PipelineStatus.CANCELLED]:
            raise RuntimeError(f"管道状态 {self.status.value} 不支持恢复")
        
        # 重置失败的阶段
        for stage in self.stages:
            if stage.name in self.failed_stages:
                stage.reset()
        
        self.failed_stages = []
        
        # 确定恢复点
        if from_stage:
            resume_from = from_stage
        elif self.current_stage:
            resume_from = self.current_stage
        else:
            # 从第一个未完成的阶段开始
            execution_order = self.get_execution_order()
            resume_from = None
            for stage_name in execution_order:
                if stage_name not in self.completed_stages:
                    resume_from = stage_name
                    break
        
        if not resume_from:
            raise RuntimeError("没有找到合适的恢复点")
        
        self.logger.info(f"从阶段 '{resume_from}' 恢复执行")
        return self.execute(resume_from=resume_from)
    
    def cancel(self):
        """取消执行"""
        self.status = PipelineStatus.CANCELLED
        self.end_time = datetime.now()
        self._save_execution_state()
        self.logger.info(f"管道 {self.name} 已取消")
    
    def get_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        return {
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'current_stage': self.current_stage,
            'completed_stages': self.completed_stages,
            'failed_stages': self.failed_stages,
            'total_stages': len(self.stages),
            'progress': len(self.completed_stages) / len(self.stages) if self.stages else 0,
            'duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None
        }
    
    def get_stage_status(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """获取阶段状态"""
        stage = next((s for s in self.stages if s.name == stage_name), None)
        if stage:
            return stage.to_dict()
        return None
    
    def get_artifacts(self) -> Dict[str, Any]:
        """获取所有工件"""
        artifacts = {
            'pipeline_artifacts': self.pipeline_artifacts,
            'stage_artifacts': {}
        }
        
        for stage in self.stages:
            if stage.artifacts:
                artifacts['stage_artifacts'][stage.name] = stage.artifacts
        
        return artifacts
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        metrics = {
            'pipeline_metrics': {
                'total_duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
                'completed_stages': len(self.completed_stages),
                'failed_stages': len(self.failed_stages),
                'success_rate': len(self.completed_stages) / len(self.stages) if self.stages else 0
            },
            'stage_metrics': {}
        }
        
        for stage in self.stages:
            if stage.metrics:
                metrics['stage_metrics'][stage.name] = stage.metrics
        
        return metrics
    
    def cleanup(self, keep_artifacts: bool = True):
        """清理工作空间"""
        if not keep_artifacts:
            if self.workspace_dir.exists():
                shutil.rmtree(self.workspace_dir)
                self.logger.info(f"已清理工作空间: {self.workspace_dir}")
        else:
            # 只清理临时文件，保留工件和状态
            temp_dirs = ['temp', 'cache', 'logs']
            for temp_dir in temp_dirs:
                temp_path = self.workspace_dir / temp_dir
                if temp_path.exists():
                    shutil.rmtree(temp_path)
            
            self.logger.info("已清理临时文件")
    
    def export_report(self, output_path: Optional[str] = None) -> str:
        """导出执行报告"""
        if not output_path:
            output_path = self.workspace_dir / f"{self.name}_report.json"
        
        report = {
            'pipeline': self.get_status(),
            'stages': [stage.to_dict() for stage in self.stages],
            'artifacts': self.get_artifacts(),
            'metrics': self.get_metrics(),
            'execution_log': self.execution_log,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"执行报告已导出: {output_path}")
        return str(output_path)


# 预定义的管道阶段处理函数
def data_validation_stage(context: Dict[str, Any]) -> Dict[str, Any]:
    """数据验证阶段"""
    inputs = context['inputs']
    workspace = context['workspace']
    
    # 实现数据验证逻辑
    validation_results = {
        'data_valid': True,
        'sample_count': 1000,  # 示例值
        'feature_count': 224   # 示例值
    }
    
    return {
        'outputs': {
            'validation_results': validation_results
        },
        'metrics': {
            'validation_time': 5.2
        },
        'artifacts': {
            'validation_report': f"{workspace}/validation_report.json"
        }
    }


def data_preprocessing_stage(context: Dict[str, Any]) -> Dict[str, Any]:
    """数据预处理阶段"""
    inputs = context['inputs']
    workspace = context['workspace']
    
    # 实现数据预处理逻辑
    processed_data_path = f"{workspace}/processed_data"
    
    return {
        'outputs': {
            'processed_data_path': processed_data_path,
            'data_stats': {
                'train_samples': 800,
                'val_samples': 200
            }
        },
        'metrics': {
            'preprocessing_time': 15.7
        },
        'artifacts': {
            'processed_data': processed_data_path
        }
    }


def model_training_stage(context: Dict[str, Any]) -> Dict[str, Any]:
    """模型训练阶段"""
    inputs = context['inputs']
    workspace = context['workspace']
    config = context['global_config']
    
    # 实现模型训练逻辑
    model_path = f"{workspace}/trained_model.pth"
    
    return {
        'outputs': {
            'model_path': model_path,
            'training_history': {
                'epochs': config.get('epochs', 10),
                'best_accuracy': 0.95
            }
        },
        'metrics': {
            'training_time': 300.5,
            'best_val_accuracy': 0.95,
            'final_loss': 0.05
        },
        'artifacts': {
            'model_file': model_path,
            'training_curves': f"{workspace}/training_curves.png"
        }
    }


def model_evaluation_stage(context: Dict[str, Any]) -> Dict[str, Any]:
    """模型评估阶段"""
    inputs = context['inputs']
    workspace = context['workspace']
    
    # 实现模型评估逻辑
    evaluation_results = {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.96,
        'f1_score': 0.95
    }
    
    return {
        'outputs': {
            'evaluation_results': evaluation_results
        },
        'metrics': evaluation_results,
        'artifacts': {
            'confusion_matrix': f"{workspace}/confusion_matrix.png",
            'evaluation_report': f"{workspace}/evaluation_report.json"
        }
    }


def model_registration_stage(context: Dict[str, Any]) -> Dict[str, Any]:
    """模型注册阶段"""
    inputs = context['inputs']
    workspace = context['workspace']
    
    # 实现模型注册逻辑
    model_id = f"model_{int(time.time())}"
    
    return {
        'outputs': {
            'model_id': model_id,
            'registration_status': 'success'
        },
        'metrics': {
            'registration_time': 2.1
        },
        'artifacts': {
            'model_metadata': f"{workspace}/model_metadata.json"
        },
        'pipeline_artifacts': {
            'registered_model_id': model_id
        }
    }