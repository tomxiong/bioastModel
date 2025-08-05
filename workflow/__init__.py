"""工作流自动化模块

提供模型生命周期的自动化管理功能。
"""

from .automation import WorkflowAutomation, WorkflowStep, WorkflowConfig
from .pipeline import ModelPipeline, PipelineStage
from .scheduler import TaskScheduler, ScheduledTask

# 全局实例
workflow_automation = WorkflowAutomation()
task_scheduler = TaskScheduler()

__all__ = [
    'WorkflowAutomation',
    'WorkflowStep', 
    'WorkflowConfig',
    'ModelPipeline',
    'PipelineStage',
    'TaskScheduler',
    'ScheduledTask',
    'workflow_automation',
    'task_scheduler'
]