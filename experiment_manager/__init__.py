"""实验管理模块

提供实验跟踪、管理和分析功能。
"""

from .experiment import Experiment
from .tracker import ExperimentTracker
from .database import ExperimentDatabase

__all__ = ['Experiment', 'ExperimentTracker', 'ExperimentDatabase']

# 全局实验跟踪器实例
tracker = ExperimentTracker()