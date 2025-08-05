"""模型注册中心模块

提供模型注册、发现、元数据管理和版本控制功能。
"""

from .registry import ModelRegistry
from .model_metadata import ModelMetadata
from .version_control import VersionControl

__all__ = ['ModelRegistry', 'ModelMetadata', 'VersionControl']

# 全局模型注册表实例
registry = ModelRegistry()