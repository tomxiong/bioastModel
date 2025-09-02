"""
模型集成器

统一管理不同来源的模型，提供版本控制、元数据管理和模型注册功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type
from pathlib import Path
import json
import pickle
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import importlib
import sys
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """模型格式枚举"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    CUSTOM = "custom"


class ModelStatus(Enum):
    """模型状态枚举"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ModelCapabilities:
    """模型能力描述"""
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    supported_tasks: List[str] = field(default_factory=list)
    input_size: Optional[Tuple[int, ...]] = None
    output_size: Optional[Tuple[int, ...]] = None
    preprocessing_required: List[str] = field(default_factory=list)
    postprocessing_available: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCapabilities':
        """从字典创建"""
        return cls(**data)


@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    version: str
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    framework: str = "pytorch"
    model_format: ModelFormat = ModelFormat.PYTORCH
    file_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    model_size: Optional[int] = None  # 参数数量
    model_hash: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: ModelStatus = ModelStatus.DRAFT
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换datetime和enum
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['model_format'] = self.model_format.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建"""
        # 转换datetime和enum
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['model_format'] = ModelFormat(data['model_format'])
        data['status'] = ModelStatus(data['status'])
        return cls(**data)


@dataclass
class ModelVersion:
    """模型版本"""
    version_id: str
    metadata: ModelMetadata
    model_path: str
    config: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    changelog: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def load_model(self) -> nn.Module:
        """加载模型"""
        if self.metadata.model_format == ModelFormat.PYTORCH:
            return torch.load(self.model_path, map_location='cpu')
        elif self.metadata.model_format == ModelFormat.TORCHSCRIPT:
            return torch.jit.load(self.model_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported model format: {self.metadata.model_format}")


class ModelRegistry:
    """模型注册表"""
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        初始化模型注册表
        
        Args:
            registry_path: 注册表存储路径
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        
        self.models: Dict[str, ModelMetadata] = {}
        self.versions: Dict[str, ModelVersion] = {}
        self.active_versions: Dict[str, str] = {}  # model_name -> version_id
        
        self._load_registry()
    
    def _load_registry(self):
        """加载注册表"""
        # 加载模型元数据
        metadata_file = self.registry_path / "models.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.models = {name: ModelMetadata.from_dict(meta) 
                             for name, meta in data.items()}
        
        # 加载版本信息
        versions_file = self.registry_path / "versions.json"
        if versions_file.exists():
            with open(versions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.versions = {vid: self._dict_to_version(vdata) 
                               for vid, vdata in data.items()}
        
        # 加载活跃版本
        active_file = self.registry_path / "active_versions.json"
        if active_file.exists():
            with open(active_file, 'r', encoding='utf-8') as f:
                self.active_versions = json.load(f)
    
    def _dict_to_version(self, data: Dict[str, Any]) -> ModelVersion:
        """从字典创建ModelVersion"""
        data['metadata'] = ModelMetadata.from_dict(data['metadata'])
        return ModelVersion(**data)
    
    def _version_to_dict(self, version: ModelVersion) -> Dict[str, Any]:
        """将ModelVersion转换为字典"""
        data = asdict(version)
        data['metadata'] = version.metadata.to_dict()
        return data
    
    def register_model(self, metadata: ModelMetadata) -> str:
        """
        注册新模型
        
        Args:
            metadata: 模型元数据
            
        Returns:
            模型ID
        """
        model_id = f"{metadata.name}_{metadata.version}"
        
        if model_id in self.models:
            raise ValueError(f"Model {model_id} already exists")
        
        self.models[model_id] = metadata
        self._save_registry()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def add_version(self, model_id: str, version: ModelVersion) -> str:
        """
        添加模型版本
        
        Args:
            model_id: 模型ID
            version: 版本信息
            
        Returns:
            版本ID
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if version.version_id in self.versions:
            raise ValueError(f"Version {version.version_id} already exists")
        
        self.versions[version.version_id] = version
        self._save_registry()
        
        logger.info(f"Added version: {version.version_id} for model {model_id}")
        return version.version_id
    
    def set_active_version(self, model_id: str, version_id: str):
        """
        设置活跃版本
        
        Args:
            model_id: 模型ID
            version_id: 版本ID
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        self.active_versions[model_id] = version_id
        self._save_registry()
        
        logger.info(f"Set active version: {version_id} for model {model_id}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        return self.models.get(model_id)
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """获取版本信息"""
        return self.versions.get(version_id)
    
    def get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        """获取活跃版本"""
        version_id = self.active_versions.get(model_id)
        if version_id:
            return self.versions.get(version_id)
        return None
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[str]:
        """
        列出模型
        
        Args:
            status: 模型状态过滤
            
        Returns:
            模型ID列表
        """
        if status is None:
            return list(self.models.keys())
        
        return [mid for mid, meta in self.models.items() 
                if meta.status == status]
    
    def list_versions(self, model_id: str) -> List[str]:
        """
        列出模型的所有版本
        
        Args:
            model_id: 模型ID
            
        Returns:
            版本ID列表
        """
        return [vid for vid, ver in self.versions.items() 
                if ver.metadata.name == self.models[model_id].name]
    
    def _save_registry(self):
        """保存注册表"""
        # 保存模型元数据
        metadata_file = self.registry_path / "models.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({mid: meta.to_dict() for mid, meta in self.models.items()}, 
                     f, indent=2, ensure_ascii=False)
        
        # 保存版本信息
        versions_file = self.registry_path / "versions.json"
        with open(versions_file, 'w', encoding='utf-8') as f:
            json.dump({vid: self._version_to_dict(ver) 
                     for vid, ver in self.versions.items()}, 
                     f, indent=2, ensure_ascii=False)
        
        # 保存活跃版本
        active_file = self.registry_path / "active_versions.json"
        with open(active_file, 'w', encoding='utf-8') as f:
            json.dump(self.active_versions, f, indent=2, ensure_ascii=False)
    
    def remove_model(self, model_id: str):
        """移除模型"""
        if model_id in self.models:
            # 移除所有相关版本
            versions_to_remove = self.list_versions(model_id)
            for vid in versions_to_remove:
                del self.versions[vid]
            
            # 移除活跃版本
            if model_id in self.active_versions:
                del self.active_versions[model_id]
            
            # 移除模型
            del self.models[model_id]
            self._save_registry()
            
            logger.info(f"Removed model: {model_id}")
    
    def cleanup_unused_models(self):
        """清理未使用的模型文件"""
        used_paths = set()
        for version in self.versions.values():
            used_paths.add(version.model_path)
        
        # 扫描模型目录
        models_dir = self.registry_path / "models"
        if models_dir.exists():
            for model_file in models_dir.glob("**/*"):
                if model_file.is_file() and str(model_file) not in used_paths:
                    model_file.unlink()
                    logger.info(f"Removed unused model file: {model_file}")


class ModelIntegrator:
    """模型集成器主类"""
    
    def __init__(self, 
                 registry_path: str = "./model_registry",
                 models_dir: str = "./models"):
        """
        初始化模型集成器
        
        Args:
            registry_path: 注册表路径
            models_dir: 模型存储目录
        """
        self.registry = ModelRegistry(registry_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # 模型缓存
        self._model_cache: Dict[str, nn.Module] = {}
        
        # 支持的模型工厂
        self._model_factories: Dict[str, Callable] = {}
        
        logger.info("ModelIntegrator initialized")
    
    def register_model_factory(self, model_type: str, factory: Callable):
        """
        注册模型工厂函数
        
        Args:
            model_type: 模型类型
            factory: 工厂函数
        """
        self._model_factories[model_type] = factory
        logger.info(f"Registered model factory: {model_type}")
    
    def integrate_pytorch_model(self,
                               model: nn.Module,
                               name: str,
                               version: str,
                               description: str = "",
                               author: str = "",
                               tags: List[str] = None,
                               config: Dict[str, Any] = None) -> str:
        """
        集成PyTorch模型
        
        Args:
            model: PyTorch模型
            name: 模型名称
            version: 版本号
            description: 描述
            author: 作者
            tags: 标签列表
            config: 配置信息
            
        Returns:
            版本ID
        """
        if tags is None:
            tags = []
        if config is None:
            config = {}
        
        # 生成模型ID
        model_id = f"{name}_{version}"
        
        # 计算模型哈希
        model_hash = self._compute_model_hash(model)
        
        # 保存模型文件
        model_filename = f"{model_id}.pth"
        model_path = self.models_dir / model_filename
        torch.save(model, model_path)
        
        # 创建元数据
        metadata = ModelMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            tags=tags,
            model_format=ModelFormat.PYTORCH,
            file_path=str(model_path),
            model_size=sum(p.numel() for p in model.parameters()),
            model_hash=model_hash
        )
        
        # 注册模型
        self.registry.register_model(metadata)
        
        # 创建版本
        version_id = f"{model_id}_v1"
        model_version = ModelVersion(
            version_id=version_id,
            metadata=metadata,
            model_path=str(model_path),
            config=config
        )
        
        # 添加版本并设为活跃
        self.registry.add_version(model_id, model_version)
        self.registry.set_active_version(model_id, version_id)
        
        # 缓存模型
        self._model_cache[version_id] = model
        
        logger.info(f"Integrated PyTorch model: {model_id}")
        return version_id
    
    def integrate_onnx_model(self,
                            onnx_path: str,
                            name: str,
                            version: str,
                            description: str = "",
                            author: str = "",
                            tags: List[str] = None,
                            config: Dict[str, Any] = None) -> str:
        """
        集成ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            name: 模型名称
            version: 版本号
            description: 描述
            author: 作者
            tags: 标签列表
            config: 配置信息
            
        Returns:
            版本ID
        """
        if tags is None:
            tags = []
        if config is None:
            config = {}
        
        import onnx
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 生成模型ID
        model_id = f"{name}_{version}"
        
        # 计算文件哈希
        with open(onnx_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # 复制ONNX文件
        model_filename = f"{model_id}.onnx"
        dest_path = self.models_dir / model_filename
        import shutil
        shutil.copy2(onnx_path, dest_path)
        
        # 创建元数据
        metadata = ModelMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            tags=tags,
            model_format=ModelFormat.ONNX,
            file_path=str(dest_path),
            model_hash=file_hash
        )
        
        # 注册模型
        self.registry.register_model(metadata)
        
        # 创建版本
        version_id = f"{model_id}_v1"
        model_version = ModelVersion(
            version_id=version_id,
            metadata=metadata,
            model_path=str(dest_path),
            config=config
        )
        
        # 添加版本并设为活跃
        self.registry.add_version(model_id, model_version)
        self.registry.set_active_version(model_id, version_id)
        
        logger.info(f"Integrated ONNX model: {model_id}")
        return version_id
    
    def create_model_from_factory(self,
                                 factory_name: str,
                                 name: str,
                                 version: str,
                                 factory_args: Dict[str, Any] = None,
                                 **kwargs) -> str:
        """
        从工厂函数创建模型
        
        Args:
            factory_name: 工厂函数名
            name: 模型名称
            version: 版本号
            factory_args: 工厂函数参数
            **kwargs: 其他参数
            
        Returns:
            版本ID
        """
        if factory_name not in self._model_factories:
            raise ValueError(f"Model factory '{factory_name}' not registered")
        
        if factory_args is None:
            factory_args = {}
        
        # 创建模型
        factory = self._model_factories[factory_name]
        model = factory(**factory_args)
        
        # 集成模型
        return self.integrate_pytorch_model(
            model=model,
            name=name,
            version=version,
            **kwargs
        )
    
    def load_model(self, model_id: str, version_id: Optional[str] = None) -> nn.Module:
        """
        加载模型
        
        Args:
            model_id: 模型ID
            version_id: 版本ID（可选，默认使用活跃版本）
            
        Returns:
            加载的模型
        """
        if version_id is None:
            version = self.registry.get_active_version(model_id)
            if version is None:
                raise ValueError(f"No active version found for model {model_id}")
        else:
            version = self.registry.get_version(version_id)
            if version is None:
                raise ValueError(f"Version {version_id} not found")
        
        # 检查缓存
        if version.version_id in self._model_cache:
            return self._model_cache[version.version_id]
        
        # 加载模型
        model = version.load_model()
        
        # 缓存模型
        self._model_cache[version.version_id] = model
        
        return model
    
    def create_new_version(self,
                         model_id: str,
                         new_version: str,
                         model: Optional[nn.Module] = None,
                         changelog: str = "",
                         config: Dict[str, Any] = None) -> str:
        """
        创建新版本
        
        Args:
            model_id: 模型ID
            new_version: 新版本号
            model: 新模型（可选）
            changelog: 变更日志
            config: 配置信息
            
        Returns:
            新版本ID
        """
        metadata = self.registry.get_model(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")
        
        # 获取当前活跃版本作为父版本
        current_version = self.registry.get_active_version(model_id)
        parent_version_id = current_version.version_id if current_version else None
        
        if model is None:
            # 如果没有提供新模型，使用父版本的模型
            model = self.load_model(model_id)
        
        # 保存新模型
        new_model_id = f"{metadata.name}_{new_version}"
        model_filename = f"{new_model_id}.pth"
        model_path = self.models_dir / model_filename
        torch.save(model, model_path)
        
        # 计算模型哈希
        model_hash = self._compute_model_hash(model)
        
        # 更新元数据
        new_metadata = ModelMetadata(
            name=metadata.name,
            version=new_version,
            description=metadata.description,
            author=metadata.author,
            tags=metadata.tags.copy(),
            framework=metadata.framework,
            model_format=metadata.model_format,
            file_path=str(model_path),
            model_size=sum(p.numel() for p in model.parameters()),
            model_hash=model_hash,
            status=ModelStatus.ACTIVE
        )
        
        # 创建新版本
        version_id = f"{new_model_id}_v1"
        model_version = ModelVersion(
            version_id=version_id,
            metadata=new_metadata,
            model_path=str(model_path),
            parent_version=parent_version_id,
            changelog=changelog,
            config=config or {}
        )
        
        # 添加版本并设为活跃
        self.registry.add_version(new_model_id, model_version)
        self.registry.set_active_version(new_model_id, version_id)
        
        # 缓存模型
        self._model_cache[version_id] = model
        
        logger.info(f"Created new version: {version_id}")
        return version_id
    
    def compare_versions(self, model_id: str, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        比较两个版本
        
        Args:
            model_id: 模型ID
            version1_id: 版本1 ID
            version2_id: 版本2 ID
            
        Returns:
            比较结果
        """
        v1 = self.registry.get_version(version1_id)
        v2 = self.registry.get_version(version2_id)
        
        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'version1': {
                'id': v1.version_id,
                'created_at': v1.metadata.created_at.isoformat(),
                'model_size': v1.metadata.model_size,
                'metrics': v1.metrics
            },
            'version2': {
                'id': v2.version_id,
                'created_at': v2.metadata.created_at.isoformat(),
                'model_size': v2.metadata.model_size,
                'metrics': v2.metrics
            },
            'differences': {}
        }
        
        # 比较模型大小
        if v1.metadata.model_size and v2.metadata.model_size:
            size_diff = v2.metadata.model_size - v1.metadata.model_size
            comparison['differences']['model_size'] = {
                'absolute': size_diff,
                'relative': size_diff / v1.metadata.model_size if v1.metadata.model_size > 0 else 0
            }
        
        # 比较指标
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for metric in all_metrics:
            if metric in v1.metrics and metric in v2.metrics:
                diff = v2.metrics[metric] - v1.metrics[metric]
                comparison['differences'][metric] = {
                    'v1_value': v1.metrics[metric],
                    'v2_value': v2.metrics[metric],
                    'difference': diff
                }
        
        return comparison
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_id: 模型ID
            
        Returns:
            模型信息
        """
        metadata = self.registry.get_model(model_id)
        if metadata is None:
            raise ValueError(f"Model {model_id} not found")
        
        active_version = self.registry.get_active_version(model_id)
        all_versions = self.registry.list_versions(model_id)
        
        return {
            'metadata': metadata.to_dict(),
            'active_version': active_version.version_id if active_version else None,
            'total_versions': len(all_versions),
            'version_history': [self.registry.get_version(vid).metadata.to_dict() 
                               for vid in all_versions]
        }
    
    def _compute_model_hash(self, model: nn.Module) -> str:
        """计算模型哈希"""
        # 收集所有参数
        state_dict = model.state_dict()
        
        # 创建哈希
        hash_obj = hashlib.md5()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            hash_obj.update(tensor.cpu().numpy().tobytes())
        
        return hash_obj.hexdigest()
    
    def clear_cache(self):
        """清理缓存"""
        self._model_cache.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cached_models': len(self._model_cache),
            'cache_size_mb': sum(
                sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
                for model in self._model_cache.values()
            )
        }
    
    def export_registry(self, export_path: str):
        """导出注册表"""
        export_path = Path(export_path)
        export_path.mkdir(exist_ok=True)
        
        # 导出所有模型文件
        models_export_dir = export_path / "models"
        models_export_dir.mkdir(exist_ok=True)
        
        for version in self.registry.versions.values():
            if Path(version.model_path).exists():
                dest = models_export_dir / Path(version.model_path).name
                import shutil
                shutil.copy2(version.model_path, dest)
        
        # 导出注册表文件
        registry_export_dir = export_path / "registry"
        registry_export_dir.mkdir(exist_ok=True)
        
        import shutil
        shutil.copytree(self.registry.registry_path, registry_export_dir, dirs_exist_ok=True)
        
        logger.info(f"Registry exported to: {export_path}")
    
    def import_registry(self, import_path: str):
        """导入注册表"""
        import_path = Path(import_path)
        
        # 导入注册表文件
        registry_import_dir = import_path / "registry"
        if registry_import_dir.exists():
            import shutil
            shutil.copytree(registry_import_dir, self.registry.registry_path, dirs_exist_ok=True)
            
            # 重新加载注册表
            self.registry._load_registry()
        
        logger.info(f"Registry imported from: {import_path}")


def create_model_integrator(registry_path: str = "./model_registry",
                           models_dir: str = "./models") -> ModelIntegrator:
    """创建模型集成器实例"""
    return ModelIntegrator(registry_path, models_dir)