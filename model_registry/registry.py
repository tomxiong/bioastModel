"""模型注册器

负责模型的注册、发现、查询和管理。
"""

import os
import json
import importlib.util
import inspect
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch.nn as nn

from .model_metadata import ModelMetadata
from .version_control import VersionControl


class ModelRegistry:
    """模型注册中心"""
    
    def __init__(self, registry_file: str = "data/model_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelMetadata] = {}
        self.version_control = VersionControl()
        self._load_registry()
    
    def _load_registry(self):
        """加载注册表"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        self.models[model_id] = ModelMetadata.from_dict(model_data)
            except Exception as e:
                print(f"加载注册表失败: {e}")
    
    def _save_registry(self):
        """保存注册表"""
        try:
            data = {model_id: metadata.to_dict() 
                   for model_id, metadata in self.models.items()}
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存注册表失败: {e}")
    
    def discover_models(self, models_dir: str = "models") -> List[str]:
        """自动发现模型
        
        Args:
            models_dir: 模型目录路径
            
        Returns:
            发现的模型文件列表
        """
        models_path = Path(models_dir)
        if not models_path.exists():
            return []
        
        discovered_models = []
        
        # 扫描Python文件
        for py_file in models_path.glob("**/*.py"):
            if py_file.name.startswith('__'):
                continue
                
            try:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(
                    py_file.stem, py_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 查找模型类
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, nn.Module) and 
                            obj != nn.Module):
                            discovered_models.append({
                                'file_path': str(py_file),
                                'class_name': name,
                                'module_name': py_file.stem
                            })
            except Exception as e:
                print(f"扫描文件 {py_file} 时出错: {e}")
        
        return discovered_models
    
    def register_model(self, 
                      model_name: str,
                      model_file: str,
                      class_name: str,
                      description: str = "",
                      author: str = "Unknown",
                      tags: List[str] = None,
                      **kwargs) -> str:
        """注册新模型
        
        Args:
            model_name: 模型名称
            model_file: 模型文件路径
            class_name: 模型类名
            description: 模型描述
            author: 作者
            tags: 标签列表
            **kwargs: 其他元数据
            
        Returns:
            模型ID
        """
        # 生成模型ID
        version = self.version_control.get_next_version(model_name)
        model_id = f"{model_name}_v{version}"
        
        # 创建模型元数据
        metadata = ModelMetadata(
            model_id=model_id,
            name=model_name,
            version=version,
            description=description,
            author=author,
            model_file=model_file,
            class_name=class_name,
            tags=tags or [],
            created_at=datetime.now().isoformat(),
            **kwargs
        )
        
        # 尝试获取模型架构信息
        try:
            arch_info = self._analyze_model_architecture(model_file, class_name)
            metadata.architecture.update(arch_info)
        except Exception as e:
            print(f"分析模型架构失败: {e}")
        
        # 注册模型
        self.models[model_id] = metadata
        self.version_control.add_version(model_name, version, metadata.to_dict())
        self._save_registry()
        
        return model_id
    
    def _analyze_model_architecture(self, model_file: str, class_name: str) -> Dict[str, Any]:
        """分析模型架构"""
        try:
            # 动态导入模型
            spec = importlib.util.spec_from_file_location("temp_module", model_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                model_class = getattr(module, class_name)
                
                # 创建模型实例（使用默认参数）
                try:
                    model = model_class()
                    
                    # 计算参数量
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    return {
                        "total_parameters": total_params,
                        "trainable_parameters": trainable_params,
                        "model_size_mb": total_params * 4 / (1024 * 1024),  # 假设float32
                        "architecture_type": self._infer_architecture_type(model)
                    }
                except Exception:
                    # 如果无法实例化，返回基本信息
                    return {
                        "architecture_type": "Unknown",
                        "note": "无法自动分析架构信息"
                    }
        except Exception as e:
            return {"error": str(e)}
    
    def _infer_architecture_type(self, model) -> str:
        """推断架构类型"""
        model_str = str(model).lower()
        
        if 'conv' in model_str and 'linear' in model_str:
            return "CNN"
        elif 'transformer' in model_str or 'attention' in model_str:
            return "Transformer"
        elif 'lstm' in model_str or 'gru' in model_str:
            return "RNN"
        elif 'linear' in model_str:
            return "MLP"
        else:
            return "Unknown"
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """获取模型信息"""
        return self.models.get(model_id)
    
    def list_models(self, 
                   status: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """列出模型
        
        Args:
            status: 过滤状态
            tags: 过滤标签
            
        Returns:
            模型列表
        """
        models = list(self.models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """搜索模型"""
        query = query.lower()
        results = []
        
        for model in self.models.values():
            if (query in model.name.lower() or 
                query in model.description.lower() or 
                any(query in tag.lower() for tag in model.tags)):
                results.append(model)
        
        return results
    
    def update_model_performance(self, 
                               model_id: str, 
                               performance: Dict[str, float]):
        """更新模型性能指标"""
        if model_id in self.models:
            self.models[model_id].performance.update(performance)
            self.models[model_id].updated_at = datetime.now().isoformat()
            self._save_registry()
    
    def deactivate_model(self, model_id: str):
        """停用模型"""
        if model_id in self.models:
            self.models[model_id].status = "inactive"
            self.models[model_id].updated_at = datetime.now().isoformat()
            self._save_registry()
    
    def get_model_versions(self, model_name: str) -> List[str]:
        """获取模型的所有版本"""
        return self.version_control.get_versions(model_name)
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """获取模型的最新版本"""
        versions = self.get_model_versions(model_name)
        if versions:
            return max(versions)
        return None
    
    def export_registry(self, output_file: str):
        """导出注册表"""
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_models": len(self.models),
            "models": {model_id: metadata.to_dict() 
                      for model_id, metadata in self.models.items()}
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        total_models = len(self.models)
        active_models = len([m for m in self.models.values() if m.status == "active"])
        
        # 按架构类型统计
        arch_stats = {}
        for model in self.models.values():
            arch_type = model.architecture.get("architecture_type", "Unknown")
            arch_stats[arch_type] = arch_stats.get(arch_type, 0) + 1
        
        # 按标签统计
        tag_stats = {}
        for model in self.models.values():
            for tag in model.tags:
                tag_stats[tag] = tag_stats.get(tag, 0) + 1
        
        return {
            "total_models": total_models,
            "active_models": active_models,
            "inactive_models": total_models - active_models,
            "architecture_distribution": arch_stats,
            "tag_distribution": tag_stats,
            "registry_file": str(self.registry_file),
            "last_updated": datetime.now().isoformat()
        }