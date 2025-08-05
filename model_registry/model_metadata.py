"""模型元数据管理

定义模型元数据结构和管理方法。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class ModelMetadata:
    """模型元数据类"""
    
    # 基本信息
    model_id: str
    name: str
    version: str
    description: str = ""
    author: str = "Unknown"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 文件信息
    model_file: str = ""
    class_name: str = ""
    config_file: str = ""
    weights_file: str = ""
    
    # 架构信息
    architecture: Dict[str, Any] = field(default_factory=dict)
    
    # 性能指标
    performance: Dict[str, float] = field(default_factory=dict)
    
    # 训练信息
    training_info: Dict[str, Any] = field(default_factory=dict)
    
    # 标签和分类
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    
    # 状态信息
    status: str = "active"  # active, inactive, deprecated, experimental
    
    # 依赖信息
    dependencies: List[str] = field(default_factory=list)
    
    # 额外信息
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保架构信息包含基本字段
        if not self.architecture:
            self.architecture = {
                "architecture_type": "Unknown",
                "total_parameters": 0,
                "trainable_parameters": 0,
                "model_size_mb": 0.0
            }
        
        # 确保性能指标包含基本字段
        if not self.performance:
            self.performance = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建实例"""
        return cls(**data)
    
    def update_performance(self, metrics: Dict[str, float]):
        """更新性能指标"""
        self.performance.update(metrics)
        self.updated_at = datetime.now().isoformat()
    
    def add_tag(self, tag: str):
        """添加标签"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now().isoformat()
    
    def remove_tag(self, tag: str):
        """移除标签"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now().isoformat()
    
    def set_status(self, status: str):
        """设置状态"""
        valid_statuses = ["active", "inactive", "deprecated", "experimental"]
        if status in valid_statuses:
            self.status = status
            self.updated_at = datetime.now().isoformat()
        else:
            raise ValueError(f"无效状态: {status}. 有效状态: {valid_statuses}")
    
    def update_architecture_info(self, arch_info: Dict[str, Any]):
        """更新架构信息"""
        self.architecture.update(arch_info)
        self.updated_at = datetime.now().isoformat()
    
    def update_training_info(self, training_info: Dict[str, Any]):
        """更新训练信息"""
        self.training_info.update(training_info)
        self.updated_at = datetime.now().isoformat()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "status": self.status,
            "architecture_type": self.architecture.get("architecture_type", "Unknown"),
            "parameters": self.architecture.get("total_parameters", 0),
            "model_size_mb": self.architecture.get("model_size_mb", 0.0),
            "best_accuracy": max(self.performance.values()) if self.performance else 0.0,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def is_compatible_with(self, other: 'ModelMetadata') -> bool:
        """检查与另一个模型的兼容性"""
        # 简单的兼容性检查：相同的架构类型和类似的参数量
        if self.architecture.get("architecture_type") != other.architecture.get("architecture_type"):
            return False
        
        self_params = self.architecture.get("total_parameters", 0)
        other_params = other.architecture.get("total_parameters", 0)
        
        if self_params == 0 or other_params == 0:
            return True  # 无法比较时认为兼容
        
        # 参数量差异在50%以内认为兼容
        ratio = min(self_params, other_params) / max(self_params, other_params)
        return ratio >= 0.5
    
    def validate(self) -> List[str]:
        """验证元数据完整性"""
        errors = []
        
        # 检查必需字段
        if not self.model_id:
            errors.append("model_id不能为空")
        
        if not self.name:
            errors.append("name不能为空")
        
        if not self.version:
            errors.append("version不能为空")
        
        # 检查文件路径
        if self.model_file and not self.model_file.endswith('.py'):
            errors.append("model_file应该是Python文件")
        
        # 检查状态
        valid_statuses = ["active", "inactive", "deprecated", "experimental"]
        if self.status not in valid_statuses:
            errors.append(f"status应该是以下之一: {valid_statuses}")
        
        # 检查性能指标
        for metric, value in self.performance.items():
            if not isinstance(value, (int, float)):
                errors.append(f"性能指标 {metric} 应该是数值")
            elif value < 0 or value > 1:
                errors.append(f"性能指标 {metric} 应该在0-1之间")
        
        return errors
    
    def to_markdown(self) -> str:
        """生成Markdown格式的模型信息"""
        md = f"# {self.name} (v{self.version})\n\n"
        
        if self.description:
            md += f"**描述**: {self.description}\n\n"
        
        md += f"**作者**: {self.author}\n"
        md += f"**状态**: {self.status}\n"
        md += f"**创建时间**: {self.created_at}\n"
        md += f"**更新时间**: {self.updated_at}\n\n"
        
        # 架构信息
        if self.architecture:
            md += "## 架构信息\n\n"
            for key, value in self.architecture.items():
                md += f"- **{key}**: {value}\n"
            md += "\n"
        
        # 性能指标
        if self.performance:
            md += "## 性能指标\n\n"
            for metric, value in self.performance.items():
                if isinstance(value, float):
                    md += f"- **{metric}**: {value:.4f}\n"
                else:
                    md += f"- **{metric}**: {value}\n"
            md += "\n"
        
        # 标签
        if self.tags:
            md += "## 标签\n\n"
            md += ", ".join(f"`{tag}`" for tag in self.tags) + "\n\n"
        
        # 文件信息
        md += "## 文件信息\n\n"
        if self.model_file:
            md += f"- **模型文件**: `{self.model_file}`\n"
        if self.class_name:
            md += f"- **类名**: `{self.class_name}`\n"
        if self.config_file:
            md += f"- **配置文件**: `{self.config_file}`\n"
        if self.weights_file:
            md += f"- **权重文件**: `{self.weights_file}`\n"
        
        return md
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"ModelMetadata(id={self.model_id}, name={self.name}, version={self.version}, status={self.status})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()