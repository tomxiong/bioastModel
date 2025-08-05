"""配置管理模块

提供配置文件的加载、保存和管理功能。
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging


@dataclass
class Config:
    """配置类"""
    
    # 基础配置
    base_dir: str = "."
    log_level: str = "INFO"
    
    # 模型注册配置
    registry_file: str = "registry/models.json"
    model_discovery_dirs: list = field(default_factory=lambda: ["models"])
    
    # 实验管理配置
    experiments_dir: str = "experiments"
    experiment_db_file: str = "experiments/experiments.db"
    
    # 工作流配置
    workflows_dir: str = "workflows"
    max_parallel_steps: int = 4
    
    # 任务调度配置
    scheduler_dir: str = "scheduler"
    max_workers: int = 4
    
    # 仪表板配置
    dashboard_port: int = 5000
    dashboard_host: str = "localhost"
    dashboard_debug: bool = False
    
    # 报告配置
    reports_dir: str = "reports"
    visualizations_dir: str = "visualizations"
    
    # 存储配置
    storage_backend: str = "local"  # local, s3, azure
    storage_config: Dict[str, Any] = field(default_factory=dict)
    
    # 安全配置
    enable_auth: bool = False
    auth_config: Dict[str, Any] = field(default_factory=dict)
    
    # 性能配置
    cache_enabled: bool = True
    cache_size: int = 1000
    
    # 通知配置
    notifications_enabled: bool = False
    notification_config: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        支持点号分隔的嵌套键，如 'dashboard.port'
        """
        keys = key.split('.')
        value = self
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        
        if len(keys) == 1:
            setattr(self, keys[0], value)
        else:
            # 处理嵌套键
            obj = self
            for k in keys[:-1]:
                if hasattr(obj, k):
                    obj = getattr(obj, k)
                else:
                    # 创建嵌套字典
                    new_dict = {}
                    setattr(obj, k, new_dict)
                    obj = new_dict
            
            if isinstance(obj, dict):
                obj[keys[-1]] = value
            else:
                setattr(obj, keys[-1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                result[key] = value
            else:
                result[key] = str(value)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def update(self, other: Union['Config', Dict[str, Any]]):
        """更新配置"""
        if isinstance(other, Config):
            other_dict = other.to_dict()
        else:
            other_dict = other
        
        for key, value in other_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """验证配置"""
        try:
            # 验证端口号
            if not (1 <= self.dashboard_port <= 65535):
                raise ValueError(f"无效的端口号: {self.dashboard_port}")
            
            # 验证工作线程数
            if self.max_workers <= 0:
                raise ValueError(f"无效的工作线程数: {self.max_workers}")
            
            # 验证日志级别
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if self.log_level not in valid_log_levels:
                raise ValueError(f"无效的日志级别: {self.log_level}")
            
            # 验证存储后端
            valid_backends = ['local', 's3', 'azure']
            if self.storage_backend not in valid_backends:
                raise ValueError(f"无效的存储后端: {self.storage_backend}")
            
            return True
            
        except Exception as e:
            logging.error(f"配置验证失败: {e}")
            return False


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.yaml"
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # 尝试加载配置文件
        self.load_config()
    
    def load_config(self, config_file: Optional[str] = None) -> Config:
        """加载配置文件"""
        if config_file:
            self.config_file = config_file
        
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            self.logger.info(f"配置文件不存在，使用默认配置: {self.config_file}")
            return self.config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            if data:
                self.config.update(data)
            
            # 验证配置
            if not self.config.validate():
                self.logger.warning("配置验证失败，使用默认配置")
                self.config = Config()
            
            self.logger.info(f"配置已加载: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            self.logger.info("使用默认配置")
        
        return self.config
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """保存配置文件"""
        if config_file:
            self.config_file = config_file
        
        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = self.config.to_dict()
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                elif config_path.suffix.lower() == '.json':
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            self.logger.info(f"配置已保存: {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            return False
    
    def get_config(self) -> Config:
        """获取当前配置"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            self.config.update(updates)
            
            if not self.config.validate():
                self.logger.error("配置更新后验证失败")
                return False
            
            self.logger.info("配置已更新")
            return True
            
        except Exception as e:
            self.logger.error(f"更新配置失败: {e}")
            return False
    
    def reset_config(self):
        """重置为默认配置"""
        self.config = Config()
        self.logger.info("配置已重置为默认值")
    
    def create_sample_config(self, output_file: str = "config.sample.yaml") -> bool:
        """创建示例配置文件"""
        sample_config = Config()
        
        # 添加注释的示例配置
        sample_data = {
            '# 基础配置': None,
            'base_dir': sample_config.base_dir,
            'log_level': sample_config.log_level,
            
            '# 模型注册配置': None,
            'registry_file': sample_config.registry_file,
            'model_discovery_dirs': sample_config.model_discovery_dirs,
            
            '# 实验管理配置': None,
            'experiments_dir': sample_config.experiments_dir,
            'experiment_db_file': sample_config.experiment_db_file,
            
            '# 工作流配置': None,
            'workflows_dir': sample_config.workflows_dir,
            'max_parallel_steps': sample_config.max_parallel_steps,
            
            '# 任务调度配置': None,
            'scheduler_dir': sample_config.scheduler_dir,
            'max_workers': sample_config.max_workers,
            
            '# 仪表板配置': None,
            'dashboard_port': sample_config.dashboard_port,
            'dashboard_host': sample_config.dashboard_host,
            'dashboard_debug': sample_config.dashboard_debug,
            
            '# 报告配置': None,
            'reports_dir': sample_config.reports_dir,
            'visualizations_dir': sample_config.visualizations_dir,
            
            '# 存储配置': None,
            'storage_backend': sample_config.storage_backend,
            'storage_config': {
                '# S3配置示例': None,
                'aws_access_key_id': 'your_access_key',
                'aws_secret_access_key': 'your_secret_key',
                'bucket_name': 'your_bucket',
                'region': 'us-east-1'
            },
            
            '# 安全配置': None,
            'enable_auth': sample_config.enable_auth,
            'auth_config': {
                'auth_type': 'basic',  # basic, oauth, jwt
                'username': 'admin',
                'password': 'password'
            },
            
            '# 性能配置': None,
            'cache_enabled': sample_config.cache_enabled,
            'cache_size': sample_config.cache_size,
            
            '# 通知配置': None,
            'notifications_enabled': sample_config.notifications_enabled,
            'notification_config': {
                'email': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': 'your_email@gmail.com',
                    'password': 'your_password',
                    'recipients': ['admin@example.com']
                },
                'slack': {
                    'webhook_url': 'https://hooks.slack.com/services/...',
                    'channel': '#ml-notifications'
                }
            }
        }
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # 手动写入带注释的YAML
                for key, value in sample_data.items():
                    if key.startswith('#'):
                        f.write(f"\n{key}\n")
                    elif value is not None:
                        if isinstance(value, dict):
                            f.write(f"{key}:\n")
                            self._write_dict_yaml(f, value, indent=2)
                        elif isinstance(value, list):
                            f.write(f"{key}:\n")
                            for item in value:
                                f.write(f"  - {item}\n")
                        else:
                            f.write(f"{key}: {value}\n")
            
            self.logger.info(f"示例配置文件已创建: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建示例配置文件失败: {e}")
            return False
    
    def _write_dict_yaml(self, f, data: dict, indent: int = 0):
        """写入字典到YAML文件"""
        for key, value in data.items():
            if key.startswith('#'):
                f.write(f"{' ' * indent}{key}\n")
            elif value is not None:
                if isinstance(value, dict):
                    f.write(f"{' ' * indent}{key}:\n")
                    self._write_dict_yaml(f, value, indent + 2)
                elif isinstance(value, list):
                    f.write(f"{' ' * indent}{key}:\n")
                    for item in value:
                        f.write(f"{' ' * (indent + 2)}- {item}\n")
                else:
                    f.write(f"{' ' * indent}{key}: {value}\n")
    
    def load_environment_config(self):
        """从环境变量加载配置"""
        env_mapping = {
            'BIOAST_BASE_DIR': 'base_dir',
            'BIOAST_LOG_LEVEL': 'log_level',
            'BIOAST_DASHBOARD_PORT': 'dashboard_port',
            'BIOAST_DASHBOARD_HOST': 'dashboard_host',
            'BIOAST_MAX_WORKERS': 'max_workers',
            'BIOAST_STORAGE_BACKEND': 'storage_backend',
            'BIOAST_ENABLE_AUTH': 'enable_auth',
            'BIOAST_CACHE_ENABLED': 'cache_enabled'
        }
        
        updates = {}
        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                # 类型转换
                if config_key in ['dashboard_port', 'max_workers', 'cache_size']:
                    try:
                        value = int(value)
                    except ValueError:
                        self.logger.warning(f"无效的环境变量值: {env_var}={value}")
                        continue
                elif config_key in ['enable_auth', 'dashboard_debug', 'cache_enabled', 'notifications_enabled']:
                    value = value.lower() in ['true', '1', 'yes', 'on']
                
                updates[config_key] = value
        
        if updates:
            self.update_config(updates)
            self.logger.info(f"已从环境变量加载 {len(updates)} 个配置项")
    
    def get_effective_config(self) -> Dict[str, Any]:
        """获取有效配置（包括环境变量覆盖）"""
        # 先加载文件配置
        self.load_config()
        
        # 再加载环境变量配置
        self.load_environment_config()
        
        return self.config.to_dict()