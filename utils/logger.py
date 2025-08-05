"""日志管理模块

提供统一的日志配置和管理功能。
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class JSONFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record):
        # 添加颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 格式化消息
        formatted = super().format(record)
        
        # 只在终端输出时添加颜色
        if hasattr(os, 'isatty') and os.isatty(2):  # stderr
            return f"{color}{formatted}{reset}"
        else:
            return formatted


class ModelLifecycleLoggerAdapter(logging.LoggerAdapter):
    """模型生命周期日志适配器"""
    
    def process(self, msg, kwargs):
        # 添加上下文信息
        extra = kwargs.get('extra', {})
        
        if 'model_id' in self.extra:
            extra['model_id'] = self.extra['model_id']
        
        if 'experiment_id' in self.extra:
            extra['experiment_id'] = self.extra['experiment_id']
        
        if 'workflow_id' in self.extra:
            extra['workflow_id'] = self.extra['workflow_id']
        
        kwargs['extra'] = extra
        return msg, kwargs


def setup_logger(name: str = None,
                level: str = "INFO",
                log_dir: str = "logs",
                console_output: bool = True,
                file_output: bool = True,
                json_format: bool = False,
                colored_output: bool = True,
                max_file_size: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """设置日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        json_format: 是否使用JSON格式
        colored_output: 是否使用彩色输出
        max_file_size: 最大文件大小
        backup_count: 备份文件数量
    
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name or __name__)
    
    # 避免重复配置
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 日志格式
    if json_format:
        formatter = JSONFormatter()
    else:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
        if colored_output and console_output:
            formatter = ColoredFormatter(format_string)
        else:
            formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if colored_output and not json_format:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        else:
            console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    # 文件处理器
    if file_output:
        # 主日志文件
        log_file = log_path / f"{name or 'bioast'}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 错误日志文件
        error_log_file = log_path / f"{name or 'bioast'}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str = None, **kwargs) -> logging.Logger:
    """获取日志器
    
    如果日志器不存在，则创建一个新的。
    """
    logger = logging.getLogger(name or __name__)
    
    if not logger.handlers:
        return setup_logger(name, **kwargs)
    
    return logger


def get_model_logger(model_id: str, **kwargs) -> ModelLifecycleLoggerAdapter:
    """获取模型专用日志器"""
    logger = get_logger(f"model.{model_id}", **kwargs)
    return ModelLifecycleLoggerAdapter(logger, {'model_id': model_id})


def get_experiment_logger(experiment_id: str, **kwargs) -> ModelLifecycleLoggerAdapter:
    """获取实验专用日志器"""
    logger = get_logger(f"experiment.{experiment_id}", **kwargs)
    return ModelLifecycleLoggerAdapter(logger, {'experiment_id': experiment_id})


def get_workflow_logger(workflow_id: str, **kwargs) -> ModelLifecycleLoggerAdapter:
    """获取工作流专用日志器"""
    logger = get_logger(f"workflow.{workflow_id}", **kwargs)
    return ModelLifecycleLoggerAdapter(logger, {'workflow_id': workflow_id})


class LogManager:
    """日志管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loggers: Dict[str, logging.Logger] = {}
        
        # 默认配置
        self.default_config = {
            'level': 'INFO',
            'log_dir': 'logs',
            'console_output': True,
            'file_output': True,
            'json_format': False,
            'colored_output': True,
            'max_file_size': 10 * 1024 * 1024,
            'backup_count': 5
        }
        
        # 合并配置
        self.effective_config = {**self.default_config, **self.config}
    
    def get_logger(self, name: str) -> logging.Logger:
        """获取或创建日志器"""
        if name not in self.loggers:
            self.loggers[name] = setup_logger(name, **self.effective_config)
        
        return self.loggers[name]
    
    def set_level(self, level: str, logger_name: str = None):
        """设置日志级别"""
        log_level = getattr(logging, level.upper())
        
        if logger_name:
            if logger_name in self.loggers:
                self.loggers[logger_name].setLevel(log_level)
        else:
            # 设置所有日志器的级别
            for logger in self.loggers.values():
                logger.setLevel(log_level)
            
            # 更新默认配置
            self.effective_config['level'] = level
    
    def add_handler(self, handler: logging.Handler, logger_name: str = None):
        """添加处理器"""
        if logger_name:
            if logger_name in self.loggers:
                self.loggers[logger_name].addHandler(handler)
        else:
            # 添加到所有日志器
            for logger in self.loggers.values():
                logger.addHandler(handler)
    
    def remove_handler(self, handler: logging.Handler, logger_name: str = None):
        """移除处理器"""
        if logger_name:
            if logger_name in self.loggers:
                self.loggers[logger_name].removeHandler(handler)
        else:
            # 从所有日志器移除
            for logger in self.loggers.values():
                logger.removeHandler(handler)
    
    def cleanup_logs(self, days_to_keep: int = 30):
        """清理旧日志文件"""
        log_dir = Path(self.effective_config['log_dir'])
        
        if not log_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        cleaned_files = 0
        
        for log_file in log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_files += 1
                except OSError:
                    pass
        
        logger = self.get_logger('log_manager')
        logger.info(f"清理了 {cleaned_files} 个旧日志文件")
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        log_dir = Path(self.effective_config['log_dir'])
        
        if not log_dir.exists():
            return {'total_files': 0, 'total_size': 0}
        
        total_files = 0
        total_size = 0
        
        for log_file in log_dir.glob('*.log*'):
            if log_file.is_file():
                total_files += 1
                total_size += log_file.stat().st_size
        
        return {
            'total_files': total_files,
            'total_size': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'log_dir': str(log_dir),
            'active_loggers': len(self.loggers)
        }
    
    def export_logs(self, 
                   output_file: str,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   level: Optional[str] = None,
                   logger_name: Optional[str] = None):
        """导出日志"""
        # 这里可以实现日志导出功能
        # 由于日志文件可能很大，这里只是一个示例实现
        
        log_dir = Path(self.effective_config['log_dir'])
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for log_file in log_dir.glob('*.log'):
                if logger_name and logger_name not in log_file.name:
                    continue
                
                try:
                    with open(log_file, 'r', encoding='utf-8') as in_file:
                        for line in in_file:
                            # 这里可以添加时间和级别过滤逻辑
                            out_file.write(line)
                except Exception:
                    continue
        
        logger = self.get_logger('log_manager')
        logger.info(f"日志已导出到: {output_file}")


# 全局日志管理器实例
log_manager = LogManager()


# 便捷函数
def configure_logging(config: Dict[str, Any]):
    """配置全局日志"""
    global log_manager
    log_manager = LogManager(config)


def log_function_call(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"调用函数: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            raise
    
    return wrapper


def log_method_call(cls):
    """类方法调用日志装饰器"""
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_function_call(attr))
    
    return cls