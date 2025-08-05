"""辅助工具函数模块

提供各种实用的辅助函数。
"""

import os
import json
import uuid
import hashlib
import zipfile
import tarfile
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import re


def generate_id(prefix: str = "", length: int = 8) -> str:
    """生成唯一ID
    
    Args:
        prefix: ID前缀
        length: ID长度
    
    Returns:
        生成的ID字符串
    """
    unique_id = str(uuid.uuid4()).replace('-', '')[:length]
    return f"{prefix}{unique_id}" if prefix else unique_id


def generate_model_id() -> str:
    """生成模型ID"""
    return generate_id("model_", 12)


def generate_experiment_id() -> str:
    """生成实验ID"""
    return generate_id("exp_", 10)


def generate_workflow_id() -> str:
    """生成工作流ID"""
    return generate_id("wf_", 8)


def format_timestamp(timestamp: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化时间戳
    
    Args:
        timestamp: 时间戳，默认为当前时间
        format_str: 格式字符串
    
    Returns:
        格式化的时间字符串
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(format_str)


def format_duration(seconds: float) -> str:
    """格式化持续时间
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的持续时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_size(size_bytes: int) -> str:
    """格式化文件大小
    
    Args:
        size_bytes: 字节数
    
    Returns:
        格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"


def format_number(number: Union[int, float], precision: int = 2) -> str:
    """格式化数字
    
    Args:
        number: 数字
        precision: 精度
    
    Returns:
        格式化的数字字符串
    """
    if isinstance(number, int):
        return f"{number:,}"
    else:
        return f"{number:,.{precision}f}"


def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """安全加载JSON文件
    
    Args:
        file_path: 文件路径
        default: 默认值
    
    Returns:
        JSON数据或默认值
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception):
        return default


def safe_json_save(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """安全保存JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进
    
    Returns:
        是否保存成功
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        return True
    except Exception:
        return False


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """确保目录存在
    
    Args:
        dir_path: 目录路径
    
    Returns:
        目录路径对象
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def clean_filename(filename: str) -> str:
    """清理文件名
    
    Args:
        filename: 原始文件名
    
    Returns:
        清理后的文件名
    """
    # 移除或替换非法字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除多余的空格和点
    filename = re.sub(r'\s+', ' ', filename).strip()
    filename = re.sub(r'\.+', '.', filename)
    # 限制长度
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    return filename


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """计算文件哈希
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法
    
    Returns:
        文件哈希值
    """
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def get_directory_size(dir_path: Union[str, Path]) -> int:
    """获取目录大小
    
    Args:
        dir_path: 目录路径
    
    Returns:
        目录大小（字节）
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def copy_directory(src: Union[str, Path], dst: Union[str, Path], 
                  ignore_patterns: Optional[List[str]] = None) -> bool:
    """复制目录
    
    Args:
        src: 源目录
        dst: 目标目录
        ignore_patterns: 忽略的模式
    
    Returns:
        是否复制成功
    """
    try:
        src = Path(src)
        dst = Path(dst)
        
        if dst.exists():
            shutil.rmtree(dst)
        
        ignore_func = None
        if ignore_patterns:
            ignore_func = shutil.ignore_patterns(*ignore_patterns)
        
        shutil.copytree(src, dst, ignore=ignore_func)
        return True
    except Exception:
        return False


def compress_directory(dir_path: Union[str, Path], 
                      output_path: Union[str, Path],
                      format: str = 'zip') -> bool:
    """压缩目录
    
    Args:
        dir_path: 目录路径
        output_path: 输出路径
        format: 压缩格式 ('zip', 'tar', 'tar.gz')
    
    Returns:
        是否压缩成功
    """
    try:
        dir_path = Path(dir_path)
        output_path = Path(output_path)
        
        if format == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(dir_path)
                        zipf.write(file_path, arcname)
        
        elif format in ['tar', 'tar.gz']:
            mode = 'w:gz' if format == 'tar.gz' else 'w'
            with tarfile.open(output_path, mode) as tarf:
                tarf.add(dir_path, arcname=dir_path.name)
        
        else:
            raise ValueError(f"不支持的压缩格式: {format}")
        
        return True
    except Exception:
        return False


def extract_archive(archive_path: Union[str, Path], 
                   extract_path: Union[str, Path]) -> bool:
    """解压缩文件
    
    Args:
        archive_path: 压缩文件路径
        extract_path: 解压路径
    
    Returns:
        是否解压成功
    """
    try:
        archive_path = Path(archive_path)
        extract_path = Path(extract_path)
        
        extract_path.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_path)
        
        elif archive_path.suffix.lower() in ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:*') as tarf:
                tarf.extractall(extract_path)
        
        else:
            raise ValueError(f"不支持的压缩格式: {archive_path.suffix}")
        
        return True
    except Exception:
        return False


def create_temp_directory(prefix: str = "bioast_") -> Path:
    """创建临时目录
    
    Args:
        prefix: 目录前缀
    
    Returns:
        临时目录路径
    """
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup_temp_directory(temp_dir: Union[str, Path]) -> bool:
    """清理临时目录
    
    Args:
        temp_dir: 临时目录路径
    
    Returns:
        是否清理成功
    """
    try:
        shutil.rmtree(temp_dir)
        return True
    except Exception:
        return False


def validate_email(email: str) -> bool:
    """验证邮箱格式
    
    Args:
        email: 邮箱地址
    
    Returns:
        是否有效
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_url(url: str) -> bool:
    """验证URL格式
    
    Args:
        url: URL地址
    
    Returns:
        是否有效
    """
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    return re.match(pattern, url) is not None


def sanitize_string(text: str, max_length: int = 1000) -> str:
    """清理字符串
    
    Args:
        text: 原始字符串
        max_length: 最大长度
    
    Returns:
        清理后的字符串
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 移除控制字符
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # 限制长度
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], 
               deep: bool = True) -> Dict[str, Any]:
    """合并字典
    
    Args:
        dict1: 字典1
        dict2: 字典2
        deep: 是否深度合并
    
    Returns:
        合并后的字典
    """
    if not deep:
        result = dict1.copy()
        result.update(dict2)
        return result
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """展平字典
    
    Args:
        d: 原始字典
        parent_key: 父键
        sep: 分隔符
    
    Returns:
        展平后的字典
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """反展平字典
    
    Args:
        d: 展平的字典
        sep: 分隔符
    
    Returns:
        反展平后的字典
    """
    result = {}
    
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                      backoff: float = 2.0, exceptions: Tuple = (Exception,)):
    """重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟
        backoff: 退避因子
        exceptions: 需要重试的异常类型
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        raise e
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


def timing_decorator(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger = logging.getLogger(func.__module__)
            logger.debug(f"{func.__name__} 执行时间: {format_duration(duration)}")
    return wrapper


def memoize(maxsize: int = 128):
    """记忆化装饰器
    
    Args:
        maxsize: 最大缓存大小
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        cache = {}
        
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            # 限制缓存大小
            if len(cache) >= maxsize:
                # 移除最旧的条目
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = result
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'size': len(cache), 'maxsize': maxsize}
        
        return wrapper
    return decorator


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """解析版本号
    
    Args:
        version_str: 版本字符串 (如 "1.2.3")
    
    Returns:
        版本元组 (major, minor, patch)
    """
    try:
        parts = version_str.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except (ValueError, IndexError):
        return (0, 0, 0)


def compare_versions(version1: str, version2: str) -> int:
    """比较版本号
    
    Args:
        version1: 版本1
        version2: 版本2
    
    Returns:
        -1: version1 < version2
         0: version1 == version2
         1: version1 > version2
    """
    v1 = parse_version(version1)
    v2 = parse_version(version2)
    
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def get_system_info() -> Dict[str, Any]:
    """获取系统信息
    
    Returns:
        系统信息字典
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'disk_usage': dict(psutil.disk_usage('/'))._asdict() if os.name != 'nt' else dict(psutil.disk_usage('C:'))._asdict()
    }