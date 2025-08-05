"""验证器模块

提供模型和数据的验证功能。
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]
    
    def add_error(self, message: str):
        """添加错误"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """添加警告"""
        self.warnings.append(message)
    
    def add_detail(self, key: str, value: Any):
        """添加详细信息"""
        self.details[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'details': self.details
        }


class BaseValidator(ABC):
    """基础验证器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def validate(self, target: Any) -> ValidationResult:
        """验证目标对象"""
        pass
    
    def _create_result(self) -> ValidationResult:
        """创建验证结果"""
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            details={}
        )


class ModelValidator(BaseValidator):
    """模型验证器"""
    
    def __init__(self, 
                 required_files: Optional[List[str]] = None,
                 max_size_mb: float = 1000.0,
                 allowed_formats: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        
        self.required_files = required_files or ['model.pkl', 'config.json']
        self.max_size_mb = max_size_mb
        self.allowed_formats = allowed_formats or ['.pkl', '.joblib', '.h5', '.pt', '.pth', '.onnx']
    
    def validate(self, model_path: Union[str, Path]) -> ValidationResult:
        """验证模型"""
        result = self._create_result()
        model_path = Path(model_path)
        
        # 检查路径存在性
        if not model_path.exists():
            result.add_error(f"模型路径不存在: {model_path}")
            return result
        
        # 验证文件结构
        self._validate_file_structure(model_path, result)
        
        # 验证文件大小
        self._validate_file_size(model_path, result)
        
        # 验证文件格式
        self._validate_file_format(model_path, result)
        
        # 验证模型可加载性
        self._validate_model_loadable(model_path, result)
        
        # 验证配置文件
        self._validate_config_file(model_path, result)
        
        # 计算文件哈希
        self._calculate_file_hash(model_path, result)
        
        return result
    
    def _validate_file_structure(self, model_path: Path, result: ValidationResult):
        """验证文件结构"""
        if model_path.is_file():
            # 单文件模型
            result.add_detail('structure_type', 'single_file')
            result.add_detail('main_file', str(model_path))
        elif model_path.is_dir():
            # 目录结构模型
            result.add_detail('structure_type', 'directory')
            
            files = list(model_path.glob('*'))
            result.add_detail('total_files', len(files))
            result.add_detail('files', [str(f.name) for f in files])
            
            # 检查必需文件
            missing_files = []
            for required_file in self.required_files:
                if not (model_path / required_file).exists():
                    missing_files.append(required_file)
            
            if missing_files:
                result.add_warning(f"缺少推荐文件: {missing_files}")
        else:
            result.add_error(f"无效的模型路径类型: {model_path}")
    
    def _validate_file_size(self, model_path: Path, result: ValidationResult):
        """验证文件大小"""
        total_size = 0
        
        if model_path.is_file():
            total_size = model_path.stat().st_size
        elif model_path.is_dir():
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        size_mb = total_size / (1024 * 1024)
        result.add_detail('total_size_bytes', total_size)
        result.add_detail('total_size_mb', round(size_mb, 2))
        
        if size_mb > self.max_size_mb:
            result.add_warning(f"模型大小 ({size_mb:.2f}MB) 超过建议大小 ({self.max_size_mb}MB)")
    
    def _validate_file_format(self, model_path: Path, result: ValidationResult):
        """验证文件格式"""
        if model_path.is_file():
            suffix = model_path.suffix.lower()
            if suffix not in self.allowed_formats:
                result.add_error(f"不支持的文件格式: {suffix}")
            result.add_detail('file_format', suffix)
        elif model_path.is_dir():
            formats = set()
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    formats.add(file_path.suffix.lower())
            
            result.add_detail('file_formats', list(formats))
            
            # 检查是否有主模型文件
            main_model_files = []
            for file_path in model_path.glob('*'):
                if file_path.suffix.lower() in self.allowed_formats:
                    main_model_files.append(str(file_path.name))
            
            if not main_model_files:
                result.add_error("未找到主模型文件")
            else:
                result.add_detail('main_model_files', main_model_files)
    
    def _validate_model_loadable(self, model_path: Path, result: ValidationResult):
        """验证模型可加载性"""
        try:
            if model_path.is_file():
                self._try_load_single_file(model_path, result)
            elif model_path.is_dir():
                self._try_load_directory_model(model_path, result)
            
            result.add_detail('loadable', True)
            
        except Exception as e:
            result.add_error(f"模型加载失败: {str(e)}")
            result.add_detail('loadable', False)
            result.add_detail('load_error', str(e))
    
    def _try_load_single_file(self, model_path: Path, result: ValidationResult):
        """尝试加载单文件模型"""
        suffix = model_path.suffix.lower()
        
        if suffix == '.pkl':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            result.add_detail('model_type', type(model).__name__)
        
        elif suffix == '.joblib':
            import joblib
            model = joblib.load(model_path)
            result.add_detail('model_type', type(model).__name__)
        
        elif suffix in ['.h5', '.hdf5']:
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                result.add_detail('model_type', 'Keras')
                result.add_detail('model_summary', str(model.summary()))
            except ImportError:
                result.add_warning("TensorFlow未安装，无法验证Keras模型")
        
        elif suffix in ['.pt', '.pth']:
            try:
                import torch
                model = torch.load(model_path, map_location='cpu')
                result.add_detail('model_type', 'PyTorch')
                if hasattr(model, '__class__'):
                    result.add_detail('model_class', model.__class__.__name__)
            except ImportError:
                result.add_warning("PyTorch未安装，无法验证PyTorch模型")
        
        elif suffix == '.onnx':
            try:
                import onnx
                model = onnx.load(model_path)
                result.add_detail('model_type', 'ONNX')
                result.add_detail('onnx_version', model.ir_version)
            except ImportError:
                result.add_warning("ONNX未安装，无法验证ONNX模型")
    
    def _try_load_directory_model(self, model_path: Path, result: ValidationResult):
        """尝试加载目录结构模型"""
        # 查找主模型文件
        for file_path in model_path.glob('*'):
            if file_path.suffix.lower() in self.allowed_formats:
                try:
                    self._try_load_single_file(file_path, result)
                    break
                except Exception:
                    continue
    
    def _validate_config_file(self, model_path: Path, result: ValidationResult):
        """验证配置文件"""
        config_files = []
        
        if model_path.is_file():
            # 查找同名配置文件
            config_path = model_path.with_suffix('.json')
            if config_path.exists():
                config_files.append(config_path)
        elif model_path.is_dir():
            # 查找目录中的配置文件
            for config_name in ['config.json', 'model_config.json', 'settings.json']:
                config_path = model_path / config_name
                if config_path.exists():
                    config_files.append(config_path)
        
        result.add_detail('config_files', [str(f) for f in config_files])
        
        # 验证配置文件内容
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                result.add_detail(f'config_{config_file.name}', config)
                
                # 检查必要的配置项
                required_keys = ['model_type', 'version']
                missing_keys = [key for key in required_keys if key not in config]
                
                if missing_keys:
                    result.add_warning(f"配置文件 {config_file.name} 缺少推荐字段: {missing_keys}")
                
            except json.JSONDecodeError as e:
                result.add_error(f"配置文件 {config_file.name} JSON格式错误: {e}")
            except Exception as e:
                result.add_error(f"读取配置文件 {config_file.name} 失败: {e}")
    
    def _calculate_file_hash(self, model_path: Path, result: ValidationResult):
        """计算文件哈希"""
        try:
            if model_path.is_file():
                hash_value = self._get_file_hash(model_path)
                result.add_detail('file_hash', hash_value)
            elif model_path.is_dir():
                # 计算目录中所有文件的哈希
                file_hashes = {}
                for file_path in model_path.rglob('*'):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(model_path)
                        file_hashes[str(relative_path)] = self._get_file_hash(file_path)
                
                result.add_detail('file_hashes', file_hashes)
                
                # 计算整体哈希
                combined_hash = hashlib.md5()
                for path in sorted(file_hashes.keys()):
                    combined_hash.update(file_hashes[path].encode())
                
                result.add_detail('directory_hash', combined_hash.hexdigest())
        
        except Exception as e:
            result.add_warning(f"计算文件哈希失败: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """获取文件哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class DataValidator(BaseValidator):
    """数据验证器"""
    
    def __init__(self, 
                 min_samples: int = 10,
                 max_missing_ratio: float = 0.1,
                 required_columns: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        
        self.min_samples = min_samples
        self.max_missing_ratio = max_missing_ratio
        self.required_columns = required_columns or []
    
    def validate(self, data: Union[str, Path, pd.DataFrame, np.ndarray]) -> ValidationResult:
        """验证数据"""
        result = self._create_result()
        
        # 加载数据
        df = self._load_data(data, result)
        if df is None:
            return result
        
        # 基础统计
        self._validate_basic_stats(df, result)
        
        # 验证数据质量
        self._validate_data_quality(df, result)
        
        # 验证列结构
        self._validate_columns(df, result)
        
        # 验证数据类型
        self._validate_data_types(df, result)
        
        # 检测异常值
        self._detect_outliers(df, result)
        
        return result
    
    def _load_data(self, data: Union[str, Path, pd.DataFrame, np.ndarray], result: ValidationResult) -> Optional[pd.DataFrame]:
        """加载数据"""
        try:
            if isinstance(data, pd.DataFrame):
                return data
            
            elif isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            
            elif isinstance(data, (str, Path)):
                data_path = Path(data)
                
                if not data_path.exists():
                    result.add_error(f"数据文件不存在: {data_path}")
                    return None
                
                suffix = data_path.suffix.lower()
                
                if suffix == '.csv':
                    return pd.read_csv(data_path)
                elif suffix in ['.xlsx', '.xls']:
                    return pd.read_excel(data_path)
                elif suffix == '.json':
                    return pd.read_json(data_path)
                elif suffix == '.parquet':
                    return pd.read_parquet(data_path)
                else:
                    result.add_error(f"不支持的数据格式: {suffix}")
                    return None
            
            else:
                result.add_error(f"不支持的数据类型: {type(data)}")
                return None
        
        except Exception as e:
            result.add_error(f"加载数据失败: {e}")
            return None
    
    def _validate_basic_stats(self, df: pd.DataFrame, result: ValidationResult):
        """验证基础统计信息"""
        n_rows, n_cols = df.shape
        
        result.add_detail('n_rows', n_rows)
        result.add_detail('n_columns', n_cols)
        result.add_detail('memory_usage_mb', df.memory_usage(deep=True).sum() / (1024 * 1024))
        
        # 检查最小样本数
        if n_rows < self.min_samples:
            result.add_error(f"样本数量 ({n_rows}) 少于最小要求 ({self.min_samples})")
        
        # 检查空数据
        if n_rows == 0:
            result.add_error("数据为空")
        
        if n_cols == 0:
            result.add_error("没有列")
    
    def _validate_data_quality(self, df: pd.DataFrame, result: ValidationResult):
        """验证数据质量"""
        # 缺失值统计
        missing_stats = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        total_missing = missing_stats.sum()
        missing_ratio = total_missing / total_cells if total_cells > 0 else 0
        
        result.add_detail('missing_values_per_column', missing_stats.to_dict())
        result.add_detail('total_missing_values', int(total_missing))
        result.add_detail('missing_ratio', round(missing_ratio, 4))
        
        # 检查缺失值比例
        if missing_ratio > self.max_missing_ratio:
            result.add_warning(f"缺失值比例 ({missing_ratio:.2%}) 超过阈值 ({self.max_missing_ratio:.2%})")
        
        # 重复行统计
        n_duplicates = df.duplicated().sum()
        duplicate_ratio = n_duplicates / len(df) if len(df) > 0 else 0
        
        result.add_detail('duplicate_rows', int(n_duplicates))
        result.add_detail('duplicate_ratio', round(duplicate_ratio, 4))
        
        if duplicate_ratio > 0.1:  # 10%
            result.add_warning(f"重复行比例较高: {duplicate_ratio:.2%}")
    
    def _validate_columns(self, df: pd.DataFrame, result: ValidationResult):
        """验证列结构"""
        columns = df.columns.tolist()
        result.add_detail('columns', columns)
        
        # 检查必需列
        missing_columns = [col for col in self.required_columns if col not in columns]
        if missing_columns:
            result.add_error(f"缺少必需列: {missing_columns}")
        
        # 检查列名重复
        duplicate_columns = [col for col in columns if columns.count(col) > 1]
        if duplicate_columns:
            result.add_error(f"重复的列名: {set(duplicate_columns)}")
        
        # 检查空列名
        empty_columns = [i for i, col in enumerate(columns) if not str(col).strip()]
        if empty_columns:
            result.add_warning(f"空列名位置: {empty_columns}")
    
    def _validate_data_types(self, df: pd.DataFrame, result: ValidationResult):
        """验证数据类型"""
        dtypes = df.dtypes.to_dict()
        result.add_detail('data_types', {k: str(v) for k, v in dtypes.items()})
        
        # 统计数值列和分类列
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()
        
        result.add_detail('numeric_columns', numeric_columns)
        result.add_detail('categorical_columns', categorical_columns)
        result.add_detail('datetime_columns', datetime_columns)
        
        # 检查混合类型列
        for col in df.columns:
            if df[col].dtype == 'object':
                # 尝试转换为数值
                try:
                    pd.to_numeric(df[col], errors='raise')
                    result.add_warning(f"列 '{col}' 可能应该是数值类型")
                except (ValueError, TypeError):
                    pass
    
    def _detect_outliers(self, df: pd.DataFrame, result: ValidationResult):
        """检测异常值"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_stats = {}
        
        for col in numeric_columns:
            if df[col].notna().sum() == 0:
                continue
            
            # 使用IQR方法检测异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            outlier_stats[col] = {
                'count': len(outliers),
                'ratio': len(outliers) / len(df) if len(df) > 0 else 0,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        result.add_detail('outlier_stats', outlier_stats)
        
        # 警告异常值比例过高的列
        for col, stats in outlier_stats.items():
            if stats['ratio'] > 0.05:  # 5%
                result.add_warning(f"列 '{col}' 异常值比例较高: {stats['ratio']:.2%}")


class PipelineValidator(BaseValidator):
    """管道验证器"""
    
    def __init__(self, 
                 model_validator: Optional[ModelValidator] = None,
                 data_validator: Optional[DataValidator] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        
        self.model_validator = model_validator or ModelValidator()
        self.data_validator = data_validator or DataValidator()
    
    def validate(self, pipeline_config: Dict[str, Any]) -> ValidationResult:
        """验证完整管道"""
        result = self._create_result()
        
        # 验证配置结构
        self._validate_config_structure(pipeline_config, result)
        
        # 验证数据路径
        if 'data_path' in pipeline_config:
            data_result = self.data_validator.validate(pipeline_config['data_path'])
            if not data_result.is_valid:
                result.add_error("数据验证失败")
                result.details['data_validation'] = data_result.to_dict()
        
        # 验证模型路径
        if 'model_path' in pipeline_config:
            model_result = self.model_validator.validate(pipeline_config['model_path'])
            if not model_result.is_valid:
                result.add_error("模型验证失败")
                result.details['model_validation'] = model_result.to_dict()
        
        # 验证超参数
        self._validate_hyperparameters(pipeline_config.get('hyperparameters', {}), result)
        
        return result
    
    def _validate_config_structure(self, config: Dict[str, Any], result: ValidationResult):
        """验证配置结构"""
        required_keys = ['model_type', 'data_path']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            result.add_error(f"配置缺少必需字段: {missing_keys}")
        
        # 检查配置类型
        type_checks = {
            'model_type': str,
            'data_path': str,
            'hyperparameters': dict,
            'training_config': dict
        }
        
        for key, expected_type in type_checks.items():
            if key in config and not isinstance(config[key], expected_type):
                result.add_error(f"配置字段 '{key}' 类型错误，期望 {expected_type.__name__}")
    
    def _validate_hyperparameters(self, hyperparams: Dict[str, Any], result: ValidationResult):
        """验证超参数"""
        # 检查常见超参数的合理性
        validations = {
            'learning_rate': lambda x: 0 < x < 1,
            'batch_size': lambda x: isinstance(x, int) and x > 0,
            'epochs': lambda x: isinstance(x, int) and x > 0,
            'validation_split': lambda x: 0 < x < 1
        }
        
        for param, validator in validations.items():
            if param in hyperparams:
                try:
                    if not validator(hyperparams[param]):
                        result.add_warning(f"超参数 '{param}' 值可能不合理: {hyperparams[param]}")
                except Exception:
                    result.add_error(f"超参数 '{param}' 类型错误: {hyperparams[param]}")