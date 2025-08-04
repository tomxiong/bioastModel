"""
增强型ONNX转换基础模块
提供多种转换策略、自动回退机制和详细的验证功能
"""

import os
import torch
import torch.onnx
import logging
from pathlib import Path
import json
import numpy as np
import importlib
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ConversionStrategy:
    """转换策略类"""
    
    def __init__(self, name: str, opset_version: int, dynamic_axes: bool = True, 
                 do_constant_folding: bool = True, verbose: bool = False):
        self.name = name
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes
        self.do_constant_folding = do_constant_folding
        self.verbose = verbose
    
    def __str__(self):
        return f"{self.name} (opset={self.opset_version}, dynamic={self.dynamic_axes})"

class EnhancedONNXConverterBase:
    """增强型ONNX转换基础类"""
    
    def __init__(self, model_name: str):
        """初始化转换器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.onnx_dir = Path("onnx_models")
        self.onnx_dir.mkdir(exist_ok=True)
        self.onnx_path = self.onnx_dir / f"{model_name}.onnx"
        
        # 转换日志
        self.conversion_log = []
        
        # 默认转换策略（按优先级排序）
        self.default_strategies = [
            ConversionStrategy("高版本动态", 13, True, True),
            ConversionStrategy("中版本动态", 12, True, True),
            ConversionStrategy("标准版本动态", 11, True, True),
            ConversionStrategy("高版本静态", 13, False, True),
            ConversionStrategy("中版本静态", 12, False, True),
            ConversionStrategy("标准版本静态", 11, False, True),
            ConversionStrategy("低版本静态", 10, False, True),
            ConversionStrategy("最低版本静态", 9, False, False),
        ]
    
    def log_message(self, message: str, level: str = "INFO"):
        """记录日志消息"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        self.conversion_log.append(log_entry)
        
        if level == "ERROR":
            logging.error(message)
        elif level == "WARNING":
            logging.warning(message)
        else:
            logging.info(message)
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """查找最新的模型检查点"""
        self.log_message(f"查找{self.model_name}的检查点文件...")
        
        experiments_dir = Path("experiments")
        model_dirs = list(experiments_dir.glob(f"**/{self.model_name}"))
        
        if not model_dirs:
            self.log_message(f"未找到{self.model_name}的实验目录", "WARNING")
            return None
        
        # 按修改时间排序，获取最新的实验目录
        latest_dir = max(model_dirs, key=os.path.getmtime)
        self.log_message(f"找到实验目录: {latest_dir}")
        
        # 查找最新的检查点文件
        checkpoint_files = list(latest_dir.glob("*.pth"))
        
        if not checkpoint_files:
            self.log_message(f"未找到{self.model_name}的检查点文件", "WARNING")
            return None
        
        # 按修改时间排序，获取最新的检查点文件
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        self.log_message(f"找到检查点文件: {latest_checkpoint}")
        return latest_checkpoint
    
    def load_model_config(self) -> Optional[Dict]:
        """加载模型配置"""
        self.log_message(f"加载{self.model_name}的配置...")
        
        experiments_dir = Path("experiments")
        model_dirs = list(experiments_dir.glob(f"**/{self.model_name}"))
        
        if not model_dirs:
            self.log_message(f"未找到{self.model_name}的实验目录", "WARNING")
            return None
        
        # 按修改时间排序，获取最新的实验目录
        latest_dir = max(model_dirs, key=os.path.getmtime)
        
        # 查找配置文件
        config_file = latest_dir / "config.json"
        
        if not config_file.exists():
            self.log_message(f"未找到{self.model_name}的配置文件", "WARNING")
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.log_message("配置文件加载成功")
            return config
        except Exception as e:
            self.log_message(f"加载{self.model_name}配置失败: {e}", "ERROR")
            return None
    
    def load_model_with_checkpoint(self, model_class, checkpoint_path: Path) -> Optional[torch.nn.Module]:
        """加载模型和检查点
        
        Args:
            model_class: 模型类或创建函数
            checkpoint_path: 检查点路径
            
        Returns:
            加载的模型
        """
        try:
            self.log_message("创建模型实例...")
            
            # 创建模型实例
            if callable(model_class):
                # 如果是函数，调用它
                model = model_class()
            else:
                # 如果是类，实例化它
                model = model_class()
            
            model.eval()
            
            self.log_message("加载模型权重...")
            
            # 加载模型权重
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            
            # 尝试多种权重键名
            state_dict = None
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.log_message("使用 'model_state_dict' 键")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                self.log_message("使用 'state_dict' 键")
            else:
                # 尝试直接加载
                state_dict = checkpoint
                self.log_message("直接使用检查点作为状态字典")
            
            # 处理可能的键名前缀问题
            processed_state_dict = self.process_state_dict(state_dict)
            
            # 加载状态字典
            model.load_state_dict(processed_state_dict)
            
            self.log_message("模型权重加载成功")
            return model
        except Exception as e:
            self.log_message(f"加载模型失败: {e}", "ERROR")
            import traceback
            self.log_message(f"详细错误信息: {traceback.format_exc()}", "ERROR")
            return None
    
    def process_state_dict(self, state_dict: Dict) -> Dict:
        """处理状态字典，移除可能的前缀"""
        processed_dict = {}
        
        # 常见的前缀列表
        prefixes_to_remove = ['base_model.', 'model.', 'module.']
        
        for key, value in state_dict.items():
            new_key = key
            
            # 尝试移除前缀
            for prefix in prefixes_to_remove:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    self.log_message(f"移除前缀 '{prefix}': {key} -> {new_key}")
                    break
            
            processed_dict[new_key] = value
        
        return processed_dict
    
    def convert_to_onnx_with_strategy(self, model: torch.nn.Module, input_shape: Tuple, 
                                    strategy: ConversionStrategy) -> bool:
        """使用指定策略将模型转换为ONNX格式
        
        Args:
            model: 模型实例
            input_shape: 输入形状，如(3, 224, 224)
            strategy: 转换策略
            
        Returns:
            是否成功
        """
        try:
            self.log_message(f"尝试使用策略: {strategy}")
            
            # 创建输入张量
            dummy_input = torch.randn(1, *input_shape)
            
            # 设置动态轴
            if strategy.dynamic_axes:
                dynamic_axes_dict = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            else:
                dynamic_axes_dict = None
            
            # 导出ONNX模型
            torch.onnx.export(
                model,
                dummy_input,
                self.onnx_path,
                export_params=True,
                opset_version=strategy.opset_version,
                do_constant_folding=strategy.do_constant_folding,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes_dict,
                verbose=strategy.verbose
            )
            
            self.log_message(f"ONNX模型已保存至: {self.onnx_path}")
            return True
        except Exception as e:
            self.log_message(f"使用策略 {strategy} 导出ONNX模型失败: {e}", "ERROR")
            return False
    
    def validate_onnx_model(self, input_shape: Tuple) -> Tuple[bool, Dict]:
        """验证ONNX模型
        
        Args:
            input_shape: 输入形状，如(3, 224, 224)
            
        Returns:
            (是否成功, 验证信息)
        """
        validation_info = {}
        
        try:
            import onnx
            import onnxruntime as ort
            
            self.log_message("开始验证ONNX模型...")
            
            # 加载ONNX模型
            onnx_model = onnx.load(self.onnx_path)
            validation_info['model_loaded'] = True
            
            # 检查模型
            onnx.checker.check_model(onnx_model)
            self.log_message("ONNX模型检查通过")
            validation_info['model_valid'] = True
            
            # 获取模型信息
            validation_info['opset_version'] = onnx_model.opset_import[0].version
            validation_info['model_size_mb'] = round(self.onnx_path.stat().st_size / (1024 * 1024), 2)
            
            # 创建推理会话
            ort_session = ort.InferenceSession(str(self.onnx_path))
            validation_info['session_created'] = True
            
            # 获取输入输出信息
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            
            validation_info['input_name'] = input_info.name
            validation_info['input_shape'] = input_info.shape
            validation_info['output_name'] = output_info.name
            validation_info['output_shape'] = output_info.shape
            
            self.log_message(f"输入信息: {input_info.name} {input_info.shape}")
            self.log_message(f"输出信息: {output_info.name} {output_info.shape}")
            
            # 创建输入张量
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            
            # 运行推理
            ort_inputs = {input_info.name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            validation_info['inference_successful'] = True
            validation_info['output_shape_actual'] = ort_outputs[0].shape
            validation_info['output_range'] = [float(ort_outputs[0].min()), float(ort_outputs[0].max())]
            
            self.log_message("ONNX模型推理测试通过")
            self.log_message(f"输出形状: {ort_outputs[0].shape}")
            self.log_message(f"输出范围: [{ort_outputs[0].min():.6f}, {ort_outputs[0].max():.6f}]")
            
            return True, validation_info
        except Exception as e:
            self.log_message(f"验证ONNX模型失败: {e}", "ERROR")
            validation_info['error'] = str(e)
            return False, validation_info
    
    def convert_with_fallback(self, model: torch.nn.Module, input_shape: Tuple, 
                            strategies: Optional[List[ConversionStrategy]] = None) -> Tuple[bool, Dict]:
        """使用回退机制转换模型
        
        Args:
            model: 模型实例
            input_shape: 输入形状
            strategies: 转换策略列表，如果为None则使用默认策略
            
        Returns:
            (是否成功, 转换信息)
        """
        if strategies is None:
            strategies = self.default_strategies
        
        conversion_info = {
            'strategies_tried': [],
            'successful_strategy': None,
            'validation_info': {}
        }
        
        for strategy in strategies:
            self.log_message(f"尝试转换策略: {strategy}")
            conversion_info['strategies_tried'].append(str(strategy))
            
            # 尝试转换
            success = self.convert_to_onnx_with_strategy(model, input_shape, strategy)
            
            if success:
                # 验证转换结果
                valid, validation_info = self.validate_onnx_model(input_shape)
                conversion_info['validation_info'] = validation_info
                
                if valid:
                    self.log_message(f"转换成功，使用策略: {strategy}")
                    conversion_info['successful_strategy'] = str(strategy)
                    return True, conversion_info
                else:
                    self.log_message(f"转换成功但验证失败，尝试下一个策略", "WARNING")
            else:
                self.log_message(f"转换失败，尝试下一个策略", "WARNING")
        
        self.log_message("所有转换策略都失败了", "ERROR")
        return False, conversion_info
    
    def generate_conversion_report(self, success: bool, conversion_info: Dict) -> str:
        """生成转换报告"""
        report = f"""# {self.model_name} ONNX转换报告

## 转换结果
- 模型名称: {self.model_name}
- 转换状态: {'成功' if success else '失败'}
- 转换时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        if success:
            strategy = conversion_info.get('successful_strategy', '未知')
            validation_info = conversion_info.get('validation_info', {})
            
            report += f"""## 成功信息
- 使用策略: {strategy}
- 模型大小: {validation_info.get('model_size_mb', 'N/A')} MB
- Opset版本: {validation_info.get('opset_version', 'N/A')}
- 输入形状: {validation_info.get('input_shape', 'N/A')}
- 输出形状: {validation_info.get('output_shape_actual', 'N/A')}
- 输出范围: {validation_info.get('output_range', 'N/A')}

"""
        
        # 添加尝试的策略
        strategies_tried = conversion_info.get('strategies_tried', [])
        if strategies_tried:
            report += f"""## 尝试的策略
"""
            for i, strategy in enumerate(strategies_tried, 1):
                status = "✓ 成功" if strategy == conversion_info.get('successful_strategy') else "✗ 失败"
                report += f"{i}. {strategy} - {status}\n"
            report += "\n"
        
        # 添加转换日志
        if self.conversion_log:
            report += f"""## 转换日志
```
"""
            for log_entry in self.conversion_log:
                report += f"{log_entry}\n"
            report += "```\n"
        
        return report
    
    def save_conversion_report(self, success: bool, conversion_info: Dict):
        """保存转换报告"""
        report = self.generate_conversion_report(success, conversion_info)
        
        # 创建报告目录
        reports_dir = Path("reports/onnx_conversion")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存报告
        report_path = reports_dir / f"{self.model_name}_conversion_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.log_message(f"转换报告已保存至: {report_path}")
    
    def convert(self) -> bool:
        """转换模型为ONNX格式
        
        此方法应由子类实现
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子类必须实现此方法")