"""
现代化ONNX转换基础模块
基于PyTorch 2.5+ TorchDynamo导出器和最新最佳实践
"""

import os
import torch
import torch.onnx
import logging
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ModernConversionStrategy:
    """现代化转换策略类"""
    
    def __init__(self, name: str, opset_version: int = 18, dynamic_axes: bool = True, 
                 use_dynamo: bool = True, do_constant_folding: bool = True, 
                 verbose: bool = False):
        self.name = name
        self.opset_version = opset_version
        self.dynamic_axes = dynamic_axes
        self.use_dynamo = use_dynamo  # 新增：是否使用TorchDynamo导出器
        self.do_constant_folding = do_constant_folding
        self.verbose = verbose
    
    def __str__(self):
        dynamo_str = "Dynamo" if self.use_dynamo else "Legacy"
        return f"{self.name} ({dynamo_str}, opset={self.opset_version}, dynamic={self.dynamic_axes})"

class ModernONNXConverterBase:
    """现代化ONNX转换基础类"""
    
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
        
        # 现代化转换策略（按优先级排序）
        self.modern_strategies = [
            # 首选：TorchDynamo导出器 + 最新opset
            ModernConversionStrategy("现代化动态", 18, True, True, True),
            ModernConversionStrategy("现代化静态", 18, False, True, True),
            
            # 备选：稍低版本但仍使用Dynamo
            ModernConversionStrategy("现代化兼容动态", 17, True, True, True),
            ModernConversionStrategy("现代化兼容静态", 17, False, True, True),
            
            # 兼容：对于不支持Dynamo的情况，回退到传统方式但使用较高opset
            ModernConversionStrategy("传统高版本动态", 16, True, False, True),
            ModernConversionStrategy("传统高版本静态", 16, False, False, True),
            
            # 最后备用：传统方式 + 中等opset
            ModernConversionStrategy("传统中版本动态", 13, True, False, True),
            ModernConversionStrategy("传统中版本静态", 13, False, False, True),
            
            # 调试模式
            ModernConversionStrategy("调试模式", 18, False, True, True, True),
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
    
    def load_model_safely(self, model_class, checkpoint_path: Path) -> Optional[torch.nn.Module]:
        """安全加载模型和检查点
        
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
                model = model_class()
            else:
                model = model_class()
            
            model.eval()
            
            self.log_message("安全加载模型权重...")
            
            # 使用 weights_only=True 提高安全性
            try:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
                self.log_message("使用安全模式加载检查点")
            except Exception as e:
                self.log_message(f"安全模式加载失败，回退到传统模式: {e}", "WARNING")
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            
            # 处理不同的权重键名格式
            state_dict = self._extract_state_dict(checkpoint)
            processed_state_dict = self._process_state_dict(state_dict)
            
            # 加载状态字典
            model.load_state_dict(processed_state_dict)
            
            self.log_message("模型权重加载成功")
            return model
        except Exception as e:
            self.log_message(f"加载模型失败: {e}", "ERROR")
            import traceback
            self.log_message(f"详细错误信息: {traceback.format_exc()}", "ERROR")
            return None
    
    def _extract_state_dict(self, checkpoint: Dict) -> Dict:
        """从检查点中提取状态字典"""
        if 'model_state_dict' in checkpoint:
            self.log_message("使用 'model_state_dict' 键")
            return checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            self.log_message("使用 'state_dict' 键")
            return checkpoint['state_dict']
        else:
            self.log_message("直接使用检查点作为状态字典")
            return checkpoint
    
    def _process_state_dict(self, state_dict: Dict) -> Dict:
        """处理状态字典，移除可能的前缀"""
        processed_dict = {}
        prefixes_to_remove = ['base_model.', 'model.', 'module.']
        
        for key, value in state_dict.items():
            new_key = key
            for prefix in prefixes_to_remove:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    self.log_message(f"移除前缀 '{prefix}': {key} -> {new_key}")
                    break
            processed_dict[new_key] = value
        
        return processed_dict
    
    def convert_to_onnx_modern(self, model: torch.nn.Module, input_shape: Tuple, 
                              strategy: ModernConversionStrategy) -> bool:
        """使用现代化策略将模型转换为ONNX格式
        
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
            dynamic_axes_dict = None
            if strategy.dynamic_axes:
                dynamic_axes_dict = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            
            # 准备导出参数
            export_kwargs = {
                'export_params': True,
                'opset_version': strategy.opset_version,
                'do_constant_folding': strategy.do_constant_folding,
                'input_names': ['input'],
                'output_names': ['output'],
                'dynamic_axes': dynamic_axes_dict,
                'verbose': strategy.verbose
            }
            
            # 根据策略选择导出方式
            if strategy.use_dynamo:
                # 使用现代化TorchDynamo导出器
                self.log_message("使用 TorchDynamo 导出器")
                export_kwargs['dynamo'] = True
                
                # 检查PyTorch版本是否支持dynamo参数
                pytorch_version = torch.__version__
                self.log_message(f"PyTorch版本: {pytorch_version}")
                
                # 对于某些版本，可能需要使用torch.export
                try:
                    torch.onnx.export(model, dummy_input, self.onnx_path, **export_kwargs)
                except TypeError as e:
                    if "dynamo" in str(e):
                        self.log_message("当前PyTorch版本不支持dynamo参数，回退到传统方式", "WARNING")
                        export_kwargs.pop('dynamo')
                        torch.onnx.export(model, dummy_input, self.onnx_path, **export_kwargs)
                    else:
                        raise e
            else:
                # 使用传统导出器
                self.log_message("使用传统 TorchScript 导出器")
                torch.onnx.export(model, dummy_input, self.onnx_path, **export_kwargs)
            
            self.log_message(f"ONNX模型已保存至: {self.onnx_path}")
            return True
        except Exception as e:
            self.log_message(f"使用策略 {strategy} 导出ONNX模型失败: {e}", "ERROR")
            return False
    
    def validate_onnx_model_comprehensive(self, input_shape: Tuple) -> Tuple[bool, Dict]:
        """全面验证ONNX模型
        
        Args:
            input_shape: 输入形状，如(3, 224, 224)
            
        Returns:
            (是否成功, 验证信息)
        """
        validation_info = {}
        
        try:
            import onnx
            import onnxruntime as ort
            
            self.log_message("开始全面验证ONNX模型...")
            
            # 1. 加载并检查模型结构
            onnx_model = onnx.load(self.onnx_path)
            validation_info['model_loaded'] = True
            
            onnx.checker.check_model(onnx_model)
            self.log_message("ONNX模型结构检查通过")
            validation_info['model_valid'] = True
            
            # 2. 获取模型详细信息
            validation_info['opset_version'] = onnx_model.opset_import[0].version
            validation_info['model_size_mb'] = round(self.onnx_path.stat().st_size / (1024 * 1024), 2)
            validation_info['ir_version'] = onnx_model.ir_version
            
            self.log_message(f"模型信息 - Opset: {validation_info['opset_version']}, "
                           f"大小: {validation_info['model_size_mb']} MB, "
                           f"IR版本: {validation_info['ir_version']}")
            
            # 3. 创建推理会话并测试
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
            
            # 4. 执行推理测试
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            ort_inputs = {input_info.name: dummy_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            validation_info['inference_successful'] = True
            validation_info['output_shape_actual'] = ort_outputs[0].shape
            validation_info['output_range'] = [float(ort_outputs[0].min()), float(ort_outputs[0].max())]
            
            # 5. 检查输出合理性
            output_tensor = ort_outputs[0]
            if np.any(np.isnan(output_tensor)):
                self.log_message("警告：输出包含NaN值", "WARNING")
                validation_info['has_nan'] = True
            if np.any(np.isinf(output_tensor)):
                self.log_message("警告：输出包含无穷大值", "WARNING")
                validation_info['has_inf'] = True
            
            self.log_message("ONNX模型全面验证通过")
            self.log_message(f"输出形状: {output_tensor.shape}")
            self.log_message(f"输出范围: [{output_tensor.min():.6f}, {output_tensor.max():.6f}]")
            
            return True, validation_info
        except Exception as e:
            self.log_message(f"验证ONNX模型失败: {e}", "ERROR")
            validation_info['error'] = str(e)
            return False, validation_info
    
    def convert_with_modern_fallback(self, model: torch.nn.Module, input_shape: Tuple, 
                                   strategies: Optional[List[ModernConversionStrategy]] = None) -> Tuple[bool, Dict]:
        """使用现代化回退机制转换模型
        
        Args:
            model: 模型实例
            input_shape: 输入形状
            strategies: 转换策略列表，如果为None则使用默认策略
            
        Returns:
            (是否成功, 转换信息)
        """
        if strategies is None:
            strategies = self.modern_strategies
        
        conversion_info = {
            'strategies_tried': [],
            'successful_strategy': None,
            'validation_info': {},
            'pytorch_version': torch.__version__
        }
        
        for strategy in strategies:
            self.log_message(f"尝试转换策略: {strategy}")
            conversion_info['strategies_tried'].append(str(strategy))
            
            # 尝试转换
            success = self.convert_to_onnx_modern(model, input_shape, strategy)
            
            if success:
                # 全面验证转换结果
                valid, validation_info = self.validate_onnx_model_comprehensive(input_shape)
                conversion_info['validation_info'] = validation_info
                
                if valid:
                    self.log_message(f"转换成功，使用策略: {strategy}")
                    conversion_info['successful_strategy'] = str(strategy)
                    return True, conversion_info
                else:
                    self.log_message(f"转换成功但验证失败，尝试下一个策略", "WARNING")
            else:
                self.log_message(f"转换失败，尝试下一个策略", "WARNING")
        
        self.log_message("所有现代化转换策略都失败了", "ERROR")
        return False, conversion_info
    
    def generate_modern_conversion_report(self, success: bool, conversion_info: Dict) -> str:
        """生成现代化转换报告"""
        report = f"""# {self.model_name} 现代化ONNX转换报告

## 转换结果
- 模型名称: {self.model_name}
- 转换状态: {'成功' if success else '失败'}
- 转换时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- PyTorch版本: {conversion_info.get('pytorch_version', 'N/A')}

"""
        
        if success:
            strategy = conversion_info.get('successful_strategy', '未知')
            validation_info = conversion_info.get('validation_info', {})
            
            report += f"""## 成功信息
- 使用策略: {strategy}
- 模型大小: {validation_info.get('model_size_mb', 'N/A')} MB
- Opset版本: {validation_info.get('opset_version', 'N/A')}
- IR版本: {validation_info.get('ir_version', 'N/A')}
- 输入形状: {validation_info.get('input_shape', 'N/A')}
- 输出形状: {validation_info.get('output_shape_actual', 'N/A')}
- 输出范围: {validation_info.get('output_range', 'N/A')}

## 质量检查
- 包含NaN值: {'是' if validation_info.get('has_nan', False) else '否'}
- 包含无穷大值: {'是' if validation_info.get('has_inf', False) else '否'}

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
            for log_entry in self.conversion_log[-20:]:  # 只显示最后20条日志
                report += f"{log_entry}\n"
            report += "```\n"
        
        return report
    
    def save_modern_conversion_report(self, success: bool, conversion_info: Dict):
        """保存现代化转换报告"""
        report = self.generate_modern_conversion_report(success, conversion_info)
        
        # 创建报告目录
        reports_dir = Path("reports/modern_onnx_conversion")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f"{self.model_name}_modern_conversion_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.log_message(f"现代化转换报告已保存至: {report_path}")
    
    def convert(self) -> bool:
        """转换模型为ONNX格式
        
        此方法应由子类实现
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子类必须实现此方法")