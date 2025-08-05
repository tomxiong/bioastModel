"""
批量现代化ONNX转换脚本
使用最新PyTorch ONNX最佳实践批量转换所有模型
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import importlib

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.modern_onnx_converter_base import ModernONNXConverterBase, ModernConversionStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class BatchModernONNXConverter:
    """批量现代化ONNX转换器"""
    
    def __init__(self):
        """初始化批量转换器"""
        self.results = {}
        self.start_time = datetime.now()
        
        # 模型配置映射
        self.model_configs = {
            'airbubble_hybrid_net': {
                'input_shape': (3, 70, 70),
                'factory_function': 'create_airbubble_hybrid_net',
                'module': 'models.airbubble_hybrid_net',
                'class_name': 'AirBubbleHybridNet',
                'priority': 1,  # 最高优先级
                'requires_wrapper': True,  # 需要输出包装器
            },
            'resnet18_improved': {
                'input_shape': (3, 70, 70),
                'factory_function': 'create_resnet18_improved',
                'module': 'models.resnet_improved',
                'class_name': 'ResNet18Improved',
                'priority': 2,
            },
            'efficientnet_b0': {
                'input_shape': (3, 224, 224),
                'factory_function': 'create_efficientnet_b0',
                'module': 'models.efficientnet',
                'class_name': 'EfficientNetCustom',
                'priority': 3,
            },
            'mic_mobilenetv3': {
                'input_shape': (3, 70, 70),
                'factory_function': 'create_mic_mobilenetv3',
                'module': 'models.mic_mobilenetv3',
                'class_name': 'MIC_MobileNetV3',
                'priority': 4,
            },
            'micro_vit': {
                'input_shape': (3, 70, 70),
                'factory_function': 'create_micro_vit',
                'module': 'models.micro_vit',
                'class_name': 'MicroViT',
                'priority': 5,
                'requires_wrapper': True,  # Vision Transformer可能需要包装器
            },
            'simplified_airbubble_detector': {
                'input_shape': (3, 70, 70),
                'factory_function': 'create_simplified_airbubble_detector',
                'module': 'models.simplified_airbubble_detector',
                'class_name': 'SimplifiedAirBubbleDetector',
                'priority': 6,
            },
            'coatnet': {
                'input_shape': (3, 70, 70),
                'factory_function': 'create_coatnet',
                'module': 'models.coatnet',
                'class_name': 'CoAtNet',
                'priority': 7,
                'requires_wrapper': True,  # 混合架构可能需要包装器
            },
            'convnext_tiny': {
                'input_shape': (3, 70, 70),
                'factory_function': 'create_convnext_tiny',
                'module': 'models.convnext_tiny',
                'class_name': 'ConvNeXtTiny',
                'priority': 8,
            },
            'vit_tiny': {
                'input_shape': (3, 70, 70),
                'factory_function': 'create_vit_tiny',
                'module': 'models.vit_tiny',
                'class_name': 'ViTTiny',
                'priority': 9,
                'requires_wrapper': True,  # Vision Transformer需要包装器
            },
        }
    
    def create_generic_modern_converter(self, model_name: str, config: Dict) -> ModernONNXConverterBase:
        """创建通用现代化转换器"""
        
        class GenericModernConverter(ModernONNXConverterBase):
            def __init__(self, model_name: str, config: Dict):
                super().__init__(model_name)
                self.input_shape = config['input_shape']
                self.config = config
                
                # 根据模型类型定制转换策略
                if config.get('requires_wrapper', False):
                    # 对于复杂模型（Transformer、混合架构），使用更保守的策略
                    self.custom_strategies = [
                        ModernConversionStrategy(f"{model_name}_现代化静态", 18, False, True, True, False),
                        ModernConversionStrategy(f"{model_name}_现代化动态", 18, True, True, True, False),
                        ModernConversionStrategy(f"{model_name}_兼容静态", 17, False, True, True, False),
                        ModernConversionStrategy(f"{model_name}_兼容动态", 17, True, True, True, False),
                        ModernConversionStrategy(f"{model_name}_传统高版本", 16, False, False, True, False),
                        ModernConversionStrategy(f"{model_name}_传统中版本", 13, False, False, True, False),
                        ModernConversionStrategy(f"{model_name}_调试模式", 18, False, True, True, True),
                    ]
                else:
                    # 对于标准CNN模型，使用完整的现代化策略
                    self.custom_strategies = self.modern_strategies
            
            def create_model_instance(self):
                """创建模型实例"""
                try:
                    # 动态导入模块
                    module = importlib.import_module(self.config['module'])
                    
                    # 首先尝试工厂函数
                    if 'factory_function' in self.config:
                        factory_func = getattr(module, self.config['factory_function'])
                        model = factory_func()
                        self.log_message(f"使用工厂函数 {self.config['factory_function']} 创建模型成功")
                    else:
                        # 使用类构造函数
                        model_class = getattr(module, self.config['class_name'])
                        model = model_class()
                        self.log_message(f"使用类 {self.config['class_name']} 创建模型成功")
                    
                    model.eval()
                    return model
                except Exception as e:
                    self.log_message(f"创建模型实例失败: {e}", "ERROR")
                    return None
            
            def create_wrapper_if_needed(self, model):
                """如果需要，创建ONNX兼容包装器"""
                if not self.config.get('requires_wrapper', False):
                    return model
                
                class GenericONNXWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, x):
                        outputs = self.model(x)
                        
                        # 处理字典输出
                        if isinstance(outputs, dict):
                            # 优先返回分类结果
                            for key in ['classification', 'logits', 'output']:
                                if key in outputs:
                                    return outputs[key]
                            # 返回第一个张量
                            for value in outputs.values():
                                if hasattr(value, 'shape'):
                                    return value
                            return list(outputs.values())[0]
                        
                        # 处理元组/列表输出
                        if isinstance(outputs, (tuple, list)):
                            return outputs[0]
                        
                        return outputs
                
                wrapped_model = GenericONNXWrapper(model)
                wrapped_model.eval()
                self.log_message("创建通用ONNX包装器成功")
                return wrapped_model
            
            def convert(self) -> bool:
                """执行转换"""
                self.log_message(f"开始现代化转换 {self.model_name}...")
                
                # 1. 查找检查点
                checkpoint_path = self.find_latest_checkpoint()
                if checkpoint_path is None:
                    self.log_message("未找到检查点文件", "ERROR")
                    return False
                
                # 2. 创建模型
                model = self.create_model_instance()
                if model is None:
                    return False
                
                # 3. 加载权重
                loaded_model = self.load_model_safely(lambda: model, checkpoint_path)
                if loaded_model is None:
                    return False
                
                # 4. 创建包装器（如果需要）
                final_model = self.create_wrapper_if_needed(loaded_model)
                
                # 5. 转换
                success, conversion_info = self.convert_with_modern_fallback(
                    final_model, self.input_shape, self.custom_strategies
                )
                
                # 6. 保存报告
                self.save_modern_conversion_report(success, conversion_info)
                
                return success
        
        return GenericModernConverter(model_name, config)
    
    def convert_single_model(self, model_name: str) -> Dict[str, Any]:
        """转换单个模型"""
        if model_name not in self.model_configs:
            return {
                'success': False,
                'error': f'未知模型: {model_name}',
                'duration': 0
            }
        
        start_time = datetime.now()
        logging.info(f"开始现代化转换模型: {model_name}")
        
        try:
            config = self.model_configs[model_name]
            converter = self.create_generic_modern_converter(model_name, config)
            success = converter.convert()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': success,
                'duration': duration,
                'input_shape': config['input_shape'],
                'priority': config['priority']
            }
            
            if success:
                logging.info(f"✓ 模型 {model_name} 现代化转换成功 (耗时: {duration:.2f}秒)")
            else:
                logging.error(f"✗ 模型 {model_name} 现代化转换失败 (耗时: {duration:.2f}秒)")
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logging.error(f"转换模型 {model_name} 时发生异常: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    def convert_all_models(self, priority_order: bool = True) -> Dict[str, Any]:
        """批量转换所有模型
        
        Args:
            priority_order: 是否按优先级顺序转换
        """
        logging.info("开始批量现代化ONNX转换...")
        
        # 确定转换顺序
        if priority_order:
            models_to_convert = sorted(
                self.model_configs.keys(),
                key=lambda x: self.model_configs[x]['priority']
            )
            logging.info("按优先级顺序转换模型")
        else:
            models_to_convert = list(self.model_configs.keys())
            logging.info("按默认顺序转换模型")
        
        logging.info(f"计划转换 {len(models_to_convert)} 个模型: {', '.join(models_to_convert)}")
        
        # 逐个转换
        for model_name in models_to_convert:
            self.results[model_name] = self.convert_single_model(model_name)
        
        # 生成总结报告
        self.generate_batch_report()
        
        return self.results
    
    def generate_batch_report(self):
        """生成批量转换报告"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        successful = sum(1 for r in self.results.values() if r['success'])
        failed = len(self.results) - successful
        
        report = f"""# 批量现代化ONNX转换报告

## 总体结果
- 转换时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 总耗时: {total_time:.2f} 秒
- 模型总数: {len(self.results)}
- 成功转换: {successful}
- 转换失败: {failed}
- 成功率: {(successful/len(self.results)*100):.1f}%

## 详细结果

"""
        
        # 按成功/失败分组显示
        successful_models = []
        failed_models = []
        
        for model_name, result in self.results.items():
            if result['success']:
                successful_models.append((model_name, result))
            else:
                failed_models.append((model_name, result))
        
        if successful_models:
            report += "### ✅ 成功转换的模型\n\n"
            for model_name, result in successful_models:
                report += f"- **{model_name}** (耗时: {result['duration']:.2f}秒)\n"
                report += f"  - 输入形状: {result['input_shape']}\n"
                report += f"  - 优先级: {result['priority']}\n\n"
        
        if failed_models:
            report += "### ❌ 转换失败的模型\n\n"
            for model_name, result in failed_models:
                report += f"- **{model_name}** (耗时: {result['duration']:.2f}秒)\n"
                error = result.get('error', '未知错误')
                report += f"  - 错误: {error}\n\n"
        
        # 保存报告
        reports_dir = Path("reports/batch_modern_conversion")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f"batch_modern_conversion_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"批量转换报告已保存至: {report_path}")
        logging.info(f"批量现代化转换完成! 成功: {successful}/{len(self.results)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='批量现代化ONNX转换')
    parser.add_argument('--model', type=str, help='转换指定模型（不指定则转换所有模型）')
    parser.add_argument('--no-priority', action='store_true', help='不按优先级顺序转换')
    
    args = parser.parse_args()
    
    converter = BatchModernONNXConverter()
    
    if args.model:
        # 转换单个模型
        if args.model not in converter.model_configs:
            logging.error(f"未知模型: {args.model}")
            logging.info(f"可用模型: {', '.join(converter.model_configs.keys())}")
            sys.exit(1)
        
        result = converter.convert_single_model(args.model)
        if result['success']:
            logging.info(f"模型 {args.model} 现代化转换成功!")
        else:
            logging.error(f"模型 {args.model} 现代化转换失败!")
            sys.exit(1)
    else:
        # 批量转换所有模型
        results = converter.convert_all_models(priority_order=not args.no_priority)
        
        successful = sum(1 for r in results.values() if r['success'])
        total = len(results)
        
        if successful == total:
            logging.info("所有模型现代化转换成功!")
        else:
            logging.warning(f"部分模型转换失败: {successful}/{total} 成功")
            if successful == 0:
                sys.exit(1)

if __name__ == "__main__":
    main()