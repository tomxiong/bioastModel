"""
ONNX 模型导出器

提供高性能的 ONNX 模型导出功能，支持各种优化选项
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from ..core.interfaces import ModelInterface
from ..core.model_adapters import ModelAdapter

logger = logging.getLogger(__name__)


class ONNXExporter:
    """ONNX 模型导出器"""
    
    def __init__(self):
        self.supported_optimizations = [
            'model_clean',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_conv_bias_bn',
            'gelu_approximation',
            'remove_unused_initializers',
            'eliminate_unused_initializer',
            'extract_constant_to_initializers',
            'fuse_layer_normalization',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_conv',
            'fuse_transpose_conv_gemm',
            'shape_inference'
        ]
        
        # 优化级别
        self.optimization_levels = {
            'basic': ['model_clean', 'shape_inference'],
            'intermediate': ['model_clean', 'fuse_bn_into_conv', 'fuse_conv_bias_bn', 
                           'eliminate_unused_initializer', 'shape_inference'],
            'advanced': ['model_clean', 'fuse_add_bias_into_conv', 'fuse_bn_into_conv',
                        'fuse_conv_bias_bn', 'fuse_layer_normalization', 
                        'fuse_matmul_add_bias_into_gemm', 'remove_unused_initializers',
                        'eliminate_unused_initializer', 'extract_constant_to_initializers',
                        'shape_inference']
        }
    
    def export_model(self, 
                    model: Union[ModelInterface, ModelAdapter, torch.nn.Module],
                    save_path: str,
                    input_shape: tuple = (1, 3, 70, 70),
                    optimizations: Optional[List[str]] = None,
                    optimization_level: str = 'basic',
                    quantization: Optional[str] = None) -> bool:
        """导出模型到 ONNX 格式
        
        Args:
            model: 要导出的模型
            save_path: 保存路径
            input_shape: 输入张量形状
            optimizations: 自定义优化列表
            optimization_level: 预设优化级别 (basic/intermediate/advanced)
            quantization: 量化选项 (None, 'fp16', 'int8')
            
        Returns:
            bool: 导出是否成功
        """
        start_time = datetime.now()
        export_metadata = {
            'model_name': getattr(model, 'model_name', 'unknown'),
            'input_shape': input_shape,
            'optimizations': optimizations or self.optimization_levels.get(optimization_level, []),
            'quantization': quantization,
            'export_time': start_time.isoformat()
        }
        
        try:
            logger.info(f"开始导出模型: {export_metadata['model_name']}")
            
            # 获取 PyTorch 模型
            pytorch_model = self._extract_pytorch_model(model)
            
            # 设置为评估模式
            pytorch_model.eval()
            
            # 创建示例输入
            dummy_input = torch.randn(*input_shape)
            
            # 导出 ONNX
            self._export_to_onnx(pytorch_model, dummy_input, save_path)
            
            # 应用优化
            opts = optimizations or self.optimization_levels.get(optimization_level, [])
            if opts:
                self._apply_optimizations(save_path, opts)
            
            # 应用量化
            if quantization:
                self._apply_quantization(save_path, quantization)
            
            # 验证导出的模型
            validation_results = self._validate_onnx_model(save_path, input_shape)
            
            # 添加元数据
            self._add_metadata(save_path, export_metadata)
            
            export_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"模型导出成功！耗时: {export_time:.2f}秒")
            logger.info(f"保存路径: {save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型导出失败: {e}")
            return False
    
    def _extract_pytorch_model(self, model: Union[ModelInterface, ModelAdapter, torch.nn.Module]) -> torch.nn.Module:
        """从各种模型类型中提取 PyTorch 模型"""
        if isinstance(model, ModelAdapter):
            # 从 ModelAdapter 获取模型
            if hasattr(model, 'model') and model.model is not None:
                return model.model
            else:
                # 如果模型尚未创建，先创建它
                model.model = model.model_factory(num_classes=2, **model.config)
                return model.model
        elif isinstance(model, ModelInterface):
            # 其他 ModelInterface 实现
            if hasattr(model, 'model'):
                return model.model
            else:
                raise ValueError("无法从 ModelInterface 提取 PyTorch 模型")
        elif isinstance(model, torch.nn.Module):
            # 直接是 PyTorch 模型
            return model
        else:
            raise ValueError(f"不支持的模型类型: {type(model)}")
    
    def _export_to_onnx(self, model: torch.nn.Module, dummy_input: torch.Tensor, save_path: str):
        """导出模型到 ONNX 格式"""
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=14,  # 使用较新的 opset 版本
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output', 'logits'] if hasattr(model, 'logits') else ['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            } if hasattr(model, 'logits') else {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        logger.info("   ✓ ONNX 导出完成")
    
    def _apply_optimizations(self, model_path: str, optimizations: List[str]):
        """应用 ONNX 优化"""
        try:
            # 使用 onnxoptimizer 进行优化
            import onnxoptimizer
            
            # 加载模型
            onnx_model = onnx.load(model_path)
            
            # 获取所有可用的优化
            all_opts = onnxoptimizer.get_fused_names()
            
            # 过滤出支持的优化
            valid_opts = [opt for opt in optimizations if opt in all_opts]
            
            if valid_opts:
                # 应用优化
                optimized_model = onnxoptimizer.optimize(onnx_model, valid_opts)
                
                # 保存优化后的模型
                onnx.save(optimized_model, model_path)
                
                logger.info(f"   ✓ 应用了 {len(valid_opts)} 个优化: {', '.join(valid_opts)}")
            else:
                logger.warning("   ⚠ 没有有效的优化可应用")
                
        except ImportError:
            logger.warning("   ⚠ onnxoptimizer 未安装，跳过优化")
        except Exception as e:
            logger.warning(f"   ⚠ 优化失败: {e}")
    
    def _apply_quantization(self, model_path: str, quantization: str):
        """应用模型量化"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            if quantization == 'fp16':
                # FP16 量化
                model_fp16_path = model_path.replace('.onnx', '_fp16.onnx')
                quantize_dynamic(
                    model_path,
                    model_fp16_path,
                    weight_type=QuantType.QFloat16  # 修正：使用 QFloat16 而不是 QUInt8
                )
                # 替换原文件
                Path(model_fp16_path).replace(model_path)
                logger.info("   ✓ FP16 量化完成")
                
            elif quantization == 'int8':
                # INT8 量化
                model_int8_path = model_path.replace('.onnx', '_int8.onnx')
                quantize_dynamic(
                    model_path,
                    model_int8_path,
                    weight_type=QuantType.QInt8
                )
                # 替换原文件
                Path(model_int8_path).replace(model_path)
                logger.info("   ✓ INT8 量化完成")
                
        except ImportError:
            logger.warning("   ⚠ ONNX Runtime 量化工具未安装")
        except Exception as e:
            logger.warning(f"   ⚠ 量化失败: {e}")
    
    def _validate_onnx_model(self, model_path: str, input_shape: tuple) -> Dict[str, Any]:
        """验证 ONNX 模型并返回性能指标"""
        validation_results = {}
        
        try:
            # 加载 ONNX 模型
            onnx_model = onnx.load(model_path)
            
            # 检查模型
            onnx.checker.check_model(onnx_model)
            validation_results['model_check'] = 'passed'
            
            # 创建推理会话
            ort_session = ort.InferenceSession(model_path)
            
            # 获取输入输出信息
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            
            validation_results['input_shape'] = input_info.shape
            validation_results['output_shape'] = output_info.shape
            validation_results['input_type'] = input_info.type
            validation_results['output_type'] = output_info.type
            
            # 测试推理性能
            warmup_runs = 10
            test_runs = 50
            
            # 预热
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            for _ in range(warmup_runs):
                ort_session.run(None, {'input': dummy_input})
            
            # 性能测试
            import time
            start_time = time.time()
            for _ in range(test_runs):
                outputs = ort_session.run(None, {'input': dummy_input})
            end_time = time.time()
            
            avg_inference_time = (end_time - start_time) / test_runs
            validation_results['avg_inference_time_ms'] = avg_inference_time * 1000
            validation_results['throughput'] = 1000 / avg_inference_time
            
            # 验证输出
            outputs = ort_session.run(None, {'input': dummy_input})
            validation_results['output_test'] = 'passed'
            validation_results['output_values_shape'] = outputs[0].shape
            
            logger.info(f"   ✓ ONNX 模型验证通过")
            logger.info(f"   ✓ 平均推理时间: {avg_inference_time*1000:.2f}ms")
            logger.info(f"   ✓ 吞吐量: {validation_results['throughput']:.2f} 推理/秒")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"   ✗ ONNX 模型验证失败: {e}")
            validation_results['error'] = str(e)
            return validation_results
    
    def _add_metadata(self, model_path: str, metadata: Dict[str, Any]):
        """添加元数据到 ONNX 模型"""
        try:
            onnx_model = onnx.load(model_path)
            
            # 创建元数据
            for key, value in metadata.items():
                meta = onnx.StringStringEntryProto()
                meta.key = key
                meta.value = str(value)
                onnx_model.metadata_props.append(meta)
            
            # 保存模型
            onnx.save(onnx_model, model_path)
            
        except Exception as e:
            logger.warning(f"   ⚠ 添加元数据失败: {e}")
    
    def batch_export(self,
                   models: List[Tuple[Union[ModelInterface, torch.nn.Module], str]],
                   output_dir: str,
                   **kwargs) -> Dict[str, bool]:
        """批量导出多个模型
        
        Args:
            models: 模型和文件名列表 [(model, filename), ...]
            output_dir: 输出目录
            **kwargs: export_model 的其他参数
            
        Returns:
            Dict: 导出结果 {filename: success}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for model, filename in models:
            save_path = output_path / filename
            success = self.export_model(model, str(save_path), **kwargs)
            results[filename] = success
            
            if success:
                logger.info(f"✓ {filename} 导出成功")
            else:
                logger.error(f"✗ {filename} 导出失败")
        
        return results
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """获取 ONNX 模型信息"""
        try:
            onnx_model = onnx.load(model_path)
            ort_session = ort.InferenceSession(model_path)
            
            info = {
                'file_size_mb': Path(model_path).stat().st_size / (1024 * 1024),
                'opset_version': onnx_model.opset_import[0].version if onnx_model.opset_import else None,
                'inputs': [{'name': inp.name, 'shape': inp.shape, 'type': inp.type} 
                          for inp in ort_session.get_inputs()],
                'outputs': [{'name': out.name, 'shape': out.shape, 'type': out.type} 
                           for out in ort_session.get_outputs()],
                'providers': ort_session.get_providers(),
                'metadata': {prop.key: prop.value for prop in onnx_model.metadata_props}
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {'error': str(e)}
    
    def compare_export_options(self, 
                              model: Union[ModelInterface, ModelAdapter, torch.nn.Module],
                              base_path: str,
                              input_shape: tuple = (1, 3, 70, 70)) -> Dict[str, Dict[str, Any]]:
        """比较不同导出选项的效果
        
        Args:
            model: 要导出的模型
            base_path: 基础路径（不带扩展名）
            input_shape: 输入形状
            
        Returns:
            Dict: 各种配置的结果比较
        """
        logger.info("开始比较不同导出选项...")
        
        results = {}
        
        # 测试所有优化级别和量化组合
        test_configs = [
            ('basic', None),
            ('intermediate', None),
            ('advanced', None),
            ('basic', 'fp16'),
            ('advanced', 'fp16'),
            ('basic', 'int8'),
            ('advanced', 'int8'),
        ]
        
        base_model_size = None
        
        for opt_level, quant in test_configs:
            config_name = f"{opt_level}"
            if quant:
                config_name += f"_{quant}"
            
            save_path = f"{base_path}_{config_name}.onnx"
            
            logger.info(f"测试配置: {config_name}")
            
            # 导出模型
            success = self.export_model(
                model,
                save_path,
                input_shape=input_shape,
                optimization_level=opt_level,
                quantization=quant
            )
            
            if success:
                # 获取模型信息
                info = self.get_model_info(save_path)
                
                # 计算压缩率
                if base_model_size is None:
                    # 找到基础模型（basic, 无量化）
                    basic_info = self.get_model_info(f"{base_path}_basic.onnx")
                    base_model_size = basic_info.get('file_size_mb', 0)
                
                compression_ratio = base_model_size / info.get('file_size_mb', 1) if info.get('file_size_mb', 0) > 0 else 1
                
                results[config_name] = {
                    'success': True,
                    'file_size_mb': info.get('file_size_mb', 0),
                    'compression_ratio': compression_ratio,
                    'size_reduction_percent': (1 - 1/compression_ratio) * 100 if compression_ratio > 1 else 0,
                    'avg_inference_time_ms': info.get('avg_inference_time_ms', 0),
                    'throughput': info.get('throughput', 0),
                    'optimization_level': opt_level,
                    'quantization': quant
                }
                
                logger.info(f"   ✓ 文件大小: {info.get('file_size_mb', 0):.2f} MB "
                           f"({(1 - 1/compression_ratio) * 100 if compression_ratio > 1 else 0:+.1f}%)")
            else:
                results[config_name] = {
                    'success': False,
                    'error': 'Export failed'
                }
                logger.error(f"   ✗ 导出失败")
        
        # 生成比较报告
        self._generate_comparison_report(results, f"{base_path}_comparison_report.json")
        
        return results
    
    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]], output_path: str):
        """生成比较报告"""
        import json
        
        # 计算统计信息
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if successful_results:
            # 找到最佳配置
            best_size = min(successful_results.items(), key=lambda x: x[1]['file_size_mb'])
            best_speed = max(successful_results.items(), key=lambda x: x[1]['throughput'])
            
            report = {
                'summary': {
                    'total_configs': len(results),
                    'successful_exports': len(successful_results),
                    'best_size_config': best_size[0],
                    'best_speed_config': best_speed[0],
                    'smallest_size_mb': best_size[1]['file_size_mb'],
                    'highest_throughput': best_speed[1]['throughput']
                },
                'detailed_results': results,
                'recommendations': self._generate_recommendations(successful_results)
            }
            
            # 保存报告
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"比较报告已保存到: {output_path}")
    
    def _generate_recommendations(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if not results:
            return ["没有成功的导出结果"]
        
        # 分析不同配置的优势
        size_configs = sorted(results.items(), key=lambda x: x[1]['file_size_mb'])
        speed_configs = sorted(results.items(), key=lambda x: x[1]['throughput'], reverse=True)
        
        # 文件大小建议
        if size_configs:
            smallest = size_configs[0]
            recommendations.append(
                f"最小文件大小: {smallest[0]} ({smallest[1]['file_size_mb']:.2f} MB, "
                f"压缩 {smallest[1]['size_reduction_percent']:+.1f}%)"
            )
        
        # 速度建议
        if speed_configs:
            fastest = speed_configs[0]
            recommendations.append(
                f"最高吞吐量: {fastest[0]} ({fastest[1]['throughput']:.1f} 推理/秒)"
            )
        
        # 平衡建议
        balanced_candidates = [r for r in results.values() 
                            if r['file_size_mb'] < np.mean([v['file_size_mb'] for v in results.values()]) 
                            and r['throughput'] > np.mean([v['throughput'] for v in results.values()])]
        
        if balanced_candidates:
            best_balanced = min(balanced_candidates, key=lambda x: x['file_size_mb'] / x['throughput'])
            recommendations.append("建议使用平衡配置以获得最佳性价比")
        
        return recommendations


# 工厂函数
def create_onnx_exporter() -> ONNXExporter:
    """创建 ONNX 导出器实例"""
    return ONNXExporter()


# 便捷函数
def export_model_to_onnx(model: Union[ModelInterface, torch.nn.Module],
                        save_path: str,
                        **kwargs) -> bool:
    """便捷的模型导出函数"""
    exporter = create_onnx_exporter()
    return exporter.export_model(model, save_path, **kwargs)
