"""
FUA ONNX 导出性能测试

测试 ONNX 导出功能的性能和正确性
"""

import unittest
import tempfile
import os
import time
import numpy as np
from pathlib import Path

# Import FUA components
import fua


class TestONNXExportPerformance(unittest.TestCase):
    """ONNX 导出性能测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = fua.ModelManager()
        
        # 创建测试模型
        self.model_id = self.model_manager.create_model('mic_mobilenetv3', {
            'learning_rate': 0.001,
            'batch_size': 32
        })
        self.model = self.model_manager.get_model(self.model_id)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_basic_export_performance(self):
        """测试基本导出性能"""
        print("\n测试基本导出性能...")
        
        if not fua.DEPLOYMENT_AVAILABLE:
            self.skipTest("部署模块不可用")
        
        exporter = fua.create_onnx_exporter()
        
        # 测试多次导出
        export_times = []
        for i in range(5):
            path = os.path.join(self.temp_dir, f'basic_{i}.onnx')
            
            start_time = time.time()
            success = exporter.export_model(self.model, path)
            end_time = time.time()
            
            self.assertTrue(success, f"导出失败: {path}")
            export_time = end_time - start_time
            export_times.append(export_time)
        
        # 计算性能指标
        avg_export_time = np.mean(export_times)
        std_export_time = np.std(export_times)
        
        print(f"   平均导出时间: {avg_export_time:.3f}±{std_export_time:.3f}秒")
        print(f"   最快导出时间: {min(export_times):.3f}秒")
        
        # 性能断言
        self.assertLess(avg_export_time, 2.0, "平均导出时间过长")
        self.assertLess(std_export_time, 0.5, "导出时间波动过大")
    
    def test_optimization_levels_performance(self):
        """测试不同优化级别的性能"""
        print("\n测试优化级别性能...")
        
        if not fua.DEPLOYMENT_AVAILABLE:
            self.skipTest("部署模块不可用")
        
        exporter = fua.create_onnx_exporter()
        
        optimization_levels = ['basic', 'intermediate', 'advanced']
        results = {}
        
        for level in optimization_levels:
            path = os.path.join(self.temp_dir, f'opt_{level}.onnx')
            
            # 测量导出时间
            start_time = time.time()
            success = exporter.export_model(
                self.model,
                path,
                optimization_level=level
            )
            export_time = time.time() - start_time
            
            self.assertTrue(success, f"优化级别 {level} 导出失败")
            
            # 获取模型信息
            info = exporter.get_model_info(path)
            
            # 测量推理性能
            session = info.get('session')
            if session is None:
                import onnxruntime as ort
                session = ort.InferenceSession(path)
            
            # 预热
            dummy_input = np.random.randn(1, 3, 70, 70).astype(np.float32)
            for _ in range(10):
                session.run(None, {'input': dummy_input})
            
            # 性能测试
            inference_times = []
            for _ in range(50):
                start_time = time.time()
                session.run(None, {'input': dummy_input})
                inference_times.append(time.time() - start_time)
            
            avg_inference_time = np.mean(inference_times) * 1000  # ms
            
            results[level] = {
                'export_time': export_time,
                'file_size': info['file_size_mb'],
                'avg_inference_time': avg_inference_time,
                'throughput': 1000 / (avg_inference_time / 1000)
            }
        
        # 打印结果
        print(f"{'优化级别':<12} {'导出时间(s)':<12} {'文件大小(MB)':<12} {'推理时间(ms)':<15} {'吞吐量'}")
        print("-" * 70)
        
        for level, result in results.items():
            print(f"{level:<12} {result['export_time']:<12.3f} {result['file_size']:<12.2f} "
                  f"{result['avg_inference_time']:<15.2f} {result['throughput']:.1f}")
        
        # 验证优化效果
        self.assertLess(results['advanced']['file_size'], 
                       results['basic']['file_size'],
                       "高级优化应该减少文件大小")
    
    def test_quantization_performance(self):
        """测试量化性能"""
        print("\n测试量化性能...")
        
        if not fua.DEPLOYMENT_AVAILABLE:
            self.skipTest("部署模块不可用")
        
        exporter = fua.create_onnx_exporter()
        
        quantization_options = [None, 'fp16']  # int8 需要校准数据
        results = {}
        
        for quant in quantization_options:
            quant_name = quant if quant else 'none'
            path = os.path.join(self.temp_dir, f'quant_{quant_name}.onnx')
            
            start_time = time.time()
            success = exporter.export_model(
                self.model,
                path,
                optimization_level='basic',
                quantization=quant
            )
            export_time = time.time() - start_time
            
            self.assertTrue(success, f"量化 {quant} 导出失败")
            
            info = exporter.get_model_info(path)
            
            results[quant_name] = {
                'export_time': export_time,
                'file_size': info['file_size_mb']
            }
        
        # 打印结果
        print(f"{'量化选项':<12} {'导出时间(s)':<12} {'文件大小(MB)':<12}")
        print("-" * 40)
        
        for quant, result in results.items():
            print(f"{quant:<12} {result['export_time']:<12.3f} {result['file_size']:<12.2f}")
        
        # 验证量化效果
        if 'fp16' in results and 'none' in results:
            fp16_size = results['fp16']['file_size']
            none_size = results['none']['file_size']
            reduction = (none_size - fp16_size) / none_size * 100
            print(f"\nFP16 量化减少: {reduction:.1f}%")
            self.assertGreater(reduction, 5, "FP16 量化应该显著减少文件大小")
    
    def test_batch_export_performance(self):
        """测试批量导出性能"""
        print("\n测试批量导出性能...")
        
        if not fua.DEPLOYMENT_AVAILABLE:
            self.skipTest("部署模块不可用")
        
        exporter = fua.create_onnx_exporter()
        
        # 创建多个模型
        models_to_export = []
        model_types = ['mic_mobilenetv3', 'airbubble_hybrid_net']
        
        for model_type in model_types:
            try:
                mid = self.model_manager.create_model(model_type, {'epochs': 5})
                m = self.model_manager.get_model(mid)
                models_to_export.append((m, f'{model_type}.onnx'))
            except Exception as e:
                print(f"跳过 {model_type}: {e}")
        
        # 批量导出
        batch_dir = os.path.join(self.temp_dir, 'batch')
        start_time = time.time()
        results = exporter.batch_export(models_to_export, batch_dir)
        total_time = time.time() - start_time
        
        successful = sum(1 for r in results.values() if r)
        total_models = len(results)
        
        print(f"   批量导出 {successful}/{total_models} 个模型")
        print(f"   总耗时: {total_time:.3f}秒")
        print(f"   平均每模型: {total_time/total_models:.3f}秒")
        
        # 验证批量导出
        self.assertEqual(successful, total_models, "所有模型都应该导出成功")
        self.assertLess(total_time / total_models, 1.0, "平均每个模型导出时间过长")
    
    def test_model_info_extraction(self):
        """测试模型信息提取"""
        print("\n测试模型信息提取...")
        
        if not fua.DEPLOYMENT_AVAILABLE:
            self.skipTest("部署模块不可用")
        
        exporter = fua.create_onnx_exporter()
        path = os.path.join(self.temp_dir, 'info_test.onnx')
        
        # 导出模型
        success = exporter.export_model(self.model, path)
        self.assertTrue(success, "模型导出失败")
        
        # 提取信息
        info = exporter.get_model_info(path)
        
        # 验证信息完整性
        required_keys = ['file_size_mb', 'opset_version', 'inputs', 'outputs', 'providers']
        for key in required_keys:
            self.assertIn(key, info, f"缺少信息: {key}")
        
        # 打印信息
        print(f"   文件大小: {info['file_size_mb']:.2f} MB")
        print(f"   OPSET 版本: {info['opset_version']}")
        print(f"   输入: {info['inputs'][0]['shape']} ({info['inputs'][0]['type']})")
        print(f"   输出: {info['outputs'][0]['shape']} ({info['outputs'][0]['type']})")
        print(f"   提供商: {', '.join(info['providers'])}")
        
        if info['metadata']:
            print("   元数据:")
            for key, value in info['metadata'].items():
                print(f"     {key}: {value}")


if __name__ == '__main__':
    # 运行性能测试
    unittest.main(verbosity=2)