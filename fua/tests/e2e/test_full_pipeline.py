"""
FUA 端到端测试

测试从模型创建到推理服务的完整流程
"""

import unittest
import tempfile
import os
import time
import json
import threading
import requests
import subprocess
from pathlib import Path
import asyncio

# Import FUA components
import fua


class TestFUAFullPipeline(unittest.TestCase):
    """FUA 完整流程测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = fua.ModelManager()
        self.server_process = None
        self.server_url = "http://localhost:8001"
        
    def tearDown(self):
        """清理测试环境"""
        # 停止服务器
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
        
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """测试完整的 FUA 流程"""
        print("\n测试 FUA 完整流程...")
        
        if not fua.DEPLOYMENT_AVAILABLE:
            self.skipTest("部署模块不可用")
        
        # 1. 创建模型
        print("\n1. 创建模型...")
        model_id = self.model_manager.create_model('mic_mobilenetv3', {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5
        })
        
        model = self.model_manager.get_model(model_id)
        self.assertIsNotNone(model)
        
        # 2. 训练模型
        print("\n2. 训练模型...")
        training_data = {'samples': 100, 'image_size': [70, 70]}
        training_results = self.model_manager.train_model(model_id, training_data)
        
        self.assertIsInstance(training_results, dict)
        self.assertIn('accuracy', training_results)
        print(f"   训练准确率: {training_results['accuracy']:.4f}")
        
        # 3. 评估模型
        print("\n3. 评估模型...")
        eval_results = self.model_manager.evaluate_model(model_id, {'samples': 50})
        
        self.assertIsInstance(eval_results, dict)
        self.assertIn('accuracy', eval_results)
        print(f"   评估准确率: {eval_results['accuracy']:.4f}")
        
        # 4. 导出模型
        print("\n4. 导出模型...")
        exporter = fua.create_onnx_exporter()
        model_path = os.path.join(self.temp_dir, 'pipeline_model.onnx')
        
        success = exporter.export_model(
            model,
            model_path,
            optimization_level='advanced',
            quantization='fp16'
        )
        
        self.assertTrue(success, "模型导出失败")
        
        # 验证导出的模型
        model_info = exporter.get_model_info(model_path)
        self.assertLess(model_info['file_size_mb'], 10, "模型文件过大")
        print(f"   模型大小: {model_info['file_size_mb']:.2f} MB")
        
        # 5. 启动推理服务器
        print("\n5. 启动推理服务器...")
        self._start_inference_server()
        
        # 等待服务器启动
        time.sleep(2)
        
        # 检查服务器健康状态
        health_response = requests.get(f"{self.server_url}/health")
        self.assertEqual(health_response.status_code, 200)
        print("   ✓ 服务器运行正常")
        
        # 6. 加载模型到服务器
        print("\n6. 加载模型到服务器...")
        load_response = requests.post(
            f"{self.server_url}/load_model",
            json={
                "model_name": "pipeline_model",
                "model_path": model_path
            }
        )
        
        self.assertEqual(load_response.status_code, 200)
        print("   ✓ 模型加载成功")
        
        # 7. 执行推理
        print("\n7. 执行推理...")
        
        # 准备测试数据（模拟 70x70x3 图像）
        test_input = [[0.5] * 70 * 70 * 3]
        
        # 单个推理
        predict_response = requests.post(
            f"{self.server_url}/predict",
            json={
                "model_name": "pipeline_model",
                "input_data": test_input,
                "threshold": 0.5
            }
        )
        
        self.assertEqual(predict_response.status_code, 200)
        result = predict_response.json()
        
        self.assertIn('predictions', result)
        self.assertIn('confidence', result)
        self.assertIn('processing_time', result)
        
        print(f"   ✓ 推理成功，置信度: {result['confidence']:.4f}")
        print(f"   ✓ 处理时间: {result['processing_time']*1000:.2f}ms")
        
        # 批量推理
        print("\n8. 批量推理...")
        batch_inputs = [test_input] * 5
        
        batch_response = requests.post(
            f"{self.server_url}/predict/batch",
            json={
                "model_name": "pipeline_model",
                "inputs": batch_inputs,
                "threshold": 0.5
            }
        )
        
        self.assertEqual(batch_response.status_code, 200)
        batch_result = batch_response.json()
        
        self.assertEqual(len(batch_result['results']), 5)
        self.assertGreater(batch_result['throughput'], 10)
        
        print(f"   ✓ 批量推理成功，吞吐量: {batch_result['throughput']:.1f}")
        
        # 9. 性能测试
        print("\n9. 性能测试...")
        
        # 执行多次推理
        num_requests = 20
        times = []
        
        for i in range(num_requests):
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/predict",
                json={
                    "model_name": "pipeline_model",
                    "input_data": test_input
                }
            )
            end_time = time.time()
            
            self.assertEqual(response.status_code, 200)
            times.append(end_time - start_time)
        
        # 计算性能指标
        avg_time = sum(times) / len(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]
        throughput = num_requests / sum(times)
        
        print(f"   平均延迟: {avg_time*1000:.2f}ms")
        print(f"   P95 延迟: {p95_time*1000:.2f}ms")
        print(f"   吞吐量: {throughput:.1f} 请求/秒")
        
        # 性能断言
        self.assertLess(avg_time, 0.1, "平均延迟过高")
        self.assertGreater(throughput, 10, "吞吐量过低")
        
        # 10. 获取服务器指标
        print("\n10. 检查服务器指标...")
        metrics_response = requests.get(f"{self.server_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        
        metrics = metrics_response.json()
        self.assertGreater(metrics['total_requests'], 0)
        self.assertGreater(metrics['throughput'], 0)
        
        print(f"   总请求数: {metrics['total_requests']}")
        print(f"   错误率: {metrics['error_rate']:.2%}")
        
        # 11. 模型管理
        print("\n11. 测试模型管理...")
        
        # 获取模型列表
        models_response = requests.get(f"{self.server_url}/models")
        self.assertEqual(models_response.status_code, 200)
        
        models = models_response.json()
        self.assertEqual(models['count'], 1)
        
        # 获取特定模型信息
        model_info_response = requests.get(f"{self.server_url}/models/pipeline_model")
        self.assertEqual(model_info_response.status_code, 200)
        
        model_detail = model_info_response.json()
        self.assertEqual(model_detail['name'], 'pipeline_model')
        
        print("   ✓ 模型管理功能正常")
        
        # 12. 模型预热
        print("\n12. 测试模型预热...")
        warmup_response = requests.post(
            f"{self.server_url}/models/pipeline_model/warmup",
            json={"iterations": 5}
        )
        
        self.assertEqual(warmup_response.status_code, 200)
        print("   ✓ 模型预热成功")
        
        print("\n" + "="*50)
        print("✅ 完整流程测试通过！")
        print("FUA 成功实现了从模型训练到生产推理的完整流程")
        print("="*50)
    
    def _start_inference_server(self):
        """启动推理服务器"""
        # 创建服务器脚本
        server_script = os.path.join(self.temp_dir, 'server.py')
        
        with open(server_script, 'w') as f:
            f.write(f'''
import sys
sys.path.append("{os.path.dirname(os.path.dirname(__file__))}")

from fua.deployment import create_inference_server

# 创建并运行服务器
server = create_inference_server(max_models=10)
server.run(host="0.0.0.0", port=8001, workers=1)
''')
        
        # 启动服务器进程
        self.server_process = subprocess.Popen(
            [sys.executable, server_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n测试错误处理...")
        
        if not fua.DEPLOYMENT_AVAILABLE:
            self.skipTest("部署模块不可用")
        
        # 启动服务器
        self._start_inference_server()
        time.sleep(2)
        
        # 测试不存在的模型
        response = requests.post(
            f"{self.server_url}/predict",
            json={
                "model_name": "nonexistent_model",
                "input_data": [[0.1] * 10]
            }
        )
        
        self.assertEqual(response.status_code, 404)
        
        # 测试无效输入
        response = requests.post(
            f"{self.server_url}/load_model",
            json={
                "model_name": "test",
                "model_path": "/nonexistent/path.onnx"
            }
        )
        
        self.assertEqual(response.status_code, 404)
        
        print("   ✓ 错误处理正常")


if __name__ == '__main__':
    # 运行端到端测试
    unittest.main(verbosity=2)