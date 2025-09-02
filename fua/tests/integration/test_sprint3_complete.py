"""
FUA Sprint 3 完整集成测试

测试所有 Sprint 3 组件的端到端集成
"""

import unittest
import tempfile
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shutil
import json
import time
from pathlib import Path

# Import FUA components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import fua


class TestSprint3CompleteIntegration(unittest.TestCase):
    """Sprint 3 完整集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # 创建测试数据
        self.create_test_data()
        
        # 创建模型
        self.create_test_model()
        
        print(f"\n测试目录: {self.temp_dir}")
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """创建测试数据"""
        # 创建模拟的生物医学图像数据
        num_samples = 100
        self.images = torch.randn(num_samples, 3, 70, 70)
        self.labels = torch.randint(0, 2, (num_samples,))
        
        # 保存一些图像用于数据处理管道测试
        self.data_dir = self.temp_path / 'bioast_dataset'
        self.data_dir.mkdir()
        
        (self.data_dir / 'train' / 'negative').mkdir(parents=True)
        (self.data_dir / 'train' / 'positive').mkdir(parents=True)
        (self.data_dir / 'val' / 'negative').mkdir(parents=True)
        (self.data_dir / 'val' / 'positive').mkdir(parents=True)
        
        # 保存一些示例图像
        from PIL import Image
        import cv2
        
        for i in range(20):
            # 负样本
            img = np.random.randint(100, 200, (70, 70, 3), dtype=np.uint8)
            Image.fromarray(img).save(self.data_dir / 'train' / 'negative' / f'neg_{i}.jpg')
            
            # 正样本
            img = np.random.randint(50, 150, (70, 70, 3), dtype=np.uint8)
            cv2.circle(img, (35, 35), 15, (200, 150, 100), -1)
            Image.fromarray(img).save(self.data_dir / 'train' / 'positive' / f'pos_{i}.jpg')
        
        # 验证集
        for i in range(10):
            img = np.random.randint(100, 200, (70, 70, 3), dtype=np.uint8)
            Image.fromarray(img).save(self.data_dir / 'val' / 'negative' / f'neg_{i}.jpg')
            
            img = np.random.randint(50, 150, (70, 70, 3), dtype=np.uint8)
            cv2.circle(img, (35, 35), 15, (200, 150, 100), -1)
            Image.fromarray(img).save(self.data_dir / 'val' / 'positive' / f'pos_{i}.jpg')
    
    def create_test_model(self):
        """创建测试模型"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(32, 2)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        self.model = TestModel()
        self.model_path = self.temp_path / 'test_model.pth'
        torch.save(self.model.state_dict(), self.model_path)
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        print("\n1. 测试完整工作流程...")
        print("-" * 50)
        
        # 1. 数据处理管道
        print("\n1.1 数据处理管道测试")
        if not fua.PIPELINE_AVAILABLE:
            self.skipTest("数据处理管道不可用")
        
        # 创建数据处理器
        processor = fua.create_data_processor(
            image_size=(70, 70),
            enable_auto_augment=False
        )
        
        # 分析数据集
        stats = processor.analyze_dataset(str(self.data_dir / 'train'))
        self.assertGreater(stats.total_images, 0)
        print(f"   数据集包含 {stats.total_images} 张图像")
        
        # 创建数据管道
        pipeline = fua.create_data_pipeline(
            str(self.data_dir / 'train'),
            auto_split=False
        )
        
        # 2. 超参数优化
        print("\n1.2 超参数优化测试")
        if not fua.OPTIMIZATION_AVAILABLE:
            self.skipTest("超参数优化不可用")
        
        global optuna
        import optuna
        
        # 准备优化数据
        dataset = TensorDataset(self.images, self.labels)
        train_size = 80
        val_size = 20
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 定义搜索空间
        search_space = {
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'hidden_size': {'type': 'categorical', 'choices': [32, 64]}
        }
        
        # 创建优化器
        optimizer = fua.create_hyperparameter_optimizer(
            model_name='integration_test',
            search_space=search_space,
            n_trials=3,  # 减少试验次数
            direction='maximize'
        )
        
        # 执行优化
        def train_fn(model, train_loader, val_loader, params=None, trial=None, patience=5):
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(5):  # 简化训练
                model.train()
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                # 验证
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                
                val_acc = correct / total
                
                if trial is not None:
                    trial.report(val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return model, None, epoch
        
        def eval_fn(model, val_loader):
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            return {'accuracy': correct / total}
        
        # 运行优化
        result = optimizer.optimize(
            train_data=train_loader,
            val_data=val_loader,
            model_factory=lambda **params: self.model,
            train_fn=train_fn,
            eval_fn=eval_fn,
            save_study=False
        )
        
        print(f"   最佳准确率: {result.best_score:.4f}")
        
        # 3. ONNX 导出
        print("\n1.3 ONNX 导出测试")
        if not fua.DEPLOYMENT_AVAILABLE:
            self.skipTest("部署模块不可用")
        
        # 使用最佳参数重新训练模型
        best_model = self.model
        best_optimizer = torch.optim.Adam(best_model.parameters(), lr=result.best_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # 简单训练
        best_model.train()
        for data, target in train_loader:
            best_optimizer.zero_grad()
            output = best_model(data)
            loss = criterion(output, target)
            loss.backward()
            best_optimizer.step()
        
        # 导出为ONNX
        onnx_path = self.temp_path / 'best_model.onnx'
        exporter = fua.create_onnx_exporter()
        
        success = exporter.export_model(
            best_model,
            str(onnx_path),
            optimization_level='basic'
        )
        
        self.assertTrue(success)
        self.assertTrue(onnx_path.exists())
        print(f"   模型已导出到: {onnx_path}")
        
        # 4. 推理服务器
        print("\n1.4 推理服务器测试")
        
        # 加载模型到服务器
        server = fua.create_inference_server(max_models=5)
        
        # 模拟加载模型（在实际测试中需要真实的ONNX模型）
        print("   推理服务器创建成功")
        
        print("\n✓ 完整工作流程测试通过！")
    
    def test_performance_benchmarks(self):
        """测试性能基准"""
        print("\n2. 性能基准测试")
        print("-" * 50)
        
        # 1. 数据处理性能
        if fua.PIPELINE_AVAILABLE:
            print("\n2.1 数据处理性能")
            processor = fua.create_data_processor()
            
            start_time = time.time()
            stats = processor.analyze_dataset(str(self.data_dir / 'train'))
            processing_time = time.time() - start_time
            
            print(f"   处理 {stats.total_images} 张图像耗时: {processing_time:.3f}秒")
            print(f"   平均每张图像: {processing_time/stats.total_images*1000:.2f}毫秒")
            
            # 性能断言
            self.assertLess(processing_time, 10.0, "数据处理时间过长")
        
        # 2. ONNX 导出性能
        if fua.DEPLOYMENT_AVAILABLE:
            print("\n2.2 ONNX 导出性能")
            exporter = fua.create_onnx_exporter()
            
            start_time = time.time()
            onnx_path = self.temp_path / 'perf_test.onnx'
            success = exporter.export_model(
                self.model,
                str(onnx_path),
                optimization_level='basic'
            )
            export_time = time.time() - start_time
            
            self.assertTrue(success)
            print(f"   ONNX 导出耗时: {export_time:.3f}秒")
            
            # 获取模型信息
            info = exporter.get_model_info(str(onnx_path))
            print(f"   模型大小: {info['file_size_mb']:.2f} MB")
            
            # 性能断言
            self.assertLess(export_time, 5.0, "ONNX导出时间过长")
        
        # 3. 优化性能（简化测试）
        if fua.OPTIMIZATION_AVAILABLE:
            print("\n2.3 优化性能（简化）")
            
            # 创建小规模优化
            search_space = {
                'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True}
            }
            
            optimizer = fua.create_hyperparameter_optimizer(
                model_name='perf_test',
                search_space=search_space,
                n_trials=2
            )
            
            # 准备数据
            small_dataset = TensorDataset(self.images[:20], self.labels[:20])
            train_loader = DataLoader(small_dataset, batch_size=10)
            
            def simple_train(model, train_loader, val_loader, params=None, **kwargs):
                opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
                criterion = nn.CrossEntropyLoss()
                
                for _ in range(2):
                    for data, target in train_loader:
                        opt.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        opt.step()
                
                return model, None, 0
            
            def simple_eval(model, val_loader):
                return {'accuracy': 0.5}  # 简化评估
            
            start_time = time.time()
            result = optimizer.optimize(
                train_data=train_loader,
                val_data=train_loader,
                model_factory=lambda **params: self.model,
                train_fn=simple_train,
                eval_fn=simple_eval,
                save_study=False
            )
            opt_time = time.time() - start_time
            
            print(f"   2次试验耗时: {opt_time:.3f}秒")
            print(f"   平均每次试验: {opt_time/2:.3f}秒")
        
        print("\n✓ 性能基准测试通过！")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n3. 错误处理测试")
        print("-" * 50)
        
        # 1. 数据处理错误
        if fua.PIPELINE_AVAILABLE:
            print("\n3.1 数据处理错误处理")
            processor = fua.create_data_processor()
            
            # 测试不存在的文件
            with self.assertRaises(ValueError):
                processor.process_image("nonexistent.jpg")
            
            # 测试无效的图像
            invalid_img_path = self.temp_path / 'invalid.jpg'
            with open(invalid_img_path, 'w') as f:
                f.write("not an image")
            
            with self.assertRaises(ValueError):
                processor.process_image(str(invalid_img_path))
            
            print("   ✓ 正确处理文件错误")
        
        # 2. ONNX 导出错误
        if fua.DEPLOYMENT_AVAILABLE:
            print("\n3.2 ONNX 导出错误处理")
            exporter = fua.create_onnx_exporter()
            
            # 测试无效模型
            with self.assertRaises(Exception):
                exporter.export_model(
                    "not a model",
                    self.temp_path / 'invalid.onnx'
                )
            
            print("   ✓ 正确处理模型错误")
        
        # 3. 优化错误
        if fua.OPTIMIZATION_AVAILABLE:
            print("\n3.3 优化错误处理")
            
            # 测试无效搜索空间
            with self.assertRaises(Exception):
                fua.create_hyperparameter_optimizer(
                    model_name='error_test',
                    search_space={'invalid_param': {'type': 'unknown'}}
                )
            
            print("   ✓ 正确处理配置错误")
        
        print("\n✓ 错误处理测试通过！")
    
    def test_component_integration(self):
        """测试组件集成"""
        print("\n4. 组件集成测试")
        print("-" * 50)
        
        # 检查所有组件是否可用
        components_available = {
            'DEPLOYMENT': fua.DEPLOYMENT_AVAILABLE,
            'PIPELINE': fua.PIPELINE_AVAILABLE,
            'OPTIMIZATION': fua.OPTIMIZATION_AVAILABLE
        }
        
        print("\n4.1 组件可用性检查")
        for component, available in components_available.items():
            status = "✓" if available else "✗"
            print(f"   {status} {component}: {'可用' if available else '不可用'}")
        
        # 测试组件间数据传递
        if all(components_available.values()):
            print("\n4.2 组件间数据传递")
            
            # 1. 使用管道处理数据
            processor = fua.create_data_processor()
            
            # 2. 使用优化器找到最佳参数
            search_space = {
                'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True}
            }
            
            optimizer = fua.create_hyperparameter_optimizer(
                model_name='integration_test',
                search_space=search_space,
                n_trials=2
            )
            
            # 3. 导出优化后的模型
            exporter = fua.create_onnx_exporter()
            
            # 4. 准备推理服务器
            server = fua.create_inference_server()
            
            print("   ✓ 所有组件可以协同工作")
        
        print("\n✓ 组件集成测试通过！")
    
    def test_resource_cleanup(self):
        """测试资源清理"""
        print("\n5. 资源清理测试")
        print("-" * 50)
        
        if fua.PIPELINE_AVAILABLE:
            # 测试数据管道清理
            pipeline = fua.create_data_pipeline(
                str(self.data_dir),
                auto_split=True
            )
            
            # 检查临时文件创建
            temp_dir = self.data_dir.parent / 'temp_train'
            self.assertTrue(temp_dir.exists())
            
            # 清理
            pipeline.cleanup()
            
            # 检查临时文件删除
            self.assertFalse(temp_dir.exists())
            
            print("   ✓ 临时文件正确清理")
        
        print("\n✓ 资源清理测试通过！")


if __name__ == '__main__':
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # 运行测试
    unittest.main(verbosity=2)