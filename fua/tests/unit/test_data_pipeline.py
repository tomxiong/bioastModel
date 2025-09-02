"""
FUA 数据处理管道测试

测试数据处理管道的各项功能
"""

import unittest
import tempfile
import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import shutil
import json

# Import FUA components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import fua


class TestDataProcessor(unittest.TestCase):
    """数据处理器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'test_dataset'
        self.data_dir.mkdir()
        
        # 创建测试图像
        self.create_test_images()
        
        # 创建处理器
        self.processor = fua.create_data_processor(
            image_size=(70, 70),
            enable_auto_augment=False  # 关闭随机增强以确保测试稳定
        )
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_images(self):
        """创建测试图像"""
        # 创建目录
        (self.data_dir / 'negative').mkdir()
        (self.data_dir / 'positive').mkdir()
        
        # 创建负样本
        for i in range(5):
            # 简单的噪声图像
            img = np.random.randint(100, 200, (70, 70, 3), dtype=np.uint8)
            Image.fromarray(img).save(self.data_dir / 'negative' / f'neg_{i}.jpg')
        
        # 创建正样本
        for i in range(5):
            # 带有圆形的图像
            img = np.random.randint(50, 150, (70, 70, 3), dtype=np.uint8)
            cv2.circle(img, (35, 35), 20, (200, 150, 100), -1)
            Image.fromarray(img).save(self.data_dir / 'positive' / f'pos_{i}.jpg')
    
    def test_basic_processing(self):
        """测试基本处理功能"""
        print("\n测试基本处理...")
        
        # 处理单张图像
        image_path = self.data_dir / 'positive' / 'pos_0.jpg'
        result = self.processor.process_image(str(image_path))
        
        # 验证结果
        self.assertIsInstance(result, fua.pipeline.data_processor.ProcessingResult)
        self.assertEqual(result.image.shape, (70, 70, 3))
        self.assertIsInstance(result.metrics, dict)
        self.assertIsInstance(result.quality_level, fua.pipeline.data_processor.QualityLevel)
        self.assertGreater(result.processing_time, 0)
    
    def test_quality_metrics(self):
        """测试质量指标计算"""
        print("\n测试质量指标...")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (70, 70, 3), dtype=np.uint8)
        
        # 计算质量指标
        metrics = self.processor.check_image_quality(test_image)
        
        # 验证指标
        required_metrics = ['sharpness', 'brightness', 'contrast', 'bubble_ratio',
                          'exposure', 'saturation', 'noise_level', 'edge_density', 'blur_score']
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_quality_assessment(self):
        """测试质量评估"""
        print("\n测试质量评估...")
        
        # 创建高质量图像
        high_quality_img = np.random.randint(100, 200, (70, 70, 3), dtype=np.uint8)
        # 添加边缘以提高清晰度
        high_quality_img[20:50, 20:50] = 255
        
        metrics = self.processor.check_image_quality(high_quality_img)
        warnings = []
        quality_level = self.processor._assess_quality_level(metrics, warnings)
        
        # 应该是良好或优秀
        self.assertIn(quality_level, [fua.pipeline.data_processor.QualityLevel.GOOD,
                                     fua.pipeline.data_processor.QualityLevel.EXCELLENT])
    
    def test_batch_processing(self):
        """测试批量处理"""
        print("\n测试批量处理...")
        
        # 收集所有图像
        image_paths = []
        for class_dir in ['negative', 'positive']:
            class_path = self.data_dir / class_dir
            image_paths.extend([str(p) for p in class_path.glob('*.jpg')])
        
        # 批量处理
        results = self.processor.process_batch(image_paths, parallel=False)
        
        # 验证结果
        self.assertEqual(len(results), len(image_paths))
        for result in results:
            self.assertIsInstance(result, fua.pipeline.data_processor.ProcessingResult)
            self.assertEqual(result.image.shape, (70, 70, 3))
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        print("\n测试数据集创建...")
        
        # 创建数据集
        dataset = self.processor.create_dataset(str(self.data_dir))
        
        # 验证数据集
        self.assertGreater(len(dataset), 0)
        
        # 测试获取样本
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 70, 70))
        self.assertIn(label, [0, 1])
    
    def test_dataset_analysis(self):
        """测试数据集分析"""
        print("\n测试数据集分析...")
        
        # 分析数据集
        stats = self.processor.analyze_dataset(str(self.data_dir))
        
        # 验证统计信息
        self.assertIsInstance(stats, fua.pipeline.data_processor.DatasetStats)
        self.assertEqual(stats.total_images, 10)
        self.assertEqual(stats.class_distribution['negative'], 5)
        self.assertEqual(stats.class_distribution['positive'], 5)
        self.assertIsInstance(stats.average_metrics, dict)
    
    def test_data_splits(self):
        """测试数据划分"""
        print("\n测试数据划分...")
        
        # 创建平衡划分
        splits = self.processor.create_balanced_splits(
            str(self.data_dir),
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # 验证划分
        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)
        
        # 检查划分比例
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        self.assertAlmostEqual(len(splits['train']) / total, 0.6, delta=0.1)
        self.assertAlmostEqual(len(splits['val']) / total, 0.2, delta=0.1)
        self.assertAlmostEqual(len(splits['test']) / total, 0.2, delta=0.1)
    
    def test_quality_filtering(self):
        """测试质量过滤"""
        print("\n测试质量过滤...")
        
        # 创建带质量过滤的数据集
        dataset = self.processor.create_dataset(
            str(self.data_dir),
            quality_filter=fua.pipeline.data_processor.QualityLevel.ACCEPTABLE
        )
        
        # 验证过滤（应该仍然有图像，因为测试图像质量都还可以）
        self.assertGreater(len(dataset), 0)
    
    def test_quality_report(self):
        """测试质量报告"""
        print("\n测试质量报告...")
        
        # 生成报告
        report_path = os.path.join(self.temp_dir, 'quality_report.json')
        self.processor.export_quality_report(str(self.data_dir), report_path)
        
        # 验证报告文件
        self.assertTrue(os.path.exists(report_path))
        
        # 读取并验证内容
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        self.assertIn('dataset_stats', report)
        self.assertIn('quality_thresholds', report)
        self.assertIn('processing_stats', report)


class TestDataPipeline(unittest.TestCase):
    """数据管道测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'test_dataset'
        self.data_dir.mkdir()
        
        # 创建测试图像
        self.create_test_images()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_images(self):
        """创建测试图像"""
        # 创建目录
        (self.data_dir / 'negative').mkdir()
        (self.data_dir / 'positive').mkdir()
        
        # 创建负样本
        for i in range(15):
            img = np.random.randint(100, 200, (70, 70, 3), dtype=np.uint8)
            Image.fromarray(img).save(self.data_dir / 'negative' / f'neg_{i}.jpg')
        
        # 创建正样本
        for i in range(15):
            img = np.random.randint(50, 150, (70, 70, 3), dtype=np.uint8)
            cv2.circle(img, (35, 35), 20, (200, 150, 100), -1)
            Image.fromarray(img).save(self.data_dir / 'positive' / f'pos_{i}.jpg')
    
    def test_pipeline_creation(self):
        """测试管道创建"""
        print("\n测试管道创建...")
        
        # 创建管道
        pipeline = fua.create_data_pipeline(
            str(self.data_dir),
            auto_split=True,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # 验证管道
        self.assertIsInstance(pipeline, fua.pipeline.data_processor.DataPipeline)
    
    def test_dataloader_creation(self):
        """测试数据加载器创建"""
        print("\n测试数据加载器...")
        
        # 创建管道
        pipeline = fua.create_data_pipeline(str(self.data_dir))
        
        # 获取数据加载器
        train_loader = pipeline.get_dataloader('train', batch_size=4)
        val_loader = pipeline.get_dataloader('val', batch_size=4)
        test_loader = pipeline.get_dataloader('test', batch_size=4)
        
        # 验证数据加载
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            images, labels = batch
            self.assertEqual(images.shape[0], 4)  # batch size
            self.assertEqual(images.shape[1], 3)  # channels
            self.assertEqual(images.shape[2], 70)  # height
            self.assertEqual(images.shape[3], 70)  # width
    
    def test_pipeline_analysis(self):
        """测试管道分析"""
        print("\n测试管道分析...")
        
        # 创建管道
        pipeline = fua.create_data_pipeline(str(self.data_dir))
        
        # 分析所有数据集
        stats = pipeline.analyze_all_datasets()
        
        # 验证统计
        self.assertIsInstance(stats, dict)
        self.assertIn('train', stats)
        self.assertIn('val', stats)
        self.assertIn('test', stats)
        
        # 验证总样本数
        total_samples = sum(s.total_images for s in stats.values())
        self.assertEqual(total_samples, 30)
    
    def test_pipeline_cleanup(self):
        """测试管道清理"""
        print("\n测试管道清理...")
        
        # 创建管道
        pipeline = fua.create_data_pipeline(str(self.data_dir))
        
        # 检查临时目录是否存在
        temp_train = self.data_dir.parent / 'temp_train'
        self.assertTrue(temp_train.exists())
        
        # 清理
        pipeline.cleanup()
        
        # 验证临时目录被删除
        self.assertFalse(temp_train.exists())


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / 'perf_dataset'
        self.data_dir.mkdir()
        
        # 创建更多测试图像
        self.create_test_images(100)
        
        # 创建处理器
        self.processor = fua.create_data_processor(num_workers=2)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_images(self, num_images: int):
        """创建测试图像"""
        # 创建目录
        (self.data_dir / 'negative').mkdir()
        (self.data_dir / 'positive').mkdir()
        
        # 创建图像
        for i in range(num_images // 2):
            # 负样本
            img = np.random.randint(100, 200, (70, 70, 3), dtype=np.uint8)
            Image.fromarray(img).save(self.data_dir / 'negative' / f'neg_{i}.jpg')
            
            # 正样本
            img = np.random.randint(50, 150, (70, 70, 3), dtype=np.uint8)
            cv2.circle(img, (35, 35), 20, (200, 150, 100), -1)
            Image.fromarray(img).save(self.data_dir / 'positive' / f'pos_{i}.jpg')
    
    def test_processing_speed(self):
        """测试处理速度"""
        print("\n测试处理速度...")
        
        # 收集图像路径
        image_paths = []
        for class_dir in ['negative', 'positive']:
            class_path = self.data_dir / class_dir
            image_paths.extend([str(p) for p in class_path.glob('*.jpg')])
        
        # 测试串行处理
        import time
        start_time = time.time()
        results_serial = self.processor.process_batch(image_paths[:20], parallel=False)
        serial_time = time.time() - start_time
        
        # 测试并行处理
        start_time = time.time()
        results_parallel = self.processor.process_batch(image_paths[:20], parallel=True)
        parallel_time = time.time() - start_time
        
        # 验证结果一致性
        self.assertEqual(len(results_serial), len(results_parallel))
        
        # 打印性能对比
        print(f"  串行处理时间: {serial_time:.3f}秒")
        print(f"  并行处理时间: {parallel_time:.3f}秒")
        print(f"  加速比: {serial_time/parallel_time:.2f}x")
        
        # 并行处理应该更快（但考虑到启动开销，只要求不慢太多）
        self.assertLess(parallel_time, serial_time * 1.5)
    
    def test_memory_usage(self):
        """测试内存使用"""
        print("\n测试内存速度...")
        
        # 创建大型数据集
        large_dataset = self.processor.create_dataset(str(self.data_dir))
        
        # 测试多次迭代
        for i in range(5):
            for j in range(min(10, len(large_dataset))):
                image, label = large_dataset[j]
                self.assertEqual(image.shape, (3, 70, 70))


if __name__ == '__main__':
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # 运行测试
    unittest.main(verbosity=2)