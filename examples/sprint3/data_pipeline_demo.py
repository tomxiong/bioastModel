"""
FUA 数据处理管道演示

展示如何使用 FUA 的数据处理管道功能
"""

import sys
import os
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fua
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import json


def create_sample_dataset(data_dir: Path, num_samples: int = 20):
    """创建示例数据集"""
    # 创建目录结构
    for class_name in ['negative', 'positive']:
        (data_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # 生成示例图像
    for i in range(num_samples // 2):
        # 生成负样本（纯色+噪声）
        img = np.random.randint(100, 200, (70, 70, 3), dtype=np.uint8)
        # 添加一些空泡效果
        for _ in range(np.random.randint(0, 5)):
            x, y = np.random.randint(0, 70, 2)
            radius = np.random.randint(3, 8)
            cv2.circle(img, (x, y), radius, (255, 255, 255), -1)
        
        Image.fromarray(img).save(data_dir / 'negative' / f'neg_{i:03d}.jpg')
        
        # 生成正样本（有图案）
        img = np.random.randint(50, 150, (70, 70, 3), dtype=np.uint8)
        # 添加一个"菌落"
        center = np.random.randint(20, 50, 2)
        radius = np.random.randint(10, 20)
        cv2.circle(img, tuple(center), radius, (200, 150, 100), -1)
        # 添加纹理
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        Image.fromarray(img).save(data_dir / 'positive' / f'pos_{i:03d}.jpg')


def demo_basic_processing():
    """演示基本处理功能"""
    print("\n1. 基本数据处理演示")
    print("-" * 40)
    
    # 创建临时目录和示例数据
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / 'sample_dataset'
    create_sample_dataset(data_dir, 20)
    
    try:
        # 创建数据处理器
        processor = fua.create_data_processor(
            image_size=(70, 70),
            enable_auto_augment=True
        )
        
        # 处理单张图像
        sample_image = data_dir / 'positive' / 'pos_000.jpg'
        result = processor.process_image(str(sample_image))
        
        print(f"处理图像: {sample_image.name}")
        print(f"质量等级: {result.quality_level.value}")
        print(f"处理时间: {result.processing_time*1000:.2f}ms")
        print(f"主要指标:")
        for metric, value in result.metrics.items():
            print(f"  - {metric}: {value:.3f}")
        
        if result.warnings:
            print(f"警告: {'; '.join(result.warnings)}")
        
        return processor
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        shutil.rmtree(temp_dir)


def demo_batch_processing():
    """演示批量处理功能"""
    print("\n2. 批量处理演示")
    print("-" * 40)
    
    # 创建临时目录和示例数据
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / 'sample_dataset'
    create_sample_dataset(data_dir, 30)
    
    try:
        # 创建数据处理器
        processor = fua.create_data_processor(num_workers=2)
        
        # 收集所有图像
        image_paths = []
        for class_dir in ['negative', 'positive']:
            class_path = data_dir / class_dir
            if class_path.exists():
                image_paths.extend([str(p) for p in class_path.glob('*.jpg')])
        
        # 批量处理
        print(f"批量处理 {len(image_paths)} 张图像...")
        results = processor.process_batch(image_paths, parallel=True)
        
        # 统计结果
        quality_counts = {level.value: 0 for level in processor.QualityLevel}
        total_time = 0
        
        for result in results:
            quality_counts[result.quality_level.value] += 1
            total_time += result.processing_time
        
        print(f"\n处理结果:")
        print(f"  - 总图像数: {len(results)}")
        print(f"  - 平均处理时间: {total_time/len(results)*1000:.2f}ms")
        print(f"  - 质量分布:")
        for level, count in quality_counts.items():
            print(f"    * {level}: {count}")
        
        # 显示一些有问题的图像
        poor_quality = [r for r in results if r.quality_level == processor.QualityLevel.POOR]
        if poor_quality:
            print(f"\n发现 {len(poor_quality)} 张低质量图像")
        
        return processor
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        shutil.rmtree(temp_dir)


def demo_quality_analysis():
    """演示质量分析功能"""
    print("\n3. 数据集质量分析")
    print("-" * 40)
    
    # 创建临时目录和示例数据
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / 'sample_dataset'
    create_sample_dataset(data_dir, 50)
    
    try:
        # 创建数据处理器
        processor = fua.create_data_processor()
        
        # 分析数据集
        print("分析数据集质量...")
        stats = processor.analyze_dataset(str(data_dir))
        
        print(f"\n数据集统计:")
        print(f"  - 总图像数: {stats.total_images}")
        print(f"  - 类别分布:")
        for class_name, count in stats.class_distribution.items():
            print(f"    * {class_name}: {count}")
        
        print(f"\n质量分布:")
        for level, count in stats.quality_distribution.items():
            percentage = count / stats.total_images * 100
            print(f"  - {level}: {count} ({percentage:.1f}%)")
        
        print(f"\n平均质量指标:")
        for metric, value in stats.average_metrics.items():
            print(f"  - {metric}: {value:.3f}")
        
        print(f"\n处理错误: {stats.processing_errors}")
        
        # 导出质量报告
        report_path = temp_dir / 'quality_report.json'
        processor.export_quality_report(str(data_dir), str(report_path))
        print(f"\n质量报告已保存到: {report_path}")
        
        return processor
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        shutil.rmtree(temp_dir)


def demo_data_pipeline():
    """演示完整的数据管道"""
    print("\n4. 完整数据管道演示")
    print("-" * 40)
    
    # 创建临时目录和示例数据
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / 'sample_dataset'
    create_sample_dataset(data_dir, 60)
    
    try:
        # 创建数据管道
        pipeline = fua.create_data_pipeline(
            str(data_dir),
            auto_split=True,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        # 获取数据加载器
        train_loader = pipeline.get_dataloader('train', batch_size=8, shuffle=True)
        val_loader = pipeline.get_dataloader('val', batch_size=8, shuffle=False)
        test_loader = pipeline.get_dataloader('test', batch_size=8, shuffle=False)
        
        print(f"数据集划分:")
        print(f"  - 训练集: {len(train_loader.dataset)} 样本")
        print(f"  - 验证集: {len(val_loader.dataset)} 样本")
        print(f"  - 测试集: {len(test_loader.dataset)} 样本")
        
        # 测试数据加载
        print(f"\n测试数据加载...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"批次 {batch_idx + 1}:")
            print(f"  - 图像形状: {images.shape}")
            print(f"  - 标签: {labels}")
            print(f"  - 像素值范围: [{images.min():.3f}, {images.max():.3f}]")
            
            if batch_idx >= 2:  # 只显示前3个批次
                break
        
        # 分析所有数据集
        print(f"\n分析所有数据集...")
        all_stats = pipeline.analyze_all_datasets()
        
        for split_name, stats in all_stats.items():
            print(f"\n{split_name.upper()} 数据集:")
            print(f"  - 样本数: {stats.total_images}")
            print(f"  - 正样本: {stats.class_distribution.get('positive', 0)}")
            print(f"  - 负样本: {stats.class_distribution.get('negative', 0)}")
            print(f"  - 高质量比例: {stats.quality_distribution.get('excellent', 0) / stats.total_images * 100:.1f}%")
        
        return pipeline
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        pipeline.cleanup()
        shutil.rmtree(temp_dir)


def demo_advanced_features():
    """演示高级功能"""
    print("\n5. 高级功能演示")
    print("-" * 40)
    
    # 创建临时目录和示例数据
    temp_dir = Path(tempfile.mkdtemp())
    data_dir = temp_dir / 'sample_dataset'
    create_sample_dataset(data_dir, 40)
    
    try:
        # 自定义质量阈值
        custom_thresholds = {
            'sharpness': {'excellent': 150, 'good': 80, 'acceptable': 30},
            'brightness': {'min': 50, 'max': 200},
            'contrast': {'min': 30, 'excellent': 60},
            'bubble_ratio': {'max': 0.2}
        }
        
        # 创建自定义处理器
        processor = fua.create_data_processor(
            quality_thresholds=custom_thresholds,
            enable_auto_augment=False
        )
        
        # 创建带质量过滤的数据集
        dataset = processor.create_dataset(
            str(data_dir),
            mode=processor.ProcessingMode.VAL,
            quality_filter=processor.QualityLevel.GOOD
        )
        
        print(f"使用自定义质量阈值:")
        for metric, thresholds in custom_thresholds.items():
            print(f"  - {metric}: {thresholds}")
        
        print(f"\n过滤后的数据集大小: {len(dataset)}")
        
        # 获取质量统计
        quality_stats = dataset.get_quality_stats()
        print(f"\n数据集质量分布:")
        for level, count in quality_stats.items():
            print(f"  - {level}: {count}")
        
        # 处理摘要
        summary = processor.get_processing_summary()
        print(f"\n处理摘要:")
        print(f"  - 总处理数: {summary['total_processed']}")
        print(f"  - 错误率: {summary['error_rate']:.2%}")
        print(f"  - 图像尺寸: {summary['image_size']}")
        
        return processor
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理
        shutil.rmtree(temp_dir)


def main():
    """主函数"""
    print("FUA 数据处理管道功能演示")
    print("=" * 50)
    
    # 检查可用性
    if not fua.PIPELINE_AVAILABLE:
        print("❌ 管道模块不可用")
        return
    
    # 运行各项演示
    demo_basic_processing()
    demo_batch_processing()
    demo_quality_analysis()
    demo_data_pipeline()
    demo_advanced_features()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n主要功能:")
    print("- ✓ 图像质量评估和过滤")
    print("- ✓ 高级数据增强")
    print("- ✓ 批量并行处理")
    print("- ✓ 自动数据集划分")
    print("- ✓ 质量报告生成")
    print("- ✓ 自定义处理管道")
    print("- ✓ 缓存机制优化")
    print("- ✓ 多种数据加载模式")
    print("=" * 50)


if __name__ == "__main__":
    main()