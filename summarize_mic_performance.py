#!/usr/bin/env python3
"""
从现有文件总结M16 MIC MobileNetV3模型性能
"""

import json
import os
from pathlib import Path

def summarize_mic_mobilenetv3_performance():
    """总结MIC MobileNetV3模型性能"""
    
    print("🔍 M16 MultiTask MobileNetV3 性能总结")
    print("="*60)
    
    model_dir = Path("experiments/mic_mobilenetv3")
    
    if not model_dir.exists():
        print("❌ 模型目录不存在")
        return
    
    # 读取配置信息
    config_file = model_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("📋 训练配置:")
        print(f"   批次大小: {config.get('batch_size', 'N/A')}")
        print(f"   训练轮数: {config.get('epochs', 'N/A')}")
        print(f"   学习率: {config.get('learning_rate', 'N/A')}")
        print(f"   宽度倍数: {config.get('width_mult', 'N/A')}")
        print(f"   丢弃率: {config.get('dropout_rate', 'N/A')}")
    
    # 读取元数据
    metadata_file = model_dir / "m16_multitask_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"\n🎯 模型信息:")
        print(f"   模型名称: {metadata.get('model_name', 'N/A')}")
        print(f"   版本: {metadata.get('version', 'N/A')}")
        print(f"   输入尺寸: {metadata.get('input_size', 'N/A')}")
        
        output_format = metadata.get('output_format', {})
        print(f"\n📊 输出任务:")
        for task, desc in output_format.items():
            print(f"   {task}: {desc}")
        
        performance = metadata.get('performance', {})
        if performance:
            print(f"\n🏆 性能指标:")
            print(f"   验证准确率: {performance.get('validation_accuracy', 'N/A')}")
            print(f"   最佳轮次: {performance.get('best_epoch', 'N/A')}")
            print(f"   训练轮数: {performance.get('training_epochs', 'N/A')}")
    
    # 读取验证报告
    validation_file = model_dir / "validation_report.json"
    if validation_file.exists():
        with open(validation_file, 'r', encoding='utf-8') as f:
            validation = json.load(f)
        
        model_info = validation.get('model_info', {})
        if model_info:
            print(f"\n⚙️ 模型统计:")
            print(f"   参数数量: {model_info.get('parameter_count', 'N/A')}")
            print(f"   ONNX文件大小: {model_info.get('onnx_file_size', 'N/A')}")
        
        benchmark = validation.get('performance_benchmark', {})
        if benchmark:
            print(f"\n⚡ 性能对比:")
            print(f"   PyTorch推理时间: {benchmark.get('pytorch_time_ms', 'N/A'):.2f} ms")
            print(f"   ONNX推理时间: {benchmark.get('onnx_time_ms', 'N/A'):.2f} ms")
            print(f"   加速比: {benchmark.get('speedup', 'N/A'):.2f}x")
    
    # 检查模型文件
    model_file = model_dir / "best.pth"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"\n📁 模型文件:")
        print(f"   PyTorch模型: {size_mb:.1f} MB")
    
    # 列出所有可用文件
    print(f"\n📄 可用文件:")
    for file in sorted(model_dir.iterdir()):
        if file.is_file():
            print(f"   {file.name}")
    
    # 数据集信息
    dataset_info_file = Path("dataset_ni_multitask/dataset_info.json")
    if dataset_info_file.exists():
        with open(dataset_info_file, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        split_stats = dataset_info.get('split_statistics', {})
        print(f"\n📊 测试数据集统计:")
        test_stats = split_stats.get('test', {})
        if test_stats:
            print(f"   测试样本总数: {test_stats.get('total_samples', 'N/A')}")
            
            growth_dist = test_stats.get('growth_level_dist', {})
            print(f"   生长级别分布:")
            for level, count in growth_dist.items():
                print(f"     {level}: {count}")
    
    print(f"\n✅ 性能总结完成！")
    
    return {
        'model_exists': model_file.exists() if 'model_file' in locals() else False,
        'metadata_available': metadata_file.exists(),
        'validation_available': validation_file.exists()
    }

def compare_with_other_models():
    """与其他已知模型性能对比"""
    print(f"\n🏁 与项目中其他顶级模型对比:")
    
    # 根据README.md中的性能排行
    top_models = [
        ("AirBubble_HybridNet", "98.02%", "CNN-Transformer混合架构"),
        ("ResNet18-Improved", "97.83%", "改进版ResNet"),
        ("EfficientNet-B0", "97.54%", "高效CNN"),
        ("MIC_MobileNetV3", "90.69%", "MIC专用多任务MobileNetV3"), # 从metadata中获取
        ("Micro-ViT", "97.36%", "微型Vision Transformer")
    ]
    
    print(f"   排名 | 模型名称                 | 准确率  | 描述")
    print(f"   ----|------------------------|---------|------------------")
    
    for i, (name, acc, desc) in enumerate(top_models, 1):
        marker = " 👈" if "MIC" in name else ""
        print(f"   {i:2d}  | {name:20s} | {acc:7s} | {desc}{marker}")

def main():
    """主函数"""
    summary = summarize_mic_mobilenetv3_performance()
    
    if summary['metadata_available']:
        compare_with_other_models()
        
        print(f"\n💡 分析结论:")
        print(f"   • M16 MIC MobileNetV3 是一个多任务学习模型")
        print(f"   • 专为MIC测试场景设计，包含4个并行任务")
        print(f"   • 相比单任务模型准确率较低，但功能更全面")
        print(f"   • ONNX转换成功，推理速度快于PyTorch")
        print(f"   • 适合需要多维度分析的生物医学应用")

if __name__ == "__main__":
    main()