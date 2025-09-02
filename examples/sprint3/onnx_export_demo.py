"""
FUA ONNX 导出演示

展示如何使用 FUA 的 ONNX 导出功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fua
import tempfile
import shutil
from pathlib import Path


def demo_onnx_export():
    """演示 ONNX 导出功能"""
    print("FUA ONNX 导出演示")
    print("=" * 50)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 1. 创建模型管理器和模型
        print("\n1. 创建模型...")
        model_manager = fua.ModelManager()
        
        # 创建一个测试模型
        model_id = model_manager.create_model('mic_mobilenetv3', {
            'learning_rate': 0.001,
            'batch_size': 32
        })
        
        # 获取模型实例
        model = model_manager.get_model(model_id)
        print(f"   ✓ 创建了模型: {model_id}")
        
        # 2. 基本导出
        print("\n2. 基本导出...")
        from fua.deployment import create_onnx_exporter
        
        exporter = create_onnx_exporter()
        
        # 基本导出
        basic_path = os.path.join(temp_dir, 'basic_model.onnx')
        success = exporter.export_model(
            model, 
            basic_path,
            optimization_level='basic'
        )
        
        if success:
            print("   ✓ 基本导出成功")
            
            # 获取模型信息
            info = exporter.get_model_info(basic_path)
            print(f"   - 文件大小: {info['file_size_mb']:.2f} MB")
            print(f"   - 输入形状: {info['inputs'][0]['shape']}")
            print(f"   - 输出形状: {info['outputs'][0]['shape']}")
        else:
            print("   ✗ 基本导出失败")
        
        # 3. 高级优化导出
        print("\n3. 高级优化导出...")
        advanced_path = os.path.join(temp_dir, 'advanced_model.onnx')
        success = exporter.export_model(
            model,
            advanced_path,
            optimization_level='advanced',
            quantization='fp16'
        )
        
        if success:
            print("   ✓ 高级优化导出成功")
            
            # 比较文件大小
            basic_size = Path(basic_path).stat().st_size
            advanced_size = Path(advanced_path).stat().st_size
            reduction = (basic_size - advanced_size) / basic_size * 100
            
            print(f"   - 基本版本: {basic_size / 1024:.2f} KB")
            print(f"   - 优化版本: {advanced_size / 1024:.2f} KB")
            print(f"   - 大小减少: {reduction:.1f}%")
        else:
            print("   ✗ 高级优化导出失败")
        
        # 4. 批量导出
        print("\n4. 批量导出...")
        
        # 创建多个模型
        models_to_export = []
        model_types = ['airbubble_hybrid_net', 'mic_mobilenetv3', 'micro_vit']
        
        for model_type in model_types:
            try:
                mid = model_manager.create_model(model_type, {'epochs': 10})
                m = model_manager.get_model(mid)
                models_to_export.append((m, f'{model_type}.onnx'))
            except Exception as e:
                print(f"   ⚠ 创建 {model_type} 失败: {e}")
        
        # 批量导出
        batch_dir = os.path.join(temp_dir, 'batch_export')
        results = exporter.batch_export(models_to_export, batch_dir)
        
        successful = sum(1 for r in results.values() if r)
        print(f"   ✓ 批量导出完成: {successful}/{len(results)} 成功")
        
        # 5. 性能对比
        print("\n5. 性能对比...")
        
        # 测试不同优化级别的性能
        optimization_levels = ['basic', 'intermediate', 'advanced']
        perf_results = {}
        
        for level in optimization_levels:
            path = os.path.join(temp_dir, f'perf_test_{level}.onnx')
            
            success = exporter.export_model(
                model,
                path,
                optimization_level=level
            )
            
            if success:
                info = exporter.get_model_info(path)
                perf_results[level] = {
                    'size_kb': info['file_size_mb'] * 1024,
                    'avg_inference_time_ms': info.get('avg_inference_time_ms', 0)
                }
        
        print("优化级别对比:")
        print(f"{'级别':<12} {'大小(KB)':<10} {'推理时间(ms)':<15}")
        print("-" * 40)
        
        for level, results in perf_results.items():
            print(f"{level:<12} {results['size_kb']:<10.1f} {results['avg_inference_time_ms']:<15.2f}")
        
        # 6. 导出元数据
        print("\n6. 导出元数据...")
        metadata_path = os.path.join(temp_dir, 'metadata_model.onnx')
        
        # 添加自定义元数据
        custom_metadata = {
            'author': 'FUA Demo',
            'description': '演示用的 ONNX 模型',
            'framework': 'PyTorch',
            'task': 'binary_classification'
        }
        
        success = exporter.export_model(
            model,
            metadata_path,
            optimization_level='basic'
        )
        
        if success:
            info = exporter.get_model_info(metadata_path)
            print("   ✓ 导出的元数据:")
            for key, value in info['metadata'].items():
                print(f"     - {key}: {value}")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir)
        print(f"\n清理临时目录: {temp_dir}")


def demo_integration_with_fua():
    """演示与 FUA 系统的集成"""
    print("\n" + "=" * 50)
    print("FUA 集成演示")
    print("=" * 50)
    
    # 创建模型管理器
    model_manager = fua.ModelManager()
    
    # 创建和训练一个模型
    print("\n1. 创建和训练模型...")
    model_id = model_manager.create_model('airbubble_hybrid_net', {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 5
    })
    
    # 模拟训练
    training_data = {'samples': 100, 'image_size': [70, 70]}
    training_results = model_manager.train_model(model_id, training_data)
    
    print(f"   ✓ 训练完成，准确率: {training_results.get('accuracy', 0):.4f}")
    
    # 评估模型
    eval_results = model_manager.evaluate_model(model_id, {'samples': 50})
    print(f"   ✓ 评估完成，准确率: {eval_results.get('accuracy', 0):.4f}")
    
    # 导出为 ONNX
    print("\n2. 导出为 ONNX...")
    model = model_manager.get_model(model_id)
    
    from fua.deployment import create_onnx_exporter
    exporter = create_onnx_exporter()
    
    # 导出到临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx_path = f.name
    
    try:
        success = exporter.export_model(
            model,
            onnx_path,
            optimization_level='advanced',
            quantization='fp16'
        )
        
        if success:
            print("   ✓ 模型导出成功")
            
            # 验证导出的模型
            info = exporter.get_model_info(onnx_path)
            print(f"   - 文件大小: {info['file_size_mb']:.2f} MB")
            print(f"   - 优化级别: advanced + fp16")
            
            # 使用 ONNX Runtime 进行推理
            import onnxruntime as ort
            import numpy as np
            
            session = ort.InferenceSession(onnx_path)
            
            # 测试推理
            dummy_input = np.random.randn(1, 3, 70, 70).astype(np.float32)
            outputs = session.run(None, {'input': dummy_input})
            
            print(f"   ✓ ONNX 推理测试成功")
            print(f"   - 输出形状: {outputs[0].shape}")
            print(f"   - 预测结果: {outputs[0][0]:.4f}")
            
        else:
            print("   ✗ 模型导出失败")
    
    finally:
        # 清理
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


def main():
    """主函数"""
    print("FUA ONNX 导出功能演示")
    print("这个演示展示了 FUA 的 ONNX 导出和优化功能")
    
    # 基本导出演示
    demo_onnx_export()
    
    # FUA 集成演示
    demo_integration_with_fua()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n关键功能:")
    print("- ✓ 基本和高级 ONNX 导出")
    print("- ✓ 模型优化（basic/intermediate/advanced）")
    print("- ✓ 量化支持（FP16/INT8）")
    print("- ✓ 批量导出")
    print("- ✓ 性能基准测试")
    print("- ✓ 元数据管理")
    print("- ✓ 与 FUA 系统深度集成")
    print("=" * 50)


if __name__ == "__main__":
    main()