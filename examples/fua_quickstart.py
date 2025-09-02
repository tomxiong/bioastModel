"""
FUA 快速开始示例

这个示例展示了如何使用 FUA 进行生物医学图像分析模型的快速开发
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fua


def quick_start_example():
    """快速开始示例：创建、训练和评估一个模型"""
    print("FUA 快速开始示例")
    print("=" * 40)
    
    # 1. 创建模型管理器
    print("\n1. 创建模型管理器...")
    model_manager = fua.ModelManager()
    
    # 2. 创建一个 AirBubble_HybridNet 模型
    print("\n2. 创建 AirBubble_HybridNet 模型...")
    model_id = model_manager.create_model('airbubble_hybrid_net', {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 20
    })
    print(f"   模型ID: {model_id}")
    
    # 3. 获取模型信息
    print("\n3. 模型信息:")
    model = model_manager.get_model(model_id)
    metadata = model.get_metadata()
    capabilities = model.get_capabilities()
    
    print(f"   模型名称: {metadata.name}")
    print(f"   参数数量: {metadata.parameter_count:,}")
    print(f"   计算复杂度: {capabilities.computational_complexity}")
    print(f"   输入尺寸: {capabilities.input_size_range}")
    
    # 4. 模拟训练（使用 bioast 数据集格式）
    print("\n4. 开始训练...")
    training_data = {
        'samples': 1000,
        'image_size': [70, 70],
        'channels': 3,
        'classes': ['negative', 'positive'],
        'description': '70x70 生物医学菌落检测数据集'
    }
    
    training_results = model_manager.train_model(model_id, training_data)
    print(f"   训练完成！")
    print(f"   最终准确率: {training_results.get('accuracy', 0):.4f}")
    print(f"   最终损失: {training_results.get('loss', 0):.4f}")
    
    # 5. 评估模型
    print("\n5. 评估模型...")
    eval_results = model_manager.evaluate_model(model_id, {
        'samples': 200
    })
    print(f"   评估准确率: {eval_results.get('accuracy', 0):.4f}")
    print(f"   精确率: {eval_results.get('precision', 0):.4f}")
    print(f"   召回率: {eval_results.get('recall', 0):.4f}")
    
    return model_id, model_manager


def auto_config_example():
    """自动配置示例"""
    print("\n\nFUA 自动配置示例")
    print("=" * 40)
    
    # 1. 创建配置系统
    print("\n1. 创建配置系统...")
    config_system = fua.ModelConfigurationSystem()
    
    # 2. 定义模型能力
    print("\n2. 定义模型能力...")
    capabilities = fua.ModelCapabilities(
        input_size_range=((60, 60), (80, 80)),
        recommended_batch_size=(32, 64),
        supported_optimizers=['adam', 'sgd'],
        supported_schedulers=['cosine', 'step'],
        special_preprocessing=['normalization', 'bubble_detection'],
        memory_requirements={'min_memory': 1024, 'recommended_memory': 2048},
        computational_complexity='medium',
        training_time_estimate='medium'
    )
    
    # 3. 自动生成配置
    print("\n3. 自动生成配置...")
    config_id = config_system.generate_config_from_capabilities(
        'bioast_optimized_model', 
        capabilities
    )
    
    config = config_system.get_config(config_id)
    print("   生成的配置:")
    print(f"     - 学习率: {config['training']['learning_rate']}")
    print(f"     - 批次大小: {config['training']['batch_size']}")
    print(f"     - 优化器: {config['training']['optimizer']}")
    print(f"     - 预处理: {config['model']['preprocessing']}")
    
    return config_id


def model_comparison_example():
    """模型对比示例"""
    print("\n\nFUA 模型对比示例")
    print("=" * 40)
    
    model_manager = fua.ModelManager()
    
    # 要对比的模型
    models_to_compare = [
        ('airbubble_hybrid_net', {'learning_rate': 0.001, 'batch_size': 32}),
        ('mic_mobilenetv3', {'learning_rate': 0.0005, 'batch_size': 64}),
        ('micro_vit', {'learning_rate': 0.0001, 'batch_size': 16})
    ]
    
    results = {}
    
    print("\n创建和训练模型...")
    for model_name, config in models_to_compare:
        print(f"\n处理 {model_name}:")
        
        # 创建模型
        model_id = model_manager.create_model(model_name, config)
        
        # 训练
        training_data = {'samples': 500, 'image_size': [70, 70]}
        training_results = model_manager.train_model(model_id, training_data)
        
        # 评估
        eval_results = model_manager.evaluate_model(model_id, {'samples': 100})
        
        results[model_name] = {
            'accuracy': eval_results.get('accuracy', 0),
            'training_time': training_results.get('training_time', 0),
            'parameter_count': model_manager.get_model(model_id).get_metadata().parameter_count
        }
        
        print(f"   准确率: {results[model_name]['accuracy']:.4f}")
        print(f"   训练时间: {results[model_name]['training_time']:.3f}秒")
        print(f"   参数量: {results[model_name]['parameter_count']:,}")
    
    # 显示对比结果
    print("\n" + "-" * 40)
    print("模型对比结果")
    print("-" * 40)
    print(f"{'模型':<25} {'准确率':<10} {'训练时间':<12} {'参数量':<15}")
    print("-" * 65)
    
    for model_name, result in results.items():
        print(f"{model_name:<25} {result['accuracy']:<10.4f} "
              f"{result['training_time']:<12.3f} {result['parameter_count']:<15,}")
    
    # 找出最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.4f})")


def main():
    """主函数"""
    print("FUA 快速开始指南")
    print("这些示例展示了 FUA 的核心功能")
    
    try:
        # 示例1：快速开始
        model_id, model_manager = quick_start_example()
        
        # 示例2：自动配置
        config_id = auto_config_example()
        
        # 示例3：模型对比
        model_comparison_example()
        
        print("\n" + "=" * 60)
        print("所有示例完成！")
        print("查看 examples/fua_demo.py 获取更多高级功能演示")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()