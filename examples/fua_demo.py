"""
FUA Demo Script - 演示 FUA (Flexible Unified Architecture) 的核心功能

这个脚本展示了如何使用 FUA 进行：
1. 模型创建和管理
2. 自动配置生成
3. 模型训练和评估
4. 性能对比分析
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import fua
import time
import json
from typing import Dict, Any


def demo_basic_model_management():
    """演示基本的模型管理功能"""
    print("=" * 60)
    print("FUA 基本模型管理演示")
    print("=" * 60)
    
    # 创建模型管理器
    model_manager = fua.ModelManager()
    
    # 可用模型列表
    print("\n1. 查看可用模型:")
    factory = fua.ModelFactory()
    available_models = factory.get_available_models()
    print(f"   可用模型: {available_models}")
    
    # 创建多个模型
    print("\n2. 创建模型:")
    model_configs = [
        ('airbubble_hybrid_net', {'learning_rate': 0.001, 'batch_size': 32}),
        ('mic_mobilenetv3', {'learning_rate': 0.0005, 'batch_size': 64}),
        ('micro_vit', {'learning_rate': 0.0001, 'batch_size': 16})
    ]
    
    model_ids = []
    for model_name, config in model_configs:
        try:
            model_id = model_manager.create_model(model_name, config)
            model_ids.append((model_name, model_id))
            print(f"   ✓ 创建 {model_name}: {model_id}")
        except Exception as e:
            print(f"   ✗ 创建 {model_name} 失败: {e}")
    
    return model_manager, model_ids


def demo_model_capabilities(model_manager, model_ids):
    """演示模型能力分析"""
    print("\n" + "=" * 60)
    print("模型能力分析演示")
    print("=" * 60)
    
    for model_name, model_id in model_ids:
        print(f"\n模型: {model_name}")
        
        # 获取模型实例
        model = model_manager.get_model(model_id)
        if model:
            # 获取能力
            capabilities = model.get_capabilities()
            print(f"   计算复杂度: {capabilities.computational_complexity}")
            print(f"   输入尺寸范围: {capabilities.input_size_range}")
            print(f"   推荐批次大小: {capabilities.recommended_batch_size}")
            print(f"   支持的优化器: {capabilities.supported_optimizers}")
            print(f"   特殊预处理: {capabilities.special_preprocessing}")
            
            # 获取元数据
            metadata = model.get_metadata()
            print(f"   参数数量: {metadata.parameter_count:,}")
            print(f"   架构类型: {metadata.architecture_type}")
            print(f"   内存使用: {metadata.memory_usage} MB")


def demo_configuration_system():
    """演示配置系统功能"""
    print("\n" + "=" * 60)
    print("配置系统演示")
    print("=" * 60)
    
    # 创建配置系统
    config_system = fua.ModelConfigurationSystem()
    
    # 1. 基于能力自动生成配置
    print("\n1. 基于模型能力自动生成配置:")
    
    # 创建不同类型的模型能力
    capability_profiles = {
        'lightweight_cnn': fua.ModelCapabilities(
            input_size_range=((60, 60), (80, 80)),
            recommended_batch_size=(32, 128),
            supported_optimizers=['adam', 'sgd'],
            supported_schedulers=['cosine', 'step'],
            special_preprocessing=['normalization'],
            memory_requirements={'min_memory': 512, 'recommended_memory': 1024},
            computational_complexity='low',
            training_time_estimate='low'
        ),
        'high_performance_hybrid': fua.ModelCapabilities(
            input_size_range=((60, 60), (80, 80)),
            recommended_batch_size=(16, 64),
            supported_optimizers=['adam', 'adamw'],
            supported_schedulers=['cosine', 'reduce_on_plateau'],
            special_preprocessing=['bubble_detection', 'multi_scale'],
            memory_requirements={'min_memory': 2048, 'recommended_memory': 4096},
            computational_complexity='high',
            training_time_estimate='medium'
        )
    }
    
    for profile_name, capabilities in capability_profiles.items():
        config_id = config_system.generate_config_from_capabilities(
            profile_name, capabilities
        )
        config = config_system.get_config(config_id)
        print(f"   ✓ {profile_name}:")
        print(f"     - 批次大小: {config['training']['batch_size']}")
        print(f"     - 学习率: {config['training']['learning_rate']}")
        print(f"     - 预处理: {config['model']['preprocessing']}")
    
    # 2. 使用配置模板
    print("\n2. 使用配置模板:")
    template_configs = ['high_performance', 'memory_efficient', 'fast_training']
    
    for template in template_configs:
        try:
            config_id = config_system.apply_template(template, {
                'model_name': f'{template}_model',
                'custom_param': 'test_value'
            })
            print(f"   ✓ {template} 模板应用成功")
        except Exception as e:
            print(f"   ✗ {template} 模板应用失败: {e}")


def demo_training_and_evaluation(model_manager, model_ids):
    """演示训练和评估功能"""
    print("\n" + "=" * 60)
    print("训练和评估演示")
    print("=" * 60)
    
    # 模拟训练数据
    training_data = {
        'samples': 1000,
        'image_size': [70, 70],
        'channels': 3,
        'classes': ['negative', 'positive'],
        'description': 'Biomedical colony detection dataset'
    }
    
    print(f"\n使用数据集: {training_data['description']}")
    print(f"样本数量: {training_data['samples']}")
    print(f"图像尺寸: {training_data['image_size']}")
    
    results = {}
    
    for model_name, model_id in model_ids:
        print(f"\n训练模型: {model_name}")
        
        # 训练模型
        start_time = time.time()
        training_results = model_manager.train_model(model_id, training_data)
        training_time = time.time() - start_time
        
        print(f"   训练时间: {training_time:.3f}秒")
        print(f"   训练结果: {training_results}")
        
        # 评估模型
        eval_data = {'samples': 200}
        eval_results = model_manager.evaluate_model(model_id, eval_data)
        
        print(f"   评估结果: {eval_results}")
        
        results[model_name] = {
            'training_time': training_time,
            'training_results': training_results,
            'eval_results': eval_results
        }
    
    # 性能对比
    print("\n" + "-" * 40)
    print("模型性能对比")
    print("-" * 40)
    
    best_model = None
    best_accuracy = 0
    
    for model_name, result in results.items():
        accuracy = result['eval_results'].get('accuracy', 0)
        training_time = result['training_time']
        
        print(f"{model_name}:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  训练时间: {training_time:.3f}秒")
        print(f"  损失: {result['training_results'].get('loss', 'N/A')}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    print(f"\n🏆 最佳模型: {best_model} (准确率: {best_accuracy:.4f})")
    
    return results, best_model


def demo_model_persistence(model_manager, best_model_id):
    """演示模型持久化功能"""
    print("\n" + "=" * 60)
    print("模型持久化演示")
    print("=" * 60)
    
    # 保存模型
    model_path = "demo_best_model.json"
    print(f"\n保存最佳模型到: {model_path}")
    
    save_success = model_manager.save_model(best_model_id, model_path)
    if save_success:
        print("   ✓ 模型保存成功")
        
        # 加载模型
        print(f"\n从文件加载模型: {model_path}")
        loaded_model_id = model_manager.load_model(model_path, "loaded_model")
        
        if loaded_model_id:
            print("   ✓ 模型加载成功")
            print(f"   新模型ID: {loaded_model_id}")
            
            # 验证加载的模型
            loaded_model = model_manager.get_model(loaded_model_id)
            if loaded_model:
                metadata = loaded_model.get_metadata()
                print(f"   模型名称: {metadata.name}")
                print(f"   参数数量: {metadata.parameter_count:,}")
        else:
            print("   ✗ 模型加载失败")
    else:
        print("   ✗ 模型保存失败")
    
    # 清理演示文件
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"\n清理演示文件: {model_path}")


def demo_performance_benchmarks():
    """演示性能基准测试"""
    print("\n" + "=" * 60)
    print("性能基准测试演示")
    print("=" * 60)
    
    model_manager = fua.ModelManager()
    
    # 测试模型创建性能
    print("\n1. 模型创建性能测试:")
    num_models = 10
    start_time = time.time()
    
    model_ids = []
    for i in range(num_models):
        model_id = model_manager.create_model('mic_mobilenetv3', {
            'learning_rate': 0.001,
            'batch_size': 32
        })
        model_ids.append(model_id)
    
    creation_time = time.time() - start_time
    avg_creation_time = creation_time / num_models
    models_per_second = num_models / creation_time
    
    print(f"   创建 {num_models} 个模型总时间: {creation_time:.4f}秒")
    print(f"   平均创建时间: {avg_creation_time:.6f}秒/模型")
    print(f"   创建速度: {models_per_second:.2f} 模型/秒")
    
    # 测试配置系统性能
    print("\n2. 配置系统性能测试:")
    config_system = fua.ModelConfigurationSystem()
    
    num_configs = 100
    test_configs = []
    
    for i in range(num_configs):
        config = {
            'name': f'test_model_{i}',
            'architecture_type': 'cnn',
            'input_size': [70, 70],
            'layers': 10 + i,
            'filters': 32 + i
        }
        test_configs.append(config)
    
    start_time = time.time()
    for config in test_configs:
        config_system.validate_model_config('cnn', config)
    
    validation_time = time.time() - start_time
    avg_validation_time = validation_time / num_configs
    validations_per_second = num_configs / validation_time
    
    print(f"   验证 {num_configs} 个配置总时间: {validation_time:.4f}秒")
    print(f"   平均验证时间: {avg_validation_time:.6f}秒/配置")
    print(f"   验证速度: {validations_per_second:.2f} 配置/秒")
    
    # 清理测试模型
    cleanup_count = model_manager.cleanup_old_models(max_age_hours=0)
    print(f"\n清理了 {cleanup_count} 个测试模型")


def main():
    """主演示函数"""
    print("FUA (Flexible Unified Architecture) 功能演示")
    print("=" * 60)
    print("这个演示展示了 FUA 的核心功能")
    print("包括模型管理、配置系统、训练评估和性能基准")
    print("=" * 60)
    
    try:
        # 1. 基本模型管理
        model_manager, model_ids = demo_basic_model_management()
        
        # 2. 模型能力分析
        demo_model_capabilities(model_manager, model_ids)
        
        # 3. 配置系统演示
        demo_configuration_system()
        
        # 4. 训练和评估
        results, best_model_name = demo_training_and_evaluation(model_manager, model_ids)
        
        # 获取最佳模型ID
        best_model_id = None
        for model_name, model_id in model_ids:
            if model_name == best_model_name:
                best_model_id = model_id
                break
        
        # 5. 模型持久化
        if best_model_id:
            demo_model_persistence(model_manager, best_model_id)
        
        # 6. 性能基准测试
        demo_performance_benchmarks()
        
        # 总结
        print("\n" + "=" * 60)
        print("演示总结")
        print("=" * 60)
        print("✓ 模型创建和管理")
        print("✓ 模型能力分析")
        print("✓ 自动配置生成")
        print("✓ 训练和评估流程")
        print("✓ 模型持久化")
        print("✓ 性能基准测试")
        print("\nFUA 提供了完整的机器学习模型生命周期管理解决方案！")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()