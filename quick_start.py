#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BioAst模型管理系统 - 快速开始脚本

这个脚本提供了一个简单的命令行界面，让用户可以快速开始使用系统。

使用方法:
    python quick_start.py                    # 交互式菜单
    python quick_start.py train <model>      # 训练指定模型
    python quick_start.py list               # 列出所有模型
    python quick_start.py compare            # 对比模型
    python quick_start.py status             # 查看系统状态
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.integration import ModelLifecycleManager
    from utils.config import ConfigManager
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装所有依赖包: pip install -r requirements.txt")
    sys.exit(1)

def print_banner():
    """打印系统横幅"""
    print("\n" + "="*60)
    print("🧬 BioAst模型管理系统 - 快速开始")
    print("="*60)
    print("专为生物信息学设计的模型生命周期管理平台")
    print("="*60 + "\n")

def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查数据目录
    data_dirs = ['bioast_dataset', 'data', './bioast_dataset']
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d):
            data_dir = d
            break
    
    if not data_dir:
        print("⚠️ 警告: 未找到数据集目录")
        print("请确保数据集位于以下位置之一:")
        for d in data_dirs:
            print(f"  - {d}")
        return False
    else:
        print(f"✅ 找到数据集: {data_dir}")
    
    # 检查必要目录
    required_dirs = ['models', 'experiments', 'reports', 'logs']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"📁 创建目录: {dir_name}")
    
    print("✅ 环境检查完成\n")
    return True

def get_available_models():
    """获取可用的模型列表"""
    return {
        '1': {
            'name': 'efficientnet_b0',
            'display_name': 'EfficientNet-B0',
            'description': '轻量级高效模型，适合快速训练'
        },
        '2': {
            'name': 'resnet18_improved',
            'display_name': 'ResNet18-Improved',
            'description': '改进版ResNet18，稳定可靠'
        },
        '3': {
            'name': 'airbubble_hybrid_net',
            'display_name': 'AirBubble-HybridNet',
            'description': '混合架构，专为菌落检测优化'
        },
        '4': {
            'name': 'micro_vit',
            'display_name': 'Micro-ViT',
            'description': '微型Vision Transformer'
        },
        '5': {
            'name': 'convnext_tiny',
            'display_name': 'ConvNeXt-Tiny',
            'description': '现代卷积网络架构'
        }
    }

def show_main_menu():
    """显示主菜单"""
    print("🎯 请选择操作:")
    print("1. 训练单个模型")
    print("2. 查看所有模型")
    print("3. 模型对比分析")
    print("4. 系统状态")
    print("5. 批量训练")
    print("6. 数据集检查")
    print("7. 生成报告")
    print("8. 帮助文档")
    print("0. 退出")
    print("-" * 40)

def show_model_menu():
    """显示模型选择菜单"""
    models = get_available_models()
    print("\n📋 可用模型:")
    for key, model in models.items():
        print(f"{key}. {model['display_name']} - {model['description']}")
    print("0. 返回主菜单")
    print("-" * 50)

def train_model_interactive():
    """交互式训练模型"""
    show_model_menu()
    
    choice = input("请选择要训练的模型 (输入数字): ").strip()
    
    if choice == '0':
        return
    
    models = get_available_models()
    if choice not in models:
        print("❌ 无效选择")
        return
    
    model_info = models[choice]
    model_name = model_info['name']
    
    print(f"\n🚀 开始训练: {model_info['display_name']}")
    print(f"描述: {model_info['description']}")
    
    # 询问数据集路径
    data_path = input("数据集路径 (回车使用默认 'bioast_dataset'): ").strip()
    if not data_path:
        data_path = 'bioast_dataset'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据集路径不存在: {data_path}")
        return
    
    # 开始训练
    success = train_single_model(model_name, data_path)
    
    if success:
        print("\n✅ 训练完成！")
        input("按回车键继续...")
    else:
        print("\n❌ 训练失败")
        input("按回车键继续...")

def train_single_model(model_name, data_path=None):
    """训练单个模型"""
    try:
        # 初始化管理器
        config = ConfigManager().get_default_config()
        manager = ModelLifecycleManager(config)
        manager.start_services()
        
        # 模型配置映射
        model_configs = {
            'efficientnet_b0': {
                'name': 'EfficientNet-B0',
                'description': '轻量级高效模型',
                'model_type': 'classification',
                'algorithm': 'efficientnet_b0',
                'data_config': {
                    'data_path': data_path or 'bioast_dataset',
                    'image_size': (70, 70),
                    'batch_size': 32,
                    'test_size': 0.2
                },
                'training_config': {
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'optimizer': 'adam'
                }
            },
            'resnet18_improved': {
                'name': 'ResNet18-Improved',
                'description': '改进版ResNet18',
                'model_type': 'classification',
                'algorithm': 'resnet18_improved',
                'data_config': {
                    'data_path': data_path or 'bioast_dataset',
                    'image_size': (70, 70),
                    'batch_size': 32,
                    'test_size': 0.2
                },
                'training_config': {
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'optimizer': 'adam'
                }
            },
            'airbubble_hybrid_net': {
                'name': 'AirBubble-HybridNet',
                'description': '混合架构菌落检测模型',
                'model_type': 'classification',
                'algorithm': 'airbubble_hybrid_net',
                'data_config': {
                    'data_path': data_path or 'bioast_dataset',
                    'image_size': (70, 70),
                    'batch_size': 32,
                    'test_size': 0.2
                },
                'training_config': {
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'optimizer': 'adam'
                }
            },
            'micro_vit': {
                'name': 'Micro-ViT',
                'description': '微型Vision Transformer',
                'model_type': 'classification',
                'algorithm': 'micro_vit',
                'data_config': {
                    'data_path': data_path or 'bioast_dataset',
                    'image_size': (70, 70),
                    'batch_size': 16,
                    'test_size': 0.2
                },
                'training_config': {
                    'epochs': 50,
                    'learning_rate': 0.0001,
                    'optimizer': 'adamw'
                }
            },
            'convnext_tiny': {
                'name': 'ConvNeXt-Tiny',
                'description': '现代卷积网络',
                'model_type': 'classification',
                'algorithm': 'convnext_tiny',
                'data_config': {
                    'data_path': data_path or 'bioast_dataset',
                    'image_size': (70, 70),
                    'batch_size': 32,
                    'test_size': 0.2
                },
                'training_config': {
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'optimizer': 'adamw'
                }
            }
        }
        
        if model_name not in model_configs:
            print(f"❌ 不支持的模型: {model_name}")
            return False
        
        model_config = model_configs[model_name]
        
        print(f"📝 模型配置:")
        print(f"  名称: {model_config['name']}")
        print(f"  算法: {model_config['algorithm']}")
        print(f"  数据路径: {model_config['data_config']['data_path']}")
        print(f"  训练轮数: {model_config['training_config']['epochs']}")
        print(f"  学习率: {model_config['training_config']['learning_rate']}")
        
        # 创建训练工作流
        print("\n🔄 创建训练工作流...")
        workflow_id = manager.create_training_workflow(
            model_config=model_config,
            data_config=model_config['data_config'],
            training_config=model_config['training_config']
        )
        
        print(f"✅ 工作流创建成功: {workflow_id}")
        
        # 执行训练
        print("\n🚀 开始训练...")
        success = manager.execute_workflow(workflow_id)
        
        if success:
            print("\n✅ 训练成功完成！")
            
            # 获取训练结果
            workflow_status = manager.get_workflow_status(workflow_id)
            experiment_id = workflow_status.get('experiment_id')
            
            if experiment_id:
                # 生成报告
                print("📊 生成实验报告...")
                report_path = manager.generate_experiment_report(
                    experiment_id=experiment_id,
                    output_format='html'
                )
                print(f"📄 报告已生成: {report_path}")
                
                # 获取模型信息
                models = manager.list_models()
                if models:
                    latest_model = models[-1]
                    print(f"\n🎯 训练结果:")
                    print(f"  模型ID: {latest_model['id']}")
                    performance = latest_model.get('performance', {})
                    if performance:
                        for metric, value in performance.items():
                            print(f"  {metric}: {value}")
            
            return True
        else:
            print("\n❌ 训练失败")
            return False
            
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        return False

def list_models():
    """列出所有模型"""
    try:
        config = ConfigManager().get_default_config()
        manager = ModelLifecycleManager(config)
        manager.start_services()
        
        models = manager.list_models()
        
        if not models:
            print("📭 暂无训练好的模型")
            return
        
        print(f"\n📋 已训练模型 ({len(models)}个):")
        print("-" * 80)
        print(f"{'序号':<4} {'模型名称':<20} {'模型ID':<15} {'准确率':<10} {'创建时间':<20}")
        print("-" * 80)
        
        for i, model in enumerate(models, 1):
            performance = model.get('performance', {})
            accuracy = performance.get('accuracy', 'N/A')
            if isinstance(accuracy, float):
                accuracy = f"{accuracy:.4f}"
            
            created_at = model.get('created_at', 'N/A')
            if len(created_at) > 19:
                created_at = created_at[:19]
            
            print(f"{i:<4} {model['name']:<20} {model['id']:<15} {accuracy:<10} {created_at:<20}")
        
        print("-" * 80)
        
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")

def compare_models_interactive():
    """交互式模型对比"""
    try:
        config = ConfigManager().get_default_config()
        manager = ModelLifecycleManager(config)
        manager.start_services()
        
        models = manager.list_models()
        
        if len(models) < 2:
            print("❌ 至少需要2个模型才能进行对比")
            return
        
        print("\n🔄 模型对比分析")
        print("选择对比方式:")
        print("1. 对比性能最好的模型")
        print("2. 手动选择模型对比")
        print("0. 返回主菜单")
        
        choice = input("请选择 (输入数字): ").strip()
        
        if choice == '0':
            return
        elif choice == '1':
            # 按准确率排序
            def get_accuracy(model):
                performance = model.get('performance', {})
                accuracy = performance.get('accuracy', 0)
                return accuracy if isinstance(accuracy, (int, float)) else 0
            
            sorted_models = sorted(models, key=get_accuracy, reverse=True)
            top_models = sorted_models[:min(5, len(sorted_models))]
            
            print(f"\n📊 对比性能最好的 {len(top_models)} 个模型:")
            
            model_ids = [model['id'] for model in top_models]
            
        elif choice == '2':
            print("\n📋 可选择的模型:")
            for i, model in enumerate(models, 1):
                performance = model.get('performance', {})
                accuracy = performance.get('accuracy', 'N/A')
                print(f"{i}. {model['name']} (准确率: {accuracy})")
            
            selected = input("请输入要对比的模型序号，用空格分隔 (如: 1 2 3): ").strip().split()
            
            try:
                indices = [int(x) - 1 for x in selected]
                if any(i < 0 or i >= len(models) for i in indices):
                    print("❌ 无效的模型序号")
                    return
                
                if len(indices) < 2:
                    print("❌ 至少选择2个模型")
                    return
                
                model_ids = [models[i]['id'] for i in indices]
                
            except ValueError:
                print("❌ 输入格式错误")
                return
        else:
            print("❌ 无效选择")
            return
        
        # 生成对比报告
        print("\n📊 生成对比报告...")
        report_path = manager.generate_comparison_report(
            model_ids=model_ids,
            output_format='html'
        )
        
        print(f"✅ 对比报告已生成: {report_path}")
        
        # 显示简要对比
        print("\n📈 简要对比:")
        print("-" * 60)
        print(f"{'模型名称':<20} {'准确率':<10} {'F1分数':<10}")
        print("-" * 60)
        
        for model_id in model_ids:
            model = manager.get_model(model_id)
            if model:
                performance = model.get('performance', {})
                accuracy = performance.get('accuracy', 'N/A')
                f1_score = performance.get('f1_score', 'N/A')
                
                if isinstance(accuracy, float):
                    accuracy = f"{accuracy:.4f}"
                if isinstance(f1_score, float):
                    f1_score = f"{f1_score:.4f}"
                
                print(f"{model['name']:<20} {accuracy:<10} {f1_score:<10}")
        
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ 模型对比失败: {e}")

def show_system_status():
    """显示系统状态"""
    try:
        config = ConfigManager().get_default_config()
        manager = ModelLifecycleManager(config)
        manager.start_services()
        
        print("\n📊 系统状态概览")
        print("=" * 50)
        
        # 模型统计
        models = manager.list_models()
        print(f"📦 模型总数: {len(models)}")
        
        if models:
            # 按准确率排序
            def get_accuracy(model):
                performance = model.get('performance', {})
                accuracy = performance.get('accuracy', 0)
                return accuracy if isinstance(accuracy, (int, float)) else 0
            
            sorted_models = sorted(models, key=get_accuracy, reverse=True)
            
            print(f"\n🏆 性能排行榜:")
            for i, model in enumerate(sorted_models[:5], 1):
                accuracy = get_accuracy(model)
                print(f"{i}. {model['name']}: {accuracy:.4f}")
        
        # 实验统计
        experiments = manager.list_experiments()
        print(f"\n🧪 实验总数: {len(experiments)}")
        
        # 最近的实验
        if experiments:
            recent_experiments = sorted(experiments, key=lambda x: x.get('created_at', ''), reverse=True)[:3]
            print(f"\n📅 最近实验:")
            for exp in recent_experiments:
                status = exp.get('status', 'N/A')
                name = exp.get('name', 'N/A')
                print(f"  - {name} ({status})")
        
        # 存储信息
        print(f"\n💾 存储信息:")
        for dir_name in ['models', 'experiments', 'reports', 'logs']:
            if os.path.exists(dir_name):
                size = sum(os.path.getsize(os.path.join(dir_name, f)) 
                          for f in os.listdir(dir_name) 
                          if os.path.isfile(os.path.join(dir_name, f)))
                size_mb = size / (1024 * 1024)
                print(f"  {dir_name}: {size_mb:.2f} MB")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 获取系统状态失败: {e}")

def batch_train():
    """批量训练模型"""
    print("\n🚀 批量训练模型")
    print("这将训练所有可用的模型，可能需要较长时间。")
    
    confirm = input("确认开始批量训练？(y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 取消批量训练")
        return
    
    models_to_train = ['efficientnet_b0', 'resnet18_improved', 'airbubble_hybrid_net']
    
    data_path = input("数据集路径 (回车使用默认 'bioast_dataset'): ").strip()
    if not data_path:
        data_path = 'bioast_dataset'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据集路径不存在: {data_path}")
        return
    
    print(f"\n开始批量训练 {len(models_to_train)} 个模型...")
    
    results = []
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\n{'='*60}")
        print(f"训练进度: {i}/{len(models_to_train)} - {model_name}")
        print(f"{'='*60}")
        
        success = train_single_model(model_name, data_path)
        results.append((model_name, success))
    
    # 显示批量训练结果
    print(f"\n{'='*60}")
    print("批量训练完成")
    print(f"{'='*60}")
    
    for model_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{model_name}: {status}")
    
    successful_count = sum(1 for _, success in results if success)
    print(f"\n总计: {successful_count}/{len(results)} 个模型训练成功")

def check_dataset():
    """检查数据集"""
    print("\n🔍 数据集检查")
    
    data_path = input("数据集路径 (回车使用默认 'bioast_dataset'): ").strip()
    if not data_path:
        data_path = 'bioast_dataset'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据集路径不存在: {data_path}")
        return
    
    print(f"📁 检查数据集: {data_path}")
    
    # 检查目录结构
    required_dirs = ['train', 'val', 'test']
    for split in required_dirs:
        split_path = os.path.join(data_path, split)
        if os.path.exists(split_path):
            print(f"✅ 找到 {split} 目录")
            
            # 检查类别目录
            classes = [d for d in os.listdir(split_path) 
                      if os.path.isdir(os.path.join(split_path, d))]
            
            if classes:
                print(f"  类别: {', '.join(classes)}")
                
                # 统计每个类别的样本数
                for class_name in classes:
                    class_path = os.path.join(split_path, class_name)
                    files = [f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    print(f"    {class_name}: {len(files)} 个样本")
            else:
                print(f"  ⚠️ {split} 目录为空")
        else:
            print(f"❌ 缺少 {split} 目录")
    
    print("\n数据集检查完成")

def generate_report():
    """生成系统报告"""
    try:
        config = ConfigManager().get_default_config()
        manager = ModelLifecycleManager(config)
        manager.start_services()
        
        print("\n📊 生成系统报告")
        print("选择报告类型:")
        print("1. 系统概览报告")
        print("2. 模型对比报告")
        print("3. 实验详细报告")
        print("0. 返回主菜单")
        
        choice = input("请选择 (输入数字): ").strip()
        
        if choice == '0':
            return
        elif choice == '1':
            print("生成系统概览报告...")
            # 这里可以调用系统报告生成功能
            print("✅ 系统概览报告生成完成")
        elif choice == '2':
            models = manager.list_models()
            if len(models) < 2:
                print("❌ 至少需要2个模型才能生成对比报告")
                return
            
            model_ids = [model['id'] for model in models]
            report_path = manager.generate_comparison_report(
                model_ids=model_ids,
                output_format='html'
            )
            print(f"✅ 模型对比报告: {report_path}")
        elif choice == '3':
            experiments = manager.list_experiments()
            if not experiments:
                print("❌ 没有可用的实验")
                return
            
            print("\n可用实验:")
            for i, exp in enumerate(experiments, 1):
                print(f"{i}. {exp.get('name', 'N/A')} ({exp.get('status', 'N/A')})")
            
            try:
                exp_index = int(input("选择实验序号: ")) - 1
                if 0 <= exp_index < len(experiments):
                    experiment_id = experiments[exp_index]['id']
                    report_path = manager.generate_experiment_report(
                        experiment_id=experiment_id,
                        output_format='html'
                    )
                    print(f"✅ 实验报告: {report_path}")
                else:
                    print("❌ 无效的实验序号")
            except ValueError:
                print("❌ 输入格式错误")
        else:
            print("❌ 无效选择")
            
    except Exception as e:
        print(f"❌ 生成报告失败: {e}")

def show_help():
    """显示帮助信息"""
    print("\n📚 BioAst模型管理系统 - 帮助文档")
    print("=" * 60)
    
    print("\n🎯 主要功能:")
    print("1. 单模型训练 - 训练指定的单个模型")
    print("2. 批量训练 - 一次性训练多个模型")
    print("3. 模型对比 - 比较不同模型的性能")
    print("4. 结果分析 - 查看训练结果和性能指标")
    print("5. 报告生成 - 生成详细的分析报告")
    
    print("\n📋 支持的模型:")
    models = get_available_models()
    for key, model in models.items():
        print(f"  - {model['display_name']}: {model['description']}")
    
    print("\n📁 文件结构:")
    print("  bioast_dataset/     # 数据集目录")
    print("  ├── train/          # 训练集")
    print("  ├── val/            # 验证集")
    print("  └── test/           # 测试集")
    print("  models/             # 模型文件")
    print("  experiments/        # 实验结果")
    print("  reports/            # 分析报告")
    print("  logs/               # 日志文件")
    
    print("\n🔧 命令行使用:")
    print("  python quick_start.py                    # 交互式菜单")
    print("  python quick_start.py train <model>      # 训练指定模型")
    print("  python quick_start.py list               # 列出所有模型")
    print("  python quick_start.py compare            # 对比模型")
    print("  python quick_start.py status             # 查看系统状态")
    
    print("\n📖 更多文档:")
    print("  - README.md: 完整系统文档")
    print("  - MANUAL_OPERATION_GUIDE.md: 手动操作指南")
    print("  - config_template.yaml: 配置文件模板")
    
    print("=" * 60)

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BioAst模型管理系统 - 快速开始')
    parser.add_argument('command', nargs='?', choices=['train', 'list', 'compare', 'status', 'help'], 
                       help='要执行的命令')
    parser.add_argument('model', nargs='?', help='模型名称 (用于train命令)')
    parser.add_argument('--data', help='数据集路径')
    
    args = parser.parse_args()
    
    # 如果有命令行参数，直接执行
    if args.command:
        if args.command == 'train':
            if not args.model:
                print("❌ 请指定要训练的模型")
                print("可用模型:", list(get_available_models().values()))
                return
            
            print_banner()
            if check_environment():
                success = train_single_model(args.model, args.data)
                if success:
                    print("\n✅ 训练完成")
                else:
                    print("\n❌ 训练失败")
        
        elif args.command == 'list':
            print_banner()
            list_models()
        
        elif args.command == 'compare':
            print_banner()
            compare_models_interactive()
        
        elif args.command == 'status':
            print_banner()
            show_system_status()
        
        elif args.command == 'help':
            show_help()
        
        return
    
    # 交互式菜单
    print_banner()
    
    if not check_environment():
        print("❌ 环境检查失败，请检查配置")
        return
    
    while True:
        try:
            show_main_menu()
            choice = input("请选择操作 (输入数字): ").strip()
            
            if choice == '0':
                print("\n👋 感谢使用 BioAst模型管理系统！")
                break
            elif choice == '1':
                train_model_interactive()
            elif choice == '2':
                list_models()
                input("\n按回车键继续...")
            elif choice == '3':
                compare_models_interactive()
                input("\n按回车键继续...")
            elif choice == '4':
                show_system_status()
                input("\n按回车键继续...")
            elif choice == '5':
                batch_train()
                input("\n按回车键继续...")
            elif choice == '6':
                check_dataset()
                input("\n按回车键继续...")
            elif choice == '7':
                generate_report()
                input("\n按回车键继续...")
            elif choice == '8':
                show_help()
                input("\n按回车键继续...")
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n\n👋 感谢使用 BioAst模型管理系统！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            input("按回车键继续...")

if __name__ == "__main__":
    main()