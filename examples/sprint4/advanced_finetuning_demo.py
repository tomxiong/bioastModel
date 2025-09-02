"""
高级微调演示

展示 FUA 高级微调功能的使用方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

# 导入 FUA 组件
import fua
from fua.finetuning.advanced_finetuner import create_advanced_finetuner, get_default_finetuning_config


def create_simple_model(num_classes=2):
    """创建简单的测试模型"""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, num_classes)
    )


def create_sample_data(num_samples=200, image_size=70):
    """创建示例数据"""
    # 生成随机图像数据
    X = torch.randn(num_samples, 3, image_size, image_size)
    
    # 生成简单标签（基于像素和的模式）
    y = (X.sum(dim=[1, 2, 3]) > 0).long()
    
    # 创建数据集
    dataset = TensorDataset(X, y)
    
    # 划分训练集和验证集
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset


def demo_basic_finetuning():
    """演示基本微调功能"""
    print("=" * 60)
    print("演示 1: 基本微调功能")
    print("=" * 60)
    
    # 创建模型和数据
    model = create_simple_model()
    train_data, val_data = create_sample_data()
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    
    # 获取默认配置
    config = get_default_finetuning_config('resnet')
    config['num_epochs'] = 5  # 减少轮数用于演示
    
    # 创建微调器
    finetuner = create_advanced_finetuner(model, config)
    
    # 执行微调
    print("开始微调...")
    result = finetuner.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir='./demo_outputs/basic_finetuning'
    )
    
    # 打印结果
    print("\n微调结果:")
    print(f"最佳准确率: {result.best_metric:.4f}")
    print(f"最佳轮数: {result.best_epoch}")
    print(f"训练时间: {result.training_time:.2f} 秒")
    
    return result


def demo_architecture_modification():
    """演示架构修改功能"""
    print("\n" + "=" * 60)
    print("演示 2: 架构修改功能")
    print("=" * 60)
    
    # 创建模型
    model = create_simple_model()
    print("原始模型结构:")
    print(model)
    
    # 创建架构修改器
    from fua.finetuning.architecture_modifier import create_architecture_modifier
    modifier = create_architecture_modifier(model)
    
    # 添加 Dropout 层
    print("\n添加 Dropout 层...")
    modifier.add_layer(
        parent_name='1',
        layer_type='dropout',
        layer_config={'p': 0.5},
        insert_position='after'
    )
    
    # 添加批归一化层
    print("添加批归一化层...")
    modifier.add_layer(
        parent_name='3',
        layer_type='batchnorm',
        layer_config={'num_features': 16},
        insert_position='before'
    )
    
    # 冻结早期层
    print("冻结早期层...")
    modifier.freeze_layers(['0', '1'])
    
    # 调整最后一层的维度
    print("调整输出层维度...")
    modifier.adjust_layer_dimensions(
        layer_name='9',
        new_dimensions={'out_features': 5}
    )
    
    print("\n修改后的模型结构:")
    print(model)
    
    # 获取修改摘要
    summary = modifier.get_modification_summary()
    print(f"\n修改摘要:")
    print(f"总修改数: {summary['total_modifications']}")
    print(f"修改类型: {summary['modifications_by_type']}")
    
    return model


def demo_custom_loss_function():
    """演示自定义损失函数"""
    print("\n" + "=" * 60)
    print("演示 3: 自定义损失函数")
    print("=" * 60)
    
    # 测试不同的损失函数
    from fua.finetuning.loss_function_factory import create_loss, get_classification_loss_configs
    
    # 获取预定义配置
    loss_configs = get_classification_loss_configs()
    
    print("可用的分类损失配置:")
    for name, config in loss_configs.items():
        print(f"  - {name}: {config}")
    
    # 创建不同的损失函数
    print("\n创建损失函数:")
    
    # 标准交叉熵
    ce_loss = create_loss('cross_entropy')
    print(f"交叉熵损失: {ce_loss}")
    
    # 带标签平滑的损失
    smooth_loss = create_loss({
        'type': 'label_smoothing',
        'params': {
            'num_classes': 2,
            'smoothing': 0.1
        }
    })
    print(f"标签平滑损失: {smooth_loss}")
    
    # Focal 损失
    focal_loss = create_loss({
        'type': 'focal',
        'params': {
            'alpha': 0.25,
            'gamma': 2.0
        }
    })
    print(f"Focal 损失: {focal_loss}")
    
    # 组合损失
    combined_loss = create_loss({
        'type': 'combined',
        'params': {
            'losses': [
                {'type': 'cross_entropy'},
                {'type': 'focal', 'params': {'gamma': 1.0}}
            ],
            'weights': [0.7, 0.3]
        }
    })
    print(f"组合损失: {combined_loss}")
    
    return [ce_loss, smooth_loss, focal_loss, combined_loss]


def demo_layered_lr_scheduler():
    """演示分层学习率调度器"""
    print("\n" + "=" * 60)
    print("演示 4: 分层学习率调度器")
    print("=" * 60)
    
    # 创建模型
    model = create_simple_model()
    
    # 创建分层学习率调度器
    from fua.finetuning.layered_lr_scheduler import create_layered_scheduler, get_resnet_layer_groups
    
    # 获取预定义层组
    layer_groups = get_resnet_layer_groups()
    print("预定义层组配置:")
    for group in layer_groups:
        print(f"  - {group['name']}: lr_multiplier={group['lr_multiplier']}")
    
    # 创建调度器
    scheduler = create_layered_scheduler(
        model=model,
        base_lr=0.001,
        layer_groups=layer_groups,
        scheduler_type='cosine',
        scheduler_params={'T_max': 100}
    )
    
    print(f"\n调度器类型: {scheduler.scheduler_type}")
    print(f"基础学习率: {scheduler.base_lr}")
    
    # 获取各层组的学习率
    lr_by_group = scheduler.get_lr_by_group()
    print("\n各层组学习率:")
    for group_name, lr in lr_by_group.items():
        print(f"  - {group_name}: {lr:.6f}")
    
    # 模拟几个步骤
    print("\n模拟学习率调度...")
    for i in range(5):
        scheduler.step()
        current_lrs = scheduler.get_lr()
        print(f"Step {i+1}: LRs = {[f'{lr:.6f}' for lr in current_lrs]}")
    
    return scheduler


def demo_finetuning_monitor():
    """演示微调监控器"""
    print("\n" + "=" * 60)
    print("演示 5: 微调监控器")
    print("=" * 60)
    
    # 创建模型
    model = create_simple_model()
    
    # 创建监控器
    from fua.finetuning.finetuning_monitor import create_finetuning_monitor
    monitor = create_finetuning_monitor(
        model=model,
        log_dir='./demo_outputs/monitoring'
    )
    
    # 模拟训练步骤
    print("模拟训练监控...")
    for step in range(10):
        # 模拟指标
        metrics = {
            'train_loss': 2.0 - step * 0.1,
            'train_accuracy': 0.5 + step * 0.03,
            'lr': 0.001 * (0.95 ** step)
        }
        
        monitor.update_metrics(metrics, step)
        monitor.log_gradients(step)
        monitor.log_activations(step)
        monitor.log_model_stats(step)
        
        if step % 5 == 0:
            print(f"Step {step}: Loss={metrics['train_loss']:.4f}, Acc={metrics['train_accuracy']:.4f}")
    
    # 生成报告
    report = monitor.generate_report()
    print(f"\n监控报告:")
    print(f"训练步数: {report['training_summary']['total_steps']}")
    print(f"模型参数数: {report['gradient_summary']['total_parameters']}")
    
    # 清理
    monitor.cleanup()
    
    return monitor


def demo_complete_workflow():
    """演示完整工作流程"""
    print("\n" + "=" * 60)
    print("演示 6: 完整微调工作流程")
    print("=" * 60)
    
    # 1. 创建模型
    model = create_simple_model()
    print("1. 创建模型完成")
    
    # 2. 应用架构修改
    from fua.finetuning.architecture_modifier import create_architecture_modifier
    modifier = create_architecture_modifier(model)
    
    # 添加 dropout
    modifier.add_layer(
        parent_name='1',
        layer_type='dropout',
        layer_config={'p': 0.3},
        insert_position='after'
    )
    print("2. 架构修改完成")
    
    # 3. 准备数据
    train_data, val_data = create_sample_data(num_samples=500)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    print("3. 数据准备完成")
    
    # 4. 配置微调
    config = get_default_finetuning_config('resnet')
    config.update({
        'num_epochs': 3,
        'loss_config': {
            'type': 'label_smoothing',
            'params': {'num_classes': 2, 'smoothing': 0.1}
        },
        'architecture_modifications': [
            {
                'type': 'freeze_layers',
                'layer_names': ['0']
            }
        ]
    })
    print("4. 微调配置完成")
    
    # 5. 执行微调
    finetuner = create_advanced_finetuner(model, config)
    print("5. 开始微调...")
    
    result = finetuner.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir='./demo_outputs/complete_workflow'
    )
    
    # 6. 显示结果
    print("\n6. 微调结果:")
    summary = result.get_summary()
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    
    print("\n完整工作流程演示完成！")
    
    return result


def main():
    """主函数"""
    print("FUA 高级微调功能演示")
    print("=" * 60)
    
    # 创建输出目录
    Path('./demo_outputs').mkdir(exist_ok=True)
    
    # 运行各个演示
    try:
        # 演示 1: 基本微调
        result1 = demo_basic_finetuning()
        
        # 演示 2: 架构修改
        model2 = demo_architecture_modification()
        
        # 演示 3: 自定义损失函数
        losses = demo_custom_loss_function()
        
        # 演示 4: 分层学习率
        scheduler = demo_layered_lr_scheduler()
        
        # 演示 5: 监控器
        monitor = demo_finetuning_monitor()
        
        # 演示 6: 完整工作流程
        result6 = demo_complete_workflow()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        print("\n输出文件保存在: ./demo_outputs/")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()