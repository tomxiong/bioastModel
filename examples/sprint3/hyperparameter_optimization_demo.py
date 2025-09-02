"""
FUA 超参数优化演示

展示如何使用 FUA 的超参数优化功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fua
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import tempfile
from pathlib import Path
import json


def create_sample_data(num_samples=1000, input_size=70*70*3, num_classes=2):
    """创建示例数据"""
    # 生成随机数据
    X = torch.randn(num_samples, input_size)
    # 生成随机标签
    y = torch.randint(0, num_classes, (num_samples,))
    
    # 创建数据加载器
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def create_simple_model(learning_rate=0.001, hidden_size=128, dropout=0.1, optimizer='adam'):
    """创建简单模型用于演示"""
    class SimpleModel(nn.Module):
        def __init__(self, input_size=70*70*3, hidden_size=128, num_classes=2, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.dropout1 = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.bn2 = nn.BatchNorm1d(hidden_size // 2)
            self.dropout2 = nn.Dropout(dropout)
            self.fc3 = nn.Linear(hidden_size // 2, num_classes)
            
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = self.fc3(x)
            return x
    
    model = SimpleModel(hidden_size=hidden_size, dropout=dropout)
    
    # 选择优化器
    if optimizer == 'adam':
        optimizer_instance = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer_instance = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer == 'adamw':
        optimizer_instance = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer_instance = optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, optimizer_instance


def train_model(model, train_loader, val_loader, params=None, trial=None, patience=10):
    """训练模型"""
    optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    best_epoch = 0
    
    # 简化的训练历史
    history = {'accuracy': [], 'loss': []}
    
    model.train()
    for epoch in range(50):  # 最多50个epoch
        # 训练
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        
        # 记录历史
        history['accuracy'].append(val_acc)
        history['loss'].append(val_loss)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        # 报告给Optuna
        if trial is not None:
            trial.report(val_acc, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if patience_counter >= patience:
            break
        
        # 打印进度
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    return model, history, best_epoch


def evaluate_model(model, val_loader):
    """评估模型"""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_acc = 100. * correct / total
    val_loss = val_loss / len(val_loader)
    
    return {
        'accuracy': val_acc / 100,  # 转换为0-1范围
        'loss': val_loss,
        'correct': correct,
        'total': total
    }


def demo_basic_optimization():
    """演示基本优化功能"""
    print("\n1. 基本超参数优化演示")
    print("-" * 50)
    
    # 创建示例数据
    train_loader, val_loader = create_sample_data(num_samples=500)
    
    # 定义搜索空间
    search_space = {
        'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
        'hidden_size': {'type': 'categorical', 'choices': [64, 128, 256]},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
        'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']}
    }
    
    # 创建优化器
    optimizer = fua.create_hyperparameter_optimizer(
        model_name='simple_model',
        search_space=search_space,
        n_trials=20,  # 减少试验次数以便演示
        n_jobs=1,
        direction='maximize',
        metric='accuracy'
    )
    
    # 执行优化
    print("开始优化...")
    result = optimizer.optimize(
        train_data=train_loader,
        val_data=val_loader,
        model_factory=lambda **params: create_simple_model(**params)[0],
        train_fn=train_model,
        eval_fn=evaluate_model,
        save_study=True
    )
    
    # 显示结果
    print(f"\n优化完成！")
    print(f"最佳准确率: {result.best_score:.4f}")
    print(f"最佳参数: {result.best_params}")
    print(f"总试验次数: {result.total_trials}")
    print(f"优化时间: {result.optimization_time:.2f}秒")
    
    return optimizer


def demo_cv_optimization():
    """演示交叉验证优化"""
    print("\n2. 交叉验证优化演示")
    print("-" * 50)
    
    # 创建示例数据
    X = torch.randn(500, 70*70*3)
    y = torch.randint(0, 2, (500,))
    
    # 定义搜索空间
    search_space = {
        'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
        'hidden_size': {'type': 'categorical', 'choices': [64, 128]},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.3}
    }
    
    # 创建交叉验证优化器
    cv_optimizer = fua.create_cv_optimizer(
        model_name='simple_model_cv',
        search_space=search_space,
        n_trials=10,  # 减少试验次数
        cv_folds=3,
        direction='maximize',
        metric='accuracy'
    )
    
    # 执行优化
    print("开始交叉验证优化...")
    result = cv_optimizer.optimize(
        data=X,
        labels=y,
        model_factory=lambda **params: create_simple_model(**params)[0],
        train_fn=train_model,
        eval_fn=evaluate_model,
        save_study=True
    )
    
    # 显示结果
    print(f"\n交叉验证优化完成！")
    print(f"最佳准确率: {result.best_score:.4f}")
    print(f"最佳参数: {result.best_params}")
    
    return cv_optimizer


def demo_advanced_features():
    """演示高级功能"""
    print("\n3. 高级功能演示")
    print("-" * 50)
    
    # 创建示例数据
    train_loader, val_loader = create_sample_data(num_samples=300)
    
    # 更复杂的搜索空间
    search_space = {
        'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-1, 'log': True},
        'hidden_size': {'type': 'int', 'low': 32, 'high': 512, 'log': True},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.7},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
        'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'adamw']},
        'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True}
    }
    
    # 创建优化器（使用不同的采样器和剪枝器）
    optimizer = fua.create_hyperparameter_optimizer(
        model_name='advanced_model',
        search_space=search_space,
        n_trials=15,
        sampler='random',  # 使用随机采样
        pruner='halving',  # 使用逐半剪枝
        direction='maximize',
        metric='accuracy'
    )
    
    # 执行优化
    print("开始高级优化...")
    result = optimizer.optimize(
        train_data=train_loader,
        val_data=val_loader,
        model_factory=lambda **params: create_simple_model(**params)[0],
        train_fn=train_model,
        eval_fn=evaluate_model,
        save_study=True
    )
    
    # 分析结果
    analysis = optimizer.analyze_results()
    
    print(f"\n优化完成！")
    print(f"最佳准确率: {result.best_score:.4f}")
    print(f"成功率: {analysis['basic_stats']['success_rate']:.1f}%")
    print(f"平均分数: {analysis['performance_stats']['mean_score']:.4f}")
    print(f"参数重要性: {analysis['parameter_importance']}")
    
    if analysis['convergence_analysis']:
        print(f"收敛分析: {analysis['convergence_analysis']}")
    
    # 显示试验历史
    print(f"\n试验历史（前5个）:")
    df = optimizer.get_trial_results_df()
    print(df.head().to_string())
    
    return optimizer


def demo_default_search_spaces():
    """演示默认搜索空间"""
    print("\n4. 默认搜索空间演示")
    print("-" * 50)
    
    # 获取不同模型的默认搜索空间
    model_types = ['resnet', 'efficientnet', 'mobilenet', 'vit']
    
    for model_type in model_types:
        search_space = fua.get_default_search_space(model_type)
        print(f"\n{model_type.upper()} 搜索空间:")
        for param, config in search_space.items():
            print(f"  - {param}: {config}")
    
    # 使用默认搜索空间进行优化
    print(f"\n使用ResNet默认搜索空间进行优化...")
    train_loader, val_loader = create_sample_data(num_samples=200)
    
    search_space = fua.get_default_search_space('resnet')
    
    optimizer = fua.create_hyperparameter_optimizer(
        model_name='resnet_default',
        search_space=search_space,
        n_trials=10,
        direction='maximize'
    )
    
    result = optimizer.optimize(
        train_data=train_loader,
        val_data=val_loader,
        model_factory=lambda **params: create_simple_model(**params)[0],
        train_fn=train_model,
        eval_fn=evaluate_model
    )
    
    print(f"最佳准确率: {result.best_score:.4f}")


def main():
    """主函数"""
    print("FUA 超参数优化功能演示")
    print("=" * 60)
    
    # 检查可用性
    if not fua.OPTIMIZATION_AVAILABLE:
        print("❌ 优化模块不可用")
        return
    
    # 导入optuna（用于演示中的TrialPruned）
    global optuna
    import optuna
    
    # 运行各项演示
    demo_basic_optimization()
    demo_cv_optimization()
    demo_advanced_features()
    demo_default_search_spaces()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("\n主要功能:")
    print("- ✓ 自动超参数搜索")
    print("- ✓ 多种采样策略（TPE、随机）")
    print("- ✓ 智能剪枝（中位数、逐半）")
    print("- ✓ 交叉验证支持")
    print("- ✓ 并行优化")
    print("- ✓ 结果分析和可视化")
    print("- ✓ 预定义搜索空间")
    print("- ✓ 早停机制")
    print("- ✓ 完整的试验历史记录")
    print("=" * 60)


if __name__ == "__main__":
    main()