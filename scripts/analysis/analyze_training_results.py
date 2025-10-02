#!/usr/bin/env python3
"""
分析训练结果和性能指标
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_training_results():
    """分析训练结果"""
    
    # 读取训练历史
    history_file = Path("experiments/enhanced_multilevel_mobilenetv3/enhanced_training_history.json")
    
    if not history_file.exists():
        print("❌ 训练历史文件不存在")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print("📊 训练结果分析")
    print("=" * 60)
    
    # 基本信息
    total_epochs = len(history)
    print(f"总训练轮数: {total_epochs}")
    
    # 获取第一轮和最后一轮的结果
    first_epoch = history[0]
    last_epoch = history[-1]
    
    print(f"\n🔄 训练进度对比:")
    print(f"第1轮 -> 第{total_epochs}轮")
    
    # 训练损失对比
    print(f"\n📉 训练损失变化:")
    first_train_loss = first_epoch['train_losses']['total']
    last_train_loss = last_epoch['train_losses']['total']
    train_improvement = ((first_train_loss - last_train_loss) / first_train_loss) * 100
    
    print(f"  总损失: {first_train_loss:.4f} -> {last_train_loss:.4f} (改善 {train_improvement:.1f}%)")
    
    for task in ['growth_level', 'growth_pattern', 'interference_factors', 'pores_detection']:
        first_loss = first_epoch['train_losses'][task]
        last_loss = last_epoch['train_losses'][task]
        improvement = ((first_loss - last_loss) / first_loss) * 100
        print(f"  {task}: {first_loss:.4f} -> {last_loss:.4f} (改善 {improvement:.1f}%)")
    
    # 验证损失对比
    print(f"\n📈 验证损失变化:")
    first_val_loss = first_epoch['val_losses']['total']
    last_val_loss = last_epoch['val_losses']['total']
    val_change = ((last_val_loss - first_val_loss) / first_val_loss) * 100
    
    print(f"  总损失: {first_val_loss:.4f} -> {last_val_loss:.4f} (变化 {val_change:+.1f}%)")
    
    for task in ['growth_level', 'growth_pattern', 'interference_factors', 'pores_detection']:
        first_loss = first_epoch['val_losses'][task]
        last_loss = last_epoch['val_losses'][task]
        change = ((last_loss - first_loss) / first_loss) * 100
        print(f"  {task}: {first_loss:.4f} -> {last_loss:.4f} (变化 {change:+.1f}%)")
    
    # 准确率对比
    print(f"\n🎯 验证准确率变化:")
    for task in ['growth_level', 'growth_pattern', 'interference_factors', 'pores_detection']:
        first_acc = first_epoch['val_accuracies'][task]
        last_acc = last_epoch['val_accuracies'][task]
        change = ((last_acc - first_acc) / first_acc) * 100
        print(f"  {task}: {first_acc:.3f} -> {last_acc:.3f} (变化 {change:+.1f}%)")
    
    # 找到最佳性能
    print(f"\n🏆 最佳性能:")
    
    # 最低训练损失
    min_train_loss_epoch = min(history, key=lambda x: x['train_losses']['total'])
    print(f"  最低训练损失: {min_train_loss_epoch['train_losses']['total']:.4f} (第{min_train_loss_epoch['epoch']}轮)")
    
    # 最低验证损失
    min_val_loss_epoch = min(history, key=lambda x: x['val_losses']['total'])
    print(f"  最低验证损失: {min_val_loss_epoch['val_losses']['total']:.4f} (第{min_val_loss_epoch['epoch']}轮)")
    
    # 最高总体准确率
    max_acc_epoch = max(history, key=lambda x: sum(x['val_accuracies'].values()) / len(x['val_accuracies']))
    avg_acc = sum(max_acc_epoch['val_accuracies'].values()) / len(max_acc_epoch['val_accuracies'])
    print(f"  最高平均准确率: {avg_acc:.3f} (第{max_acc_epoch['epoch']}轮)")
    
    # 各任务最高准确率
    for task in ['growth_level', 'growth_pattern', 'interference_factors', 'pores_detection']:
        max_task_acc_epoch = max(history, key=lambda x: x['val_accuracies'][task])
        max_acc = max_task_acc_epoch['val_accuracies'][task]
        print(f"  {task}最高准确率: {max_acc:.3f} (第{max_task_acc_epoch['epoch']}轮)")
    
    # 学习率变化
    print(f"\n📚 学习率变化:")
    first_lr = first_epoch['learning_rate']
    last_lr = last_epoch['learning_rate']
    print(f"  {first_lr:.6f} -> {last_lr:.6f}")
    
    # 训练时间统计
    print(f"\n⏱️  训练时间统计:")
    total_time = sum(epoch['epoch_time'] for epoch in history)
    avg_time = total_time / len(history)
    print(f"  总训练时间: {total_time:.1f}秒")
    print(f"  平均每轮时间: {avg_time:.2f}秒")
    
    # 收敛性分析
    print(f"\n📈 收敛性分析:")
    
    # 计算最后10轮的损失变化
    if len(history) >= 10:
        last_10_train_losses = [epoch['train_losses']['total'] for epoch in history[-10:]]
        last_10_val_losses = [epoch['val_losses']['total'] for epoch in history[-10:]]
        
        train_std = np.std(last_10_train_losses)
        val_std = np.std(last_10_val_losses)
        
        print(f"  最后10轮训练损失标准差: {train_std:.4f}")
        print(f"  最后10轮验证损失标准差: {val_std:.4f}")
        
        if train_std < 0.1 and val_std < 1.0:
            print("  ✅ 模型已收敛")
        else:
            print("  ⚠️  模型可能未完全收敛")
    
    print("\n" + "=" * 60)
    print("✅ 训练结果分析完成")

if __name__ == "__main__":
    analyze_training_results()