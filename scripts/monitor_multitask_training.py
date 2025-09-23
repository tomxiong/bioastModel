"""
多任务训练监控脚本
专门监控生长模式和干扰因素的训练进度
"""

import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def monitor_training_progress(experiment_dir="experiments/multitask_grayscale_focused"):
    """监控多任务训练进度"""
    
    experiment_path = Path(experiment_dir)
    if not experiment_path.exists():
        print(f"实验目录不存在: {experiment_path}")
        return
        
    history_file = experiment_path / "training_history.json"
    
    print("🔍 多任务MIC训练监控")
    print("=" * 50)
    
    while True:
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if history:
                    latest = history[-1]
                    epoch = latest['epoch']
                    
                    print(f"\n📊 Epoch {epoch}/25 最新进度:")
                    print("-" * 30)
                    
                    # 训练指标
                    if 'train_main_acc' in latest:
                        print(f"训练 - 主分类: {latest['train_main_acc']:.2%}")
                        print(f"     生长模式: {latest['train_pattern_acc']:.2%}")
                        print(f"     干扰F1: {latest['train_interference_f1']:.2%}")
                    
                    # 验证指标  
                    if 'val_main_acc' in latest:
                        print(f"验证 - 主分类: {latest['val_main_acc']:.2%}")
                        print(f"     生长模式: {latest['val_pattern_acc']:.2%}")
                        print(f"     干扰F1: {latest['val_interference_f1']:.2%}")
                    
                    print(f"学习率: {latest.get('lr', 0):.6f}")
                    
                    # 检查是否有改进
                    if len(history) > 1:
                        prev = history[-2]
                        val_main_diff = latest.get('val_main_acc', 0) - prev.get('val_main_acc', 0)
                        val_pattern_diff = latest.get('val_pattern_acc', 0) - prev.get('val_pattern_acc', 0) 
                        val_interference_diff = latest.get('val_interference_f1', 0) - prev.get('val_interference_f1', 0)
                        
                        print(f"\n📈 相比上一轮变化:")
                        print(f"   主分类: {val_main_diff:+.2%}")
                        print(f"   生长模式: {val_pattern_diff:+.2%}")
                        print(f"   干扰因素: {val_interference_diff:+.2%}")
            
            else:
                print("⏳ 等待训练历史文件生成...")
                
        except Exception as e:
            print(f"监控出错: {e}")
        
        time.sleep(10)  # 每10秒检查一次

def analyze_task_performance(experiment_dir="experiments/multitask_grayscale_focused"):
    """分析各任务性能趋势"""
    
    experiment_path = Path(experiment_dir) 
    history_file = experiment_path / "training_history.json"
    
    if not history_file.exists():
        print("训练历史文件不存在")
        return
        
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("训练历史为空")
        return
        
    # 转换为DataFrame
    df = pd.DataFrame(history)
    
    print(f"\n📊 多任务性能分析 (共{len(df)}个epoch)")
    print("=" * 50)
    
    # 最新性能
    latest = df.iloc[-1]
    print(f"最新验证性能:")
    print(f"  主分类准确率: {latest.get('val_main_acc', 0):.2%}")
    print(f"  生长模式准确率: {latest.get('val_pattern_acc', 0):.2%}")
    print(f"  干扰因素F1: {latest.get('val_interference_f1', 0):.2%}")
    
    # 最佳性能
    best_main = df['val_main_acc'].max() if 'val_main_acc' in df else 0
    best_pattern = df['val_pattern_acc'].max() if 'val_pattern_acc' in df else 0
    best_interference = df['val_interference_f1'].max() if 'val_interference_f1' in df else 0
    
    print(f"\n最佳验证性能:")
    print(f"  主分类准确率: {best_main:.2%}")
    print(f"  生长模式准确率: {best_pattern:.2%}")
    print(f"  干扰因素F1: {best_interference:.2%}")
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # 主分类性能
    plt.subplot(2, 3, 1)
    if 'val_main_acc' in df:
        plt.plot(df['epoch'], df['val_main_acc'], 'b-', label='验证', linewidth=2)
        plt.plot(df['epoch'], df['train_main_acc'], 'b--', label='训练', alpha=0.7)
    plt.title('主分类准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 生长模式性能
    plt.subplot(2, 3, 2)
    if 'val_pattern_acc' in df:
        plt.plot(df['epoch'], df['val_pattern_acc'], 'g-', label='验证', linewidth=2)
        plt.plot(df['epoch'], df['train_pattern_acc'], 'g--', label='训练', alpha=0.7)
    plt.title('生长模式准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 干扰因素性能
    plt.subplot(2, 3, 3)
    if 'val_interference_f1' in df:
        plt.plot(df['epoch'], df['val_interference_f1'], 'r-', label='验证', linewidth=2)
        plt.plot(df['epoch'], df['train_interference_f1'], 'r--', label='训练', alpha=0.7)
    plt.title('干扰因素F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('F1分数')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失曲线
    plt.subplot(2, 3, 4)
    if 'train_loss' in df:
        plt.plot(df['epoch'], df['train_loss'], 'orange', label='训练损失', linewidth=2)
        plt.plot(df['epoch'], df['val_loss'], 'red', label='验证损失', linewidth=2)
    plt.title('损失函数')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 学习率变化
    plt.subplot(2, 3, 5)
    if 'lr' in df:
        plt.plot(df['epoch'], df['lr'], 'purple', linewidth=2)
    plt.title('学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.grid(True, alpha=0.3)
    
    # 综合性能对比
    plt.subplot(2, 3, 6)
    if all(col in df for col in ['val_main_acc', 'val_pattern_acc', 'val_interference_f1']):
        plt.plot(df['epoch'], df['val_main_acc'], 'b-', label='主分类', linewidth=2)
        plt.plot(df['epoch'], df['val_pattern_acc'], 'g-', label='生长模式', linewidth=2)
        plt.plot(df['epoch'], df['val_interference_f1'], 'r-', label='干扰因素', linewidth=2)
    plt.title('综合性能对比')
    plt.xlabel('Epoch')
    plt.ylabel('性能指标')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = experiment_path / "multitask_performance_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 性能分析图表已保存: {chart_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['monitor', 'analyze'], default='analyze',
                       help='运行模式: monitor(实时监控) 或 analyze(分析)')
    parser.add_argument('--experiment_dir', default='experiments/multitask_grayscale_focused',
                       help='实验目录')
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor_training_progress(args.experiment_dir)
    else:
        analyze_task_performance(args.experiment_dir)