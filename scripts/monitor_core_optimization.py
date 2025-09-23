"""
核心误判问题优化训练监控脚本
专门监控边界样本准确率改善情况
"""

import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

def monitor_core_optimization_training(experiment_dir="experiments/core_boundary_optimization"):
    """监控核心误判优化训练进度"""
    
    experiment_path = Path(experiment_dir)
    if not experiment_path.exists():
        print(f"实验目录不存在: {experiment_path}")
        return
        
    history_file = experiment_path / "training_history.json"
    
    print("🎯 核心误判优化训练监控")
    print("=" * 60)
    print("重点关注指标:")
    print("  • 边界样本准确率 (阳性弱特征样本)")
    print("  • 主分类准确率")
    print("  • 生长模式准确率")
    print("=" * 60)
    
    while True:
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if history:
                    latest = history[-1]
                    epoch = latest['epoch']
                    
                    print(f"\\n🎯 Epoch {epoch}/20 最新优化进度:")
                    print("-" * 40)
                    
                    # 训练指标 
                    if 'train_main_acc' in latest:
                        print(f"训练 - 主分类: {latest['train_main_acc']:.2%}")
                        print(f"     边界样本: {latest.get('train_boundary_sample_acc', 0):.2%}")
                        print(f"     生长模式: {latest['train_pattern_acc']:.2%}")
                    
                    # 验证指标
                    if 'val_main_acc' in latest:
                        print(f"验证 - 主分类: {latest['val_main_acc']:.2%}")
                        print(f"     边界样本: {latest.get('val_boundary_sample_acc', 0):.2%}")
                        print(f"     生长模式: {latest['val_pattern_acc']:.2%}")
                    
                    print(f"学习率 - 主干: {latest.get('lr', 0):.6f}")
                    
                    # 边界样本改善分析
                    if len(history) > 1:
                        prev = history[-2]
                        boundary_improvement = latest.get('val_boundary_sample_acc', 0) - prev.get('val_boundary_sample_acc', 0)
                        main_improvement = latest.get('val_main_acc', 0) - prev.get('val_main_acc', 0)
                        
                        print(f"\\n📈 相比上轮变化:")
                        print(f"   边界样本: {boundary_improvement:+.2%} {'🔥' if boundary_improvement > 0 else '📉' if boundary_improvement < 0 else '➡️'}")
                        print(f"   主分类: {main_improvement:+.2%}")
                    
                    # 综合性能评估
                    boundary_acc = latest.get('val_boundary_sample_acc', 0)
                    main_acc = latest.get('val_main_acc', 0)
                    
                    if boundary_acc > 0.85:  # 边界样本准确率超过85%
                        print(f"🎉 边界样本优化良好! (准确率: {boundary_acc:.2%})")
                    elif boundary_acc > 0.75:
                        print(f"🟡 边界样本持续改善中... (准确率: {boundary_acc:.2%})")
                    else:
                        print(f"🔴 边界样本仍需优化 (准确率: {boundary_acc:.2%})")
            
            else:
                print("⏳ 等待核心优化训练历史文件生成...")
                
        except Exception as e:
            print(f"监控出错: {e}")
        
        time.sleep(15)  # 每15秒检查一次

def analyze_boundary_optimization_progress(experiment_dir="experiments/core_boundary_optimization"):
    """分析边界样本优化进展"""
    
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
    
    print(f"\\n🎯 边界样本优化分析 (共{len(df)}个epoch)")
    print("=" * 60)
    
    # 最新性能
    if len(df) > 0:
        latest = df.iloc[-1]
        print(f"最新验证性能:")
        print(f"  主分类准确率: {latest.get('val_main_acc', 0):.2%}")
        print(f"  边界样本准确率: {latest.get('val_boundary_sample_acc', 0):.2%}")
        print(f"  生长模式准确率: {latest.get('val_pattern_acc', 0):.2%}")
        
        # 最佳性能
        best_main = df['val_main_acc'].max() if 'val_main_acc' in df else 0
        best_boundary = df['val_boundary_sample_acc'].max() if 'val_boundary_sample_acc' in df else 0
        best_pattern = df['val_pattern_acc'].max() if 'val_pattern_acc' in df else 0
        
        print(f"\\n最佳验证性能:")
        print(f"  主分类准确率: {best_main:.2%}")
        print(f"  边界样本准确率: {best_boundary:.2%}")
        print(f"  生长模式准确率: {best_pattern:.2%}")
        
        # 边界样本改善趋势
        if 'val_boundary_sample_acc' in df and len(df) > 1:
            boundary_trend = np.polyfit(range(len(df)), df['val_boundary_sample_acc'], 1)[0]
            print(f"\\n边界样本改善趋势:")
            if boundary_trend > 0.001:
                print(f"  📈 持续改善 (斜率: {boundary_trend:+.4f})")
            elif boundary_trend > -0.001:
                print(f"  ➡️ 趋于稳定 (斜率: {boundary_trend:+.4f})")
            else:
                print(f"  📉 需要调整 (斜率: {boundary_trend:+.4f})")
    
    # 创建可视化
    if len(df) > 0:
        plt.figure(figsize=(15, 10))
        
        # 边界样本准确率 vs 主分类准确率
        plt.subplot(2, 3, 1)
        if 'val_boundary_sample_acc' in df:
            plt.plot(df['epoch'], df['val_boundary_sample_acc'], 'r-', linewidth=3, label='边界样本')
            plt.plot(df['epoch'], df['train_boundary_sample_acc'], 'r--', alpha=0.7, label='边界样本(训练)')
        if 'val_main_acc' in df:
            plt.plot(df['epoch'], df['val_main_acc'], 'b-', linewidth=2, label='主分类')
        plt.title('边界样本 vs 主分类准确率', fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 生长模式准确率
        plt.subplot(2, 3, 2)
        if 'val_pattern_acc' in df:
            plt.plot(df['epoch'], df['val_pattern_acc'], 'g-', linewidth=2, label='验证')
            plt.plot(df['epoch'], df['train_pattern_acc'], 'g--', alpha=0.7, label='训练')
        plt.title('生长模式准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失曲线
        plt.subplot(2, 3, 3)
        if 'train_loss' in df:
            plt.plot(df['epoch'], df['train_loss'], 'orange', linewidth=2, label='训练损失')
            plt.plot(df['epoch'], df['val_loss'], 'red', linewidth=2, label='验证损失')
        plt.title('损失函数')
        plt.xlabel('Epoch')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学习率变化
        plt.subplot(2, 3, 4)
        if 'lr' in df:
            plt.plot(df['epoch'], df['lr'], 'purple', linewidth=2)
        plt.title('学习率变化')
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.grid(True, alpha=0.3)
        
        # 综合性能对比
        plt.subplot(2, 3, 5)
        if all(col in df for col in ['val_main_acc', 'val_boundary_sample_acc', 'val_pattern_acc']):
            plt.plot(df['epoch'], df['val_main_acc'], 'b-', linewidth=2, label='主分类')
            plt.plot(df['epoch'], df['val_boundary_sample_acc'], 'r-', linewidth=3, label='边界样本')
            plt.plot(df['epoch'], df['val_pattern_acc'], 'g-', linewidth=2, label='生长模式')
        plt.title('综合性能对比')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 边界样本改善速度
        plt.subplot(2, 3, 6)
        if 'val_boundary_sample_acc' in df and len(df) > 1:
            boundary_diff = df['val_boundary_sample_acc'].diff()
            plt.plot(df['epoch'][1:], boundary_diff[1:], 'ro-', linewidth=2, markersize=4)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.title('边界样本改善速度')
        plt.xlabel('Epoch')  
        plt.ylabel('准确率变化')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = experiment_path / "boundary_optimization_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"\\n📊 边界优化分析图表已保存: {chart_path}")
        
        return df
    
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['monitor', 'analyze'], default='analyze',
                       help='运行模式: monitor(实时监控) 或 analyze(分析)')
    parser.add_argument('--experiment_dir', default='experiments/core_boundary_optimization',
                       help='实验目录')
    
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor_core_optimization_training(args.experiment_dir)
    else:
        analyze_boundary_optimization_progress(args.experiment_dir)