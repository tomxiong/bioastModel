"""
监控简化版气孔检测器训练进度
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import time

def find_latest_experiment():
    """查找最新的实验目录"""
    base_dir = Path("experiments")
    if not base_dir.exists():
        print(f"❌ 实验目录不存在: {base_dir}")
        return None
    
    # 查找所有simplified_airbubble_detector目录
    model_dirs = list(base_dir.glob("**/simplified_airbubble_detector"))
    
    if not model_dirs:
        print("❌ 未找到simplified_airbubble_detector的实验目录")
        return None
    
    # 按修改时间排序
    latest_dir = max(model_dirs, key=os.path.getmtime)
    print(f"✅ 找到最新实验目录: {latest_dir}")
    return latest_dir

def load_history(experiment_dir):
    """加载训练历史"""
    history_files = list(Path(experiment_dir).glob("*history.json"))
    
    if not history_files:
        print(f"❌ 未找到训练历史文件")
        return None
    
    history_file = history_files[0]
    print(f"📄 加载训练历史: {history_file}")
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        print(f"❌ 加载训练历史失败: {e}")
        return None

def plot_training_curves(history):
    """绘制训练曲线"""
    if not history:
        return
    
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 绘制F1分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('F1 Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_training_summary(history):
    """打印训练摘要"""
    if not history:
        return
    
    epochs = len(history['train_loss'])
    
    print("\n📊 训练摘要:")
    print(f"已完成 {epochs} 个epoch")
    
    if epochs > 0:
        print(f"最新训练损失: {history['train_loss'][-1]:.4f}")
        print(f"最新验证损失: {history['val_loss'][-1]:.4f}")
        print(f"最新训练准确率: {history['train_acc'][-1]*100:.2f}%")
        print(f"最新验证准确率: {history['val_acc'][-1]*100:.2f}%")
        print(f"最新验证F1分数: {history['val_f1'][-1]*100:.2f}%")
        
        # 找出最佳验证准确率
        best_epoch = np.argmax(history['val_acc'])
        print(f"\n🏆 最佳验证准确率: {history['val_acc'][best_epoch]*100:.2f}% (Epoch {best_epoch+1})")
        print(f"   对应训练准确率: {history['train_acc'][best_epoch]*100:.2f}%")
        print(f"   对应验证F1分数: {history['val_f1'][best_epoch]*100:.2f}%")

def monitor_training(interval=5):
    """监控训练进度"""
    print("🔍 开始监控简化版气孔检测器训练进度")
    print("=" * 50)
    
    try:
        while True:
            experiment_dir = find_latest_experiment()
            if experiment_dir:
                history = load_history(experiment_dir)
                if history:
                    print_training_summary(history)
                    plot_training_curves(history)
            
            print(f"\n⏱️ 等待 {interval} 秒后刷新...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n✋ 监控已停止")

if __name__ == "__main__":
    monitor_training()