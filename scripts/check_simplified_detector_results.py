"""
检查简化版气孔检测器训练结果
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector

def load_checkpoint(checkpoint_path):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(f"✅ 成功加载检查点: {checkpoint_path}")
        return checkpoint
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return None

def check_model_parameters(model):
    """检查模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 模型参数总数: {total_params:,}")
    print(f"📊 可训练参数数: {trainable_params:,}")
    
    # 检查每层参数
    print("\n📋 模型结构:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  - {name}: {params:,} 参数")

def plot_training_history(history):
    """绘制训练历史"""
    if not history:
        print("❌ 没有训练历史数据")
        return
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    plt.figure(figsize=(12, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, history['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    plt.plot(epochs, history['val_acc'], 'r-', label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制学习率曲线
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['learning_rates'], 'g-')
    plt.title('学习率调度')
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 绘制训练/验证差距
    plt.subplot(2, 2, 4)
    gaps = np.array(history['train_acc']) - np.array(history['val_acc'])
    plt.plot(epochs, gaps, 'purple')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('训练/验证准确率差距')
    plt.xlabel('轮次')
    plt.ylabel('差距 (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/simplified_airbubble_detector/training_history.png', dpi=300)
    plt.close()
    
    print(f"✅ 训练历史图表已保存到: experiments/simplified_airbubble_detector/training_history.png")

def check_test_results(results_path):
    """检查测试结果"""
    if not os.path.exists(results_path):
        print(f"❌ 测试结果文件不存在: {results_path}")
        return
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("\n📊 测试结果:")
        print(f"  - 准确率: {results['test_accuracy']:.2f}%")
        print(f"  - 精确率: {results['test_precision']:.2f}%")
        print(f"  - 召回率: {results['test_recall']:.2f}%")
        print(f"  - F1分数: {results['test_f1']:.2f}%")
        
        if 'improvement_over_original' in results:
            print(f"  - 相比原始模型改进: +{results['improvement_over_original']:.2f}%")
        
        if 'target_achievement' in results:
            status = "✅ 已达成" if results['target_achievement'] else "❌ 未达成"
            print(f"  - 目标达成状态: {status}")
        
        if 'best_val_accuracy' in results:
            print(f"  - 最佳验证准确率: {results['best_val_accuracy']:.2f}%")
    
    except Exception as e:
        print(f"❌ 读取测试结果失败: {e}")

def main():
    """主函数"""
    print("🔍 检查简化版气孔检测器训练结果")
    print("=" * 60)
    
    # 检查点路径
    checkpoint_path = "experiments/simplified_airbubble_detector/simplified_airbubble_best.pth"
    results_path = "experiments/simplified_airbubble_detector/simplified_test_results.json"
    
    # 加载检查点
    checkpoint = load_checkpoint(checkpoint_path)
    if not checkpoint:
        return
    
    # 创建模型
    model = SimplifiedAirBubbleDetector()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 检查模型参数
    check_model_parameters(model)
    
    # 绘制训练历史
    if 'train_history' in checkpoint:
        plot_training_history(checkpoint['train_history'])
    
    # 检查测试结果
    check_test_results(results_path)
    
    print("\n✅ 检查完成")

if __name__ == "__main__":
    main()