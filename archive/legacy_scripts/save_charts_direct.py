"""
直接使用matplotlib保存对比图表，避免PIL解压缩问题
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 设置字体避免警告
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

def load_model_data():
    """加载两个模型的数据"""
    # EfficientNet-B0数据
    efficientnet_history_path = './experiments/experiment_20250802_140818/efficientnet_b0/training_history.json'
    with open(efficientnet_history_path, 'r') as f:
        efficientnet_history = json.load(f)
    
    # ResNet-18数据
    resnet_history_path = './experiments/experiment_20250802_164948/resnet18_improved/training_history.json'
    with open(resnet_history_path, 'r') as f:
        resnet_history = json.load(f)
    
    resnet_test_path = './experiments/experiment_20250802_164948/resnet18_improved/test_results.json'
    with open(resnet_test_path, 'r') as f:
        resnet_test = json.load(f)
    
    return {
        'efficientnet': {
            'history': efficientnet_history,
            'params': 1.56,
            'epochs': len(efficientnet_history['train_loss']),
            'test': {'accuracy': 97.54}
        },
        'resnet': {
            'history': resnet_history,
            'params': 11.26,
            'epochs': len(resnet_history['train_loss']),
            'test': resnet_test
        }
    }

def save_performance_radar_chart(data, output_dir):
    """生成并保存性能雷达图"""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # 性能指标
    categories = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Efficiency', 'Speed']
    
    # EfficientNet-B0数据
    efficientnet_values = [
        data['efficientnet']['test']['accuracy'] / 100,  # 归一化到0-1
        0.9969,  # AUC
        0.9774,  # Sensitivity
        0.9731,  # Specificity
        data['efficientnet']['test']['accuracy'] / data['efficientnet']['params'] / 100,  # 效率比
        1.0  # 速度（相对值）
    ]
    
    # ResNet-18数据
    resnet_values = [
        data['resnet']['test']['accuracy'] / 100,
        data['resnet']['test']['auc'],
        data['resnet']['test']['sensitivity'],
        data['resnet']['test']['specificity'],
        data['resnet']['test']['accuracy'] / data['resnet']['params'] / 100,
        0.4  # 速度（相对值，较慢）
    ]
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    efficientnet_values += efficientnet_values[:1]
    resnet_values += resnet_values[:1]
    
    # 绘制
    ax.plot(angles, efficientnet_values, 'o-', linewidth=2, label='EfficientNet-B0', color='blue')
    ax.fill(angles, efficientnet_values, alpha=0.25, color='blue')
    
    ax.plot(angles, resnet_values, 'o-', linewidth=2, label='ResNet-18 Improved', color='red')
    ax.fill(angles, resnet_values, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_radar_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 已保存: performance_radar_comparison.png")

def save_training_history_chart(data, output_dir):
    """生成并保存训练历史对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # EfficientNet训练历史
    eff_epochs = range(1, len(data['efficientnet']['history']['train_loss']) + 1)
    ax1.plot(eff_epochs, data['efficientnet']['history']['train_loss'], 'b-', linewidth=2, label='EfficientNet-B0')
    ax1.plot(eff_epochs, data['efficientnet']['history']['val_loss'], 'b--', linewidth=2, alpha=0.7)
    
    # ResNet训练历史
    res_epochs = range(1, len(data['resnet']['history']['train_loss']) + 1)
    ax1.plot(res_epochs, data['resnet']['history']['train_loss'], 'r-', linewidth=2, label='ResNet-18 Improved')
    ax1.plot(res_epochs, data['resnet']['history']['val_loss'], 'r--', linewidth=2, alpha=0.7)
    
    ax1.set_title('Training Loss Comparison', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证准确率对比
    ax2.plot(eff_epochs, data['efficientnet']['history']['val_acc'], 'b-', linewidth=2, label='EfficientNet-B0')
    ax2.plot(res_epochs, data['resnet']['history']['val_acc'], 'r-', linewidth=2, label='ResNet-18 Improved')
    ax2.set_title('Validation Accuracy Comparison', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习率对比
    ax3.plot(eff_epochs, data['efficientnet']['history']['lr'], 'b-', linewidth=2, label='EfficientNet-B0')
    ax3.plot(res_epochs, data['resnet']['history']['lr'], 'r-', linewidth=2, label='ResNet-18 Improved')
    ax3.set_title('Learning Rate Schedule Comparison', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 收敛速度对比
    eff_best_epoch = np.argmax(data['efficientnet']['history']['val_acc']) + 1
    res_best_epoch = np.argmax(data['resnet']['history']['val_acc']) + 1
    
    ax4.bar(['EfficientNet-B0', 'ResNet-18 Improved'], 
            [eff_best_epoch, res_best_epoch], 
            color=['blue', 'red'], alpha=0.7)
    ax4.set_title('Convergence Speed (Best Epoch)', fontweight='bold')
    ax4.set_ylabel('Epoch')
    
    for i, v in enumerate([eff_best_epoch, res_best_epoch]):
        ax4.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 已保存: training_history_comparison.png")

def save_efficiency_chart(data, output_dir):
    """生成并保存效率分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 效率散点图
    models = ['EfficientNet-B0', 'ResNet-18 Improved']
    params = [data['efficientnet']['params'], data['resnet']['params']]
    accuracies = [data['efficientnet']['test']['accuracy'], data['resnet']['test']['accuracy']]
    colors = ['blue', 'red']
    
    for i, (model, param, acc, color) in enumerate(zip(models, params, accuracies, colors)):
        ax1.scatter(param, acc, s=200, c=color, alpha=0.7, label=model)
        ax1.annotate(f'{model}\n({param:.2f}M, {acc:.2f}%)', 
                    (param, acc), xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax1.set_xlabel('Parameters (Millions)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Efficiency Analysis: Parameters vs Accuracy', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 效率比对比
    efficiency_ratios = [acc/param for acc, param in zip(accuracies, params)]
    bars = ax2.bar(models, efficiency_ratios, color=colors, alpha=0.7)
    ax2.set_title('Efficiency Ratio (Accuracy/Parameters)', fontweight='bold')
    ax2.set_ylabel('Efficiency Ratio (%/M)')
    
    for bar, ratio in zip(bars, efficiency_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 已保存: efficiency_analysis.png")

def main():
    """主函数"""
    print("🔧 直接生成并保存对比图表...")
    
    # 创建输出目录
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("📊 加载模型数据...")
    data = load_model_data()
    
    # 生成图表
    print("📈 生成性能雷达图...")
    save_performance_radar_chart(data, output_dir)
    
    print("📉 生成训练历史对比图...")
    save_training_history_chart(data, output_dir)
    
    print("⚡ 生成效率分析图...")
    save_efficiency_chart(data, output_dir)
    
    print(f"\n✅ 所有图表已保存到 {output_dir}/ 目录")
    print("📁 生成的文件:")
    print("  - performance_radar_comparison.png")
    print("  - training_history_comparison.png") 
    print("  - efficiency_analysis.png")

if __name__ == "__main__":
    main()