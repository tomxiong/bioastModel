"""
修复字体问题的模型对比可视化图表生成脚本
生成EfficientNet-B0和ResNet-18 Improved的综合对比图表
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import json
import base64
from io import BytesIO
import seaborn as sns

# 设置字体和样式 - 使用英文避免字体问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

def load_model_data():
    """加载两个模型的评估数据"""
    
    # EfficientNet-B0 数据
    with open('experiments/experiment_20250802_140818/efficientnet_b0/training_history.json', 'r') as f:
        efficientnet_history = json.load(f)
    
    # ResNet-18 Improved 数据
    with open('experiments/experiment_20250802_164948/resnet18_improved/training_history.json', 'r') as f:
        resnet_history = json.load(f)
    
    with open('experiments/experiment_20250802_164948/resnet18_improved/test_results.json', 'r') as f:
        resnet_test = json.load(f)
    
    # EfficientNet-B0 测试结果（从分类报告中提取）
    efficientnet_test = {
        'accuracy': 0.9754,
        'auc': 0.9969,
        'sensitivity': 0.9774,
        'specificity': 0.9731,
        'precision': 0.975,  # 加权平均
        'recall': 0.975,     # 加权平均
        'f1_score': 0.975    # 加权平均
    }
    
    return {
        'efficientnet': {
            'history': efficientnet_history,
            'test': efficientnet_test,
            'params': 1.56,  # Million parameters
            'epochs': 16
        },
        'resnet': {
            'history': resnet_history,
            'test': resnet_test,
            'params': 11.26,  # Million parameters
            'epochs': 21
        }
    }

def create_performance_radar_chart(data):
    """创建性能雷达图对比"""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # 性能指标
    metrics = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score']
    
    # 数据准备
    efficientnet_values = [
        data['efficientnet']['test']['accuracy'],
        data['efficientnet']['test']['auc'],
        data['efficientnet']['test']['sensitivity'],
        data['efficientnet']['test']['specificity'],
        data['efficientnet']['test']['precision'],
        data['efficientnet']['test']['f1_score']
    ]
    
    resnet_values = [
        data['resnet']['test']['accuracy'],
        data['resnet']['test']['auc'],
        data['resnet']['test']['sensitivity'],
        data['resnet']['test']['specificity'],
        data['resnet']['test']['precision'],
        data['resnet']['test']['f1_score']
    ]
    
    # 角度设置
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    efficientnet_values += efficientnet_values[:1]  # 闭合图形
    resnet_values += resnet_values[:1]
    angles += angles[:1]
    
    # 绘制雷达图
    ax.plot(angles, efficientnet_values, 'o-', linewidth=2, label='EfficientNet-B0', color='#2E8B57')
    ax.fill(angles, efficientnet_values, alpha=0.25, color='#2E8B57')
    
    ax.plot(angles, resnet_values, 'o-', linewidth=2, label='ResNet-18 Improved', color='#4169E1')
    ax.fill(angles, resnet_values, alpha=0.25, color='#4169E1')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.94, 1.0)
    ax.set_title('Model Performance Comparison\n(Higher is Better)', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def create_training_history_comparison(data):
    """创建训练历史对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 训练损失对比
    ax1.plot(data['efficientnet']['history']['train_loss'], 'o-', label='EfficientNet-B0', color='#2E8B57', linewidth=2)
    ax1.plot(data['resnet']['history']['train_loss'], 's-', label='ResNet-18 Improved', color='#4169E1', linewidth=2)
    ax1.set_title('Training Loss Comparison', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证损失对比
    ax2.plot(data['efficientnet']['history']['val_loss'], 'o-', label='EfficientNet-B0', color='#2E8B57', linewidth=2)
    ax2.plot(data['resnet']['history']['val_loss'], 's-', label='ResNet-18 Improved', color='#4169E1', linewidth=2)
    ax2.set_title('Validation Loss Comparison', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 训练准确率对比
    ax3.plot(data['efficientnet']['history']['train_acc'], 'o-', label='EfficientNet-B0', color='#2E8B57', linewidth=2)
    ax3.plot(data['resnet']['history']['train_acc'], 's-', label='ResNet-18 Improved', color='#4169E1', linewidth=2)
    ax3.set_title('Training Accuracy Comparison', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 验证准确率对比
    ax4.plot(data['efficientnet']['history']['val_acc'], 'o-', label='EfficientNet-B0', color='#2E8B57', linewidth=2)
    ax4.plot(data['resnet']['history']['val_acc'], 's-', label='ResNet-18 Improved', color='#4169E1', linewidth=2)
    ax4.set_title('Validation Accuracy Comparison', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_efficiency_performance_scatter(data):
    """创建效率-性能散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算效率指标（准确率/参数量）
    efficientnet_efficiency = data['efficientnet']['test']['accuracy'] / data['efficientnet']['params']
    resnet_efficiency = data['resnet']['test']['accuracy'] / data['resnet']['params']
    
    # 散点图
    ax.scatter(data['efficientnet']['params'], data['efficientnet']['test']['accuracy'], 
              s=300, alpha=0.7, color='#2E8B57', label='EfficientNet-B0', edgecolors='black', linewidth=2)
    ax.scatter(data['resnet']['params'], data['resnet']['test']['accuracy'], 
              s=300, alpha=0.7, color='#4169E1', label='ResNet-18 Improved', edgecolors='black', linewidth=2)
    
    # 添加效率线
    x_range = np.linspace(0, 12, 100)
    for eff in [0.6, 0.7, 0.8]:
        ax.plot(x_range, eff * x_range, '--', alpha=0.3, color='gray')
        ax.text(10, eff * 10, f'Efficiency = {eff:.1f}', alpha=0.5, fontsize=10)
    
    # 标注点
    ax.annotate(f'EfficientNet-B0\n{data["efficientnet"]["params"]:.2f}M params\n{data["efficientnet"]["test"]["accuracy"]:.4f} acc\nEff: {efficientnet_efficiency:.3f}', 
                xy=(data['efficientnet']['params'], data['efficientnet']['test']['accuracy']), 
                xytext=(3, 0.976), fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=1.5))
    
    ax.annotate(f'ResNet-18 Improved\n{data["resnet"]["params"]:.2f}M params\n{data["resnet"]["test"]["accuracy"]:.4f} acc\nEff: {resnet_efficiency:.3f}', 
                xy=(data['resnet']['params'], data['resnet']['test']['accuracy']), 
                xytext=(8, 0.972), fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='#4169E1', lw=1.5))
    
    ax.set_xlabel('Model Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontweight='bold')
    ax.set_title('Efficiency vs Performance Trade-off\n(Higher accuracy with fewer parameters is better)', 
                fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.970, 0.980)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_comparison(data):
    """创建混淆矩阵对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # EfficientNet-B0 混淆矩阵 (从分类报告推算)
    efficientnet_cm = np.array([[470, 13], [13, 563]])
    
    # ResNet-18 Improved 混淆矩阵
    resnet_cm = np.array(data['resnet']['test']['confusion_matrix'])
    
    # 绘制热力图
    sns.heatmap(efficientnet_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax1.set_title('EfficientNet-B0\nConfusion Matrix', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    sns.heatmap(resnet_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax2.set_title('ResNet-18 Improved\nConfusion Matrix', fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def create_convergence_analysis(data):
    """创建收敛分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 损失收敛分析
    efficientnet_val_loss = data['efficientnet']['history']['val_loss']
    resnet_val_loss = data['resnet']['history']['val_loss']
    
    ax1.plot(efficientnet_val_loss, 'o-', label='EfficientNet-B0', color='#2E8B57', linewidth=2)
    ax1.plot(resnet_val_loss, 's-', label='ResNet-18 Improved', color='#4169E1', linewidth=2)
    
    # 标记最佳点
    efficientnet_best_epoch = np.argmin(efficientnet_val_loss)
    resnet_best_epoch = np.argmin(resnet_val_loss)
    
    ax1.scatter(efficientnet_best_epoch, efficientnet_val_loss[efficientnet_best_epoch], 
               color='red', s=100, zorder=5, label=f'EfficientNet Best (Epoch {efficientnet_best_epoch+1})')
    ax1.scatter(resnet_best_epoch, resnet_val_loss[resnet_best_epoch], 
               color='orange', s=100, zorder=5, label=f'ResNet Best (Epoch {resnet_best_epoch+1})')
    
    ax1.set_title('Validation Loss Convergence', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 学习率对比
    efficientnet_lr = data['efficientnet']['history']['lr']
    resnet_lr = data['resnet']['history']['lr']
    
    ax2.plot(efficientnet_lr, 'o-', label='EfficientNet-B0 (Cosine Annealing)', color='#2E8B57', linewidth=2)
    ax2.plot(resnet_lr, 's-', label='ResNet-18 Improved (Step Decay)', color='#4169E1', linewidth=2)
    ax2.set_title('Learning Rate Schedule Comparison', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_resnet_training_curves(data):
    """为ResNet-18创建单独的训练曲线图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(data['resnet']['history']['train_loss']) + 1)
    
    # 训练损失
    ax1.plot(epochs, data['resnet']['history']['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, data['resnet']['history']['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('ResNet-18 Improved - Loss Curves', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 训练准确率
    ax2.plot(epochs, data['resnet']['history']['train_acc'], 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, data['resnet']['history']['val_acc'], 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_title('ResNet-18 Improved - Accuracy Curves', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 学习率变化
    ax3.plot(epochs, data['resnet']['history']['lr'], 'g-', linewidth=2, label='Learning Rate')
    ax3.set_title('ResNet-18 Improved - Learning Rate Schedule', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 训练稳定性分析
    val_loss_smooth = np.convolve(data['resnet']['history']['val_loss'], np.ones(3)/3, mode='valid')
    ax4.plot(epochs, data['resnet']['history']['val_loss'], 'r-', alpha=0.5, label='Validation Loss')
    ax4.plot(epochs[1:-1], val_loss_smooth, 'r-', linewidth=2, label='Smoothed Val Loss')
    ax4.set_title('ResNet-18 Improved - Training Stability', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    """将matplotlib图表转换为base64编码"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return image_base64

def generate_all_comparison_charts():
    """生成所有对比图表并返回base64编码"""
    print("Loading model data...")
    data = load_model_data()
    
    charts = {}
    
    print("Generating performance radar chart...")
    charts['radar'] = fig_to_base64(create_performance_radar_chart(data))
    
    print("Generating training history comparison...")
    charts['training_history'] = fig_to_base64(create_training_history_comparison(data))
    
    print("Generating efficiency scatter plot...")
    charts['efficiency'] = fig_to_base64(create_efficiency_performance_scatter(data))
    
    print("Generating confusion matrix comparison...")
    charts['confusion_matrix'] = fig_to_base64(create_confusion_matrix_comparison(data))
    
    print("Generating convergence analysis...")
    charts['convergence'] = fig_to_base64(create_convergence_analysis(data))
    
    print("Generating ResNet training curves...")
    charts['resnet_training'] = fig_to_base64(create_resnet_training_curves(data))
    
    return charts, data

if __name__ == "__main__":
    charts, data = generate_all_comparison_charts()
    print("All comparison charts generated successfully!")
    print(f"Generated {len(charts)} charts")
    for chart_name in charts.keys():
        print(f"  - {chart_name}")