"""
快速验证简化版气孔检测器模型的有效性
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector, generate_synthetic_data

def load_model(checkpoint_path):
    """加载模型"""
    print(f"🔍 加载模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model = SimplifiedAirBubbleDetector()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ 成功加载模型")
        return model
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def generate_test_samples(num_samples=100):
    """生成测试样本"""
    print(f"🔍 生成{num_samples}个测试样本...")
    
    X, y = generate_synthetic_data(num_samples)
    print(f"✅ 成功生成测试样本: X形状={X.shape}, y形状={y.shape}")
    
    return X, y

def evaluate_model(model, X, y):
    """评估模型性能"""
    print("🔍 评估模型性能...")
    
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32))
        _, preds = torch.max(outputs, 1)
        
    accuracy = torch.sum(preds == torch.tensor(y)).item() / len(y) * 100
    
    # 计算混淆矩阵
    confusion = np.zeros((2, 2), dtype=int)
    for i in range(len(y)):
        confusion[y[i]][preds[i].item()] += 1
    
    # 计算精确率、召回率和F1分数
    tp = confusion[1][1]
    fp = confusion[0][1]
    fn = confusion[1][0]
    
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"📊 准确率: {accuracy:.2f}%")
    print(f"📊 精确率: {precision:.2f}%")
    print(f"📊 召回率: {recall:.2f}%")
    print(f"📊 F1分数: {f1:.2f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion': confusion,
        'predictions': preds.numpy(),
        'true_labels': y
    }

def visualize_results(results, X, save_path):
    """可视化结果"""
    print("🔍 可视化结果...")
    
    # 选择一些样本进行可视化
    n_samples = min(10, len(X))
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        
        # 将特征向量重塑为图像
        img = X[idx].reshape(10, 10)
        
        # 绘制图像
        plt.imshow(img, cmap='viridis')
        
        # 添加标题
        true_label = results['true_labels'][idx]
        pred_label = results['predictions'][idx]
        title = f"真实: {true_label}, 预测: {pred_label}"
        color = 'green' if true_label == pred_label else 'red'
        plt.title(title, color=color)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ 可视化结果已保存到: {save_path}")

def plot_confusion_matrix(confusion, save_path):
    """绘制混淆矩阵"""
    print("🔍 绘制混淆矩阵...")
    
    plt.figure(figsize=(8, 6))
    
    # 绘制混淆矩阵
    plt.imshow(confusion, cmap='Blues')
    
    # 添加数值标签
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion[i, j]), ha='center', va='center', color='black')
    
    # 添加标签
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks([0, 1], ['无气孔', '有气孔'])
    plt.yticks([0, 1], ['无气孔', '有气孔'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ 混淆矩阵已保存到: {save_path}")

def main():
    """主函数"""
    print("🔍 快速验证简化版气孔检测器模型的有效性")
    print("=" * 60)
    
    # 路径设置
    checkpoint_path = "experiments/simplified_airbubble_detector/simplified_airbubble_best.pth"
    results_path = "experiments/simplified_airbubble_detector/validation_results.png"
    confusion_path = "experiments/simplified_airbubble_detector/confusion_matrix.png"
    
    # 加载模型
    model = load_model(checkpoint_path)
    if not model:
        return
    
    # 生成测试样本
    X, y = generate_test_samples(num_samples=500)
    
    # 评估模型性能
    results = evaluate_model(model, X, y)
    
    # 可视化结果
    visualize_results(results, X, results_path)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(results['confusion'], confusion_path)
    
    print("\n📋 验证结果摘要:")
    print(f"  - 准确率: {results['accuracy']:.2f}%")
    print(f"  - 精确率: {results['precision']:.2f}%")
    print(f"  - 召回率: {results['recall']:.2f}%")
    print(f"  - F1分数: {results['f1']:.2f}%")
    
    print("\n✅ 验证完成")

if __name__ == "__main__":
    main()