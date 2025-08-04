"""
比较原始AirBubbleHybridNet模型和简化模型的输出
"""

import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.airbubble_hybrid_net import create_airbubble_hybrid_net
from scripts.create_simple_airbubble_model import SimpleAirBubbleModel
from core.data_loader import MICDataLoader, create_data_loaders

def load_original_model():
    """加载原始AirBubbleHybridNet模型"""
    try:
        # 查找最新的检查点文件
        checkpoint_path = Path("experiments/experiment_20250803_115344/airbubble_hybrid_net/best_model.pth")
        
        if not checkpoint_path.exists():
            logging.error(f"检查点文件不存在: {checkpoint_path}")
            return None
        
        # 创建模型实例
        model = create_airbubble_hybrid_net(num_classes=2, model_size='base')
        model.eval()
        
        # 加载模型权重
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        
        # 检查权重键名是否匹配
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 尝试直接加载
            state_dict = checkpoint
        
        # 处理base_model前缀问题
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('base_model.'):
                new_key = key[len('base_model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 尝试加载处理后的权重
        model.load_state_dict(new_state_dict)
        
        logging.info("原始模型加载成功")
        return model
    except Exception as e:
        logging.error(f"加载原始模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_simple_model():
    """加载简化模型"""
    try:
        model = SimpleAirBubbleModel(num_classes=2)
        model.eval()
        logging.info("简化模型加载成功")
        return model
    except Exception as e:
        logging.error(f"加载简化模型失败: {e}")
        return None

def compare_models_on_random_data(original_model, simple_model, num_samples=100):
    """在随机数据上比较两个模型的输出"""
    if original_model is None or simple_model is None:
        return
    
    # 创建随机输入数据
    random_inputs = torch.randn(num_samples, 3, 70, 70)
    
    # 获取原始模型的输出
    original_outputs = []
    with torch.no_grad():
        for i in range(num_samples):
            output = original_model(random_inputs[i:i+1])
            if isinstance(output, dict):
                # 如果输出是字典，获取分类结果
                output = output['classification']
            original_outputs.append(output.numpy())
    
    original_outputs = np.vstack(original_outputs)
    original_preds = np.argmax(original_outputs, axis=1)
    
    # 获取简化模型的输出
    simple_outputs = []
    with torch.no_grad():
        for i in range(num_samples):
            output = simple_model(random_inputs[i:i+1])
            simple_outputs.append(output.numpy())
    
    simple_outputs = np.vstack(simple_outputs)
    simple_preds = np.argmax(simple_outputs, axis=1)
    
    # 计算一致性
    agreement = np.mean(original_preds == simple_preds)
    logging.info(f"模型预测一致性: {agreement:.4f} ({int(agreement * num_samples)}/{num_samples})")
    
    # 计算混淆矩阵
    cm = confusion_matrix(original_preds, simple_preds)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('原始模型 vs 简化模型 混淆矩阵')
    plt.colorbar()
    
    classes = ['类别0', '类别1']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # 在混淆矩阵中添加文本
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('原始模型预测')
    plt.xlabel('简化模型预测')
    
    # 保存图像
    reports_dir = Path("reports/model_comparison")
    reports_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(reports_dir / "airbubble_model_comparison.png")
    
    # 生成报告
    report = f"""# AirBubbleHybridNet模型比较报告

## 随机数据测试结果

- 测试样本数: {num_samples}
- 模型预测一致性: {agreement:.4f} ({int(agreement * num_samples)}/{num_samples})

## 混淆矩阵

![混淆矩阵](airbubble_model_comparison.png)

## 结论

简化模型与原始模型的预测一致性为{agreement:.2%}。

"""
    
    # 保存报告
    with open(reports_dir / "airbubble_model_comparison.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    logging.info(f"比较报告已保存至: {reports_dir / 'airbubble_model_comparison.md'}")

def compare_models_on_test_data(original_model, simple_model):
    """在测试数据上比较两个模型的性能"""
    if original_model is None or simple_model is None:
        return
    
    try:
        # 创建数据加载器
        data_loader = MICDataLoader()
        
        # 获取测试数据加载器
        _, _, test_loader = create_data_loaders(data_loader, batch_size=32)
        
        if test_loader is None:
            logging.error("无法加载测试数据")
            return
        
        # 收集原始模型的预测和真实标签
        original_preds = []
        simple_preds = []
        true_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                # 原始模型预测
                original_outputs = original_model(images)
                if isinstance(original_outputs, dict):
                    original_outputs = original_outputs['classification']
                original_preds.extend(torch.argmax(original_outputs, dim=1).numpy())
                
                # 简化模型预测
                simple_outputs = simple_model(images)
                simple_preds.extend(torch.argmax(simple_outputs, dim=1).numpy())
                
                # 真实标签
                true_labels.extend(labels.numpy())
        
        # 转换为numpy数组
        original_preds = np.array(original_preds)
        simple_preds = np.array(simple_preds)
        true_labels = np.array(true_labels)
        
        # 计算性能指标
        original_accuracy = accuracy_score(true_labels, original_preds)
        simple_accuracy = accuracy_score(true_labels, simple_preds)
        
        original_f1 = f1_score(true_labels, original_preds, average='weighted')
        simple_f1 = f1_score(true_labels, simple_preds, average='weighted')
        
        # 计算一致性
        agreement = np.mean(original_preds == simple_preds)
        
        # 生成报告
        reports_dir = Path("reports/model_comparison")
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        report = f"""# AirBubbleHybridNet模型在测试数据上的比较

## 性能指标

| 模型 | 准确率 | F1分数 |
|-----|-------|-------|
| 原始模型 | {original_accuracy:.4f} | {original_f1:.4f} |
| 简化模型 | {simple_accuracy:.4f} | {simple_f1:.4f} |

## 模型一致性

- 预测一致性: {agreement:.4f} ({int(agreement * len(true_labels))}/{len(true_labels)})

## 结论

1. 原始模型准确率: {original_accuracy:.2%}
2. 简化模型准确率: {simple_accuracy:.2%}
3. 准确率差异: {abs(original_accuracy - simple_accuracy):.2%}
4. 简化模型与原始模型的预测一致性为{agreement:.2%}

"""
        
        # 保存报告
        with open(reports_dir / "airbubble_model_test_comparison.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        logging.info(f"测试数据比较报告已保存至: {reports_dir / 'airbubble_model_test_comparison.md'}")
        
    except Exception as e:
        logging.error(f"在测试数据上比较模型失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    # 加载原始模型
    original_model = load_original_model()
    
    # 加载简化模型
    simple_model = load_simple_model()
    
    # 在随机数据上比较模型
    compare_models_on_random_data(original_model, simple_model)
    
    # 在测试数据上比较模型
    try:
        compare_models_on_test_data(original_model, simple_model)
    except Exception as e:
        logging.error(f"测试数据比较失败，可能是测试数据加载器不可用: {e}")
        logging.info("仅使用随机数据进行比较")

if __name__ == "__main__":
    main()