"""
验证ONNX模型在真实数据上的性能
使用bioast_dataset中的真实数据比较原始PyTorch模型和转换后的ONNX模型的性能
"""

import os
import sys
import logging
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

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

def load_onnx_model(model_name):
    """加载ONNX模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        tuple: (model, input_size) ONNX模型和输入尺寸
    """
    try:
        onnx_path = Path(f"onnx_models/{model_name}.onnx")
        
        if not onnx_path.exists():
            logging.error(f"ONNX模型文件不存在: {onnx_path}")
            return None, None
        
        # 创建ONNX运行时会话
        session = ort.InferenceSession(str(onnx_path))
        
        # 获取输入和输出名称
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 获取输入形状
        input_shape = session.get_inputs()[0].shape
        logging.info(f"ONNX模型输入形状: {input_shape}")
        
        # 提取输入尺寸（假设输入格式为 [batch, channels, height, width]）
        if len(input_shape) >= 4:
            input_size = input_shape[2] if isinstance(input_shape[2], int) else 224  # 默认224
        else:
            input_size = 70  # 默认70
        
        logging.info(f"ONNX模型 {model_name} 加载成功，输入尺寸: {input_size}")
        
        # 返回一个函数，该函数接受输入张量并返回模型输出
        def onnx_model(input_tensor):
            # 确保输入是numpy数组
            if isinstance(input_tensor, torch.Tensor):
                input_np = input_tensor.numpy()
            else:
                input_np = input_tensor
            
            # 运行ONNX模型
            output = session.run([output_name], {input_name: input_np})
            
            # 转换为torch张量
            return torch.tensor(output[0])
        
        return onnx_model, input_size
    except Exception as e:
        logging.error(f"加载ONNX模型 {model_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_real_test_data(dataset_path="C:/Users/tomxiong/codebuddy/bioastModel/bioast_dataset", batch_size=32, input_size=70):
    """加载真实测试数据"""
    dataset_path = Path(dataset_path)
    
    # 检查数据集路径是否存在
    if not dataset_path.exists():
        logging.error(f"数据集路径不存在: {dataset_path}")
        return None
    
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # 调整为模型输入大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据
    test_images = []
    test_labels = []
    
    # 加载阳性样本
    positive_test_dir = dataset_path / "positive" / "test"
    if positive_test_dir.exists():
        for img_path in positive_test_dir.glob("*.png"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                test_images.append(img_tensor)
                test_labels.append(1)  # 阳性标签为1
            except Exception as e:
                logging.error(f"加载图像失败: {img_path}, 错误: {e}")
    
    # 加载阴性样本
    negative_test_dir = dataset_path / "negative" / "test"
    if negative_test_dir.exists():
        for img_path in negative_test_dir.glob("*.png"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                test_images.append(img_tensor)
                test_labels.append(0)  # 阴性标签为0
            except Exception as e:
                logging.error(f"加载图像失败: {img_path}, 错误: {e}")
    
    # 检查是否加载了测试数据
    if len(test_images) == 0:
        logging.error("未加载任何测试数据")
        return None
    
    logging.info(f"加载了 {len(test_images)} 个测试样本")
    
    # 创建数据加载器
    test_dataset = list(zip(test_images, test_labels))
    
    # 创建批次
    batches = []
    for i in range(0, len(test_dataset), batch_size):
        batch = test_dataset[i:i+batch_size]
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        batches.append((images, labels))
    
    return batches

def evaluate_models_on_real_data(original_model, simple_model, onnx_model, test_batches):
    """在真实数据上比较模型的性能"""
    # 收集预测结果
    original_preds = []
    original_probs = []
    simple_preds = []
    simple_probs = []
    onnx_preds = []
    onnx_probs = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in test_batches:
            # 原始模型预测
            if original_model is not None:
                original_outputs = original_model(images)
                if isinstance(original_outputs, dict):
                    original_outputs = original_outputs['classification']
                original_probs.extend(torch.softmax(original_outputs, dim=1)[:, 1].numpy())
                original_preds.extend(torch.argmax(original_outputs, dim=1).numpy())
            
            # 简化模型预测
            if simple_model is not None:
                simple_outputs = simple_model(images)
                simple_probs.extend(torch.softmax(simple_outputs, dim=1)[:, 1].numpy())
                simple_preds.extend(torch.argmax(simple_outputs, dim=1).numpy())
            
            # ONNX模型预测 - 需要逐个处理，因为ONNX模型只支持batch_size=1
            if onnx_model is not None:
                batch_onnx_outputs = []
                for i in range(images.shape[0]):
                    single_image = images[i:i+1]  # 保持4D形状 [1, 3, 70, 70]
                    single_output = onnx_model(single_image)
                    batch_onnx_outputs.append(single_output)
                onnx_outputs = torch.cat(batch_onnx_outputs, dim=0)
                onnx_probs.extend(torch.softmax(onnx_outputs, dim=1)[:, 1].numpy())
                onnx_preds.extend(torch.argmax(onnx_outputs, dim=1).numpy())
            
            # 真实标签
            true_labels.extend(labels.numpy())
    
    # 转换为numpy数组
    true_labels = np.array(true_labels)
    
    # 计算性能指标
    results = {}
    
    if original_model is not None and len(original_preds) > 0:
        original_preds = np.array(original_preds)
        original_probs = np.array(original_probs)
        results['original'] = {
            'accuracy': accuracy_score(true_labels, original_preds),
            'f1': f1_score(true_labels, original_preds, average='weighted'),
            'predictions': original_preds,
            'probabilities': original_probs
        }
    
    if simple_model is not None and len(simple_preds) > 0:
        simple_preds = np.array(simple_preds)
        simple_probs = np.array(simple_probs)
        results['simple'] = {
            'accuracy': accuracy_score(true_labels, simple_preds),
            'f1': f1_score(true_labels, simple_preds, average='weighted'),
            'predictions': simple_preds,
            'probabilities': simple_probs
        }
    
    if onnx_model is not None and len(onnx_preds) > 0:
        onnx_preds = np.array(onnx_preds)
        onnx_probs = np.array(onnx_probs)
        results['onnx'] = {
            'accuracy': accuracy_score(true_labels, onnx_preds),
            'f1': f1_score(true_labels, onnx_preds, average='weighted'),
            'predictions': onnx_preds,
            'probabilities': onnx_probs
        }
    
    # 计算模型之间的一致性
    if 'original' in results and 'simple' in results:
        results['original_vs_simple'] = {
            'agreement': np.mean(results['original']['predictions'] == results['simple']['predictions'])
        }
    
    if 'original' in results and 'onnx' in results:
        results['original_vs_onnx'] = {
            'agreement': np.mean(results['original']['predictions'] == results['onnx']['predictions'])
        }
    
    if 'simple' in results and 'onnx' in results:
        results['simple_vs_onnx'] = {
            'agreement': np.mean(results['simple']['predictions'] == results['onnx']['predictions'])
        }
    
    # 生成报告
    reports_dir = Path("reports/real_data_validation")
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(15, 5))
    
    # 原始模型混淆矩阵
    if 'original' in results:
        plt.subplot(1, 3, 1)
        cm = confusion_matrix(true_labels, results['original']['predictions'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('原始模型混淆矩阵')
        plt.colorbar()
        plt.xticks([0, 1], ['阴性', '阳性'])
        plt.yticks([0, 1], ['阴性', '阳性'])
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.grid(False)
        
        # 在混淆矩阵中显示数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    # 简化模型混淆矩阵
    if 'simple' in results:
        plt.subplot(1, 3, 2)
        cm = confusion_matrix(true_labels, results['simple']['predictions'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('简化模型混淆矩阵')
        plt.colorbar()
        plt.xticks([0, 1], ['阴性', '阳性'])
        plt.yticks([0, 1], ['阴性', '阳性'])
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.grid(False)
        
        # 在混淆矩阵中显示数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    # ONNX模型混淆矩阵
    if 'onnx' in results:
        plt.subplot(1, 3, 3)
        cm = confusion_matrix(true_labels, results['onnx']['predictions'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('ONNX模型混淆矩阵')
        plt.colorbar()
        plt.xticks([0, 1], ['阴性', '阳性'])
        plt.yticks([0, 1], ['阴性', '阳性'])
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.grid(False)
        
        # 在混淆矩阵中显示数字
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrices.png")
    plt.close()
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    
    # 原始模型ROC曲线
    if 'original' in results:
        fpr, tpr, _ = roc_curve(true_labels, results['original']['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'原始模型 (AUC = {roc_auc:.4f})')
    
    # 简化模型ROC曲线
    if 'simple' in results:
        fpr, tpr, _ = roc_curve(true_labels, results['simple']['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'简化模型 (AUC = {roc_auc:.4f})')
    
    # ONNX模型ROC曲线
    if 'onnx' in results:
        fpr, tpr, _ = roc_curve(true_labels, results['onnx']['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ONNX模型 (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.savefig(reports_dir / "roc_curves.png")
    plt.close()
    
    # 绘制精确率-召回率曲线
    plt.figure(figsize=(10, 8))
    
    # 原始模型PR曲线
    if 'original' in results:
        precision, recall, _ = precision_recall_curve(true_labels, results['original']['probabilities'])
        ap = average_precision_score(true_labels, results['original']['probabilities'])
        plt.plot(recall, precision, label=f'原始模型 (AP = {ap:.4f})')
    
    # 简化模型PR曲线
    if 'simple' in results:
        precision, recall, _ = precision_recall_curve(true_labels, results['simple']['probabilities'])
        ap = average_precision_score(true_labels, results['simple']['probabilities'])
        plt.plot(recall, precision, label=f'简化模型 (AP = {ap:.4f})')
    
    # ONNX模型PR曲线
    if 'onnx' in results:
        precision, recall, _ = precision_recall_curve(true_labels, results['onnx']['probabilities'])
        ap = average_precision_score(true_labels, results['onnx']['probabilities'])
        plt.plot(recall, precision, label=f'ONNX模型 (AP = {ap:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    plt.savefig(reports_dir / "pr_curves.png")
    plt.close()
    
    # 生成文本报告
    report = f"""# 真实数据上的模型验证报告

## 数据集信息

- 测试样本数: {len(true_labels)}
- 阳性样本数: {np.sum(true_labels == 1)}
- 阴性样本数: {np.sum(true_labels == 0)}

## 性能指标

| 模型 | 准确率 | F1分数 |
|-----|-------|-------|
"""
    
    if 'original' in results:
        report += f"| 原始模型 | {results['original']['accuracy']:.4f} | {results['original']['f1']:.4f} |\n"
    
    if 'simple' in results:
        report += f"| 简化模型 | {results['simple']['accuracy']:.4f} | {results['simple']['f1']:.4f} |\n"
    
    if 'onnx' in results:
        report += f"| ONNX模型 | {results['onnx']['accuracy']:.4f} | {results['onnx']['f1']:.4f} |\n"
    
    report += f"""
## 模型一致性

"""
    
    if 'original_vs_simple' in results:
        agreement = results['original_vs_simple']['agreement']
        report += f"- 原始模型与简化模型预测一致性: {agreement:.4f} ({int(agreement * len(true_labels))}/{len(true_labels)})\n"
    
    if 'original_vs_onnx' in results:
        agreement = results['original_vs_onnx']['agreement']
        report += f"- 原始模型与ONNX模型预测一致性: {agreement:.4f} ({int(agreement * len(true_labels))}/{len(true_labels)})\n"
    
    if 'simple_vs_onnx' in results:
        agreement = results['simple_vs_onnx']['agreement']
        report += f"- 简化模型与ONNX模型预测一致性: {agreement:.4f} ({int(agreement * len(true_labels))}/{len(true_labels)})\n"
    
    report += f"""
## 混淆矩阵

![混淆矩阵](confusion_matrices.png)

## ROC曲线

![ROC曲线](roc_curves.png)

## 精确率-召回率曲线

![PR曲线](pr_curves.png)

## 详细分类报告

### 原始模型

```
"""
    
    if 'original' in results:
        report += classification_report(true_labels, results['original']['predictions'])
    
    report += f"""
```

### 简化模型

```
"""
    
    if 'simple' in results:
        report += classification_report(true_labels, results['simple']['predictions'])
    
    report += f"""
```

### ONNX模型

```
"""
    
    if 'onnx' in results:
        report += classification_report(true_labels, results['onnx']['predictions'])
    
    report += f"""
```

## 结论

"""
    
    # 添加结论
    if 'original' in results and 'onnx' in results:
        accuracy_diff = abs(results['original']['accuracy'] - results['onnx']['accuracy'])
        agreement = results['original_vs_onnx']['agreement']
        
        if accuracy_diff < 0.01 and agreement > 0.99:
            report += "ONNX模型与原始PyTorch模型表现几乎完全一致，转换过程没有导致性能下降。\n"
        elif accuracy_diff < 0.05 and agreement > 0.95:
            report += "ONNX模型与原始PyTorch模型表现非常接近，转换过程导致了轻微的性能变化。\n"
        else:
            report += "ONNX模型与原始PyTorch模型表现存在差异，转换过程可能导致了一定的性能下降。\n"
    
    if 'simple' in results and 'onnx' in results:
        accuracy_diff = abs(results['simple']['accuracy'] - results['onnx']['accuracy'])
        agreement = results['simple_vs_onnx']['agreement']
        
        if accuracy_diff < 0.01 and agreement > 0.99:
            report += "ONNX模型与简化PyTorch模型表现几乎完全一致，转换过程没有导致性能下降。\n"
        elif accuracy_diff < 0.05 and agreement > 0.95:
            report += "ONNX模型与简化PyTorch模型表现非常接近，转换过程导致了轻微的性能变化。\n"
        else:
            report += "ONNX模型与简化PyTorch模型表现存在差异，转换过程可能导致了一定的性能下降。\n"
    
    # 保存报告
    with open(reports_dir / "real_data_validation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    logging.info(f"验证报告已保存至: {reports_dir / 'real_data_validation_report.md'}")
    
    return results

def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='验证ONNX模型在真实数据上的性能')
    parser.add_argument('--model', type=str, default='airbubble_hybrid_net', 
                       help='要验证的模型名称')
    args = parser.parse_args()
    
    # 加载原始模型
    original_model = load_original_model()
    
    # 加载简化模型
    simple_model = load_simple_model()
    
    # 加载ONNX模型
    onnx_model, input_size = load_onnx_model(args.model)
    
    # 加载真实测试数据
    test_batches = load_real_test_data(input_size=input_size if input_size else 70)
    
    if test_batches is None or len(test_batches) == 0:
        logging.error("未能加载测试数据，退出验证")
        return
    
    # 在真实数据上比较模型
    evaluate_models_on_real_data(original_model, simple_model, onnx_model, test_batches)

if __name__ == "__main__":
    main()