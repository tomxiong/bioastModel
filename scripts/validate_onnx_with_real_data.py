"""
使用真实训练集数据验证ONNX模型性能
比较原始PyTorch模型和转换后的ONNX模型在真实测试数据上的性能
"""

import os
import sys
import logging
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import cv2
import json
import random
from tqdm import tqdm

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

class BioastDataset(Dataset):
    """Bioast数据集类"""
    
    def __init__(self, data_dir, split='test', image_size=(70, 70), transform=None):
        """初始化数据集
        
        Args:
            data_dir: 数据目录
            split: 数据集划分（'train', 'val', 'test'）
            image_size: 图像大小
            transform: 数据变换
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # 加载数据
        self.images = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 加载数据划分信息
        split_file = self.data_dir / f"{self.split}_split.json"
        if not split_file.exists():
            # 如果没有划分文件，尝试自动划分
            self._auto_split_data()
        else:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            # 加载图像和标签
            for item in split_data:
                image_path = self.data_dir / item['image_path']
                if image_path.exists():
                    self.images.append(image_path)
                    self.labels.append(item['label'])
        
        logging.info(f"加载了 {len(self.images)} 个 {self.split} 样本")
    
    def _auto_split_data(self):
        """自动划分数据"""
        # 查找所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(self.data_dir.glob(f"**/{ext}")))
        
        # 如果没有图像文件，尝试查找子目录
        if not image_files:
            for subdir in self.data_dir.iterdir():
                if subdir.is_dir():
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        image_files.extend(list(subdir.glob(f"**/{ext}")))
        
        if not image_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到图像文件")
        
        # 根据目录名称确定标签
        for image_path in image_files:
            # 尝试从父目录名称确定标签
            parent_dir = image_path.parent.name.lower()
            
            # 根据目录名称确定标签
            if 'positive' in parent_dir or 'pos' in parent_dir:
                label = 1
            elif 'negative' in parent_dir or 'neg' in parent_dir:
                label = 0
            else:
                # 如果无法从目录名称确定标签，尝试从文件名确定
                filename = image_path.stem.lower()
                if 'positive' in filename or 'pos' in filename:
                    label = 1
                elif 'negative' in filename or 'neg' in filename:
                    label = 0
                else:
                    # 如果仍然无法确定，跳过该图像
                    continue
            
            # 根据划分确定是否添加到当前数据集
            if self.split == 'train' and random.random() < 0.7:
                self.images.append(image_path)
                self.labels.append(label)
            elif self.split == 'val' and 0.7 <= random.random() < 0.85:
                self.images.append(image_path)
                self.labels.append(label)
            elif self.split == 'test' and random.random() >= 0.85:
                self.images.append(image_path)
                self.labels.append(label)
        
        # 保存划分信息
        split_data = [{'image_path': str(img.relative_to(self.data_dir)), 'label': lbl} for img, lbl in zip(self.images, self.labels)]
        with open(self.data_dir / f"{self.split}_split.json", 'w') as f:
            json.dump(split_data, f)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.images[idx]
        image = cv2.imread(str(image_path))
        
        # 如果图像加载失败，返回随机噪声图像
        if image is None:
            logging.warning(f"无法加载图像: {image_path}")
            image = np.random.rand(70, 70, 3) * 255
        
        # 转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小
        image = cv2.resize(image, self.image_size)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

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
        model: ONNX模型
    """
    try:
        onnx_path = Path(f"onnx_models/{model_name}.onnx")
        
        if not onnx_path.exists():
            logging.error(f"ONNX模型文件不存在: {onnx_path}")
            return None
        
        # 创建ONNX运行时会话
        session = ort.InferenceSession(str(onnx_path))
        
        # 获取输入和输出名称
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 获取输入形状
        input_shape = session.get_inputs()[0].shape
        logging.info(f"ONNX模型输入形状: {input_shape}")
        
        logging.info(f"ONNX模型 {model_name} 加载成功")
        
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
        
        return onnx_model
    except Exception as e:
        logging.error(f"加载ONNX模型 {model_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_models_on_real_data(original_model, simple_model, onnx_model, data_dir):
    """在真实数据上比较模型的性能
    
    Args:
        original_model: 原始模型
        simple_model: 简化模型
        onnx_model: ONNX模型
        data_dir: 数据目录
    
    Returns:
        results: 评估结果
    """
    # 创建测试数据集
    test_dataset = BioastDataset(data_dir, split='test')
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # 收集预测结果
    original_preds = []
    original_probs = []
    simple_preds = []
    simple_probs = []
    onnx_preds = []
    onnx_probs = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估模型"):
            # 原始模型预测
            if original_model is not None:
                original_outputs = original_model(images)
                if isinstance(original_outputs, dict):
                    original_outputs = original_outputs['classification']
                original_probs.extend(torch.softmax(original_outputs, dim=1)[:, 1].cpu().numpy())
                original_preds.extend(torch.argmax(original_outputs, dim=1).cpu().numpy())
            
            # 简化模型预测
            if simple_model is not None:
                simple_outputs = simple_model(images)
                simple_probs.extend(torch.softmax(simple_outputs, dim=1)[:, 1].cpu().numpy())
                simple_preds.extend(torch.argmax(simple_outputs, dim=1).cpu().numpy())
            
            # ONNX模型预测
            if onnx_model is not None:
                onnx_outputs = onnx_model(images)
                onnx_probs.extend(torch.softmax(onnx_outputs, dim=1)[:, 1].cpu().numpy())
                onnx_preds.extend(torch.argmax(onnx_outputs, dim=1).cpu().numpy())
            
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
    
    # 生成混淆矩阵图
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
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 在混淆矩阵中添加文本
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
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
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 在混淆矩阵中添加文本
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
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
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 在混淆矩阵中添加文本
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrices.png")
    
    # 生成ROC曲线
    plt.figure(figsize=(15, 5))
    
    # 原始模型ROC曲线
    if 'original' in results:
        plt.subplot(1, 3, 1)
        fpr, tpr, _ = roc_curve(true_labels, results['original']['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('原始模型ROC曲线')
        plt.legend(loc="lower right")
    
    # 简化模型ROC曲线
    if 'simple' in results:
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(true_labels, results['simple']['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('简化模型ROC曲线')
        plt.legend(loc="lower right")
    
    # ONNX模型ROC曲线
    if 'onnx' in results:
        plt.subplot(1, 3, 3)
        fpr, tpr, _ = roc_curve(true_labels, results['onnx']['probabilities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ONNX模型ROC曲线')
        plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curves.png")
    
    # 生成精确率-召回率曲线
    plt.figure(figsize=(15, 5))
    
    # 原始模型PR曲线
    if 'original' in results:
        plt.subplot(1, 3, 1)
        precision, recall, _ = precision_recall_curve(true_labels, results['original']['probabilities'])
        plt.plot(recall, precision, lw=2)
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('原始模型PR曲线')
        plt.ylim([0.0, 1.05])
    
    # 简化模型PR曲线
    if 'simple' in results:
        plt.subplot(1, 3, 2)
        precision, recall, _ = precision_recall_curve(true_labels, results['simple']['probabilities'])
        plt.plot(recall, precision, lw=2)
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('简化模型PR曲线')
        plt.ylim([0.0, 1.05])
    
    # ONNX模型PR曲线
    if 'onnx' in results:
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(true_labels, results['onnx']['probabilities'])
        plt.plot(recall, precision, lw=2)
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('ONNX模型PR曲线')
        plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(reports_dir / "pr_curves.png")
    
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
    # 设置数据目录
    data_dir = "C:/Users/tomxiong/codebuddy/bioastModel/bioast_dataset"
    
    # 加载原始模型
    original_model = load_original_model()
    
    # 加载简化模型
    simple_model = load_simple_model()
    
    # 加载ONNX模型
    onnx_model = load_onnx_model("airbubble_hybrid_net")
    
    # 在真实数据上比较模型
    evaluate_models_on_real_data(original_model, simple_model, onnx_model, data_dir)

if __name__ == "__main__":
    main()