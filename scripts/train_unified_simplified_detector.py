"""
统一训练框架下的简化版气孔检测器训练脚本
使用与其他模型相同的数据集进行训练
调整标签映射：阳性代表生长，阴性代表不生长
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import logging
from datetime import datetime
import time
import random
from tqdm import tqdm

from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector, create_simplified_airbubble_detector
from training.dataset import create_data_loaders, BioastDataset, get_transforms
from core.config import get_experiment_path, DATA_DIR

# 设置日志
def setup_logger(log_file=None):
    """设置日志"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_epoch(model, data_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 用于记录每个类别的性能
    class_correct = {0: 0, 1: 0}  # 0=阴性(不生长), 1=阳性(生长)
    class_total = {0: 0, 1: 0}
    
    for inputs, labels in tqdm(data_loader, desc="训练中", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 记录每个类别的性能
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # 计算每个类别的准确率
    class_acc = {
        cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0)
        for cls in class_total
    }
    
    return epoch_loss, epoch_acc, class_acc

def validate(model, data_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    # 用于记录每个类别的性能
    class_correct = {0: 0, 1: 0}  # 0=阴性(不生长), 1=阳性(生长)
    class_total = {0: 0, 1: 0}
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="验证中", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 记录每个类别的性能
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # 计算每个类别的准确率
    class_acc = {
        cls: (class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0)
        for cls in class_total
    }
    
    # 计算F1分数
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # 计算每个类别的精确率和召回率
    precision = precision_score(all_labels, all_predictions, average=None)
    recall = recall_score(all_labels, all_predictions, average=None)
    
    metrics = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'f1': f1,
        'class_acc': class_acc,
        'precision': {0: precision[0], 1: precision[1]},
        'recall': {0: recall[0], 1: recall[1]}
    }
    
    return metrics

def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(20, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 绘制准确率曲线
    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 绘制F1分数曲线
    plt.subplot(2, 3, 3)
    plt.plot(history['val_f1'], label='验证F1')
    plt.title('F1分数曲线')
    plt.xlabel('轮次')
    plt.ylabel('F1分数')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 绘制学习率曲线
    plt.subplot(2, 3, 4)
    plt.plot(history['lr'], label='学习率')
    plt.title('学习率曲线')
    plt.xlabel('轮次')
    plt.ylabel('学习率')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 绘制类别准确率曲线（如果有）
    if 'class_acc_0' in history and 'class_acc_1' in history:
        plt.subplot(2, 3, 5)
        plt.plot(history['class_acc_0'], label='阴性(不生长)')
        plt.plot(history['class_acc_1'], label='阳性(生长)')
        plt.title('类别准确率曲线')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(alpha=0.3)
    
    # 绘制训练/验证差距曲线
    plt.subplot(2, 3, 6)
    gaps = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    plt.plot(gaps, label='训练/验证差距')
    plt.title('训练/验证差距曲线')
    plt.xlabel('轮次')
    plt.ylabel('差距')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_checkpoint(model, optimizer, epoch, val_metrics, history, save_path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'history': history
    }, save_path)

def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 设置模型名称
    model_name = 'simplified_airbubble_detector'
    
    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = get_experiment_path(model_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(experiment_dir, f'{model_name}_{timestamp}.log')
    logger = setup_logger(log_file)
    
    logger.info(f"开始训练 {model_name}...")
    logger.info(f"使用真实数据集，标签映射: 阴性=不生长(孔明显), 阳性=生长(孔不明显)")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    # 创建模型
    model = create_simplified_airbubble_detector()
    model.to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数: {total_params}")
    
    # 创建数据加载器
    data_loaders = create_data_loaders(
        str(DATA_DIR),
        batch_size=32,
        num_workers=2,
        image_size=70
    )
    
    # 打印数据集信息
    logger.info("使用真实数据集进行训练")
    logger.info("标签映射: 阴性(negative)=不生长(孔明显), 阳性(positive)=生长(孔不明显)")
    
    # 打印数据集大小
    logger.info(f"Data prepared - Train: {len(data_loaders['train'].dataset)} "
                f"Val: {len(data_loaders['val'])} "
                f"Test: {len(data_loaders['test'])}")
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # 训练参数
    num_epochs = 30
    best_val_acc = 0.0
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': [],
        'class_acc_0': [],  # 阴性(不生长)准确率
        'class_acc_1': [],  # 阳性(生长)准确率
        'precision_0': [],  # 阴性(不生长)精确率
        'precision_1': [],  # 阳性(生长)精确率
        'recall_0': [],     # 阴性(不生长)召回率
        'recall_1': []      # 阳性(生长)召回率
    }
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        
        # 训练
        train_loss, train_acc, train_class_acc = train_epoch(model, data_loaders['train'], criterion, optimizer, device)
        
        # 验证
        val_metrics = validate(model, data_loaders['val'], criterion, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1']
        val_class_acc = val_metrics['class_acc']
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # 记录每个类别的性能
        history['class_acc_0'].append(val_class_acc[0])
        history['class_acc_1'].append(val_class_acc[1])
        history['precision_0'].append(val_precision[0])
        history['precision_1'].append(val_precision[1])
        history['recall_0'].append(val_recall[0])
        history['recall_1'].append(val_recall[1])
        
        # 打印结果
        logger.info(f"训练损失: {train_loss:.4f} 训练准确率: {train_acc*100:.2f}%")
        logger.info(f"验证损失: {val_loss:.4f} 验证准确率: {val_acc*100:.2f}% 验证F1: {val_f1*100:.2f}%")
        logger.info(f"学习率: {current_lr:.6f}")
        logger.info(f"训练/验证差距: {(train_acc-val_acc)*100:.2f}%")
        
        # 打印每个类别的性能
        logger.info(f"训练类别准确率 - 阴性(不生长): {train_class_acc[0]*100:.2f}%, 阳性(生长): {train_class_acc[1]*100:.2f}%")
        logger.info(f"验证类别准确率 - 阴性(不生长): {val_class_acc[0]*100:.2f}%, 阳性(生长): {val_class_acc[1]*100:.2f}%")
        logger.info(f"验证精确率 - 阴性(不生长): {val_precision[0]*100:.2f}%, 阳性(生长): {val_precision[1]*100:.2f}%")
        logger.info(f"验证召回率 - 阴性(不生长): {val_recall[0]*100:.2f}%, 阳性(生长): {val_recall[1]*100:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(experiment_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_metrics, history, best_model_path)
            logger.info(f"检查点已保存: {best_model_path}")
            logger.info(f"新的最佳验证准确率: {val_acc*100:.2f}%")
        
        # 每10个epoch保存一次检查点
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(experiment_dir, f'epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, val_acc, history, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # 保存训练历史
    history_path = os.path.join(experiment_dir, f'{model_name}_{timestamp}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # 绘制训练曲线
    curves_path = os.path.join(experiment_dir, f'{model_name}_{timestamp}_training_curves.png')
    plot_training_curves(history, curves_path)
    
    # 在测试集上评估最佳模型
    logger.info("\n在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    test_metrics = validate(model, data_loaders['test'], criterion, device)
    
    test_loss = test_metrics['loss']
    test_acc = test_metrics['accuracy']
    test_f1 = test_metrics['f1']
    test_class_acc = test_metrics['class_acc']
    test_precision = test_metrics['precision']
    test_recall = test_metrics['recall']
    
    logger.info(f"测试损失: {test_loss:.4f} 测试准确率: {test_acc*100:.2f}% 测试F1: {test_f1*100:.2f}%")
    logger.info(f"测试类别准确率 - 阴性(不生长): {test_class_acc[0]*100:.2f}%, 阳性(生长): {test_class_acc[1]*100:.2f}%")
    logger.info(f"测试精确率 - 阴性(不生长): {test_precision[0]*100:.2f}%, 阳性(生长): {test_precision[1]*100:.2f}%")
    logger.info(f"测试召回率 - 阴性(不生长): {test_recall[0]*100:.2f}%, 阳性(生长): {test_recall[1]*100:.2f}%")
    
    # 保存测试结果
    test_results = {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_class_acc': {str(k): v for k, v in test_class_acc.items()},
        'test_precision': {str(k): v for k, v in test_precision.items()},
        'test_recall': {str(k): v for k, v in test_recall.items()}
    }
    test_results_path = os.path.join(experiment_dir, f'{model_name}_test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f)
    
    # 生成简短报告
    report = f"""# {model_name} 训练报告

## 训练详情
- 日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 轮次: {num_epochs}
- 最佳验证准确率: {best_val_acc*100:.2f}%
- 测试准确率: {test_acc*100:.2f}%
- 测试F1分数: {test_f1*100:.2f}%

## 类别性能
- 阴性(不生长)准确率: {test_class_acc[0]*100:.2f}%
- 阳性(生长)准确率: {test_class_acc[1]*100:.2f}%
- 阴性(不生长)精确率: {test_precision[0]*100:.2f}%
- 阳性(生长)精确率: {test_precision[1]*100:.2f}%
- 阴性(不生长)召回率: {test_recall[0]*100:.2f}%
- 阳性(生长)召回率: {test_recall[1]*100:.2f}%

## 模型信息
- 参数数量: {total_params}
- 架构: SimplifiedAirBubbleDetector

## 数据集
- 训练集: {len(data_loaders['train'].dataset)} 样本
- 验证集: {len(data_loaders['val'].dataset)} 样本
- 测试集: {len(data_loaders['test'].dataset)} 样本

## 标签映射
- 0: 阴性(不生长) - 气泡更明显
- 1: 阳性(生长) - 气泡可能被细菌生长覆盖
"""
    
    report_path = os.path.join(experiment_dir, f'{model_name}_{timestamp}_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"\nTraining completed. Results saved to {experiment_dir}")

if __name__ == "__main__":
    main()