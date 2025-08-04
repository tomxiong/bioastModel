"""
使用知识蒸馏技术从原始AirBubbleHybridNet模型向简化模型转移知识
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
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

class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, targets):
        """计算蒸馏损失
        
        Args:
            student_logits: 学生模型的输出
            teacher_logits: 教师模型的输出
            targets: 真实标签
        
        Returns:
            total_loss: 总损失
        """
        # 硬目标损失（学生模型与真实标签的交叉熵）
        hard_loss = self.ce_loss(student_logits, targets)
        
        # 软目标损失（学生模型与教师模型的KL散度）
        soft_student = torch.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = torch.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # 总损失 = alpha * 硬目标损失 + (1 - alpha) * 软目标损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss

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

def train_distilled_model(teacher_model, epochs=50, batch_size=32, learning_rate=0.001, 
                         temperature=3.0, alpha=0.5, device='cpu'):
    """训练蒸馏模型
    
    Args:
        teacher_model: 教师模型（原始模型）
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        temperature: 蒸馏温度
        alpha: 硬目标损失权重
        device: 训练设备
    
    Returns:
        student_model: 训练好的学生模型
    """
    # 创建数据加载器
    data_loader = MICDataLoader()
    train_loader, val_loader, _ = create_data_loaders(data_loader, batch_size=batch_size)
    
    # 创建学生模型
    student_model = SimpleAirBubbleModel(num_classes=2)
    student_model.to(device)
    
    # 将教师模型移动到设备
    teacher_model.to(device)
    teacher_model.eval()  # 教师模型设置为评估模式
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    distill_loss_fn = DistillationLoss(temperature=temperature, alpha=alpha)
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 创建保存目录
    save_dir = Path("experiments/distilled_airbubble_model")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 训练循环
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # 训练阶段
        student_model.train()
        train_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            
            # 获取教师模型的输出
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
                if isinstance(teacher_outputs, dict):
                    teacher_outputs = teacher_outputs['classification']
            
            # 获取学生模型的输出
            student_outputs = student_model(images)
            
            # 计算损失
            loss = distill_loss_fn(student_outputs, teacher_outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证阶段
        student_model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                # 获取教师模型的输出
                teacher_outputs = teacher_model(images)
                if isinstance(teacher_outputs, dict):
                    teacher_outputs = teacher_outputs['classification']
                
                # 获取学生模型的输出
                student_outputs = student_model(images)
                
                # 计算损失
                loss = distill_loss_fn(student_outputs, teacher_outputs, targets)
                val_loss += loss.item()
                
                # 收集预测结果
                preds = torch.argmax(student_outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算平均验证损失和准确率
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logging.info(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_dir / "best_model.pth")
            logging.info(f"保存最佳模型，验证准确率: {val_acc:.4f}")
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_history.png")
    
    # 加载最佳模型
    checkpoint = torch.load(save_dir / "best_model.pth")
    student_model.load_state_dict(checkpoint['model_state_dict'])
    
    return student_model

def evaluate_distilled_model(teacher_model, student_model, device='cpu'):
    """评估蒸馏模型
    
    Args:
        teacher_model: 教师模型
        student_model: 学生模型
        device: 评估设备
    """
    # 创建数据加载器
    data_loader = MICDataLoader()
    _, _, test_loader = create_data_loaders(data_loader, batch_size=32)
    
    # 将模型移动到设备
    teacher_model.to(device)
    student_model.to(device)
    
    # 设置为评估模式
    teacher_model.eval()
    student_model.eval()
    
    # 收集预测结果
    teacher_preds = []
    student_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            
            # 教师模型预测
            teacher_outputs = teacher_model(images)
            if isinstance(teacher_outputs, dict):
                teacher_outputs = teacher_outputs['classification']
            teacher_preds.extend(torch.argmax(teacher_outputs, dim=1).cpu().numpy())
            
            # 学生模型预测
            student_outputs = student_model(images)
            student_preds.extend(torch.argmax(student_outputs, dim=1).cpu().numpy())
            
            # 真实标签
            all_targets.extend(targets.numpy())
    
    # 转换为numpy数组
    teacher_preds = np.array(teacher_preds)
    student_preds = np.array(student_preds)
    all_targets = np.array(all_targets)
    
    # 计算性能指标
    teacher_acc = accuracy_score(all_targets, teacher_preds)
    student_acc = accuracy_score(all_targets, student_preds)
    
    teacher_f1 = f1_score(all_targets, teacher_preds, average='weighted')
    student_f1 = f1_score(all_targets, student_preds, average='weighted')
    
    # 计算一致性
    agreement = np.mean(teacher_preds == student_preds)
    
    # 计算混淆矩阵
    cm = confusion_matrix(teacher_preds, student_preds)
    
    # 创建保存目录
    save_dir = Path("reports/distilled_model")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Teacher vs Student Model Confusion Matrix')
    plt.colorbar()
    
    classes = ['Class 0', 'Class 1']
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
    plt.ylabel('Teacher Predictions')
    plt.xlabel('Student Predictions')
    
    # 保存图像
    plt.savefig(save_dir / "teacher_student_comparison.png")
    
    # 生成报告
    report = f"""# 蒸馏模型评估报告

## 性能指标

| 模型 | 准确率 | F1分数 |
|-----|-------|-------|
| 教师模型 (原始) | {teacher_acc:.4f} | {teacher_f1:.4f} |
| 学生模型 (蒸馏) | {student_acc:.4f} | {student_f1:.4f} |

## 模型一致性

- 预测一致性: {agreement:.4f} ({int(agreement * len(all_targets))}/{len(all_targets)})

## 混淆矩阵

![混淆矩阵](teacher_student_comparison.png)

## 结论

1. 教师模型准确率: {teacher_acc:.2%}
2. 学生模型准确率: {student_acc:.2%}
3. 准确率差异: {abs(teacher_acc - student_acc):.2%}
4. 学生模型与教师模型的预测一致性为{agreement:.2%}

## 蒸馏效果

通过知识蒸馏，我们成功地将教师模型的知识转移到了更简单的学生模型中。学生模型现在可以更好地模仿教师模型的行为，同时保持了更简单的架构，更适合转换为ONNX格式和部署。

"""
    
    # 保存报告
    with open(save_dir / "distillation_evaluation.md", "w") as f:
        f.write(report)
    
    logging.info(f"评估报告已保存至: {save_dir / 'distillation_evaluation.md'}")
    
    # 导出为ONNX格式
    try:
        # 确保ONNX模型目录存在
        onnx_dir = Path("onnx_models")
        onnx_dir.mkdir(exist_ok=True)
        
        onnx_path = onnx_dir / "airbubble_hybrid_net_distilled.onnx"
        
        # 创建示例输入
        dummy_input = torch.randn(1, 3, 70, 70)
        
        # 导出为ONNX格式
        torch.onnx.export(
            student_model.cpu(),
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )
        
        logging.info(f"蒸馏模型已导出为ONNX格式: {onnx_path}")
    except Exception as e:
        logging.error(f"导出ONNX模型失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 加载原始模型（教师模型）
    teacher_model = load_original_model()
    if teacher_model is None:
        logging.error("无法加载教师模型，退出")
        return
    
    # 训练蒸馏模型（学生模型）
    student_model = train_distilled_model(
        teacher_model=teacher_model,
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        temperature=3.0,
        alpha=0.5,
        device=device
    )
    
    # 评估蒸馏模型
    evaluate_distilled_model(teacher_model, student_model, device=device)

if __name__ == "__main__":
    main()