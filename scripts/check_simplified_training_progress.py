"""
检查simplified_airbubble_detector模型的训练进度
"""

import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def find_latest_experiment():
    """查找最新的实验目录"""
    experiments_dir = Path("experiments")
    model_dirs = list(experiments_dir.glob("**/simplified_airbubble_detector"))
    
    if not model_dirs:
        logging.warning("未找到simplified_airbubble_detector的实验目录")
        return None
    
    # 按修改时间排序，获取最新的实验目录
    latest_dir = max(model_dirs, key=os.path.getmtime)
    return latest_dir

def load_training_history(experiment_dir):
    """加载训练历史记录"""
    history_file = experiment_dir / "training_history.json"
    
    if not history_file.exists():
        logging.warning(f"未找到训练历史记录文件: {history_file}")
        return None
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        logging.error(f"加载训练历史记录失败: {e}")
        return None

def plot_training_curves(history):
    """绘制训练曲线"""
    if not history:
        logging.error("没有可用的训练历史记录")
        return
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 绘制准确率
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # 绘制损失
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制类别准确率
    plt.subplot(2, 2, 3)
    
    if 'train_class_acc' in history and '0' in history['train_class_acc']:
        plt.plot(history['train_class_acc']['0'], label='训练阴性(不生长)准确率')
        plt.plot(history['train_class_acc']['1'], label='训练阳性(生长)准确率')
    
    if 'val_class_acc' in history and '0' in history['val_class_acc']:
        plt.plot(history['val_class_acc']['0'], label='验证阴性(不生长)准确率')
        plt.plot(history['val_class_acc']['1'], label='验证阳性(生长)准确率')
    
    plt.title('类别准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    # 绘制F1分数
    plt.subplot(2, 2, 4)
    plt.plot(history['val_f1'], label='验证F1分数')
    plt.title('F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('F1分数')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('reports/simplified_detector_training', exist_ok=True)
    plt.savefig('reports/simplified_detector_training/training_curves.png', dpi=300)
    plt.close()
    
    logging.info("训练曲线已保存至: reports/simplified_detector_training/training_curves.png")

def print_latest_metrics(history):
    """打印最新的指标"""
    if not history:
        logging.error("没有可用的训练历史记录")
        return
    
    # 获取最新的指标
    latest_epoch = len(history['train_acc'])
    latest_train_acc = history['train_acc'][-1]
    latest_val_acc = history['val_acc'][-1]
    latest_val_f1 = history['val_f1'][-1]
    
    logging.info(f"当前训练进度: Epoch {latest_epoch}")
    logging.info(f"训练准确率: {latest_train_acc:.2%}")
    logging.info(f"验证准确率: {latest_val_acc:.2%}")
    logging.info(f"验证F1分数: {latest_val_f1:.2%}")
    
    # 打印类别准确率
    if 'val_class_acc' in history and '0' in history['val_class_acc']:
        latest_val_neg_acc = history['val_class_acc']['0'][-1]
        latest_val_pos_acc = history['val_class_acc']['1'][-1]
        logging.info(f"验证阴性(不生长)准确率: {latest_val_neg_acc:.2%}")
        logging.info(f"验证阳性(生长)准确率: {latest_val_pos_acc:.2%}")
    
    # 打印精确率
    if 'val_precision' in history and '0' in history['val_precision']:
        latest_val_neg_prec = history['val_precision']['0'][-1]
        latest_val_pos_prec = history['val_precision']['1'][-1]
        logging.info(f"验证阴性(不生长)精确率: {latest_val_neg_prec:.2%}")
        logging.info(f"验证阳性(生长)精确率: {latest_val_pos_prec:.2%}")
    
    # 打印召回率
    if 'val_recall' in history and '0' in history['val_recall']:
        latest_val_neg_rec = history['val_recall']['0'][-1]
        latest_val_pos_rec = history['val_recall']['1'][-1]
        logging.info(f"验证阴性(不生长)召回率: {latest_val_neg_rec:.2%}")
        logging.info(f"验证阳性(生长)召回率: {latest_val_pos_rec:.2%}")

def main():
    """主函数"""
    logging.info("检查simplified_airbubble_detector模型的训练进度...")
    
    # 查找最新的实验目录
    experiment_dir = find_latest_experiment()
    
    if not experiment_dir:
        return
    
    logging.info(f"找到最新的实验目录: {experiment_dir}")
    
    # 加载训练历史记录
    history = load_training_history(experiment_dir)
    
    if not history:
        return
    
    # 打印最新的指标
    print_latest_metrics(history)
    
    # 绘制训练曲线
    plot_training_curves(history)
    
    logging.info("检查完成")

if __name__ == "__main__":
    main()