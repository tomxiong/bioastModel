"""
为simplified_airbubble_detector模型生成详细的验证报告
包括正确和错误样本的分析
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import cv2
from tqdm import tqdm
import shutil
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 导入项目模块
from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector
from core.data_loader import MICDataLoader
import torch.nn.functional as F

def find_latest_experiment():
    """查找最新的实验目录"""
    base_dir = Path("experiments")
    if not base_dir.exists():
        logging.error(f"实验目录不存在: {base_dir}")
        return None
    
    # 查找所有simplified_airbubble_detector目录
    model_dirs = list(base_dir.glob("**/simplified_airbubble_detector"))
    
    if not model_dirs:
        logging.error("未找到simplified_airbubble_detector的实验目录")
        return None
    
    # 按修改时间排序
    latest_dir = max(model_dirs, key=os.path.getmtime)
    logging.info(f"找到最新实验目录: {latest_dir}")
    
    # 检查是否有best_model.pth文件
    best_model_path = latest_dir / "best_model.pth"
    if not best_model_path.exists():
        logging.warning(f"最佳模型文件不存在: {best_model_path}")
        logging.warning("训练可能尚未完成，将使用最新的checkpoint")
        
        # 查找最新的checkpoint
        checkpoints = list(latest_dir.glob("checkpoint_*.pth"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            logging.info(f"找到最新checkpoint: {latest_checkpoint}")
            # 复制最新的checkpoint为best_model.pth
            shutil.copy(latest_checkpoint, best_model_path)
            logging.info(f"已将{latest_checkpoint}复制为{best_model_path}")
    
    return latest_dir

def load_model(experiment_dir):
    """加载模型"""
    model_path = os.path.join(experiment_dir, "best_model.pth")
    
    if not os.path.exists(model_path):
        logging.error(f"模型文件不存在: {model_path}")
        return None
    
    # 创建模型实例
    model = SimplifiedAirBubbleDetector()
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logging.info(f"成功加载模型: {model_path}")
        return model
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        return None

def create_data_loader():
    """创建数据加载器"""
    try:
        data_loader = MICDataLoader()
        _, _, test_images, test_labels = data_loader.get_all_data()
        logging.info(f"成功加载测试数据: {len(test_images)}个样本")
        return test_images, test_labels
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        return None, None

def preprocess_image(image):
    """预处理图像"""
    # 确保图像是3通道的
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 调整大小为模型输入尺寸
    image = cv2.resize(image, (224, 224))
    
    # 归一化
    image = image.astype(np.float32) / 255.0
    
    # 转换为PyTorch张量
    image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    
    return image

def evaluate_model(model, test_images, test_labels):
    """评估模型"""
    if model is None or test_images is None or test_labels is None:
        return None, None, None, None, None
    
    # 创建结果目录
    report_dir = Path("reports/simplified_detector_validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    correct_dir = report_dir / "correct_samples"
    incorrect_dir = report_dir / "incorrect_samples"
    correct_dir.mkdir(exist_ok=True)
    incorrect_dir.mkdir(exist_ok=True)
    
    # 清空目录
    for dir_path in [correct_dir, incorrect_dir]:
        for file in dir_path.glob("*"):
            file.unlink()
    
    # 评估结果
    all_preds = []
    all_labels = []
    all_probs = []
    correct_samples = []
    incorrect_samples = []
    
    # 遍历测试样本
    for i, (image, label) in enumerate(tqdm(zip(test_images, test_labels), total=len(test_images), desc="评估样本")):
        # 预处理图像
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
        
        # 模型预测
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        # 记录结果
        pred = prediction.item()
        prob = confidence.item()
        all_preds.append(pred)
        all_labels.append(label)
        all_probs.append(prob)
        
        # 保存样本
        sample_info = {
            "index": i,
            "true_label": label,
            "predicted_label": pred,
            "confidence": prob,
            "image": image
        }
        
        if pred == label:
            correct_samples.append(sample_info)
        else:
            incorrect_samples.append(sample_info)
    
    # 计算指标
    accuracy = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # 保存样本图像
    save_sample_images(correct_samples, correct_dir, "correct", max_samples=50)
    save_sample_images(incorrect_samples, incorrect_dir, "incorrect", max_samples=50)
    
    return accuracy, f1, precision, recall, {
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "correct_samples": correct_samples,
        "incorrect_samples": incorrect_samples
    }

def save_sample_images(samples, output_dir, prefix, max_samples=50):
    """保存样本图像"""
    # 按置信度排序
    samples.sort(key=lambda x: x["confidence"], reverse=True)
    
    # 限制样本数量
    samples = samples[:max_samples]
    
    # 保存图像
    for i, sample in enumerate(samples):
        image = sample["image"]
        true_label = sample["true_label"]
        pred_label = sample["predicted_label"]
        conf = sample["confidence"]
        
        # 构建文件名
        filename = f"{prefix}_{i}_true{true_label}_pred{pred_label}_conf{conf:.2f}.png"
        filepath = output_dir / filename
        
        # 保存图像
        cv2.imwrite(str(filepath), image)

def generate_confusion_matrix(labels, predictions, class_names):
    """生成混淆矩阵"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在每个单元格中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存图像
    plt.savefig("reports/simplified_detector_validation/confusion_matrix.png")
    plt.close()

def generate_report(accuracy, f1, precision, recall, results):
    """生成报告"""
    report_dir = Path("reports/simplified_detector_validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成混淆矩阵
    generate_confusion_matrix(results["labels"], results["predictions"], ["无气泡", "有气泡"])
    
    # 生成分类报告
    class_report = classification_report(results["labels"], results["predictions"], target_names=["无气泡", "有气泡"])
    
    # 创建报告文件
    report_path = report_dir / "validation_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# SimplifiedAirBubbleDetector 验证报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 模型性能指标\n\n")
        f.write(f"- 准确率: {accuracy*100:.2f}%\n")
        f.write(f"- F1分数: {f1*100:.2f}%\n")
        f.write(f"- 精确率: {precision*100:.2f}%\n")
        f.write(f"- 召回率: {recall*100:.2f}%\n\n")
        
        f.write("## 分类报告\n\n")
        f.write("```\n")
        f.write(class_report)
        f.write("\n```\n\n")
        
        f.write("## 混淆矩阵\n\n")
        f.write("![混淆矩阵](confusion_matrix.png)\n\n")
        
        f.write("## 样本分析\n\n")
        
        f.write("### 正确分类样本\n\n")
        f.write(f"总计: {len(results['correct_samples'])}个样本\n\n")
        f.write("前50个高置信度正确样本已保存在`correct_samples`目录中。\n\n")
        
        f.write("### 错误分类样本\n\n")
        f.write(f"总计: {len(results['incorrect_samples'])}个样本\n\n")
        
        if results['incorrect_samples']:
            f.write("#### 错误样本详情\n\n")
            f.write("| 样本索引 | 真实标签 | 预测标签 | 置信度 | 图像链接 |\n")
            f.write("|---------|---------|---------|--------|--------|\n")
            
            for i, sample in enumerate(results['incorrect_samples'][:50]):
                idx = sample["index"]
                true_label = "有气泡" if sample["true_label"] == 1 else "无气泡"
                pred_label = "有气泡" if sample["predicted_label"] == 1 else "无气泡"
                conf = sample["confidence"]
                image_path = f"incorrect_samples/incorrect_{i}_true{sample['true_label']}_pred{sample['predicted_label']}_conf{conf:.2f}.png"
                
                f.write(f"| {idx} | {true_label} | {pred_label} | {conf:.2f} | [查看]({image_path}) |\n")
        else:
            f.write("没有错误分类的样本！模型表现完美。\n")
    
    logging.info(f"验证报告已生成: {report_path}")
    return report_path

def main():
    """主函数"""
    logging.info("开始生成SimplifiedAirBubbleDetector验证报告")
    
    # 查找最新实验目录
    experiment_dir = find_latest_experiment()
    if experiment_dir is None:
        return
    
    # 加载模型
    model = load_model(experiment_dir)
    if model is None:
        return
    
    # 创建数据加载器
    test_images, test_labels = create_data_loader()
    if test_images is None:
        return
    
    # 评估模型
    logging.info("开始评估模型...")
    accuracy, f1, precision, recall, results = evaluate_model(model, test_images, test_labels)
    
    if results is None:
        logging.error("模型评估失败")
        return
    
    # 生成报告
    report_path = generate_report(accuracy, f1, precision, recall, results)
    
    logging.info(f"验证完成! 准确率: {accuracy*100:.2f}%, F1分数: {f1*100:.2f}%")
    logging.info(f"报告已保存至: {report_path}")

if __name__ == "__main__":
    main()