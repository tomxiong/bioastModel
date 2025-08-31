#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型验证脚本
使用训练数据集验证ONNX模型性能
"""

import os
import sys
import torch
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import BioastDataset

def validate_onnx_model():
    """验证ONNX模型性能"""
    
    print("开始验证ONNX模型性能...")
    
    # ONNX模型路径
    onnx_model_path = "onnx_models/efficientnet_v2_s.onnx"
    
    if not os.path.exists(onnx_model_path):
        print(f"错误: ONNX模型文件不存在: {onnx_model_path}")
        return
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("加载验证数据集...")
    val_dataset = BioastDataset(
        data_dir="bioast_dataset",
        split="val",
        transform=transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 创建ONNX Runtime会话
    print("加载ONNX模型...")
    ort_session = ort.InferenceSession(onnx_model_path)
    
    # 获取输入输出名称
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    print(f"输入名称: {input_name}")
    print(f"输出名称: {output_name}")
    
    # 验证模型
    all_predictions = []
    all_labels = []
    all_file_paths = []
    error_samples = []
    total_time = 0
    
    print("开始验证...")
    with torch.no_grad():
        sample_idx = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            # 转换为numpy数组
            images_np = images.numpy()
            
            # ONNX推理
            start_time = time.time()
            ort_outputs = ort_session.run([output_name], {input_name: images_np})
            end_time = time.time()
            
            total_time += (end_time - start_time)
            
            # 获取预测结果
            predictions = np.argmax(ort_outputs[0], axis=1)
            
            # 记录每个样本的信息
            for i in range(len(predictions)):
                file_path = val_dataset.samples[sample_idx + i][0]
                file_name = os.path.basename(file_path)
                true_label = labels[i].item()
                pred_label = predictions[i]
                
                all_predictions.append(pred_label)
                all_labels.append(true_label)
                all_file_paths.append(file_name)
                
                # 记录错误样本
                if true_label != pred_label:
                    class_names = ['Benign', 'Malignant']
                    error_samples.append({
                        'file_name': file_name,
                        'true_label': class_names[true_label],
                        'predicted_label': class_names[pred_label],
                        'file_path': file_path
                    })
            
            sample_idx += len(predictions)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"已处理 {batch_idx + 1}/{len(val_loader)} 批次")
    
    # 计算性能指标
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_inference_time = total_time / len(val_dataset) * 1000  # ms per sample
    
    print("\n=== ONNX模型验证结果 ===")
    print(f"验证准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"平均推理时间: {avg_inference_time:.2f} ms/样本")
    print(f"总推理时间: {total_time:.2f} 秒")
    
    # 错误样本统计
    correct_predictions = sum(1 for true, pred in zip(all_labels, all_predictions) if true == pred)
    num_error_samples = len(all_labels) - correct_predictions
    print(f"正确预测: {correct_predictions}/{len(all_labels)}")
    print(f"错误样本: {num_error_samples}/{len(all_labels)}")
    
    # 输出错误样本详情
    if error_samples:
        print(f"\n=== 错误样本详情 ({len(error_samples)}个) ===")
        for i, error in enumerate(error_samples, 1):
            print(f"{i:2d}. {error['file_name']} - 真实: {error['true_label']}, 预测: {error['predicted_label']}")
    else:
        print("\n🎉 所有样本都预测正确！")
    
    # 分类报告
    print("\n=== 分类报告 ===")
    class_names = ['Benign', 'Malignant']
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # 混淆矩阵
    print("\n=== 混淆矩阵 ===")
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"真实\\预测  Benign  Malignant")
    print(f"Benign      {cm[0][0]:6d}  {cm[0][1]:9d}")
    print(f"Malignant   {cm[1][0]:6d}  {cm[1][1]:9d}")
    
    # 保存结果
    result = {
        'model_name': 'EfficientNetV2-S (ONNX)',
        'accuracy': float(accuracy),
        'error_samples': int(num_error_samples),
        'total_samples': len(all_labels),
        'avg_inference_time_ms': float(avg_inference_time),
        'total_inference_time_s': float(total_time),
        'error_sample_details': error_samples
    }
    
    import json
    result_file = "onnx_validation_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n验证结果已保存到: {result_file}")
    
    return result

if __name__ == "__main__":
    validate_onnx_model()