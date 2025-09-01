#!/usr/bin/env python3
"""
提取高精度模型的错误样本清单
"""

import json
import os

def extract_error_samples():
    """提取高精度模型的错误样本"""
    
    print("🔍 提取高精度模型错误样本...")
    
    # 读取ONNX性能分析结果
    with open("onnx_performance_analysis.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 目标高精度模型
    target_models = ["inception_micro", "mic_mobilenetv3", "resnet_micro"]
    
    print("\n📊 高精度模型验证报告")
    print("="*80)
    
    for result in data.get("detailed_results", []):
        model_type = result.get("model_type", "")
        if model_type not in target_models:
            continue
            
        filename = result.get("filename", "")
        accuracy = result.get("accuracy_results", {}).get("accuracy", 0)
        
        if accuracy < 0.99:
            continue
            
        print(f"\n🏆 模型: {model_type.upper()}")
        print(f"   文件: {filename}")
        print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 检查数据集
        accuracy_results = result.get("accuracy_results", {})
        filenames = accuracy_results.get("filenames", [])
        
        if filenames and any('bioast_dataset' in str(f) for f in filenames[:10]):
            print("   ✅ 确认使用bioast_dataset训练")
            print(f"   样本路径示例: {filenames[0]}")
        else:
            print("   ❓ 无法确认数据集来源")
        
        # 提取错误样本
        predictions = accuracy_results.get("predictions", [])
        true_labels = accuracy_results.get("true_labels", [])
        predicted_labels = accuracy_results.get("predicted_labels", [])
        confidences = accuracy_results.get("confidences", [])
        
        if len(predictions) > 0 and len(true_labels) > 0:
            error_samples = []
            
            for i, (true_label, pred_label) in enumerate(zip(true_labels, predicted_labels)):
                if true_label != pred_label and i < len(filenames) and i < len(confidences):
                    error_type = "假阳性 (False Positive)" if true_label == 0 and pred_label == 1 else "假阴性 (False Negative)"
                    error_samples.append({
                        "filename": filenames[i] if i < len(filenames) else f"sample_{i}",
                        "true_label": int(true_label),
                        "predicted_label": int(pred_label),
                        "confidence": float(confidences[i]) if i < len(confidences) else 0.0,
                        "error_type": error_type
                    })
            
            print(f"   错误样本数: {len(error_samples)}")
            
            if error_samples:
                fp_count = sum(1 for e in error_samples if "False Positive" in e["error_type"])
                fn_count = sum(1 for e in error_samples if "False Negative" in e["error_type"])
                
                print(f"   错误类型分布:")
                print(f"     - 假阳性: {fp_count}")
                print(f"     - 假阴性: {fn_count}")
                
                print(f"   错误样本清单 (前10个):")
                for j, error in enumerate(error_samples[:10], 1):
                    print(f"     {j}. {error['filename']}")
                    print(f"        真实: {error['true_label']}, 预测: {error['predicted_label']}")
                    print(f"        置信度: {error['confidence']:.4f}, 类型: {error['error_type']}")
                
                # 保存该模型的错误样本
                error_file = f"{model_type}_error_samples.json"
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "model_type": model_type,
                        "filename": filename,
                        "accuracy": accuracy,
                        "total_errors": len(error_samples),
                        "false_positives": fp_count,
                        "false_negatives": fn_count,
                        "error_samples": error_samples
                    }, f, indent=2, ensure_ascii=False)
                
                print(f"   💾 错误样本已保存到: {error_file}")
        else:
            print("   ❌ 无法提取错误样本数据")
    
    print(f"\n✅ 高精度模型验证完成!")

if __name__ == "__main__":
    extract_error_samples()