#!/usr/bin/env python3
"""
验证高精度模型的训练数据集和错误样本分析
"""

import json
import os
from pathlib import Path

def verify_high_accuracy_models():
    """验证高精度模型的训练情况和错误样本"""
    
    print("🔍 验证高精度模型的训练数据集和错误样本...")
    
    # 读取ONNX性能分析结果
    onnx_analysis_file = "onnx_performance_analysis.json"
    if not os.path.exists(onnx_analysis_file):
        print(f"❌ 找不到ONNX分析文件: {onnx_analysis_file}")
        return
    
    with open(onnx_analysis_file, 'r', encoding='utf-8') as f:
        onnx_data = json.load(f)
    
    # 目标高精度模型
    target_models = [
        "inception_micro",
        "mic_mobilenetv3", 
        "resnet_micro"
    ]
    
    print(f"\n📊 分析目标模型: {target_models}")
    
    high_accuracy_models = []
    
    # 分析每个模型
    for model_result in onnx_data.get("detailed_results", []):
        model_type = model_result.get("model_type", "")
        accuracy = model_result.get("accuracy_results", {}).get("accuracy", 0)
        
        if model_type in target_models and accuracy > 0.99:
            high_accuracy_models.append({
                "model_type": model_type,
                "filename": model_result.get("filename", ""),
                "accuracy": accuracy,
                "file_size_mb": model_result.get("file_size_mb", 0),
                "fps": model_result.get("speed_results", {}).get("fps", 0),
                "predictions": model_result.get("accuracy_results", {}).get("predictions", [])
            })
    
    print(f"\n✅ 找到 {len(high_accuracy_models)} 个高精度模型 (准确率 > 99%)")
    
    # 验证训练数据集
    print("\n" + "="*80)
    print("📋 高精度模型验证报告")
    print("="*80)
    
    for i, model in enumerate(high_accuracy_models, 1):
        print(f"\n🏆 模型 {i}: {model['model_type']}")
        print(f"   文件名: {model['filename']}")
        print(f"   准确率: {model['accuracy']:.4f} ({model['accuracy']*100:.2f}%)")
        print(f"   文件大小: {model['file_size_mb']:.2f} MB")
        print(f"   推理速度: {model['fps']:.1f} FPS")
        
        # 检查是否使用bioast_dataset训练
        accuracy_results = model_result.get("accuracy_results", {})
        predictions = accuracy_results.get('predictions', [])
        filenames = accuracy_results.get('filenames', [])
        
        # 检查文件名列表来验证数据集
        if filenames:
            bioast_count = sum(1 for path in filenames if 'bioast_dataset' in str(path))
            print(f"   数据集验证: {'✅ 使用bioast_dataset训练' if bioast_count > 0 else '❌ 未确认使用bioast_dataset'}")
            if bioast_count > 0:
                print(f"   样本路径示例: {filenames[0] if filenames else 'N/A'}")
        else:
            print("   数据集验证: ❓ 无预测数据可验证")
    
    # 生成错误样本清单
    print(f"\n" + "="*80)
    print("📝 错误样本清单")
    print("="*80)
    
    error_report = {
        "analysis_timestamp": onnx_data.get("analysis_timestamp", ""),
        "high_accuracy_models": [],
        "summary": {
            "total_models_analyzed": len(high_accuracy_models),
            "all_use_bioast_dataset": True,
            "total_error_samples": 0
        }
    }
    
    for model in high_accuracy_models:
        model_type = model['model_type']
        predictions = model.get('predictions', [])
        
        # 提取错误样本
        error_samples = []
        correct_samples = []
        
        for pred in predictions:
            filename = pred.get('filename', '')
            true_label = pred.get('true_label', -1)
            predicted_label = pred.get('predicted_label', -1)
            confidence = pred.get('confidence', 0.0)
            
            if true_label != predicted_label:
                error_samples.append({
                    "filename": filename,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "error_type": "False Positive" if true_label == 0 and predicted_label == 1 else "False Negative"
                })
            else:
                correct_samples.append({
                    "filename": filename,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence
                })
        
        # 检查是否使用bioast_dataset
        uses_bioast = any('bioast_dataset' in sample.get('filename', '') for sample in error_samples + correct_samples)
        
        model_report = {
            "model_type": model_type,
            "filename": model['filename'],
            "accuracy": model['accuracy'],
            "uses_bioast_dataset": uses_bioast,
            "total_predictions": len(predictions),
            "error_samples_count": len(error_samples),
            "correct_samples_count": len(correct_samples),
            "error_samples": error_samples,
            "error_analysis": {
                "false_positives": len([e for e in error_samples if e['error_type'] == 'False Positive']),
                "false_negatives": len([e for e in error_samples if e['error_type'] == 'False Negative']),
                "avg_error_confidence": sum(e['confidence'] for e in error_samples) / len(error_samples) if error_samples else 0
            }
        }
        
        error_report["high_accuracy_models"].append(model_report)
        error_report["summary"]["total_error_samples"] += len(error_samples)
        
        if not uses_bioast:
            error_report["summary"]["all_use_bioast_dataset"] = False
        
        print(f"\n🔍 {model_type.upper()}")
        print(f"   准确率: {model['accuracy']:.4f}")
        print(f"   使用bioast_dataset: {'✅ 是' if uses_bioast else '❌ 否'}")
        print(f"   总预测样本: {len(predictions)}")
        print(f"   错误样本数: {len(error_samples)}")
        print(f"   正确样本数: {len(correct_samples)}")
        
        if error_samples:
            print(f"   错误类型分布:")
            fp_count = len([e for e in error_samples if e['error_type'] == 'False Positive'])
            fn_count = len([e for e in error_samples if e['error_type'] == 'False Negative'])
            print(f"     - 假阳性 (False Positive): {fp_count}")
            print(f"     - 假阴性 (False Negative): {fn_count}")
            print(f"   平均错误置信度: {sum(e['confidence'] for e in error_samples) / len(error_samples):.4f}")
            
            print(f"   错误样本示例 (前5个):")
            for j, error in enumerate(error_samples[:5], 1):
                print(f"     {j}. {error['filename']}")
                print(f"        真实标签: {error['true_label']}, 预测标签: {error['predicted_label']}")
                print(f"        置信度: {error['confidence']:.4f}, 错误类型: {error['error_type']}")
    
    # 保存详细报告
    report_file = "high_accuracy_models_verification_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(error_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细报告已保存到: {report_file}")
    
    # 总结
    print(f"\n" + "="*80)
    print("📊 验证总结")
    print("="*80)
    print(f"✅ 分析了 {len(high_accuracy_models)} 个高精度模型")
    print(f"✅ 所有模型都使用bioast_dataset训练: {'是' if error_report['summary']['all_use_bioast_dataset'] else '否'}")
    print(f"✅ 总错误样本数: {error_report['summary']['total_error_samples']}")
    
    for model_report in error_report["high_accuracy_models"]:
        print(f"   - {model_report['model_type']}: {model_report['accuracy']:.4f} 准确率, {model_report['error_samples_count']} 个错误样本")
    
    return error_report

if __name__ == "__main__":
    verify_high_accuracy_models()