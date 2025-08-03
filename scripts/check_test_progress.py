#!/usr/bin/env python3
"""
检查批量测试进度
"""

import os
import json
from datetime import datetime

def check_test_progress():
    """检查测试进度"""
    experiments_to_check = [
        ('experiments/experiment_20250802_140818/efficientnet_b0', 'EfficientNet-B0'),
        ('experiments/experiment_20250802_164948/resnet18_improved', 'ResNet18-Improved'),
        ('experiments/experiment_20250802_231639/convnext_tiny', 'ConvNext-Tiny'),
        ('experiments/experiment_20250803_020217/vit_tiny', 'ViT-Tiny'),
        ('experiments/experiment_20250803_032628/coatnet', 'CoAtNet'),
        ('experiments/experiment_20250803_101438/mic_mobilenetv3', 'MIC_MobileNetV3'),
        ('experiments/experiment_20250803_102845/micro_vit', 'Micro-ViT'),
        ('experiments/experiment_20250803_115344/airbubble_hybrid_net', 'AirBubble_HybridNet')
    ]
    
    print("=" * 60)
    print("批量测试进度检查")
    print("=" * 60)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    completed = []
    missing = []
    
    for exp_path, model_name in experiments_to_check:
        test_results_path = os.path.join(exp_path, 'test_results.json')
        
        if os.path.exists(test_results_path):
            try:
                with open(test_results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                accuracy = results.get('accuracy', 0) * 100
                print(f"✅ {model_name:<20} - 准确率: {accuracy:.2f}%")
                completed.append((model_name, accuracy))
            except Exception as e:
                print(f"❌ {model_name:<20} - 文件损坏: {e}")
                missing.append(model_name)
        else:
            print(f"⏳ {model_name:<20} - 测试中或未开始")
            missing.append(model_name)
    
    print()
    print("=" * 60)
    print(f"完成: {len(completed)}/{len(experiments_to_check)} 个模型")
    
    if completed:
        print("\n已完成的模型:")
        for model_name, accuracy in completed:
            print(f"  - {model_name}: {accuracy:.2f}%")
    
    if missing:
        print(f"\n待完成的模型: {len(missing)} 个")
        for model_name in missing:
            print(f"  - {model_name}")
    
    return len(completed), len(experiments_to_check)

if __name__ == "__main__":
    completed, total = check_test_progress()
    
    if completed == total:
        print("\n🎉 所有模型测试已完成!")
    else:
        print(f"\n⏳ 还有 {total - completed} 个模型正在测试中...")