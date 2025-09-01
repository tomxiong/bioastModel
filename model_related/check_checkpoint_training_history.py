#!/usr/bin/env python3
"""
检查高精度模型checkpoints中的训练历史记录以确认使用的数据集
"""

import torch
import json
import os
from pathlib import Path

def check_checkpoint_training_history():
    """检查checkpoint文件中的训练历史记录"""
    
    print("🔍 检查高精度模型checkpoints的训练历史记录...")
    
    # 目标模型的checkpoint文件
    target_checkpoints = {
        "inception_micro": "checkpoints/inception_micro_20250808_000513_best.pth",
        "mic_mobilenetv3": "checkpoints/mic_mobilenetv3_20250807_231138_best.pth", 
        "resnet_micro": "checkpoints/resnet_micro_20250808_005254_best.pth",
        "densenet_compact": "checkpoints/densenet_compact_20250808_010530_best.pth"
    }
    
    verification_results = {
        "analysis_timestamp": "2025-08-08T23:00:00",
        "models_checked": [],
        "summary": {
            "total_models": len(target_checkpoints),
            "successfully_loaded": 0,
            "confirmed_bioast_dataset": 0,
            "training_history_available": 0
        }
    }
    
    print(f"\n📊 检查 {len(target_checkpoints)} 个高精度模型的checkpoints")
    print("="*80)
    
    for model_name, checkpoint_path in target_checkpoints.items():
        print(f"\n🔍 检查模型: {model_name.upper()}")
        print(f"   Checkpoint路径: {checkpoint_path}")
        
        model_result = {
            "model_name": model_name,
            "checkpoint_path": checkpoint_path,
            "checkpoint_exists": False,
            "checkpoint_loaded": False,
            "dataset_confirmed": False,
            "dataset_info": {},
            "training_config": {},
            "training_history": {},
            "error_message": None
        }
        
        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"   ❌ Checkpoint文件不存在")
            model_result["error_message"] = "Checkpoint file not found"
            verification_results["models_checked"].append(model_result)
            continue
        
        model_result["checkpoint_exists"] = True
        print(f"   ✅ Checkpoint文件存在")
        
        try:
            # 加载checkpoint
            print(f"   📂 加载checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_result["checkpoint_loaded"] = True
            verification_results["summary"]["successfully_loaded"] += 1
            
            print(f"   ✅ Checkpoint加载成功")
            print(f"   📋 Checkpoint包含的键: {list(checkpoint.keys())}")
            
            # 检查数据集信息
            dataset_info = {}
            training_config = {}
            training_history = {}
            
            # 检查各种可能的键
            possible_keys = [
                'dataset_info', 'dataset_config', 'data_info',
                'training_config', 'config', 'args',
                'training_history', 'history', 'train_history',
                'epoch', 'best_accuracy', 'best_val_accuracy',
                'model_info', 'metadata'
            ]
            
            for key in possible_keys:
                if key in checkpoint:
                    value = checkpoint[key]
                    print(f"   📝 发现键 '{key}': {type(value)}")
                    
                    if 'dataset' in key.lower() or 'data' in key.lower():
                        dataset_info[key] = value
                    elif 'config' in key.lower() or 'args' in key.lower():
                        training_config[key] = value
                    elif 'history' in key.lower():
                        training_history[key] = value
                    
                    # 如果是字典类型，显示部分内容
                    if isinstance(value, dict):
                        print(f"      内容预览: {list(value.keys())[:5]}...")
                    elif isinstance(value, (int, float, str)):
                        print(f"      值: {value}")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"      列表长度: {len(value)}, 首项类型: {type(value[0])}")
            
            # 特别检查是否有数据集路径信息
            dataset_paths = []
            bioast_confirmed = False
            
            def search_for_bioast(obj, path=""):
                """递归搜索bioast_dataset相关信息"""
                nonlocal bioast_confirmed, dataset_paths
                
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        current_path = f"{path}.{k}" if path else k
                        if isinstance(v, str) and 'bioast' in v.lower():
                            dataset_paths.append(f"{current_path}: {v}")
                            bioast_confirmed = True
                        elif isinstance(v, (dict, list)):
                            search_for_bioast(v, current_path)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        current_path = f"{path}[{i}]"
                        if isinstance(item, str) and 'bioast' in item.lower():
                            dataset_paths.append(f"{current_path}: {item}")
                            bioast_confirmed = True
                        elif isinstance(item, (dict, list)):
                            search_for_bioast(item, current_path)
                elif isinstance(obj, str) and 'bioast' in obj.lower():
                    dataset_paths.append(f"{path}: {obj}")
                    bioast_confirmed = True
            
            # 搜索整个checkpoint中的bioast相关信息
            search_for_bioast(checkpoint)
            
            model_result["dataset_info"] = dataset_info
            model_result["training_config"] = training_config
            model_result["training_history"] = training_history
            model_result["dataset_confirmed"] = bioast_confirmed
            
            if bioast_confirmed:
                verification_results["summary"]["confirmed_bioast_dataset"] += 1
                print(f"   ✅ 确认使用bioast_dataset训练")
                for path in dataset_paths:
                    print(f"      📍 {path}")
            else:
                print(f"   ❓ 未在checkpoint中找到明确的bioast_dataset信息")
            
            if training_history:
                verification_results["summary"]["training_history_available"] += 1
                print(f"   ✅ 包含训练历史记录")
            
            # 显示关键训练信息
            if 'best_val_accuracy' in checkpoint:
                print(f"   🎯 最佳验证准确率: {checkpoint['best_val_accuracy']:.4f}")
            if 'epoch' in checkpoint:
                print(f"   📈 训练轮数: {checkpoint['epoch']}")
            
        except Exception as e:
            print(f"   ❌ 加载checkpoint失败: {str(e)}")
            model_result["error_message"] = str(e)
        
        verification_results["models_checked"].append(model_result)
    
    # 保存详细结果
    report_file = "checkpoint_training_history_verification.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n" + "="*80)
    print("📊 检查总结")
    print("="*80)
    
    summary = verification_results["summary"]
    print(f"✅ 总模型数: {summary['total_models']}")
    print(f"✅ 成功加载: {summary['successfully_loaded']}")
    print(f"✅ 确认使用bioast_dataset: {summary['confirmed_bioast_dataset']}")
    print(f"✅ 包含训练历史: {summary['training_history_available']}")
    
    print(f"\n💾 详细报告已保存到: {report_file}")
    
    # 生成简要总结
    print(f"\n📋 各模型验证状态:")
    for result in verification_results["models_checked"]:
        status = "✅" if result["dataset_confirmed"] else "❓"
        print(f"   {status} {result['model_name']}: {'确认bioast_dataset' if result['dataset_confirmed'] else '未确认数据集'}")
    
    return verification_results

if __name__ == "__main__":
    check_checkpoint_training_history()