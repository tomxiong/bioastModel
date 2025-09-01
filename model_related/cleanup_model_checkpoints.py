#!/usr/bin/env python3
"""
清理模型检查点，为每种模型类型只保留最佳模型

说明：
1. 本脚本会分析所有模型检查点，并按模型类型分组
2. 对于每种模型类型，只保留性能最佳的一个模型
3. 其他模型会被移动到备份目录，而不是删除
4. 对于已知的模型（如EfficientNet），会根据性能分析结果选择最佳模型
5. 对于其他模型，会选择最新的模型作为最佳模型

保留策略：
- EfficientNet: 保留 efficientnet_20250808_014214_best.pth (验证准确率: 99.09%, 测试准确率: 98.66%)
- MicroViT: 保留 micro_vit_20250807_214640_best.pth (验证准确率: 98.94%, 测试准确率: 98.58%)
- MIC_MobileNetV3: 保留 mic_mobilenetv3_20250807_231138_best.pth (验证准确率: 99.16%, 测试准确率: 98.78%)
- 其他模型: 保留最新的版本（假设最新训练的模型性能最好）
"""

import os
import sys
import json
import shutil
import re
from datetime import datetime
from collections import defaultdict

# 检查点目录
CHECKPOINTS_DIR = "/home/aaa/ws/bioastModel/checkpoints"
# 备份目录
BACKUP_DIR = os.path.join(CHECKPOINTS_DIR, "backup")

# 已知的最佳模型（基于性能分析）
KNOWN_BEST_MODELS = {
    "efficientnet": "efficientnet_20250808_014214_best.pth",  # 验证准确率: 99.09%, 测试准确率: 98.66%
    "micro_vit": "micro_vit_20250807_214640_best.pth",        # 验证准确率: 98.94%, 测试准确率: 98.58%
    "mic_mobilenetv3": "mic_mobilenetv3_20250807_231138_best.pth"  # 验证准确率: 99.16%, 测试准确率: 98.78%
}

def ensure_backup_dir():
    """确保备份目录存在"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"📁 创建备份目录: {BACKUP_DIR}")

def get_model_files():
    """获取所有模型检查点文件"""
    if not os.path.exists(CHECKPOINTS_DIR):
        print(f"❌ 检查点目录不存在: {CHECKPOINTS_DIR}")
        return []
    
    files = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.pth')]
    return files

def group_model_files(files):
    """将模型文件按类型分组"""
    model_groups = defaultdict(list)
    
    for file in files:
        # 提取模型类型
        model_type = re.sub(r'_\d{8}_\d{6}_best\.pth$', '', file)
        model_groups[model_type].append(file)
    
    return model_groups

def find_best_model(model_type, files):
    """查找特定模型类型的最佳模型"""
    # 检查是否是已知的最佳模型
    if model_type in KNOWN_BEST_MODELS:
        best_file = KNOWN_BEST_MODELS[model_type]
        if best_file in files:
            return best_file
    
    # 对于其他模型，我们需要分析时间戳
    # 假设最新的模型是最好的（如果没有其他性能指标）
    latest_timestamp = ""
    best_file = None
    
    for file in files:
        # 提取时间戳
        match = re.search(r'(\d{8}_\d{6})_best\.pth$', file)
        if match:
            timestamp = match.group(1)
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                best_file = file
    
    return best_file

def cleanup_models():
    """清理模型检查点，只保留每种类型的最佳模型"""
    print("\n" + "="*80)
    print("🧹 模型检查点清理工具")
    print("="*80)
    print("📝 说明:")
    print("  1. 本工具将分析所有模型检查点，并按模型类型分组")
    print("  2. 对于每种模型类型，只保留性能最佳的一个模型")
    print("  3. 其他模型会被移动到备份目录，而不是删除")
    print("  4. 已知最佳模型:")
    for model_type, best_file in KNOWN_BEST_MODELS.items():
        print(f"     - {model_type}: {best_file}")
    print("  5. 其他模型: 保留最新的版本（假设最新训练的模型性能最好）")
    print("="*80 + "\n")
    
    # 确保备份目录存在
    ensure_backup_dir()
    
    # 获取所有模型文件
    all_files = get_model_files()
    print(f"📊 找到 {len(all_files)} 个模型检查点文件")
    
    # 按类型分组
    model_groups = group_model_files(all_files)
    print(f"📊 模型分为 {len(model_groups)} 种类型")
    
    # 处理每种模型类型
    for model_type, files in model_groups.items():
        print(f"\n🔍 处理模型类型: {model_type}")
        print(f"   共有 {len(files)} 个检查点")
        
        # 找到最佳模型
        best_file = find_best_model(model_type, files)
        
        if best_file:
            # 显示选择原因
            if model_type in KNOWN_BEST_MODELS:
                if model_type == "efficientnet":
                    print(f"🔒 将保留最佳模型: {best_file} (验证准确率: 99.09%, 测试准确率: 98.66%)")
                elif model_type == "micro_vit":
                    print(f"🔒 将保留最佳模型: {best_file} (验证准确率: 98.94%, 测试准确率: 98.58%)")
                elif model_type == "mic_mobilenetv3":
                    print(f"🔒 将保留最佳模型: {best_file} (验证准确率: 99.16%, 测试准确率: 98.78%)")
            else:
                print(f"🔒 将保留最佳模型: {best_file} (选择最新的模型)")
            
            # 创建模型类型的备份目录
            model_backup_dir = os.path.join(BACKUP_DIR, model_type)
            if not os.path.exists(model_backup_dir):
                os.makedirs(model_backup_dir)
            
            # 移动其他文件到备份目录
            for file in files:
                if file != best_file:
                    src = os.path.join(CHECKPOINTS_DIR, file)
                    dst = os.path.join(model_backup_dir, file)
                    shutil.move(src, dst)
                    print(f"   ✅ 已移动: {file}")
        else:
            print(f"❌ 未找到最佳模型")
    
    # 统计结果
    remaining_files = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.pth')]
    print(f"\n🎉 清理完成!")
    print(f"   📊 原始检查点数: {len(all_files)}")
    print(f"   📊 保留的检查点: {len(remaining_files)}")
    print(f"   📊 移动到备份的检查点: {len(all_files) - len(remaining_files)}")
    
    # 保存清理报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "original_count": len(all_files),
        "remaining_count": len(remaining_files),
        "backup_count": len(all_files) - len(remaining_files),
        "model_types": list(model_groups.keys()),
        "remaining_models": remaining_files
    }
    
    with open("checkpoint_cleanup_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"   📄 清理报告已保存至: checkpoint_cleanup_report.json")

if __name__ == "__main__":
    print("🚀 开始清理模型检查点...")
    cleanup_models()