#!/usr/bin/env python3
"""
清理 EfficientNet 检查点，只保留最佳模型
"""

import os
import sys
import glob
import shutil

def cleanup_efficientnet_checkpoints():
    """清理 EfficientNet 检查点，只保留最佳模型"""
    print("🧹 开始清理 EfficientNet 检查点...")
    
    # 检查点目录
    checkpoint_dir = "/home/aaa/ws/bioastModel/checkpoints/"
    
    # 最佳模型路径
    best_model_path = "/home/aaa/ws/bioastModel/checkpoints/efficientnet_20250808_014214_best.pth"
    best_model_filename = os.path.basename(best_model_path)
    
    # 确保最佳模型存在
    if not os.path.exists(best_model_path):
        print(f"❌ 错误: 最佳模型 {best_model_path} 不存在!")
        return False
    
    # 查找所有 EfficientNet 检查点
    patterns = [
        "efficientnet*.pth",
        "EfficientNet*.pth", 
        "*efficientnet*.pth"
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        all_checkpoints.extend(files)
    
    # 去重并排序
    all_checkpoints = list(set(all_checkpoints))
    all_checkpoints.sort()
    
    # 创建备份目录
    backup_dir = os.path.join(checkpoint_dir, "efficientnet_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # 统计
    total_checkpoints = len(all_checkpoints)
    moved_count = 0
    
    print(f"📊 找到 {total_checkpoints} 个 EfficientNet 检查点")
    print(f"🔒 将保留最佳模型: {best_model_filename}")
    print(f"📁 其他检查点将移动到: {backup_dir}")
    
    # 移动非最佳模型到备份目录
    for checkpoint in all_checkpoints:
        if checkpoint != best_model_path:
            checkpoint_filename = os.path.basename(checkpoint)
            backup_path = os.path.join(backup_dir, checkpoint_filename)
            
            try:
                shutil.move(checkpoint, backup_path)
                moved_count += 1
                print(f"   ✅ 已移动: {checkpoint_filename}")
            except Exception as e:
                print(f"   ❌ 移动失败 {checkpoint_filename}: {str(e)}")
    
    print(f"\n🎉 清理完成!")
    print(f"   📊 总检查点数: {total_checkpoints}")
    print(f"   📊 已移动到备份: {moved_count}")
    print(f"   📊 保留的检查点: 1 ({best_model_filename})")
    
    return True

if __name__ == "__main__":
    success = cleanup_efficientnet_checkpoints()
    if not success:
        sys.exit(1)