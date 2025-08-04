#!/usr/bin/env python3
"""
单独训练模型的使用示例
演示如何训练新增的模型
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """运行命令并打印输出"""
    print(f"\n🚀 执行命令: {cmd}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"执行失败: {e}")
        return False

def main():
    print("📋 单独训练模型示例")
    print("=" * 60)
    
    # 1. 列出所有可用模型
    print("\n1️⃣ 查看所有可用模型:")
    run_command("python train_single_model.py --list_models")
    
    # 2. 训练一个轻量级模型 (快速测试)
    print("\n2️⃣ 训练 ShuffleNet V2 0.5x (轻量级模型):")
    run_command("python train_single_model.py --model shufflenetv2_x0_5 --epochs 2 --batch_size 32")
    
    # 3. 训练 MobileNet V3 Small
    print("\n3️⃣ 训练 MobileNet V3 Small:")
    run_command("python train_single_model.py --model mobilenetv3_small --epochs 3 --batch_size 64")
    
    # 4. 训练 EfficientNet V2-S
    print("\n4️⃣ 训练 EfficientNet V2-S:")
    run_command("python train_single_model.py --model efficientnetv2_s --epochs 2 --batch_size 32")
    
    # 5. 训练 GhostNet
    print("\n5️⃣ 训练 GhostNet:")
    run_command("python train_single_model.py --model ghostnet --epochs 3")
    
    print("\n✅ 示例训练完成!")
    print("\n📊 训练结果文件:")
    print("- 模型检查点保存在: checkpoints/{model_name}/")
    print("- 训练历史保存在: checkpoints/{model_name}/training_history.json")
    print("- 单次训练结果: single_model_result_{model_name}_{timestamp}.json")

if __name__ == "__main__":
    main()