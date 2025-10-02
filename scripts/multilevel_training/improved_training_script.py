#!/usr/bin/env python3
"""
改进的立即优化版训练脚本
基于性能差距分析的优化配置
目标：将87.90%的准确率提升至91%+
"""

import subprocess
import sys
import os
from datetime import datetime

def run_improved_training():
    """运行改进的训练配置"""
    
    print("🚀 启动改进的立即优化版训练")
    print("=" * 60)
    print("📊 优化配置:")
    print("  - 任务权重: 完全平衡 (1.0, 1.0, 1.0)")
    print("  - 训练轮次: 20 epochs (避免过度训练)")
    print("  - 早停patience: 8 (更快收敛)")
    print("  - 学习率调度器: CosineAnnealingLR")
    print("  - 目标准确率: >91%")
    print("=" * 60)
    
    # 训练命令
    cmd = [
        sys.executable, "train_optimized_simple_enhanced.py",
        "--data_root", "/home/aaa/ws/bioastModel/ds/images",
        "--json_path", "/home/aaa/ws/bioastModel/ds/images/m9e1n170.json",
        "--batch_size", "32",
        "--epochs", "20",
        "--learning_rate", "0.001",
        "--patience", "8",
        "--growth_level_weight", "1.0",
        "--growth_pattern_weight", "1.0", 
        "--interference_weight", "1.0",
        "--experiment_name", "improved_immediate_optimized",
        "--model_size", "small",
        "--dropout_rate", "0.2"
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # 执行训练
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ 训练完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        return False

def main():
    """主函数"""
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查训练脚本是否存在
    if not os.path.exists("train_optimized_simple_enhanced.py"):
        print("❌ 找不到训练脚本: train_optimized_simple_enhanced.py")
        return
    
    # 运行改进的训练
    success = run_improved_training()
    
    if success:
        print("\n🎉 改进训练完成!")
        print("📈 预期改进:")
        print("  - 整体准确率: 87.90% → 91%+")
        print("  - 任务平衡性: 显著改善")
        print("  - 训练稳定性: 提升")
        print("\n📁 请查看 experiments/improved_immediate_optimized_* 目录获取结果")
    else:
        print("\n💡 训练失败，请检查:")
        print("  - 数据路径是否正确")
        print("  - GPU内存是否充足")
        print("  - 依赖包是否完整")
    
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()