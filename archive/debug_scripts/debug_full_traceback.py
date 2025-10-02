#!/usr/bin/env python3
"""
捕获完整的错误堆栈跟踪
"""

import sys
import os
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    try:
        # 导入并运行原始训练脚本的main函数
        from train_enhanced_multilevel import main as train_main
        
        # 设置命令行参数
        sys.argv = [
            'train_enhanced_multilevel.py',
            '--data_dir', './data',
            '--batch_size', '16',
            '--epochs', '1',
            '--learning_rate', '0.001'
        ]
        
        print("🚀 开始运行训练脚本...")
        train_main()
        
    except Exception as e:
        print(f"\n❌ 捕获到异常: {e}")
        print(f"异常类型: {type(e)}")
        print("\n📋 完整堆栈跟踪:")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80)
        
        # 获取异常的详细信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        print("\n🔍 异常详细信息:")
        print(f"  异常类型: {exc_type}")
        print(f"  异常值: {exc_value}")
        
        # 打印调用栈
        print("\n📚 调用栈:")
        for i, frame_info in enumerate(traceback.extract_tb(exc_traceback)):
            print(f"  {i+1}. 文件: {frame_info.filename}")
            print(f"     行号: {frame_info.lineno}")
            print(f"     函数: {frame_info.name}")
            print(f"     代码: {frame_info.line}")
            print()

if __name__ == "__main__":
    main()