#!/usr/bin/env python3
"""
优化的多任务训练配置和启动脚本
"""

import subprocess
import sys
import time
from pathlib import Path

def run_optimized_training():
    """运行优化的训练配置"""
    
    # 优化配置
    configs = [
        {
            'name': 'fast_training',
            'params': [
                '--batch_size', '48',
                '--epochs', '50',
                '--lr', '0.005',
                '--width_mult', '1.0',
                '--dropout_rate', '0.1',
                '--num_workers', '4'
            ],
            'description': '快速训练配置 - 大批次量，高学习率'
        },
        {
            'name': 'balanced_training',
            'params': [
                '--batch_size', '32',
                '--epochs', '80',
                '--lr', '0.003',
                '--width_mult', '1.2',
                '--dropout_rate', '0.15',
                '--num_workers', '4'
            ],
            'description': '平衡训练配置 - 中等批次量，适中学习率'
        },
        {
            'name': 'stable_training',
            'params': [
                '--batch_size', '24',
                '--epochs', '100',
                '--lr', '0.002',
                '--width_mult', '1.1',
                '--dropout_rate', '0.2',
                '--num_workers', '2'
            ],
            'description': '稳定训练配置 - 小批次量，低学习率'
        }
    ]
    
    print("=== 优化的多任务MobileNetV3训练配置 ===")
    print()
    
    for i, config in enumerate(configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   描述: {config['description']}")
        print(f"   参数: {' '.join(config['params'])}")
        print()
    
    try:
        choice = input("请选择训练配置 (1-3): ").strip()
        choice = int(choice) - 1
        
        if 0 <= choice < len(configs):
            selected_config = configs[choice]
            print(f"\n选择的配置: {selected_config['name']}")
            print(f"描述: {selected_config['description']}")
            print("\n开始训练...")
            
            # 构建命令
            cmd = [sys.executable, 'train_stable_m16_multitask.py'] + selected_config['params']
            
            # 运行训练
            process = subprocess.Popen(cmd, 
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True,
                                     encoding='utf-8')
            
            # 实时输出日志
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            
            if process.returncode == 0:
                print("\n✅ 训练完成!")
            else:
                print(f"\n❌ 训练失败，返回码: {process.returncode}")
                
        else:
            print("❌ 无效选择")
            
    except ValueError:
        print("❌ 请输入有效数字")
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")

def check_gpu_memory():
    """检查GPU内存使用情况"""
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU总内存: {total_memory:.1f} GB")
            
            # 根据内存大小推荐配置
            if total_memory >= 8:
                print("推荐配置: batch_size=48-64, num_workers=4")
            elif total_memory >= 4:
                print("推荐配置: batch_size=32-48, num_workers=2-4")
            else:
                print("推荐配置: batch_size=16-32, num_workers=2")
        else:
            print("未检测到CUDA设备，将使用CPU训练")
    except ImportError:
        print("未安装PyTorch，无法检查GPU内存")

def main():
    """主函数"""
    print("=== 优化的多任务MobileNetV3训练助手 ===")
    print()
    
    # 检查GPU内存
    check_gpu_memory()
    print()
    
    # 运行训练
    run_optimized_training()

if __name__ == "__main__":
    main()