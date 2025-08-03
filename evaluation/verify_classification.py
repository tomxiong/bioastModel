#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证数据分类的正确性
"""

import os
from pathlib import Path

def read_cfg_file(cfg_path):
    """读取cfg文件"""
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
        parts = line.split(',')
        if len(parts) == 2:
            return parts[1]  # 返回标签字符串
    except:
        pass
    return None

def verify_sample_classification():
    """验证几个样本的分类是否正确"""
    source_dir = Path("D:\\image_analysis")
    dataset_dir = Path("bioast_dataset")
    
    print("验证数据分类正确性...")
    print("=" * 50)
    
    # 检查几个具体的样本
    test_cases = [
        ("EB20000078", 24),  # hole_24
        ("EB20000078", 35),  # hole_35 
        ("EB20000078", 50),  # hole_50
        ("EB20000086", 35),  # hole_35
        ("EB20000086", 50),  # hole_50
        ("EB20000091", 24),  # hole_24
    ]
    
    for sample_name, hole_num in test_cases:
        print(f"\n检查样本 {sample_name} 的 hole_{hole_num}:")
        
        # 读取cfg文件
        cfg_path = source_dir / f"{sample_name}.cfg"
        labels_str = read_cfg_file(cfg_path)
        
        if not labels_str:
            print(f"  ❌ 无法读取cfg文件: {cfg_path}")
            continue
            
        # 获取对应位置的标签（hole_num对应cfg中索引hole_num）
        if hole_num < len(labels_str):
            cfg_label = labels_str[hole_num]
            expected_class = 'positive' if cfg_label == '+' else 'negative' if cfg_label == '-' else 'unknown'
            
            print(f"  CFG标签[{hole_num}]: '{cfg_label}' -> {expected_class}")
            
            # 检查文件是否在正确的目录中
            filename = f"{sample_name}_hole_{hole_num}.png"
            found_in = []
            
            for class_dir in ['positive', 'negative']:
                for split_dir in ['train', 'test', 'val']:
                    file_path = dataset_dir / class_dir / split_dir / filename
                    if file_path.exists():
                        found_in.append(f"{class_dir}/{split_dir}")
            
            if not found_in:
                print(f"  ❌ 文件未找到: {filename}")
            elif len(found_in) > 1:
                print(f"  ❌ 文件重复: {found_in}")
            else:
                actual_class = found_in[0].split('/')[0]
                if actual_class == expected_class:
                    print(f"  ✅ 分类正确: {found_in[0]}")
                else:
                    print(f"  ❌ 分类错误: 期望{expected_class}, 实际{actual_class} ({found_in[0]})")
        else:
            print(f"  ❌ hole_{hole_num} 超出cfg范围")

def main():
    verify_sample_classification()
    print("\n验证完成!")

if __name__ == "__main__":
    main()