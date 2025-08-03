#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目结构整理工具
将root目录下的文件整理到合适的目录结构中
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """创建新的目录结构"""
    # 定义新的目录结构
    directories = [
        "scripts/dataset",
        "scripts/training",
        "scripts/evaluation",
        "scripts/improvement",
        "scripts/utils",
        "docs/guides",
        "docs/reports",
        "configs"
    ]
    
    # 创建目录
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def get_file_mapping():
    """获取文件映射关系，定义文件应该移动到哪个目录"""
    # 数据集相关脚本
    dataset_scripts = [
        "dataset_builder.py",
        "dataset_manager.py",
        "enhanced_dataset_manager.py",
        "dataset_analyzer.py",
        "create_project_structure.py"
    ]
    
    # 训练相关脚本
    training_scripts = [
        "train.py",
        "smart_train.py",
        "debug_model_config.py"
    ]
    
    # 评估相关脚本
    evaluation_scripts = [
        "batch_test_model.py",
        "simple_batch_test.py",
        "comprehensive_verify.py",
        "verify_classification.py",
        "export_onnx.py",
        "generate_report.py"
    ]
    
    # 改进相关脚本
    improvement_scripts = [
        "implement_data_augmentation.py",
        "implement_data_augmentation_complete.py",
        "implement_data_augmentation_final.py",
        "implement_data_augmentation_fixed.py",
        "implement_data_augmentation_working.py",
        "confidence_calibration.py",
        "threshold_optimization.py",
        "test_improved_model.py",
        "test_improved_model_complete.py",
        "test_improved_model_final.py",
        "test_improved_model_fixed.py",
        "error_analysis.py",
        "short_term_improvements.py"
    ]
    
    # 工具脚本
    utils_scripts = [
        "organize_project_structure.py"
    ]
    
    # 指南文档
    guide_docs = [
        "CFG_SYNC_GUIDE.md",
        "DATASET_MANAGEMENT_GUIDE.md",
        "OVERFITTING_PREVENTION_GUIDE.md",
        "TRAINING_GUIDE.md"
    ]
    
    # 报告文档
    report_docs = [
        "PROJECT_SUMMARY.md",
        "short_term_improvements_summary.md"
    ]
    
    # 配置文件
    config_files = [
        "bioast_train.code-workspace",
        "requirements.txt"
    ]
    
    # 构建文件映射
    file_mapping = {}
    
    for script in dataset_scripts:
        file_mapping[script] = "scripts/dataset"
    
    for script in training_scripts:
        file_mapping[script] = "scripts/training"
    
    for script in evaluation_scripts:
        file_mapping[script] = "scripts/evaluation"
    
    for script in improvement_scripts:
        file_mapping[script] = "scripts/improvement"
    
    for script in utils_scripts:
        file_mapping[script] = "scripts/utils"
    
    for doc in guide_docs:
        file_mapping[doc] = "docs/guides"
    
    for doc in report_docs:
        file_mapping[doc] = "docs/reports"
    
    for config in config_files:
        file_mapping[config] = "configs"
    
    return file_mapping

def move_files(file_mapping):
    """移动文件到新目录"""
    # 获取当前目录下的所有文件
    current_dir = Path(".")
    files = [f for f in current_dir.iterdir() if f.is_file()]
    
    # 移动文件
    for file in files:
        filename = file.name
        
        # 跳过一些特殊文件
        if filename.startswith(".") or filename in ["README.md", "readme.MD", ".gitignore"]:
            print(f"⏭️  跳过文件: {filename}")
            continue
        
        # 检查文件是否在映射中
        if filename in file_mapping:
            target_dir = Path(file_mapping[filename])
            target_path = target_dir / filename
            
            # 确保目标目录存在
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 移动文件
            try:
                shutil.move(str(file), str(target_path))
                print(f"✅ 移动文件: {filename} -> {target_dir}")
            except Exception as e:
                print(f"❌ 移动文件失败 {filename}: {e}")
        else:
            print(f"⚠️  未找到映射关系: {filename}")

def update_main_readme():
    """更新主README文件，添加新的项目结构说明"""
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print("❌ 未找到主README文件")
        return
    
    # 读取现有内容
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 添加项目结构说明
    structure_section = """
## 项目结构

```
bioast_train/
├── bioast_dataset/                 # 数据集目录
│   ├── dataset_stats.json         # 数据集统计信息（JSON格式）
│   ├── negative/                   # 阴性样本
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── positive/                   # 阳性样本
│       ├── train/
│       ├── val/
│       └── test/
├── scripts/                        # 脚本目录
│   ├── dataset/                    # 数据集相关脚本
│   │   ├── dataset_builder.py      # 数据集构建工具
│   │   ├── dataset_manager.py      # 数据集管理工具
│   │   ├── enhanced_dataset_manager.py  # 增强数据集管理工具
│   │   ├── dataset_analyzer.py    # 数据集分析工具
│   │   └── create_project_structure.py  # 项目结构创建脚本
│   ├── training/                   # 训练相关脚本
│   │   ├── train.py                # 主训练脚本
│   │   └── smart_train.py          # 智能训练脚本
│   ├── evaluation/                 # 评估相关脚本
│   │   ├── batch_test_model.py     # 批量测试脚本
│   │   ├── simple_batch_test.py    # 简单批量测试脚本
│   │   └── ...
│   ├── improvement/                # 改进相关脚本
│   │   ├── implement_data_augmentation_working.py  # 数据增强脚本
│   │   ├── confidence_calibration.py  # 置信度校准脚本
│   │   ├── test_improved_model_final.py  # 模型测试脚本
│   │   └── ...
│   └── utils/                      # 工具脚本
├── docs/                          # 文档目录
│   ├── guides/                     # 指南文档
│   │   ├── CFG_SYNC_GUIDE.md       # 配置同步指南
│   │   ├── DATASET_MANAGEMENT_GUIDE.md  # 数据集管理指南
│   │   ├── OVERFITTING_PREVENTION_GUIDE.md  # 过拟合预防指南
│   │   └── TRAINING_GUIDE.md       # 训练指南
│   └── reports/                    # 报告文档
├── configs/                        # 配置文件目录
│   ├── config.yaml                 # 主配置文件
│   ├── config_continue.yaml        # 继续训练配置文件
│   └── bioast_train.code-workspace # VS Code工作区
├── data/                          # 数据处理模块
│   ├── __init__.py
│   └── dataset.py                 # 数据集类定义
├── evaluation/                     # 评估模块
│   ├── __init__.py
│   ├── metrics.py                 # 评估指标
│   └── visualizer.py              # 可视化工具
├── models/                        # 模型定义
│   ├── __init__.py
│   └── faster_vit.py              # Faster ViT模型实现
├── training/                      # 训练模块
│   ├── __init__.py
│   ├── optimizer.py               # 优化器配置
│   └── trainer.py                 # 训练器
├── utils/                         # 工具函数
├── results/                       # 结果输出
│   ├── checkpoints/               # 模型检查点
│   ├── logs/                      # 训练日志
│   ├── models/                    # 导出模型
│   ├── plots/                     # 可视化图表
│   └── reports/                   # 评估报告
├── csharp_integration/            # C#集成模块
├── .gitignore                     # Git忽略文件
├── README.md                      # 项目说明（本文件）
└── requirements.txt               # Python依赖
```

"""
    
    # 将结构说明添加到README中
    # 查找"## 目录结构"部分，如果存在则替换，否则添加
    if "## 目录结构" in content:
        # 替换现有的目录结构部分
        start = content.find("## 目录结构")
        end = content.find("\n## ", start + 1)
        if end == -1:
            end = len(content)
        
        content = content[:start] + structure_section + content[end:]
    else:
        # 在项目概述后添加结构说明
        overview_end = content.find("## 目录结构")
        if overview_end == -1:
            # 如果没有找到目录结构，查找项目概述的结尾
            overview_end = content.find("\n## ", content.find("## 项目概述") + 1)
            if overview_end == -1:
                # 如果还是没有找到，就在项目概述后添加
                overview_end = content.find("\n\n", content.find("## 项目概述") + 1)
        
        if overview_end != -1:
            content = content[:overview_end] + structure_section + content[overview_end:]
        else:
            # 如果还是找不到，就在文件开头添加
            content = structure_section + content
    
    # 写入更新后的内容
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 更新主README文件")

def main():
    """主函数"""
    print("🚀 开始整理项目结构...")
    
    # 创建目录结构
    create_directory_structure()
    
    # 获取文件映射
    file_mapping = get_file_mapping()
    
    # 移动文件
    move_files(file_mapping)
    
    # 更新主README
    update_main_readme()
    
    print("\n✅ 项目结构整理完成！")

if __name__ == "__main__":
    main()