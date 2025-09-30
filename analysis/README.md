# 分析文档中心

本目录包含项目的所有分析文档和脚本，用于训练结果分析、性能评估和改进建议。

## 目录结构

```
analysis/
├── README.md                           # 本文件，分析文档总览
├── analyze_data_distribution.py       # 数据分布分析脚本
├── classification_performance_analysis.py  # 分类性能深度分析脚本
└── gpu_training_run/                  # GPU训练运行分析
    ├── README.md                       # GPU训练分析详细说明
    ├── FINAL_CORRECT_GPU_TRAINING_ANALYSIS_REPORT.md  # 最终分析报告
    ├── correct_*.py                    # 分析脚本
    └── correct_*.json                  # 分析数据
```

## 通用分析工具

### 数据分布分析 (`analyze_data_distribution.py`)
用于分析数据集的分布情况，重点关注：
- negative样本中pores的分布情况
- center_dots vs litter_center_dots的差异
- 各类别的样本不平衡情况
- 生成优化建议和可视化图表

**使用方法**:
```bash
cd analysis
python analyze_data_distribution.py --json_path /path/to/data.json --output_dir ./data_analysis_results
```

### 分类性能深度分析 (`classification_performance_analysis.py`)
基于训练历史数据分析各类别的实际分类性能：
- 任务性能趋势分析
- 损失收敛情况分析
- 各类别性能估算
- 干扰因素检测性能分析

**使用方法**:
```bash
cd analysis
python classification_performance_analysis.py
```

## GPU训练运行分析概览

`gpu_training_run` 目录包含了MultiLevel-MobileNetV3-small模型的完整训练分析：

- **模型**: MultiLevel-MobileNetV3-small
- **训练时间**: 2024-09-19 02:12:08 - 04:32:15
- **总轮数**: 100轮
- **最终性能**: 
  - Growth Level: 96.77%
  - Growth Pattern: 78.99%
  - Interference Factors: 77.43%

## 📋 使用指南

### 查看特定训练分析
```bash
# 进入特定训练分析目录
cd analysis/gpu_training_run

# 查看详细README
cat README.md

# 查看最终分析报告
cat FINAL_CORRECT_GPU_TRAINING_ANALYSIS_REPORT.md
```

### 重新运行分析
```bash
# 进入特定训练目录
cd analysis/gpu_training_run

# 运行完整分析流程
python correct_gpu_training_analysis.py
python correct_training_metrics_extraction.py
python correct_bottleneck_analysis.py
python correct_improvement_suggestions.py
```

## 🔄 添加新训练分析

当有新的训练运行时，按以下结构添加：

```bash
# 创建新训练分析目录
mkdir analysis/[训练名称]

# 移动分析文件
mv [分析文件] analysis/[训练名称]/

# 创建README
cp analysis/gpu_training_run/README.md analysis/[训练名称]/
# 然后修改README内容
```

## 📊 分析文件标准结构

每个训练分析目录应包含：

### 必需文件
- `README.md` - 训练分析说明文档
- `FINAL_*_ANALYSIS_REPORT.md` - 最终分析报告
- `*_detailed_metrics.json` - 详细训练指标
- `*_bottleneck_analysis.json` - 瓶颈分析结果
- `*_improvement_suggestions.json` - 改进建议

### 分析脚本
- `*_analysis.py` - 主分析脚本
- `*_metrics_extraction.py` - 指标提取脚本
- `*_bottleneck_analysis.py` - 瓶颈分析脚本
- `*_improvement_suggestions.py` - 改进建议脚本

## 🎯 最佳实践

1. **命名规范**: 使用描述性的训练名称
2. **文档完整**: 每个训练都应有完整的README
3. **脚本可复用**: 分析脚本应该可以独立运行
4. **数据保留**: 保留原始分析数据用于对比
5. **版本管理**: 记录分析版本和修正历史

---

*此文档作为所有训练分析的入口点，便于管理和查阅不同训练的分析结果。*