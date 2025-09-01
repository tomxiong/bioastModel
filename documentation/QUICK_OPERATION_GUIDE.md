# BioAst模型管理系统 - 快速操作指南

本指南提供了手动进行单个模型训练、结果分析和对比分析的详细步骤。

## 📋 目录

1. [环境准备](#环境准备)
2. [单模型训练](#单模型训练)
3. [结果分析](#结果分析)
4. [模型对比](#模型对比)
5. [数据集管理](#数据集管理)
6. [批量操作](#批量操作)
7. [报告生成](#报告生成)
8. [常见问题](#常见问题)

## 🚀 环境准备

### 1. 检查项目结构
```bash
# 确保项目目录结构正确
dir /b
# 应该看到以下文件：
# train_single_model.py
# compare_models.py
# dataset_manager.py
# quick_start.py
# config_template.yaml
```

### 2. 准备数据集
```bash
# 数据集应该按以下结构组织：
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

### 3. 检查数据集状态
```bash
python dataset_manager.py --check
```

## 🎯 单模型训练

### 方法1: 使用交互式脚本（推荐新手）
```bash
# 启动交互式训练界面
python quick_start.py
# 选择选项 1: 训练单个模型
# 按提示选择模型类型和配置参数
```

### 方法2: 直接命令行训练
```bash
# 查看可用的预定义模型
python train_single_model.py --list_models

# 训练指定模型（使用默认配置）
python train_single_model.py --model EfficientNet-B0

# 使用自定义配置训练
python train_single_model.py --model ResNet18-Improved --config config_template.yaml

# 训练时指定更多参数
python train_single_model.py --model ConvNeXt-Tiny --epochs 50 --batch_size 32 --lr 0.001
```

### 方法3: 生成训练脚本后手动执行
```bash
# 生成训练脚本但不立即执行
python train_single_model.py --model Micro-ViT --generate_only

# 查看生成的脚本
type generated_training_script.py

# 手动执行生成的脚本
python generated_training_script.py
```

### 训练过程监控
训练过程中会显示：
- 每个epoch的训练和验证损失
- 每个epoch的训练和验证准确率
- 最佳模型保存信息
- 训练时间统计

训练完成后会自动保存：
- `checkpoints/{model_name}/best.pth` - 最佳模型权重
- `checkpoints/{model_name}/training_history.json` - 训练历史
- `checkpoints/{model_name}/config.json` - 训练配置

## 📊 结果分析

### 1. 查看单个模型结果
```bash
# 查看训练历史
type checkpoints\EfficientNet-B0\training_history.json

# 查看模型配置
type checkpoints\EfficientNet-B0\config.json
```

### 2. 使用交互式界面查看
```bash
python quick_start.py
# 选择选项 2: 列出所有模型
# 选择选项 4: 查看系统状态
```

### 3. 生成单模型报告
```bash
# 通过对比脚本生成单模型详细分析
python compare_models.py --models EfficientNet-B0 --generate-report
```

### 4. 详细测试结果分析（新增）
```bash
# 运行详细测试分析（识别失败样本）
python test_result_analyzer.py

# 单个模型分析
python -c "from test_result_analyzer import TestResultAnalyzer; analyzer = TestResultAnalyzer(); analyzer.test_model_from_experiment('experiments/efficientnet_b0/20241220_143022')"
```

**分析结果包含：**
- 失败样本详细列表（CSV格式）
- 错误类型分析（假阳性/假阴性）
- 置信度分布图表
- 改进建议报告
- 可视化分析图表

**输出文件：**
```
experiments/{model_name}/{timestamp}/test_analysis/
├── detailed_test_results.json      # 完整测试数据
├── test_analysis_report.png        # 可视化图表
└── failed_samples_analysis/
    ├── failed_samples_detail.csv    # 失败样本列表
    └── failed_samples_report.md     # 分析报告
```

## 🔍 模型对比

### 1. 对比所有已训练模型
```bash
# 对比所有模型并生成可视化图表
python compare_models.py

# 对比所有模型并生成详细报告
python compare_models.py --generate-report
```

### 2. 对比指定模型
```bash
# 对比两个特定模型
python compare_models.py --models EfficientNet-B0 ResNet18-Improved

# 对比多个模型并生成报告
python compare_models.py --models EfficientNet-B0 ResNet18-Improved ConvNeXt-Tiny --generate-report
```

### 3. 对比性能最好的模型
```bash
# 对比性能最好的3个模型
python compare_models.py --top 3

# 对比性能最好的5个模型并生成报告
python compare_models.py --top 5 --generate-report
```

### 4. 使用交互式界面对比
```bash
python quick_start.py
# 选择选项 3: 对比模型
# 按提示选择要对比的模型
```

### 对比结果包含：
- **对比表格**: CSV格式，包含各模型的关键指标
- **训练曲线图**: 显示训练过程中损失和准确率的变化
- **性能对比图**: 柱状图和饼图显示模型性能对比
- **详细报告**: Markdown格式，包含排行榜、统计信息和建议

## 📁 数据集管理

### 1. 检查数据集状态
```bash
# 检查当前数据集状态
python dataset_manager.py --check
```

### 2. 更新数据集
```bash
# 从新路径更新数据集
python dataset_manager.py --update-dataset "path/to/new/dataset"

# 仅更新数据集配置（如果数据已手动更新）
python dataset_manager.py --update-dataset
```

### 3. 检测需要重训练的模型
数据集更新后，系统会自动检测哪些模型需要重训练：
```bash
python dataset_manager.py --check
# 查看输出中的 "需要重训练" 标记
```

## 🔄 批量操作

### 1. 批量重训练
```bash
# 重训练所有模型
python dataset_manager.py --retrain-all

# 重训练性能最好的模型
python dataset_manager.py --retrain-best

# 重训练指定模型
python dataset_manager.py --retrain-models EfficientNet-B0 ResNet18-Improved
```

### 2. 批量训练新模型
```bash
python quick_start.py
# 选择选项 5: 批量训练
# 按提示选择要训练的模型类型
```

### 3. 计划重训练任务
```bash
# 计划明天同一时间重训练
python dataset_manager.py --schedule-retrain

# 计划指定时间重训练
python dataset_manager.py --schedule-retrain "2024-12-20 02:00"
```

## 📋 报告生成

### 1. 生成系统状态报告
```bash
python quick_start.py
# 选择选项 7: 生成报告
# 选择报告类型（系统状态、模型对比、训练总结）
```

### 2. 生成模型对比报告
```bash
# 生成所有模型的对比报告
python compare_models.py --generate-report

# 生成指定模型的对比报告
python compare_models.py --models model1 model2 --generate-report
```

### 3. 报告文件位置
所有报告默认保存在 `reports/` 目录下：
- `model_comparison_table_*.csv` - 模型对比表格
- `training_curves_*.png` - 训练曲线图
- `performance_comparison_*.png` - 性能对比图
- `detailed_comparison_report_*.md` - 详细对比报告

## 🔧 常见问题

### Q1: 训练过程中出现内存不足错误
**解决方案**:
```bash
# 减小批次大小
python train_single_model.py --model EfficientNet-B0 --batch_size 16

# 或者使用更小的模型
python train_single_model.py --model Micro-ViT
```

### Q2: 找不到数据集
**解决方案**:
```bash
# 检查数据集路径
python dataset_manager.py --check

# 确保数据集结构正确
dir data\train
dir data\val
```

### Q3: 模型训练速度太慢
**解决方案**:
```bash
# 减少训练轮次
python train_single_model.py --model ResNet18-Improved --epochs 20

# 增大批次大小（如果内存允许）
python train_single_model.py --model ResNet18-Improved --batch_size 64

# 使用更小的模型
python train_single_model.py --model Micro-ViT
```

### Q4: 如何恢复中断的训练
**解决方案**:
```bash
# 使用重训练标志继续训练
python train_single_model.py --model EfficientNet-B0 --retrain
```

### Q5: 如何清理旧的模型和报告
**解决方案**:
```bash
# 删除指定模型的检查点
rmdir /s checkpoints\old_model_name

# 清理旧报告
del reports\*_old_timestamp.*
```

## 💡 最佳实践

### 1. 训练前准备
- 确保数据集结构正确
- 检查可用的GPU内存
- 备份重要的模型检查点

### 2. 训练过程
- 从小模型开始测试
- 监控训练曲线，及时调整参数
- 定期保存中间结果

### 3. 结果分析
- 对比多个模型的性能
- 分析训练曲线找出问题
- 生成详细报告便于后续参考

### 4. 数据集管理
- 定期备份数据集
- 记录数据集变更历史
- 及时重训练受影响的模型

### 5. 文件组织
- 使用有意义的模型名称
- 定期清理旧的检查点和报告
- 保持项目目录结构整洁

## 🔗 相关文件

- `train_single_model.py` - 单模型训练脚本
- `compare_models.py` - 模型对比分析脚本
- `dataset_manager.py` - 数据集管理和批量重训练脚本
- `quick_start.py` - 交互式操作界面
- `config_template.yaml` - 配置模板
- `MANUAL_OPERATION_GUIDE.md` - 详细操作手册

---

**提示**: 如果遇到问题，可以使用 `python script_name.py --help` 查看详细的命令行参数说明。