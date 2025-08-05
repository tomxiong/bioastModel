# BioAst模型训练与对比分析工具

一个专为手动模型训练、结果分析和对比分析设计的标准化工具集。

## 🎯 项目目标

本项目提供了一套标准化的流程和工具，用于：
- **单模型训练**: 灵活的手动模型训练流程
- **结果分析**: 统一的训练结果分析和可视化
- **模型对比**: 全面的模型性能对比和报告生成
- **数据集管理**: 智能的数据集版本控制和批量重训练
- **持续更新**: 支持新数据集的集成和模型重训练

## 🚀 快速开始

### 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 检查数据集状态
python dataset_manager.py --check
```

### 交互式操作（推荐新手）
```bash
# 启动交互式界面
python quick_start.py
```

### 命令行操作
```bash
# 训练单个模型
python train_single_model.py --model EfficientNet-B0

# 对比所有模型
python compare_models.py --generate-report

# 批量重训练
python dataset_manager.py --retrain-all
```

## 📁 核心文件

### 主要脚本
- **`train_single_model.py`** - 单模型训练脚本
- **`compare_models.py`** - 模型对比分析脚本
- **`dataset_manager.py`** - 数据集管理和批量重训练脚本
- **`quick_start.py`** - 交互式操作界面

### 配置文件
- **`config_template.yaml`** - 标准化配置模板
- **`dataset_config.json`** - 数据集版本管理配置

### 文档指南
- **`QUICK_OPERATION_GUIDE.md`** - 快速操作指南
- **`MANUAL_OPERATION_GUIDE.md`** - 详细操作手册
- **`BIOAST_SYSTEM_INTRODUCTION.md`** - 完整系统介绍

## 🎯 使用场景

### 1. 单模型训练
```bash
# 查看可用模型
python train_single_model.py --list_models

# 训练指定模型
python train_single_model.py --model ResNet18-Improved --epochs 50

# 使用自定义配置
python train_single_model.py --model ConvNeXt-Tiny --config config_template.yaml
```

### 2. 模型对比分析
```bash
# 对比所有已训练模型
python compare_models.py

# 对比指定模型
python compare_models.py --models EfficientNet-B0 ResNet18-Improved

# 对比性能最好的3个模型
python compare_models.py --top 3 --generate-report
```

### 3. 数据集更新和重训练
```bash
# 检查数据集状态
python dataset_manager.py --check

# 更新数据集
python dataset_manager.py --update-dataset "path/to/new/dataset"

# 重训练所有模型
python dataset_manager.py --retrain-all
```

## 📊 输出结果

### 训练结果
- `checkpoints/{model_name}/best.pth` - 最佳模型权重
- `checkpoints/{model_name}/training_history.json` - 训练历史
- `checkpoints/{model_name}/config.json` - 训练配置

### 对比报告
- `reports/model_comparison_table_*.csv` - 模型对比表格
- `reports/training_curves_*.png` - 训练曲线图
- `reports/performance_comparison_*.png` - 性能对比图
- `reports/detailed_comparison_report_*.md` - 详细对比报告

### 数据集管理
- `dataset_config.json` - 数据集版本和模型训练记录
- `backups/dataset_backup_*` - 数据集备份

## 🔧 配置说明

### 数据集结构
```
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

### 预定义模型
- **EfficientNet-B0**: 高效的卷积神经网络
- **ResNet18-Improved**: 改进的残差网络
- **ConvNeXt-Tiny**: 现代卷积网络架构
- **Micro-ViT**: 轻量级视觉Transformer
- **AirBubble-HybridNet**: 混合网络架构

## 📈 特色功能

### 1. 标准化流程
- 统一的训练配置格式
- 标准化的结果输出
- 一致的报告生成

### 2. 智能管理
- 自动检测数据集变化
- 智能标记需要重训练的模型
- 版本控制和历史追踪

### 3. 灵活操作
- 支持交互式和命令行两种方式
- 可自定义训练参数
- 灵活的模型选择和对比

### 4. 可视化分析
- 训练曲线对比图
- 性能排行榜
- 详细的统计分析

## 🔄 持续更新支持

### 数据集版本管理
- 自动计算数据集哈希值
- 跟踪数据集变更历史
- 标记受影响的模型

### 批量重训练
- 一键重训练所有模型
- 选择性重训练指定模型
- 重训练性能最好的模型

### 计划任务
- 支持定时重训练任务
- 灵活的任务调度

## 💡 最佳实践

1. **训练前**: 检查数据集状态和可用资源
2. **训练中**: 监控训练曲线，及时调整参数
3. **训练后**: 生成对比报告，分析模型性能
4. **数据更新**: 及时重训练受影响的模型
5. **结果管理**: 定期清理旧的检查点和报告

## 🔗 扩展性

本工具集设计时考虑了未来的扩展需求：
- **Web界面集成**: 代码结构支持Web界面的集成
- **API接口**: 可以轻松添加RESTful API
- **自动化流程**: 支持更复杂的自动化工作流
- **云端部署**: 支持云端训练和存储

## 📚 文档资源

- **[快速操作指南](QUICK_OPERATION_GUIDE.md)** - 详细的使用步骤
- **[手动操作手册](MANUAL_OPERATION_GUIDE.md)** - 完整的操作说明
- **[系统介绍](BIOAST_SYSTEM_INTRODUCTION.md)** - 完整系统架构介绍

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具集。

## 📄 许可证

本项目采用MIT许可证。

---

**BioAst模型训练与对比分析工具** - 让模型训练、分析和对比变得简单高效。