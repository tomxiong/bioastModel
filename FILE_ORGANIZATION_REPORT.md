# 文件整理报告

## 整理概述

根据核心功能对项目文件进行了分类整理，将分散在根目录的文件按照功能分类到不同的目录中。

## 目录结构

### 根目录核心文件
- `README.md` - 项目说明文档
- `CLAUDE.md` - Claude Code 开发指南
- `requirements.txt` - 项目依赖
- `main.py` - 主程序入口
- `quick_start.py` - 快速启动脚本
- `dataset_manager.py` - 数据集管理器
- `model_manager.py` - 模型管理器
- `train_single_model.py` - 单模型训练脚本
- `compare_models.py` - 模型对比脚本
- `config_template.yaml` - 配置模板
- `training_config.json` - 训练配置
- `model_registry.json` - 模型注册表
- `sync_bioast_dataset.sh` - 数据集同步脚本

### 分类目录

#### `config/` - 配置文件
包含所有 JSON 配置文件和训练结果文件：
- 模型训练结果 JSON 文件
- 性能分析报告 JSON 文件
- ONNX 转换结果文件
- 检查点清理报告文件

#### `documentation/` - 文档文件
包含所有文档和报告：
- HTML 格式的分析报告
- Markdown 格式的指南和文档
- 模型分析报告
- 训练进度摘要

#### `analysis/` - 分析相关
- 分析脚本和工具
- 错误分析文件

#### `cleanup/` - 清理相关
- 清理脚本和维护工具

#### `converters/` - 转换器
- ONNX 模型转换器

#### `core/` - 核心功能
- 核心模块和工具
- 配置管理
- 数据加载器

#### `models/` - 模型定义
- 所有模型架构定义
- 模型配置文件

#### `training/` - 训练相关
- 训练器实现
- 评估器
- 可视化工具

#### `scripts/` - 脚本文件
- 自动化脚本
- 批处理脚本

#### `deployment/` - 部署相关
- 部署配置和文件

#### `utils/` - 工具函数
- 通用工具函数

#### `checkpoints/` - 模型检查点
- 训练好的模型权重
- 检查点文件

#### `reports/` - 报告文件
- 生成的分析报告
- 性能对比报告

#### `experiments/` - 实验目录
- 训练实验结果
- 实验配置文件

## 文件统计

根据 organization_report.json 的统计：
- config: 33 个文件
- documentation: 38 个文件
- model_related: 75 个文件
- core: 12 个文件
- uncategorized: 43 个文件（主要是备份文件和隐藏目录）

## 使用说明

1. **配置文件**：所有 JSON 配置文件现在位于 `config/` 目录
2. **文档查看**：所有文档和报告位于 `documentation/` 目录
3. **模型开发**：模型相关文件位于 `models/` 目录
4. **训练脚本**：训练相关脚本位于 `training/` 目录
5. **分析工具**：分析脚本位于 `analysis/` 目录

## 维护建议

1. 新增文件时请按照功能分类放置到对应目录
2. 定期清理 `config/` 目录中的临时文件
3. 更新文档时请放置到 `documentation/` 目录
4. 模型开发请在 `models/` 目录进行

## 注意事项

- 核心入口文件保持在根目录以便于访问
- 隐藏目录和系统文件保持原位置
- 备份文件（.backup_*) 主要在根目录，可定期清理
- 虚拟环境 `.venv` 保持原位置