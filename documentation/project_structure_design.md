# 标准化项目结构设计

## 新目录结构规划

```
bioastModel/
├── core/                           # 核心功能模块
│   ├── __init__.py
│   ├── config/                     # 配置管理
│   │   ├── __init__.py
│   │   ├── model_configs.py        # 模型配置
│   │   ├── training_configs.py     # 训练配置
│   │   └── paths.py               # 路径配置
│   ├── models/                     # 模型定义（保持现有）
│   │   ├── __init__.py
│   │   ├── efficientnet.py
│   │   ├── resnet_improved.py
│   │   └── model_factory.py       # 模型工厂
│   ├── training/                   # 训练相关（保持现有）
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── visualizer.py
│   ├── evaluation/                 # 评估模块
│   │   ├── __init__.py
│   │   ├── model_evaluator.py     # 统一模型评估
│   │   ├── report_generator.py    # 报告生成器
│   │   └── metrics_calculator.py  # 指标计算
│   └── comparison/                 # 对比分析模块
│       ├── __init__.py
│       ├── model_comparator.py    # 模型对比器
│       ├── visualization.py       # 对比可视化
│       └── report_builder.py      # 对比报告构建
├── scripts/                        # 标准化脚本
│   ├── train_model.py             # 统一训练脚本
│   ├── evaluate_model.py          # 统一评估脚本
│   ├── compare_models.py          # 统一对比脚本
│   ├── add_new_model.py           # 新模型添加脚本
│   └── cleanup_experiments.py     # 实验清理脚本
├── templates/                      # 模板文件
│   ├── model_template.py          # 新模型模板
│   ├── training_template.py       # 训练脚本模板
│   ├── evaluation_template.py     # 评估脚本模板
│   └── report_templates/          # 报告模板
│       ├── html_template.html
│       └── markdown_template.md
├── reports/                        # 统一报告输出
│   ├── individual/                # 单模型报告
│   ├── comparisons/               # 对比报告
│   └── summaries/                 # 总结报告
├── experiments/                    # 实验结果（保持现有结构）
│   ├── experiment_YYYYMMDD_HHMMSS/
│   │   └── model_name/
│   │       ├── config.json
│   │       ├── best_model.pth
│   │       ├── training_history.json
│   │       ├── test_results.json
│   │       ├── evaluation/
│   │       ├── sample_analysis/
│   │       └── visualizations/
├── docs/                          # 文档
│   ├── user_guide.md             # 用户指南
│   ├── api_reference.md          # API参考
│   ├── model_comparison_guide.md # 模型对比指南
│   └── troubleshooting.md        # 故障排除
├── legacy/                        # 遗留文件（临时存放）
└── [现有根目录文件保持不变]
```

## 标准化命名规范

### 文件命名
- **脚本文件**: `action_target.py` (如: `train_model.py`, `evaluate_resnet.py`)
- **模块文件**: `功能名.py` (如: `model_evaluator.py`, `report_generator.py`)
- **配置文件**: `config_type.py` (如: `model_configs.py`, `training_configs.py`)

### 目录命名
- 使用小写字母和下划线
- 功能相关的目录分组
- 避免缩写，使用完整单词

### 实验命名
- 格式: `experiment_YYYYMMDD_HHMMSS`
- 模型子目录: 使用模型的标准名称 (如: `efficientnet_b0`, `resnet18_improved`)

## 配置管理系统

### 模型配置 (`core/config/model_configs.py`)
```python
MODEL_CONFIGS = {
    'efficientnet_b0': {
        'class_name': 'EfficientNetB0',
        'params': 1.56,
        'input_size': 70,
        'num_classes': 2
    },
    'resnet18_improved': {
        'class_name': 'ResNet18Improved', 
        'params': 11.26,
        'input_size': 70,
        'num_classes': 2
    }
}
```

### 路径配置 (`core/config/paths.py`)
```python
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'bioast_dataset')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

def get_experiment_path(model_name):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(EXPERIMENTS_DIR, f'experiment_{timestamp}', model_name)
```

## 标准化流程设计

### 新模型添加流程
1. **模型定义**: 在 `core/models/` 中添加模型类
2. **配置注册**: 在 `model_configs.py` 中注册模型配置
3. **训练脚本**: 使用 `scripts/train_model.py --model MODEL_NAME`
4. **评估报告**: 使用 `scripts/evaluate_model.py --model MODEL_NAME`
5. **对比分析**: 使用 `scripts/compare_models.py --models MODEL1,MODEL2`

### 统一接口设计
所有脚本都支持统一的命令行参数：
- `--model`: 指定模型名称
- `--config`: 指定配置文件
- `--output`: 指定输出目录
- `--experiment`: 指定实验ID

## 向后兼容性

### 保持现有功能
- 现有的 `models/`, `training/`, `experiments/` 目录结构不变
- 现有的训练脚本继续可用
- 现有的实验结果保持原有路径

### 渐进迁移策略
1. 新功能使用新结构
2. 逐步将现有功能迁移到新模块
3. 保留旧脚本的兼容性包装器

## 实施优先级

### 第一优先级（立即实施）
1. 创建 `core/config/` 配置管理
2. 创建 `scripts/` 标准化脚本
3. 整合重复的可视化和报告功能

### 第二优先级（后续实施）
1. 创建 `templates/` 模板系统
2. 建立 `reports/` 统一报告输出
3. 完善 `docs/` 文档系统

### 第三优先级（长期目标）
1. 迁移现有功能到新结构
2. 清理 `legacy/` 遗留文件
3. 完全自动化的工作流程

这个设计确保了在不破坏现有功能的前提下，建立标准化的项目结构和工作流程。