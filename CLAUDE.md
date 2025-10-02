# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioAst 是一个用于生物医学图像分析的深度学习训练平台,专注于 **70×70 像素菌落检测**的二分类任务:
- **Positive**: 图像中存在菌落
- **Negative**: 仅有气泡或无菌落

项目已实现并训练了 **30+ 种不同的模型架构**,包括完整的性能分析、ONNX 转换能力和扩展的评估框架。代码库采用模块化架构,具有组织良好的文件结构和自动化训练管道。

**关键特性**:
- 统一的模型训练框架,支持单模型和批量训练
- FUA (Flexible Unified Architecture) 高级架构层,提供统一的模型管理
- 多任务学习支持 (多级分类、灰度菌落检测等)
- 完整的 ONNX 转换和部署管道
- MLflow 实验跟踪和 Web 界面
- 生产级监控和自动回滚系统

## Architecture and Structure

### 架构层次

项目采用三层架构设计:

1. **基础模型层** ([models/](models/)): 30+ 种模型实现,每个模型提供 `create_<model_name>(num_classes=2)` 工厂函数
2. **训练框架层** ([training/](training/), [core/](core/)): 统一的训练、评估和可视化组件
3. **FUA 高级架构层** ([fua/](fua/)): 提供统一模型管理、实验跟踪、生产部署等高级功能

### Core Components

**Data Pipeline**:
- [training/dataset.py](training/dataset.py) - 基础 PyTorch Dataset 实现
- [training/multitask_dataset.py](training/multitask_dataset.py) - 多任务学习数据集
- [training/multilevel_dataset.py](training/multilevel_dataset.py) - 多级分类数据集
- `bioast_dataset/` - 主数据集目录 (train/val/test 结构)

**Model Definitions**:
- **核心模型** ([models/](models/)): 所有模型使用统一的工厂函数模式
  - [models/airbubble_hybrid_net.py](models/airbubble_hybrid_net.py) - 性能最佳 (98.02% 准确率)
  - [models/resnet_improved.py](models/resnet_improved.py) - 改进的 ResNet (97.83%)
  - [models/efficientnet.py](models/efficientnet.py) - EfficientNet-B0/B1
  - [models/mic_mobilenetv3.py](models/mic_mobilenetv3.py) - 移动端优化 (97.45%)
  - 更多模型见 [models/](models/) 目录

**Training Framework**:
- [training/trainer.py](training/trainer.py) - 通用模型训练器
- [training/multitask_trainer.py](training/multitask_trainer.py) - 多任务训练器
- [training/evaluator.py](training/evaluator.py) - 模型评估器
- [training/visualizer.py](training/visualizer.py) - 训练可视化

**FUA Architecture** ([fua/](fua/)):
- [fua/core/](fua/core/) - 核心接口和数据结构
  - [interfaces.py](fua/core/interfaces.py) - 统一的模型接口定义
  - [model_adapters.py](fua/core/model_adapters.py) - 模型适配器系统
  - [model_config.py](fua/core/model_config.py) - 分层配置管理
- [fua/model_integration/](fua/model_integration/) - 模型集成系统
- [fua/production/](fua/production/) - 生产级功能
  - [model_monitor.py](fua/production/model_monitor.py) - 模型监控
  - [auto_rollback.py](fua/production/auto_rollback.py) - 自动回滚
  - [ab_test_framework.py](fua/production/ab_test_framework.py) - A/B 测试
- [fua/experiment_tracking/](fua/experiment_tracking/) - MLflow 集成
- [fua/deployment/](fua/deployment/) - ONNX 导出和推理

**Conversion Pipeline**:
- [core/onnx_converter_base.py](core/onnx_converter_base.py) - 基础转换类
- [core/enhanced_onnx_converter_base.py](core/enhanced_onnx_converter_base.py) - 增强转换类
- [converters/](converters/) - 模型特定转换器
- [deployment/onnx_models/](deployment/onnx_models/) - 生产 ONNX 模型

### Configuration System

**配置优先级**: 基础配置 → 模型族配置 → 模型特定配置 → 运行时参数

**核心配置文件**:
- `core/config/model_configs.py` - 模型元数据和参数
- `core/config/training_configs.py` - 训练配置
- `config_template.yaml` - 标准化配置模板
- `dataset_config.json` - 数据集版本管理

## Development Environment Setup

**重要**: 项目使用本地虚拟环境 (`.venv`) 和 uv 进行包管理。

**环境规则**:
- **始终使用 .venv 环境**: 所有 Python 命令必须在 `.venv` 虚拟环境中运行
- **包安装**: 使用 `uv pip install <package>` 代替 `pip install`
- **环境激活**: 运行脚本前确保已激活 `.venv`
- **编码设置**: 在 Windows 上设置控制台编码为 UTF-8 避免中文显示问题
- **代码标准**: Python 脚本应使用英文输出以避免编码问题

**包管理命令**:
```bash
# 安装新包 (必需方法)
uv pip install <package_name>

# 从 requirements 安装
uv pip install -r requirements.txt

# 列出已安装包
uv pip list

# 升级包
uv pip install --upgrade <package_name>
```

**激活环境**:
```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

## Common Development Commands

### 训练模型

**单模型训练**:
```bash
# 列出可用模型
python train_single_model.py --list_models

# 训练指定模型
python train_single_model.py --model efficientnet_b0 --epochs 30 --batch_size 32 --lr 0.001

# 使用配置文件
python train_single_model.py --model ResNet18-Improved --config config_template.yaml
```

**批量训练**:
```bash
# 顺序训练所有模型
python scripts/auto_train_sequence.py

# 训练多个指定模型
python train_all_models.py
```

**多任务训练**:
```bash
# 训练多级分类模型
python train_enhanced_multilevel.py

# 训练稳定版 M16 多任务模型
python train_stable_m16_multitask.py
```

### 模型测试和验证

**模型测试**:
```bash
# 批量测试所有训练好的模型
python scripts/batch_test_models.py

# 验证 ONNX 模型
python scripts/batch_validate_all_onnx_models.py

# 快速验证 ONNX 模型
python scripts/quick_validate_all_onnx.py

# 检查训练进度
python scripts/check_test_progress.py
```

**模型分析**:
```bash
# 分析单个模型性能
python scripts/analyze_individual_models.py

# 生成综合模型分析
python scripts/comprehensive_model_analysis.py

# 对比模型性能
python compare_models.py --models EfficientNet-B0 ResNet18-Improved

# 对比性能最好的 3 个模型
python compare_models.py --top 3 --generate-report
```

### ONNX 转换

**单模型转换**:
```bash
python scripts/convert_single_model_to_onnx.py --model <model_name>
```

**批量转换**:
```bash
python scripts/batch_convert_models_to_onnx.py
```

### 数据集管理

**数据集操作**:
```bash
# 检查数据集状态
python dataset_manager.py --check

# 更新数据集
python dataset_manager.py --update-dataset "path/to/new/dataset"

# 重新训练所有模型
python dataset_manager.py --retrain-all
```

### FUA 高级功能

**使用 FUA 接口**:
```python
import fua

# 创建模型管理器
model_manager = fua.ModelManager()

# 创建模型
model_id = model_manager.create_model('airbubble_hybrid_net', {
    'learning_rate': 0.001,
    'batch_size': 32
})

# 训练模型
training_results = model_manager.train_model(model_id, {
    'samples': 1000,
    'image_size': [70, 70]
})

# 评估模型
eval_results = model_manager.evaluate_model(model_id)
```

**启动 Web 界面**:
```bash
# 启动 FUA Web UI
python start_web_ui.py

# 启动 MLflow UI
mlflow ui --port 5000
```

**运行 FUA 测试**:
```bash
# 运行所有 FUA 测试
cd fua
python tests/run_tests.py

# 运行集成测试
python -m pytest fua/tests/integration/

# 运行端到端测试
python -m pytest fua/tests/e2e/
```

### MobileNetV5 训练 (独立模块)

```bash
# 激活环境并检查
cd mobilenetv5
python check_env.py

# 快速测试 (5 epochs)
python train.py --model mobilenetv5 --config quick_test --test_only

# 标准训练 (50 epochs)
python train.py --model mobilenetv5 --config standard

# 训练小型变体
python train.py --model mobilenetv5_small --config standard

# 自定义参数训练
python train.py --model mobilenetv5 --config standard --batch_size 16 --learning_rate 0.0005 --num_epochs 75

# 评估模型
python evaluation.py --model mobilenetv5 --checkpoint experiments/mobilenetv5/model_best.pth
```

### 监控和分析

**训练监控**:
```bash
# 监控特定模型训练
python scripts/monitor_<model_name>_training.py

# 测试分布式监控
python test_distributed_monitoring_demo.py
```

**生成报告**:
```bash
# 生成详细分析报告
python scripts/generate_detailed_analysis.py

# 生成最终分析报告
python scripts/generate_final_analysis.py
```

## Build and Development Commands

**环境设置**:
```bash
# 安装依赖
uv pip install -r requirements.txt

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

**GPU 支持**: 所有训练脚本自动检测并使用 CUDA (如果可用)

**Python 路径**: 脚本会自动将项目根目录添加到 sys.path

## Model Performance Hierarchy

### 二分类模型 (Binary Classification)

当前最佳性能模型 (70×70 像素菌落检测):
1. **AirBubble_HybridNet**: 98.02% (CNN-Transformer 混合架构)
2. **ResNet18-Improved**: 97.83% (增强 ResNet)
3. **EfficientNet-B0**: 97.54% (高效 CNN)
4. **MIC_MobileNetV3**: 97.45% (移动端优化)
5. **Micro-ViT**: 97.36% (微型 Vision Transformer)

**新增模型**: MobileNetV5 模块 (独立实现):
- **MobileNetV5**: 2.8M 参数,带 SE 注意力机制
- **MobileNetV5 Small**: 1.6M 参数,适合快速推理
- 为 70×70 输入图像优化的现代架构

### 多任务学习模型 (Multitask Learning)

**多级分类系统** (Multilevel Classification):

当前最佳版本: **改进版 Multilevel MobileNetV3**
- **总体准确率**: 92.65% (验证集)
- **训练效率**: 10 epochs 达到最佳性能
- **参数量**: 2.29M
- **模型大小**: 8.80MB

**各任务性能** (改进版):
1. **Growth Level** (二分类): 98.13%
   - 精确率: 98.13%
   - 召回率: 98.13%
   - F1 分数: 98.13%

2. **Growth Pattern** (10分类): 86.07%
   - 精确率: 85.66%
   - 召回率: 86.07%
   - F1 分数: 85.76%
   - 主要类别: clustered, clean, weak_scattered, heavy_growth

3. **Interference Factors** (多标签): 92.64%
   - artifacts: 92.63%
   - contamination: 99.83%
   - debris: 95.30%
   - pores: 82.80%

**版本演进对比**:

| 版本 | 总体准确率 | 参数量 | 训练轮次 | 验证损失 | 状态 |
|------|------------|--------|----------|----------|------|
| 原版本 | 90.73% | 1.62M | 21轮 | 0.717 | 基准 |
| 简单优化版 | 91.61% | 2.29M | 3轮 | 1.036 | **推荐生产** |
| 改进版 | **92.65%** | 2.29M | **10轮** | 0.926 | **当前最佳** |
| 第一次优化版 | 87.50% | 4.08M | 8轮 | 1.170 | 已废弃 |
| 立即优化版 | 87.90% | 2.29M | 15轮 | 1.140 | 改进中 |

**关键发现**:
- ✅ **改进版表现最佳**: 92.65% 准确率,相比标准版提升 2.64%
- ✅ **训练效率显著提升**: 仅需 10 epochs,相比原版本减少 67%
- ✅ **各任务性能均衡**: Growth Level (98.13%), Growth Pattern (86.07%), Interference (92.64%)
- ⚠️ **复杂架构未必更好**: 第一次优化版参数量翻倍但性能下降 4.11%
- ✅ **轻量化策略成功**: 简单优化版以最少训练时间达到 91.61%

**优化策略总结**:
1. **任务权重平衡**: 统一设置为 (1.0, 1.0, 1.0) 效果最好
2. **学习率调度**: CosineAnnealingLR 优于 ReduceLROnPlateau
3. **早停机制**: patience=8-10 为最佳平衡点
4. **架构设计**: 轻量化 > 过度复杂化

所有模型针对 70×70 输入图像进行优化。

## Key Patterns and Conventions

### 模型实现模式
```python
def create_<model_name>(num_classes=2, **kwargs):
    """工厂函数,返回配置好的模型"""
    model = ModelClass(num_classes=num_classes, **kwargs)
    return model
```

每个模型应包含:
- `create_<model_name>()` 工厂函数
- `MODEL_CONFIG` 字典,包含模型元数据
- 输入尺寸验证 (默认 70×70)
- 设备自适应 (CPU/CUDA)

### 训练脚本模式
- 使用 `core/config/model_configs.py` 获取模型元数据
- 使用 `core/config/training_configs.py` 获取训练参数
- 保存检查点到 `experiments/experiment_<timestamp>/`
- 生成训练历史 JSON 和曲线图
- 自动记录到 MLflow (如果启用)

### ONNX 转换模式
- 继承 `OnnxConverterBase` 或 `EnhancedOnnxConverterBase`
- 实现模型特定的输入/输出处理
- 保存到 `onnx_models/` 或 `deployment/onnx_models/`
- 转换后进行模型验证

### FUA 集成模式
- 使用模型适配器包装现有模型
- 通过分层配置系统管理参数
- 实现 `IModel` 接口以支持高级功能
- 使用 `ModelCapability` 声明模型能力

### 文件组织
- **实验目录**: `experiments/experiment_<timestamp>/` (不提交到 git)
- **检查点**: 模型特定子目录,包含 `best.pth`, `latest.pth`
- **报告**: `reports/` 包含综合分析文件
- **脚本**: `scripts/` 包含所有自动化和训练脚本
- **FUA 模块**: `fua/` 独立模块,具有完整的测试套件
- **MobileNetV5**: `mobilenetv5/` 独立模块,具有独立的训练管道
- **功能目录**: 文件按功能分类:
  - `config/` - 配置文件和 JSON 结果
  - `documentation/` - 报告、HTML 分析和文档
  - `analysis/` - 分析和错误调查工具
  - `converters/` - ONNX 模型转换工具
  - `cleanup/` - 维护和清理脚本

### 架构模式
- **工厂模式**: 所有模型使用 `create_<model_name>(num_classes=2)` 工厂函数
- **配置驱动**: 训练参数和模型元数据集中在配置文件中
- **模块化训练**: 独立的训练器、评估器和可视化组件
- **管道架构**: 数据加载 → 训练 → 评估 → 转换 → 部署
- **注册表模式**: 模型注册表跟踪训练的模型和性能指标
- **适配器模式**: FUA 使用适配器包装现有模型而不修改原始代码

## Dataset Structure

预期数据集布局:
```
bioast_dataset/
├── train/
│   ├── negative/
│   └── positive/
├── val/
│   ├── negative/
│   └── positive/
└── test/
    ├── negative/
    └── positive/
```

**多任务数据集** (如需要):
```
bioast_dataset/
├── train/
│   ├── P1/  # 阳性 - 级别 1
│   ├── P2/  # 阳性 - 级别 2
│   ├── P3/  # 阳性 - 级别 3
│   └── N/   # 阴性
├── val/
└── test/
```

数据加载通过 `BioastDataset` 类自动处理结构。

## Important Notes

- **输入尺寸**: 模型使用 70×70 输入分辨率 (不是标准的 224×224)
- **分类任务**: 所有模型输出 2 类用于二分类
- **实验目录**: 排除在 git 之外 (.gitignore)
- **ONNX 模型**: 已提交用于部署
- **双语代码库**: 中文注释和英文代码混合
- **包管理**: 使用 uv 代替 pip
- **编码设置**: Windows 上设置控制台编码为 UTF-8
- **文件组织**: 项目已使用功能目录组织以提高可维护性
- **测试套件**: FUA 有完整的测试套件,核心模块通过验证脚本测试
- **核心入口文件**: 保留在根目录便于访问

## FUA Architecture Details

### FUA 发展阶段

**已完成**:
- ✅ Sprint 1: 核心基础设施 - 接口定义和数据结构
- ✅ Sprint 2: 模型集成系统 - 模型适配器框架和自动配置生成
- ✅ Sprint 6: 生产部署 - 监控、回滚、A/B 测试
- ✅ Sprint 8: 分布式监控优化
- ✅ Sprint 9: MLflow 集成和 Web 界面

**规划中**:
- 📋 Sprint 3+: 高级优化、自动化训练管道、持续改进机制

### FUA 核心概念

**分层配置系统**:
- 基础配置 → 模型族配置 → 模型特定配置
- 运行时参数覆盖
- 配置验证和版本控制

**模型适配器系统**:
- 包装现有模型而不修改原始代码
- 提供统一的接口
- 支持模型能力声明

**实验跟踪**:
- MLflow 集成记录所有实验
- 自动参数和指标记录
- Web UI 用于可视化和比较

**生产级功能**:
- 实时模型性能监控
- 性能降级自动检测
- 自动回滚机制
- A/B 测试框架

## Training Best Practices

### 多任务学习训练建议

**任务权重配置**:
- **推荐配置**: 统一权重 (1.0, 1.0, 1.0) 用于平衡训练
- **避免**: 过度偏向单一任务的权重配置
- **经验**: 权重比例不应超过 2:1

**训练参数优化**:
```python
# 推荐配置 (改进版 Multilevel)
config = {
    'batch_size': 64,          # 平衡训练速度和稳定性
    'learning_rate': 0.002,    # 初始学习率
    'weight_decay': 0.01,      # L2 正则化
    'num_epochs': 20,          # 足够的训练轮次
    'patience': 8-10,          # 早停耐心值
    'warmup_epochs': 5,        # 学习率预热
    'dropout_rate': 0.3        # 防止过拟合
}
```

**学习率调度器选择**:
- ✅ **推荐**: `CosineAnnealingLR` - 平滑衰减,收敛稳定
- ⚠️ **谨慎使用**: `ReduceLROnPlateau` - 可能导致训练不稳定
- 📝 **配置**: `T_max=num_epochs`, `eta_min=1e-6`

**避免常见陷阱**:
1. ❌ **过度复杂化**: 参数量翻倍不一定带来性能提升
2. ❌ **权重失衡**: 过度关注难度高的任务会损害其他任务
3. ❌ **训练不足**: patience 过小导致过早停止
4. ❌ **训练过度**: 过多 epochs 导致过拟合

### 性能优化检查清单

**训练前**:
- [ ] 验证数据集完整性和标注质量
- [ ] 确认任务权重合理性
- [ ] 设置合适的 batch_size (根据显存)
- [ ] 配置学习率调度器

**训练中**:
- [ ] 监控验证损失和准确率曲线
- [ ] 检查各任务性能是否平衡
- [ ] 观察是否出现过拟合迹象
- [ ] 记录最佳检查点

**训练后**:
- [ ] 对比各版本性能指标
- [ ] 分析混淆矩阵找出弱点
- [ ] 验证模型在测试集上的泛化能力
- [ ] 生成详细的性能报告

## Troubleshooting Common Issues

### 训练相关问题

**训练不稳定 / 性能波动大**:
- 原因: 学习率过高、任务权重失衡、batch_size 过小
- 解决方案:
  - 降低初始学习率 (0.002 → 0.001)
  - 使用 CosineAnnealingLR 替代 ReduceLROnPlateau
  - 增加 batch_size (32 → 64)
  - 检查任务权重配置,确保平衡

**模型性能不如预期**:
- 检查训练配置是否使用推荐参数
- 对比版本演进表,确认使用最佳版本
- 分析各任务性能,找出瓶颈任务
- 参考改进版 (92.65%) 的配置

**过拟合问题**:
- 增加 dropout_rate (0.3 → 0.4)
- 增加 weight_decay (0.01 → 0.02)
- 使用数据增强
- 减少训练轮次或提前早停

**收敛速度慢**:
- 增加学习率 (谨慎调整)
- 使用学习率预热 (warmup_epochs=3-5)
- 检查 batch_size 是否过小
- 验证数据加载是否存在瓶颈

### 系统相关问题

**CUDA 内存不足**:
- 减小 batch_size (64 → 32 → 16)
- 使用梯度累积
- 启用混合精度训练 (AMP)
- 减小模型参数量

**模型加载失败**:
- 检查检查点路径是否正确
- 验证模型架构与保存时一致
- 确保 num_classes 配置正确
- 检查 PyTorch 版本兼容性

**ONNX 转换错误**:
- 验证输入尺寸 (70×70)
- 检查模型中的动态操作
- 使用 `enhanced_onnx_converter_base.py` 提高兼容性
- 设置 opset_version=11 或更高

**数据集加载问题**:
- 验证目录结构是否符合规范
- 检查图像格式 (支持 PNG, JPG)
- 确认类别文件夹命名正确
- 检查 JSON 标注文件格式

**性能数据不一致**:
- 验证实验目录是否包含完整的训练历史
- 检查 training_summary.json 文件
- 对比多次训练结果确认可复现性
- 使用 MLflow 追踪所有实验

## Additional Resources

### 核心文档

**架构设计**:
- [FUA 需求文档](FUA_REQUIREMENTS.md) - 详细的用户故事和验收标准
- [FUA 高级设计](FUA_HIGH_LEVEL_DESIGN.md) - 架构设计和模块说明
- [项目结构设计](PROJECT_STRUCTURE.md) - 文件组织和模块划分

**部署指南**:
- [ONNX 部署指南](deployment/ONNX_Deployment_Guide.md) - ONNX 模型部署
- [C# 部署指南](deployment/CSharp_Deployment_Guide.md) - .NET 集成
- [快速操作指南](QUICK_OPERATION_GUIDE.md) - 详细使用步骤
- [系统介绍](BIOAST_SYSTEM_INTRODUCTION.md) - 完整系统架构

### 训练结果与分析报告

**多任务学习性能报告**:
- [四版本全面对比](comprehensive_four_version_comparison_report.md) - 原版本/简单优化版/第一次优化版/立即优化版对比
- [改进版性能分析](improved_multilevel_performance_analysis.md) - 改进版 Multilevel MobileNetV3 详细分析 (92.65%)
- [性能改进总结](performance_improvement_summary.md) - 立即优化版改进措施和效果
- [版本对比报告](multilevel_vs_simple_enhanced_comparison_report.md) - Multilevel 与 Simple Enhanced 对比
- [性能差距分析](performance_gap_analysis.md) - 各版本性能差距深度分析

**关键发现总结**:
- [当前版本关键问题分析](current_version_critical_issues_analysis.md) - Simple Enhanced 版本问题诊断
- [优化趋势分析](optimization_trends_analysis.md) - 优化策略演进分析
- [置信度阈值分析](confidence_threshold_analysis.md) - 阈值对性能的影响

**模型对比分析**:
- [综合模型对比](reports/model_comparison/comprehensive_model_comparison.md) - 所有模型横向对比
- [综合性能分析](reports/comprehensive_performance_analysis.md) - 性能指标详细分析
- [最终完整分析](reports/final_complete_analysis.md) - 最终分析报告
- [简单模型对比](reports/simple_model_comparison.md) - 轻量级模型对比

**增强版 MobileNetV3 系列**:
- [MIC MobileNetV3 优化报告](MIC_MobileNetV3_Optimization_Report.md) - MIC 版本优化分析
- [增强版性能分析](Enhanced_MIC_MobileNetV3_Performance_Analysis.md) - 增强版详细性能
- [大模型性能分析](large_model_performance_analysis.md) - 大规模模型性能评估

### 实验数据

**训练摘要** (最新):
- `experiments/improved_multilevel/training_summary.json` - 改进版训练结果 (92.65%)
- `experiments/*/training_summary.json` - 其他版本训练历史

**MLflow 实验追踪**:
- 启动 MLflow UI 查看所有实验: `mlflow ui --port 5000`
- 实验数据存储: `mlruns/` 目录

### 配置示例

**训练配置**:
- `config_template.yaml` - 标准化配置模板
- `dataset_config.json` - 数据集版本管理示例

**推荐配置** (改进版):
```yaml
model:
  size: small
  input_channels: 1
  dropout_rate: 0.3

training:
  batch_size: 64
  learning_rate: 0.002
  weight_decay: 0.01
  num_epochs: 20
  warmup_epochs: 5
  patience: 10

task_weights:
  growth_level: 1.0
  growth_pattern: 1.0
  interference_factors: 1.0
```

### 快速开始指南

**新手推荐**:
1. 阅读 [系统介绍](BIOAST_SYSTEM_INTRODUCTION.md) 了解整体架构
2. 查看 [快速操作指南](QUICK_OPERATION_GUIDE.md) 学习基本操作
3. 参考 [改进版性能分析](improved_multilevel_performance_analysis.md) 了解最佳实践
4. 使用推荐配置开始训练

**高级用户**:
1. 研究 [FUA 高级设计](FUA_HIGH_LEVEL_DESIGN.md) 理解架构
2. 阅读 [四版本全面对比](comprehensive_four_version_comparison_report.md) 了解优化历程
3. 使用 FUA 接口进行高级定制
4. 集成 MLflow 进行实验管理
