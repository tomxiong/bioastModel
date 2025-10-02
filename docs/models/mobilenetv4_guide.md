# MultiLevelMobileNetV4 使用指南

## 📋 模型概述

MultiLevelMobileNetV4 是基于 MobileNetV4 架构的多级分类模型,专为 70×70 像素的细菌图像分类设计。

### 核心特性

- ✅ **MobileNetV4 架构**: 使用 Universal Inverted Bottleneck (UIB) 模块
- ✅ **注意力机制**: 集成 SE 和 ECA 注意力块
- ✅ **多任务学习**: 三个分类任务同时优化
- ✅ **轻量高效**: 参数量 0.95M - 1.83M
- ✅ **灰度图像优化**: 针对单通道 70×70 图像优化

### 参考来源

- LARS-MobileNet-V4: https://github.com/lars-uav/LARS-MobileNet-V4
- 改进版 multilevel_mobilenetv3 的成功经验 (92.65% 准确率)

## 🏗️ 模型架构

### 三种模型大小

| 模型 | 参数量 | Width Mult | 适用场景 |
|------|--------|------------|----------|
| **Small** | 952K | 0.75 | 快速推理,资源受限 |
| **Medium** | 1.33M | 1.0 | 平衡性能和效率 |
| **Large** | 1.83M | 1.25 | 追求最高精度 |

### 任务定义

1. **Growth Level** (二分类)
   - 类别: positive / negative
   - 任务: 判断是否存在菌落生长

2. **Growth Pattern** (10分类)
   - 类别: 10种不同的生长模式
   - 任务: 识别具体的生长特征

3. **Interference Factors** (多标签)
   - 类别: 4种干扰因子
   - 任务: 检测图像中的干扰因素

## 🚀 快速开始

### 1. 创建模型

```python
from models.multilevel_mobilenetv4 import (
    create_multilevel_mobilenetv4_small,
    create_multilevel_mobilenetv4_medium,
    create_multilevel_mobilenetv4_large
)

# 创建小型模型
model = create_multilevel_mobilenetv4_small(
    input_channels=1,
    dropout_rate=0.3
)

# 查看模型信息
info = model.get_model_info()
print(f"Total Parameters: {info['total_parameters']:,}")
```

### 2. 训练模型

#### 基础训练 (推荐配置)

```bash
python scripts/multilevel_training/train_mobilenetv4.py \
    --model_size small \
    --batch_size 64 \
    --learning_rate 0.002 \
    --num_epochs 20 \
    --patience 10
```

#### 自定义配置

```bash
python scripts/multilevel_training/train_mobilenetv4.py \
    --model_size medium \
    --data_root /path/to/dataset \
    --json_path /path/to/annotations.json \
    --batch_size 64 \
    --learning_rate 0.002 \
    --weight_decay 0.01 \
    --num_epochs 30 \
    --warmup_epochs 5 \
    --patience 10 \
    --dropout_rate 0.3 \
    --growth_level_weight 1.0 \
    --growth_pattern_weight 1.0 \
    --interference_weight 1.0
```

### 3. 模型推理

```python
import torch
from models.multilevel_mobilenetv4 import create_multilevel_mobilenetv4_small

# 加载模型
model = create_multilevel_mobilenetv4_small()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# 推理
with torch.no_grad():
    # 输入: [batch_size, 1, 70, 70]
    x = torch.randn(1, 1, 70, 70)
    outputs = model(x)

    # 获取预测结果
    growth_level_pred = outputs['growth_level'].argmax(dim=1)
    growth_pattern_pred = outputs['growth_pattern'].argmax(dim=1)
    interference_pred = (outputs['interference_factors'].sigmoid() > 0.5).float()
```

## ⚙️ 训练配置

### 推荐配置 (基于改进版 V3 的成功经验)

```yaml
# 模型配置
model_size: small
input_channels: 1
dropout_rate: 0.3

# 训练参数
batch_size: 64
learning_rate: 0.002
weight_decay: 0.01
num_epochs: 20
warmup_epochs: 5
patience: 10

# 任务权重 (统一权重效果最好)
task_weights:
  growth_level: 1.0
  growth_pattern: 1.0
  interference_factors: 1.0

# 数据划分
train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
```

### 关键训练策略

1. **学习率调度**: 使用 `CosineAnnealingLR` (效果优于 ReduceLROnPlateau)
2. **任务权重**: 统一设为 1.0 (不平衡权重会损害整体性能)
3. **早停机制**: patience=10 为最佳平衡点
4. **Warmup**: 5 epochs 的学习率预热提升收敛稳定性

## 📊 性能目标

基于改进版 MobileNetV3 的基准 (92.65%),MobileNetV4 的目标:

| 任务 | 目标准确率 | 基准 (V3) |
|------|-----------|-----------|
| **Overall** | **>93%** | 92.65% |
| Growth Level | >98% | 98.13% |
| Growth Pattern | >87% | 86.07% |
| Interference Factors | >93% | 92.64% |

### 优化重点

1. **Growth Pattern**: 当前最具挑战性,重点提升
2. **收敛速度**: 目标在 15 epochs 内达到最佳性能
3. **参数效率**: 保持轻量级,参数量 < 2M

## 🔧 常见问题

### Q1: 如何选择模型大小?

- **Small**: 推荐用于快速实验和资源受限场景
- **Medium**: 推荐用于生产环境,平衡性能和效率
- **Large**: 追求最高精度,资源充足时使用

### Q2: 训练不稳定怎么办?

参考训练最佳实践:
- 降低学习率 (0.002 → 0.001)
- 增加 batch_size (32 → 64)
- 检查任务权重是否平衡
- 确认使用 CosineAnnealingLR

### Q3: 如何提升 Growth Pattern 性能?

- 增加数据增强强度
- 调整 dropout_rate (0.3 → 0.4)
- 适当增加训练轮次
- 使用 Focal Loss (可选)

## 📈 与其他模型对比

| 模型 | 参数量 | 准确率 | 训练时间 | 推理速度 |
|------|--------|--------|----------|----------|
| MobileNetV3 (原版) | 1.62M | 90.73% | 21 epochs | 快 |
| MobileNetV3 (改进版) | 2.29M | **92.65%** | 10 epochs | 中等 |
| **MobileNetV4 (Small)** | **0.95M** | **目标>93%** | **<15 epochs** | **最快** |
| MobileNetV4 (Medium) | 1.33M | 目标>93% | <15 epochs | 快 |
| MobileNetV4 (Large) | 1.83M | 目标>93% | <15 epochs | 中等 |

### 优势

- ✅ **参数更少**: Small 版本仅 0.95M 参数
- ✅ **架构更新**: UIB 模块 + 双注意力机制
- ✅ **训练更快**: 优化的架构加速收敛
- ✅ **精度更高**: 目标超越改进版 V3

## 📚 相关文档

- [改进版性能分析](../performance_analysis/improved_multilevel_performance_analysis.md)
- [训练最佳实践](../../CLAUDE.md#training-best-practices)
- [四版本对比报告](../performance_analysis/comprehensive_four_version_comparison_report.md)

## 🔗 参考资料

- LARS-MobileNet-V4: https://github.com/lars-uav/LARS-MobileNet-V4
- MobileNetV4 Paper: https://arxiv.org/abs/2404.10518
- 项目文档: [CLAUDE.md](../../CLAUDE.md)
