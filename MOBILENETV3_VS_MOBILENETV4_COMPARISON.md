# MobileNetV3 v0.10.0 vs MobileNetV4 v0.11.0 架构对比报告

## 📋 实验概述

本报告对比了基于相同 Pattern-Conditional Loss 的两种不同架构:
- **MobileNetV3 v0.10.0**: 传统 MobileNetV3 架构
- **MobileNetV4 v0.11.0**: 先进 MobileNetV4 架构 (UIB + SE/ECA)

**实验设置完全一致**:
- 数据集: `m9e1n170_cleaned_round2.json` (数据清洗版)
- 数据划分: `dataset_split_seed44.json` (固定划分)
- Pattern-Conditional Loss: Negative 15.0, Positive Critical 15.0, Other 0.1
- 任务权重: [1.0, 2.0, 1.5]
- 类别权重: [8.0, 3.0, 5.0, 10.0]
- 训练参数: 相同学习率、batch size、优化器配置

## 🏗️ 架构对比

### MobileNetV3 v0.10.0

**架构特点**:
- **基础模块**: Inverted Residual Blocks
- **注意力机制**: SE (Squeeze-and-Excitation)
- **激活函数**: Hard-Swish
- **参数量**: 1.62M
- **模型大小**: ~19.69 MB

**优势**:
- ✅ 成熟稳定的架构
- ✅ 经过充分验证
- ✅ 参数量适中

**劣势**:
- ⚠️ 架构相对传统
- ⚠️ 特征表达能力有限

### MobileNetV4 v0.11.0

**架构特点**:
- **基础模块**: Universal Inverted Bottleneck (UIB)
  - 统一可配置的模块设计
  - Expansion + Depthwise + Attention + Projection
- **注意力机制**: SE + ECA (Efficient Channel Attention)
  - 双重注意力机制提升特征表达
- **激活函数**: ReLU6 (更稳定)
- **参数量**: 0.95M (**减少 41.4%**)
- **模型大小**: ~11.71 MB (**减少 40.5%**)

**优势**:
- ✅ **更轻量**: 参数量仅为 MobileNetV3 的 58.6%
- ✅ **更现代**: UIB 设计更灵活可配置
- ✅ **双重注意力**: SE + ECA 增强特征表达

**劣势**:
- ⚠️ 架构更复杂,训练难度更高
- ⚠️ 需要更精细的超参数调优

## 📊 性能对比 (验证集最佳性能)

### Overall Summary

| 指标 | MobileNetV3 v0.10.0 | MobileNetV4 v0.11.0 | 差距 |
|------|---------------------|---------------------|------|
| **参数量** | 1.62M | 0.95M | **-41.4%** ✅ |
| **模型大小** | 19.69 MB | 11.71 MB | **-40.5%** ✅ |
| **最佳 Epoch** | 19 | 14 | **-26.3%** ✅ |
| **验证损失** | 1.739 | 1.711 | **-1.6%** ✅ |

### Growth Level (二分类)

| 指标 | MobileNetV3 v0.10.0 | MobileNetV4 v0.11.0 | 差距 |
|------|---------------------|---------------------|------|
| **Accuracy** | **98.73%** | 98.73% | 0.00% ≈ |

**结论**: 两种架构在 Growth Level 上表现相同,均达到极高准确率。

### Growth Pattern (10分类)

| 指标 | MobileNetV3 v0.10.0 | MobileNetV4 v0.11.0 | 差距 |
|------|---------------------|---------------------|------|
| **Accuracy** | **87.05%** | 87.19% | **+0.14%** ≈ |

**结论**: MobileNetV4 略优,但差距极小。

### Interference Factors (多标签)

| 指标 | MobileNetV3 v0.10.0 | MobileNetV4 v0.11.0 | 差距 |
|------|---------------------|---------------------|------|
| **Overall** | **49.52%** | 44.89% | **-4.63%** ⚠️ |

**详细分析** (基于测试集数据 v0.10.0):

#### Pores (核心指标)

| 指标 | MobileNetV3 v0.10.0 | MobileNetV4 v0.11.0 | 差距 |
|------|---------------------|---------------------|------|
| **F1 Score** | **91.76%** | ~85-90% (估计) | **-2~7%** ⚠️ |
| **Precision** | 94.05% | ? | ? |
| **Recall** | 89.58% | ? | ? |

> ⚠️ **注意**: v0.11.0 的 Pores 性能基于 Overall Interference 指标推测,实际需要完整测试评估确认。

#### 其他 Interference Factors

MobileNetV4 在 Interference Factors 整体表现下降,可能原因:
1. 轻量化架构容量不足
2. 多标签任务需要更强特征表达
3. 需要针对性调优

## 🔍 深度分析

### MobileNetV3 v0.10.0 的优势

1. **Pores 检测突破性**:
   - Pattern-Conditional Loss 成功解决代理学习问题
   - Pores F1 从 v0.9.9 的 28.57% 提升到 **91.76%** (+63.19%)
   - 业务关键样本 Recall 达到 92.06% (目标 ≥75%)

2. **各任务平衡**:
   - Growth Level: 98.40% ✅
   - Growth Pattern: 87.05% ✅
   - Interference: 45.54% (Pores 91.76% ✅)

3. **训练稳定性**:
   - 收敛平稳,无过拟合
   - 最佳 epoch 19 (共 34 epochs)

### MobileNetV4 v0.11.0 的表现

1. **效率优势明显**:
   - 参数量减少 41.4%
   - 更快收敛 (epoch 14 vs 19)
   - 验证损失略优 (1.711 vs 1.739)

2. **Growth 任务表现相当**:
   - Growth Level: 98.73% ≈ MobileNetV3
   - Growth Pattern: 87.19% ≈ MobileNetV3

3. **Interference 任务不足**:
   - Overall: 44.89% < MobileNetV3 49.52% ⚠️
   - 可能因架构容量限制导致多标签任务表现下降

## 🎯 架构对比结论

### MobileNetV3 v0.10.0 ✅ **推荐用于生产**

**优势**:
- ✅ **Pores 检测性能最佳**: F1 91.76%,完全满足业务需求
- ✅ **各任务平衡稳定**: Growth、Pattern、Interference 全面达标
- ✅ **训练成熟可靠**: Pattern-Conditional Loss 效果经过验证

**劣势**:
- 参数量和模型大小较 MobileNetV4 多 40%

**适用场景**:
- ✅ 生产环境首选
- ✅ 对 Pores 检测精度要求高的场景
- ✅ 需要稳定可靠性能的场景

### MobileNetV4 v0.11.0 ⚠️ **需进一步优化**

**优势**:
- ✅ **极度轻量**: 参数量仅 0.95M,模型仅 11.71 MB
- ✅ **更快收敛**: 训练效率高,Epoch 14 即达最佳
- ✅ **Growth 任务优秀**: Level 和 Pattern 性能与 MobileNetV3 相当

**劣势**:
- ⚠️ **Interference 性能不足**: Overall 44.89% < MobileNetV3 49.52%
- ⚠️ **Pores 性能未知**: 缺少完整测试评估,推测可能下降

**优化建议**:
1. **增加模型容量**:
   - 使用 MobileNetV4 Medium/Large 变体
   - 增加 UIB 深度或宽度
2. **调整任务权重**:
   - 提高 Interference Factors 权重 (1.5 → 2.0)
   - 单独优化 Pores 损失权重
3. **数据增强**:
   - 针对 Pores 样本增强
   - 平衡多标签样本分布
4. **完整测试评估**:
   - 获取详细 Pores 性能指标
   - 分析 FN/FP 错误案例

**适用场景**:
- 🔬 资源受限环境 (边缘设备、移动端)
- 🔬 只关注 Growth Level/Pattern,对 Interference 要求不高
- 🔬 研究实验,探索轻量化极限

## 📈 性能趋势

### v0.9.8 → v0.9.9 → v0.10.0 → v0.11.0 演进

| 版本 | 核心创新 | Pores F1 | 参数量 | 状态 |
|------|----------|----------|--------|------|
| v0.9.8 | 数据清洗 Round 2 | **0.00%** | 1.62M | 基准 |
| v0.9.9 | 提升 Interference 权重 | 28.57% | 1.62M | 改进 |
| **v0.10.0** | **Pattern-Conditional Loss** | **91.76%** | **1.62M** | **突破** ✅ |
| v0.11.0 | MobileNetV4 架构 | ~85-90%? | 0.95M | 探索 🔬 |

**关键里程碑**:
- ✅ **v0.10.0**: Pattern-Conditional Loss 实现 Pores 检测突破 (0% → 91.76%)
- 🔬 **v0.11.0**: 架构轻量化探索,参数减少 41.4%,但 Interference 性能下降

## 🚀 下一步建议

### 短期 (v0.11.1 优化)

1. **完整测试评估** ⭐⭐⭐
   - 获取 v0.11.0 详细测试集性能
   - 对比 v0.10.0 各指标差距
   - 分析 Pores 错误案例

2. **架构容量提升**:
   - 训练 MobileNetV4 Medium 变体
   - 验证容量对 Interference 性能的影响

3. **超参数调优**:
   - 调整任务权重 (提升 Interference 权重)
   - 调整 Pores 损失权重 (15.0 → 20.0)

### 中期 (架构融合)

1. **混合架构**:
   - MobileNetV4 Backbone + 增强 Interference Head
   - 保持轻量化同时提升 Pores 性能

2. **多阶段训练**:
   - Stage 1: 冻结 Backbone,训练 Pattern-Conditional Loss
   - Stage 2: Fine-tune 全模型

### 长期 (架构探索)

1. **Transformer 集成**:
   - UIB + Self-Attention 增强全局特征
   - 专门针对 Pores 的注意力模块

2. **知识蒸馏**:
   - v0.10.0 作为 Teacher 模型
   - 蒸馏到轻量 MobileNetV4 Student

## 📌 总结

### 当前最佳方案: MobileNetV3 v0.10.0 ✅

**理由**:
1. **Pores F1 91.76%**: 完全满足业务需求
2. **全面均衡**: Growth、Pattern、Interference 全达标
3. **生产可靠**: Pattern-Conditional Loss 效果经过验证

### MobileNetV4 探索价值

**积极发现**:
- 参数量减少 41.4%,证明架构轻量化可行性
- Growth 任务表现相当,说明基础特征提取能力足够

**待解决问题**:
- Interference 性能下降需优化
- 需完整测试确认 Pores 性能

**结论**: MobileNetV4 架构有潜力,但需进一步调优才能达到 v0.10.0 水平。当前**推荐 v0.10.0 用于生产环境**。
