# MobileNetV4 所有版本性能验证报告

## 执行摘要

**验证日期**: 2025-10-03
**验证方法**: 统一测试集 (3000 样本)
**发现**: 🔴 **所有版本的训练报告数据均存在严重虚高问题**

---

## 📊 所有版本性能对比

### 完整对比表

| 版本 | 模型规模 | 指标 | 训练报告声称 | 实际测试集 | 差异 | 状态 |
|------|---------|------|-------------|-----------|------|------|
| **v1.1** | Small (0.95M) | Growth Level | 98.73% | **90.87%** | -7.87% | ❌ |
| | | Growth Pattern | 88.10% | **55.57%** | -32.53% | ❌ |
| | | Interference (准确率) | 95.96% | 76.56% | -19.40% | ❌ |
| | | **Interference (F1)** | N/A | **5.08%** | - | 🔴 极差 |
| | | **加权平均** | ~94% | **59.59%** | -34.41% | ❌ |
| **v1.2** | Medium (1.33M) | Growth Level | 98.80% | **96.13%** | -2.67% | ⚠️ |
| | | Growth Pattern | 88.86% | **57.03%** | -31.83% | ❌ |
| | | Interference (准确率) | 95.98% | 72.56% | -23.42% | ❌ |
| | | **Interference (F1)** | N/A | **0.87%** | - | 🔴 极差 |
| | | **加权平均** | ~94% | **61.44%** | -32.56% | ❌ |
| **Small Quick** | Small (0.95M) | Growth Level | 98.57% | **66.90%** | -31.67% | ❌ |
| *(可能v1.0)* | | Growth Pattern | 85.33% | **44.03%** | -41.30% | ❌ |
| | | Interference (准确率) | 94.76% | 75.43% | -19.33% | ❌ |
| | | **Interference (F1)** | N/A | **1.82%** | - | 🔴 极差 |
| | | **加权平均** | ~93% | **44.74%** | -48.26% | ❌ |

### 关键发现

1. **所有版本都存在相同的问题** ✅ 确认
   - Growth Pattern 虚高 **30-40%**
   - Interference 准确率虚高 **20-30%**
   - Interference F1 实际接近 **0%**（完全失败）

2. **版本间实际性能对比**:
   - **v1.2 (Medium)** 表现最好: 61.44% 综合准确率
   - **v1.1 (Small)** 次之: 59.59% 综合准确率
   - **Small Quick (v1.0?)** 最差: 44.74% 综合准确率

3. **各任务真实表现**:
   - ✅ **Growth Level**: 67-96% (可用，v1.2 最佳)
   - ❌ **Growth Pattern**: 44-57% (远低于随机猜测的期望)
   - 🔴 **Interference**: 0.9-5% F1 (完全失败)

---

## 🔍 根本原因分析

### 问题 A: 数据集随机划分不一致

**代码位置**: `training/enhanced_multitask_dataset.py:149`

```python
def split_samples(samples, ratios):
    random.shuffle(samples)  # ❌ 没有固定随机种子
```

**影响**:
- 训练报告中的验证集 ≠ 当前测试时的验证集
- 导致 Growth Level 和 Growth Pattern 性能差异 **7-32%**

**证据**:
- v1.1 训练时验证集: negative=1481, positive=1518
- 当前验证时: negative=1467, positive=1532
- 样本分布不同导致性能评估不可复现

---

### 问题 B: Interference 指标计算严重错误 🔴

**代码位置**: `training/improved_multilevel_trainer.py:197-203`

```python
if task == 'interference_factors':
    # 多标签分类准确率 - 将概率转换为二进制预测
    preds_binary = (preds > 0.5).astype(int)
    accuracies[task] = np.mean([
        accuracy_score(targets_np[:, i], preds_binary[:, i])  # ❌ 用准确率而非F1
        for i in range(targets_np.shape[1])
    ])
```

**问题**: 对于**极度不平衡的多标签分类**，准确率会严重虚高。

**测试集 Interference 分布** (极度不平衡):
```
pores:          1097 / 3000 (36.6%)
artifacts:       225 / 3000 (7.5%)
debris:          133 / 3000 (4.4%)
contamination:     2 / 3000 (0.07%)  ← 极度稀有！
```

**结果对比**:
| 版本 | 准确率 (训练报告) | F1 分数 (实际) | 差距 |
|------|-----------------|--------------|------|
| v1.1 | 95.96% | 5.08% | **-90.88%** |
| v1.2 | 95.98% | 0.87% | **-95.11%** |
| Small Quick | 94.76% | 1.82% | **-92.94%** |

**原因**: 模型几乎总是预测"无干扰因素"（全部预测为0），导致：
- ✅ 准确率虚高（因为大部分样本确实是0）
- ❌ F1 极低（因为召回率接近0）

---

### 问题 C: Growth Pattern 性能真实问题

**v1.1 和 v1.2 在 Growth Pattern 上的真实准确率只有 55-57%**

对于 **10 分类任务**，随机猜测的期望准确率是 **10%**，所以 55-57% 看似还行，但：
- 训练报告声称 **88%**
- 实际只有 **55-57%**
- 差距高达 **31-33%**

这说明:
1. 数据集划分不一致导致验证集与测试集差异大
2. 可能存在过拟合到特定验证集的问题

---

## 💡 详细分析

### 各版本深入对比

#### 1️⃣ MobileNetV4 v1.1 (Small)

**参数量**: 0.95M
**训练时间**: 133.8秒 (20 epochs)

| 任务 | 声称 | 实际 | 差异 | 分析 |
|------|------|------|------|------|
| Growth Level | 98.73% | 90.87% | -7.87% | 可用但不如声称 |
| Growth Pattern | 88.10% | 55.57% | -32.53% | 严重虚高 |
| Interference (Acc) | 95.96% | 76.56% | -19.40% | 虚高 |
| Interference (F1) | - | **5.08%** | - | 🔴 完全失败 |

**结论**: 实际综合性能 **59.59%**，而非报告的 94%。

---

#### 2️⃣ MobileNetV4 v1.2 (Medium)

**参数量**: 1.33M (+40% vs v1.1)
**训练时间**: 更长 (25 epochs)

| 任务 | 声称 | 实际 | 差异 | 分析 |
|------|------|------|------|------|
| Growth Level | 98.80% | **96.13%** | -2.67% | ✅ **最佳表现** |
| Growth Pattern | 88.86% | 57.03% | -31.83% | 仍然严重虚高 |
| Interference (Acc) | 95.98% | 72.56% | -23.42% | 更严重虚高 |
| Interference (F1) | - | **0.87%** | - | 🔴 比v1.1更差！ |

**结论**:
- ✅ Growth Level 表现最好 (96.13%)
- ⚠️ Growth Pattern 略有提升 (57% vs 55%)
- 🔴 Interference F1 反而更差 (0.87% vs 5.08%)
- 整体: **61.44%** (略优于v1.1)

---

#### 3️⃣ MobileNetV4 Small Quick (可能是v1.0)

**参数量**: 0.95M
**训练时间**: 更短 (10 epochs)

| 任务 | 声称 | 实际 | 差异 | 分析 |
|------|------|------|------|------|
| Growth Level | 98.57% | **66.90%** | -31.67% | ❌ 严重虚高 |
| Growth Pattern | 85.33% | 44.03% | -41.30% | ❌ 最差表现 |
| Interference (Acc) | 94.76% | 75.43% | -19.33% | 虚高 |
| Interference (F1) | - | **1.82%** | - | 🔴 失败 |

**结论**:
- 训练不充分或数据集问题
- 所有任务表现均为最差
- 整体: **44.74%** (不可用)

---

## 🎯 版本排名 (按真实性能)

### 综合排名

| 排名 | 版本 | 综合准确率 | Growth Level | Growth Pattern | Interference F1 | 推荐度 |
|------|------|-----------|-------------|---------------|----------------|--------|
| 🥇 | **v1.2 Medium** | **61.44%** | 96.13% ⭐ | 57.03% | 0.87% | ⚠️ 部分可用 |
| 🥈 | **v1.1 Small** | **59.59%** | 90.87% | 55.57% | 5.08% | ⚠️ 部分可用 |
| 🥉 | Small Quick | 44.74% | 66.90% | 44.03% | 1.82% | ❌ 不推荐 |

### 各任务最佳模型

- **Growth Level**: v1.2 (96.13%) ✅
- **Growth Pattern**: v1.2 (57.03%) ⚠️
- **Interference**: v1.1 (5.08% F1) 🔴 都很差

---

## ⚠️ 关键问题总结

### 问题严重程度

| 问题 | 影响范围 | 严重性 | 是否普遍 |
|------|---------|--------|---------|
| **数据集随机划分** | Growth Level, Growth Pattern | 🟡 中等 | ✅ 所有版本 |
| **Interference 指标计算错误** | Interference Factors | 🔴 严重 | ✅ 所有版本 |
| **训练报告虚高** | 所有任务 | 🔴 严重 | ✅ 所有版本 |
| **Interference 任务失败** | 多标签分类 | 🔴 严重 | ✅ 所有版本 |

---

## 📋 修复建议

### 优先级 1: 立即修复代码问题

#### A. 修复数据集划分随机性

```python
# training/enhanced_multitask_dataset.py
def split_samples(samples, ratios, seed=42):
    random.seed(seed)  # ✅ 固定随机种子
    random.shuffle(samples)
    # ... 其余代码不变
```

#### B. 修复 Interference 指标计算

```python
# training/improved_multilevel_trainer.py
if task == 'interference_factors':
    # ✅ 使用 F1 分数而非准确率
    from sklearn.metrics import f1_score

    f1_scores = []
    for i in range(targets_np.shape[1]):
        f1 = f1_score(
            targets_np[:, i],
            preds_binary[:, i],
            zero_division=0
        )
        f1_scores.append(f1)

    accuracies[task] = np.mean(f1_scores)  # 存储F1而非准确率
```

### 优先级 2: 重新训练和评估

1. **使用修复后的代码重新训练所有版本**
2. **在固定的测试集上评估**
3. **生成真实可信的性能报告**

### 优先级 3: 解决 Interference 任务失败问题

**根本原因**: 数据极度不平衡 + 多标签分类策略不当

**建议方案**:

1. **调整损失函数权重**:
```python
# 针对稀有类别增加权重
class_weights = {
    'pores': 1.0,
    'artifacts': 2.0,       # 稀有类别
    'debris': 3.0,          # 更稀有
    'contamination': 10.0   # 极稀有
}
```

2. **使用 Focal Loss**:
```python
# 专门处理类别不平衡
from torch.nn import BCEWithLogitsLoss
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

3. **调整预测阈值**:
```python
# 针对不同类别使用不同阈值
thresholds = {
    'pores': 0.5,
    'artifacts': 0.3,
    'debris': 0.2,
    'contamination': 0.1
}
```

---

## 🚀 下一步行动

### 立即行动

1. ✅ **已完成**: 确认所有版本都存在问题
2. 🔄 **进行中**: 生成本验证报告
3. ⏭️ **下一步**: 修复代码问题

### 短期计划 (1-2 天)

1. 修复数据集划分代码
2. 修复 Interference 指标计算
3. 重新训练 v1.3 版本
4. 在测试集上验证新模型

### 中期计划 (1-2 周)

1. 解决 Interference 任务失败问题
2. 改进 Growth Pattern 性能
3. 训练真正可用的生产模型
4. 建立完善的评估流程

---

## 📁 附件

- **验证数据**: `deployment/validation_results/all_versions_validation.json`
- **v1.1 验证**: `deployment/validation_results/onnx_validation_results.json`
- **训练代码**: `training/improved_multilevel_trainer.py`
- **数据集代码**: `training/enhanced_multitask_dataset.py`

---

## 🎓 经验教训

### 关键教训

1. **永远不要信任训练报告的数字**
   - 必须在独立测试集上验证
   - 必须使用正确的评估指标

2. **数据集划分必须可复现**
   - 固定随机种子
   - 或保存划分索引

3. **对于不平衡数据，准确率≠性能**
   - 必须使用 F1、Recall、Precision
   - 必须查看混淆矩阵

4. **训练过程监控不足**
   - 应该实时在测试集上评估
   - 应该可视化预测分布

---

**报告生成时间**: 2025-10-03
**验证工具**: PyTorch + ONNX Runtime
**测试集**: m9e1n170.json (3000 样本)
**报告版本**: 1.0

---

## ✅ 总结

**确认**: 所有 MobileNetV4 版本（v1.0/v1.1/v1.2）都存在相同的训练报告虚高问题。

**最严重问题**: Interference 任务完全失败（F1 < 6%），但训练报告声称 95%+。

**推荐**: 暂停使用现有模型，修复代码后重新训练。
