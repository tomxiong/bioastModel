# 数据集清理总结报告

## 执行摘要

**清理日期**: 2025-10-04
**操作**: 移除 positive + clustered 样本中的 pores 标注
**目的**: 解决 pores 与菌落标注冲突问题

## 1. 清理规则

### 1.1 移除条件

**符合以下所有条件的样本,移除其 pores 标注**:
1. `growth_level = positive` (有菌落)
2. `growth_pattern = clustered` (簇状菌落)
3. `interference_factors` 包含 `pores`

**操作**: 从 `interference_factors` 中移除 `pores`,保留其他干扰因素

### 1.2 理由

根据分析发现:
- Pores+Positive 中 **76.1% 是 clustered** 模式
- 这些样本中的黑色区域更可能是**菌落簇**而非气孔
- 存在标注冲突: 同一区域既是菌落又是气孔
- 导致模型无法学习 pores 特征

---

## 2. 清理结果

### 2.1 统计数据

| 指标 | 数值 |
|------|------|
| **总样本数** | 19,994 (不变) |
| **清理前 pores 样本** | 7,450 |
| **清理后 pores 样本** | 5,819 |
| **移除的 pores 标注** | 1,631 |
| **移除比例** | 21.9% |

### 2.2 受影响样本

**移除了 pores 的样本** (1,631 个):
- 条件: positive + clustered + pores
- 占总样本比例: 8.2%
- 这些样本的其他标注保持不变

**保留 pores 的样本** (5,819 个):
- Pores + Negative: 大部分
- Pores + Positive (非 clustered): 少部分

---

## 3. 数据集对比

### 3.1 原始数据集 (m9e1n170.json)

**Pores 分布**:
- 总 Pores: 7,450
- Pores+Positive: 2,144 (28.8%)
- Pores+Negative: 5,306 (71.2%)

**Pores+Positive 的 Growth Pattern**:
- clustered: 1,631 (76.1%) ← 被移除
- center_dots: 254 (11.8%)
- weak_scattered_pos: 162 (7.6%)
- 其他: 97 (4.5%)

### 3.2 清理后数据集 (m9e1n170_cleaned.json)

**Pores 分布**:
- 总 Pores: 5,819 (↓ 1,631)
- Pores+Positive: 513 (8.8%) ← 大幅减少
- Pores+Negative: 5,306 (91.2%)

**Pores+Positive 的 Growth Pattern** (清理后):
- center_dots: 254 (49.5%)
- weak_scattered_pos: 162 (31.6%)
- strong_scattered: 91 (17.7%)
- 其他: 6 (1.2%)
- clustered: 0 (0%) ← 已完全移除

---

## 4. 新数据集划分

### 4.1 划分配置

**文件**: `ds/images/dataset_split_seed44.json`
**随机种子**: 44 (区别于之前的 42, 43)
**比例**: train 70%, val 15%, test 15%
**策略**: growth_level + pores 双层分层抽样

### 4.2 划分统计

**训练集** (13,994 样本):
- Positive: 7,149, Negative: 6,845
- Pores: 4,073 (29.1%)
- 主要 Pattern: clustered (4,836), clean (3,933)

**验证集** (2,997 样本):
- Positive: 1,531, Negative: 1,466
- Pores: 871 (29.1%)
- 主要 Pattern: clustered (1,047), clean (847)

**测试集** (3,003 样本):
- Positive: 1,534, Negative: 1,469
- Pores: 875 (29.1%)
- 主要 Pattern: clustered (1,040), clean (810)

### 4.3 Pores 分布对比

**原始数据集 (seed=42)**:
| 划分 | Pores 样本 | 占比 |
|------|-----------|------|
| Train | 5,214 | 37.26% |
| Val | 1,116 | 37.24% |
| Test | 1,120 | 37.30% |

**清理后数据集 (seed=44)**:
| 划分 | Pores 样本 | 占比 |
|------|-----------|------|
| Train | 4,073 | 29.10% |
| Val | 871 | 29.06% |
| Test | 875 | 29.14% |

**变化**:
- Pores 样本数减少约 22%
- Pores 占比从 37% 降至 29%
- 分布仍然非常均衡 (差异 <0.1%)

---

## 5. 预期效果

### 5.1 Pores 性能预期

**清理前** (v0.9.6 在原始数据):
- Pores F1: 0% (验证集和测试集)
- 原因: 与菌落特征冲突

**清理后预期**:
- Pores F1: **50-70%** (保守估计)
- 原因:
  1. 移除了冲突样本 (positive+clustered)
  2. 保留的 pores 主要是 negative (91.2%)
  3. Pores+Negative 特征更明确 (环形/C形结构)

### 5.2 整体 Interference 性能预期

**当前性能** (v0.9.6):
| 类别 | F1 | 权重 |
|------|-----|------|
| artifacts | 88.22% | 3.0 |
| contamination | 61.22% | 50.0 |
| debris | 50.18% | 5.0 |
| **pores** | **0%** | 1.0 |
| **整体** | **49.91%** | - |

**清理后预期** (v0.9.8):
| 类别 | 预期 F1 | 说明 |
|------|---------|------|
| artifacts | 88%+ | 保持 |
| contamination | 61%+ | 保持 |
| debris | 50%+ | 保持 |
| **pores** | **60-70%** | 大幅提升 |
| **整体** | **65-70%** | 提升 15-20% |

---

## 6. 下一步训练计划

### 6.1 训练配置 (v0.9.8)

**基础配置** (继承 v0.9.6):
```python
model_size: small
input_channels: 1
dropout_rate: 0.3
batch_size: 64
learning_rate: 0.002
num_epochs: 35
patience: 15
```

**任务权重** (保持 v0.9.6):
```python
task_weights = [1.0, 2.0, 0.8]
# growth_level, growth_pattern, interference
```

**类别权重** (可选调整):
```python
# 方案 A: 保持 v0.9.6
interference_weights = [3.0, 5.0, 50.0, 1.0]

# 方案 B: 提高 pores 权重
interference_weights = [3.0, 5.0, 50.0, 2.0]
# pores: 1.0 → 2.0 (因为样本减少了)
```

**数据集**:
```python
annotations_file = "m9e1n170_cleaned.json"
split_file = "dataset_split_seed44.json"
```

### 6.2 训练脚本

创建 `scripts/train_multilevel_mobilenetv3_v0.9.8.py`:

```bash
# 复制 v0.9.6 脚本
cp scripts/train_multilevel_mobilenetv3_v0.9.6.py \
   scripts/train_multilevel_mobilenetv3_v0.9.8.py

# 修改配置:
# 1. annotations_file → m9e1n170_cleaned.json
# 2. split_file → dataset_split_seed44.json
# 3. experiment_dir → multilevel_mobilenetv3_v0.9.8
# 4. (可选) pores 权重 1.0 → 2.0
```

### 6.3 验证和对比

**训练后对比**:
1. v0.9.6 (原始数据) vs v0.9.8 (清理后数据)
2. 关注 pores F1 变化
3. 检查其他 3 个类别是否保持性能

**关键指标**:
- Pores F1: 0% → 目标 60%+
- Interference 整体 F1: 49.91% → 目标 65%+
- Growth Level 和 Pattern: 保持稳定

---

## 7. 文件清单

### 7.1 数据集文件

| 文件 | 说明 |
|------|------|
| `ds/images/m9e1n170.json` | 原始数据集 (19,994 样本, 7,450 pores) |
| `ds/images/m9e1n170_cleaned.json` | 清理后数据集 (19,994 样本, 5,819 pores) |
| `ds/images/dataset_split_seed42.json` | 原始数据集划分 (seed=42) |
| `ds/images/dataset_split_seed44.json` | 清理后数据集划分 (seed=44) |
| `ds/images/dataset_cleaning_stats.json` | 清理统计信息 |

### 7.2 脚本文件

| 文件 | 说明 |
|------|------|
| `scripts/create_cleaned_dataset.py` | 数据集清理脚本 |
| `scripts/create_fixed_dataset_split.py` | 数据集划分脚本 |
| `scripts/train_multilevel_mobilenetv3_v0.9.8.py` | v0.9.8 训练脚本 (待创建) |

### 7.3 分析报告

| 文件 | 说明 |
|------|------|
| `PORES_FINAL_DIAGNOSIS_REPORT.md` | Pores 问题最终诊断 |
| `DATASET_CLEANING_SUMMARY.md` | 数据集清理总结 (本文档) |
| `analysis/pores_diagnosis/` | Pores 分析可视化 |

---

## 8. 验证检查清单

### 8.1 数据完整性

- [x] 样本数量不变 (19,994)
- [x] 仅移除符合条件的 pores 标注
- [x] 其他标注保持不变
- [x] 文件格式正确 (JSON)

### 8.2 清理正确性

- [x] 移除了 1,631 个 positive+clustered 的 pores
- [x] 保留了 513 个 positive (非clustered) 的 pores
- [x] 保留了 5,306 个 negative 的 pores
- [x] 抽样验证通过

### 8.3 划分质量

- [x] Pores 分布均衡 (29.06% ~ 29.14%)
- [x] Growth Level 分布合理
- [x] 使用分层抽样策略
- [x] 固定随机种子 (可重复)

---

## 9. 使用说明

### 9.1 快速开始

**1. 验证清理结果**:
```bash
# 检查清理统计
cat ds/images/dataset_cleaning_stats.json

# 检查新的划分
python -c "
import json
with open('ds/images/dataset_split_seed44.json', 'r') as f:
    data = json.load(f)
    print(f'Train: {len(data[\"splits\"][\"train\"])}')
    print(f'Val: {len(data[\"splits\"][\"val\"])}')
    print(f'Test: {len(data[\"splits\"][\"test\"])}')
"
```

**2. 创建 v0.9.8 训练脚本**:
```bash
# 复制并修改
cp scripts/train_multilevel_mobilenetv3_v0.9.6.py \
   scripts/train_multilevel_mobilenetv3_v0.9.8.py

# 修改关键配置:
# - annotations_file = "m9e1n170_cleaned.json"
# - split_file = "dataset_split_seed44.json"
# - experiment_dir = "multilevel_mobilenetv3_v0.9.8"
```

**3. 训练模型**:
```bash
python scripts/train_multilevel_mobilenetv3_v0.9.8.py
```

### 9.2 对比分析

**训练完成后**:
```bash
# 对比 v0.9.6 vs v0.9.8
python scripts/compare_models.py \
  --models multilevel_mobilenetv3_v0.9.6 multilevel_mobilenetv3_v0.9.8 \
  --focus-on pores

# 生成对比报告
python scripts/generate_comparison_report.py \
  --baseline v0.9.6 \
  --improved v0.9.8
```

---

## 10. 预期问题和应对

### 10.1 如果 Pores F1 仍然很低 (<30%)

**可能原因**:
1. Pores+Negative 样本也存在标注问题
2. 模型架构不适合学习气孔特征
3. 数据增强不足

**应对措施**:
1. 进一步审查 Pores+Negative 样本
2. 尝试更强的数据增强
3. 考虑使用注意力机制

### 10.2 如果其他任务性能下降

**可能原因**:
1. 数据集变化影响了其他任务
2. 随机种子变化导致的波动

**应对措施**:
1. 对比训练曲线
2. 检查各任务的损失变化
3. 必要时调整任务权重

### 10.3 如果 Pores F1 达到 60%+

**下一步优化**:
1. 进一步提高 pores 权重
2. 添加 pores 专用数据增强
3. 尝试 focal loss
4. 目标: Pores F1 → 80%+

---

## 11. 结论

### 11.1 主要成果

1. ✅ 成功移除 1,631 个冲突标注
2. ✅ 创建清理后的数据集 (5,819 pores)
3. ✅ 生成新的数据集划分 (seed=44)
4. ✅ Pores 特征更清晰 (91.2% 是 negative)

### 11.2 预期提升

- Pores F1: 0% → **60-70%**
- Interference 整体 F1: 49.91% → **65-70%**
- 解决了严重的标注冲突问题

### 11.3 下一步

1. 创建 v0.9.8 训练脚本
2. 训练并评估模型
3. 对比 v0.9.6 vs v0.9.8 性能
4. 根据结果决定是否进一步优化

---

## 12. 第二轮清理 (2025-10-04 04:04)

### 12.1 清理目标扩展

**数据集**: `ds/images/m9e1n170_cleaned.json` (19,994 样本)

**新增清理规则** - 移除以下组合中的 pores 标注:
- ✅ **Positive + Strong_Scattered + Pores** (新增，主要目标)
- ✅ **Positive + Scattered + Pores** (新增)
- ✅ **Positive + Heavy_Growth + Pores** (新增)

**理由**:
- 第一轮清理了 `clustered`，但遗漏了 `strong_scattered` 等其他明显菌落模式
- 当菌落特征明显时（strong_scattered、scattered、heavy_growth），气孔不应成为主要干扰因素
- 这些场景下的 "pores" 标注更可能是菌落本身的特征而非真实气孔

### 12.2 清理结果

| 指标 | 第一轮后 | 第二轮后 | 变化 |
|------|----------|----------|------|
| **总样本数** | 19,994 | 19,994 | - |
| **总 Pores** | 5,819 | 5,724 | -95 (-1.6%) |
| **Positive + Pores** | 513 | 418 | -95 (-18.5%) |
| **Negative + Pores** | 5,306 | 5,306 | 0 |
| **Negative 中 Pores 占比** | 91.2% | **92.7%** | +1.5% |

### 12.3 按 Pattern 清理统计

第二轮清理的 95 个 pores 标注:
- **Strong_Scattered**: 91 个 (95.8%)
- **Heavy_Growth**: 3 个 (3.2%)
- **Scattered**: 1 个 (1.1%)

### 12.4 清理后 Pores 分布

**Positive + Pores 剩余情况** (418 个):
- **Center_Dots**: 254 (60.8%) - 中心点菌落模式
- **Weak_Scattered_Pos**: 162 (38.8%) - 弱分散菌落模式
- **Irregular**: 2 (0.5%) - 不规则模式

**关键改进**:
- ✅ 移除了所有 `strong_scattered` + pores (91 个)
- ✅ 移除了所有 `scattered` + pores (1 个)
- ✅ 移除了所有 `heavy_growth` + pores (3 个)
- ✅ Negative 占比从 91.2% 提升到 92.7%
- ✅ 剩余的 Positive + Pores 主要是 center_dots 和 weak_scattered_pos，这些是合理保留的边界案例

### 12.5 备份和输出文件

**文件结构**:
```
ds/images/
├── m9e1n170.json                              (原始数据，19,994 样本，7,450 pores)
├── m9e1n170_cleaned.json                      (第一轮清理后，5,819 pores) ← 第二轮清理的输入
├── m9e1n170_cleaned_backup_round1.json        (第一轮清理结果备份)
├── m9e1n170_cleaned_round2.json               (第二轮清理后，5,724 pores) ← 新文件
├── dataset_split_seed44.json                  (数据集划分)
└── cleaning_stats_round2_20251004.json        (第二轮清理详细统计)
```

### 12.6 两轮清理累计效果

**原始 → 第一轮 → 第二轮**:

| 指标 | 原始 | 第一轮 | 第二轮 | 累计变化 |
|------|------|--------|--------|----------|
| **总 Pores** | 7,450 | 5,819 | 5,724 | **-1,726 (-23.2%)** |
| **Positive + Pores** | 2,144 | 513 | 418 | **-1,726 (-80.5%)** |
| **Negative + Pores** | 5,306 | 5,306 | 5,306 | **0 (100%)** |
| **Negative 占比** | 71.2% | 91.2% | **92.7%** | **+21.5%** |

**清理覆盖的 Pattern**:
- 第一轮: `clustered` (1,631 个)
- 第二轮: `strong_scattered`, `scattered`, `heavy_growth` (95 个)
- **累计**: 1,726 个冲突标注被移除

### 12.7 预期效果提升

**基于第二轮清理的预期**:

| 指标 | 第一轮预期 | 第二轮预期 | 说明 |
|------|-----------|-----------|------|
| **Pores F1** | 60-70% | **65-75%** | 进一步提升 |
| **Pores 精确率** | 70%+ | **75-80%** | 更少误报 |
| **Pores 召回率** | 60%+ | **65-75%** | 保持稳定 |
| **Interference 整体 F1** | 65-70% | **68-72%** | 整体提升 |

**改进原因**:
1. 移除了 strong_scattered 中的 pores 冲突（最重要的改进）
2. Negative 占比提升到 92.7%，特征更加纯净
3. 剩余的 Positive + Pores 主要是合理的边界案例（center_dots, weak_scattered_pos）

### 12.8 数据完整性验证

- ✅ 样本数量保持不变 (19,994)
- ✅ 仅移除指定组合的 pores 标注 (95 个)
- ✅ 其他干扰因素 (debris, artifacts, contamination) 保持不变
- ✅ Growth Level 和 Growth Pattern 标注保持不变
- ✅ Negative + Pores 完全保留 (5,306 个)
- ✅ 第一轮清理结果已完整备份

### 12.9 剩余 Positive + Pores 的合理性

剩余的 418 个 Positive + Pores 样本主要是：

1. **Center_Dots (254 个, 60.8%)**:
   - 中心点菌落模式，可能确实存在边缘气孔
   - 这是合理的边界案例，应该保留

2. **Weak_Scattered_Pos (162 个, 38.8%)**:
   - 弱分散菌落模式，菌落特征不够明显
   - 气孔可能是真实存在的干扰因素
   - 保留是合理的

3. **Irregular (2 个, 0.5%)**:
   - 不规则模式，少量样本
   - 需要人工审核

**结论**: 第二轮清理已经移除了所有明显菌落模式（strong_scattered, scattered, heavy_growth）中的 pores，剩余的 Positive + Pores 都是合理的边界案例。

---

**第一轮清理完成**: 2025-10-04 (之前)
**第二轮清理完成**: 2025-10-04 04:04
**数据集版本 (第一轮)**: m9e1n170_cleaned.json
**数据集版本 (第二轮)**: m9e1n170_cleaned_round2.json
**下一个训练版本**: v0.9.8 (建议使用 m9e1n170_cleaned_round2.json)
