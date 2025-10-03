# 固定数据集划分使用指南

## 概述

为了确保模型训练的**可复现性**和**公平对比**，我们提供了固定数据集划分机制。

### 为什么需要固定划分？

**问题**: 之前所有模型（v1.0/v1.1/v1.2）使用随机划分，导致：
- ❌ 每次训练时 train/val/test 划分不同
- ❌ 不同模型使用不同的验证集，性能无法公平对比
- ❌ 训练报告中的性能无法复现
- ❌ 性能评估结果虚高或虚低（差异可达 30-40%）

**解决方案**: 使用固定的数据集划分文件
- ✅ 所有模型使用相同的 train/val/test 划分
- ✅ 性能对比公平可信
- ✅ 结果完全可复现
- ✅ 便于调试和错误分析

---

## 快速开始

### 1. 生成固定划分文件

```bash
# 使用默认参数（推荐）
python scripts/create_fixed_dataset_split.py

# 或自定义参数
python scripts/create_fixed_dataset_split.py \
    --json-path ds/images/m9e1n170.json \
    --image-root ds/images \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --output-dir ds/images
```

**输出文件**:
- `ds/images/dataset_split_seed42.json` - 固定划分文件
- `ds/images/dataset_split_latest.json` - 符号链接（指向最新划分）

### 2. 在训练中使用固定划分

#### 方法 A: 在代码中直接指定

```python
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset

# 训练集
train_dataset = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='train',
    split_file='ds/images/dataset_split_seed42.json',  # ✅ 使用固定划分
    transform=train_transform
)

# 验证集
val_dataset = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='val',
    split_file='ds/images/dataset_split_seed42.json',  # ✅ 使用固定划分
    transform=None
)

# 测试集
test_dataset = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='test',
    split_file='ds/images/dataset_split_seed42.json',  # ✅ 使用固定划分
    transform=None
)
```

#### 方法 B: 在训练脚本中添加参数

```bash
# 训练时指定划分文件
python scripts/multilevel_training/train_mobilenetv4.py \
    --split-file ds/images/dataset_split_seed42.json \
    --model-size small \
    --num-epochs 20
```

---

## 数据集划分详情

### 划分统计 (seed=42)

| 数据集 | 样本数 | Negative | Positive | 占比 |
|--------|--------|----------|----------|------|
| **TRAIN** | 13,995 | 6,846 | 7,149 | 70% |
| **VAL** | 2,999 | 1,467 | 1,532 | 15% |
| **TEST** | 3,000 | 1,467 | 1,533 | 15% |
| **总计** | 19,994 | 9,780 | 10,214 | 100% |

### Growth Pattern 分布 (VAL)

| 模式 | 样本数 | 占比 |
|------|--------|------|
| clustered | 1,018 | 33.9% |
| clean | 830 | 27.7% |
| weak_scattered | 482 | 16.1% |
| heavy_growth | 259 | 8.6% |
| litter_center_dots | 155 | 5.2% |
| 其他 | 255 | 8.5% |

### Interference Factors 分布 (VAL)

| 因素 | 样本数 | 占比 |
|------|--------|------|
| none | 1,617 | 53.9% |
| pores | 1,070 | 35.7% |
| artifacts | 232 | 7.7% |
| debris | 126 | 4.2% |
| contamination | 7 | 0.2% |

---

## 固定划分文件格式

```json
{
  "metadata": {
    "created_at": "20251003_025800",
    "json_path": "ds/images/m9e1n170.json",
    "image_root": "ds/images",
    "seed": 42,
    "ratios": {
      "train": 0.7,
      "val": 0.15,
      "test": 0.15
    },
    "total_samples": 19994,
    "missing_files": 0
  },
  "splits": {
    "train": ["image_path_1", "image_path_2", ...],
    "val": ["image_path_a", "image_path_b", ...],
    "test": ["image_path_x", "image_path_y", ...]
  },
  "statistics": {
    "train": {
      "total": 13995,
      "growth_level": {"negative": 6846, "positive": 7149},
      "growth_pattern": {...},
      "interference_factors": {...}
    },
    "val": {...},
    "test": {...}
  }
}
```

---

## 验证固定划分

### 验证可复现性

```python
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset

# 多次加载验证集
val_1 = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='val',
    split_file='ds/images/dataset_split_seed42.json'
)

val_2 = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='val',
    split_file='ds/images/dataset_split_seed42.json'
)

# 检查是否完全一致
assert len(val_1) == len(val_2)
for i in range(len(val_1)):
    assert val_1.annotations[i]['image_path'] == val_2.annotations[i]['image_path']

print("✅ 可复现性验证通过")
```

### 对比新旧划分方式

```python
# 固定划分（推荐）
val_fixed = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='val',
    split_file='ds/images/dataset_split_seed42.json'
)

# 随机划分（旧方式，不推荐）
val_random = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='val',
    split_file=None,  # 使用随机划分
    seed=42
)

print(f"固定划分样本数: {len(val_fixed)}")
print(f"随机划分样本数: {len(val_random)}")
# 注意：样本数相同，但具体样本可能不同
```

---

## 最佳实践

### ✅ 推荐做法

1. **始终使用固定划分文件**
   ```python
   split_file='ds/images/dataset_split_seed42.json'
   ```

2. **将划分文件提交到版本控制**
   ```bash
   git add ds/images/dataset_split_seed42.json
   git commit -m "Add fixed dataset split for reproducibility"
   ```

3. **在训练报告中记录使用的划分文件**
   ```markdown
   数据集: ds/images/dataset_split_seed42.json
   Train: 13,995 样本
   Val:   2,999 样本
   Test:  3,000 样本
   ```

4. **所有模型使用相同的划分文件**
   - 确保公平对比
   - 便于性能排名

### ❌ 避免做法

1. **不要使用随机划分进行正式训练**
   ```python
   # ❌ 不推荐
   split_file=None
   ```

2. **不要为不同模型生成不同的划分**
   - 会导致性能无法公平对比

3. **不要忽略划分文件的版本管理**
   - 必须记录使用的是哪个划分文件

---

## 常见问题

### Q1: 如何生成不同比例的划分？

```bash
python scripts/create_fixed_dataset_split.py \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42
```

### Q2: 如何使用不同的随机种子？

```bash
python scripts/create_fixed_dataset_split.py \
    --seed 123  # 使用不同的种子
```

**注意**: 建议整个项目统一使用 `seed=42`，除非有特殊需求。

### Q3: 已有的模型如何迁移到固定划分？

**步骤**:
1. 生成固定划分文件（如果还没有）
2. 修改训练脚本，添加 `split_file` 参数
3. 使用固定划分重新训练模型
4. 在测试集上验证新旧模型性能

### Q4: 如何验证两个划分文件是否相同？

```python
import json

with open('ds/images/dataset_split_seed42.json') as f:
    split1 = json.load(f)

with open('ds/images/dataset_split_seed43.json') as f:
    split2 = json.load(f)

# 检查训练集是否相同
train_same = set(split1['splits']['train']) == set(split2['splits']['train'])
print(f"训练集相同: {train_same}")
```

### Q5: 划分文件损坏或丢失怎么办？

**解决方案**:
1. 从版本控制恢复
   ```bash
   git checkout ds/images/dataset_split_seed42.json
   ```

2. 或使用相同参数重新生成
   ```bash
   python scripts/create_fixed_dataset_split.py --seed 42
   ```

---

## 迁移指南

### 从随机划分迁移到固定划分

#### 步骤 1: 生成固定划分
```bash
python scripts/create_fixed_dataset_split.py
```

#### 步骤 2: 修改训练代码

**修改前**:
```python
train_dataset = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='train',
    seed=42  # ❌ 每次运行结果可能不同
)
```

**修改后**:
```python
train_dataset = EnhancedMultitaskDataset(
    data_root='ds/images',
    split='train',
    split_file='ds/images/dataset_split_seed42.json'  # ✅ 固定划分
)
```

#### 步骤 3: 重新训练

```bash
# 使用固定划分训练新模型
python scripts/multilevel_training/train_mobilenetv4.py \
    --split-file ds/images/dataset_split_seed42.json \
    --model-size small \
    --experiment-dir experiments/mobilenetv4_v1.3_fixed
```

#### 步骤 4: 验证

```bash
# 在测试集上评估
python scripts/validate_onnx_vs_pytorch.py \
    --checkpoint experiments/mobilenetv4_v1.3_fixed/best_model.pth \
    --split-file ds/images/dataset_split_seed42.json
```

---

## 技术细节

### 分层抽样策略

固定划分使用**分层抽样**确保 positive/negative 样本在各划分中比例一致：

1. 按 `growth_level` 分为 negative 和 positive 两组
2. 对每组独立按比例划分为 train/val/test
3. 合并对应的划分并打乱顺序
4. 保存 `image_path` 列表作为索引

### 加载流程

```
加载固定划分文件
  ↓
读取当前split的image_path列表
  ↓
从完整标注中筛选对应样本
  ↓
验证图像文件存在性
  ↓
返回样本列表
```

---

## 总结

### ✅ 优势

- **可复现**: 完全相同的train/val/test划分
- **公平对比**: 所有模型使用相同的评估标准
- **便于调试**: 错误样本固定，便于分析
- **版本管理**: 可追溯使用的数据集版本

### 📋 使用要求

1. 生成固定划分文件
2. 在训练时指定 `split_file` 参数
3. 将划分文件提交到版本控制
4. 在训练报告中记录使用的划分文件

### 🚀 下一步

- [ ] 所有新训练都必须使用固定划分
- [ ] 已有模型使用固定划分重新训练
- [ ] 更新训练脚本默认使用固定划分
- [ ] 在文档中强调固定划分的重要性

---

**创建时间**: 2025-10-03
**版本**: 1.0
**维护者**: BioAst 团队
