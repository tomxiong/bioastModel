# MobileNetV4 训练数据集说明

## 📊 数据集概览

### 基本信息

- **数据集名称**: 新数据集 - 增强标注
- **数据集路径**: `/home/aaa/ws/bioastModel/ds/images/`
- **标注文件**: `m9e1n170.json`
- **总样本数**: **19,994** 个标注
- **图像格式**: PNG (70×70 像素, 灰度图)
- **创建时间**: 2025-09-18

### 数据集结构

```
ds/images/
├── EB10000026/        # 全景图文件夹
│   ├── hole_26.png
│   ├── hole_27.png
│   └── ...
├── EB10000027/
├── EB10000028/
├── ...
└── m9e1n170.json      # 标注文件 (11MB)
```

---

## 🏷️ 标签分布 (基于前 10,000 样本分析)

### 1. Growth Level (生长水平) - 二分类

| 类别 | 数量 | 比例 |
|------|------|------|
| **positive** | 5,084 | 50.84% |
| **negative** | 4,916 | 49.16% |

✅ **类别平衡**: 非常均衡,差异 < 2%

### 2. Growth Pattern (生长模式) - 10分类

| 类别 | 数量 | 比例 | 难度 |
|------|------|------|------|
| **clustered** | 4,301 | 43.01% | ⭐ 简单 |
| **clean** | 2,980 | 29.80% | ⭐ 简单 |
| **weak_scattered** | 1,384 | 13.84% | ⭐⭐ 中等 |
| **litter_center_dots** | 552 | 5.52% | ⭐⭐ 中等 |
| **center_dots** | 289 | 2.89% | ⭐⭐⭐ 困难 |
| **strong_scattered** | 199 | 1.99% | ⭐⭐⭐ 困难 |
| **heavy_growth** | 194 | 1.94% | ⭐⭐⭐ 困难 |
| **scattered** | 36 | 0.36% | ⭐⭐⭐⭐ 很困难 |
| **irregular** | 34 | 0.34% | ⭐⭐⭐⭐ 很困难 |
| **weak_scattered_pos** | 31 | 0.31% | ⭐⭐⭐⭐ 很困难 |

⚠️ **类别不平衡**:
- 前2类占 72.81%
- 最少的3类合计仅 1.01%
- 这是当前 Growth Pattern 准确率较低的主要原因

### 3. Interference Factors (干扰因子) - 多标签

| 类别 | 出现次数 | 样本比例 |
|------|----------|----------|
| **pores** (孔洞) | 3,300 | 33.00% |
| **artifacts** (伪影) | 922 | 9.22% |
| **debris** (碎片) | 614 | 6.14% |
| **contamination** (污染) | 6 | 0.06% |

📝 **多标签说明**: 一张图像可能同时有多个干扰因子

---

## 📈 数据集特点

### 优势

1. ✅ **样本量充足**: 近 20,000 个标注样本
2. ✅ **Growth Level 平衡**: 正负样本几乎 1:1
3. ✅ **高质量标注**: 人工确认的增强标注
4. ✅ **完整元数据**: 包含置信度、来源等信息

### 挑战

1. ⚠️ **Growth Pattern 不平衡**:
   - 长尾分布,部分类别样本 < 50
   - 可能导致模型对少数类过拟合不足

2. ⚠️ **Interference 分布不均**:
   - pores 占主导 (33%)
   - contamination 极少 (0.06%)

3. ⚠️ **小图像**: 70×70 像素限制了细节信息

---

## 🎯 训练策略建议

### 1. 数据增强

```python
# 推荐的数据增强策略
transforms.Compose([
    # 几何变换
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),

    # 颜色变换 (灰度图)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),

    # 噪声 (可选)
    AddGaussianNoise(mean=0, std=0.01),

    # 标准化
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

### 2. 类别平衡处理

#### Growth Pattern 不平衡解决方案:

```python
# 方案1: 加权采样
from torch.utils.data import WeightedRandomSampler

class_counts = [4301, 2980, 1384, ...]  # 各类别数量
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = weights[labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# 方案2: Focal Loss
from torch.nn import CrossEntropyLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重
```

#### Interference Factors:

```python
# 多标签使用 BCEWithLogitsLoss + pos_weight
pos_weight = torch.tensor([
    1.0,   # artifacts
    1.0,   # contamination (权重可以提高)
    1.0,   # debris
    1.0    # pores
])

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 3. 数据划分

```python
# 推荐划分比例
train_ratio = 0.70  # 13,996 samples
val_ratio = 0.15    # 2,999 samples
test_ratio = 0.15   # 2,999 samples

# 分层采样 (保持各类别比例)
from sklearn.model_selection import train_test_split

train_val, test = train_test_split(
    data,
    test_size=0.15,
    stratify=growth_levels,  # 按 growth_level 分层
    random_state=42
)

train, val = train_test_split(
    train_val,
    test_size=0.176,  # 0.15 / 0.85 ≈ 0.176
    stratify=train_val_growth_levels,
    random_state=42
)
```

---

## 📊 与改进版 V3 对比

### 数据集一致性

改进版 MobileNetV3 使用 **相同的数据集**:
- ✅ 相同的标注文件: `m9e1n170.json`
- ✅ 相同的样本数量: 19,994
- ✅ 相同的数据划分策略

### 性能基准参考

| 模型 | Growth Level | Growth Pattern | Interference | Overall |
|------|-------------|----------------|--------------|---------|
| **V3 改进版** | 98.13% | 86.07% | 92.64% | 92.65% |
| **V4 目标** | >98% | **>87%** | **>93%** | **>93%** |

---

## 🔧 数据集加载

### 使用现有 Dataset 类

```python
from training.enhanced_multitask_dataset import EnhancedMultitaskDataset

# 创建数据集
dataset = EnhancedMultitaskDataset(
    json_path='/home/aaa/ws/bioastModel/ds/images/m9e1n170.json',
    image_root='/home/aaa/ws/bioastModel/ds/images',
    split='train'
)

print(f"Dataset size: {len(dataset)}")

# 获取一个样本
image, labels = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Labels: {labels}")
```

### 标签格式

```python
labels = {
    'growth_level': torch.tensor(1),           # 0=negative, 1=positive
    'growth_pattern': torch.tensor(5),         # 0-9 (10类)
    'interference_factors': torch.tensor([     # 多标签 binary
        0, 1, 0, 1                             # [artifacts, contamination, debris, pores]
    ])
}
```

---

## 📝 数据质量

### 标注质量

- ✅ **人工确认**: `is_confirmed: true`
- ✅ **置信度**: 大部分标注 `confidence: 1.0`
- ✅ **元数据完整**: 包含来源、时间戳等

### 已知问题

1. **Growth Pattern 类别定义**:
   - 部分类别界限模糊 (如 scattered vs weak_scattered)
   - 建议: 可以考虑合并相似类别

2. **Contamination 样本极少**:
   - 仅 6 个样本 (0.06%)
   - 建议: 可能需要过采样或单独处理

3. **小样本类别**:
   - 3个类别 < 50 样本
   - 建议: 使用数据增强或迁移学习

---

## 🚀 训练配置推荐

基于数据集特点和改进版 V3 的成功经验:

```yaml
# 数据加载
batch_size: 64
num_workers: 4
pin_memory: true

# 数据增强
use_augmentation: true
augmentation_strength: medium

# 类别平衡
use_weighted_sampling: false  # 改进版不用也达到92.65%
use_focal_loss: false         # 可选,用于进一步提升

# 任务权重 (关键: 统一权重效果最好!)
growth_level_weight: 1.0
growth_pattern_weight: 1.0
interference_weight: 1.0

# 数据划分
train_ratio: 0.70
val_ratio: 0.15
test_ratio: 0.15
stratify_by: growth_level  # 分层采样
```

---

## 📚 相关文档

- [MobileNetV4 使用指南](mobilenetv4_guide.md)
- [改进版 V3 性能分析](../performance_analysis/improved_multilevel_performance_analysis.md)
- [训练脚本](../../scripts/multilevel_training/train_mobilenetv4.py)

---

**数据集状态**: ✅ 已验证,可用于训练
**样本质量**: ⭐⭐⭐⭐⭐ 优秀
**标注完整性**: ✅ 完整
**推荐使用**: ✅ 是
