# Pores 优化策略（基于完整业务需求）

## 业务需求澄清

### 完整业务逻辑

**Pores 检测的作用**：
1. **Negative 样本**：Pores 是 negative 的验证证据
   - 所有 negative 样本都需要检测 pores
   - 有 pores → 支持 negative 判断

2. **Positive 特定 Pattern**：Pores 是误判的指示器
   - `center_dots` + pores → 可能误判
   - `weak_scattered_pos` + pores → 可能误判
   - 有 pores → 需要人工复核

### 业务关键数据（测试集）

```
需要检测 pores 的样本: 863/864 个 pores (99.9%)

1. Negative pores: 797 个 (92.4%)
   - clean + pores: 485
   - weak_scattered + pores: 248  
   - litter_center_dots + pores: 64

2. Positive critical pores: 66 个 (7.6%)
   - center_dots + pores: 37
   - weak_scattered_pos + pores: 29

不需要检测: 仅 1 个 (Positive 其他 pattern 的 pores)
```

---

## 当前性能分析

### v0.9.9 结果

**整体 Pores 性能**：
- Precision: 50.0%
- Recall: 20.0%
- F1: 28.57%

**业务关键指标**：
- 检测到：172/863
- **业务关键 Recall: 19.9%**
- 业务目标：75% (647/863)
- **差距：475 个样本（55.1 个百分点）**

### 问题诊断

**根本原因**（已确认）：
1. ✅ **特征冲突**：Pattern 和 pores 高度相关（χ² = 5316）
2. ✅ **代理学习**：模型用 pattern 预测 pores，而非学习视觉特征
3. ✅ **权重不足**：即使提升 8 倍仍无法打破代理学习
4. ✅ **多任务冲突**：Shared backbone 被 pattern 任务主导

**数据验证**：
- 数据质量：92.7% pores 纯净度 ✅
- 样本量：863 个业务关键 pores ✅
- 数据分布：各集合均衡 ✅

**结论**：不是数据问题，是架构和训练策略问题

---

## 优化方案对比

### 方案 1: 继续提升权重 + Focal Loss

```python
# v0.10.0
interference_weights = [3.0, 5.0, 10.0, 25.0]  # pores: 8 → 25
use_focal_loss = True
focal_gamma = 2.0
task_weights = [1.0, 1.5, 1.5]  # 平衡调整
```

**预期效果**：
- Pores Recall: 19.9% → **35-45%**
- 仍无法达到 75% 目标
- 无法完全打破代理学习

**优势**：
- ✅ 快速验证（3 小时）
- ✅ 无需修改架构

**劣势**：
- ❌ 效果有限
- ❌ 仍存在特征冲突

---

### 方案 2: 两阶段训练

```python
# 阶段 1: 仅训练 level + pattern (20 epochs)
task_weights = [1.0, 2.0, 0.0]

# 阶段 2: 冻结 level + pattern，仅训练 interference (30 epochs)
freeze_level_head()
freeze_pattern_head()
task_weights = [0.0, 0.0, 1.0]
interference_weights = [1.0, 1.0, 1.0, 10.0]  # 仅强调 pores
```

**预期效果**：
- Pores Recall: 19.9% → **50-65%**
- 减少但未完全消除代理学习

**优势**：
- ✅ 减少任务冲突
- ✅ 允许 interference 独立学习

**劣势**：
- ❌ 仍共享 backbone
- ❌ 训练时间加倍（6 小时）

---

### 方案 3: Pattern-Conditional Pores Loss（推荐⭐）

```python
# 关键创新：根据 pattern 动态调整 pores loss 权重
def pattern_conditional_pores_loss(pores_pred, pores_target, pattern_pred):
    """
    Pattern-conditional pores loss
    在 negative 和关键 positive pattern 上强化 pores 学习
    """
    # 获取 pattern 预测
    pattern_class = torch.argmax(pattern_pred, dim=-1)
    
    # Pattern index mapping
    # negative patterns: clean (1), weak_scattered (2), litter_center_dots (4)
    # positive critical: center_dots (6), weak_scattered_pos (7)
    
    # 创建权重 mask
    weight_mask = torch.ones_like(pores_pred)
    
    for i, pattern_idx in enumerate(pattern_class):
        if pattern_idx in [1, 2, 4]:  # Negative patterns
            weight_mask[i] = 15.0  # 高权重
        elif pattern_idx in [6, 7]:  # Positive critical
            weight_mask[i] = 15.0  # 高权重
        else:
            weight_mask[i] = 0.1  # 低权重（几乎忽略）
    
    # BCE Loss with dynamic weights
    loss = F.binary_cross_entropy_with_logits(
        pores_pred, 
        pores_target,
        reduction='none'
    )
    
    weighted_loss = loss * weight_mask
    return weighted_loss.mean()
```

**训练配置**：
```python
# v0.10.0: Pattern-Conditional Pores
task_weights = [1.0, 2.0, 1.5]  # 提升 interference
interference_weights = [3.0, 5.0, 10.0, 1.0]  # pores 基础权重保持
use_pattern_conditional_loss = True  # 核心改进
num_epochs = 40
```

**预期效果**：
- Pores Recall: 19.9% → **60-75%** ⭐
- 接近或达到业务目标
- 针对性学习，避免浪费在无关 pattern 上

**优势**：
- ✅ 精准针对业务需求
- ✅ 无需修改架构
- ✅ 训练时间合理（3-4 小时）
- ✅ 符合业务逻辑（negative + 关键 positive）

**劣势**：
- ⚠️ 需要实现自定义 loss
- ⚠️ 仍存在一定代理学习风险

---

### 方案 4: 独立 Pores 检测器

```python
# 完全独立的二分类模型
class IndependentPoresDetector(nn.Module):
    def __init__(self):
        self.backbone = MobileNetV3Small()
        self.head = nn.Linear(1024, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# 训练时仅使用业务关键样本
# 1. 所有 negative 样本
# 2. Positive 中的 center_dots + weak_scattered_pos
```

**预期效果**：
- Pores Recall: 19.9% → **75-85%** ⭐⭐
- 彻底解决问题
- 无任务冲突

**优势**：
- ✅ 彻底消除代理学习
- ✅ 无特征冲突
- ✅ 性能最优

**劣势**：
- ❌ 需要额外模型（部署复杂度）
- ❌ 训练时间较长（1-2 天）
- ❌ 推理时间增加

---

## 推荐执行路径

### 短期（验证上限）- v0.10.0

**方案 3: Pattern-Conditional Pores Loss**

**理由**：
1. 精准针对业务需求（negative + 关键 positive）
2. 无需架构修改
3. 预期可达到或接近 75% 目标
4. 实施成本低

**预期时间**：3-4 小时

**成功标准**：
- Pores Recall >= 60% → 继续优化
- Pores Recall >= 75% → 达到目标 ✅

---

### 中期（如果 v0.10.0 < 60%）- v0.10.1

**方案 2: 两阶段训练**

**理由**：
1. 进一步减少任务冲突
2. 允许 interference 完全独立学习
3. 预期 50-65% recall

**预期时间**：6 小时

---

### 长期（最终方案）- v0.11.0

**方案 4: 独立 Pores 检测器**

**理由**：
1. 彻底解决代理学习问题
2. 性能最优（75-85% recall）
3. 可解释性强

**实施时机**：
- 如果 v0.10.0 和 v0.10.1 都无法达到 75%
- 或业务要求更高准确率（>85%）

**预期时间**：1-2 天

---

## 实施细节（v0.10.0）

### 代码实现

```python
# training/pattern_conditional_loss.py
import torch
import torch.nn.functional as F

class PatternConditionalPoresLoss:
    def __init__(self, pattern_mapping):
        """
        pattern_mapping: dict mapping pattern names to indices
        """
        self.pattern_mapping = pattern_mapping
        
        # 业务关键 patterns
        self.negative_patterns = ['clean', 'weak_scattered', 'litter_center_dots']
        self.positive_critical = ['center_dots', 'weak_scattered_pos']
        
        # 获取索引
        self.negative_indices = [
            self.pattern_mapping[p] for p in self.negative_patterns
        ]
        self.positive_critical_indices = [
            self.pattern_mapping[p] for p in self.positive_critical
        ]
    
    def __call__(self, pores_pred, pores_target, pattern_pred, growth_level):
        """
        Args:
            pores_pred: [B] pores logits
            pores_target: [B] pores labels (0/1)
            pattern_pred: [B, num_patterns] pattern logits
            growth_level: [B] growth level (0=negative, 1=positive)
        """
        batch_size = pores_pred.size(0)
        pattern_class = torch.argmax(pattern_pred, dim=-1)
        
        # 动态权重
        weights = torch.zeros(batch_size, device=pores_pred.device)
        
        for i in range(batch_size):
            level = growth_level[i].item()
            pattern = pattern_class[i].item()
            
            if level == 0:  # Negative
                weights[i] = 15.0  # 高权重
            elif pattern in self.positive_critical_indices:  # Positive critical
                weights[i] = 15.0  # 高权重
            else:
                weights[i] = 0.1  # 低权重
        
        # BCE Loss
        loss = F.binary_cross_entropy_with_logits(
            pores_pred, 
            pores_target.float(),
            reduction='none'
        )
        
        weighted_loss = loss * weights
        return weighted_loss.mean()
```

### 训练脚本修改

```python
# scripts/train_multilevel_mobilenetv3_v0.10.0.py

# 添加 pattern-conditional loss
from training.pattern_conditional_loss import PatternConditionalPoresLoss

# 初始化
pattern_mapping = dataset.label_mappings['growth_pattern']
pores_loss_fn = PatternConditionalPoresLoss(pattern_mapping)

# 训练循环中
for batch in train_loader:
    # ... forward pass ...
    
    # Interference loss (使用 pattern-conditional)
    pores_loss = pores_loss_fn(
        interference_outputs[:, 0],  # pores
        interference_labels[:, 0],   # pores target
        pattern_outputs,             # pattern logits
        growth_level_labels          # growth level
    )
    
    # 其他 interference factors 使用标准 BCE
    other_loss = F.binary_cross_entropy_with_logits(
        interference_outputs[:, 1:],
        interference_labels[:, 1:],
        pos_weight=torch.tensor([3.0, 5.0, 10.0])
    )
    
    interference_loss = pores_loss + other_loss
    
    # 总损失
    total_loss = (
        task_weights[0] * level_loss +
        task_weights[1] * pattern_loss +
        task_weights[2] * interference_loss
    )
```

---

## 成功标准

### v0.10.0 目标

**主要指标**：
- **Pores Recall >= 60%** (517/863)
- Pores Precision >= 40%
- Pores F1 >= 45%

**业务目标**：
- **Pores Recall >= 75%** (647/863)

### 其他任务保持

- Growth Level F1 >= 98%
- Growth Pattern Accuracy >= 80%
- Interference Overall F1 >= 55%

---

## 后续优化（如果需要）

### 如果 v0.10.0 达到 60-70%

**微调建议**：
1. 增加 pores 条件权重 15.0 → 20.0
2. 延长训练 40 → 50 epochs
3. 使用 Focal Loss 替代 BCE

### 如果 v0.10.0 < 60%

**考虑方案 4**（独立检测器）或**方案 2**（两阶段训练）

---

**报告时间**: 2025-10-04  
**推荐方案**: v0.10.0 (Pattern-Conditional Pores Loss)  
**预期效果**: Pores Recall 19.9% → 60-75%
