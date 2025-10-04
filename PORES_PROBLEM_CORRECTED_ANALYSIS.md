# Pores 问题分析（业务需求修正版）

## 业务需求澄清

### 原先理解（错误）
- Pores 是独立的干扰因素，需要单独检测
- 目标：提高 pores 检测的 F1 分数

### 实际业务需求（正确）
- **Pores 是 Pattern 误判的指示器**
- 业务逻辑：
  ```
  IF pattern == 'center_dots' OR pattern == 'weak_scattered_pos'
     AND pores == True
  THEN
     pattern 可能是误判
     需要重新评估或标记为不确定
  ```

---

## 数据验证

让我们验证这个业务逻辑在数据中的体现：

### 训练集统计（已知）
```
center_dots:
  - 总样本: 424
  - 有 pores: 179 (42.2%)
  - 无 pores: 245 (57.8%)

weak_scattered_pos:
  - 总样本: 182  
  - 有 pores: 112 (61.5%)
  - 无 pores: 70 (38.5%)
```

### 业务解读

**如果 pores 是误判指示器**：
1. `center_dots` + `pores` (179 样本, 42.2%)
   - 这些可能是误判为 center_dots 的样本
   - 真实 pattern 可能是其他（clean? litter_center_dots?）

2. `weak_scattered_pos` + `pores` (112 样本, 61.5%)
   - 这些可能是误判为 weak_scattered_pos 的样本
   - 真实 pattern 可能是 weak_scattered 或 clean

### 关键问题

**数据清理的影响**：
我们在第二轮清理时，移除了：
- `strong_scattered` + `pores` (91 个)
- `scattered` + `pores` (1 个)
- `heavy_growth` + `pores` (3 个)

但保留了：
- `center_dots` + `pores` (254 个，全数据集)
- `weak_scattered_pos` + `pores` (162 个，全数据集)

这说明：**我们正确地保留了业务上有意义的 pores 样本**！

---

## 正确的建模策略

### 方案 1: 规则后处理（推荐，最简单）

不需要模型学习 pores，使用规则：

```python
def correct_pattern_with_pores(pattern_pred, pores_pred, confidence_threshold=0.5):
    """
    使用 pores 修正 pattern 预测
    """
    # 如果检测到 pores
    if pores_pred > confidence_threshold:
        # 如果 pattern 是容易误判的类型
        if pattern_pred in ['center_dots', 'weak_scattered_pos']:
            # 降低置信度或标记为需要人工复核
            return {
                'pattern': pattern_pred,
                'pores': True,
                'confidence': 'low',
                'action': 'manual_review',
                'reason': 'Pores detected in ambiguous pattern'
            }
    
    # 正常情况
    return {
        'pattern': pattern_pred,
        'pores': pores_pred > confidence_threshold,
        'confidence': 'high'
    }
```

**优势**：
- ✅ 简单直接，符合业务逻辑
- ✅ 不需要重新训练模型
- ✅ 可解释性强
- ✅ 易于调整阈值

### 方案 2: 联合建模（如果需要自动化）

将 pores 作为 pattern 的辅助任务：

```python
class PatternWithPoresCorrector(nn.Module):
    def __init__(self):
        self.backbone = MobileNetV3()
        self.pattern_head = nn.Linear(1024, 10)  # 10 个 pattern
        self.pores_head = nn.Linear(1024, 1)     # pores (仅用于 center_dots/weak_scattered_pos)
        
    def forward(self, x):
        features = self.backbone(x)
        pattern_logits = self.pattern_head(features)
        pores_logits = self.pores_head(features)
        
        # Pattern softmax
        pattern_probs = F.softmax(pattern_logits, dim=-1)
        pattern_pred = torch.argmax(pattern_probs, dim=-1)
        
        # Pores sigmoid
        pores_prob = torch.sigmoid(pores_logits)
        
        # 业务规则：如果是 center_dots/weak_scattered_pos + pores，降低置信度
        # (在推理阶段实现)
        
        return pattern_pred, pores_prob
```

**训练策略**：
```python
# 仅在 center_dots 和 weak_scattered_pos 样本上训练 pores
if pattern_label in ['center_dots', 'weak_scattered_pos']:
    pores_loss = F.binary_cross_entropy_with_logits(
        pores_logits, pores_label, 
        pos_weight=torch.tensor([5.0])  # 强调 pores
    )
else:
    pores_loss = 0  # 其他 pattern 不学习 pores

total_loss = pattern_loss + 0.5 * pores_loss
```

### 方案 3: 分层决策树（最符合业务逻辑）

```
Step 1: 预测 Pattern
   ↓
Step 2: IF pattern in [center_dots, weak_scattered_pos]
        THEN 检测 Pores
        ELSE 跳过
   ↓
Step 3: IF pores == True
        THEN 标记为 "需要复核" 或 重新分类
        ELSE 确认 pattern
```

**实现**：
```python
class HierarchicalPatternPoresModel:
    def __init__(self):
        self.pattern_model = load_pattern_model()  # 现有模型
        self.pores_detector = load_pores_detector()  # 独立 pores 检测器
        
    def predict(self, image):
        # Step 1: 预测 pattern
        pattern, pattern_conf = self.pattern_model(image)
        
        # Step 2: 仅对特定 pattern 检测 pores
        if pattern in ['center_dots', 'weak_scattered_pos']:
            pores_prob = self.pores_detector(image)
            
            # Step 3: 如果检测到 pores，降低置信度
            if pores_prob > 0.5:
                return {
                    'pattern': pattern,
                    'confidence': 'low',
                    'pores': True,
                    'action': 'manual_review',
                    'alternative_patterns': ['clean', 'weak_scattered']
                }
        
        # 正常情况
        return {
            'pattern': pattern,
            'confidence': 'high',
            'pores': False
        }
```

---

## 重新评估 v0.9.9 结果

### 当前模型表现（v0.9.9）

**Pores 检测性能**：
- Precision: 50.0%
- Recall: 20.0%
- F1: 28.57%

**在业务逻辑下的解读**：

1. **Recall 20% 的意义**：
   - 864 个 pores 样本中，检测到 173 个
   - 如果这 173 个主要分布在 `center_dots` 和 `weak_scattered_pos`
   - 那么模型已经在部分实现业务逻辑！

让我们验证：检测到的 pores 主要在哪些 pattern？

### 需要验证的关键问题

1. **检测到的 173 个 pores 样本的 pattern 分布是什么？**
   - 如果主要是 center_dots + weak_scattered_pos → 符合业务需求 ✅
   - 如果是 clean + weak_scattered → 不符合业务需求 ❌

2. **遗漏的 691 个 pores 样本中，有多少是业务上不重要的？**
   - clean + pores: 可以遗漏（不影响业务）
   - center_dots + pores: 必须检测（影响误判修正）

---

## 正确的优化目标

### 业务目标（修正后）

**不是**：提升 pores 整体 F1 分数  
**而是**：提升 pores 在特定 pattern 上的检测准确率

**具体指标**：
```
Target Metric:
  Pores Recall on (center_dots + weak_scattered_pos)

当前可能的表现：
  Total center_dots + weak_scattered_pos with pores: ~291 (测试集)
  Detected: ~173 (假设)
  Recall: ~59% (假设)

业务目标:
  Recall >= 75% on (center_dots + weak_scattered_pos) with pores
```

### 优化策略（修正后）

#### 策略 1: 针对性训练（推荐）

```python
# 仅在 center_dots 和 weak_scattered_pos 上训练 pores
def custom_pores_loss(pores_pred, pores_target, pattern_label):
    # 仅计算 center_dots 和 weak_scattered_pos 的 pores loss
    mask = (pattern_label == 'center_dots') | (pattern_label == 'weak_scattered_pos')
    
    if mask.sum() > 0:
        loss = F.binary_cross_entropy_with_logits(
            pores_pred[mask], 
            pores_target[mask],
            pos_weight=torch.tensor([10.0])  # 高权重
        )
    else:
        loss = 0
    
    return loss
```

#### 策略 2: 后处理规则（最简单）

使用现有模型，添加规则：

```python
# 在推理时
if pattern in ['center_dots', 'weak_scattered_pos']:
    # 降低 pores 检测阈值
    pores = pores_prob > 0.3  # 原来 0.5，现在 0.3
else:
    # 其他 pattern 不关心 pores
    pores = False
```

#### 策略 3: 独立 Pores 检测器（针对特定 pattern）

```python
# 训练数据：仅 center_dots + weak_scattered_pos 样本
train_data = filter_samples(
    dataset, 
    patterns=['center_dots', 'weak_scattered_pos']
)

# 二分类：pores vs no_pores
pores_detector = BinaryClassifier(
    backbone='mobilenetv3',
    num_classes=2
)
```

---

## 实施建议

### 短期方案（立即可用）

1. **分析 v0.9.9 检测到的 pores 分布**
   ```python
   # 检查检测到的 173 个 pores 主要在哪些 pattern
   # 如果主要在 center_dots + weak_scattered_pos → 已经基本符合需求
   ```

2. **调整推理规则**
   ```python
   # 针对性降低阈值
   if pattern in ['center_dots', 'weak_scattered_pos']:
       pores_threshold = 0.3  # 提高召回率
   else:
       pores_threshold = 0.8  # 提高精确率（或直接忽略）
   ```

### 中期方案（重新训练）

3. **训练 pattern-specific pores 检测器**
   ```python
   # v0.10.0: 针对性 pores 检测
   # 仅在 center_dots + weak_scattered_pos 上优化 pores
   # 其他 pattern 忽略 pores
   ```

### 长期方案（架构优化）

4. **分层决策系统**
   ```python
   # 先预测 pattern
   # 再根据 pattern 决定是否检测 pores
   # 最后应用业务规则修正
   ```

---

## 结论

### 关键洞察

1. **Pores 不是独立任务**，而是 pattern 误判的指示器
2. **不需要在所有样本上检测 pores**，仅需在 center_dots 和 weak_scattered_pos 上
3. **当前的"特征冲突"实际上是合理的业务关联**
4. **v0.9.9 的 28.57% F1 可能已经部分满足业务需求**

### 下一步行动

**优先级 1**: 分析 v0.9.9 检测到的 pores 分布
- 检查 173 个检测到的 pores 主要在哪些 pattern
- 如果主要在 center_dots + weak_scattered_pos → 调整阈值即可

**优先级 2**: 实施规则后处理
- 不需要重新训练
- 仅在特定 pattern 上应用 pores 检测
- 降低这些 pattern 的 pores 阈值

**优先级 3**: 训练 pattern-specific pores 检测器（如果规则不够）
- 仅在 center_dots + weak_scattered_pos 样本上训练
- 目标：Recall >= 75% on these patterns

---

**报告时间**: 2025-10-04  
**状态**: 业务需求已澄清，策略已修正
