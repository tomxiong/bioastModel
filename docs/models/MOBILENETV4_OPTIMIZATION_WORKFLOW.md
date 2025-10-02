# MobileNetV4 优化工作流程

本文档说明MobileNetV4模型的完整优化迭代机制。

---

## 工作流程概览

```
┌─────────────────────────────────────────────────────────────┐
│                    迭代优化循环                              │
│                                                             │
│  1. 训练新版本                                               │
│      ↓                                                      │
│  2. 性能分析                                                 │
│      ↓                                                      │
│  3. 错误样本分析                                             │
│      ↓                                                      │
│  4. 制定改进方案                                             │
│      ↓                                                      │
│  5. 更新版本历史                                             │
│      ↓                                                      │
│  6. 下一轮迭代 ────────────────────────────────┐             │
│      │                                         │             │
│      └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心工具

### 1. 自动化训练脚本

**脚本**: `scripts/mobilenetv4_train_iteration.sh`

**功能**:
- ✅ 自动执行完整训练流程
- ✅ 自动生成性能分析报告
- ✅ 自动进行错误样本分析
- ✅ 统一的版本管理

**用法**:

```bash
# 基础用法（使用默认参数）
./scripts/mobilenetv4_train_iteration.sh -v v1.1

# 自定义参数
./scripts/mobilenetv4_train_iteration.sh \
    -v v1.2 \
    -s medium \        # 模型大小
    -e 30 \            # 训练轮数
    -l 0.001 \         # 学习率
    -w 5 \             # Warmup轮数
    -p 20 \            # Early stopping patience
    -b 64              # 批量大小
```

**参数说明**:

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `-v` | 版本号 (必需) | - | v1.1, v1.2, ... |
| `-s` | 模型大小 | small | small, medium, large |
| `-e` | 训练轮数 | 20 | 任意正整数 |
| `-l` | 学习率 | 0.0015 | 0.0001-0.01 |
| `-w` | Warmup轮数 | 3 | 0-10 |
| `-p` | Patience | 15 | 5-30 |
| `-b` | 批量大小 | 64 | 16, 32, 64, 128 |

### 2. 错误分析工具

**脚本**: `scripts/mobilenetv4_error_analysis.py`

**功能**:
- ✅ 分析所有错误预测样本
- ✅ 生成混淆矩阵
- ✅ 识别低置信度错误
- ✅ 统计错误模式

**用法**:

```bash
python scripts/mobilenetv4_error_analysis.py \
    --checkpoint experiments/mobilenetv4_v1.1/best_model.pth \
    --model_size small \
    --output_dir experiments/mobilenetv4_v1.1
```

**生成文件**:
- `ERROR_ANALYSIS_REPORT.md` - 错误分析报告
- `error_samples.json` - 错误样本详情
- `error_statistics.json` - 错误统计信息

### 3. 版本历史管理

**文档**: `docs/models/MOBILENETV4_VERSION_HISTORY.md`

**内容**:
- ✅ 所有版本的性能记录
- ✅ 版本间对比分析
- ✅ 改进历程追踪
- ✅ 文件清单管理

---

## 标准工作流程

### 步骤1: 启动新版本训练

```bash
# 例如：训练v1.1版本
./scripts/mobilenetv4_train_iteration.sh -v v1.1
```

**自动执行**:
1. 训练模型 (20 epochs)
2. 生成训练报告
3. 错误样本分析
4. 保存所有结果

**预计时间**: 约2-3分钟

### 步骤2: 查看训练结果

```bash
# 查看训练日志
cat experiments/mobilenetv4_v1.1/training.log

# 查看性能报告
cat experiments/mobilenetv4_v1.1/TRAINING_ANALYSIS_REPORT.md

# 查看错误分析
cat experiments/mobilenetv4_v1.1/ERROR_ANALYSIS_REPORT.md
```

### 步骤3: 分析关键指标

**必看指标**:

1. **总体准确率**
   - 目标: 持续提升
   - 基准: v1.0 = 92.75%

2. **各任务准确率**
   - Growth Level: 目标 >98%
   - Growth Pattern: 目标 >86%
   - Interference Factors: 目标 >92%

3. **训练稳定性**
   - 有无过拟合
   - Loss收敛情况
   - 训练/验证Gap

4. **错误模式**
   - 主要错误类型
   - 低置信度样本
   - 混淆矩阵分析

### 步骤4: 制定改进方案

基于分析结果，识别问题并制定方案：

**常见问题 → 解决方案**:

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| Growth Pattern准确率低 | 数据不平衡 | 增加类别权重 |
| 训练Loss波动大 | 学习率过高 | 降低学习率 |
| 验证Loss不降 | 欠拟合 | 延长训练/增大模型 |
| 训练/验证Gap大 | 过拟合 | 增加Dropout/正则化 |
| 低置信度错误多 | 特征不足 | 尝试更大模型 |

### 步骤5: 更新版本历史

编辑 `docs/models/MOBILENETV4_VERSION_HISTORY.md`:

1. 填写v1.1的完整性能数据
2. 添加对比分析
3. 记录改进效果
4. 制定v1.2计划

### 步骤6: 启动下一轮迭代

```bash
# 根据改进方案启动v1.2
./scripts/mobilenetv4_train_iteration.sh \
    -v v1.2 \
    -e 25 \              # 调整参数
    -l 0.001             # 根据分析优化
```

---

## 版本规划示例

### v1.0 → v1.1 (当前)

**v1.0问题**:
- ❌ Growth Pattern 85.33% (未达86%目标)
- ❌ 训练时间过短 (10 epochs)
- ❌ 学习率波动导致Loss震荡

**v1.1改进**:
- ✅ 延长训练至20 epochs
- ✅ 降低学习率 0.002 → 0.0015
- ✅ 减少Warmup 5 → 3 epochs

**v1.1目标**:
- 总体准确率: >93.2%
- Growth Pattern: >86.5%

### v1.1 → v1.2 (计划)

**待定，基于v1.1分析结果**

可能方向:
1. 如v1.1达标 → 尝试Medium模型
2. 如仍有差距 → 数据增强/类别权重
3. 如过拟合 → 增加正则化

---

## 文件组织规范

### 实验目录结构

```
experiments/mobilenetv4_v1.1/
├── best_model.pth                    # 最佳模型权重
├── latest_model.pth                  # 最新模型权重
├── config.json                       # 训练配置
├── model_info.json                   # 模型结构信息
├── label_info.json                   # 标签映射
├── improved_training_history.json    # 训练曲线数据
├── training.log                      # 完整训练日志
├── TRAINING_ANALYSIS_REPORT.md       # 性能分析报告
├── ERROR_ANALYSIS_REPORT.md          # 错误样本分析报告
├── error_samples.json                # 错误样本详情
└── error_statistics.json             # 错误统计
```

### 必需文件检查

每个版本完成后，确保包含：

- [ ] `best_model.pth` - 模型权重
- [ ] `config.json` - 配置文件
- [ ] `improved_training_history.json` - 训练历史
- [ ] `TRAINING_ANALYSIS_REPORT.md` - 性能报告
- [ ] `ERROR_ANALYSIS_REPORT.md` - 错误分析
- [ ] `training.log` - 训练日志

---

## 性能追踪

### 关键指标追踪表

| 版本 | 总体准确率 | Growth Level | Growth Pattern | Interference | 训练时长 | 改进点 |
|------|-----------|--------------|---------------|-------------|---------|--------|
| v1.0 | 92.75% | 98.57% | 85.33% | 94.76% | 67s | 基线 |
| v1.1 | TBD | TBD | TBD | TBD | ~150s | 延长训练+学习率优化 |
| v1.2 | - | - | - | - | - | 待定 |

### 目标设定

**短期目标** (v1.1-v1.2):
- 总体准确率: >93.5%
- Growth Pattern: >87%
- 消除训练波动

**中期目标** (v1.3-v1.5):
- 总体准确率: >94.5%
- 尝试Medium模型
- 建立Ensemble系统

**长期目标** (v2.0+):
- 总体准确率: >95%
- 架构创新
- 部署优化

---

## 最佳实践

### 训练策略

1. **渐进式改进**
   - 每次只改变1-2个参数
   - 便于分析改进效果

2. **充分训练**
   - 至少20 epochs
   - 等待Early Stopping触发
   - 观察收敛趋势

3. **对照实验**
   - 保持数据集一致
   - 使用相同随机种子（可选）
   - 记录所有超参数

### 分析策略

1. **全面分析**
   - 训练曲线
   - 错误样本
   - 混淆矩阵
   - 置信度分布

2. **对比分析**
   - 与前一版本对比
   - 与基准版本对比
   - 与目标对比

3. **深入挖掘**
   - 识别系统性错误
   - 分析边界案例
   - 找出数据问题

### 版本管理

1. **清晰命名**
   - 使用语义化版本号
   - v1.x: 同架构优化
   - v2.x: 架构升级

2. **完整记录**
   - 所有改动都记录
   - 性能数据完整
   - 保留所有报告

3. **及时归档**
   - 定期清理临时文件
   - 保留最佳模型
   - 备份重要版本

---

## 快速参考

### 常用命令

```bash
# 1. 训练新版本
./scripts/mobilenetv4_train_iteration.sh -v v1.x

# 2. 查看训练进度
tail -f experiments/mobilenetv4_v1.x/training.log

# 3. 检查训练是否完成
ls -lh experiments/mobilenetv4_v1.x/*.pth

# 4. 查看性能
cat experiments/mobilenetv4_v1.x/TRAINING_ANALYSIS_REPORT.md

# 5. 查看错误分析
cat experiments/mobilenetv4_v1.x/ERROR_ANALYSIS_REPORT.md

# 6. 单独运行错误分析
python scripts/mobilenetv4_error_analysis.py \
    --checkpoint experiments/mobilenetv4_v1.x/best_model.pth \
    --model_size small \
    --output_dir experiments/mobilenetv4_v1.x
```

### 问题排查

**训练失败**:
1. 检查CUDA是否可用
2. 检查数据集路径
3. 查看training.log最后几行

**性能不理想**:
1. 查看训练曲线是否收敛
2. 检查是否过拟合/欠拟合
3. 分析错误样本模式

**文件缺失**:
1. 确认训练完成
2. 检查实验目录权限
3. 重新运行对应步骤

---

## 附录

### A. 超参数调优指南

| 超参数 | 范围 | 推荐值 | 影响 |
|--------|------|--------|------|
| Learning Rate | 0.0001-0.01 | 0.0015 | 收敛速度和稳定性 |
| Batch Size | 16-128 | 64 | 训练速度和泛化 |
| Num Epochs | 10-50 | 20-30 | 收敛程度 |
| Warmup Epochs | 0-10 | 3-5 | 训练稳定性 |
| Dropout | 0.1-0.5 | 0.3 | 过拟合控制 |
| Weight Decay | 0.001-0.1 | 0.01 | 正则化强度 |

### B. 架构变体对比

| 模型 | 参数量 | 推理速度 | 推荐场景 |
|------|--------|---------|---------|
| Small | 952K | ~5ms | 边缘设备、实时应用 |
| Medium | 1.33M | ~8ms | 平衡性能和速度 |
| Large | 1.83M | ~12ms | 追求最高准确率 |

### C. 版本历史文档链接

- [版本历史总览](MOBILENETV4_VERSION_HISTORY.md)
- [v1.0详细报告](../../experiments/mobilenetv4_small_quick/TRAINING_ANALYSIS_REPORT.md)
- [v1.1详细报告](../../experiments/mobilenetv4_v1.1/TRAINING_ANALYSIS_REPORT.md) (训练中)

---

**文档维护**: 随优化流程更新
**最后更新**: 2025-10-03
**维护者**: Claude Code + User
