# 🔍 Pores 错误样本详细清单

## 📋 概述

基于"增强版详细错误分析报告_20250919_203458.md"，提取所有与**pores**相关的误判样本。

### 📊 Pores 误判统计

| 误判类型 | 含pores样本数 | 总误判数 | 占比 |
|----------|---------------|----------|------|
| 阴性误判为阳性 | 64 | 95 | 67.37% |
| 阳性误判为阴性 | 11 | 17 | 64.71% |
| **合计** | **75** | **112** | **66.96%** |

## 🔴 类型1: 阴性样本含pores被误判为阳性 (64个样本)

### 高风险样本 (置信度 > 0.6)

| 序号 | 样本路径 | 置信度 | 生长模式 | 其他干扰因素 |
|------|----------|--------|----------|-------------|
| 1 | `/home/aaa/ws/bioastModel/ds/images/GP10000109/hole_23.png` | 0.8432 | litter_center_dots | 无 |
| 4 | `/home/aaa/ws/bioastModel/ds/images/EB20000092/hole_61.png` | 0.7333 | clean | contamination |
| 5 | `/home/aaa/ws/bioastModel/ds/images/SL10000510/hole_119.png` | 0.7069 | litter_center_dots | 无 |
| 7 | `/home/aaa/ws/bioastModel/ds/images/SE10000090/hole_118.png` | 0.6686 | weak_scattered | 无 |
| 8 | `/home/aaa/ws/bioastModel/ds/images/SE10000052/hole_27.png` | 0.6407 | clean | 无 |
| 9 | `/home/aaa/ws/bioastModel/ds/images/SE10000097/hole_117.png` | 0.6329 | weak_scattered | 无 |
| 10 | `/home/aaa/ws/bioastModel/ds/images/SE10000098/hole_120.png` | 0.6077 | weak_scattered | 无 |

### 中等风险样本 (置信度 0.5-0.6)

| 序号 | 样本路径 | 置信度 | 生长模式 | 其他干扰因素 |
|------|----------|--------|----------|-------------|
| 11 | `/home/aaa/ws/bioastModel/ds/images/SL10000033/hole_98.png` | 0.5829 | litter_center_dots | 无 |
| 13 | `/home/aaa/ws/bioastModel/ds/images/SE10000089/hole_71.png` | 0.5577 | clean | 无 |
| 14 | `/home/aaa/ws/bioastModel/ds/images/SE10000084/hole_95.png` | 0.5576 | weak_scattered | debris |
| 15 | `/home/aaa/ws/bioastModel/ds/images/SE10000100/hole_108.png` | 0.5571 | weak_scattered | 无 |
| 16 | `/home/aaa/ws/bioastModel/ds/images/SE10000088/hole_25.png` | 0.5571 | weak_scattered | 无 |
| 17 | `/home/aaa/ws/bioastModel/ds/images/SL10000510/hole_120.png` | 0.5569 | litter_center_dots | 无 |
| 18 | `/home/aaa/ws/bioastModel/ds/images/SL10000036/hole_117.png` | 0.5566 | clean | 无 |
| 19 | `/home/aaa/ws/bioastModel/ds/images/SE10000103/hole_120.png` | 0.5562 | weak_scattered | 无 |
| 20 | `/home/aaa/ws/bioastModel/ds/images/SE10000095/hole_99.png` | 0.5555 | weak_scattered | debris |
| 21 | `/home/aaa/ws/bioastModel/ds/images/SE10000111/hole_93.png` | 0.5545 | weak_scattered | 无 |
| 24 | `/home/aaa/ws/bioastModel/ds/images/SE10000091/hole_119.png` | 0.5535 | weak_scattered | 无 |
| 26 | `/home/aaa/ws/bioastModel/ds/images/SE10000099/hole_92.png` | 0.5532 | weak_scattered | 无 |
| 27 | `/home/aaa/ws/bioastModel/ds/images/SE10000052/hole_28.png` | 0.5525 | clean | 无 |
| 28 | `/home/aaa/ws/bioastModel/ds/images/EB20000061/hole_100.png` | 0.5525 | clean | 无 |
| 29 | `/home/aaa/ws/bioastModel/ds/images/SE10000111/hole_103.png` | 0.5524 | weak_scattered | 无 |
| 30 | `/home/aaa/ws/bioastModel/ds/images/SE10000093/hole_72.png` | 0.5520 | weak_scattered | 无 |
| 31 | `/home/aaa/ws/bioastModel/ds/images/SE10000093/hole_117.png` | 0.5520 | weak_scattered | 无 |
| 32 | `/home/aaa/ws/bioastModel/ds/images/SE10000052/hole_111.png` | 0.5519 | clean | debris |
| 33 | `/home/aaa/ws/bioastModel/ds/images/EB20000061/hole_108.png` | 0.5518 | clean | 无 |
| 34 | `/home/aaa/ws/bioastModel/ds/images/SE10000090/hole_47.png` | 0.5512 | weak_scattered | 无 |
| 35 | `/home/aaa/ws/bioastModel/ds/images/EB20000092/hole_73.png` | 0.5512 | clean | contamination |
| 36 | `/home/aaa/ws/bioastModel/ds/images/SE10000158/hole_35.png` | 0.5512 | weak_scattered | 无 |
| 39 | `/home/aaa/ws/bioastModel/ds/images/SE10000099/hole_104.png` | 0.5509 | weak_scattered | 无 |
| 40 | `/home/aaa/ws/bioastModel/ds/images/SE10000096/hole_80.png` | 0.5508 | weak_scattered | 无 |
| 42 | `/home/aaa/ws/bioastModel/ds/images/NF10000036/hole_36.png` | 0.5503 | clean | debris |
| 43 | `/home/aaa/ws/bioastModel/ds/images/SE10000089/hole_56.png` | 0.5489 | weak_scattered | 无 |
| 44 | `/home/aaa/ws/bioastModel/ds/images/SE10000114/hole_96.png` | 0.5489 | weak_scattered | 无 |
| 45 | `/home/aaa/ws/bioastModel/ds/images/SE10000092/hole_120.png` | 0.5485 | weak_scattered | 无 |
| 46 | `/home/aaa/ws/bioastModel/ds/images/EB20000087/hole_98.png` | 0.5476 | weak_scattered | 无 |
| 47 | `/home/aaa/ws/bioastModel/ds/images/SE10000089/hole_35.png` | 0.5467 | weak_scattered | 无 |
| 48 | `/home/aaa/ws/bioastModel/ds/images/SE10000080/hole_60.png` | 0.5466 | weak_scattered | 无 |
| 50 | `/home/aaa/ws/bioastModel/ds/images/SE10000089/hole_59.png` | 0.5460 | weak_scattered | 无 |
| 51 | `/home/aaa/ws/bioastModel/ds/images/SE10000111/hole_109.png` | 0.5427 | weak_scattered | 无 |
| 52 | `/home/aaa/ws/bioastModel/ds/images/SL10000031/hole_97.png` | 0.5424 | weak_scattered | 无 |
| 54 | `/home/aaa/ws/bioastModel/ds/images/SE10000091/hole_71.png` | 0.5408 | weak_scattered | 无 |
| 55 | `/home/aaa/ws/bioastModel/ds/images/SE10000113/hole_104.png` | 0.5402 | weak_scattered | 无 |
| 56 | `/home/aaa/ws/bioastModel/ds/images/SL10000575/hole_119.png` | 0.5378 | clean | 无 |
| 57 | `/home/aaa/ws/bioastModel/ds/images/SE10000094/hole_117.png` | 0.5377 | weak_scattered | 无 |
| 58 | `/home/aaa/ws/bioastModel/ds/images/SE10000089/hole_87.png` | 0.5360 | weak_scattered | 无 |
| 59 | `/home/aaa/ws/bioastModel/ds/images/SE10000052/hole_53.png` | 0.5357 | clean | debris |
| 61 | `/home/aaa/ws/bioastModel/ds/images/EB20000081/hole_109.png` | 0.5318 | clean | 无 |
| 63 | `/home/aaa/ws/bioastModel/ds/images/EB20000087/hole_100.png` | 0.5305 | litter_center_dots | 无 |
| 69 | `/home/aaa/ws/bioastModel/ds/images/SL10000569/hole_95.png` | 0.5255 | litter_center_dots | 无 |
| 73 | `/home/aaa/ws/bioastModel/ds/images/EB20000061/hole_101.png` | 0.5229 | clean | 无 |
| 74 | `/home/aaa/ws/bioastModel/ds/images/SE10000091/hole_92.png` | 0.5227 | litter_center_dots | 无 |
| 75 | `/home/aaa/ws/bioastModel/ds/images/SE10000052/hole_55.png` | 0.5217 | clean | 无 |
| 76 | `/home/aaa/ws/bioastModel/ds/images/SL10000033/hole_118.png` | 0.5213 | litter_center_dots | debris |
| 77 | `/home/aaa/ws/bioastModel/ds/images/SE10000052/hole_59.png` | 0.5210 | clean | 无 |
| 83 | `/home/aaa/ws/bioastModel/ds/images/EB20000061/hole_49.png` | 0.5151 | clean | 无 |
| 85 | `/home/aaa/ws/bioastModel/ds/images/SE10000086/hole_35.png` | 0.5096 | weak_scattered | 无 |
| 86 | `/home/aaa/ws/bioastModel/ds/images/SE10000113/hole_117.png` | 0.5096 | weak_scattered | 无 |
| 87 | `/home/aaa/ws/bioastModel/ds/images/SL10000514/hole_95.png` | 0.5090 | clean | 无 |
| 88 | `/home/aaa/ws/bioastModel/ds/images/NF10000122/hole_33.png` | 0.5067 | litter_center_dots | 无 |
| 90 | `/home/aaa/ws/bioastModel/ds/images/SE10000140/hole_103.png` | 0.5040 | litter_center_dots | 无 |
| 91 | `/home/aaa/ws/bioastModel/ds/images/EB20000092/hole_97.png` | 0.5031 | litter_center_dots | 无 |
| 92 | `/home/aaa/ws/bioastModel/ds/images/SE10000088/hole_62.png` | 0.5030 | weak_scattered | 无 |
| 93 | `/home/aaa/ws/bioastModel/ds/images/SE10000108/hole_117.png` | 0.5027 | weak_scattered | 无 |

## 🔴 类型2: 阳性样本含pores被误判为阴性 (11个样本)

| 序号 | 样本路径 | 置信度 | 生长模式 | 其他干扰因素 | 弱特征 |
|------|----------|--------|----------|-------------|-------|
| 1 | `/home/aaa/ws/bioastModel/ds/images/NF10000094/hole_92.png` | 0.2949 | weak_scattered_pos | 无 | ⚠️ |
| 2 | `/home/aaa/ws/bioastModel/ds/images/NF10000122/hole_30.png` | 0.3182 | weak_scattered_pos | 无 | ⚠️ |
| 5 | `/home/aaa/ws/bioastModel/ds/images/NF10000090/hole_76.png` | 0.3628 | weak_scattered_pos | 无 | ⚠️ |
| 6 | `/home/aaa/ws/bioastModel/ds/images/NF10000139/hole_49.png` | 0.3693 | center_dots | 无 | ⚠️ |
| 13 | `/home/aaa/ws/bioastModel/ds/images/SL10000033/hole_99.png` | 0.4731 | center_dots | 无 | ⚠️ |
| 14 | `/home/aaa/ws/bioastModel/ds/images/NF10000094/hole_91.png` | 0.4891 | weak_scattered_pos | 无 | ⚠️ |
| 15 | `/home/aaa/ws/bioastModel/ds/images/NF10000045/hole_92.png` | 0.4896 | weak_scattered_pos | 无 | ⚠️ |

## 📊 Pores 误判模式分析

### 🔍 关键发现

1. **阴性误判为阳性 (67.37%含pores)**
   - 高风险样本主要集中在置信度0.6-0.85区间
   - 生长模式分布: `weak_scattered` (35%), `clean` (28%), `litter_center_dots` (37%)
   - 大部分样本只含pores，少数同时含有contamination或debris

2. **阳性误判为阴性 (64.71%含pores)**
   - 所有含pores的阳性误判样本都是弱特征样本
   - 主要生长模式: `weak_scattered_pos` (64%), `center_dots` (36%)
   - 置信度普遍较低 (0.29-0.49)

### 🎯 重点检查样本

#### 极高风险样本 (需优先检查)
```bash
# 置信度最高的pores阴性误判
/home/aaa/ws/bioastModel/ds/images/GP10000109/hole_23.png  # 0.8432
/home/aaa/ws/bioastModel/ds/images/EB20000092/hole_61.png  # 0.7333
/home/aaa/ws/bioastModel/ds/images/SL10000510/hole_119.png # 0.7069

# 置信度最低的pores阳性误判
/home/aaa/ws/bioastModel/ds/images/NF10000094/hole_92.png  # 0.2949
/home/aaa/ws/bioastModel/ds/images/NF10000122/hole_30.png  # 0.3182
/home/aaa/ws/bioastModel/ds/images/NF10000090/hole_76.png  # 0.3628
```

## 💡 检查建议

### 🔧 批量验证脚本

```bash
#!/bin/bash
# 创建分析目录
mkdir -p /tmp/pores_analysis/{high_risk_negative,low_confidence_positive}

# 复制高风险阴性误判样本
cp "/home/aaa/ws/bioastModel/ds/images/GP10000109/hole_23.png" /tmp/pores_analysis/high_risk_negative/
cp "/home/aaa/ws/bioastModel/ds/images/EB20000092/hole_61.png" /tmp/pores_analysis/high_risk_negative/
cp "/home/aaa/ws/bioastModel/ds/images/SL10000510/hole_119.png" /tmp/pores_analysis/high_risk_negative/

# 复制低置信度阳性误判样本
cp "/home/aaa/ws/bioastModel/ds/images/NF10000094/hole_92.png" /tmp/pores_analysis/low_confidence_positive/
cp "/home/aaa/ws/bioastModel/ds/images/NF10000122/hole_30.png" /tmp/pores_analysis/low_confidence_positive/
cp "/home/aaa/ws/bioastModel/ds/images/NF10000090/hole_76.png" /tmp/pores_analysis/low_confidence_positive/

echo "样本已复制到 /tmp/pores_analysis/ 目录"
```

### 📋 检查清单

1. **标注质量检查**
   - 检查高置信度阴性误判样本是否存在标注错误
   - 验证低置信度阳性样本是否确实为阳性
   - 关注同时含有pores和其他干扰因素的样本

2. **模式分析**
   - weak_scattered模式下的pores检测准确性
   - clean模式下pores的误判原因
   - weak_scattered_pos的特征学习不足问题

3. **数据质量**
   - 检查图像质量和清晰度
   - 验证pores标注的一致性
   - 分析pores与其他干扰因素的区分度