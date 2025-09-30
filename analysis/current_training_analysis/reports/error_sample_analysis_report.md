# 模型错误样本分析报告

生成时间: 2025-09-30 19:15:24

## 执行摘要

### Growth Level 任务
- 总样本数: 3,000
- 错误样本数: 62
- 错误率: 2.07%
- 假阴性 (FN): 36 (漏检率: 2.34%)
- 假阳性 (FP): 26 (误检率: 1.78%)

### Growth Pattern 任务
- 总体准确率: 84.17%
- 错误率最高的类别:
  1. irregular: 100.00% (5/5)
     主要误分类为: clustered (2次)
  2. scattered: 100.00% (6/6)
     主要误分类为: heavy_growth (2次)
  3. weak_scattered_pos: 70.27% (26/37)
     主要误分类为: weak_scattered (15次)

### Interference Factors 任务
- 总体准确率: 92.66%
- 各因子错误率:
  - pores: 17.13%
  - artifacts: 7.37%
  - debris: 4.70%
  - contamination: 0.17%

## 关键发现
1. **最需要改进的任务**: Growth Pattern (错误率: 15.83%)
2. **Growth Pattern最大问题**: irregular类别错误率高达100.00%
3. **Interference Factors最大问题**: pores因子错误率17.13%

## 改进建议
1. **Growth Level**: 假阴性率较高，建议降低分类阈值或增加正样本的训练权重
2. **Growth Pattern**: 重点关注irregular类别，考虑:
   - 增加该类别的训练样本
   - 使用数据增强技术
   - 调整类别权重
3. **Interference Factors**: pores因子识别困难，建议:
   - 收集更多该因子的标注样本
   - 优化特征提取方法
   - 考虑使用专门的检测模块