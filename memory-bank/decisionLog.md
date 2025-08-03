# Decision Log

菌落检测项目重要架构和实现决策记录
2025-01-02 更新 - 关键决策文档

## Decision: 采用渐进式重构方案

**时间**: 2025-01-02
**背景**: 项目存在大量冗余文件和不统一的工作流

### Rationale 
- 避免破坏现有功能和实验数据
- 保持向后兼容性，支持已有实验
- 逐步建立标准化工作流
- 降低重构风险，确保项目稳定性

### Implementation Details
- 建立 core/config/ 统一配置管理
- 创建 scripts/ 标准化脚本工具
- 保留 experiments/ 现有实验数据
- 实现配置文件向后兼容机制

## Decision: 统一配置管理系统设计

**时间**: 2025-01-02
**背景**: 多个脚本使用不同的配置方式，难以维护

### Rationale
- 集中管理所有模型和训练配置
- 简化新模型添加流程
- 提供一致的参数访问接口
- 支持配置版本控制和追踪

### Implementation Details
- model_configs.py: 模型架构配置
- training_configs.py: 训练超参数配置
- paths.py: 统一路径管理
- 支持动态配置加载和验证

## Decision: 向后兼容性优先策略

**时间**: 2025-01-02
**背景**: 现有实验缺少标准化配置文件

### Rationale
- 保护已投入的训练时间和计算资源
- 确保历史实验数据可继续使用
- 平滑过渡到新的工作流
- 避免重复训练已完成的模型

### Implementation Details
- evaluate_model_fixed.py 支持无config.json实验
- 自动重建缺失的配置信息
- 默认参数填充机制
- 兼容性警告和提示系统

## Decision: 模型扩展优先级排序

**时间**: 2025-01-02
**背景**: 需要确定新模型添加的顺序

### Rationale
- ConvNext: 现代卷积架构，性能优秀
- CoAtNet: 混合注意力机制，效率平衡
- 先轻量级后复杂模型的渐进策略
- 基于实际性能需求确定优先级

### Implementation Details
- 第一阶段: ConvNext-Tiny (轻量级)
- 第二阶段: CoAtNet-0 (效率平衡)
- 第三阶段: 更大规模模型
- 每个模型完整评估后再添加下一个

## Decision: 评估脚本修复策略

**时间**: 2025-01-02
**背景**: 原始evaluate_model.py存在缩进错误

### Rationale
- 创建修复版本而非直接修改原文件
- 保留原始文件作为参考
- 确保修复版本功能完整
- 便于问题追踪和回滚

### Implementation Details
- 创建 evaluate_model_fixed.py
- 修复所有缩进和语法错误
- 增强错误处理和用户提示
- 保持与原版本的功能一致性