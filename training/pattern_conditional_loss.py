#!/usr/bin/env python3
"""
Pattern-Conditional Interference Loss for v0.10.0
基于 growth_pattern 和 growth_level 的条件化 Interference Loss
"""

import torch
import torch.nn.functional as F


class PatternConditionalInterferenceLoss:
    """
    Pattern-Conditional Interference Loss (v0.10.0核心创新)

    业务逻辑:
    1. Negative 样本: 需要检测 pores (高权重)
    2. Positive 关键 pattern (center_dots, weak_scattered_pos): 需要检测 pores (高权重)
    3. 其他 Positive pattern: 不需要检测 pores (低权重)

    实现:
    - 动态计算每个样本的 pores 损失权重
    - 基于 growth_level 和 growth_pattern 预测结果
    - 其他 interference factors (artifacts, contamination, debris) 使用基础权重
    """

    def __init__(self,
                 pattern_mapping: dict,
                 base_weights: torch.Tensor,  # [pores, artifacts, debris, contamination] (按数据集顺序)
                 negative_pores_weight: float = 15.0,
                 positive_critical_pores_weight: float = 15.0,
                 other_pores_weight: float = 0.1,
                 pores_index: int = 0):
        """
        初始化 Pattern-Conditional Interference Loss

        Args:
            pattern_mapping: Pattern 名称到索引的映射字典
            base_weights: 基础权重 tensor [pores, artifacts, debris, contamination] (按数据集顺序)
            negative_pores_weight: Negative 样本的 pores 权重
            positive_critical_pores_weight: Positive 关键 pattern 的 pores 权重
            other_pores_weight: 其他样本的 pores 权重
            pores_index: pores 在 interference_factors 中的索引 (默认 0)
        """
        self.pattern_mapping = pattern_mapping
        self.base_weights = base_weights
        self.negative_pores_weight = negative_pores_weight
        self.positive_critical_pores_weight = positive_critical_pores_weight
        self.other_pores_weight = other_pores_weight
        self.pores_index = pores_index

        # 关键 pattern 列表
        self.positive_critical_patterns = ['center_dots', 'weak_scattered_pos']

        # 获取关键 pattern 的索引
        self.positive_critical_indices = [
            self.pattern_mapping[p] for p in self.positive_critical_patterns
            if p in self.pattern_mapping
        ]

        print(f"Pattern-Conditional Interference Loss initialized:")
        print(f"  Base weights: {self.base_weights.tolist()}")
        print(f"  Negative pores weight: {self.negative_pores_weight}")
        print(f"  Positive critical pores weight: {self.positive_critical_pores_weight}")
        print(f"  Positive critical patterns: {self.positive_critical_patterns}")
        print(f"  Positive critical indices: {self.positive_critical_indices}")
        print(f"  Other pores weight: {self.other_pores_weight}")

    def __call__(self,
                 interference_pred: torch.Tensor,
                 interference_target: torch.Tensor,
                 pattern_pred: torch.Tensor,
                 growth_level: torch.Tensor) -> torch.Tensor:
        """
        计算 Pattern-Conditional Interference Loss

        Args:
            interference_pred: Interference 预测 logits [batch_size, 4]
            interference_target: Interference 标签 [batch_size, 4]
            pattern_pred: Pattern 预测 logits [batch_size, num_patterns]
            growth_level: Growth level 标签 [batch_size]

        Returns:
            加权后的 interference 损失
        """
        batch_size = interference_pred.size(0)
        num_factors = interference_pred.size(1)

        # 获取预测的 pattern 类别
        pattern_class = torch.argmax(pattern_pred, dim=-1)

        # 创建动态权重矩阵 [batch_size, num_factors]
        pos_weights = self.base_weights.unsqueeze(0).repeat(batch_size, 1)

        # 为每个样本计算 pores 的条件权重
        for i in range(batch_size):
            level = growth_level[i].item()
            pattern = pattern_class[i].item()

            if level == 0:  # Negative
                pos_weights[i, self.pores_index] = self.negative_pores_weight
            elif pattern in self.positive_critical_indices:  # Positive 关键 pattern
                pos_weights[i, self.pores_index] = self.positive_critical_pores_weight
            else:  # 其他 Positive
                pos_weights[i, self.pores_index] = self.other_pores_weight

        # 计算逐样本、逐类别的 BCE 损失
        loss = F.binary_cross_entropy_with_logits(
            interference_pred,
            interference_target.float(),
            reduction='none'
        )

        # 应用权重
        weighted_loss = loss * pos_weights

        # 取平均
        return weighted_loss.mean()

    def get_stats(self, pattern_pred, growth_level):
        """
        获取当前批次样本分布统计
        """
        batch_size = pattern_pred.size(0)
        pattern_class = torch.argmax(pattern_pred, dim=-1)

        negative_count = (growth_level == 0).sum().item()
        positive_critical_count = 0
        other_count = 0

        for i in range(batch_size):
            level = growth_level[i].item()
            pattern = pattern_class[i].item()

            if level == 1:  # Positive
                if pattern in self.positive_critical_indices:
                    positive_critical_count += 1
                else:
                    other_count += 1

        return {
            'negative': negative_count,
            'positive_critical': positive_critical_count,
            'other': other_count,
            'total': batch_size
        }
