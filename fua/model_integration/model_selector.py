"""
模型选择器

提供智能的模型选择功能，基于多维度评估结果和业务需求，
自动选择最适合的模型部署
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import networkx as nx
from abc import ABC, abstractmethod

from .model_evaluator import EvaluationResult, EvaluationMetrics

logger = logging.getLogger(__name__)


class SelectionCriteria(Enum):
    """选择标准枚举"""
    ACCURACY = "accuracy"
    SPEED = "speed"
    MEMORY = "memory"
    BALANCED = "balanced"
    COST_EFFECTIVE = "cost_effective"
    ROBUSTNESS = "robustness"
    CUSTOM = "custom"


class SelectionStrategy(Enum):
    """选择策略枚举"""
    TOP_PERFORMER = "top_performer"
    PARETO_OPTIMAL = "pareto_optimal"
    WEIGHTED_SCORE = "weighted_score"
    THRESHOLD_BASED = "threshold_based"
    MULTI_OBJECTIVE = "multi_objective"
    ENSEMBLE = "ensemble"


@dataclass
class SelectionConstraint:
    """选择约束条件"""
    metric_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    operator: str = "ge"  # ge, le, eq
    
    def satisfies(self, value: float) -> bool:
        """检查是否满足约束"""
        if self.operator == "ge":
            return value >= self.min_value
        elif self.operator == "le":
            return value <= self.max_value
        elif self.operator == "eq":
            return abs(value - self.min_value) < 1e-6
        return True


@dataclass
class SelectionWeights:
    """选择权重配置"""
    accuracy: float = 0.3
    speed: float = 0.2
    memory: float = 0.2
    robustness: float = 0.1
    cost: float = 0.1
    custom_weights: Dict[str, float] = field(default_factory=dict)
    
    def normalize(self):
        """归一化权重"""
        total = sum([
            self.accuracy, self.speed, self.memory,
            self.robustness, self.cost, *self.custom_weights.values()
        ])
        if total > 0:
            self.accuracy /= total
            self.speed /= total
            self.memory /= total
            self.robustness /= total
            self.cost /= total
            for key in self.custom_weights:
                self.custom_weights[key] /= total


@dataclass
class SelectionResult:
    """选择结果"""
    selected_model_id: str
    selected_version_id: str
    selection_criteria: SelectionCriteria
    selection_strategy: SelectionStrategy
    score: float
    ranking: List[Tuple[str, float]]  # (model_id, score)
    constraints_satisfied: bool
    selection_reason: str
    additional_models: List[str] = field(default_factory=list)  # 用于集成或其他用途
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['selection_criteria'] = self.selection_criteria.value
        data['selection_strategy'] = self.selection_strategy.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelectionResult':
        """从字典创建"""
        data['selection_criteria'] = SelectionCriteria(data['selection_criteria'])
        data['selection_strategy'] = SelectionStrategy(data['selection_strategy'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ModelScorer(ABC):
    """模型评分器抽象基类"""
    
    @abstractmethod
    def score(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """计算模型分数"""
        pass


class AccuracyScorer(ModelScorer):
    """准确率评分器"""
    
    def score(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """基于准确率评分"""
        scores = {}
        for result in evaluation_results:
            # 加权平均准确率
            accuracy = result.metrics.accuracy
            f1 = result.metrics.f1_score
            auc = result.metrics.roc_auc
            
            scores[result.model_id] = 0.5 * accuracy + 0.3 * f1 + 0.2 * auc
        
        return scores


class SpeedScorer(ModelScorer):
    """速度评分器"""
    
    def score(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """基于推理速度评分"""
        scores = {}
        inference_times = [r.metrics.avg_inference_time for r in evaluation_results]
        
        if not inference_times:
            return scores
        
        # 归一化时间（越小越好）
        max_time = max(inference_times)
        min_time = min(inference_times)
        time_range = max_time - min_time if max_time != min_time else 1
        
        for result in evaluation_results:
            normalized_time = 1 - (result.metrics.avg_inference_time - min_time) / time_range
            scores[result.model_id] = normalized_time
        
        return scores


class MemoryScorer(ModelScorer):
    """内存评分器"""
    
    def score(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """基于内存使用评分"""
        scores = {}
        memory_usages = [r.metrics.model_size_mb for r in evaluation_results]
        
        if not memory_usages:
            return scores
        
        # 归一化内存使用（越小越好）
        max_memory = max(memory_usages)
        min_memory = min(memory_usages)
        memory_range = max_memory - min_memory if max_memory != min_memory else 1
        
        for result in evaluation_results:
            normalized_memory = 1 - (result.metrics.model_size_mb - min_memory) / memory_range
            scores[result.model_id] = normalized_memory
        
        return scores


class ParetoFrontAnalyzer:
    """帕累托前沿分析器"""
    
    @staticmethod
    def find_pareto_front(results: List[EvaluationResult], 
                         metrics: List[str] = None) -> List[str]:
        """
        找到帕累托前沿模型
        
        Args:
            results: 评估结果列表
            metrics: 要考虑的指标列表
            
        Returns:
            帕累托前沿模型ID列表
        """
        if metrics is None:
            metrics = ['accuracy', 'avg_inference_time', 'model_size_mb']
        
        # 准备数据
        model_ids = []
        values = []
        
        for result in results:
            model_ids.append(result.model_id)
            row = []
            for metric in metrics:
                if metric == 'accuracy':
                    row.append(result.metrics.accuracy)
                elif metric == 'avg_inference_time':
                    # 速度是负向指标（越小越好）
                    row.append(-result.metrics.avg_inference_time)
                elif metric == 'model_size_mb':
                    # 内存是负向指标（越小越好）
                    row.append(-result.metrics.model_size_mb)
                else:
                    # 尝试从自定义指标获取
                    row.append(result.metrics.custom_metrics.get(metric, 0))
            values.append(row)
        
        if not values:
            return []
        
        values = np.array(values)
        
        # 找到帕累托前沿
        pareto_indices = []
        for i in range(len(values)):
            dominated = False
            for j in range(len(values)):
                if i != j:
                    # 检查是否被支配
                    if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                        dominated = True
                        break
            if not dominated:
                pareto_indices.append(i)
        
        return [model_ids[i] for i in pareto_indices]


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, weights: SelectionWeights):
        self.weights = weights
        self.weights.normalize()
    
    def optimize(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """
        多目标优化
        
        Args:
            evaluation_results: 评估结果列表
            
        Returns:
            模型分数字典
        """
        scores = {}
        
        # 归一化所有指标
        scaler = MinMaxScaler()
        
        # 准备特征矩阵
        features = []
        model_ids = []
        
        for result in evaluation_results:
            model_ids.append(result.model_id)
            feature_row = [
                result.metrics.accuracy,
                1 / (result.metrics.avg_inference_time + 1e-6),  # 速度转换
                1 / (result.metrics.model_size_mb + 1e-6),  # 内存转换
                result.metrics.roc_auc,
                result.metrics.balanced_accuracy
            ]
            features.append(feature_row)
        
        if not features:
            return scores
        
        features = np.array(features)
        features_normalized = scaler.fit_transform(features)
        
        # 计算加权分数
        for i, model_id in enumerate(model_ids):
            score = (
                self.weights.accuracy * features_normalized[i, 0] +
                self.weights.speed * features_normalized[i, 1] +
                self.weights.memory * features_normalized[i, 2] +
                self.weights.robustness * features_normalized[i, 3] +
                self.weights.cost * features_normalized[i, 4]
            )
            scores[model_id] = score
        
        return scores


class ModelSelector:
    """模型选择器主类"""
    
    def __init__(self,
                 output_dir: str = "./selection_results"):
        """
        初始化模型选择器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 评分器注册
        self.scorers: Dict[SelectionCriteria, ModelScorer] = {
            SelectionCriteria.ACCURACY: AccuracyScorer(),
            SelectionCriteria.SPEED: SpeedScorer(),
            SelectionCriteria.MEMORY: MemoryScorer()
        }
        
        # 选择历史
        self.selection_history: List[SelectionResult] = []
        
        # 帕累托分析器
        self.pareto_analyzer = ParetoFrontAnalyzer()
        
        logger.info("ModelSelector initialized")
    
    def register_scorer(self, criteria: SelectionCriteria, scorer: ModelScorer):
        """注册自定义评分器"""
        self.scorers[criteria] = scorer
        logger.info(f"Registered scorer for criteria: {criteria.value}")
    
    def select_model(self,
                     evaluation_results: List[EvaluationResult],
                     criteria: SelectionCriteria = SelectionCriteria.BALANCED,
                     strategy: SelectionStrategy = SelectionStrategy.WEIGHTED_SCORE,
                     constraints: List[SelectionConstraint] = None,
                     weights: SelectionWeights = None,
                     top_k: int = 1) -> SelectionResult:
        """
        选择最佳模型
        
        Args:
            evaluation_results: 评估结果列表
            criteria: 选择标准
            strategy: 选择策略
            constraints: 约束条件
            weights: 权重配置
            top_k: 选择前K个模型
            
        Returns:
            选择结果
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided")
        
        # 应用约束过滤
        filtered_results = self._apply_constraints(evaluation_results, constraints)
        
        if not filtered_results:
            return SelectionResult(
                selected_model_id="",
                selected_version_id="",
                selection_criteria=criteria,
                selection_strategy=strategy,
                score=0.0,
                ranking=[],
                constraints_satisfied=False,
                selection_reason="No models satisfy the constraints"
            )
        
        # 根据策略选择
        if strategy == SelectionStrategy.TOP_PERFORMER:
            result = self._select_top_performer(
                filtered_results, criteria, top_k
            )
        elif strategy == SelectionStrategy.PARETO_OPTIMAL:
            result = self._select_pareto_optimal(
                filtered_results, criteria, top_k
            )
        elif strategy == SelectionStrategy.WEIGHTED_SCORE:
            result = self._select_weighted_score(
                filtered_results, criteria, top_k, weights
            )
        elif strategy == SelectionStrategy.THRESHOLD_BASED:
            result = self._select_threshold_based(
                filtered_results, criteria, top_k
            )
        elif strategy == SelectionStrategy.MULTI_OBJECTIVE:
            result = self._select_multi_objective(
                filtered_results, top_k, weights
            )
        elif strategy == SelectionStrategy.ENSEMBLE:
            result = self._select_ensemble(
                filtered_results, criteria, top_k
            )
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        # 保存结果
        self.selection_history.append(result)
        self._save_selection_result(result)
        
        logger.info(f"Model selection completed: {result.selected_model_id}")
        return result
    
    def _apply_constraints(self,
                          results: List[EvaluationResult],
                          constraints: List[SelectionConstraint] = None) -> List[EvaluationResult]:
        """应用约束条件过滤"""
        if not constraints:
            return results
        
        filtered = []
        for result in results:
            satisfies_all = True
            for constraint in constraints:
                value = self._get_metric_value(result, constraint.metric_name)
                if not constraint.satisfies(value):
                    satisfies_all = False
                    break
            
            if satisfies_all:
                filtered.append(result)
        
        return filtered
    
    def _get_metric_value(self, result: EvaluationResult, metric_name: str) -> float:
        """获取指标值"""
        if metric_name == 'accuracy':
            return result.metrics.accuracy
        elif metric_name == 'precision':
            return result.metrics.precision
        elif metric_name == 'recall':
            return result.metrics.recall
        elif metric_name == 'f1_score':
            return result.metrics.f1_score
        elif metric_name == 'roc_auc':
            return result.metrics.roc_auc
        elif metric_name == 'inference_time':
            return result.metrics.avg_inference_time
        elif metric_name == 'memory_usage':
            return result.metrics.model_size_mb
        else:
            return result.metrics.custom_metrics.get(metric_name, 0.0)
    
    def _select_top_performer(self,
                            results: List[EvaluationResult],
                            criteria: SelectionCriteria,
                            top_k: int) -> SelectionResult:
        """选择最佳表现者"""
        scorer = self.scorers.get(criteria, self.scorers[SelectionCriteria.ACCURACY])
        scores = scorer.score(results)
        
        # 排序
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_model = ranking[0][0]
        selected_result = next(r for r in results if r.model_id == selected_model)
        
        additional_models = [model_id for model_id, _ in ranking[1:top_k]]
        
        return SelectionResult(
            selected_model_id=selected_model,
            selected_version_id=selected_result.version_id,
            selection_criteria=criteria,
            selection_strategy=SelectionStrategy.TOP_PERFORMER,
            score=ranking[0][1],
            ranking=ranking,
            constraints_satisfied=True,
            selection_reason=f"Top performer based on {criteria.value}",
            additional_models=additional_models
        )
    
    def _select_pareto_optimal(self,
                              results: List[EvaluationResult],
                              criteria: SelectionCriteria,
                              top_k: int) -> SelectionResult:
        """选择帕累托最优模型"""
        pareto_models = self.pareto_analyzer.find_pareto_front(results)
        
        if not pareto_models:
            # 如果没有帕累托前沿，回退到top performer
            return self._select_top_performer(results, criteria, top_k)
        
        # 从帕累托前沿中选择
        pareto_results = [r for r in results if r.model_id in pareto_models]
        scorer = self.scorers.get(criteria, self.scorers[SelectionCriteria.ACCURACY])
        scores = scorer.score(pareto_results)
        
        # 排序
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_model = ranking[0][0]
        selected_result = next(r for r in pareto_results if r.model_id == selected_model)
        
        additional_models = [model_id for model_id, _ in ranking[1:min(top_k, len(ranking))]]
        
        return SelectionResult(
            selected_model_id=selected_model,
            selected_version_id=selected_result.version_id,
            selection_criteria=criteria,
            selection_strategy=SelectionStrategy.PARETO_OPTIMAL,
            score=ranking[0][1],
            ranking=ranking,
            constraints_satisfied=True,
            selection_reason=f"Pareto optimal model based on {criteria.value}",
            additional_models=additional_models,
            metadata={'pareto_front': pareto_models}
        )
    
    def _select_weighted_score(self,
                              results: List[EvaluationResult],
                              criteria: SelectionCriteria,
                              top_k: int,
                              weights: SelectionWeights = None) -> SelectionResult:
        """基于加权分数选择"""
        if weights is None:
            weights = SelectionWeights()
        
        optimizer = MultiObjectiveOptimizer(weights)
        scores = optimizer.optimize(results)
        
        # 排序
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_model = ranking[0][0]
        selected_result = next(r for r in results if r.model_id == selected_model)
        
        additional_models = [model_id for model_id, _ in ranking[1:top_k]]
        
        return SelectionResult(
            selected_model_id=selected_model,
            selected_version_id=selected_result.version_id,
            selection_criteria=criteria,
            selection_strategy=SelectionStrategy.WEIGHTED_SCORE,
            score=ranking[0][1],
            ranking=ranking,
            constraints_satisfied=True,
            selection_reason=f"Best weighted score with weights: {weights.to_dict()}",
            additional_models=additional_models,
            metadata={'weights': weights.to_dict()}
        )
    
    def _select_threshold_based(self,
                                results: List[EvaluationResult],
                                criteria: SelectionCriteria,
                                top_k: int) -> SelectionResult:
        """基于阈值选择"""
        # 定义默认阈值
        thresholds = {
            'accuracy': 0.9,
            'f1_score': 0.85,
            'roc_auc': 0.9,
            'inference_time': 0.01,  # 10ms
            'memory_usage': 10.0  # 10MB
        }
        
        # 过滤满足阈值的模型
        qualified = []
        for result in results:
            qualifies = True
            for metric, threshold in thresholds.items():
                value = self._get_metric_value(result, metric)
                if metric in ['inference_time', 'memory_usage']:
                    if value > threshold:
                        qualifies = False
                        break
                else:
                    if value < threshold:
                        qualifies = False
                        break
            
            if qualifies:
                qualified.append(result)
        
        if not qualified:
            # 如果没有模型满足阈值，选择最佳模型
            return self._select_top_performer(results, criteria, top_k)
        
        # 从合格模型中选择最佳
        scorer = self.scorers.get(criteria, self.scorers[SelectionCriteria.ACCURACY])
        scores = scorer.score(qualified)
        
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_model = ranking[0][0]
        selected_result = next(r for r in qualified if r.model_id == selected_model)
        
        additional_models = [model_id for model_id, _ in ranking[1:min(top_k, len(ranking))]]
        
        return SelectionResult(
            selected_model_id=selected_model,
            selected_version_id=selected_result.version_id,
            selection_criteria=criteria,
            selection_strategy=SelectionStrategy.THRESHOLD_BASED,
            score=ranking[0][1],
            ranking=ranking,
            constraints_satisfied=True,
            selection_reason=f"Best model satisfying thresholds: {thresholds}",
            additional_models=additional_models,
            metadata={'thresholds': thresholds}
        )
    
    def _select_multi_objective(self,
                              results: List[EvaluationResult],
                              top_k: int,
                              weights: SelectionWeights = None) -> SelectionResult:
        """多目标优化选择"""
        # 使用加权分数策略作为多目标优化的实现
        return self._select_weighted_score(
            results, SelectionCriteria.CUSTOM, top_k, weights
        )
    
    def _select_ensemble(self,
                        results: List[EvaluationResult],
                        criteria: SelectionCriteria,
                        top_k: int) -> SelectionResult:
        """选择集成模型"""
        # 选择多样性高的模型
        scorer = self.scorers.get(criteria, self.scorers[SelectionCriteria.ACCURACY])
        scores = scorer.score(results)
        
        # 基于分数和多样性选择
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前top_k个模型，考虑多样性
        selected_models = []
        remaining_models = ranking.copy()
        
        while len(selected_models) < top_k and remaining_models:
            # 选择下一个最佳模型
            next_model = remaining_models[0][0]
            selected_models.append(next_model)
            remaining_models.pop(0)
            
            # 移除相似的模型（基于性能）
            if remaining_models:
                current_score = scores[next_model]
                remaining_models = [
                    (model_id, score) for model_id, score in remaining_models
                    if abs(score - current_score) > 0.05  # 5%的差异阈值
                ]
        
        selected_model = selected_models[0]
        selected_result = next(r for r in results if r.model_id == selected_model)
        
        return SelectionResult(
            selected_model_id=selected_model,
            selected_version_id=selected_result.version_id,
            selection_criteria=criteria,
            selection_strategy=SelectionStrategy.ENSEMBLE,
            score=scores[selected_model],
            ranking=ranking,
            constraints_satisfied=True,
            selection_reason=f"Ensemble selection with {len(selected_models)} diverse models",
            additional_models=selected_models[1:],
            metadata={'ensemble_models': selected_models}
        )
    
    def batch_select(self,
                    evaluation_results_by_dataset: Dict[str, List[EvaluationResult]],
                    criteria: SelectionCriteria = SelectionCriteria.BALANCED,
                    strategy: SelectionStrategy = SelectionStrategy.WEIGHTED_SCORE) -> Dict[str, SelectionResult]:
        """
        批量选择模型
        
        Args:
            evaluation_results_by_dataset: 按数据集分组的评估结果
            criteria: 选择标准
            strategy: 选择策略
            
        Returns:
            每个数据集的选择结果
        """
        results = {}
        
        for dataset_name, eval_results in evaluation_results_by_dataset.items():
            logger.info(f"Selecting model for dataset: {dataset_name}")
            
            result = self.select_model(
                eval_results, criteria, strategy
            )
            
            results[dataset_name] = result
        
        return results
    
    def analyze_selection_stability(self,
                                  evaluation_results: List[EvaluationResult],
                                  n_iterations: int = 10) -> Dict[str, float]:
        """
        分析选择稳定性
        
        Args:
            evaluation_results: 评估结果
            n_iterations: 迭代次数
            
        Returns:
            模型选择稳定性分数
        """
        selection_counts = {}
        
        for i in range(n_iterations):
            # 采样（这里简化为使用全部数据）
            sampled_results = evaluation_results
            
            # 选择模型
            result = self.select_model(
                sampled_results,
                criteria=SelectionCriteria.BALANCED,
                strategy=SelectionStrategy.WEIGHTED_SCORE
            )
            
            model_id = result.selected_model_id
            selection_counts[model_id] = selection_counts.get(model_id, 0) + 1
        
        # 计算稳定性分数
        stability_scores = {}
        for model_id, count in selection_counts.items():
            stability_scores[model_id] = count / n_iterations
        
        return stability_scores
    
    def generate_selection_report(self,
                                results: List[SelectionResult],
                                output_path: str = None) -> str:
        """生成选择报告"""
        if output_path is None:
            output_path = self.output_dir / f"selection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report = f"# Model Selection Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 选择统计
        criteria_counts = {}
        strategy_counts = {}
        
        for result in results:
            criteria = result.selection_criteria.value
            strategy = result.selection_strategy.value
            
            criteria_counts[criteria] = criteria_counts.get(criteria, 0) + 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        report += "## Selection Statistics\n\n"
        report += "### Selection Criteria Usage\n"
        for criteria, count in criteria_counts.items():
            report += f"- {criteria}: {count} times\n"
        
        report += "\n### Selection Strategy Usage\n"
        for strategy, count in strategy_counts.items():
            report += f"- {strategy}: {count} times\n"
        
        # 最近选择
        if results:
            report += "\n## Recent Selections\n\n"
            recent = results[-5:]  # 最近5次选择
            
            report += "| Model | Criteria | Strategy | Score | Time |\n"
            report += "|-------|----------|----------|-------|------|\n"
            
            for result in recent:
                report += f"| {result.selected_model_id} | {result.selection_criteria.value} | "
                report += f"{result.selection_strategy.value} | {result.score:.4f} | "
                report += f"{result.timestamp.strftime('%H:%M:%S')} |\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Selection report saved to: {output_path}")
        return str(output_path)
    
    def _save_selection_result(self, result: SelectionResult):
        """保存选择结果"""
        result_path = self.output_dir / f"selection_{result.selected_model_id}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_selection_result(self, model_id: str, timestamp: str) -> Optional[SelectionResult]:
        """加载选择结果"""
        result_path = self.output_dir / f"selection_{model_id}_{timestamp}.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
                return SelectionResult.from_dict(data)
        return None
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """获取选择摘要"""
        if not self.selection_history:
            return {}
        
        # 统计最常选择的模型
        model_selections = {}
        for result in self.selection_history:
            model_id = result.selected_model_id
            model_selections[model_id] = model_selections.get(model_id, 0) + 1
        
        most_selected = max(model_selections.items(), key=lambda x: x[1])
        
        # 统计策略使用
        strategy_usage = {}
        for result in self.selection_history:
            strategy = result.selection_strategy.value
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        return {
            'total_selections': len(self.selection_history),
            'most_selected_model': {
                'model_id': most_selected[0],
                'selection_count': most_selected[1]
            },
            'strategy_usage': strategy_usage,
            'average_score': np.mean([r.score for r in self.selection_history]),
            'constraints_satisfaction_rate': sum(1 for r in self.selection_history if r.constraints_satisfied) / len(self.selection_history)
        }


def create_model_selector(output_dir: str = "./selection_results") -> ModelSelector:
    """创建模型选择器实例"""
    return ModelSelector(output_dir)