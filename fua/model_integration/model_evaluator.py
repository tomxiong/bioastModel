"""
模型评估器

提供全面的模型评估功能，包括性能基准测试、多维度指标计算、
统计分析和可视化报告生成
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score, cohen_kappa_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EvaluationType(Enum):
    """评估类型枚举"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    CONFUSION_MATRIX = "confusion_matrix"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"
    ROBUSTNESS = "robustness"
    CALIBRATION = "calibration"


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    matthews_corrcoef: float = 0.0
    cohen_kappa: float = 0.0
    
    # 推理性能
    avg_inference_time: float = 0.0
    std_inference_time: float = 0.0
    min_inference_time: float = 0.0
    max_inference_time: float = 0.0
    p95_inference_time: float = 0.0
    p99_inference_time: float = 0.0
    
    # 内存使用
    model_size_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # 混淆矩阵
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # 额外指标
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationMetrics':
        """从字典创建"""
        return cls(**data)


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    model_id: str
    version_id: str
    dataset_name: str
    metrics: EvaluationMetrics
    evaluation_time: datetime = field(default_factory=datetime.now)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    predictions: Optional[list] = None
    probabilities: Optional[list] = None
    labels: Optional[list] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['evaluation_time'] = self.evaluation_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """从字典创建"""
        data['evaluation_time'] = datetime.fromisoformat(data['evaluation_time'])
        return cls(**data)


class BenchmarkSuite:
    """基准测试套件"""
    
    def __init__(self, datasets: Dict[str, Any], evaluation_types: List[EvaluationType] = None):
        """
        初始化基准测试套件
        
        Args:
            datasets: 数据集字典
            evaluation_types: 评估类型列表
        """
        self.datasets = datasets
        self.evaluation_types = evaluation_types or [
            EvaluationType.ACCURACY,
            EvaluationType.PRECISION,
            EvaluationType.RECALL,
            EvaluationType.F1_SCORE,
            EvaluationType.ROC_AUC,
            EvaluationType.INFERENCE_TIME,
            EvaluationType.MEMORY_USAGE
        ]
        self.results: List[EvaluationResult] = []
        
        logger.info(f"BenchmarkSuite initialized with {len(datasets)} datasets")
    
    def add_dataset(self, name: str, data_loader: Callable, description: str = ""):
        """添加数据集"""
        self.datasets[name] = {
            'loader': data_loader,
            'description': description
        }
        logger.info(f"Added dataset: {name}")
    
    def run_benchmark(self, model: nn.Module, model_id: str, version_id: str,
                     dataset_names: List[str] = None) -> Dict[str, EvaluationResult]:
        """
        运行基准测试
        
        Args:
            model: 要评估的模型
            model_id: 模型ID
            version_id: 版本ID
            dataset_names: 要测试的数据集列表
            
        Returns:
            评估结果字典
        """
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        results = {}
        
        for dataset_name in dataset_names:
            logger.info(f"Evaluating model {model_id} on dataset {dataset_name}")
            
            # 加载数据
            dataset_info = self.datasets[dataset_name]
            data_loader = dataset_info['loader']
            data = data_loader()
            
            # 评估模型
            result = self._evaluate_model(
                model=model,
                model_id=model_id,
                version_id=version_id,
                dataset_name=dataset_name,
                data=data,
                evaluation_types=self.evaluation_types
            )
            
            results[dataset_name] = result
            self.results.append(result)
        
        return results
    
    def _evaluate_model(self, model: nn.Module, model_id: str, version_id: str,
                       dataset_name: str, data: Tuple[torch.Tensor, torch.Tensor],
                       evaluation_types: List[EvaluationType]) -> EvaluationResult:
        """评估单个模型"""
        device = next(model.parameters()).device
        model.eval()
        
        X, y = data
        X = X.to(device)
        y = y.to(device)
        
        # 批量预测
        predictions = []
        probabilities = []
        inference_times = []
        
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                
                # 测量推理时间
                start_time = time.time()
                outputs = model(batch_X)
                inference_time = time.time() - start_time
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                predictions.append(preds.cpu().numpy())
                probabilities.append(probs.cpu().numpy())
                inference_times.append(inference_time / len(batch_X))
        
        predictions = np.concatenate(predictions)
        probabilities = np.concatenate(probabilities)
        
        # 计算指标
        metrics = EvaluationMetrics()
        
        # 基础分类指标
        if EvaluationType.ACCURACY in evaluation_types:
            metrics.accuracy = accuracy_score(y.cpu(), predictions)
        
        if EvaluationType.PRECISION in evaluation_types:
            metrics.precision = precision_score(y.cpu(), predictions, average='weighted')
        
        if EvaluationType.RECALL in evaluation_types:
            metrics.recall = recall_score(y.cpu(), predictions, average='weighted')
        
        if EvaluationType.F1_SCORE in evaluation_types:
            metrics.f1_score = f1_score(y.cpu(), predictions, average='weighted')
        
        # ROC AUC
        if EvaluationType.ROC_AUC in evaluation_types and probabilities.shape[1] > 1:
            try:
                metrics.roc_auc = roc_auc_score(y.cpu(), probabilities[:, 1])
            except:
                metrics.roc_auc = 0.0
        
        # PR AUC
        if EvaluationType.PR_AUC in evaluation_types and probabilities.shape[1] > 1:
            try:
                metrics.pr_auc = average_precision_score(y.cpu(), probabilities[:, 1])
            except:
                metrics.pr_auc = 0.0
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y.cpu(), predictions).ravel()
        metrics.true_positives = int(tp)
        metrics.false_positives = int(fp)
        metrics.true_negatives = int(tn)
        metrics.false_negatives = int(fn)
        
        # 特异性和平衡准确率
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics.balanced_accuracy = (metrics.recall + metrics.specificity) / 2
        
        # Matthews相关系数
        metrics.matthews_corrcoef = self._calculate_matthews_corrcoef(
            tp, tn, fp, fn
        )
        
        # Cohen's Kappa
        metrics.cohen_kappa = self._calculate_cohen_kappa(y.cpu(), predictions)
        
        # 推理时间统计
        if EvaluationType.INFERENCE_TIME in evaluation_types:
            metrics.avg_inference_time = np.mean(inference_times)
            metrics.std_inference_time = np.std(inference_times)
            metrics.min_inference_time = np.min(inference_times)
            metrics.max_inference_time = np.max(inference_times)
            metrics.p95_inference_time = np.percentile(inference_times, 95)
            metrics.p99_inference_time = np.percentile(inference_times, 99)
        
        # 内存使用
        if EvaluationType.MEMORY_USAGE in evaluation_types:
            metrics.model_size_mb = self._calculate_model_size(model)
            metrics.peak_memory_mb = self._measure_peak_memory(model, X[:1])
            metrics.avg_memory_mb = self._measure_avg_memory(model, X[:10])
        
        return EvaluationResult(
            model_id=model_id,
            version_id=version_id,
            dataset_name=dataset_name,
            metrics=metrics,
            predictions=predictions.tolist() if predictions is not None else None,
            probabilities=probabilities.tolist() if probabilities is not None else None,
            labels=y.cpu().numpy().tolist(),
            evaluation_config={
                'evaluation_types': [et.value for et in evaluation_types],
                'batch_size': batch_size
            }
        )
    
    def _calculate_matthews_corrcoef(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """计算Matthews相关系数"""
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            return 0.0
        return (tp * tn - fp * fn) / denominator
    
    def _calculate_cohen_kappa(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算Cohen's Kappa"""
        return cohen_kappa_score(y_true, y_pred)
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _measure_peak_memory(self, model: nn.Module, input_data: torch.Tensor) -> float:
        """测量峰值内存使用"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(input_data)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            return peak_memory
        else:
            return 0.0
    
    def _measure_avg_memory(self, model: nn.Module, input_data: torch.Tensor) -> float:
        """测量平均内存使用"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            memory_usage = []
            for i in range(len(input_data)):
                with torch.no_grad():
                    _ = model(input_data[i:i+1])
                memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)
            
            return np.mean(memory_usage) if memory_usage else 0.0
        else:
            return 0.0
    
    def compare_models(self, model_results: Dict[str, Dict[str, EvaluationResult]]) -> pd.DataFrame:
        """比较模型性能"""
        comparison_data = []
        
        for model_id, dataset_results in model_results.items():
            for dataset_name, result in dataset_results.items():
                metrics = result.metrics
                comparison_data.append({
                    'Model': model_id,
                    'Dataset': dataset_name,
                    'Accuracy': metrics.accuracy,
                    'Precision': metrics.precision,
                    'Recall': metrics.recall,
                    'F1 Score': metrics.f1_score,
                    'ROC AUC': metrics.roc_auc,
                    'PR AUC': metrics.pr_auc,
                    'Inference Time (ms)': metrics.avg_inference_time * 1000,
                    'Model Size (MB)': metrics.model_size_mb
                })
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, output_path: str = None) -> str:
        """生成评估报告"""
        if output_path is None:
            output_path = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 创建报告
        report = f"# Model Evaluation Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 总体统计
        report += "## Summary\n\n"
        report += f"- Total models evaluated: {len(set(r.model_id for r in self.results))}\n"
        report += f"- Total datasets: {len(set(r.dataset_name for r in self.results))}\n"
        report += f"- Total evaluations: {len(self.results)}\n\n"
        
        # 按数据集分组的结果
        datasets = sorted(set(r.dataset_name for r in self.results))
        for dataset in datasets:
            report += f"## Dataset: {dataset}\n\n"
            
            dataset_results = [r for r in self.results if r.dataset_name == dataset]
            
            # 创建表格
            report += "| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Time (ms) |\n"
            report += "|-------|----------|-----------|--------|----|---------|-----------|\n"
            
            for result in sorted(dataset_results, key=lambda x: x.metrics.accuracy, reverse=True):
                metrics = result.metrics
                report += f"| {result.model_id} | {metrics.accuracy:.4f} | {metrics.precision:.4f} | "
                report += f"{metrics.recall:.4f} | {metrics.f1_score:.4f} | {metrics.roc_auc:.4f} | "
                report += f"{metrics.avg_inference_time*1000:.2f} |\n"
            
            report += "\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to: {output_path}")
        return output_path


class ModelEvaluator:
    """模型评估器主类"""
    
    def __init__(self, 
                 output_dir: str = "./evaluation_results",
                 device: str = "auto"):
        """
        初始化模型评估器
        
        Args:
            output_dir: 输出目录
            device: 计算设备
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设备设置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 评估历史
        self.evaluation_history: List[EvaluationResult] = []
        
        # 基准测试套件
        self.benchmark_suite: Optional[BenchmarkSuite] = None
        
        logger.info(f"ModelEvaluator initialized on device: {self.device}")
    
    def create_benchmark_suite(self, datasets: Dict[str, Any], 
                              evaluation_types: List[EvaluationType] = None) -> BenchmarkSuite:
        """创建基准测试套件"""
        self.benchmark_suite = BenchmarkSuite(datasets, evaluation_types)
        return self.benchmark_suite
    
    def evaluate_model(self, 
                      model: nn.Module,
                      model_id: str,
                      version_id: str,
                      test_data: Tuple[torch.Tensor, torch.Tensor],
                      evaluation_types: List[EvaluationType] = None) -> EvaluationResult:
        """
        评估单个模型
        
        Args:
            model: 要评估的模型
            model_id: 模型ID
            version_id: 版本ID
            test_data: 测试数据 (X, y)
            evaluation_types: 评估类型列表
            
        Returns:
            评估结果
        """
        if evaluation_types is None:
            evaluation_types = [
                EvaluationType.ACCURACY,
                EvaluationType.PRECISION,
                EvaluationType.RECALL,
                EvaluationType.F1_SCORE,
                EvaluationType.ROC_AUC,
                EvaluationType.INFERENCE_TIME,
                EvaluationType.MEMORY_USAGE
            ]
        
        # 移动模型到设备
        model = model.to(self.device)
        model.eval()
        
        # 使用基准测试套件或直接评估
        if self.benchmark_suite:
            # 创建临时数据集
            temp_dataset = {
                'temp': {
                    'loader': lambda: test_data,
                    'description': 'Temporary dataset'
                }
            }
            results = self.benchmark_suite.run_benchmark(
                model, model_id, version_id, ['temp']
            )
            result = results['temp']
        else:
            # 直接评估
            result = self._evaluate_model_direct(
                model, model_id, version_id, test_data, evaluation_types
            )
        
        # 保存结果
        self._save_evaluation_result(result)
        self.evaluation_history.append(result)
        
        logger.info(f"Model {model_id} evaluation completed")
        return result
    
    def _evaluate_model_direct(self, 
                             model: nn.Module,
                             model_id: str,
                             version_id: str,
                             test_data: Tuple[torch.Tensor, torch.Tensor],
                             evaluation_types: List[EvaluationType]) -> EvaluationResult:
        """直接评估模型"""
        # 创建临时基准测试套件
        temp_suite = BenchmarkSuite(
            datasets={'temp': {'loader': lambda: test_data}},
            evaluation_types=evaluation_types
        )
        
        results = temp_suite.run_benchmark(model, model_id, version_id, ['temp'])
        return results['temp']
    
    def evaluate_multiple_models(self,
                                 models: Dict[str, nn.Module],
                                 test_data: Tuple[torch.Tensor, torch.Tensor],
                                 parallel: bool = True) -> Dict[str, EvaluationResult]:
        """
        评估多个模型
        
        Args:
            models: 模型字典 {model_id: model}
            test_data: 测试数据
            parallel: 是否并行评估
            
        Returns:
            评估结果字典
        """
        results = {}
        
        if parallel and len(models) > 1:
            # 并行评估
            with ThreadPoolExecutor(max_workers=min(4, len(models))) as executor:
                futures = {}
                
                for model_id, model in models.items():
                    version_id = f"{model_id}_v1"
                    future = executor.submit(
                        self.evaluate_model,
                        model, model_id, version_id, test_data
                    )
                    futures[future] = model_id
                
                for future in tqdm(as_completed(futures), total=len(futures)):
                    model_id = futures[future]
                    try:
                        result = future.result()
                        results[model_id] = result
                    except Exception as e:
                        logger.error(f"Error evaluating model {model_id}: {e}")
                        results[model_id] = None
        else:
            # 串行评估
            for model_id, model in tqdm(models.items(), desc="Evaluating models"):
                version_id = f"{model_id}_v1"
                try:
                    result = self.evaluate_model(model, model_id, version_id, test_data)
                    results[model_id] = result
                except Exception as e:
                    logger.error(f"Error evaluating model {model_id}: {e}")
                    results[model_id] = None
        
        return results
    
    def cross_validate_model(self,
                            model: nn.Module,
                            model_id: str,
                            cv_data: List[Tuple[torch.Tensor, torch.Tensor]],
                            k_folds: int = 5) -> Dict[str, List[float]]:
        """
        交叉验证评估
        
        Args:
            model: 要评估的模型
            model_id: 模型ID
            cv_data: 交叉验证数据列表
            k_folds: 折数
            
        Returns:
            交叉验证结果
        """
        fold_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': []
        }
        
        for i, (train_data, val_data) in enumerate(cv_data):
            logger.info(f"Cross-validation fold {i+1}/{k_folds}")
            
            # 训练模型（这里简化，实际应该在每个fold上重新训练）
            # model.train_on(train_data)
            
            # 评估模型
            result = self.evaluate_model(
                model, f"{model_id}_fold{i}", f"{model_id}_v1", val_data
            )
            
            # 收集指标
            fold_metrics['accuracy'].append(result.metrics.accuracy)
            fold_metrics['precision'].append(result.metrics.precision)
            fold_metrics['recall'].append(result.metrics.recall)
            fold_metrics['f1_score'].append(result.metrics.f1_score)
            fold_metrics['roc_auc'].append(result.metrics.roc_auc)
        
        # 计算统计信息
        cv_stats = {}
        for metric, values in fold_metrics.items():
            cv_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # 保存交叉验证结果
        cv_path = self.output_dir / f"cv_results_{model_id}.json"
        with open(cv_path, 'w') as f:
            json.dump(cv_stats, f, indent=2)
        
        logger.info(f"Cross-validation results saved to: {cv_path}")
        return cv_stats
    
    def generate_comparison_report(self, 
                                 results: Dict[str, EvaluationResult],
                                 output_path: str = None) -> str:
        """生成模型对比报告"""
        if output_path is None:
            output_path = self.output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # 使用基准测试套件生成报告
        if self.benchmark_suite:
            # 转换结果格式
            model_results = {}
            for result in results.values():
                if result:
                    model_id = result.model_id
                    if model_id not in model_results:
                        model_results[model_id] = {}
                    model_results[model_id][result.dataset_name] = result
            
            if model_results:
                df = self.benchmark_suite.compare_models(model_results)
                
                # 生成Markdown报告
                report = f"# Model Comparison Report\n\n"
                report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # 性能表格
                report += "## Performance Comparison\n\n"
                report += df.to_markdown(index=False)
                
                # 最佳模型
                report += "\n\n## Top Performers\n\n"
                for metric in ['Accuracy', 'F1 Score', 'ROC AUC']:
                    best_model = df.loc[df[metric].idxmax(), 'Model']
                    best_score = df[metric].max()
                    report += f"- **{metric}**: {best_model} ({best_score:.4f})\n"
                
                # 保存报告
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                logger.info(f"Comparison report saved to: {output_path}")
                return str(output_path)
        
        return ""
    
    def visualize_results(self, results: Dict[str, EvaluationResult], 
                         output_dir: str = None):
        """可视化评估结果"""
        if output_dir is None:
            output_dir = self.output_dir / "visualizations"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 准备数据
        model_names = []
        accuracies = []
        f1_scores = []
        inference_times = []
        model_sizes = []
        
        for result in results.values():
            if result:
                model_names.append(result.model_id)
                accuracies.append(result.metrics.accuracy)
                f1_scores.append(result.metrics.f1_score)
                inference_times.append(result.metrics.avg_inference_time * 1000)
                model_sizes.append(result.metrics.model_size_mb)
        
        if not model_names:
            logger.warning("No valid results to visualize")
            return
        
        # 1. 准确率对比图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width/2, accuracies, width, label='Accuracy')
        plt.bar(x + width/2, f1_scores, width, label='F1 Score')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 推理时间对比图
        plt.subplot(1, 2, 2)
        plt.bar(model_names, inference_times)
        plt.xlabel('Models')
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Performance')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 性能vs大小散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(model_sizes, accuracies, s=100, alpha=0.7)
        
        for i, model in enumerate(model_names):
            plt.annotate(model, (model_sizes[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Model Size')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / "accuracy_vs_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {output_dir}")
    
    def _save_evaluation_result(self, result: EvaluationResult):
        """保存评估结果"""
        result_path = self.output_dir / f"evaluation_{result.model_id}_{result.version_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def load_evaluation_result(self, model_id: str, version_id: str) -> Optional[EvaluationResult]:
        """加载评估结果"""
        result_path = self.output_dir / f"evaluation_{model_id}_{version_id}.json"
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
                return EvaluationResult.from_dict(data)
        return None
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估摘要"""
        if not self.evaluation_history:
            return {}
        
        # 计算统计信息
        models = set(r.model_id for r in self.evaluation_history)
        datasets = set(r.dataset_name for r in self.evaluation_history)
        
        # 最佳模型
        best_accuracy = max(self.evaluation_history, key=lambda x: x.metrics.accuracy)
        best_f1 = max(self.evaluation_history, key=lambda x: x.metrics.f1_score)
        best_speed = min(self.evaluation_history, key=lambda x: x.metrics.avg_inference_time)
        
        return {
            'total_evaluations': len(self.evaluation_history),
            'unique_models': len(models),
            'unique_datasets': len(datasets),
            'best_accuracy': {
                'model_id': best_accuracy.model_id,
                'accuracy': best_accuracy.metrics.accuracy,
                'dataset': best_accuracy.dataset_name
            },
            'best_f1': {
                'model_id': best_f1.model_id,
                'f1_score': best_f1.metrics.f1_score,
                'dataset': best_f1.dataset_name
            },
            'fastest_model': {
                'model_id': best_speed.model_id,
                'inference_time_ms': best_speed.metrics.avg_inference_time * 1000,
                'dataset': best_speed.dataset_name
            }
        }
    
    def export_results(self, output_path: str, format: str = 'json'):
        """导出所有评估结果"""
        output_path = Path(output_path)
        
        if format.lower() == 'json':
            data = {
                'evaluation_history': [r.to_dict() for r in self.evaluation_history],
                'summary': self.get_evaluation_summary()
            }
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == 'csv':
            # 转换为DataFrame
            rows = []
            for result in self.evaluation_history:
                row = {
                    'model_id': result.model_id,
                    'version_id': result.version_id,
                    'dataset_name': result.dataset_name,
                    'evaluation_time': result.evaluation_time.isoformat(),
                    **result.metrics.to_dict()
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        logger.info(f"Results exported to: {output_path}")


def create_model_evaluator(output_dir: str = "./evaluation_results",
                          device: str = "auto") -> ModelEvaluator:
    """创建模型评估器实例"""
    return ModelEvaluator(output_dir, device)