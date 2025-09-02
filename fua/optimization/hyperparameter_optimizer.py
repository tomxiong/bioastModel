"""
FUA 超参数优化器

提供自动化的超参数搜索和优化功能
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import combinations
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Import FUA components
from ..core.model_adapters import ModelAdapter
from ..core.model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    best_trial: int
    total_trials: int
    study_summary: Dict[str, Any]
    trial_history: List[Dict[str, Any]]
    optimization_time: float
    pruned_trials: int
    failed_trials: int


@dataclass
class TrialResult:
    """单次试验结果"""
    trial_number: int
    params: Dict[str, Any]
    score: float
    accuracy: float
    loss: float
    training_time: float
    evaluation_time: float
    epoch: int
    status: str  # 'complete', 'pruned', 'failed'
    user_attrs: Dict[str, Any]


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self,
                 model_name: str,
                 search_space: Dict[str, Any],
                 direction: str = 'maximize',
                 sampler: str = 'tpe',
                 pruner: str = 'median',
                 n_trials: int = 100,
                 timeout: Optional[float] = None,
                 n_jobs: int = 1,
                 storage: Optional[str] = None,
                 study_name: Optional[str] = None,
                 cv_folds: int = 3,
                 early_stopping_patience: int = 10,
                 metric: str = 'accuracy'):
        """
        初始化超参数优化器
        
        Args:
            model_name: 模型名称
            search_space: 搜索空间定义
            direction: 优化方向 ('maximize' 或 'minimize')
            sampler: 采样器类型 ('tpe', 'random')
            pruner: 剪枝器类型 ('median', 'halving')
            n_trials: 试验次数
            timeout: 超时时间（秒）
            n_jobs: 并行任务数
            storage: Optuna存储URL
            study_name: 研究名称
            cv_folds: 交叉验证折数
            early_stopping_patience: 早停耐心值
            metric: 优化指标
        """
        self.model_name = model_name
        self.search_space = search_space
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.early_stopping_patience = early_stopping_patience
        self.metric = metric
        
        # 创建采样器
        if sampler == 'tpe':
            self.sampler = TPESampler(seed=42)
        else:
            self.sampler = RandomSampler(seed=42)
        
        # 创建剪枝器
        if pruner == 'halving':
            self.pruner = SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4
            )
        else:
            self.pruner = MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=10,
                interval_steps=1
            )
        
        # 创建Optuna研究
        self.study = optuna.create_study(
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=storage,
            study_name=study_name or f"{model_name}_optimization_{int(time.time())}",
            load_if_exists=True
        )
        
        # 结果存储
        self.trial_results = []
        self.best_model_state = None
        self.optimization_start_time = None
        
        # 统计信息
        self.stats = {
            'total_trials': 0,
            'completed_trials': 0,
            'pruned_trials': 0,
            'failed_trials': 0,
            'best_score': float('-inf') if direction == 'maximize' else float('inf')
        }
    
    def objective(self, trial: optuna.Trial,
                  train_data: Any,
                  val_data: Any,
                  model_factory: Callable,
                  train_fn: Callable,
                  eval_fn: Callable) -> float:
        """
        目标函数
        
        Args:
            trial: Optuna试验对象
            train_data: 训练数据
            val_data: 验证数据
            model_factory: 模型工厂函数
            train_fn: 训练函数
            eval_fn: 评估函数
        
        Returns:
            目标值
        """
        # 采样超参数
        params = {}
        for param_name, param_config in self.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        # 记录开始时间
        trial_start_time = time.time()
        
        try:
            # 创建模型
            model = model_factory(**params)
            
            # 训练模型
            training_start = time.time()
            model, history, early_stop_epoch = train_fn(
                model, train_data, val_data,
                params=params,
                trial=trial,
                patience=self.early_stopping_patience
            )
            training_time = time.time() - training_start
            
            # 评估模型
            eval_start = time.time()
            eval_results = eval_fn(model, val_data)
            eval_time = time.time() - eval_start
            
            # 获取目标值
            if self.metric in eval_results:
                score = eval_results[self.metric]
            else:
                raise ValueError(f"Metric '{self.metric}' not found in evaluation results")
            
            # 记录试验结果
            trial_result = TrialResult(
                trial_number=trial.number,
                params=params,
                score=score,
                accuracy=eval_results.get('accuracy', 0),
                loss=eval_results.get('loss', float('inf')),
                training_time=training_time,
                evaluation_time=eval_time,
                epoch=early_stop_epoch,
                status='complete',
                user_attrs=eval_results
            )
            self.trial_results.append(trial_result)
            
            # 更新统计
            self.stats['completed_trials'] += 1
            if score > self.stats['best_score']:
                self.stats['best_score'] = score
                self.best_model_state = model.state_dict()
            
            # 记录用户属性
            trial.set_user_attr('training_time', training_time)
            trial.set_user_attr('evaluation_time', eval_time)
            trial.set_user_attr('epoch', early_stop_epoch)
            trial.set_user_attr('params', params)
            
            for key, value in eval_results.items():
                trial.set_user_attr(key, value)
            
            # 报告中间值（用于剪枝）
            if hasattr(history, self.metric):
                for epoch, value in enumerate(getattr(history, self.metric)):
                    trial.report(value, epoch)
                    
                    # 检查是否应该剪枝
                    if trial.should_prune():
                        self.stats['pruned_trials'] += 1
                        trial_result.status = 'pruned'
                        raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            self.stats['failed_trials'] += 1
            logger.error(f"Trial {trial.number} failed: {e}")
            trial.set_user_attr('error', str(e))
            raise optuna.TrialPruned()  # 使用剪枝标记失败的试验
    
    def optimize(self,
                 train_data: Any,
                 val_data: Any,
                 model_factory: Callable,
                 train_fn: Callable,
                 eval_fn: Callable,
                 save_study: bool = True,
                 save_dir: Optional[str] = None) -> OptimizationResult:
        """
        执行优化
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            model_factory: 模型工厂函数
            train_fn: 训练函数
            eval_fn: 评估函数
            save_study: 是否保存研究
            save_dir: 保存目录
        
        Returns:
            优化结果
        """
        self.optimization_start_time = time.time()
        
        logger.info(f"Starting hyperparameter optimization for {self.model_name}")
        logger.info(f"Search space: {self.search_space}")
        logger.info(f"Number of trials: {self.n_trials}")
        
        # 执行优化
        self.study.optimize(
            lambda trial: self.objective(trial, train_data, val_data, model_factory, train_fn, eval_fn),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # 计算总时间
        optimization_time = time.time() - self.optimization_start_time
        self.stats['total_trials'] = len(self.trial_results)
        
        # 创建结果对象
        result = OptimizationResult(
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            best_trial=self.study.best_trial,
            total_trials=len(self.trial_results),
            study_summary=self._get_study_summary(),
            trial_history=[asdict(trial) for trial in self.trial_results],
            optimization_time=optimization_time,
            pruned_trials=self.stats['pruned_trials'],
            failed_trials=self.stats['failed_trials']
        )
        
        # 保存结果
        if save_study:
            self.save_results(result, save_dir)
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best score: {result.best_score:.4f}")
        logger.info(f"Best parameters: {result.best_params}")
        
        return result
    
    def _get_study_summary(self) -> Dict[str, Any]:
        """获取研究摘要"""
        return {
            'direction': self.study.direction.name,
            'best_trial': self.study.best_trial,
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'trials': len(self.study.trials),
            'datetime_start': self.study.datetime_start.isoformat() if self.study.datetime_start else None,
            'datetime_complete': self.study.datetime_complete.isoformat() if self.study.datetime_complete else None
        }
    
    def save_results(self, result: OptimizationResult, save_dir: Optional[str] = None):
        """保存优化结果"""
        if save_dir is None:
            save_dir = f"optimization_results/{self.model_name}_{int(time.time())}"
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存结果JSON
        with open(save_path / 'optimization_result.json', 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        # 保存试验历史
        df = pd.DataFrame([asdict(trial) for trial in self.trial_results])
        df.to_csv(save_path / 'trial_history.csv', index=False)
        
        # 保存Optuna研究
        if self.study.storage:
            self.study.storage.write_study(self.study)
        
        # 生成可视化
        self._generate_visualizations(save_path)
        
        logger.info(f"Results saved to {save_path}")
    
    def _generate_visualizations(self, save_path: Path):
        """生成可视化图表"""
        try:
            # 优化历史
            fig = plot_optimization_history(self.study)
            fig.write_image(save_path / 'optimization_history.png')
            plt.close(fig)
            
            # 参数重要性
            if len(self.study.trials) > 1:
                fig = plot_param_importances(self.study)
                fig.write_image(save_path / 'param_importances.png')
                plt.close(fig)
            
            # 参数关系图
            if len(self.study.best_params) > 1:
                for param1, param2 in combinations(list(self.study.best_params.keys()), 2):
                    fig = optuna.visualization.plot_contour(
                        self.study, params=[param1, param2]
                    )
                    fig.write_image(save_path / f'contour_{param1}_{param2}.png')
                    plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
    
    def get_best_model(self, model_factory: Callable) -> nn.Module:
        """获取最佳模型"""
        if self.best_model_state is None:
            raise ValueError("No optimization has been performed yet")
        
        model = model_factory(**self.study.best_params)
        model.load_state_dict(self.best_model_state)
        return model
    
    def get_trial_results_df(self) -> pd.DataFrame:
        """获取试验结果DataFrame"""
        return pd.DataFrame([asdict(trial) for trial in self.trial_results])
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析优化结果"""
        df = self.get_trial_results_df()
        
        analysis = {
            'basic_stats': {
                'total_trials': len(df),
                'completed_trials': len(df[df['status'] == 'complete']),
                'pruned_trials': len(df[df['status'] == 'pruned']),
                'failed_trials': len(df[df['status'] == 'failed']),
                'success_rate': len(df[df['status'] == 'complete']) / len(df) * 100
            },
            'performance_stats': {
                'best_score': df['score'].max(),
                'worst_score': df['score'].min(),
                'mean_score': df['score'].mean(),
                'std_score': df['score'].std(),
                'median_score': df['score'].median()
            },
            'parameter_importance': self._calculate_parameter_importance(),
            'convergence_analysis': self._analyze_convergence()
        }
        
        return analysis
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """计算参数重要性"""
        if len(self.study.trials) < 2:
            return {}
        
        try:
            importances = optuna.importance.get_param_importances(self.study)
            return {k: float(v) for k, v in importances.items()}
        except:
            return {}
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """分析收敛性"""
        df = self.get_trial_results_df()
        completed_trials = df[df['status'] == 'complete'].sort_values('trial_number')
        
        if len(completed_trials) < 2:
            return {}
        
        # 计算移动平均
        window_size = min(10, len(completed_trials) // 3)
        if window_size > 1:
            moving_avg = completed_trials['score'].rolling(window=window_size).mean()
            
            # 检查是否收敛
            last_scores = moving_avg[-window_size:]
            score_range = last_scores.max() - last_scores.min()
            
            return {
                'converged': score_range < 0.01,  # 阈值可根据需要调整
                'final_score_range': float(score_range),
                'moving_average_final': float(moving_avg.iloc[-1]),
                'improvement_rate': float((completed_trials['score'].iloc[-1] - completed_trials['score'].iloc[0]) / len(completed_trials))
            }
        
        return {}


class CrossValidationOptimizer(HyperparameterOptimizer):
    """交叉验证优化器"""
    
    def __init__(self, *args, cv_strategy: str = 'stratified', **kwargs):
        """
        初始化交叉验证优化器
        
        Args:
            cv_strategy: 交叉验证策略 ('stratified', 'kfold')
        """
        super().__init__(*args, **kwargs)
        self.cv_strategy = cv_strategy
    
    def optimize(self,
                 data: Any,
                 labels: Any,
                 model_factory: Callable,
                 train_fn: Callable,
                 eval_fn: Callable,
                 **kwargs) -> OptimizationResult:
        """执行交叉验证优化"""
        
        # 创建交叉验证
        if self.cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # 修改目标函数以支持交叉验证
        def cv_objective(trial: optuna.Trial) -> float:
            # 采样超参数
            params = {}
            for param_name, param_config in self.search_space.items():
                param_type = param_config['type']
                
                if param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # 执行交叉验证
            cv_scores = []
            cv_times = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(data, labels)):
                fold_start_time = time.time()
                
                try:
                    # 划分数据
                    fold_train_data = data[train_idx]
                    fold_val_data = data[val_idx]
                    
                    # 创建并训练模型
                    model = model_factory(**params)
                    model, history, _ = train_fn(
                        model, fold_train_data, fold_val_data,
                        params=params,
                        trial=trial,
                        patience=self.early_stopping_patience
                    )
                    
                    # 评估
                    eval_results = eval_fn(model, fold_val_data)
                    score = eval_results[self.metric]
                    cv_scores.append(score)
                    
                    fold_time = time.time() - fold_start_time
                    cv_times.append(fold_time)
                    
                    # 报告中间值
                    trial.report(score, fold)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                        
                except Exception as e:
                    logger.error(f"Fold {fold} failed: {e}")
                    continue
            
            if not cv_scores:
                raise optuna.TrialPruned()
            
            # 返回平均分数
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # 记录用户属性
            trial.set_user_attr('cv_scores', cv_scores)
            trial.set_user_attr('cv_mean', mean_score)
            trial.set_user_attr('cv_std', std_score)
            trial.set_user_attr('cv_times', cv_times)
            
            return mean_score
        
        # 执行优化
        self.optimization_start_time = time.time()
        
        logger.info(f"Starting cross-validation optimization for {self.model_name}")
        logger.info(f"CV strategy: {self.cv_strategy}, folds: {self.cv_folds}")
        
        self.study.optimize(cv_objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # 计算结果
        optimization_time = time.time() - self.optimization_start_time
        
        result = OptimizationResult(
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            best_trial=self.study.best_trial,
            total_trials=len(self.study.trials),
            study_summary=self._get_study_summary(),
            trial_history=[],  # 需要从试验中提取
            optimization_time=optimization_time,
            pruned_trials=0,  # 需要统计
            failed_trials=0
        )
        
        return result


# 工厂函数
def create_hyperparameter_optimizer(
    model_name: str,
    search_space: Dict[str, Any],
    **kwargs
) -> HyperparameterOptimizer:
    """创建超参数优化器"""
    return HyperparameterOptimizer(model_name, search_space, **kwargs)


def create_cv_optimizer(
    model_name: str,
    search_space: Dict[str, Any],
    cv_strategy: str = 'stratified',
    **kwargs
) -> CrossValidationOptimizer:
    """创建交叉验证优化器"""
    return CrossValidationOptimizer(
        model_name, search_space, cv_strategy=cv_strategy, **kwargs
    )


# 预定义的搜索空间
def get_default_search_space(model_type: str) -> Dict[str, Any]:
    """获取默认搜索空间"""
    search_spaces = {
        'resnet': {
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'adamw']},
            'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-2, 'log': True},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5}
        },
        'efficientnet': {
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw']},
            'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.4},
            'augmentation': {'type': 'categorical', 'choices': ['light', 'medium', 'heavy']}
        },
        'mobilenet': {
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [32, 64, 128]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']},
            'weight_decay': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'width_multiplier': {'type': 'categorical', 'choices': [0.75, 1.0, 1.25]}
        },
        'vit': {
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32]},
            'optimizer': {'type': 'categorical', 'choices': ['adamw']},
            'weight_decay': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.3},
            'attention_dropout': {'type': 'float', 'low': 0.0, 'high': 0.2}
        }
    }
    
    return search_spaces.get(model_type, search_spaces['resnet'])