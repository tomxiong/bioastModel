"""
FUA 参数管理和调优模块
支持参数历史追踪、智能调优建议和自动化搜索
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
import seaborn as sns


class ParameterHistoryManager:
    """参数历史管理器"""
    
    def __init__(self, storage_path: str = "fua/parameter_history"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_path / "parameter_history.json"
        self._load_history()
    
    def _load_history(self):
        """加载参数历史"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                "experiments": [],
                "best_configs": {},
                "performance_trends": {}
            }
    
    def _save_history(self):
        """保存参数历史"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_experiment(self, model_name: str, parameters: Dict, 
                         metrics: Dict, dataset_version: str = None):
        """记录实验结果"""
        experiment = {
            "model_name": model_name,
            "parameters": parameters,
            "metrics": metrics,
            "dataset_version": dataset_version or "unknown",
            "timestamp": datetime.now().isoformat(),
            "experiment_id": f"{model_name}_{int(datetime.now().timestamp())}"
        }
        
        self.history["experiments"].append(experiment)
        
        # 更新最佳配置
        self._update_best_config(model_name, experiment)
        
        # 更新性能趋势
        self._update_performance_trends(model_name, experiment)
        
        self._save_history()
    
    def _update_best_config(self, model_name: str, experiment: Dict):
        """更新最佳配置"""
        if model_name not in self.history["best_configs"]:
            self.history["best_configs"][model_name] = experiment
        else:
            current_best = self.history["best_configs"][model_name]
            # 假设使用accuracy作为主要指标
            if experiment["metrics"].get("accuracy", 0) > current_best["metrics"].get("accuracy", 0):
                self.history["best_configs"][model_name] = experiment
    
    def _update_performance_trends(self, model_name: str, experiment: Dict):
        """更新性能趋势"""
        if model_name not in self.history["performance_trends"]:
            self.history["performance_trends"][model_name] = []
        
        self.history["performance_trends"][model_name].append({
            "timestamp": experiment["timestamp"],
            "accuracy": experiment["metrics"].get("accuracy", 0),
            "loss": experiment["metrics"].get("loss", 0)
        })
    
    def get_parameter_history(self, model_name: str = None) -> List[Dict]:
        """获取参数历史"""
        if model_name:
            return [exp for exp in self.history["experiments"] 
                   if exp["model_name"] == model_name]
        return self.history["experiments"]
    
    def get_best_config(self, model_name: str) -> Dict:
        """获取最佳配置"""
        return self.history["best_configs"].get(model_name, {})
    
    def analyze_parameter_importance(self, model_name: str) -> Dict:
        """分析参数重要性"""
        experiments = self.get_parameter_history(model_name)
        if not experiments:
            return {}
        
        # 收集所有参数名
        all_params = set()
        for exp in experiments:
            all_params.update(exp["parameters"].keys())
        
        # 计算每个参数与性能的相关性
        importance = {}
        for param in all_params:
            param_values = []
            accuracies = []
            
            for exp in experiments:
                if param in exp["parameters"]:
                    param_values.append(exp["parameters"][param])
                    accuracies.append(exp["metrics"].get("accuracy", 0))
            
            if len(param_values) > 1:
                # 检查是否有足够的变化
                if len(set(param_values)) > 1 and len(set(accuracies)) > 1:
                    correlation = np.corrcoef(param_values, accuracies)[0, 1]
                else:
                    correlation = 0
                
                importance[param] = {
                    "correlation": correlation if not np.isnan(correlation) else 0,
                    "range": [min(param_values), max(param_values)],
                    "optimal_value": param_values[np.argmax(accuracies)]
                }
        
        return importance


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, model_name: str, history_manager: ParameterHistoryManager):
        self.model_name = model_name
        self.history_manager = history_manager
        self.search_space = self._define_search_space()
    
    def _define_search_space(self) -> Dict:
        """定义搜索空间"""
        return {
            "learning_rate": {
                "type": "float",
                "range": [0.0001, 0.1],
                "scale": "log"
            },
            "batch_size": {
                "type": "int",
                "range": [16, 128],
                "scale": "linear"
            },
            "epochs": {
                "type": "int",
                "range": [10, 100],
                "scale": "linear"
            },
            "optimizer": {
                "type": "categorical",
                "values": ["adam", "sgd", "rmsprop"]
            },
            "weight_decay": {
                "type": "float",
                "range": [0.00001, 0.001],
                "scale": "log"
            }
        }
    
    def suggest_parameters(self, strategy: str = "adaptive") -> Dict:
        """建议新参数"""
        if strategy == "adaptive":
            return self._adaptive_suggestion()
        elif strategy == "random":
            return self._random_suggestion()
        elif strategy == "grid":
            return self._grid_suggestion()
        else:
            return self._bayesian_suggestion()
    
    def _adaptive_suggestion(self) -> Dict:
        """自适应建议（基于历史数据）"""
        # 获取历史最佳配置
        best_config = self.history_manager.get_best_config(self.model_name)
        importance = self.history_manager.analyze_parameter_importance(self.model_name)
        
        suggestion = {}
        
        for param_name, param_info in self.search_space.items():
            if param_name in importance and importance[param_name]["correlation"] > 0.3:
                # 对重要参数进行微调
                optimal = importance[param_name]["optimal_value"]
                if param_info["type"] == "float":
                    if param_info["scale"] == "log":
                        # 对数空间微调
                        log_optimal = np.log10(optimal)
                        new_value = 10 ** (log_optimal + np.random.normal(0, 0.1))
                    else:
                        new_value = optimal * (1 + np.random.normal(0, 0.1))
                    suggestion[param_name] = np.clip(new_value, 
                                                   param_info["range"][0], 
                                                   param_info["range"][1])
                else:
                    # 整数参数
                    new_value = int(optimal + np.random.normal(0, 1))
                    suggestion[param_name] = np.clip(new_value, 
                                                   param_info["range"][0], 
                                                   param_info["range"][1])
            else:
                # 次要参数随机选择
                if param_info["type"] == "categorical":
                    suggestion[param_name] = np.random.choice(param_info["values"])
                else:
                    if param_info["scale"] == "log":
                        log_min = np.log10(param_info["range"][0])
                        log_max = np.log10(param_info["range"][1])
                        value = 10 ** np.random.uniform(log_min, log_max)
                    else:
                        value = np.random.uniform(param_info["range"][0], 
                                                   param_info["range"][1])
                    
                    if param_info["type"] == "int":
                        value = int(value)
                    
                    suggestion[param_name] = value
        
        return suggestion
    
    def _random_suggestion(self) -> Dict:
        """随机建议"""
        suggestion = {}
        
        for param_name, param_info in self.search_space.items():
            if param_info["type"] == "categorical":
                suggestion[param_name] = np.random.choice(param_info["values"])
            else:
                if param_info["scale"] == "log":
                    log_min = np.log10(param_info["range"][0])
                    log_max = np.log10(param_info["range"][1])
                    value = 10 ** np.random.uniform(log_min, log_max)
                else:
                    value = np.random.uniform(param_info["range"][0], 
                                               param_info["range"][1])
                
                if param_info["type"] == "int":
                    value = int(value)
                
                suggestion[param_name] = value
        
        return suggestion
    
    def _grid_suggestion(self) -> Dict:
        """网格搜索建议"""
        # 实现网格搜索的下一个参数点
        history = self.history_manager.get_parameter_history(self.model_name)
        used_params = [set(exp["parameters"].items()) for exp in history]
        
        # 生成网格点
        grid_points = self._generate_grid_points()
        
        # 找到未使用的网格点
        for point in grid_points:
            if set(point.items()) not in used_params:
                return point
        
        # 如果所有点都用过了，返回最佳配置
        return self.history_manager.get_best_config(self.model_name).get("parameters", {})
    
    def _bayesian_suggestion(self) -> Dict:
        """贝叶斯优化建议"""
        history = self.history_manager.get_parameter_history(self.model_name)
        
        if len(history) < 5:
            # 数据不足时使用随机搜索
            return self._random_suggestion()
        
        # 准备训练数据
        X = []
        y = []
        
        for exp in history:
            # 将参数转换为向量
            param_vector = self._param_to_vector(exp["parameters"])
            X.append(param_vector)
            y.append(exp["metrics"].get("accuracy", 0))
        
        X = np.array(X)
        y = np.array(y)
        
        # 训练高斯过程
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gp.fit(X, y)
        
        # 生成候选点并选择最优的
        candidates = [self._param_to_vector(self._random_suggestion()) 
                     for _ in range(100)]
        
        predictions, stds = gp.predict(candidates, return_std=True)
        
        # 使用UCB（Upper Confidence Bound）选择
        ucb = predictions + 1.96 * stds
        best_idx = np.argmax(ucb)
        
        return self._vector_to_param(candidates[best_idx])
    
    def _param_to_vector(self, params: Dict) -> List[float]:
        """将参数字典转换为向量"""
        vector = []
        
        # 按固定顺序处理参数
        param_order = ["learning_rate", "batch_size", "epochs", "weight_decay"]
        
        for param_name in param_order:
            if param_name in params:
                value = params[param_name]
                if param_name == "learning_rate" or param_name == "weight_decay":
                    vector.append(np.log10(value))
                else:
                    vector.append(float(value))
        
        # 处理分类变量
        if "optimizer" in params:
            optimizer_map = {"adam": 0, "sgd": 1, "rmsprop": 2}
            vector.append(optimizer_map.get(params["optimizer"], 0))
        
        return vector
    
    def _vector_to_param(self, vector: List[float]) -> Dict:
        """将向量转换为参数字典"""
        params = {}
        
        param_order = ["learning_rate", "batch_size", "epochs", "weight_decay"]
        
        for i, param_name in enumerate(param_order):
            if i < len(vector):
                if param_name == "learning_rate" or param_name == "weight_decay":
                    params[param_name] = 10 ** vector[i]
                else:
                    if param_name == "batch_size" or param_name == "epochs":
                        params[param_name] = int(vector[i])
                    else:
                        params[param_name] = vector[i]
        
        # 恢复优化器
        if len(vector) > len(param_order):
            optimizer_map = {0: "adam", 1: "sgd", 2: "rmsprop"}
            optimizer_idx = int(round(vector[len(param_order)]))
            params["optimizer"] = optimizer_map.get(optimizer_idx, "adam")
        
        return params
    
    def _generate_grid_points(self) -> List[Dict]:
        """生成网格搜索点"""
        # 简化的网格实现
        learning_rates = [0.001, 0.01, 0.1]
        batch_sizes = [32, 64, 128]
        epochs = [30, 50, 70]
        
        points = []
        for lr in learning_rates:
            for bs in batch_sizes:
                for ep in epochs:
                    points.append({
                        "learning_rate": lr,
                        "batch_size": bs,
                        "epochs": ep,
                        "optimizer": "adam",
                        "weight_decay": 0.0001
                    })
        
        return points


class ParameterVisualizer:
    """参数可视化工具"""
    
    def __init__(self, history_manager: ParameterHistoryManager):
        self.history_manager = history_manager
    
    def plot_parameter_effects(self, model_name: str, save_path: str = None):
        """绘制参数影响图"""
        history = self.history_manager.get_parameter_history(model_name)
        if not history:
            print("No history found for model:", model_name)
            return
        
        # 准备数据
        df = pd.DataFrame([
            {
                **exp["parameters"],
                "accuracy": exp["metrics"].get("accuracy", 0),
                "loss": exp["metrics"].get("loss", 0)
            }
            for exp in history
        ])
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Parameter Effects for {model_name}', fontsize=16)
        
        # 学习率 vs 准确率
        axes[0, 0].scatter(df["learning_rate"], df["accuracy"], alpha=0.6)
        axes[0, 0].set_xscale('log')
        axes[0, 0].set_xlabel('Learning Rate (log scale)')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Learning Rate vs Accuracy')
        
        # 批次大小 vs 准确率
        axes[0, 1].scatter(df["batch_size"], df["accuracy"], alpha=0.6)
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Batch Size vs Accuracy')
        
        # 训练轮数 vs 准确率
        axes[1, 0].scatter(df["epochs"], df["accuracy"], alpha=0.6)
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Epochs vs Accuracy')
        
        # 优化器对比
        optimizer_perf = df.groupby("optimizer")["accuracy"].agg(["mean", "std"])
        optimizer_perf.plot(kind="bar", y="mean", yerr="std", ax=axes[1, 1])
        axes[1, 1].set_xlabel('Optimizer')
        axes[1, 1].set_ylabel('Mean Accuracy')
        axes[1, 1].set_title('Optimizer Performance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_trend(self, model_name: str, save_path: str = None):
        """绘制性能趋势图"""
        trends = self.history_manager.history["performance_trends"].get(model_name, [])
        if not trends:
            print("No performance trends found for model:", model_name)
            return
        
        df = pd.DataFrame(trends)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 准确率趋势
        ax1.plot(df["timestamp"], df["accuracy"], marker='o', linewidth=2)
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Accuracy Trend for {model_name}')
        ax1.grid(True, alpha=0.3)
        
        # 损失趋势
        ax2.plot(df["timestamp"], df["loss"], marker='o', color='red', linewidth=2)
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Time')
        ax2.set_title(f'Loss Trend for {model_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建参数管理器
    history_manager = ParameterHistoryManager()
    
    # 记录实验
    experiment_params = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "optimizer": "adam",
        "weight_decay": 0.0001
    }
    experiment_metrics = {
        "accuracy": 0.95,
        "loss": 0.15,
        "val_accuracy": 0.92
    }
    
    history_manager.record_experiment(
        "resnet18", 
        experiment_params, 
        experiment_metrics,
        "v1.0"
    )
    
    # 创建优化器
    optimizer = ParameterOptimizer("resnet18", history_manager)
    
    # 获取参数建议
    suggestion = optimizer.suggest_parameters("adaptive")
    print("自适应参数建议:", suggestion)
    
    # 可视化
    visualizer = ParameterVisualizer(history_manager)
    # visualizer.plot_parameter_effects("resnet18")