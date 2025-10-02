#!/usr/bin/env python3
"""
FUA 训练调优流水线
整合分布式监控与MobileNetV3模型训练
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple, Optional
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # 导入FUA组件
    from fua.production import (
        create_distributed_monitor, NodeRole, NodeStatus,
        ClusterNode, MonitoringTask
    )
    from models.mobilenet_v3 import create_mobilenetv3_large
    from core.config.model_configs import get_model_config
    from core.config.training_configs import get_model_specific_config
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("请确保已安装所有必要的依赖")
    sys.exit(1)


class TrainingOptimizer:
    """训练优化器 - 整合分布式监控"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.distributed_monitor = None
        self.training_history = []
        self.best_accuracy = 0.0
        self.best_config = None
        self.current_epoch = 0
        self.total_epochs = self.config.get('epochs', 50)
        self.optimization_results = []
        
        # 优化配置
        self.search_space = self.config.get('search_space', {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'optimizer': ['adam', 'sgd', 'rmsprop'],
            'weight_decay': [0.0, 0.0001, 0.001]
        })
        
        # 训练数据
        self.train_data, self.train_labels = self._generate_synthetic_data(2000)
        self.val_data, self.val_labels = self._generate_synthetic_data(400)
        
        # 任务队列
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def _generate_synthetic_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成合成数据"""
        # 生成随机图像数据
        data = torch.randn(num_samples, 3, 70, 70)
        # 生成随机标签（0或1）
        labels = torch.randint(0, 2, (num_samples,))
        return data, labels
    
    def _create_dataloader(self, data: torch.Tensor, labels: torch.Tensor, 
                          batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def setup_model_and_monitor(self, model_config: Dict[str, Any] = None):
        """设置模型和监控器"""
        logger.info("Setting up MobileNetV3 model and distributed monitor...")
        
        # 1. 创建MobileNetV3模型
        self.model = create_mobilenetv3_large(num_classes=2)
        
        # 2. 设置分布式监控器
        monitor_config = {
            "node_role": "coordinator",
            "region": "training-region",
            "consul_enabled": False,
            "redis_enabled": False,
            "websocket_port": 8769,
            "api_port": 8084
        }
        
        self.distributed_monitor = create_distributed_monitor(
            node_id="training_optimizer",
            config=monitor_config
        )
        
        # 3. 添加模型到监控
        model_config = model_config or {
            "input_size": (3, 70, 70),
            "num_classes": 2,
            "model_type": "MobileNetV3-Large"
        }
        
        self.distributed_monitor.add_model(
            model_id="mobilenetv3_optimized",
            version_id="v1.0",
            model=self.model,
            config=model_config
        )
        
        # 4. 设置训练告警规则
        self._setup_optimization_alert_rules()
        
        # 5. 模拟训练集群
        self._simulate_training_cluster()
        
        logger.info(f"Model and monitor setup complete. Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def _setup_optimization_alert_rules(self):
        """设置优化相关的告警规则"""
        alert_manager = self.distributed_monitor.alert_manager
        
        # 训练准确率告警
        alert_manager.add_alert_rule("low_accuracy", {
            "condition": {"training.accuracy": 0.6},
            "operator": "<",
            "message": "Training accuracy below threshold"
        })
        
        # 验证损失告警
        alert_manager.add_alert_rule("high_val_loss", {
            "condition": {"validation.loss": 1.5},
            "operator": ">",
            "message": "Validation loss too high"
        })
        
        # 过拟合告警
        alert_manager.add_alert_rule("overfitting", {
            "condition": {"overfitting_ratio": 0.15},
            "operator": ">",
            "message": "Potential overfitting detected"
        })
        
        # 资源使用告警
        alert_manager.add_alert_rule("high_resource_usage", {
            "condition": {"system.cpu_percent": 90.0},
            "operator": ">",
            "message": "High CPU usage detected"
        })
        
        logger.info("Optimization alert rules configured")
    
    def _simulate_training_cluster(self):
        """模拟训练集群"""
        # 模拟多个worker节点
        worker_nodes = [
            ClusterNode(
                id=f"worker_{i}",
                host=f"10.0.0.{i+10}",
                port=9000 + i,
                role=NodeRole.MONITOR,
                status=NodeStatus.ACTIVE,
                region="training-region",
                zone=f"zone-{i%2}",
                load=np.random.uniform(0.3, 0.8)
            )
            for i in range(6)
        ]
        
        # 添加聚合节点
        aggregator_nodes = [
            ClusterNode(
                id=f"aggregator_{i}",
                host=f"10.0.0.{i+20}",
                port=9010 + i,
                role=NodeRole.AGGREGATOR,
                status=NodeStatus.ACTIVE,
                region="training-region",
                zone=f"zone-{i%2}",
                load=0.5
            )
            for i in range(2)
        ]
        
        # 添加到集群管理器
        cluster_manager = self.distributed_monitor.cluster_manager
        for node in worker_nodes + aggregator_nodes:
            cluster_manager.nodes[node.id] = node
        
        logger.info(f"Simulated training cluster with {len(worker_nodes)} workers and {len(aggregator_nodes)} aggregators")
    
    def run_hyperparameter_optimization(self, n_trials: int = 9):
        """运行超参数优化"""
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        # 启动监控
        self.distributed_monitor.start()
        
        # 生成参数组合
        param_combinations = self._generate_param_combinations(n_trials)
        
        # 并行执行优化
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, params in enumerate(param_combinations):
                future = executor.submit(self._run_single_trial, i, params)
                futures.append(future)
                
                # 限制并发数
                if len(futures) >= 3:
                    completed, _ = as_completed(futures, timeout=3600).__next__()
                    futures.remove(completed)
            
            # 等待所有任务完成
            for future in as_completed(futures, timeout=3600):
                try:
                    result = future.result()
                    self.optimization_results.append(result)
                    logger.info(f"Trial {result['trial_id']} completed: Acc={result['best_val_acc']:.4f}")
                except Exception as e:
                    logger.error(f"Trial failed: {e}")
        
        # 停止监控
        self.distributed_monitor.stop()
        
        # 分析结果
        self._analyze_optimization_results()
        
        return self.optimization_results
    
    def _generate_param_combinations(self, n_trials: int) -> List[Dict[str, Any]]:
        """生成参数组合"""
        import itertools
        
        # 生成所有可能的组合
        all_combinations = list(itertools.product(
            self.search_space['learning_rate'],
            self.search_space['batch_size'],
            self.search_space['optimizer'],
            self.search_space['weight_decay']
        ))
        
        # 随机选择n_trials个组合
        selected_indices = np.random.choice(len(all_combinations), min(n_trials, len(all_combinations)), replace=False)
        
        combinations = []
        for idx in selected_indices:
            lr, bs, opt, wd = all_combinations[idx]
            combinations.append({
                'learning_rate': lr,
                'batch_size': bs,
                'optimizer': opt,
                'weight_decay': wd
            })
        
        return combinations
    
    def _run_single_trial(self, trial_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个优化试验"""
        logger.info(f"Running trial {trial_id} with params: {params}")
        
        # 创建新的模型实例
        model = create_mobilenetv3_large(num_classes=2)
        
        # 设置数据加载器
        train_loader = self._create_dataloader(
            self.train_data, self.train_labels, 
            batch_size=params['batch_size']
        )
        val_loader = self._create_dataloader(
            self.val_data, self.val_labels, 
            batch_size=params['batch_size'], shuffle=False
        )
        
        # 设置优化器
        optimizer = self._get_optimizer(model, params)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        # 训练
        best_val_acc = 0.0
        trial_history = []
        
        for epoch in range(30):  # 每个试验30个epoch
            # 训练
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            # 学习率调整
            scheduler.step()
            
            # 记录
            trial_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # 更新最佳准确率
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # 收集监控指标
            self._collect_trial_metrics(trial_id, params, train_loss, train_acc, val_loss, val_acc)
        
        return {
            'trial_id': trial_id,
            'params': params,
            'best_val_acc': best_val_acc,
            'history': trial_history
        }
    
    def _get_optimizer(self, model: nn.Module, params: Dict[str, Any]) -> optim.Optimizer:
        """获取优化器"""
        if params['optimizer'] == 'adam':
            return optim.Adam(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == 'rmsprop':
            return optim.RMSprop(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:
            return optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float]:
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def _collect_trial_metrics(self, trial_id: int, params: Dict[str, Any], 
                              train_loss: float, train_acc: float, 
                              val_loss: float, val_acc: float):
        """收集试验指标"""
        try:
            # 计算过拟合比例
            overfitting_ratio = abs(train_acc - val_acc) / max(train_acc, val_acc)
            
            # 收集分布式指标
            metrics = self.distributed_monitor.collect_distributed_metrics(
                model_id="mobilenetv3_optimized",
                version_id="v1.0"
            )
            
            # 添加试验特定指标
            if hasattr(self.distributed_monitor.local_monitor, 'metrics_collector'):
                custom_metrics = {
                    f"trial_{trial_id}_train_loss": train_loss,
                    f"trial_{trial_id}_train_acc": train_acc,
                    f"trial_{trial_id}_val_loss": val_loss,
                    f"trial_{trial_id}_val_acc": val_acc,
                    f"trial_{trial_id}_lr": params['learning_rate'],
                    f"trial_{trial_id}_batch_size": params['batch_size'],
                    f"trial_{trial_id}_optimizer": params['optimizer'],
                    f"trial_{trial_id}_overfitting": overfitting_ratio
                }
                
                # 检查告警
                self.distributed_monitor.alert_manager.check_alerts(metrics)
                
        except Exception as e:
            logger.error(f"Error collecting trial metrics: {e}")
    
    def _analyze_optimization_results(self):
        """分析优化结果"""
        if not self.optimization_results:
            logger.warning("No optimization results to analyze")
            return
        
        # 找出最佳配置
        best_result = max(self.optimization_results, key=lambda x: x['best_val_acc'])
        self.best_config = best_result['params']
        self.best_accuracy = best_result['best_val_acc']
        
        logger.info(f"Best configuration found:")
        logger.info(f"  Validation Accuracy: {self.best_accuracy:.4f}")
        logger.info(f"  Learning Rate: {self.best_config['learning_rate']}")
        logger.info(f"  Batch Size: {self.best_config['batch_size']}")
        logger.info(f"  Optimizer: {self.best_config['optimizer']}")
        logger.info(f"  Weight Decay: {self.best_config['weight_decay']}")
        
        # 生成优化报告
        self._generate_optimization_report()
    
    def run_final_training(self):
        """使用最佳配置进行最终训练"""
        if not self.best_config:
            logger.error("No best configuration found. Run optimization first.")
            return
        
        logger.info("Running final training with best configuration...")
        
        # 创建新模型
        self.model = create_mobilenetv3_large(num_classes=2)
        
        # 使用最佳配置
        train_loader = self._create_dataloader(
            self.train_data, self.train_labels,
            batch_size=self.best_config['batch_size']
        )
        val_loader = self._create_dataloader(
            self.val_data, self.val_labels,
            batch_size=self.best_config['batch_size'], shuffle=False
        )
        
        # 设置优化器
        optimizer = self._get_optimizer(self.model, self.best_config)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        # 训练
        best_val_acc = 0.0
        
        for epoch in range(self.total_epochs):
            self.current_epoch = epoch + 1
            
            # 训练和验证
            train_loss, train_acc = self._train_epoch(self.model, train_loader, optimizer, criterion)
            val_loss, val_acc = self._validate_epoch(self.model, val_loader, criterion)
            scheduler.step()
            
            # 记录历史
            self.training_history.append({
                'epoch': self.current_epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
            
            # 打印进度
            if epoch % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.total_epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        logger.info(f"Final training completed. Best validation accuracy: {best_val_acc:.4f}")
    
    def _save_checkpoint(self, epoch: int, accuracy: float):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'config': self.best_config,
            'training_history': self.training_history
        }
        
        checkpoint_path = f"checkpoints/mobilenetv3_optimized_epoch_{epoch}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _generate_optimization_report(self):
        """生成优化报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"mobilenetv3_optimization_report_{timestamp}.md"
        
        # 生成集群报告
        cluster_report_path = self.distributed_monitor.generate_cluster_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MobileNetV3 Hyperparameter Optimization Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Optimization Summary\n\n")
            f.write(f"- **Total Trials:** {len(self.optimization_results)}\n")
            f.write(f"- **Best Validation Accuracy:** {self.best_accuracy:.4f}\n")
            f.write(f"- **Search Space:** {len(self.search_space)} parameters\n\n")
            
            f.write("## Best Configuration\n\n")
            if self.best_config:
                f.write("```python\n")
                f.write("best_config = {\n")
                for key, value in self.best_config.items():
                    f.write(f"    '{key}': {repr(value)},\n")
                f.write("}\n")
                f.write("```\n\n")
            
            f.write("## All Trials Results\n\n")
            f.write("| Trial ID | Learning Rate | Batch Size | Optimizer | Weight Decay | Best Val Acc |\n")
            f.write("|----------|---------------|------------|-----------|--------------|---------------|\n")
            
            for result in sorted(self.optimization_results, key=lambda x: x['best_val_acc'], reverse=True):
                f.write(f"| {result['trial_id']} | {result['params']['learning_rate']} | ")
                f.write(f"{result['params']['batch_size']} | {result['params']['optimizer']} | ")
                f.write(f"{result['params']['weight_decay']} | {result['best_val_acc']:.4f} |\n")
            
            f.write("\n## Optimization Insights\n\n")
            f.write("### Parameter Impact\n")
            
            # 分析参数影响
            lr_results = {}
            for result in self.optimization_results:
                lr = result['params']['learning_rate']
                if lr not in lr_results:
                    lr_results[lr] = []
                lr_results[lr].append(result['best_val_acc'])
            
            f.write("\n#### Learning Rate Analysis\n")
            for lr, accs in lr_results.items():
                avg_acc = np.mean(accs)
                f.write(f"- LR={lr}: Average accuracy = {avg_acc:.4f}\n")
            
            f.write("\n## Distributed Monitoring Integration\n\n")
            f.write("The optimization process was monitored using FUA's distributed monitoring system:\n\n")
            f.write("- **Real-time metrics collection** from all training trials\n")
            f.write("- **Automated alerting** for training anomalies\n")
            f.write("- **Resource utilization monitoring** across the cluster\n")
            f.write("- **Health monitoring** of all worker nodes\n\n")
            
            f.write(f"## Cluster Report\n\n")
            f.write(f"A detailed cluster monitoring report was generated at: `{cluster_report_path}`\n\n")
            
            f.write("## Key Achievements\n\n")
            f.write("1. **Automated Hyperparameter Optimization**: Systematic search for optimal parameters\n")
            f.write("2. **Distributed Monitoring**: Real-time monitoring of all training processes\n")
            f.write("3. **Resource Efficiency**: Parallel execution of optimization trials\n")
            f.write("4. **Comprehensive Tracking**: Detailed metrics and history for all trials\n")
            f.write("5. **Scalable Architecture**: Support for large-scale optimization tasks\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("- Use the best configuration for production deployment\n")
            f.write("- Consider expanding the search space for further improvements\n")
            f.write("- Implement early stopping for faster optimization\n")
            f.write("- Add more sophisticated optimization algorithms (e.g., Bayesian optimization)\n")
            f.write("- Integrate with model versioning system\n")
        
        logger.info(f"Optimization report generated: {report_path}")
        return report_path
    
    def run_complete_optimization_pipeline(self):
        """运行完整的优化流水线"""
        logger.info("Running complete MobileNetV3 optimization pipeline...")
        
        # 1. 设置模型和监控器
        self.setup_model_and_monitor()
        
        # 2. 运行超参数优化
        optimization_results = self.run_hyperparameter_optimization(n_trials=9)
        
        # 3. 运行最终训练
        if self.best_config:
            self.run_final_training()
        
        logger.info("Optimization pipeline completed successfully!")
        
        return {
            'optimization_results': optimization_results,
            'best_config': self.best_config,
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }


def main():
    """主函数"""
    print("=" * 80)
    print("MobileNetV3 Training Optimization with Distributed Monitoring")
    print("FUA System - Complete Pipeline Demo")
    print("=" * 80)
    
    # 创建优化器实例
    optimizer = TrainingOptimizer({
        'epochs': 50,
        'search_space': {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'optimizer': ['adam', 'sgd'],
            'weight_decay': [0.0, 0.0001]
        }
    })
    
    try:
        # 运行完整流水线
        results = optimizer.run_complete_optimization_pipeline()
        
        print("\n" + "=" * 80)
        print("Optimization pipeline completed successfully!")
        print("=" * 80)
        print(f"\nBest configuration achieved {results['best_accuracy']:.4f} validation accuracy")
        print("\nCheck the generated reports for detailed results.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nPipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)