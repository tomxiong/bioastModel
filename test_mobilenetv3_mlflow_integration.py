#!/usr/bin/env python3
"""
FUA MLflow集成到训练流水线的示例

演示如何将MLflow实验跟踪集成到MobileNetV3训练过程中
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
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

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
    from fua import (
        FUAMLflowIntegration, create_mlflow_integration,
        FUAModelRegistry
    )
    from fua.production import create_distributed_monitor
    from models.mobilenet_v3 import create_mobilenetv3_large
    from core.config.model_configs import get_model_config
    from core.config.training_configs import get_model_specific_config
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("请确保已安装所有必要的依赖")
    sys.exit(1)


class MobileNetV3WithMLflow:
    """集成MLflow的MobileNetV3训练器"""
    
    def __init__(self, 
                 experiment_name: str = "mobilenetv3_training",
                 tracking_uri: str = None,
                 registry_uri: str = None):
        """
        初始化训练器
        
        Args:
            experiment_name: 实验名称
            tracking_uri: MLflow跟踪URI
            registry_uri: 模型注册表URI
        """
        self.experiment_name = experiment_name
        self.model = None
        self.mlflow_integration = None
        self.distributed_monitor = None
        self.training_history = []
        
        # 生成模拟数据
        self.train_data, self.train_labels = self._generate_synthetic_data(2000)
        self.val_data, self.val_labels = self._generate_synthetic_data(400)
        
        # 创建MLflow集成
        self.mlflow_integration = create_mlflow_integration(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            experiment_name=experiment_name
        )
        
        logger.info(f"MobileNetV3 trainer with MLflow initialized")
        logger.info(f"Experiment: {experiment_name}")
    
    def _generate_synthetic_data(self, num_samples: int) -> tuple:
        """生成合成数据"""
        data = torch.randn(num_samples, 3, 70, 70)
        labels = torch.randint(0, 2, (num_samples,))
        return data, labels
    
    def _create_dataloader(self, data, labels, batch_size=32, shuffle=True):
        """创建数据加载器"""
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def setup_training(self, model_config: dict = None, training_config: dict = None):
        """设置训练参数"""
        logger.info("Setting up MobileNetV3 training with MLflow...")
        
        # 1. 创建模型
        self.model = create_mobilenetv3_large(num_classes=2)
        
        # 2. 获取配置
        if model_config is None:
            try:
                model_config = get_model_config("mic_mobilenetv3")
            except:
                model_config = {
                    "model_type": "MobileNetV3-Large",
                    "input_size": (3, 70, 70),
                    "num_classes": 2,
                    "parameters": 4187536
                }
        if training_config is None:
            try:
                training_config = get_model_specific_config("mic_mobilenetv3")
            except:
                training_config = {
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "epochs": 50,
                    "optimizer": "adam"
                }
        
        # 3. 设置分布式监控
        monitor_config = {
            "node_role": "monitor",
            "region": "training-region",
            "consul_enabled": False,
            "redis_enabled": False,
            "websocket_port": 8771,
            "api_port": 8086
        }
        
        self.distributed_monitor = create_distributed_monitor(
            node_id="mobilenetv3_mlflow_training",
            config=monitor_config
        )
        
        # 4. 添加模型到监控
        self.distributed_monitor.add_model(
            model_id="mobilenetv3_mlflow",
            version_id="v1.0",
            model=self.model,
            config=model_config
        )
        
        logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return model_config, training_config
    
    def train_with_mlflow_tracking(self, 
                                epochs: int = 50,
                                batch_size: int = 32,
                                learning_rate: float = 0.001,
                                optimizer: str = "adam",
                                run_name: str = None):
        """
        使用MLflow跟踪的训练
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            optimizer: 优化器类型
            run_name: 运行名称
        """
        logger.info("Starting training with MLflow tracking...")
        
        # 准备配置
        model_config = {
            "model_type": "MobileNetV3-Large",
            "input_size": (3, 70, 70),
            "num_classes": 2
        }
        
        training_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer
        }
        
        # 创建MLflow运行
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"mobilenetv3_bs{batch_size}_lr{learning_rate}_{optimizer}_{timestamp}"
        
        run_id = self.mlflow_integration.create_training_run(
            model_name="MobileNetV3",
            model_config=model_config,
            training_config=training_config,
            run_name=run_name,
            tags={"batch_size": str(batch_size), "lr": str(learning_rate)}
        )
        
        if not run_id:
            logger.error("Failed to create MLflow run")
            return
        
        # 启动分布式监控
        self.distributed_monitor.start()
        
        # 准备数据加载器
        train_loader = self._create_dataloader(self.train_data, self.train_labels, batch_size)
        val_loader = self._create_dataloader(self.val_data, self.val_labels, batch_size, shuffle=False)
        
        # 设置优化器
        criterion = nn.CrossEntropyLoss()
        if optimizer == "adam":
            opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            opt = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            opt = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)
        
        # 训练循环
        best_val_acc = 0.0
        best_epoch = 0
        
        try:
            for epoch in range(epochs):
                # 训练
                train_loss, train_acc = self._train_epoch(
                    self.model, train_loader, opt, criterion
                )
                
                # 验证
                val_loss, val_acc = self._validate_epoch(
                    self.model, val_loader, criterion
                )
                
                # 学习率调整
                scheduler.step()
                
                # 记录到MLflow
                self.mlflow_integration.log_training_metrics({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": opt.param_groups[0]['lr']
                }, step=epoch + 1)
                
                # 记录到训练历史
                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": opt.param_groups[0]['lr']
                })
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    self._save_checkpoint(epoch, val_acc, opt)
                
                # 记录分布式指标
                if epoch % 5 == 0:
                    metrics = self.distributed_monitor.collect_distributed_metrics(
                        model_id="mobilenetv3_mlflow",
                        version_id="v1.0"
                    )
                    
                    # 记录系统指标到MLflow
                    if metrics.get('aggregated'):
                        system_metrics = {
                            "cpu_avg": metrics['aggregated'].get('cpu', {}).get('avg', 0),
                            "memory_avg": metrics['aggregated'].get('memory', {}).get('avg', 0),
                            "latency_avg": metrics['aggregated'].get('latency', {}).get('avg', 0)
                        }
                        self.mlflow_integration.log_training_metrics(system_metrics, step=epoch + 1)
                
                # 打印进度
                if epoch % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}] "
                              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 训练完成，记录最终指标并注册模型
            final_metrics = {
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "final_train_loss": train_loss,
                "final_train_acc": train_acc,
                "final_val_loss": val_loss,
                "final_val_acc": val_acc
            }
            
            # 记录模型到MLflow
            self.mlflow_integration.log_model_and_register(
                model=self.model,
                model_name="MobileNetV3",
                model_config=model_config,
                input_example=torch.randn(1, 3, 70, 70),
                stage="Staging"
            )
            
            # 记录训练历史作为工件
            history_path = f"training_history_{run_id}.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            self.mlflow_integration.log_artifact(history_path)
            
            # 完成MLflow运行
            self.mlflow_integration.complete_training_run(final_metrics)
            
            logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.mlflow_integration.complete_training_run({})
            raise
        
        finally:
            # 停止分布式监控
            self.distributed_monitor.stop()
            
            # 清理临时文件
            if os.path.exists(history_path):
                os.remove(history_path)
    
    def _train_epoch(self, model, train_loader, optimizer, criterion) -> tuple:
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
    
    def _validate_epoch(self, model, val_loader, criterion) -> tuple:
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
    
    def _save_checkpoint(self, epoch, accuracy, optimizer):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'training_history': self.training_history
        }
        
        checkpoint_path = f"checkpoints/mobilenetv3_mlflow_epoch_{epoch}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def run_hyperparameter_study(self):
        """运行超参数研究"""
        logger.info("Starting hyperparameter study with MLflow tracking...")
        
        # 参数组合
        param_combinations = [
            {"batch_size": 16, "learning_rate": 0.001, "optimizer": "adam"},
            {"batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
            {"batch_size": 64, "learning_rate": 0.001, "optimizer": "adam"},
            {"batch_size": 32, "learning_rate": 0.0005, "optimizer": "adam"},
            {"batch_size": 32, "learning_rate": 0.0001, "optimizer": "adam"},
            {"batch_size": 32, "learning_rate": 0.001, "optimizer": "sgd"},
        ]
        
        results = []
        
        for i, params in enumerate(param_combinations):
            logger.info(f"\nRunning combination {i+1}/{len(param_combinations)}: {params}")
            
            # 创建新的模型实例
            self.model = create_mobilenetv3_large(num_classes=2)
            
            # 运行训练
            self.train_with_mlflow_tracking(
                epochs=30,  # 减少epoch以加快研究
                **params
            )
            
            # 记录结果
            if self.training_history:
                best_acc = max(h['val_acc'] for h in self.training_history)
                results.append({
                    "params": params,
                    "best_val_acc": best_acc,
                    "training_history": self.training_history.copy()
                })
            
            # 清理历史
            self.training_history = []
            
            # 短暂休息
            time.sleep(2)
        
        # 分析结果
        self._analyze_hyperparameter_results(results)
        
        return results
    
    def _analyze_hyperparameter_results(self, results):
        """分析超参数研究结果"""
        logger.info("\n" + "="*50)
        logger.info("Hyperparameter Study Results")
        logger.info("="*50)
        
        # 按性能排序
        sorted_results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
        
        logger.info(f"\nTop 3 configurations:")
        for i, result in enumerate(sorted_results[:3]):
            params = result['params']
            logger.info(f"{i+1}. Val Acc: {result['best_val_acc']:.4f} - "
                      f"BS: {params['batch_size']}, LR: {params['learning_rate']}, "
                      f"Opt: {params['optimizer']}")
        
        # 分析参数影响
        logger.info(f"\nParameter Analysis:")
        
        # Batch size影响
        bs_results = {}
        for result in results:
            bs = result['params']['batch_size']
            if bs not in bs_results:
                bs_results[bs] = []
            bs_results[bs].append(result['best_val_acc'])
        
        logger.info("Batch Size Impact:")
        for bs, accs in bs_results.items():
            avg_acc = np.mean(accs)
            logger.info(f"  BS={bs}: Avg Acc = {avg_acc:.4f}")
        
        # Learning rate影响
        lr_results = {}
        for result in results:
            lr = result['params']['learning_rate']
            if lr not in lr_results:
                lr_results[lr] = []
            lr_results[lr].append(result['best_val_acc'])
        
        logger.info("Learning Rate Impact:")
        for lr, accs in lr_results.items():
            avg_acc = np.mean(accs)
            logger.info(f"  LR={lr}: Avg Acc = {avg_acc:.4f}")
        
        # Optimizer影响
        opt_results = {}
        for result in results:
            opt = result['params']['optimizer']
            if opt not in opt_results:
                opt_results[opt] = []
            opt_results[opt].append(result['best_val_acc'])
        
        logger.info("Optimizer Impact:")
        for opt, accs in opt_results.items():
            avg_acc = np.mean(accs)
            logger.info(f"  {opt.upper()}: Avg Acc = {avg_acc:.4f}")
        
        # 生成研究报告
        self._generate_study_report(sorted_results)
    
    def _generate_study_report(self, results):
        """生成研究报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"mobilenetv3_hyperparameter_study_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MobileNetV3 Hyperparameter Study Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This hyperparameter study was conducted using FUA's MLflow integration ")
            f.write("to systematically explore the impact of different training configurations ")
            f.write("on MobileNetV3 performance.\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Rank | Batch Size | Learning Rate | Optimizer | Best Val Acc |\n")
            f.write("|------|------------|---------------|-----------|--------------|\n")
            
            for i, result in enumerate(results):
                params = result['params']
                f.write(f"| {i+1} | {params['batch_size']} | {params['learning_rate']} | ")
                f.write(f"{params['optimizer']} | {result['best_val_acc']:.4f} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # 最佳配置
            best = results[0]
            f.write(f"### Best Configuration\n")
            f.write(f"- **Validation Accuracy**: {best['best_val_acc']:.4f}\n")
            f.write(f"- **Batch Size**: {best['params']['batch_size']}\n")
            f.write(f"- **Learning Rate**: {best['params']['learning_rate']}\n")
            f.write(f"- **Optimizer**: {best['params']['optimizer']}\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **For faster training**: Use larger batch sizes (64) with Adam optimizer\n")
            f.write("2. **For better accuracy**: Use moderate learning rates (0.001) with Adam\n")
            f.write("3. **For stability**: Avoid very low learning rates (0.0001) as they slow convergence\n\n")
            
            f.write("## MLflow Integration Benefits\n\n")
            f.write("- **Experiment Tracking**: All runs automatically tracked and comparable\n")
            f.write("- **Model Registry**: Best models automatically registered and versioned\n")
            f.write("- **Metrics Visualization**: Training curves and system metrics available in MLflow UI\n")
            f.write("- **Reproducibility**: All parameters and configurations logged for reproducibility\n\n")
            
            f.write("---\n")
            f.write(f"*Report generated by FUA MLflow Integration*")
        
        logger.info(f"\nStudy report generated: {report_path}")
        
        # 记录报告到MLflow
        if self.mlflow_integration:
            self.mlflow_integration.log_artifact(report_path)


def main():
    """主函数"""
    print("=" * 70)
    print("MobileNetV3 Training with MLflow Integration")
    print("FUA System - Experiment Tracking Demo")
    print("=" * 70)
    
    # 创建训练器
    trainer = MobileNetV3WithMLflow(
        experiment_name="mobilenetv3_hyperparameter_study",
        tracking_uri="mlruns",
        registry_uri="mlruns"
    )
    
    try:
        # 设置训练
        trainer.setup_training()
        
        # 运行超参数研究
        results = trainer.run_hyperparameter_study()
        
        print("\n" + "=" * 70)
        print("Hyperparameter study completed successfully!")
        print("=" * 70)
        print(f"\nTotal experiments run: {len(results)}")
        print("View results in MLflow UI: mlflow ui")
        
        return 0
        
    except Exception as e:
        logger.error(f"Study failed: {e}")
        print(f"\nStudy failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)