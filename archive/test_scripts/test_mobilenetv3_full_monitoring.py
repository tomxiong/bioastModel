#!/usr/bin/env python3
"""
FUA MLflow与分布式监控集成测试

演示MLflow实验跟踪与分布式监控系统的完整集成
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path
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


class MobileNetV3WithFullMonitoring:
    """集成MLflow和分布式监控的MobileNetV3训练器"""
    
    def __init__(self, 
                 experiment_name: str = "mobilenetv3_full_monitoring",
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
        
        # 创建MLflow集成
        self.mlflow_integration = create_mlflow_integration(
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            experiment_name=experiment_name
        )
        
        logger.info(f"MobileNetV3 trainer with full monitoring initialized")
        logger.info(f"Experiment: {experiment_name}")
    
    def setup_training(self):
        """设置训练参数"""
        logger.info("Setting up MobileNetV3 training with full monitoring...")
        
        # 1. 创建模型
        self.model = create_mobilenetv3_large(num_classes=2)
        
        # 2. 获取配置
        model_config = {
            "model_type": "MobileNetV3-Large",
            "input_size": (3, 70, 70),
            "num_classes": 2,
            "parameters": 4187536
        }
        
        training_config = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 20,
            "optimizer": "adam"
        }
        
        # 3. 设置分布式监控
        monitor_config = {
            "node_role": "monitor",
            "region": "training-region",
            "consul_enabled": False,
            "redis_enabled": False,
            "websocket_port": 8772,
            "api_port": 8087
        }
        
        self.distributed_monitor = create_distributed_monitor(
            node_id="mobilenetv3_full_monitoring",
            config=monitor_config
        )
        
        # 4. 添加模型到监控
        self.distributed_monitor.add_model(
            model_id="mobilenetv3_full",
            version_id="v1.0",
            model=self.model,
            config=model_config
        )
        
        logger.info(f"Model setup complete. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return model_config, training_config
    
    def train_with_full_monitoring(self, 
                                  epochs: int = 20,
                                  batch_size: int = 32,
                                  learning_rate: float = 0.001,
                                  run_name: str = None):
        """
        使用完整监控的训练
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            run_name: 运行名称
        """
        logger.info("Starting training with full monitoring...")
        
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
            "optimizer": "adam"
        }
        
        # 创建MLflow运行
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"mobilenetv3_full_monitoring_{timestamp}"
        
        run_id = self.mlflow_integration.create_training_run(
            model_name="MobileNetV3-Full",
            model_config=model_config,
            training_config=training_config,
            run_name=run_name,
            tags={"monitoring": "full", "distributed": "true"}
        )
        
        if not run_id:
            logger.error("Failed to create MLflow run")
            return
        
        # 启动分布式监控
        self.distributed_monitor.start()
        
        # 生成数据
        train_data, train_labels = self._generate_synthetic_data(500)
        val_data, val_labels = self._generate_synthetic_data(100)
        
        # 创建数据加载器
        train_loader = self._create_dataloader(train_data, train_labels, batch_size)
        val_loader = self._create_dataloader(val_data, val_labels, batch_size, shuffle=False)
        
        # 设置优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 训练循环
        best_val_acc = 0.0
        best_epoch = 0
        
        try:
            for epoch in range(epochs):
                # 训练
                train_loss, train_acc = self._train_epoch(
                    self.model, train_loader, optimizer, criterion
                )
                
                # 验证
                val_loss, val_acc = self._validate_epoch(
                    self.model, val_loader, criterion
                )
                
                # 学习率调整
                scheduler.step()
                
                # 记录到MLflow
                mlflow_metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                self.mlflow_integration.log_training_metrics(mlflow_metrics, step=epoch + 1)
                
                # 记录到训练历史
                self.training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]['lr']
                })
                
                # 收集分布式指标
                if epoch % 5 == 0:
                    dist_metrics = self.distributed_monitor.collect_distributed_metrics(
                        model_id="mobilenetv3_full",
                        version_id="v1.0"
                    )
                    
                    # 记录系统指标到MLflow
                    if dist_metrics.get('aggregated'):
                        system_metrics = {
                            "cpu_avg": dist_metrics['aggregated'].get('cpu', {}).get('avg', 0),
                            "memory_avg": dist_metrics['aggregated'].get('memory', {}).get('avg', 0),
                            "latency_avg": dist_metrics['aggregated'].get('latency', {}).get('avg', 0)
                        }
                        self.mlflow_integration.log_training_metrics(system_metrics, step=epoch + 1)
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    self._save_checkpoint(epoch, val_acc, optimizer)
                
                # 打印进度
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
                "final_val_acc": val_acc,
                "total_epochs": epochs
            }
            
            # 记录模型到MLflow
            self.mlflow_integration.log_model_and_register(
                model=self.model,
                model_name="MobileNetV3-Full",
                model_config=model_config,
                input_example=torch.randn(1, 3, 70, 70),
                stage="Staging"
            )
            
            # 记录训练历史作为工件
            history_path = f"training_history_full_{run_id}.json"
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
    
    def _generate_synthetic_data(self, num_samples: int) -> tuple:
        """生成合成数据"""
        data = torch.randn(num_samples, 3, 70, 70)
        labels = torch.randint(0, 2, (num_samples,))
        return data, labels
    
    def _create_dataloader(self, data, labels, batch_size=32, shuffle=True):
        """创建数据加载器"""
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
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
        
        checkpoint_path = f"checkpoints/mobilenetv3_full_epoch_{epoch}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def generate_monitoring_report(self):
        """生成监控报告"""
        logger.info("Generating monitoring report...")
        
        # 获取MLflow实验摘要
        mlflow_summary = self.mlflow_integration.get_experiment_summary()
        
        # 获取分布式监控指标
        dist_metrics = self.distributed_monitor.collect_distributed_metrics(
            model_id="mobilenetv3_full",
            version_id="v1.0"
        )
        
        # 生成报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"mobilenetv3_full_monitoring_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MobileNetV3 Full Monitoring Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## MLflow Experiment Summary\n\n")
            f.write(f"- **Experiment Name**: {mlflow_summary.get('experiment_name', 'N/A')}\n")
            f.write(f"- **Total Runs**: {mlflow_summary.get('total_runs', 0)}\n")
            f.write(f"- **Completed Runs**: {mlflow_summary.get('finished_runs', 0)}\n")
            f.write(f"- **Failed Runs**: {mlflow_summary.get('failed_runs', 0)}\n\n")
            
            f.write("## Training Performance\n\n")
            if self.training_history:
                best_epoch = max(self.training_history, key=lambda x: x['val_acc'])
                f.write(f"- **Best Validation Accuracy**: {best_epoch['val_acc']:.4f} (Epoch {best_epoch['epoch']})\n")
                f.write(f"- **Final Training Accuracy**: {self.training_history[-1]['train_acc']:.4f}\n")
                f.write(f"- **Final Validation Accuracy**: {self.training_history[-1]['val_acc']:.4f}\n\n")
            
            f.write("## Distributed Monitoring Metrics\n\n")
            if dist_metrics.get('aggregated'):
                agg = dist_metrics['aggregated']
                f.write("### Aggregated Metrics\n")
                f.write(f"- **CPU Usage**: {agg.get('cpu', {}).get('avg', 0):.2f}%\n")
                f.write(f"- **Memory Usage**: {agg.get('memory', {}).get('avg', 0):.2f}%\n")
                f.write(f"- **Latency**: {agg.get('latency', {}).get('avg', 0):.2f}ms\n\n")
            
            f.write("## System Integration\n\n")
            f.write("This demo showcases the integration of:\n")
            f.write("1. **MLflow Experiment Tracking**: All training metrics and parameters logged\n")
            f.write("2. **Model Registry**: Models automatically registered and versioned\n")
            f.write("3. **Distributed Monitoring**: Real-time system and model performance monitoring\n")
            f.write("4. **Unified Architecture**: Seamless integration of all components\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Run `mlflow ui` to view experiment results\n")
            f.write("2. Check the registered models in MLflow Model Registry\n")
            f.write("3. Monitor the distributed metrics through the API\n")
            f.write("4. Extend to multi-node distributed training\n")
            
        logger.info(f"Report generated: {report_path}")
        
        # 记录报告到MLflow
        if self.mlflow_integration:
            self.mlflow_integration.log_artifact(report_path)


def main():
    """主函数"""
    print("=" * 80)
    print("MobileNetV3 Training with MLflow + Distributed Monitoring")
    print("FUA System - Complete Integration Demo")
    print("=" * 80)
    
    # 创建训练器
    trainer = MobileNetV3WithFullMonitoring(
        experiment_name="mobilenetv3_full_monitoring",
        tracking_uri="mlruns",
        registry_uri="mlruns"
    )
    
    try:
        # 设置训练
        trainer.setup_training()
        
        # 运行训练
        trainer.train_with_full_monitoring(
            epochs=20,
            batch_size=32,
            learning_rate=0.001
        )
        
        # 生成报告
        trainer.generate_monitoring_report()
        
        print("\n" + "=" * 80)
        print("Full Monitoring Demo Completed Successfully!")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("✓ MLflow experiment tracking")
        print("✓ Model registration and versioning")
        print("✓ Distributed system monitoring")
        print("✓ Real-time metrics collection")
        print("✓ Unified architecture integration")
        print("\nNext Steps:")
        print("1. View experiments: mlflow ui")
        print("2. Check generated reports")
        print("3. Explore registered models")
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)