#!/usr/bin/env python3
"""
MobileNetV3 训练调优与分布式监控集成示例

演示如何使用FUA系统对MobileNetV3进行训练调优，
并通过分布式监控系统实时监控训练过程
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
    from fua.model_integration import ModelSelector
    from models.mobilenet_v3 import create_mobilenetv3_large
    from core.config.model_configs import get_model_config
    from core.config.training_configs import get_model_specific_config
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("请确保已安装所有必要的依赖")
    sys.exit(1)


class MobileNetV3TrainingOptimizer:
    """MobileNetV3训练优化器"""
    
    def __init__(self):
        self.model = None
        self.distributed_monitor = None
        self.training_history = []
        self.best_accuracy = 0.0
        self.current_epoch = 0
        self.total_epochs = 50
        
        # 生成模拟数据
        self.train_data, self.train_labels = self._generate_synthetic_data(1000)
        self.val_data, self.val_labels = self._generate_synthetic_data(200)
        
        # 创建数据加载器
        self.train_loader = self._create_dataloader(
            self.train_data, self.train_labels, batch_size=32
        )
        self.val_loader = self._create_dataloader(
            self.val_data, self.val_labels, batch_size=32, shuffle=False
        )
        
    def _generate_synthetic_data(self, num_samples: int) -> tuple:
        """生成合成数据"""
        # 生成随机图像数据
        data = torch.randn(num_samples, 3, 70, 70)
        # 生成随机标签（0或1）
        labels = torch.randint(0, 2, (num_samples,))
        return data, labels
    
    def _create_dataloader(self, data, labels, batch_size=32, shuffle=True):
        """创建数据加载器"""
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def setup_model_and_monitor(self):
        """设置模型和监控器"""
        logger.info("Setting up MobileNetV3 model and distributed monitor...")
        
        # 1. 创建MobileNetV3模型
        self.model = create_mobilenetv3_large(num_classes=2)
        
        # 直接设置模型配置
        model_config = {
            "input_size": (3, 70, 70),
            "num_classes": 2,
            "model_type": "MobileNetV3-Large"
        }
        
        # 2. 设置分布式监控器
        monitor_config = {
            "node_role": "monitor",
            "region": "training-region",
            "consul_enabled": False,
            "redis_enabled": False,
            "websocket_port": 8767,
            "api_port": 8082
        }
        
        self.distributed_monitor = create_distributed_monitor(
            node_id="training_monitor",
            config=monitor_config
        )
        
        # 3. 添加训练告警规则
        self._setup_training_alert_rules()
        
        # 4. 模拟训练集群节点
        self._simulate_training_cluster()
        
        # 5. 添加模型到监控
        self.distributed_monitor.add_model(
            model_id="mobilenetv3_training",
            version_id="v1.0",
            model=self.model,
            config=model_config
        )
        
        logger.info(f"Model and monitor setup complete. Model parameters: {sum(p.numel() for p in self.model.parameters())}")
    
    def _setup_training_alert_rules(self):
        """设置训练相关的告警规则"""
        # 训练准确率告警
        self.distributed_monitor.alert_manager.add_alert_rule("low_accuracy", {
            "condition": {
                "training.accuracy": 0.5
            },
            "operator": "<",
            "message": "Training accuracy is too low"
        })
        
        # 验证损失告警
        self.distributed_monitor.alert_manager.add_alert_rule("high_val_loss", {
            "condition": {
                "validation.loss": 2.0
            },
            "operator": ">",
            "message": "Validation loss is too high"
        })
        
        # GPU内存使用告警
        self.distributed_monitor.alert_manager.add_alert_rule("high_gpu_memory", {
            "condition": {
                "system.gpu_memory_percent": 90.0
            },
            "operator": ">",
            "message": "GPU memory usage is too high"
        })
        
        logger.info("Training alert rules configured")
    
    def _simulate_training_cluster(self):
        """模拟训练集群节点"""
        # 添加模拟的worker节点
        worker_nodes = [
            ClusterNode(
                id=f"worker_{i}",
                host=f"192.168.1.{10+i}",
                port=9000 + i,
                role=NodeRole.MONITOR,
                status=NodeStatus.ACTIVE,
                region="training-region",
                zone=f"zone-{i%2}",
                load=np.random.uniform(0.2, 0.9)
            )
            for i in range(4)
        ]
        
        # 添加参数服务器节点
        ps_node = ClusterNode(
            id="parameter_server",
            host="192.168.1.20",
            port=9004,
            role=NodeRole.AGGREGATOR,
            status=NodeStatus.ACTIVE,
            region="training-region",
            zone="zone-0",
            load=0.5
        )
        
        # 添加到集群管理器
        cluster_manager = self.distributed_monitor.cluster_manager
        for node in worker_nodes + [ps_node]:
            cluster_manager.nodes[node.id] = node
        
        logger.info(f"Simulated training cluster with {len(worker_nodes)} workers and 1 parameter server")
    
    def train_with_monitoring(self):
        """带监控的训练过程"""
        logger.info("Starting MobileNetV3 training with distributed monitoring...")
        
        # 启动监控
        self.distributed_monitor.start()
        
        # 设置训练参数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 训练循环
        for epoch in range(self.total_epochs):
            self.current_epoch = epoch + 1
            epoch_start_time = time.time()
            
            # 训练阶段
            train_loss, train_acc = self._train_epoch(criterion, optimizer)
            
            # 验证阶段
            val_loss, val_acc = self._validate_epoch(criterion)
            
            # 学习率调整
            scheduler.step()
            
            # 记录训练历史
            epoch_time = time.time() - epoch_start_time
            self.training_history.append({
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            })
            
            # 收集和报告训练指标
            self._report_training_metrics(train_loss, train_acc, val_loss, val_acc)
            
            # 收集分布式指标
            self._collect_distributed_metrics()
            
            # 保存最佳模型
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self._save_checkpoint(epoch, val_acc)
            
            # 打印进度
            if epoch % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.total_epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 停止监控
        self.distributed_monitor.stop()
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_accuracy:.4f}")
        
        # 生成训练报告
        self._generate_training_report()
    
    def _train_epoch(self, criterion, optimizer) -> tuple:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, criterion) -> tuple:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _report_training_metrics(self, train_loss, train_acc, val_loss, val_acc):
        """报告训练指标到本地监控器"""
        if hasattr(self.distributed_monitor.local_monitor, 'metrics_collector'):
            # 生成一些测试数据
            test_data = torch.randn(16, 3, 70, 70)
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(test_data)
            
            # 收集指标
            metrics = self.distributed_monitor.local_monitor.metrics_collector.collect_metrics(
                self.model, "mobilenetv3_training", "v1.0", test_data, outputs
            )
            
            # 添加训练特定指标
            metrics.custom_metrics.update({
                "training_loss": train_loss,
                "training_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "epoch": self.current_epoch
            })
    
    def _collect_distributed_metrics(self):
        """收集分布式指标"""
        try:
            metrics = self.distributed_monitor.collect_distributed_metrics(
                model_id="mobilenetv3_training",
                version_id="v1.0"
            )
            
            # 检查告警
            self.distributed_monitor.alert_manager.check_alerts(metrics)
            
            # 每5个epoch打印一次集群状态
            if self.current_epoch % 5 == 0:
                cluster_status = self.distributed_monitor.cluster_manager.get_cluster_status()
                logger.info(f"Cluster status - Active nodes: {cluster_status['active_nodes']}, "
                          f"CPU avg: {metrics['aggregated'].get('cpu', {}).get('avg', 0):.1f}%")
                
        except Exception as e:
            logger.error(f"Error collecting distributed metrics: {e}")
    
    def _save_checkpoint(self, epoch, accuracy):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'training_history': self.training_history
        }
        
        checkpoint_path = f"checkpoints/mobilenetv3_epoch_{epoch}.pth"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _generate_training_report(self):
        """生成训练报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"mobilenetv3_training_report_{timestamp}.md"
        
        # 生成集群报告
        cluster_report_path = self.distributed_monitor.generate_cluster_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MobileNetV3 Training Report with Distributed Monitoring\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Training Summary\n\n")
            f.write(f"- **Model:** MobileNetV3-Large\n")
            f.write(f"- **Total Epochs:** {self.total_epochs}\n")
            f.write(f"- **Best Validation Accuracy:** {self.best_accuracy:.4f}\n")
            f.write(f"- **Model Parameters:** {sum(p.numel() for p in self.model.parameters()):,}\n\n")
            
            f.write("## Final Metrics\n\n")
            final_metrics = self.training_history[-1]
            f.write(f"- Training Loss: {final_metrics['train_loss']:.4f}\n")
            f.write(f"- Training Accuracy: {final_metrics['train_acc']:.4f}\n")
            f.write(f"- Validation Loss: {final_metrics['val_loss']:.4f}\n")
            f.write(f"- Validation Accuracy: {final_metrics['val_acc']:.4f}\n")
            f.write(f"- Learning Rate: {final_metrics['lr']:.6f}\n\n")
            
            f.write("## Training Progress\n\n")
            f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR\n")
            f.write("------|------------|-----------|----------|---------|------\n")
            
            for record in self.training_history[::5]:  # 每5个epoch显示一次
                f.write(f"{record['epoch']:5d} | {record['train_loss']:.4f} | {record['train_acc']:.4f} | ")
                f.write(f"{record['val_loss']:.4f} | {record['val_acc']:.4f} | {record['lr']:.6f}\n")
            
            f.write("\n## Distributed Monitoring Insights\n\n")
            f.write("The training process was monitored using FUA's distributed monitoring system:\n\n")
            f.write("- **Real-time metrics collection** from multiple cluster nodes\n")
            f.write("- **Automated alerting** for training anomalies\n")
            f.write("- **Resource utilization monitoring** across the cluster\n")
            f.write("- **Health checks** and failover capabilities\n\n")
            
            f.write(f"## Cluster Report\n\n")
            f.write(f"A detailed cluster monitoring report was generated at: `{cluster_report_path}`\n\n")
            
            f.write("## Key Achievements\n\n")
            f.write("1. **Successful Integration**: Integrated distributed monitoring with model training\n")
            f.write("2. **Real-time Visibility**: Gained real-time insights into training performance\n")
            f.write("3. **Scalable Architecture**: Demonstrated scalable training architecture\n")
            f.write("4. **Automated Monitoring**: Implemented automated monitoring and alerting\n")
            f.write("5. **Comprehensive Reporting**: Generated detailed training and cluster reports\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("- Use actual distributed dataset for production training\n")
            f.write("- Implement distributed data parallelism for large-scale training\n")
            f.write("- Integrate with MLflow for experiment tracking\n")
            f.write("- Add more sophisticated hyperparameter optimization\n")
            f.write("- Implement model checkpointing and versioning\n")
        
        logger.info(f"Training report generated: {report_path}")
        return report_path
    
    def run_complete_pipeline(self):
        """运行完整的训练和监控流水线"""
        logger.info("Running complete MobileNetV3 training with distributed monitoring pipeline...")
        
        # 1. 设置模型和监控器
        self.setup_model_and_monitor()
        
        # 2. 运行带监控的训练
        self.train_with_monitoring()
        
        logger.info("Pipeline completed successfully!")


def main():
    """主函数"""
    print("=" * 70)
    print("MobileNetV3 Training with Distributed Monitoring")
    print("FUA System Integration Demo")
    print("=" * 70)
    
    # 创建训练优化器实例
    optimizer = MobileNetV3TrainingOptimizer()
    
    try:
        # 运行完整流水线
        optimizer.run_complete_pipeline()
        
        print("\n" + "=" * 70)
        print("Training and monitoring pipeline completed successfully!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\nPipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)