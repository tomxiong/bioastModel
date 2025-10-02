#!/usr/bin/env python3
"""
分布式监控系统测试脚本

使用MobileNetV3作为样例，演示分布式监控系统的功能
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta

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
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("请确保已安装所有必要的依赖")
    sys.exit(1)


class MobileNetV3Example:
    """MobileNetV3样例类"""
    
    def __init__(self):
        self.model = None
        self.monitor = None
        self.test_data = None
        
    def setup_model(self):
        """设置MobileNetV3模型"""
        logger.info("Setting up MobileNetV3 model...")
        
        # 创建MobileNetV3模型
        try:
            self.model = create_mobilenetv3_large(num_classes=2)
            self.model.eval()
            logger.info(f"MobileNetV3 model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        except Exception as e:
            logger.error(f"Failed to create MobileNetV3: {e}")
            # 使用简单的CNN作为备选
            self.model = self._create_simple_cnn()
            logger.info("Using simple CNN as fallback")
        
        # 准备测试数据
        self.test_data = torch.randn(16, 3, 70, 70)
        
    def _create_simple_cnn(self):
        """创建简单的CNN模型作为备选"""
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes)
                )
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
                
        return SimpleCNN(num_classes=2)
    
    def setup_distributed_monitor(self):
        """设置分布式监控器"""
        logger.info("Setting up distributed monitor...")
        
        # 监控配置
        config = {
            "node_role": "monitor",
            "region": "us-west-1",
            "consul_enabled": False,  # 禁用Consul（测试环境）
            "redis_enabled": False,   # 禁用Redis（测试环境）
            "websocket_port": 8765,
            "api_port": 8080
        }
        
        # 创建分布式监控器
        self.monitor = create_distributed_monitor(
            node_id="mobilenetv3_monitor",
            config=config
        )
        
        # 添加告警规则
        self._setup_alert_rules()
        
    def _setup_alert_rules(self):
        """设置告警规则"""
        # CPU使用率告警
        self.monitor.alert_manager.add_alert_rule("high_cpu", {
            "condition": {
                "aggregated.cpu.avg": 80.0
            },
            "operator": ">",
            "message": "High CPU usage detected"
        })
        
        # 内存使用率告警
        self.monitor.alert_manager.add_alert_rule("high_memory", {
            "condition": {
                "aggregated.memory.avg": 85.0
            },
            "operator": ">",
            "message": "High memory usage detected"
        })
        
        # 延迟告警
        self.monitor.alert_manager.add_alert_rule("high_latency", {
            "condition": {
                "aggregated.latency.avg": 100.0
            },
            "operator": ">",
            "message": "High latency detected"
        })
        
        logger.info("Alert rules configured")
    
    def simulate_cluster_nodes(self):
        """模拟集群节点"""
        logger.info("Simulating cluster nodes...")
        
        # 添加模拟节点
        nodes_config = [
            {"id": "node_1", "role": NodeRole.MONITOR, "region": "us-west-1", "zone": "zone-a"},
            {"id": "node_2", "role": NodeRole.AGGREGATOR, "region": "us-west-1", "zone": "zone-b"},
            {"id": "node_3", "role": NodeRole.MONITOR, "region": "us-west-2", "zone": "zone-a"},
            {"id": "node_4", "role": NodeRole.STORAGE, "region": "us-west-2", "zone": "zone-b"}
        ]
        
        for node_config in nodes_config:
            node = ClusterNode(
                id=node_config["id"],
                host=f"192.168.1.{node_config['id'].split('_')[1]}",
                port=8000 + int(node_config['id'].split('_')[1]),
                role=node_config["role"],
                status=NodeStatus.ACTIVE,
                region=node_config["region"],
                zone=node_config["zone"],
                load=np.random.uniform(0.1, 0.8)
            )
            self.monitor.cluster_manager.nodes[node.id] = node
            
        logger.info(f"Added {len(nodes_config)} simulated nodes")
    
    def run_monitoring_demo(self, duration: int = 300):
        """运行监控演示"""
        logger.info(f"Starting monitoring demo for {duration} seconds...")
        
        # 添加模型到监控
        self.monitor.add_model(
            model_id="mobilenetv3",
            version_id="v1.0",
            model=self.model,
            config={"input_size": (3, 70, 70), "model_type": "MobileNetV3"}
        )
        
        # 启动监控
        self.monitor.start()
        
        try:
            start_time = time.time()
            iteration = 0
            
            while time.time() - start_time < duration:
                iteration += 1
                
                # 收集分布式指标
                metrics = self.monitor.collect_distributed_metrics(
                    model_id="mobilenetv3",
                    version_id="v1.0"
                )
                
                # 检查告警
                self.monitor.alert_manager.check_alerts(metrics)
                
                # 模拟模型推理
                self._simulate_inference()
                
                # 打印状态
                if iteration % 10 == 0:
                    cluster_status = self.monitor.cluster_manager.get_cluster_status()
                    logger.info(f"Iteration {iteration}: "
                               f"Active nodes: {cluster_status['active_nodes']}, "
                               f"CPU avg: {metrics['aggregated'].get('cpu', {}).get('avg', 0):.1f}%")
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        finally:
            # 停止监控
            self.monitor.stop()
            
            # 生成报告
            report_path = self.monitor.generate_cluster_report()
            logger.info(f"Cluster report generated: {report_path}")
    
    def _simulate_inference(self):
        """模拟模型推理"""
        try:
            with torch.no_grad():
                # 执行推理
                outputs = self.model(self.test_data)
                
                # 模拟一些计算
                predictions = torch.softmax(outputs, dim=1)
                confidence = predictions.mean().item()
                
                # 更新本地监控指标
                if hasattr(self.monitor.local_monitor, 'metrics_collector'):
                    metrics = self.monitor.local_monitor.metrics_collector.collect_metrics(
                        self.model, "mobilenetv3", "v1.0", self.test_data, predictions
                    )
                    
        except Exception as e:
            logger.error(f"Inference simulation error: {e}")
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        logger.info("Running comprehensive distributed monitoring test...")
        
        # 1. 设置模型
        self.setup_model()
        
        # 2. 设置监控
        self.setup_distributed_monitor()
        
        # 3. 模拟集群
        self.simulate_cluster_nodes()
        
        # 4. 运行演示
        self.run_monitoring_demo(duration=120)  # 运行2分钟
        
        logger.info("Comprehensive test completed")


def main():
    """主函数"""
    print("=" * 60)
    print("FUA Distributed Monitoring System Test")
    print("MobileNetV3 Example")
    print("=" * 60)
    
    # 创建样例实例
    example = MobileNetV3Example()
    
    try:
        # 运行综合测试
        example.run_comprehensive_test()
        
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("Check the generated cluster report for details.")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\nTest failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)