#!/usr/bin/env python3
"""
FUA MLflow集成快速测试

演示MLflow实验跟踪与MobileNetV3训练的基本集成
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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


def quick_mlflow_test():
    """快速MLflow集成测试"""
    print("=" * 70)
    print("MobileNetV3 Training with MLflow Integration - Quick Test")
    print("FUA System - Experiment Tracking Demo")
    print("=" * 70)
    
    # 创建MLflow集成
    mlflow_integration = create_mlflow_integration(
        tracking_uri="mlruns",
        registry_uri="mlruns",
        experiment_name="mobilenetv3_quick_test"
    )
    
    # 创建模型
    model = create_mobilenetv3_large(num_classes=2)
    logger.info(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成少量合成数据
    train_data = torch.randn(100, 3, 70, 70)
    train_labels = torch.randint(0, 2, (100,))
    val_data = torch.randn(50, 3, 70, 70)
    val_labels = torch.randint(0, 2, (50,))
    
    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=16, shuffle=False)
    
    # 配置
    model_config = {
        "model_type": "MobileNetV3-Large",
        "input_size": (3, 70, 70),
        "num_classes": 2
    }
    
    training_config = {
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 0.001,
        "optimizer": "adam"
    }
    
    # 创建MLflow运行
    run_id = mlflow_integration.create_training_run(
        model_name="MobileNetV3",
        model_config=model_config,
        training_config=training_config,
        run_name="quick_test",
        tags={"test": "quick"}
    )
    
    if not run_id:
        logger.error("Failed to create MLflow run")
        return
    
    logger.info(f"MLflow run created: {run_id}")
    
    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 快速训练
    logger.info("Starting quick training...")
    for epoch in range(5):
        # 训练
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # 验证
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        # 记录到MLflow
        mlflow_integration.log_training_metrics({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch + 1)
        
        logger.info(f"Epoch [{epoch+1}/5] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 记录最终指标和模型
    final_metrics = {
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc
    }
    
    # 记录并注册模型
    mlflow_integration.log_model_and_register(
        model=model,
        model_name="MobileNetV3",
        model_config=model_config,
        input_example=torch.randn(1, 3, 70, 70),
        stage="Staging"
    )
    
    # 完成运行
    mlflow_integration.complete_training_run(final_metrics)
    
    logger.info("Quick test completed successfully!")
    
    # 获取实验摘要
    summary = mlflow_integration.get_experiment_summary()
    logger.info(f"Experiment summary: {summary}")
    
    print("\n" + "=" * 70)
    print("MLflow Integration Quick Test Completed!")
    print("=" * 70)
    print("View results with: mlflow ui")
    print(f"Experiment: {mlflow_integration.experiment_name}")
    print(f"Total runs: {summary.get('total_runs', 0)}")


if __name__ == "__main__":
    quick_mlflow_test()