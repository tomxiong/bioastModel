#!/usr/bin/env python3
"""
FUA迭代平台快速开始示例
演示如何使用FUA平台进行模型迭代优化
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=== FUA迭代平台快速开始 ===\n")
    
    # 1. 数据集管理示例
    print("1. 数据集版本管理示例")
    print("-" * 30)
    
    try:
        from fua.dataset_iteration_manager import DatasetVersionManager, DatasetIncrementalUpdater
        
        # 创建版本管理器
        version_manager = DatasetVersionManager("bioast_dataset")
        
        # 创建版本
        version_info = version_manager.create_version("v1.0", "初始版本")
        print(f"✓ 创建版本: {version_info['version']}")
        
        # 获取版本信息
        current = version_manager.get_version_info()
        print(f"✓ 当前版本: {current['version']}")
        
    except Exception as e:
        print(f"⚠ 数据集管理示例跳过: {e}")
    
    print()
    
    # 2. 参数优化示例
    print("2. 参数优化示例")
    print("-" * 30)
    
    try:
        from fua.parameter_optimizer import ParameterHistoryManager, ParameterOptimizer
        
        # 创建参数管理器
        history_manager = ParameterHistoryManager("fua/parameter_history")
        optimizer = ParameterOptimizer("example_model", history_manager)
        
        # 记录一些历史实验
        experiments = [
            {"lr": 0.001, "batch": 32, "acc": 0.85},
            {"lr": 0.01, "batch": 64, "acc": 0.88},
            {"lr": 0.0005, "batch": 16, "acc": 0.82},
        ]
        
        for exp in experiments:
            params = {
                "learning_rate": exp["lr"],
                "batch_size": exp["batch"],
                "epochs": 50,
                "optimizer": "adam"
            }
            metrics = {"accuracy": exp["acc"]}
            history_manager.record_experiment("example_model", params, metrics)
        
        # 获取参数建议
        suggestion = optimizer.suggest_parameters("adaptive")
        print(f"✓ 自适应参数建议: {suggestion}")
        
        # 获取最佳配置
        best_config = history_manager.get_best_config("example_model")
        if best_config:
            print(f"✓ 最佳准确率: {best_config['metrics']['accuracy']:.3f}")
        
    except Exception as e:
        print(f"⚠ 参数优化示例跳过: {e}")
    
    print()
    
    # 3. 训练流水线示例
    print("3. 训练流水线示例")
    print("-" * 30)
    
    try:
        from fua.training_pipeline import PipelineManager
        
        # 创建流水线管理器
        manager = PipelineManager()
        
        # 获取训练摘要
        summary = manager.get_training_summary()
        print(f"✓ 训练摘要: {summary}")
        
        # 注意：实际训练需要准备模型和数据集
        # job_id = manager.quick_train("resnet18", epochs=10)
        # print(f"✓ 提交训练任务: {job_id}")
        
    except Exception as e:
        print(f"⚠ 训练流水线示例跳过: {e}")
    
    print()
    
    # 4. Bmad工作流示例
    print("4. Bmad工作流示例")
    print("-" * 30)
    
    try:
        from fua.bmad_workflow_engine import BmadWorkflowEngine
        
        # 创建工作流引擎
        engine = BmadWorkflowEngine()
        
        # 创建工作流
        workflow_id = engine.create_workflow(
            "quick_start_example",
            "resnet18",
            {
                "target_accuracy": 0.90,
                "max_iterations": 5
            }
        )
        
        print(f"✓ 创建工作流: {workflow_id}")
        
        # 获取工作流状态
        status = engine.get_workflow_status(workflow_id)
        print(f"✓ 工作流状态: {status['status']}")
        
        # 列出所有工作流
        workflows = engine.list_workflows()
        print(f"✓ 总工作流数: {len(workflows)}")
        
    except Exception as e:
        print(f"⚠ Bmad工作流示例跳过: {e}")
    
    print()
    
    # 5. 验证引擎示例
    print("5. 模型验证示例")
    print("-" * 30)
    
    try:
        from fua.validation_engine import ValidationEngine
        
        # 创建验证引擎
        validator = ValidationEngine()
        
        # 注意：实际验证需要模型文件
        # result = validator.validate_model(
        #     "path/to/model.pth",
        #     "bioast_dataset/test",
        #     "resnet18"
        # )
        # print(f"✓ 验证准确率: {result['metrics']['accuracy']:.3f}")
        
        print("✓ 验证引擎已就绪（需要模型文件进行实际验证）")
        
    except Exception as e:
        print(f"⚠ 验证引擎示例跳过: {e}")
    
    print()
    print("=" * 50)
    print("快速开始完成！")
    print("=" * 50)
    print()
    print("下一步操作：")
    print("1. 准备数据集到 bioast_dataset/ 目录")
    print("2. 配置模型文件路径")
    print("3. 运行实际训练和验证")
    print("4. 启动Web界面: python fua/web/app.py")
    print()
    print("更多信息请参考：")
    print("- 部署指南: FUA_Iteration_Deployment_Guide.md")
    print("- 用户故事: FUA_Iteration_User_Stories.md")
    print("- 架构设计: FUA_Iteration_Architecture_Design.md")


if __name__ == "__main__":
    main()