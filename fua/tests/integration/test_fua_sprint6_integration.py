"""
Sprint 6 Model Integration Demo

演示模型集成、评估、选择和部署的完整流程
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

# 导入FUA组件
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fua.model_integration import (
    ModelIntegrator, ModelEvaluator, ModelSelector, ModelDeployer,
    ModelMetadata, ModelFormat, ModelStatus,
    EvaluationType, SelectionCriteria, SelectionStrategy,
    DeploymentConfig, DeploymentPlatform, DeploymentFormat,
    OptimizationLevel, DeploymentStatus, create_model_integrator, 
    create_model_evaluator, create_model_selector, create_model_deployer
)


class SimpleModel(nn.Module):
    """简单模型用于演示"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_test_data(num_samples=100):
    """创建测试数据"""
    # 创建随机数据
    X = torch.randn(num_samples, 3, 32, 32)
    # 创建随机标签
    y = torch.randint(0, 2, (num_samples,))
    return X, y


def demo_model_integration():
    """演示模型集成"""
    print("\n=== 模型集成演示 ===")
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 创建模型集成器
        integrator = create_model_integrator(
            registry_path=str(temp_dir / "registry"),
            models_dir=str(temp_dir / "models")
        )
        
        # 创建几个不同的模型
        models = {
            'simple_cnn': SimpleModel(),
            'deep_cnn': nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 2)
            ),
            'lightweight': nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(8, 2)
            )
        }
        
        # 集成模型
        version_ids = []
        for name, model in models.items():
            print(f"\n集成模型: {name}")
            
            version_id = integrator.integrate_pytorch_model(
                model=model,
                name=name,
                version="v1.0",
                description=f"Demo model: {name}",
                author="FUA Demo",
                tags=["demo", "test"],
                config={"epochs": 10, "batch_size": 32}
            )
            
            version_ids.append(version_id)
            print(f"  版本ID: {version_id}")
        
        # 列出所有模型
        print(f"\n已注册的模型数量: {len(integrator.registry.list_models())}")
        
        return integrator, version_ids, temp_dir
        
    except Exception as e:
        print(f"模型集成演示失败: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return None, [], None


def demo_model_evaluation(integrator, version_ids, temp_dir):
    """演示模型评估"""
    print("\n=== 模型评估演示 ===")
    
    try:
        # 创建评估器
        evaluator = create_model_evaluator(
            output_dir=str(temp_dir / "evaluation"),
            device="cpu"
        )
        
        # 创建测试数据
        test_data = create_test_data(200)
        
        # 评估所有模型
        evaluation_results = {}
        for version_id in version_ids:
            model_id = version_id.rsplit('_v1', 1)[0]
            print(f"\n评估模型: {model_id}")
            
            # 加载模型
            model = integrator.load_model(model_id)
            
            # 评估模型
            result = evaluator.evaluate_model(
                model=model,
                model_id=model_id,
                version_id=version_id,
                test_data=test_data,
                evaluation_types=[
                    EvaluationType.ACCURACY,
                    EvaluationType.PRECISION,
                    EvaluationType.RECALL,
                    EvaluationType.F1_SCORE,
                    EvaluationType.INFERENCE_TIME,
                    EvaluationType.MEMORY_USAGE
                ]
            )
            
            evaluation_results[model_id] = result
            
            # 打印结果
            print(f"  准确率: {result.metrics.accuracy:.4f}")
            print(f"  F1分数: {result.metrics.f1_score:.4f}")
            print(f"  推理时间: {result.metrics.avg_inference_time*1000:.2f}ms")
            print(f"  模型大小: {result.metrics.model_size_mb:.2f}MB")
        
        return evaluator, evaluation_results
        
    except Exception as e:
        print(f"模型评估演示失败: {e}")
        return None, {}


def demo_model_selection(evaluation_results, temp_dir):
    """演示模型选择"""
    print("\n=== 模型选择演示 ===")
    
    try:
        # 创建选择器
        selector = create_model_selector(
            output_dir=str(temp_dir / "selection")
        )
        
        # 将评估结果转换为列表
        results_list = list(evaluation_results.values())
        
        # 使用不同策略选择模型
        strategies = [
            (SelectionCriteria.ACCURACY, SelectionStrategy.TOP_PERFORMER),
            (SelectionCriteria.SPEED, SelectionStrategy.TOP_PERFORMER),
            (SelectionCriteria.MEMORY, SelectionStrategy.TOP_PERFORMER),
            (SelectionCriteria.BALANCED, SelectionStrategy.WEIGHTED_SCORE),
            (SelectionCriteria.BALANCED, SelectionStrategy.PARETO_OPTIMAL)
        ]
        
        selection_results = {}
        for criteria, strategy in strategies:
            print(f"\n选择策略: {criteria.value} + {strategy.value}")
            
            result = selector.select_model(
                evaluation_results=results_list,
                criteria=criteria,
                strategy=strategy
            )
            
            selection_results[f"{criteria.value}_{strategy.value}"] = result
            
            print(f"  选择的模型: {result.selected_model_id}")
            print(f"  分数: {result.score:.4f}")
            print(f"  排名: {[m for m, s in result.ranking[:3]]}")
        
        # 生成选择报告
        report_path = selector.generate_selection_report(
            list(selection_results.values())
        )
        print(f"\n选择报告已保存: {report_path}")
        
        return selector, selection_results
        
    except Exception as e:
        print(f"模型选择演示失败: {e}")
        return None, {}


def demo_model_deployment(integrator, selection_results, temp_dir):
    """演示模型部署"""
    print("\n=== 模型部署演示 ===")
    
    try:
        # 创建部署器
        deployer = create_model_deployer(
            output_dir=str(temp_dir / "deployments"),
            model_integrator=integrator
        )
        
        # 获取最佳模型
        best_selection = None
        for result in selection_results.values():
            if result.selected_model_id:
                best_selection = result
                break
        
        if not best_selection:
            print("没有可部署的模型")
            return deployer, {}
        
        model_id = best_selection.selected_model_id
        version_id = f"{model_id}_v1"
        
        print(f"部署最佳模型: {model_id}")
        
        # 创建不同的部署配置
        deployment_configs = [
            DeploymentConfig(
                platform=DeploymentPlatform.LOCAL,
                format=DeploymentFormat.PYTORCH,
                optimization_level=OptimizationLevel.NONE,
                input_shape=(1, 3, 32, 32)
            ),
            DeploymentConfig(
                platform=DeploymentPlatform.LOCAL,
                format=DeploymentFormat.ONNX,
                optimization_level=OptimizationLevel.BASIC,
                input_shape=(1, 3, 32, 32)
            ),
            DeploymentConfig(
                platform=DeploymentPlatform.EDGE,
                format=DeploymentFormat.JIT,
                optimization_level=OptimizationLevel.ADVANCED,
                quantization=True,
                input_shape=(1, 3, 32, 32)
            )
        ]
        
        deployment_results = {}
        for i, config in enumerate(deployment_configs):
            print(f"\n部署配置 {i+1}:")
            print(f"  平台: {config.platform.value}")
            print(f"  格式: {config.format.value}")
            print(f"  优化级别: {config.optimization_level.value}")
            
            # 部署模型
            result = deployer.deploy_model(
                model_id=model_id,
                version_id=version_id,
                config=config,
                deployment_id=f"demo_deploy_{i}"
            )
            
            deployment_results[f"config_{i}"] = result
            
            print(f"  部署状态: {result.status.value}")
            if result.status == DeploymentStatus.DEPLOYED:
                print(f"  模型大小: {result.metrics.model_size_mb:.2f}MB")
                print(f"  推理时间: {result.metrics.inference_time_ms:.2f}ms")
            
            # 打印日志
            for log in result.logs[-3:]:
                print(f"    {log}")
        
        # 生成部署报告
        report_path = deployer.generate_deployment_report()
        print(f"\n部署报告已保存: {report_path}")
        
        return deployer, deployment_results
        
    except Exception as e:
        print(f"模型部署演示失败: {e}")
        return None, {}


def main():
    """主演示函数"""
    print("FUA Sprint 6 - 模型集成演示")
    print("=" * 50)
    
    temp_dir = None
    
    try:
        # 1. 模型集成
        integrator, version_ids, temp_dir = demo_model_integration()
        if not integrator:
            return
        
        # 2. 模型评估
        evaluator, evaluation_results = demo_model_evaluation(
            integrator, version_ids, temp_dir
        )
        if not evaluator:
            return
        
        # 3. 模型选择
        selector, selection_results = demo_model_selection(
            evaluation_results, temp_dir
        )
        if not selector:
            return
        
        # 4. 模型部署
        deployer, deployment_results = demo_model_deployment(
            integrator, selection_results, temp_dir
        )
        
        # 总结
        print("\n=== 演示总结 ===")
        print(f"✓ 成功集成 {len(version_ids)} 个模型")
        print(f"✓ 评估了 {len(evaluation_results)} 个模型")
        print(f"✓ 使用了 {len(selection_results)} 种选择策略")
        print(f"✓ 创建了 {len(deployment_results)} 个部署")
        
        # 打印最佳模型信息
        if evaluation_results:
            best_model = max(evaluation_results.items(), 
                           key=lambda x: x[1].metrics.accuracy)
            print(f"\n最佳模型: {best_model[0]}")
            print(f"  准确率: {best_model[1].metrics.accuracy:.4f}")
            print(f"  F1分数: {best_model[1].metrics.f1_score:.4f}")
        
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时目录
        if temp_dir and temp_dir.exists():
            print(f"\n清理临时目录: {temp_dir}")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()