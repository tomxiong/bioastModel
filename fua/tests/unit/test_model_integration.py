"""
单元测试：模型集成组件
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime

from fua.model_integration import (
    ModelIntegrator, ModelRegistry, ModelMetadata, ModelVersion,
    ModelFormat, ModelStatus, ModelCapabilities,
    EvaluationType, EvaluationMetrics, EvaluationResult,
    SelectionCriteria, SelectionStrategy, SelectionWeights,
    DeploymentConfig, DeploymentPlatform, DeploymentFormat,
    OptimizationLevel, DeploymentStatus,
    create_model_integrator, create_model_evaluator,
    create_model_selector, create_model_deployer
)


class SimpleTestModel(nn.Module):
    """简单测试模型"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestModelIntegrator:
    """测试模型集成器"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.integrator = create_model_integrator(
            registry_path=str(self.temp_dir / "registry"),
            models_dir=str(self.temp_dir / "models")
        )
        self.model = SimpleTestModel()
    
    def teardown_method(self):
        """测试后清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_integrate_pytorch_model(self):
        """测试集成PyTorch模型"""
        version_id = self.integrator.integrate_pytorch_model(
            model=self.model,
            name="test_model",
            version="v1.0",
            description="Test model for unit testing",
            author="Test Author",
            tags=["test", "unit"]
        )
        
        assert version_id is not None
        assert "test_model_v1.0_v1" in version_id
        
        # 验证模型已注册
        model_id = "test_model_v1.0"
        assert model_id in self.integrator.registry.models
        
        # 验证版本已添加
        assert version_id in self.integrator.registry.versions
    
    def test_load_model(self):
        """测试加载模型"""
        # 先集成模型
        version_id = self.integrator.integrate_pytorch_model(
            model=self.model,
            name="test_model",
            version="v1.0"
        )
        
        # 加载模型
        model_id = "test_model_v1.0"
        loaded_model = self.integrator.load_model(model_id)
        
        assert loaded_model is not None
        assert isinstance(loaded_model, nn.Module)
    
    def test_create_new_version(self):
        """测试创建新版本"""
        # 先集成模型
        version_id = self.integrator.integrate_pytorch_model(
            model=self.model,
            name="test_model",
            version="v1.0"
        )
        model_id = "test_model_v1.0"
        
        # 创建新版本
        new_version_id = self.integrator.create_new_version(
            model_id=model_id,
            new_version="v2.0",
            model=self.model,
            changelog="Updated model weights"
        )
        
        assert new_version_id is not None
        assert "test_model_v2.0_v1" in new_version_id
        
        # 验证新版本是活跃的
        active_version = self.integrator.registry.get_active_version("test_model_v2.0")
        assert active_version is not None
        assert active_version.version_id == new_version_id
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        # 先集成模型
        version_id = self.integrator.integrate_pytorch_model(
            model=self.model,
            name="test_model",
            version="v1.0"
        )
        model_id = "test_model_v1.0"
        
        # 获取信息
        info = self.integrator.get_model_info(model_id)
        
        assert info is not None
        assert info['metadata']['name'] == "test_model"
        assert info['metadata']['version'] == "v1.0"
        assert 'active_version' in info
        assert 'total_versions' in info


class TestModelEvaluator:
    """测试模型评估器"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.evaluator = create_model_evaluator(
            output_dir=str(self.temp_dir / "evaluation"),
            device="cpu"
        )
        self.model = SimpleTestModel()
        self.test_data = self._create_test_data()
    
    def teardown_method(self):
        """测试后清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self, num_samples=100):
        """创建测试数据"""
        X = torch.randn(num_samples, 3, 32, 32)
        y = torch.randint(0, 2, (num_samples,))
        return X, y
    
    def test_evaluate_model(self):
        """测试评估模型"""
        result = self.evaluator.evaluate_model(
            model=self.model,
            model_id="test_model",
            version_id="v1",
            test_data=self.test_data,
            evaluation_types=[
                EvaluationType.ACCURACY,
                EvaluationType.PRECISION,
                EvaluationType.RECALL,
                EvaluationType.F1_SCORE,
                EvaluationType.INFERENCE_TIME,
                EvaluationType.MEMORY_USAGE
            ]
        )
        
        assert result is not None
        assert result.model_id == "test_model"
        assert result.version_id == "v1"
        assert isinstance(result.metrics, EvaluationMetrics)
        
        # 验证指标范围
        assert 0 <= result.metrics.accuracy <= 1
        assert 0 <= result.metrics.precision <= 1
        assert 0 <= result.metrics.recall <= 1
        assert 0 <= result.metrics.f1_score <= 1
        assert result.metrics.avg_inference_time > 0
        assert result.metrics.model_size_mb > 0
    
    def test_evaluate_multiple_models(self):
        """测试评估多个模型"""
        models = {
            "model1": SimpleTestModel(),
            "model2": nn.Sequential(
                nn.Conv2d(3, 8, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(8, 2)
            )
        }
        
        results = self.evaluator.evaluate_multiple_models(
            models=models,
            test_data=self.test_data,
            parallel=False
        )
        
        assert len(results) == 2
        assert "model1" in results
        assert "model2" in results
        assert results["model1"] is not None
        assert results["model2"] is not None
    
    def test_generate_comparison_report(self):
        """测试生成对比报告"""
        # 先评估几个模型
        results = {}
        for i in range(3):
            model = SimpleTestModel()
            result = self.evaluator.evaluate_model(
                model=model,
                model_id=f"model_{i}",
                version_id="v1",
                test_data=self.test_data
            )
            results[f"model_{i}"] = result
        
        # 生成报告
        report_path = self.evaluator.generate_comparison_report(
            results=results,
            output_path=str(self.temp_dir / "comparison_report.md")
        )
        
        assert Path(report_path).exists()
        assert Path(report_path).stat().st_size > 0


class TestModelSelector:
    """测试模型选择器"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.selector = create_model_selector(
            output_dir=str(self.temp_dir / "selection")
        )
        self.evaluation_results = self._create_mock_results()
    
    def teardown_method(self):
        """测试后清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_mock_results(self):
        """创建模拟评估结果"""
        results = []
        
        # 模型1：高准确率，慢速
        metrics1 = EvaluationMetrics(
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            avg_inference_time=0.05,  # 50ms
            model_size_mb=10.0
        )
        result1 = EvaluationResult(
            model_id="model1",
            version_id="v1",
            dataset_name="test",
            metrics=metrics1
        )
        results.append(result1)
        
        # 模型2：中等准确率，快速
        metrics2 = EvaluationMetrics(
            accuracy=0.90,
            precision=0.89,
            recall=0.91,
            f1_score=0.90,
            roc_auc=0.95,
            avg_inference_time=0.01,  # 10ms
            model_size_mb=2.0
        )
        result2 = EvaluationResult(
            model_id="model2",
            version_id="v1",
            dataset_name="test",
            metrics=metrics2
        )
        results.append(result2)
        
        # 模型3：低准确率，极快速
        metrics3 = EvaluationMetrics(
            accuracy=0.85,
            precision=0.84,
            recall=0.86,
            f1_score=0.85,
            roc_auc=0.92,
            avg_inference_time=0.005,  # 5ms
            model_size_mb=0.5
        )
        result3 = EvaluationResult(
            model_id="model3",
            version_id="v1",
            dataset_name="test",
            metrics=metrics3
        )
        results.append(result3)
        
        return results
    
    def test_select_top_performer(self):
        """测试选择最佳表现者"""
        result = self.selector.select_model(
            evaluation_results=self.evaluation_results,
            criteria=SelectionCriteria.ACCURACY,
            strategy=SelectionStrategy.TOP_PERFORMER
        )
        
        assert result.selected_model_id == "model1"  # 最高准确率
        assert result.constraints_satisfied is True
        assert len(result.ranking) == 3
        assert result.ranking[0][0] == "model1"
    
    def test_select_speed_optimized(self):
        """测试选择速度优化模型"""
        result = self.selector.select_model(
            evaluation_results=self.evaluation_results,
            criteria=SelectionCriteria.SPEED,
            strategy=SelectionStrategy.TOP_PERFORMER
        )
        
        assert result.selected_model_id == "model3"  # 最快速度
    
    def test_select_pareto_optimal(self):
        """测试选择帕累托最优"""
        result = self.selector.select_model(
            evaluation_results=self.evaluation_results,
            criteria=SelectionCriteria.BALANCED,
            strategy=SelectionStrategy.PARETO_OPTIMAL
        )
        
        # 帕累托前沿应该包含非支配解
        assert result.selected_model_id in ["model1", "model2", "model3"]
        assert "pareto_front" in result.metadata
    
    def test_select_with_constraints(self):
        """测试带约束的选择"""
        from fua.model_integration.model_selector import SelectionConstraint
        
        constraints = [
            SelectionConstraint(
                metric_name="accuracy",
                min_value=0.90,
                operator="ge"
            ),
            SelectionConstraint(
                metric_name="inference_time",
                max_value=0.02,
                operator="le"
            )
        ]
        
        result = self.selector.select_model(
            evaluation_results=self.evaluation_results,
            criteria=SelectionCriteria.BALANCED,
            strategy=SelectionStrategy.TOP_PERFORMER,
            constraints=constraints
        )
        
        # 只有model2满足约束
        assert result.selected_model_id == "model2"


class TestModelDeployer:
    """测试模型部署器"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.deployer = create_model_deployer(
            output_dir=str(self.temp_dir / "deployments")
        )
        self.model = SimpleTestModel()
    
    def teardown_method(self):
        """测试后清理"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_deploy_local_pytorch(self):
        """测试本地PyTorch部署"""
        config = DeploymentConfig(
            platform=DeploymentPlatform.LOCAL,
            format=DeploymentFormat.PYTORCH,
            optimization_level=OptimizationLevel.NONE,
            input_shape=(1, 3, 32, 32)
        )
        
        result = self.deployer.deploy_model(
            model_id="test_model",
            version_id="v1",
            config=config,
            deployment_id="test_deploy_1"
        )
        
        assert result is not None
        assert result.deployment_id == "test_deploy_1"
        assert result.status == DeploymentStatus.DEPLOYED
        assert Path(result.output_path).exists()
    
    def test_deploy_local_onnx(self):
        """测试本地ONNX部署"""
        config = DeploymentConfig(
            platform=DeploymentPlatform.LOCAL,
            format=DeploymentFormat.ONNX,
            optimization_level=OptimizationLevel.BASIC,
            input_shape=(1, 3, 32, 32)
        )
        
        result = self.deployer.deploy_model(
            model_id="test_model",
            version_id="v1",
            config=config,
            deployment_id="test_deploy_2"
        )
        
        assert result is not None
        assert result.status == DeploymentStatus.DEPLOYED
        
        # 检查是否生成了ONNX文件
        onnx_file = Path(result.output_path) / "model.onnx"
        assert onnx_file.exists()
    
    def test_list_deployments(self):
        """测试列出部署"""
        # 先部署几个模型
        for i in range(3):
            config = DeploymentConfig(
                platform=DeploymentPlatform.LOCAL,
                format=DeploymentFormat.PYTORCH if i % 2 == 0 else DeploymentFormat.ONNX,
                optimization_level=OptimizationLevel.NONE,
                input_shape=(1, 3, 32, 32)
            )
            
            self.deployer.deploy_model(
                model_id=f"test_model_{i}",
                version_id="v1",
                config=config,
                deployment_id=f"test_deploy_{i}"
            )
        
        # 列出所有部署
        deployments = self.deployer.list_deployments()
        
        assert len(deployments) == 3
        
        # 列出特定状态的部署
        active_deployments = self.deployer.list_deployments(
            status=DeploymentStatus.DEPLOYED
        )
        assert len(active_deployments) == 3
    
    def test_get_deployment_summary(self):
        """测试获取部署摘要"""
        # 先部署一个模型
        config = DeploymentConfig(
            platform=DeploymentPlatform.LOCAL,
            format=DeploymentFormat.PYTORCH,
            optimization_level=OptimizationLevel.NONE,
            input_shape=(1, 3, 32, 32)
        )
        
        self.deployer.deploy_model(
            model_id="test_model",
            version_id="v1",
            config=config,
            deployment_id="test_deploy"
        )
        
        # 获取摘要
        summary = self.deployer.get_deployment_summary()
        
        assert summary['total_deployments'] == 1
        assert summary['active_deployments'] == 1
        assert summary['failed_deployments'] == 0
        assert summary['success_rate'] == 100.0
        assert 'platform_distribution' in summary
        assert 'format_distribution' in summary


if __name__ == "__main__":
    pytest.main([__file__])