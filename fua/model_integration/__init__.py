"""
FUA Model Integration Module

提供模型集成、评估、选择和部署功能，实现完整的模型生命周期管理
"""

from .model_integrator import (
    ModelIntegrator, ModelRegistry, ModelVersion,
    ModelMetadata, ModelCapabilities, create_model_integrator
)

from .model_evaluator import (
    ModelEvaluator, EvaluationResult, EvaluationMetrics,
    BenchmarkSuite, EvaluationType, create_model_evaluator
)

from .model_selector import (
    ModelSelector, SelectionCriteria, SelectionStrategy,
    SelectionResult, SelectionWeights, SelectionConstraint, create_model_selector
)

from .model_deployer import (
    ModelDeployer, DeploymentConfig, DeploymentResult,
    DeploymentStatus, DeploymentPlatform, DeploymentFormat,
    OptimizationLevel, DeploymentMetrics, create_model_deployer
)

__all__ = [
    # Model Integrator
    'ModelIntegrator',
    'ModelRegistry', 
    'ModelVersion',
    'ModelMetadata',
    'ModelCapabilities',
    'create_model_integrator',
    
    # Model Evaluator
    'ModelEvaluator',
    'EvaluationResult',
    'EvaluationMetrics',
    'BenchmarkSuite',
    'EvaluationType',
    'create_model_evaluator',
    
    # Model Selector
    'ModelSelector',
    'SelectionCriteria',
    'SelectionStrategy',
    'SelectionResult',
    'SelectionWeights',
    'SelectionConstraint',
    'create_model_selector',
    
    # Model Deployer
    'ModelDeployer',
    'DeploymentConfig',
    'DeploymentResult',
    'DeploymentStatus',
    'DeploymentPlatform',
    'DeploymentFormat',
    'OptimizationLevel',
    'DeploymentMetrics',
    'create_model_deployer'
]

# 模块版本
__version__ = "1.0.0"