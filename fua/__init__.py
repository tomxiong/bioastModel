"""
FUA - Flexible Unified Architecture
A flexible and unified architecture for ML model training and optimization
"""

__version__ = "1.0.0"
__author__ = "FUA Development Team"

# Core components
from .core.data_structures import ModelCapabilities, ModelMetadata, Error, Improvement
from .core.interfaces import (
    ModelInterface, ConfigManager, DataProcessor, FineTuner, AutomationEngine,
    ConfigurationManager, BaseConfigValidator, ConfigValidator
)
from .core.model_adapters import ModelAdapter, ModelFactory, ModelManager
from .core.model_config import ModelConfigurationSystem, ModelConfigurationManager

# Deployment components (Sprint 3)
try:
    from .deployment.onnx_exporter import ONNXExporter, create_onnx_exporter, export_model_to_onnx
    from .deployment.inference_server import FUAInferenceServer, create_inference_server
    DEPLOYMENT_AVAILABLE = True
except ImportError:
    DEPLOYMENT_AVAILABLE = False
    ONNXExporter = None
    create_onnx_exporter = None
    export_model_to_onnx = None
    FUAInferenceServer = None
    create_inference_server = None

# Pipeline components (Sprint 3)
try:
    from .pipeline.data_processor import (
        BioAstDataProcessor, create_data_processor, DataPipeline, 
        ProcessingMode, QualityLevel, ProcessingResult, DatasetStats
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    BioAstDataProcessor = None
    create_data_processor = None
    DataPipeline = None
    ProcessingMode = None
    QualityLevel = None
    ProcessingResult = None
    DatasetStats = None

# Optimization components (Sprint 3)
try:
    from .optimization.hyperparameter_optimizer import (
        HyperparameterOptimizer, CrossValidationOptimizer, OptimizationResult,
        TrialResult, create_hyperparameter_optimizer, create_cv_optimizer,
        get_default_search_space
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    HyperparameterOptimizer = None
    CrossValidationOptimizer = None
    OptimizationResult = None
    TrialResult = None
    create_hyperparameter_optimizer = None
    create_cv_optimizer = None
    get_default_search_space = None

# Automation Engine components (Sprint 5)
try:
    from .automation.automation_engine import (
        AutomationEngine, AutomationResult, AutomationStatus,
        create_automation_engine
    )
    from .automation.error_detector import (
        ErrorDetector, ErrorCategory, ErrorSeverity,
        create_error_detector
    )
    from .automation.root_cause_analyzer import (
        RootCauseAnalyzer, AnalysisResult, AnalysisConfidence,
        create_root_cause_analyzer
    )
    from .automation.improvement_generator import (
        ImprovementGenerator, ImprovementType, ImprovementPriority,
        create_improvement_generator
    )
    from .automation.fast_validator import (
        FastValidator, ValidationResult, ValidationLevel,
        create_fast_validator
    )
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False
    AutomationEngine = None
    AutomationResult = None
    AutomationStatus = None
    create_automation_engine = None
    ErrorDetector = None
    ErrorCategory = None
    ErrorSeverity = None
    create_error_detector = None
    RootCauseAnalyzer = None
    AnalysisResult = None
    AnalysisConfidence = None
    create_root_cause_analyzer = None
    ImprovementGenerator = None
    ImprovementType = None
    ImprovementPriority = None
    create_improvement_generator = None
    FastValidator = None
    ValidationResult = None
    ValidationLevel = None
    create_fast_validator = None

# Model Integration components (Sprint 6)
try:
    from .model_integration import (
        ModelIntegrator, ModelEvaluator, ModelSelector, ModelDeployer,
        ModelMetadata, ModelFormat, ModelStatus, EvaluationResult, EvaluationMetrics,
        BenchmarkSuite, EvaluationType, SelectionCriteria, SelectionStrategy,
        SelectionResult, SelectionWeights, SelectionConstraint,
        DeploymentConfig, DeploymentResult, DeploymentStatus, DeploymentPlatform,
        DeploymentFormat, OptimizationLevel, DeploymentMetrics,
        create_model_integrator, create_model_evaluator, create_model_selector,
        create_model_deployer
    )
    MODEL_INTEGRATION_AVAILABLE = True
except ImportError:
    MODEL_INTEGRATION_AVAILABLE = False
    ModelIntegrator = None
    ModelEvaluator = None
    ModelSelector = None
    ModelDeployer = None
    ModelMetadata = None
    ModelFormat = None
    ModelStatus = None
    EvaluationResult = None
    EvaluationMetrics = None
    BenchmarkSuite = None
    EvaluationType = None
    SelectionCriteria = None
    SelectionStrategy = None
    SelectionResult = None
    SelectionWeights = None
    SelectionConstraint = None
    DeploymentConfig = None
    DeploymentResult = None
    DeploymentStatus = None
    DeploymentPlatform = None
    DeploymentFormat = None
    OptimizationLevel = None
    DeploymentMetrics = None
    create_model_integrator = None
    create_model_evaluator = None
    create_model_selector = None
    create_model_deployer = None

# Production components (Sprint 7)
try:
    from .production import (
        ModelMonitor, MetricsCollector, AnomalyDetector, AlertManager,
        Alert, AlertSeverity, MetricType, AlertChannel, MetricThreshold,
        ModelMetrics, EmailNotifier, SlackNotifier, WebhookNotifier,
        create_model_monitor
    )
    PRODUCTION_AVAILABLE = True
except ImportError:
    PRODUCTION_AVAILABLE = False
    ModelMonitor = None
    MetricsCollector = None
    AnomalyDetector = None
    AlertManager = None
    Alert = None
    AlertSeverity = None
    MetricType = None
    AlertChannel = None
    MetricThreshold = None
    ModelMetrics = None
    EmailNotifier = None
    SlackNotifier = None
    WebhookNotifier = None
    create_model_monitor = None

__all__ = [
    # Core
    'ModelCapabilities',
    'ModelMetadata', 
    'Error',
    'Improvement',
    'ModelInterface',
    'ConfigManager',
    'DataProcessor',
    'FineTuner',
    'AutomationEngine',
    'ConfigurationManager',
    'BaseConfigValidator',
    'ConfigValidator',
    'ModelAdapter',
    'ModelFactory',
    'ModelManager',
    'ModelConfigurationSystem',
    'ModelConfigurationManager',
    # Deployment
    'ONNXExporter',
    'create_onnx_exporter',
    'export_model_to_onnx',
    'FUAInferenceServer',
    'create_inference_server',
    # Pipeline
    'BioAstDataProcessor',
    'create_data_processor',
    'DataPipeline',
    'ProcessingMode',
    'QualityLevel',
    'ProcessingResult',
    'DatasetStats',
    # Optimization
    'HyperparameterOptimizer',
    'CrossValidationOptimizer',
    'OptimizationResult',
    'TrialResult',
    'create_hyperparameter_optimizer',
    'create_cv_optimizer',
    'get_default_search_space',
    # Fine-tuning
    'LayeredLRScheduler',
    'DifferentialLearningRateFinder',
    'create_layered_scheduler',
    'create_lr_finder',
    'get_resnet_layer_groups',
    'get_vit_layer_groups',
    'get_efficientnet_layer_groups',
    'LossFunctionFactory',
    'create_loss',
    'register_custom_loss',
    'get_classification_loss_configs',
    'get_imbalanced_loss_configs',
    'get_metric_learning_loss_configs',
    'get_segmentation_loss_configs',
    'ArchitectureModifier',
    'create_architecture_modifier',
    'FineTuningMonitor',
    'TrainingMetrics',
    'GradientMonitor',
    'ActivationMonitor',
    'create_finetuning_monitor',
    'FineTuningConfig',
    'FineTuningResult',
    'AdvancedFineTuner',
    'create_advanced_finetuner',
    'get_default_finetuning_config',
    # Automation Engine
    'AutomationEngine',
    'AutomationResult',
    'AutomationStatus',
    'create_automation_engine',
    'ErrorDetector',
    'ErrorCategory',
    'ErrorSeverity',
    'create_error_detector',
    'RootCauseAnalyzer',
    'AnalysisResult',
    'AnalysisConfidence',
    'create_root_cause_analyzer',
    'ImprovementGenerator',
    'ImprovementType',
    'ImprovementPriority',
    'create_improvement_generator',
    'FastValidator',
    'ValidationResult',
    'ValidationLevel',
    'create_fast_validator',
    # Model Integration
    'ModelIntegrator',
    'ModelEvaluator',
    'ModelSelector',
    'ModelDeployer',
    'ModelMetadata',
    'ModelFormat',
    'ModelStatus',
    'EvaluationResult',
    'EvaluationMetrics',
    'BenchmarkSuite',
    'EvaluationType',
    'SelectionCriteria',
    'SelectionStrategy',
    'SelectionResult',
    'SelectionWeights',
    'SelectionConstraint',
    'DeploymentConfig',
    'DeploymentResult',
    'DeploymentStatus',
    'DeploymentPlatform',
    'DeploymentFormat',
    'OptimizationLevel',
    'DeploymentMetrics',
    'create_model_integrator',
    'create_model_evaluator',
    'create_model_selector',
    'create_model_deployer',
    # Production
    'ModelMonitor',
    'MetricsCollector',
    'AnomalyDetector',
    'AlertManager',
    'Alert',
    'AlertSeverity',
    'MetricType',
    'AlertChannel',
    'MetricThreshold',
    'ModelMetrics',
    'EmailNotifier',
    'SlackNotifier',
    'WebhookNotifier',
    'create_model_monitor',
    # Flags
    'DEPLOYMENT_AVAILABLE',
    'PIPELINE_AVAILABLE',
    'OPTIMIZATION_AVAILABLE',
    'FINETUNING_AVAILABLE',
    'AUTOMATION_AVAILABLE',
    'MODEL_INTEGRATION_AVAILABLE',
    'PRODUCTION_AVAILABLE'
]