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

# Fine-tuning components (Sprint 4)
try:
    from .finetuning.layered_lr_scheduler import (
        LayeredLRScheduler, DifferentialLearningRateFinder,
        create_layered_scheduler, create_lr_finder,
        get_resnet_layer_groups, get_vit_layer_groups, get_efficientnet_layer_groups
    )
    from .finetuning.loss_function_factory import (
        LossFunctionFactory, create_loss, register_custom_loss,
        get_classification_loss_configs, get_imbalanced_loss_configs,
        get_metric_learning_loss_configs, get_segmentation_loss_configs
    )
    from .finetuning.architecture_modifier import (
        ArchitectureModifier, create_architecture_modifier
    )
    from .finetuning.finetuning_monitor import (
        FineTuningMonitor, TrainingMetrics, GradientMonitor, ActivationMonitor,
        create_finetuning_monitor
    )
    from .finetuning.advanced_finetuner import (
        FineTuningConfig, FineTuningResult, AdvancedFineTuner,
        create_advanced_finetuner, get_default_finetuning_config
    )
    FINETUNING_AVAILABLE = True
except ImportError:
    FINETUNING_AVAILABLE = False
    # Layered LR Scheduler
    LayeredLRScheduler = None
    DifferentialLearningRateFinder = None
    create_layered_scheduler = None
    create_lr_finder = None
    get_resnet_layer_groups = None
    get_vit_layer_groups = None
    get_efficientnet_layer_groups = None
    # Loss Function Factory
    LossFunctionFactory = None
    create_loss = None
    register_custom_loss = None
    get_classification_loss_configs = None
    get_imbalanced_loss_configs = None
    get_metric_learning_loss_configs = None
    get_segmentation_loss_configs = None
    # Architecture Modifier
    ArchitectureModifier = None
    create_architecture_modifier = None
    # Fine-tuning Monitor
    FineTuningMonitor = None
    TrainingMetrics = None
    GradientMonitor = None
    ActivationMonitor = None
    create_finetuning_monitor = None
    # Advanced Fine-tuner
    FineTuningConfig = None
    FineTuningResult = None
    AdvancedFineTuner = None
    create_advanced_finetuner = None
    get_default_finetuning_config = None

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
    # Flags
    'DEPLOYMENT_AVAILABLE',
    'PIPELINE_AVAILABLE',
    'OPTIMIZATION_AVAILABLE',
    'FINETUNING_AVAILABLE'
]