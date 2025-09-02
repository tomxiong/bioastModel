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
    # Flags
    'DEPLOYMENT_AVAILABLE',
    'PIPELINE_AVAILABLE',
    'OPTIMIZATION_AVAILABLE'
]