"""
FUA Fine-tuning Module

高级微调工具集，提供分层学习率、损失函数工厂、架构修改和监控功能
"""

from .layered_lr_scheduler import (
    LayeredLRScheduler, DifferentialLearningRateFinder,
    create_layered_scheduler, create_lr_finder,
    get_resnet_layer_groups, get_vit_layer_groups, get_efficientnet_layer_groups
)

from .loss_function_factory import (
    LossFunctionFactory, create_loss, register_custom_loss,
    get_classification_loss_configs, get_imbalanced_loss_configs,
    get_metric_learning_loss_configs, get_segmentation_loss_configs
)

from .architecture_modifier import (
    ArchitectureModifier, create_architecture_modifier
)

from .finetuning_monitor import (
    FineTuningMonitor, TrainingMetrics, GradientMonitor, ActivationMonitor,
    create_finetuning_monitor
)

from .advanced_finetuner import (
    FineTuningConfig, FineTuningResult, AdvancedFineTuner,
    create_advanced_finetuner, get_default_finetuning_config
)

__all__ = [
    # Layered LR Scheduler
    'LayeredLRScheduler',
    'DifferentialLearningRateFinder',
    'create_layered_scheduler',
    'create_lr_finder',
    'get_resnet_layer_groups',
    'get_vit_layer_groups',
    'get_efficientnet_layer_groups',
    # Loss Function Factory
    'LossFunctionFactory',
    'create_loss',
    'register_custom_loss',
    'get_classification_loss_configs',
    'get_imbalanced_loss_configs',
    'get_metric_learning_loss_configs',
    'get_segmentation_loss_configs',
    # Architecture Modifier
    'ArchitectureModifier',
    'create_architecture_modifier',
    # Fine-tuning Monitor
    'FineTuningMonitor',
    'TrainingMetrics',
    'GradientMonitor',
    'ActivationMonitor',
    'create_finetuning_monitor',
    # Advanced Fine-tuner
    'FineTuningConfig',
    'FineTuningResult',
    'AdvancedFineTuner',
    'create_advanced_finetuner',
    'get_default_finetuning_config'
]