"""
模型定义模块
包含EfficientNet、ResNet改进版以及后续的ConvNeXt、CoAtNet等模型
"""

from .efficientnet import EfficientNetCustom
from .resnet_improved import ResNetImproved

__all__ = ['EfficientNetCustom', 'ResNetImproved']