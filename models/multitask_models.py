"""
多任务生物图像分类模型
基于现有模型架构适配多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import math

# 导入现有模型作为backbone
from models.airbubble_hybrid_net import create_airbubble_hybrid_net
from models.resnet_improved import create_resnet18_improved
from models.efficientnet import create_efficientnet_b0
from models.mobilenet_v3 import create_mobilenetv3_large, create_mobilenetv3_small
from models.efficientnet_v2_multitask import create_efficientnet_v2_s, create_efficientnet_v2_b0
from models.mic_mobilenetv3 import create_mic_mobilenetv3
from models.enhanced_multitask_mobilenetv3 import create_enhanced_multitask_mobilenetv3
from models.multitask_gray_colony_net import create_multitask_gray_colony_net


class MultitaskHead(nn.Module):
    """多任务头部模块"""
    
    def __init__(self, 
                 in_features: int,
                 task_configs: Dict[str, Dict],
                 dropout_rate: float = 0.2):
        super().__init__()
        self.task_configs = task_configs
        self.dropout_rate = dropout_rate
        
        # 为每个任务创建独立的头部
        self.heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            num_classes = config['num_classes']
            is_multilabel = config.get('multilabel', False)
            
            # 创建任务特定的头部
            if is_multilabel:
                # 多标签分类头部
                self.heads[task_name] = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, in_features // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features // 2, num_classes)
                )
            else:
                # 单标签分类头部
                self.heads[task_name] = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, in_features // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features // 2, num_classes)
                )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        for task_name, head in self.heads.items():
            outputs[task_name] = head(x)
        
        return outputs


class AttentionFusion(nn.Module):
    """注意力融合模块"""
    
    def __init__(self, feature_dim: int, reduction: int = 8):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, feature_dim, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class MultitaskBioastModel(nn.Module):
    """多任务生物图像分类模型"""
    
    def __init__(self,
                 backbone_name: str = 'airbubble_hybrid_net',
                 task_configs: Optional[Dict[str, Dict]] = None,
                 feature_dim: int = 576,
                 dropout_rate: float = 0.2,
                 use_attention: bool = True):
        """
        Args:
            backbone_name: 骨干网络名称
            task_configs: 任务配置
            feature_dim: 特征维度
            dropout_rate: Dropout率
            use_attention: 是否使用注意力机制
        """
        super().__init__()
        
        # 默认任务配置
        self.task_configs = task_configs or {
            'growth_level': {
                'num_classes': 3,
                'multilabel': False,
                'weight': 1.0
            },
            'growth_pattern': {
                'num_classes': 9,
                'multilabel': False,
                'weight': 1.0
            },
            'interference_mapping': {
                'num_classes': 3,
                'multilabel': True,
                'weight': 0.5
            },
            'fine_grained': {
                'num_classes': 40,
                'multilabel': False,
                'weight': 1.0
            }
        }
        
        # 创建骨干网络
        self.backbone_name = backbone_name
        self.backbone = self._create_backbone(backbone_name)
        self.feature_dim = feature_dim
        
        # 特征适配层
        self.feature_adapter = self._create_feature_adapter()
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionFusion(feature_dim)
        
        # 多任务头部
        self.multitask_head = MultitaskHead(
            in_features=feature_dim,
            task_configs=self.task_configs,
            dropout_rate=dropout_rate
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_backbone(self, backbone_name: str) -> nn.Module:
        """创建骨干网络"""
        if backbone_name == 'airbubble_hybrid_net':
            model = create_airbubble_hybrid_net(num_classes=1000)  # 临时使用1000类
            # 移除最后的分类层
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
            elif hasattr(model, 'fc'):
                model.fc = nn.Identity()
        elif backbone_name == 'resnet18_improved':
            model = create_resnet18_improved(num_classes=1000)
            if hasattr(model, 'fc'):
                model.fc = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            model = create_efficientnet_b0(num_classes=1000)
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
        elif backbone_name == 'mobilenetv3_large':
            model = create_mobilenetv3_large(num_classes=1000)
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
        elif backbone_name == 'mobilenetv3_small':
            model = create_mobilenetv3_small(num_classes=1000)
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
        elif backbone_name == 'efficientnet_v2_s':
            model = create_efficientnet_v2_s(num_classes=1000)
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
        elif backbone_name == 'efficientnet_v2_b0':
            model = create_efficientnet_v2_b0(num_classes=1000)
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
        elif backbone_name == 'mic_mobilenetv3':
            # MIC MobileNetV3 has special handling
            model = create_mic_mobilenetv3(num_classes=1000)
            # Remove classification head but keep other outputs
            if hasattr(model, 'classifier'):
                model.classifier = nn.Identity()
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")
        
        return model
    
    def _create_feature_adapter(self) -> nn.Module:
        """创建特征适配层"""
        # 根据骨干网络输出调整特征维度
        adapter = nn.Sequential(
            nn.Conv2d(self._get_backbone_out_channels(), self.feature_dim, 1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True)
        )
        return adapter
    
    def _get_backbone_out_channels(self) -> int:
        """获取骨干网络输出通道数"""
        # 这里需要根据实际的骨干网络架构返回正确的输出通道数
        # 临时返回一个默认值
        if 'efficientnet' in self.backbone_name:
            return 1280
        elif 'resnet' in self.backbone_name:
            return 512
        else:
            return 576  # 默认值
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取特征
        features = self.backbone(x)
        
        # 如果骨干网络返回的是分类结果，需要获取中间特征
        if isinstance(features, torch.Tensor) and features.dim() == 2:
            # 假设是分类结果，需要从骨干网络获取特征图
            features = self._extract_features(x)
        
        # 特征适配
        features = self.feature_adapter(features)
        
        # 应用注意力
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        # 全局池化
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = features.flatten(1)
        
        # 多任务预测
        outputs = self.multitask_head(features)
        
        return outputs
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """从骨干网络提取特征图"""
        # 这是一个简化实现，实际需要根据具体的骨干网络架构来获取特征图
        # 这里使用一个卷积层模拟特征提取
        x = F.relu(self.backbone.conv1(x))
        x = self.backbone.bn1(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        return x
    
    def get_task_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        return {
            'backbone': self.backbone_name,
            'feature_dim': self.feature_dim,
            'tasks': self.task_configs,
            'use_attention': self.use_attention,
            'total_params': sum(p.numel() for p in self.parameters())
        }
    
    def freeze_backbone(self):
        """冻结骨干网络参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻骨干网络参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class HierarchicalMultitaskModel(nn.Module):
    """分层多任务模型"""
    
    def __init__(self,
                 backbone_name: str = 'airbubble_hybrid_net',
                 task_configs: Optional[Dict[str, Dict]] = None):
        """
        分层多任务模型架构：
        1. 生长级别 -> 基础分类
        2. 生长模式 -> 依赖于生长级别
        3. 干扰因素 -> 独立任务
        4. 精细分类 -> 融合所有任务
        """
        super().__init__()
        
        self.task_configs = task_configs or {
            'growth_level': {'num_classes': 3, 'multilabel': False},
            'growth_pattern': {'num_classes': 9, 'multilabel': False},
            'interference_mapping': {'num_classes': 3, 'multilabel': True},
            'fine_grained': {'num_classes': 40, 'multilabel': False}
        }
        
        # 共享骨干网络
        self.backbone = self._create_backbone(backbone_name)
        
        # 共享特征提取器
        self.shared_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True)
        )
        
        # 任务特定分支
        self.growth_level_head = self._create_task_head(256, 3)
        
        # 生长模式分支（考虑生长级别信息）
        self.growth_pattern_head = nn.Sequential(
            nn.Linear(256 + 3, 128),  # 256特征 + 3生长级别
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 9)
        )
        
        # 干扰因素分支（独立）
        self.interference_head = self._create_task_head(256, 3)
        
        # 精细分类分支（融合所有信息）
        self.fine_grained_head = nn.Sequential(
            nn.Linear(256 + 3 + 9 + 3, 512),  # 特征 + 所有任务输出
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 40)
        )
    
    def _create_backbone(self, backbone_name: str) -> nn.Module:
        """创建骨干网络"""
        # 同上实现
        if backbone_name == 'airbubble_hybrid_net':
            return create_airbubble_hybrid_net(num_classes=1000)
        elif backbone_name == 'resnet18_improved':
            return create_resnet18_improved(num_classes=1000)
        elif backbone_name == 'efficientnet_b0':
            return create_efficientnet_b0(num_classes=1000)
        else:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")
    
    def _create_task_head(self, in_features: int, num_classes: int) -> nn.Module:
        """创建任务头部"""
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取共享特征
        features = self.backbone(x)
        shared_features = self.shared_features(features)
        
        # 生长级别预测
        growth_level_logits = self.growth_level_head(shared_features)
        growth_level_probs = F.softmax(growth_level_logits, dim=1)
        
        # 生长模式预测（使用生长级别信息）
        gp_input = torch.cat([shared_features, growth_level_probs], dim=1)
        growth_pattern_logits = self.growth_pattern_head(gp_input)
        
        # 干扰因素预测
        interference_logits = self.interference_head(shared_features)
        
        # 精细分类预测（融合所有信息）
        fg_input = torch.cat([
            shared_features,
            growth_level_probs,
            F.softmax(growth_pattern_logits, dim=1),
            torch.sigmoid(interference_logits)
        ], dim=1)
        fine_grained_logits = self.fine_grained_head(fg_input)
        
        return {
            'growth_level': growth_level_logits,
            'growth_pattern': growth_pattern_logits,
            'interference_mapping': interference_logits,
            'fine_grained': fine_grained_logits
        }


def create_multitask_model(model_type: str = 'standard',
                          backbone_name: str = 'airbubble_hybrid_net',
                          **kwargs) -> nn.Module:
    """创建多任务模型的工厂函数"""
    
    if model_type == 'standard':
        return MultitaskBioastModel(
            backbone_name=backbone_name,
            **kwargs
        )
    elif model_type == 'hierarchical':
        return HierarchicalMultitaskModel(
            backbone_name=backbone_name,
            **kwargs
        )
    elif model_type == 'enhanced':
        # 特殊处理增强版多任务模型
        from models.enhanced_multitask_mobilenetv3 import create_enhanced_multitask_mobilenetv3
        return create_enhanced_multitask_mobilenetv3(**kwargs)
    elif model_type == 'multitask_gray':
        # 多任务灰度菌落检测网络
        return create_multitask_gray_colony_net(**kwargs)
    else:
        raise ValueError(f"不支持的多任务模型类型: {model_type}")


# 模型配置
MULTITASK_MODEL_CONFIGS = {
    'multitask_airbubble_hybrid': {
        'model_type': 'standard',
        'backbone_name': 'airbubble_hybrid_net',
        'feature_dim': 576,
        'dropout_rate': 0.2,
        'use_attention': True,
        'description': '基于AirBubble HybridNet的多任务模型'
    },
    'multitask_resnet18': {
        'model_type': 'standard',
        'backbone_name': 'resnet18_improved',
        'feature_dim': 512,
        'dropout_rate': 0.3,
        'use_attention': True,
        'description': '基于ResNet18的多任务模型'
    },
    'multitask_efficientnet': {
        'model_type': 'standard',
        'backbone_name': 'efficientnet_b0',
        'feature_dim': 1280,
        'dropout_rate': 0.3,
        'use_attention': True,
        'description': '基于EfficientNet-B0的多任务模型'
    },
    'multitask_mobilenetv3_large': {
        'model_type': 'standard',
        'backbone_name': 'mobilenetv3_large',
        'feature_dim': 1280,
        'dropout_rate': 0.3,
        'use_attention': True,
        'description': '基于MobileNetV3-Large的多任务模型'
    },
    'multitask_mobilenetv3_small': {
        'model_type': 'standard',
        'backbone_name': 'mobilenetv3_small',
        'feature_dim': 1024,
        'dropout_rate': 0.2,
        'use_attention': True,
        'description': '基于MobileNetV3-Small的多任务模型'
    },
    'multitask_efficientnet_v2_s': {
        'model_type': 'standard',
        'backbone_name': 'efficientnet_v2_s',
        'feature_dim': 1280,
        'dropout_rate': 0.3,
        'use_attention': True,
        'description': '基于EfficientNetV2-S的多任务模型'
    },
    'multitask_efficientnet_v2_b0': {
        'model_type': 'standard',
        'backbone_name': 'efficientnet_v2_b0',
        'feature_dim': 1280,
        'dropout_rate': 0.3,
        'use_attention': True,
        'description': '基于EfficientNetV2-B0的多任务模型'
    },
    'multitask_mic_mobilenetv3': {
        'model_type': 'standard',
        'backbone_name': 'mic_mobilenetv3',
        'feature_dim': 576,
        'dropout_rate': 0.2,
        'use_attention': True,
        'description': '基于MIC MobileNetV3的多任务模型'
    },
    'enhanced_multitask_mobilenetv3': {
        'model_type': 'enhanced',
        'backbone_name': 'mobilenetv3_small',
        'feature_dim': 576,
        'dropout_rate': 0.2,
        'use_attention': True,
        'description': '增强版MobileNetV3多任务模型，内置多任务支持'
    },
    'hierarchical_airbubble': {
        'model_type': 'hierarchical',
        'backbone_name': 'airbubble_hybrid_net',
        'description': '分层多任务AirBubble模型'
    },
    'multitask_gray_colony': {
        'model_type': 'multitask_gray',
        'feature_dim': 128,
        'dropout_rate': 0.2,
        'enable_background_filter': True,
        'description': '多任务灰度菌落检测网络，专精于灰度图像的4层标注任务'
    }
}


def get_multitask_model_config(model_name: str) -> Dict:
    """获取多任务模型配置"""
    if model_name in MULTITASK_MODEL_CONFIGS:
        return MULTITASK_MODEL_CONFIGS[model_name]
    else:
        raise ValueError(f"未知的多任务模型: {model_name}")


# 使用示例
if __name__ == "__main__":
    # 创建标准多任务模型
    model = create_multitask_model(
        model_type='standard',
        backbone_name='airbubble_hybrid_net'
    )
    
    # 创建分层多任务模型
    hierarchical_model = create_multitask_model(
        model_type='hierarchical',
        backbone_name='resnet18_improved'
    )
    
    # 测试前向传播
    dummy_input = torch.randn(2, 3, 70, 70)
    
    print("标准多任务模型:")
    outputs = model(dummy_input)
    for task_name, output in outputs.items():
        print(f"{task_name}: {output.shape}")
    
    print("\n分层多任务模型:")
    outputs = hierarchical_model(dummy_input)
    for task_name, output in outputs.items():
        print(f"{task_name}: {output.shape}")
    
    # 打印模型信息
    print("\n模型信息:")
    print(model.get_task_info())