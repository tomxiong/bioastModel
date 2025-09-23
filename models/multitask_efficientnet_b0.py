"""
基于EfficientNet-B0的多任务学习架构
按照multitask_network_selection.md的推荐实现
专门针对70x70灰度图像的4个相关任务设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, List, Optional


class MultiHeadAttention(nn.Module):
    """跨任务多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Self-attention
        residual = x
        q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) 
        v = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.w_o(out)
        
        # 残差连接和层归一化
        out = self.layer_norm(out + residual)
        return out


class TaskSpecificHead(nn.Module):
    """任务特定分类头"""
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        hidden_dim = max(input_dim // 4, 64)  # 至少64维
        
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        return self.head(x)


class HierarchicalTaskHead(nn.Module):
    """层次化任务头，融合上级任务信息"""
    def __init__(self, feature_dim: int, prev_task_dims: List[int], 
                 num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        
        total_input_dim = feature_dim + sum(prev_task_dims) if prev_task_dims else feature_dim
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 分类器
        hidden_dim = max(feature_dim // 4, 64)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, features: torch.Tensor, prev_outputs: Optional[List[torch.Tensor]] = None):
        # 融合特征和上级任务信息
        if prev_outputs:
            combined = torch.cat([features] + prev_outputs, dim=1)
        else:
            combined = features
            
        fused_features = self.fusion(combined)
        output = self.classifier(fused_features)
        return output


class MultiTaskEfficientNetB0(nn.Module):
    """
    基于EfficientNet-B0的多任务学习架构
    实现multitask_network_selection.md推荐的设计:
    - 共享特征提取器 (EfficientNet-B0)
    - 跨任务注意力机制
    - 层次化任务头部设计
    - 专为70×70灰度图优化
    """
    
    def __init__(self, 
                 num_classes_dict: Optional[Dict[str, int]] = None,
                 dropout_rate: float = 0.3,
                 use_attention: bool = True,
                 use_hierarchical: bool = True,
                 pretrained: bool = True):
        super().__init__()
        
        # 默认任务配置
        if num_classes_dict is None:
            num_classes_dict = {
                'growth_level': 3,      # 生长级别: negative, positive, weak_growth
                'growth_pattern': 9,    # 生长模式: 9种模式
                'interference_factors': 5,  # 干扰因素: 5种类型
                'fine_grained': 40      # 精细分类: 40个细分类
            }
            
        self.num_classes_dict = num_classes_dict
        self.use_attention = use_attention
        self.use_hierarchical = use_hierarchical
        
        # 共享特征提取器: EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # 修改第一层卷积以适应单通道灰度图 (70×70×1)
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,  # 灰度图单通道
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # 如果使用预训练权重，需要处理通道数不匹配
        if pretrained:
            # 将RGB权重转换为灰度权重(取平均)
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # 获取特征维度并移除原分类器
        if isinstance(self.backbone.classifier, nn.Sequential):
            # EfficientNet的classifier是Sequential，需要从Linear层获取
            for layer in self.backbone.classifier:
                if isinstance(layer, nn.Linear):
                    feature_dim = layer.in_features
                    break
            else:
                feature_dim = 1280  # EfficientNet-B0默认特征维度
        else:
            feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # 跨任务注意力机制
        if self.use_attention:
            self.cross_attention = MultiHeadAttention(feature_dim, num_heads=8, dropout=dropout_rate)
        
        # 共享特征处理层
        self.feature_processor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1) if feature_dim != 1280 else nn.Identity(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 构建任务头部
        if self.use_hierarchical:
            self._build_hierarchical_heads(feature_dim, dropout_rate)
        else:
            self._build_standard_heads(feature_dim, dropout_rate)
        
        # 初始化新增权重
        self._initialize_new_weights()
        
    def _build_hierarchical_heads(self, feature_dim: int, dropout_rate: float):
        """构建层次化任务头部"""
        # 层次1: 生长级别 (最基础任务)
        self.growth_level_head = TaskSpecificHead(
            feature_dim, self.num_classes_dict['growth_level'], dropout_rate
        )
        
        # 层次2: 生长模式 (基于生长级别)
        self.growth_pattern_head = HierarchicalTaskHead(
            feature_dim, [self.num_classes_dict['growth_level']],
            self.num_classes_dict['growth_pattern'], dropout_rate
        )
        
        # 独立任务: 干扰因素检测
        self.interference_head = TaskSpecificHead(
            feature_dim, self.num_classes_dict['interference_factors'], dropout_rate
        )
        
        # 层次3: 精细分类 (基于生长级别和生长模式)
        self.fine_grained_head = HierarchicalTaskHead(
            feature_dim, 
            [self.num_classes_dict['growth_level'], self.num_classes_dict['growth_pattern']],
            self.num_classes_dict['fine_grained'], dropout_rate
        )
        
    def _build_standard_heads(self, feature_dim: int, dropout_rate: float):
        """构建标准多任务头部"""
        self.task_heads = nn.ModuleDict({
            task_name: TaskSpecificHead(feature_dim, num_classes, dropout_rate)
            for task_name, num_classes in self.num_classes_dict.items()
        })
        
    def _initialize_new_weights(self):
        """初始化新增的权重"""
        modules_to_init = []
        
        if hasattr(self, 'cross_attention'):
            modules_to_init.append(self.cross_attention)
        modules_to_init.append(self.feature_processor)
        
        if self.use_hierarchical:
            modules_to_init.extend([
                self.growth_level_head, self.growth_pattern_head,
                self.interference_head, self.fine_grained_head
            ])
        else:
            modules_to_init.extend(list(self.task_heads.values()))
            
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取共享特征"""
        # EfficientNet-B0特征提取
        features = self.backbone(x)  # [batch_size, 1280]
        
        # 跨任务注意力增强
        if self.use_attention:
            # 为注意力计算添加序列维度
            features_expanded = features.unsqueeze(1)  # [batch_size, 1, 1280]
            enhanced_features = self.cross_attention(features_expanded)
            features = enhanced_features.squeeze(1)  # [batch_size, 1280]
        
        # 特征后处理
        processed_features = self.feature_processor(features)
        return processed_features
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 提取共享特征
        features = self.extract_features(x)
        
        outputs = {}
        
        if self.use_hierarchical:
            # 层次化推理: 遵循任务依赖关系
            
            # 步骤1: 生长级别分类 (基础任务)
            growth_level_logits = self.growth_level_head(features)
            outputs['growth_level'] = growth_level_logits
            
            # 步骤2: 生长模式分类 (条件依赖生长级别)
            growth_level_softmax = F.softmax(growth_level_logits, dim=1)
            growth_pattern_logits = self.growth_pattern_head(features, [growth_level_softmax])
            outputs['growth_pattern'] = growth_pattern_logits
            
            # 步骤3: 干扰因素检测 (独立任务)
            interference_logits = self.interference_head(features)
            outputs['interference_factors'] = interference_logits
            
            # 步骤4: 精细分类 (综合依赖前面的任务结果)
            growth_pattern_softmax = F.softmax(growth_pattern_logits, dim=1)
            fine_grained_logits = self.fine_grained_head(
                features, [growth_level_softmax, growth_pattern_softmax]
            )
            outputs['fine_grained'] = fine_grained_logits
            
        else:
            # 标准多任务推理: 所有任务并行
            for task_name, head in self.task_heads.items():
                outputs[task_name] = head(features)
                
        return outputs
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取中间特征图用于分析和可视化"""
        feature_maps = {}
        
        # 逐层提取EfficientNet特征
        current = x
        for i, layer in enumerate(self.backbone.features):
            current = layer(current)
            # 记录关键阶段的特征图
            if i in [0, 2, 4, 6, 8]:  # stem + 主要block
                feature_maps[f'stage_{i}'] = current.clone()
        
        return feature_maps
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 计算各部分参数数量
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        
        return {
            'model_name': 'MultiTaskEfficientNetB0',
            'architecture': 'EfficientNet-B0 + Multi-Task Heads',
            'input_size': (1, 70, 70),  # 灰度图
            'tasks': list(self.num_classes_dict.keys()),
            'num_classes_per_task': self.num_classes_dict,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_parameters': backbone_params,
            'task_head_parameters': total_params - backbone_params,
            'use_attention': self.use_attention,
            'use_hierarchical': self.use_hierarchical,
            'feature_dimension': 1280
        }


def create_multitask_efficientnet_b0(num_classes_dict: Optional[Dict[str, int]] = None, **kwargs):
    """
    创建多任务EfficientNet-B0模型的工厂函数
    
    Args:
        num_classes_dict: 各任务类别数字典
        **kwargs: 其他模型参数
        
    Returns:
        MultiTaskEfficientNetB0: 多任务模型实例
    """
    return MultiTaskEfficientNetB0(num_classes_dict=num_classes_dict, **kwargs)


# 预设配置的工厂函数
def create_multitask_efficientnet_b0_standard(**kwargs):
    """标准配置: 启用注意力和层次化"""
    # 提取pretrained参数，避免重复
    pretrained = kwargs.pop('pretrained', False)
    return create_multitask_efficientnet_b0(
        dropout_rate=0.3,
        use_attention=True,
        use_hierarchical=True,
        pretrained=pretrained,
        **kwargs
    )


def create_multitask_efficientnet_b0_lightweight(**kwargs):
    """轻量配置: 关闭注意力机制"""
    return create_multitask_efficientnet_b0(
        dropout_rate=0.2,
        use_attention=False,
        use_hierarchical=False,
        pretrained=True,
        **kwargs
    )


def create_multitask_efficientnet_b0_enhanced(**kwargs):
    """增强配置: 更高dropout和完整特性"""
    return create_multitask_efficientnet_b0(
        dropout_rate=0.4,
        use_attention=True,
        use_hierarchical=True,
        pretrained=True,
        **kwargs
    )


if __name__ == "__main__":
    # 模型测试
    print("=== MultiTask EfficientNet-B0 架构测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试不同配置
    configs = [
        ("标准配置", create_multitask_efficientnet_b0_standard),
        ("轻量配置", create_multitask_efficientnet_b0_lightweight),
        ("增强配置", create_multitask_efficientnet_b0_enhanced)
    ]
    
    for config_name, create_func in configs:
        print(f"\n--- {config_name} ---")
        
        # 创建模型
        model = create_func()
        model = model.to(device)
        model_info = model.get_model_info()
        
        print(f"总参数量: {model_info['total_parameters']:,}")
        print(f"主干网络参数: {model_info['backbone_parameters']:,}")
        print(f"任务头参数: {model_info['task_head_parameters']:,}")
        print(f"使用注意力机制: {model_info['use_attention']}")
        print(f"使用层次化设计: {model_info['use_hierarchical']}")
        
        # 前向传播测试
        batch_size = 4
        test_input = torch.randn(batch_size, 1, 70, 70).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_input)
            
            print(f"输入形状: {test_input.shape}")
            print("输出形状:")
            for task_name, output_tensor in outputs.items():
                print(f"  {task_name}: {output_tensor.shape}")
        
        print(f"✓ {config_name}测试通过")
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache()
    
    print(f"\n=== 所有配置测试完成 ===")