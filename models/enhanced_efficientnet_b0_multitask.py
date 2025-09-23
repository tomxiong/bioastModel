"""
改进版EfficientNet-B0多任务学习模型
基于之前实验结果的分析，解决过拟合和任务不平衡问题
支持新的2分类growth_level和12类growth_pattern系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import sys
import os
import math

# 添加项目根路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    print("警告: efficientnet_pytorch 未安装，将使用torchvision版本")
    import torchvision.models as models
    EFFICIENTNET_AVAILABLE = False


class EnhancedEfficientNetB0MultiTask(nn.Module):
    """
    改进版EfficientNet-B0多任务学习模型
    
    主要改进：
    1. 解决过拟合问题 - 增强正则化和Dropout
    2. 优化多任务损失平衡 - 自适应权重和Focal Loss
    3. 改进特征融合 - 多尺度特征和注意力机制
    4. 适配新数据集 - 2分类growth_level，12类growth_pattern
    """
    
    def __init__(self, 
                 num_classes: Dict[str, int] = None,
                 dropout_rate: float = 0.4,
                 use_pretrained: bool = True,
                 freeze_backbone: bool = False,
                 use_attention: bool = True,
                 use_label_smoothing: bool = True,
                 feature_fusion: str = 'concat'):  # 'concat', 'add', 'attention'
        """
        Args:
            num_classes: 各任务的类别数
            dropout_rate: Dropout比率 (增加到0.4防止过拟合)
            use_pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结backbone参数
            use_attention: 是否使用跨任务注意力机制
            use_label_smoothing: 是否使用标签平滑
            feature_fusion: 特征融合方式
        """
        super(EnhancedEfficientNetB0MultiTask, self).__init__()
        
        # 默认类别数配置 (基于新数据集)
        if num_classes is None:
            num_classes = {
                'growth_level': 2,      # 简化为2分类
                'growth_pattern': 12,   # 扩展到12类
                'interference_factors': 4,  # 多标签
                'microbe_type': 4
            }
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_label_smoothing = use_label_smoothing
        self.feature_fusion = feature_fusion
        
        # 创建EfficientNet-B0 backbone
        self._create_backbone(use_pretrained, freeze_backbone)
        
        # 获取特征维度
        self.feature_dim = self._get_feature_dim()
        
        # 多尺度特征提取
        self.multi_scale_features = MultiScaleFeatureExtractor(self.feature_dim)
        
        # 共享特征处理层 (增强正则化)
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # 跨任务注意力机制
        if self.use_attention:
            self.cross_attention = EnhancedCrossAttention(
                d_model=256,
                num_heads=8,
                num_tasks=len(num_classes)
            )
        
        # 任务特定分类头 (改进设计)
        self.task_heads = nn.ModuleDict()
        for task_name, num_cls in num_classes.items():
            if task_name == 'interference_factors':
                # 多标签任务使用特殊设计
                self.task_heads[task_name] = EnhancedTaskHead(
                    256, num_cls, use_sigmoid=True, dropout_rate=dropout_rate*0.3
                )
            elif task_name == 'growth_pattern':
                # 12分类任务需要更强的表达能力
                self.task_heads[task_name] = EnhancedTaskHead(
                    256, num_cls, use_sigmoid=False, dropout_rate=dropout_rate*0.3,
                    hidden_dim=128  # 更大的隐藏层
                )
            else:
                # 标准分类任务
                self.task_heads[task_name] = EnhancedTaskHead(
                    256, num_cls, use_sigmoid=False, dropout_rate=dropout_rate*0.3
                )
        
        # 改进的自适应任务权重学习 (包含confidence任务)
        self.task_weights = EnhancedTaskWeightLearner(len(num_classes) + 1)  # +1 for confidence
        
        # 置信度预测头
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_backbone(self, use_pretrained: bool, freeze_backbone: bool):
        """创建EfficientNet-B0 backbone"""
        if EFFICIENTNET_AVAILABLE:
            try:
                if use_pretrained:
                    self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
                else:
                    self.backbone = EfficientNet.from_name('efficientnet-b0')
                
                # 修改第一层以适应灰度图输入
                original_conv = self.backbone._conv_stem
                self.backbone._conv_stem = nn.Conv2d(
                    1, original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias
                )
                
                # 复制预训练权重的第一个通道到新的单通道层
                if use_pretrained:
                    with torch.no_grad():
                        self.backbone._conv_stem.weight = nn.Parameter(
                            original_conv.weight.mean(dim=1, keepdim=True)
                        )
                
                # 移除分类头
                self.backbone._fc = nn.Identity()
                
            except Exception as e:
                print(f"EfficientNet创建失败: {e}, 使用torchvision版本")
                self._create_torchvision_backbone(use_pretrained)
        else:
            self._create_torchvision_backbone(use_pretrained)
        
        # 冻结backbone参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _create_torchvision_backbone(self, use_pretrained: bool):
        """创建torchvision版本的EfficientNet"""
        self.backbone = models.efficientnet_b0(pretrained=use_pretrained)
        
        # 修改第一层
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        if use_pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
        
        # 移除分类头
        self.backbone.classifier = nn.Identity()
    
    def _get_feature_dim(self) -> int:
        """获取backbone输出的特征维度"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 70, 70)
            features = self.backbone(dummy_input)
            return features.shape[1]
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for module in [self.shared_fc, self.confidence_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 特征提取
        features = self.backbone(x)  # [batch_size, feature_dim]
        
        # 多尺度特征提取
        if hasattr(self, 'multi_scale_features'):
            multi_scale_feat = self.multi_scale_features(features)
            features = features + multi_scale_feat  # 残差连接
        
        # 共享特征处理
        shared_features = self.shared_fc(features)  # [batch_size, 256]
        
        # 跨任务注意力
        if self.use_attention:
            attended_features = self.cross_attention(shared_features)
        else:
            attended_features = shared_features
        
        # 各任务预测
        outputs = {}
        for task_name, task_head in self.task_heads.items():
            outputs[task_name] = task_head(attended_features)
        
        # 置信度预测
        outputs['confidence'] = self.confidence_head(attended_features).squeeze(-1)
        
        return outputs
    
    def compute_loss(self, 
                     predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        改进的多任务损失计算
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            epoch: 当前训练epoch (用于动态调整)
            
        Returns:
            (total_loss, individual_losses)
        """
        individual_losses = {}
        
        # Growth Level - 2分类 (使用标签平滑)
        if 'growth_level' in predictions and 'growth_level' in targets:
            if self.use_label_smoothing:
                criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
            else:
                criterion = nn.CrossEntropyLoss()
            individual_losses['growth_level'] = criterion(
                predictions['growth_level'], targets['growth_level']
            )
        
        # Growth Pattern - 12分类 (使用Focal Loss处理类别不平衡)
        if 'growth_pattern' in predictions and 'growth_pattern' in targets:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
            individual_losses['growth_pattern'] = criterion(
                predictions['growth_pattern'], targets['growth_pattern']
            )
        
        # Interference Factors - 多标签 (使用加权BCE)
        if 'interference_factors' in predictions and 'interference_factors' in targets:
            # 动态权重：稀少类别给更高权重
            pos_weight = torch.tensor([3.0, 1.5, 2.0, 5.0], device=predictions['interference_factors'].device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            individual_losses['interference_factors'] = criterion(
                predictions['interference_factors'], targets['interference_factors']
            )
        
        # Microbe Type - 标准分类
        if 'microbe_type' in predictions and 'microbe_type' in targets:
            criterion = nn.CrossEntropyLoss()
            individual_losses['microbe_type'] = criterion(
                predictions['microbe_type'], targets['microbe_type']
            )
        
        # Confidence - 回归任务
        if 'confidence' in predictions and 'confidence' in targets:
            criterion = nn.SmoothL1Loss()  # 对异常值更鲁棒
            individual_losses['confidence'] = criterion(
                predictions['confidence'], targets['confidence']
            )
        
        # 自适应权重加权总损失
        loss_values = list(individual_losses.values())
        total_loss = self.task_weights(loss_values, epoch)
        
        return total_loss, individual_losses


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_conv = nn.Conv1d(feature_dim, feature_dim // 4, 1)
        self.fusion = nn.Linear(feature_dim + feature_dim // 4, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 全局特征
        global_feat = self.global_pool(x.unsqueeze(-1)).squeeze(-1)
        
        # 局部特征
        local_feat = self.local_conv(x.unsqueeze(-1)).squeeze(-1)
        
        # 特征融合
        combined = torch.cat([global_feat, local_feat], dim=1)
        return self.fusion(combined)


class EnhancedTaskHead(nn.Module):
    """改进的任务特定分类头"""
    
    def __init__(self, input_dim: int, num_classes: int, use_sigmoid: bool = False, 
                 dropout_rate: float = 0.2, hidden_dim: int = 64):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        if use_sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        return self.activation(logits)


class EnhancedCrossAttention(nn.Module):
    """改进的多头跨任务注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, num_tasks: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, d_model) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # 多头注意力
        Q = self.W_q(x).view(batch_size, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, self.num_heads, self.d_k)
        
        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, self.d_model)
        
        # 输出投影
        output = self.W_o(attended)
        
        # 残差连接和层归一化
        return self.layer_norm(x + output)


class EnhancedTaskWeightLearner(nn.Module):
    """改进的自适应任务权重学习器"""
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        # 使用可学习的权重而不是对数方差
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, losses: List[torch.Tensor], epoch: int = 0) -> torch.Tensor:
        """计算加权损失"""
        # 动态权重调整
        weights = F.softmax(self.task_weights / self.temperature, dim=0)
        
        # 加权求和
        weighted_loss = 0
        for i, loss in enumerate(losses):
            weighted_loss += weights[i] * loss
        
        return weighted_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def create_enhanced_efficientnet_b0_multitask(num_classes: Dict[str, int] = None, **kwargs) -> EnhancedEfficientNetB0MultiTask:
    """
    创建改进版EfficientNet-B0多任务模型
    
    Args:
        num_classes: 各任务的类别数
        **kwargs: 其他模型参数
        
    Returns:
        EnhancedEfficientNetB0MultiTask模型实例
    """
    if num_classes is None:
        num_classes = {
            'growth_level': 2,
            'growth_pattern': 12,
            'interference_factors': 4,
            'microbe_type': 4
        }
    
    model = EnhancedEfficientNetB0MultiTask(num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":
    # 测试模型
    print("创建改进版EfficientNet-B0多任务模型...")
    
    model = create_enhanced_efficientnet_b0_multitask()
    
    # 测试前向传播
    dummy_input = torch.randn(4, 1, 70, 70)
    
    print(f"输入形状: {dummy_input.shape}")
    
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print("输出形状:")
    for task_name, output in outputs.items():
        print(f"  {task_name}: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 测试损失计算
    dummy_targets = {
        'growth_level': torch.randint(0, 2, (4,)),
        'growth_pattern': torch.randint(0, 12, (4,)),
        'interference_factors': torch.randint(0, 2, (4, 4)).float(),
        'microbe_type': torch.randint(0, 4, (4,)),
        'confidence': torch.rand(4)
    }
    
    total_loss, individual_losses = model.compute_loss(outputs, dummy_targets)
    print(f"\n损失测试:")
    print(f"  总损失: {total_loss.item():.4f}")
    for task, loss in individual_losses.items():
        print(f"  {task}: {loss.item():.4f}")
    
    print("\n改进版EfficientNet-B0多任务模型创建成功！")