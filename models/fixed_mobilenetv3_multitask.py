import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    MOBILENET_AVAILABLE = True
except ImportError:
    MOBILENET_AVAILABLE = False
    print("警告: MobileNetV3 不可用")


class FixedMobileNetV3MultiTask(nn.Module):
    """修复的MobileNetV3多任务模型 - 适配新数据集"""
    
    def __init__(self, 
                 num_classes: Dict[str, int], 
                 dropout_rate: float = 0.3, 
                 use_attention: bool = True,
                 use_label_smoothing: bool = True,
                 freeze_backbone: bool = False,
                 use_pretrained: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_label_smoothing = use_label_smoothing
        self.use_attention = use_attention
        
        # 创建MobileNetV3 backbone
        self._create_backbone(use_pretrained, freeze_backbone)
        
        # 特征维度
        self.feature_dim = 960  # MobileNetV3-Large output
        
        # 特征处理器 (MobileNetV3已包含avgpool，直接处理扁平化特征)
        self.feature_processor = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # 轻量级特征增强
        if use_attention:
            self.attention_module = LightweightAttention(512)
            feature_dim_after_att = 512
        else:
            self.attention_module = None
            feature_dim_after_att = 512
        
        # 任务特定头 (专门针对interference_factors优化)
        self.task_heads = nn.ModuleDict()
        
        # Growth Level - 简单2分类
        self.task_heads['growth_level'] = nn.Sequential(
            nn.Linear(feature_dim_after_att, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(64, num_classes['growth_level'])
        )
        
        # Growth Pattern - 12分类
        self.task_heads['growth_pattern'] = nn.Sequential(
            nn.Linear(feature_dim_after_att, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(64, num_classes['growth_pattern'])
        )
        
        # Interference Factors - 关键优化！
        self.task_heads['interference_factors'] = nn.Sequential(
            nn.Linear(feature_dim_after_att, 256),  # 增大容量
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(128, num_classes['interference_factors'])
            # 不添加Sigmoid，使用BCEWithLogitsLoss
        )
        
        # Microbe Type - 4分类
        self.task_heads['microbe_type'] = nn.Sequential(
            nn.Linear(feature_dim_after_att, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(64, num_classes['microbe_type'])
        )
        
        # 简化的任务权重
        self.register_parameter('task_weights', 
                              nn.Parameter(torch.ones(len(num_classes))))
        
        # 置信度预测头
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim_after_att, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_backbone(self, use_pretrained: bool, freeze_backbone: bool):
        """创建MobileNetV3 backbone"""
        if MOBILENET_AVAILABLE:
            if use_pretrained:
                weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            else:
                weights = None
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.backbone = mobilenet_v3_large(weights=weights)
            
            # 修改第一层以适应灰度图输入
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # 移除分类头，但保留avgpool
            self.backbone.classifier = nn.Identity()
            
            # MobileNetV3-Large已包含avgpool和classifier
            
        else:
            raise RuntimeError("MobileNetV3 不可用")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # Backbone特征提取
        features = self.backbone(x)
        
        # 特征处理
        processed_features = self.feature_processor(features)
        
        # 注意力增强
        if self.attention_module:
            attended_features = self.attention_module(processed_features)
        else:
            attended_features = processed_features
        
        # 任务预测
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(attended_features)
        
        # 置信度预测
        predictions['confidence'] = self.confidence_head(attended_features)
        
        return predictions
    
    def compute_loss(self, 
                     predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算多任务损失"""
        individual_losses = {}
        
        # Growth Level - 2分类
        if 'growth_level' in predictions and 'growth_level' in targets:
            if self.use_label_smoothing:
                criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
            else:
                criterion = nn.CrossEntropyLoss()
            individual_losses['growth_level'] = criterion(
                predictions['growth_level'], targets['growth_level']
            )
        
        # Growth Pattern - 12分类 (使用类别权重)
        if 'growth_pattern' in predictions and 'growth_pattern' in targets:
            # 添加类别权重处理不平衡
            class_weights = torch.tensor([0.8, 1.2, 1.5, 2.0, 1.0, 1.3, 1.8, 1.5, 1.0, 1.2, 1.4, 2.5], 
                                       device=predictions['growth_pattern'].device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            individual_losses['growth_pattern'] = criterion(
                predictions['growth_pattern'], targets['growth_pattern']
            )
        
        # Interference Factors - 多标签 (关键优化)
        if 'interference_factors' in predictions and 'interference_factors' in targets:
            # 使用位置权重处理类别不平衡
            pos_weights = torch.tensor([0.5, 2.0, 8.0, 1.5], 
                                     device=predictions['interference_factors'].device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            individual_losses['interference_factors'] = criterion(
                predictions['interference_factors'], 
                targets['interference_factors'].float()
            )
        
        # Microbe Type - 标准分类
        if 'microbe_type' in predictions and 'microbe_type' in targets:
            criterion = nn.CrossEntropyLoss()
            individual_losses['microbe_type'] = criterion(
                predictions['microbe_type'], targets['microbe_type']
            )
        
        # Confidence - 回归任务
        if 'confidence' in predictions and 'confidence' in targets:
            criterion = nn.MSELoss()
            individual_losses['confidence'] = criterion(
                predictions['confidence'].squeeze(), 
                targets['confidence'].float()
            )
        
        # 自适应加权损失 (给困难任务更高权重)
        weights = F.softmax(self.task_weights, dim=0)
        # 手动调整权重，给interference_factors更高权重
        adjusted_weights = weights.clone()
        if len(individual_losses) >= 3:
            adjusted_weights[2] *= 2.0  # interference_factors权重翻倍
        adjusted_weights = F.softmax(adjusted_weights, dim=0)
        
        loss_values = list(individual_losses.values())
        
        total_loss = torch.tensor(0.0, device=loss_values[0].device, requires_grad=True)
        for i, loss in enumerate(loss_values):
            if i < len(adjusted_weights):
                total_loss = total_loss + adjusted_weights[i] * loss
            else:
                total_loss = total_loss + loss  # confidence任务
        
        return total_loss, individual_losses


class LightweightAttention(nn.Module):
    """轻量级注意力机制"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def create_fixed_mobilenetv3_multitask(num_classes: Dict[str, int], **kwargs):
    """创建修复的MobileNetV3多任务模型"""
    return FixedMobileNetV3MultiTask(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # 测试模型
    num_classes = {
        'growth_level': 2,
        'growth_pattern': 12,
        'interference_factors': 4,
        'microbe_type': 4
    }
    
    model = create_fixed_mobilenetv3_multitask(num_classes)
    
    # 测试前向传播
    x = torch.randn(2, 1, 70, 70)
    outputs = model(x)
    
    print("修复版MobileNetV3多任务模型输出形状:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 测试损失计算
    targets = {
        'growth_level': torch.randint(0, 2, (2,)),
        'growth_pattern': torch.randint(0, 12, (2,)),
        'interference_factors': torch.randint(0, 2, (2, 4)).float(),
        'microbe_type': torch.randint(0, 4, (2,)),
        'confidence': torch.rand(2)
    }
    
    loss, individual_losses = model.compute_loss(outputs, targets)
    print(f"\n总损失: {loss.item():.4f}")
    print("各任务损失:")
    for task, loss_val in individual_losses.items():
        print(f"  {task}: {loss_val.item():.4f}")