import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    from torchvision.models import resnet34, ResNet34_Weights
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False
    print("警告: ResNet34 不可用")


class ResNet34MultiTask(nn.Module):
    """ResNet-34多任务模型 - GPU优化版本"""
    
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
        
        # 创建ResNet-34 backbone
        self._create_backbone(use_pretrained, freeze_backbone)
        
        # 特征维度
        self.feature_dim = 512  # ResNet-34 output
        
        # 增强的特征处理器 (利用更大显存)
        self.feature_processor = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),  # 增大特征维度
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3)
        )
        
        # 多尺度特征提取 (GPU优化)
        if use_attention:
            self.multi_scale_extractor = MultiScaleFeatureExtractor(512)
            feature_dim_after_ms = 512
        else:
            self.multi_scale_extractor = None
            feature_dim_after_ms = 512
        
        # 跨任务注意力机制 (增强版)
        if use_attention:
            self.cross_task_attention = EnhancedCrossTaskAttention(
                d_model=feature_dim_after_ms, 
                num_heads=8,  # 增加注意力头数
                num_tasks=len(num_classes)
            )
        else:
            self.cross_task_attention = None
        
        # 任务特定分类头 (增强设计)
        self.task_heads = nn.ModuleDict()
        for task_name, num_cls in num_classes.items():
            if task_name == 'interference_factors':
                # 多标签任务
                self.task_heads[task_name] = EnhancedTaskHead(
                    feature_dim_after_ms, num_cls, 
                    use_sigmoid=False,  # 使用BCEWithLogitsLoss
                    dropout_rate=dropout_rate*0.2,
                    hidden_dims=[256, 128]  # 多层设计
                )
            elif task_name == 'growth_pattern':
                # 复杂分类任务
                self.task_heads[task_name] = EnhancedTaskHead(
                    feature_dim_after_ms, num_cls,
                    use_sigmoid=False,
                    dropout_rate=dropout_rate*0.2,
                    hidden_dims=[256, 128]
                )
            else:
                # 标准分类任务
                self.task_heads[task_name] = EnhancedTaskHead(
                    feature_dim_after_ms, num_cls,
                    use_sigmoid=False,
                    dropout_rate=dropout_rate*0.2,
                    hidden_dims=[128, 64]
                )
        
        # 自适应任务权重学习器
        self.task_weights = nn.Parameter(torch.ones(len(num_classes)))
        
        # 置信度预测头
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim_after_ms, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_backbone(self, use_pretrained: bool, freeze_backbone: bool):
        """创建ResNet-34 backbone"""
        if RESNET_AVAILABLE:
            if use_pretrained:
                weights = ResNet34_Weights.IMAGENET1K_V1
            else:
                weights = None
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.backbone = resnet34(weights=weights)
            
            # 修改第一层以适应灰度图输入
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # 移除分类头，保留特征提取部分
            self.backbone.fc = nn.Identity()
            
        else:
            raise RuntimeError("ResNet34 不可用")
        
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
        
        # 多尺度特征提取
        if self.multi_scale_extractor:
            multi_scale_features = self.multi_scale_extractor(processed_features)
        else:
            multi_scale_features = processed_features
        
        # 跨任务注意力
        if self.cross_task_attention:
            attended_features = self.cross_task_attention(multi_scale_features)
        else:
            attended_features = multi_scale_features
        
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
        
        # Growth Pattern - 12分类
        if 'growth_pattern' in predictions and 'growth_pattern' in targets:
            criterion = FocalLoss(alpha=1.0, gamma=2.0)  # 处理类别不平衡
            individual_losses['growth_pattern'] = criterion(
                predictions['growth_pattern'], targets['growth_pattern']
            )
        
        # Interference Factors - 多标签
        if 'interference_factors' in predictions and 'interference_factors' in targets:
            criterion = nn.BCEWithLogitsLoss()
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
        
        # 自适应加权损失
        weights = F.softmax(self.task_weights, dim=0)
        loss_values = list(individual_losses.values())
        
        total_loss = torch.tensor(0.0, device=loss_values[0].device, requires_grad=True)
        for i, loss in enumerate(loss_values):
            if i < len(weights):
                total_loss = total_loss + weights[i] * loss
            else:
                total_loss = total_loss + loss  # confidence任务
        
        return total_loss, individual_losses


class MultiScaleFeatureExtractor(nn.Module):
    """GPU优化的多尺度特征提取器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        # 并行多尺度分支
        self.scale_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.BatchNorm1d(feature_dim // 2),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.BatchNorm1d(feature_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 4, feature_dim // 2),
                nn.BatchNorm1d(feature_dim // 2),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 2 + feature_dim // 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始特征
        features = [x]
        
        # 多尺度特征
        for branch in self.scale_branches:
            features.append(branch(x))
        
        # 特征拼接和融合
        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


class EnhancedCrossTaskAttention(nn.Module):
    """增强的跨任务注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, num_tasks: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        
        # 多头注意力
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 层归一化和dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # 任务特定位置编码
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, d_model) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 添加任务位置编码
        # 这里简化处理，实际可以根据任务类型选择不同编码
        x_with_pos = x + self.task_embeddings[0]  # 使用第一个任务编码
        
        # 多头注意力
        Q = self.W_q(x_with_pos).view(batch_size, self.num_heads, self.d_k)
        K = self.W_k(x_with_pos).view(batch_size, self.num_heads, self.d_k)
        V = self.W_v(x_with_pos).view(batch_size, self.num_heads, self.d_k)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, self.d_model)
        
        # 输出投影
        output = self.W_o(attended)
        
        # 残差连接和层归一化
        return self.layer_norm(x + output)


class EnhancedTaskHead(nn.Module):
    """增强的任务特定分类头"""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 use_sigmoid: bool = False, 
                 dropout_rate: float = 0.2, 
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        
        # 构建多层网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        if use_sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        if self.activation:
            return self.activation(logits)
        return logits


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


class FocalLoss(nn.Module):
    """Focal Loss处理类别不平衡"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_resnet34_multitask(num_classes: Dict[str, int], **kwargs):
    """创建ResNet-34多任务模型"""
    return ResNet34MultiTask(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # 测试模型
    num_classes = {
        'growth_level': 2,
        'growth_pattern': 12,
        'interference_factors': 4,
        'microbe_type': 4
    }
    
    model = create_resnet34_multitask(num_classes)
    
    # 测试前向传播
    x = torch.randn(4, 1, 70, 70)  # 增大batch size测试
    outputs = model(x)
    
    print("ResNet-34多任务模型输出形状:")
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
        'growth_level': torch.randint(0, 2, (4,)),
        'growth_pattern': torch.randint(0, 12, (4,)),
        'interference_factors': torch.randint(0, 2, (4, 4)).float(),
        'microbe_type': torch.randint(0, 4, (4,)),
        'confidence': torch.rand(4)
    }
    
    loss, individual_losses = model.compute_loss(outputs, targets)
    print(f"\n总损失: {loss.item():.4f}")
    print("各任务损失:")
    for task, loss_val in individual_losses.items():
        print(f"  {task}: {loss_val.item():.4f}")