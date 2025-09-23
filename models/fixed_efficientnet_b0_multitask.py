import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False
    print("警告: efficientnet_pytorch 未安装，将使用torchvision版本")
    from torchvision.models import efficientnet_b0


class FixedEfficientNetB0MultiTask(nn.Module):
    """修复的EfficientNet-B0多任务模型 - 解决训练问题"""
    
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
        
        # 创建backbone
        self._create_backbone(use_pretrained, freeze_backbone)
        
        # 特征维度
        self.feature_dim = 1280  # EfficientNet-B0 output
        
        # 特征处理层
        self.feature_processor = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # 共享特征层
        self.shared_features = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3)
        )
        
        # 任务特定分类头
        self.task_heads = nn.ModuleDict()
        for task_name, num_cls in num_classes.items():
            if task_name == 'interference_factors':
                # 多标签任务 - 简化设计避免数值问题
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate * 0.2),
                    nn.Linear(128, num_cls)
                    # 注意：不添加Sigmoid，在loss中使用BCEWithLogitsLoss
                )
            else:
                # 标准分类任务
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate * 0.2),
                    nn.Linear(128, num_cls)
                )
        
        # 简化的任务权重（避免复杂的自适应权重学习）
        self.register_parameter('task_weights', 
                              nn.Parameter(torch.ones(len(num_classes))))
        
        # 置信度预测头
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.2),
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
                    bias=False
                )
                
                # 移除分类头
                self.backbone._fc = nn.Identity()
                
            except Exception as e:
                print(f"EfficientNet加载失败: {e}，使用torchvision版本")
                self._create_torchvision_backbone(use_pretrained)
        else:
            self._create_torchvision_backbone(use_pretrained)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def _create_torchvision_backbone(self, use_pretrained: bool):
        """使用torchvision的EfficientNet-B0"""
        if use_pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = efficientnet_b0(weights=weights)
        
        # 修改第一层以适应灰度图输入
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        # 移除分类头
        self.backbone.classifier = nn.Identity()
    
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
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Feature processing
        processed_features = self.feature_processor(features)
        shared_features = self.shared_features(processed_features)
        
        # 任务预测
        predictions = {}
        for task_name, head in self.task_heads.items():
            predictions[task_name] = head(shared_features)
        
        # 置信度预测
        predictions['confidence'] = self.confidence_head(shared_features)
        
        return predictions
    
    def compute_loss(self, 
                     predictions: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务损失 - 简化版本避免数值问题
        """
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
            criterion = nn.CrossEntropyLoss()
            individual_losses['growth_pattern'] = criterion(
                predictions['growth_pattern'], targets['growth_pattern']
            )
        
        # Interference Factors - 多标签（关键修复）
        if 'interference_factors' in predictions and 'interference_factors' in targets:
            # 使用更稳定的损失函数
            criterion = nn.BCEWithLogitsLoss()  # 内置sigmoid，更数值稳定
            individual_losses['interference_factors'] = criterion(
                predictions['interference_factors'], 
                targets['interference_factors'].float()  # 确保目标是float类型
            )
        
        # Microbe Type - 标准分类
        if 'microbe_type' in predictions and 'microbe_type' in targets:
            criterion = nn.CrossEntropyLoss()
            individual_losses['microbe_type'] = criterion(
                predictions['microbe_type'], targets['microbe_type']
            )
        
        # Confidence - 回归任务
        if 'confidence' in predictions and 'confidence' in targets:
            criterion = nn.MSELoss()  # 使用MSE而不是SmoothL1Loss
            individual_losses['confidence'] = criterion(
                predictions['confidence'].squeeze(), 
                targets['confidence'].float()
            )
        
        # 简化的加权损失计算
        weights = F.softmax(self.task_weights, dim=0)
        loss_values = list(individual_losses.values())
        
        total_loss = torch.tensor(0.0, device=loss_values[0].device, requires_grad=True)
        for i, loss in enumerate(loss_values):
            if i < len(weights):
                total_loss = total_loss + weights[i] * loss
            else:
                total_loss = total_loss + loss  # confidence任务权重为1
        
        return total_loss, individual_losses


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


def create_fixed_efficientnet_b0_multitask(num_classes: Dict[str, int], **kwargs):
    """创建修复的EfficientNet-B0多任务模型"""
    return FixedEfficientNetB0MultiTask(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # 测试模型
    num_classes = {
        'growth_level': 2,
        'growth_pattern': 12,
        'interference_factors': 4,
        'microbe_type': 4
    }
    
    model = create_fixed_efficientnet_b0_multitask(num_classes)
    
    # 测试前向传播
    x = torch.randn(2, 1, 70, 70)
    outputs = model(x)
    
    print("模型输出形状:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")
    
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