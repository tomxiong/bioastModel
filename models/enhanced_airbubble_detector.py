"""
Enhanced Air Bubble Detection Module
专用气孔检测网络 - 基于物理模型的增强版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import cv2

class PhysicsBasedAugmentation:
    """基于物理模型的数据增强"""
    
    def __init__(self):
        self.optical_model = OpticalInterferenceSimulator()
        self.turbidity_model = TurbidityGradientGenerator()
        self.airbubble_model = AirBubblePhysicsSimulator()
    
    def apply_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """应用物理模型驱动的数据增强"""
        # 转换为numpy进行处理
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        
        # 应用气孔物理模拟
        img_with_bubbles = self.airbubble_model.simulate_bubbles(img_np)
        
        # 应用光学干扰模拟
        img_with_interference = self.optical_model.simulate_interference(img_with_bubbles)
        
        # 应用浊度梯度
        img_final = self.turbidity_model.apply_gradient(img_with_interference)
        
        return torch.from_numpy(img_final.transpose(2, 0, 1))

class OpticalInterferenceSimulator:
    """光学干扰模拟器"""
    
    def __init__(self):
        self.interference_patterns = self._generate_interference_patterns()
    
    def _generate_interference_patterns(self):
        """生成光学干扰模式"""
        patterns = []
        for i in range(10):
            # 生成不同频率的干涉条纹
            pattern = np.zeros((70, 70))
            freq = 0.1 + i * 0.05
            for x in range(70):
                for y in range(70):
                    pattern[x, y] = 0.1 * np.sin(2 * np.pi * freq * x) * np.cos(2 * np.pi * freq * y)
            patterns.append(pattern)
        return patterns
    
    def simulate_interference(self, image: np.ndarray) -> np.ndarray:
        """模拟光学干扰效应"""
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
            c = 1
            image = image.reshape(h, w, 1)
        
        # 随机选择干扰模式
        pattern_idx = np.random.randint(0, len(self.interference_patterns))
        pattern = self.interference_patterns[pattern_idx]
        
        # 应用干扰
        result = image.copy()
        for ch in range(c):
            result[:, :, ch] = np.clip(result[:, :, ch] + pattern, 0, 1)
        
        return result

class TurbidityGradientGenerator:
    """浊度梯度生成器"""
    
    def apply_gradient(self, image: np.ndarray) -> np.ndarray:
        """应用浊度梯度效果"""
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
            c = 1
            image = image.reshape(h, w, 1)
        
        # 生成径向梯度
        center_x, center_y = w // 2, h // 2
        gradient = np.zeros((h, w))
        
        for y in range(h):
            for x in range(w):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                gradient[y, x] = 1.0 - (dist / (max(h, w) / 2)) * 0.3
        
        # 应用梯度
        result = image.copy()
        for ch in range(c):
            result[:, :, ch] = result[:, :, ch] * gradient.reshape(h, w, 1)[:, :, 0]
        
        return np.clip(result, 0, 1)

class AirBubblePhysicsSimulator:
    """气孔物理模拟器"""
    
    def __init__(self):
        self.bubble_templates = self._generate_bubble_templates()
    
    def _generate_bubble_templates(self):
        """生成气孔模板"""
        templates = []
        sizes = [3, 5, 7, 9, 11]  # 不同大小的气孔
        
        for size in sizes:
            template = np.zeros((size, size))
            center = size // 2
            
            for y in range(size):
                for x in range(size):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    if dist <= center:
                        # 气孔内部较亮，边缘较暗
                        intensity = 1.0 - (dist / center) * 0.7
                        template[y, x] = intensity
            
            templates.append(template)
        
        return templates
    
    def simulate_bubbles(self, image: np.ndarray, num_bubbles: int = None) -> np.ndarray:
        """在图像中模拟气孔"""
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
            c = 1
            image = image.reshape(h, w, 1)
        
        if num_bubbles is None:
            num_bubbles = np.random.randint(0, 3)  # 0-2个气孔
        
        result = image.copy()
        
        for _ in range(num_bubbles):
            # 随机选择气孔模板
            template = self.bubble_templates[np.random.randint(0, len(self.bubble_templates))]
            t_h, t_w = template.shape
            
            # 随机位置
            start_y = np.random.randint(0, max(1, h - t_h))
            start_x = np.random.randint(0, max(1, w - t_w))
            
            # 应用气孔效果
            for ch in range(c):
                region = result[start_y:start_y+t_h, start_x:start_x+t_w, ch]
                result[start_y:start_y+t_h, start_x:start_x+t_w, ch] = np.maximum(region, template)
        
        return result

class AirBubbleAttentionModule(nn.Module):
    """气孔注意力模块"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 气孔特征检测器
        self.bubble_detector = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x_sa = x_ca * sa
        
        # 气孔检测
        bubble_map = self.bubble_detector(x_sa)
        
        return x_sa, bubble_map

class EnhancedAirBubbleDetector(nn.Module):
    """增强型气孔检测器"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 2):
        super().__init__()
        
        # 特征提取骨干网络
        self.backbone = nn.Sequential(
            # 第一层：保持空间分辨率
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 第二层：轻微下采样
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 35x35
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第三层：特征增强
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第四层：深层特征
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 18x18
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # 气孔注意力模块
        self.attention_module = AirBubbleAttentionModule(256)
        
        # 多尺度特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # 气孔定位头
        self.localization_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        # 不确定性量化
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus()  # 确保输出为正
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 特征提取
        features = self.backbone(x)
        
        # 注意力机制
        attended_features, bubble_attention = self.attention_module(features)
        
        # 特征融合
        fused_features = self.feature_fusion(attended_features)
        fused_features = fused_features.view(fused_features.size(0), -1)
        
        # 分类预测
        classification = self.classifier(fused_features)
        
        # 气孔定位
        localization = self.localization_head(attended_features)
        
        # 不确定性估计
        uncertainty = self.uncertainty_head(fused_features)
        
        return {
            'classification': classification,
            'localization': localization,
            'bubble_attention': bubble_attention,
            'uncertainty': uncertainty,
            'features': fused_features
        }

class AirBubbleLoss(nn.Module):
    """气孔检测专用损失函数"""
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 localization_weight: float = 0.5,
                 uncertainty_weight: float = 0.1):
        super().__init__()
        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
        self.uncertainty_weight = uncertainty_weight
        
        self.classification_loss = nn.CrossEntropyLoss()
        self.localization_loss = nn.BCELoss()
        self.uncertainty_loss = nn.MSELoss()
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # 分类损失
        cls_loss = self.classification_loss(
            predictions['classification'], 
            targets['labels']
        )
        
        # 定位损失（如果有定位标签）
        loc_loss = torch.tensor(0.0, device=predictions['classification'].device)
        if 'localization_maps' in targets:
            loc_loss = self.localization_loss(
                predictions['localization'], 
                targets['localization_maps']
            )
        
        # 不确定性损失（基于预测置信度）
        uncertainty_target = torch.ones_like(predictions['uncertainty']) * 0.1
        unc_loss = self.uncertainty_loss(
            predictions['uncertainty'], 
            uncertainty_target
        )
        
        # 总损失
        total_loss = (self.classification_weight * cls_loss + 
                     self.localization_weight * loc_loss + 
                     self.uncertainty_weight * unc_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'localization_loss': loc_loss,
            'uncertainty_loss': unc_loss
        }

def create_enhanced_airbubble_detector(config: Dict) -> EnhancedAirBubbleDetector:
    """创建增强型气孔检测器"""
    return EnhancedAirBubbleDetector(
        input_channels=config.get('input_channels', 3),
        num_classes=config.get('num_classes', 2)
    )

# 测试代码
if __name__ == "__main__":
    # 创建模型
    model = EnhancedAirBubbleDetector()
    
    # 测试输入
    x = torch.randn(4, 3, 70, 70)
    
    # 前向传播
    outputs = model(x)
    
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    # 测试数据增强
    augmentation = PhysicsBasedAugmentation()
    test_image = torch.randn(3, 70, 70)
    augmented = augmentation.apply_augmentation(test_image)
    print(f"Augmented image shape: {augmented.shape}")