"""
改进版ResNet实现
针对70x70小图像和菌落检测任务优化的ResNet变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ImprovedBasicBlock(nn.Module):
    """改进的基础残差块，加入SE注意力和更好的正则化"""
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None, use_se: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE注意力机制
        self.se = SEBlock(out_channels) if use_se else None
        
        # Dropout正则化
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # 下采样层
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # 第一个卷积块
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 第二个卷积块
        out = self.bn2(self.conv2(out))
        
        # SE注意力
        if self.se is not None:
            out = self.se(out)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out

class ImprovedBottleneck(nn.Module):
    """改进的瓶颈残差块"""
    
    expansion = 4
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, use_se: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1卷积升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # SE注意力机制
        self.se = SEBlock(out_channels * self.expansion) if use_se else None
        
        # Dropout正则化
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
        # 下采样层
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # 1x1卷积
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 3x3卷积
        out = F.relu(self.bn2(self.conv2(out)))
        
        # 1x1卷积
        out = self.bn3(self.conv3(out))
        
        # SE注意力
        if self.se is not None:
            out = self.se(out)
        
        # Dropout
        if self.dropout is not None:
            out = self.dropout(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out

class ResNetImproved(nn.Module):
    """
    改进版ResNet，针对70x70小图像优化
    特点：
    1. 更小的初始卷积核
    2. SE注意力机制
    3. 更好的正则化
    4. 针对小图像的层数配置
    """
    
    def __init__(self, block, layers: List[int], num_classes: int = 2, 
                 use_se: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        
        self.in_channels = 64
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        
        # 针对70x70小图像的初始层 - 使用更小的卷积核和步长
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
    def _make_layer(self, block, out_channels: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample,
                           self.use_se, self.dropout_rate))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_se=self.use_se,
                               dropout_rate=self.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 初始层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类头
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x):
        """获取特征图用于可视化分析"""
        features = []
        
        # 初始层
        x = F.relu(self.bn1(self.conv1(x)))
        features.append(('conv1', x.clone()))
        x = self.maxpool(x)
        features.append(('maxpool', x.clone()))
        
        # 残差层
        x = self.layer1(x)
        features.append(('layer1', x.clone()))
        x = self.layer2(x)
        features.append(('layer2', x.clone()))
        x = self.layer3(x)
        features.append(('layer3', x.clone()))
        x = self.layer4(x)
        features.append(('layer4', x.clone()))
        
        return features

def create_resnet18_improved(num_classes: int = 2, pretrained: bool = False) -> ResNetImproved:
    """创建改进版ResNet-18"""
    model = ResNetImproved(ImprovedBasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                          use_se=True, dropout_rate=0.1)
    
    if pretrained:
        print("Warning: 预训练权重暂未实现")
    
    return model

def create_resnet34_improved(num_classes: int = 2, pretrained: bool = False) -> ResNetImproved:
    """创建改进版ResNet-34"""
    model = ResNetImproved(ImprovedBasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                          use_se=True, dropout_rate=0.1)
    
    if pretrained:
        print("Warning: 预训练权重暂未实现")
    
    return model

def create_resnet50_improved(num_classes: int = 2, pretrained: bool = False) -> ResNetImproved:
    """创建改进版ResNet-50"""
    model = ResNetImproved(ImprovedBottleneck, [3, 4, 6, 3], num_classes=num_classes,
                          use_se=True, dropout_rate=0.1)
    
    if pretrained:
        print("Warning: 预训练权重暂未实现")
    
    return model

if __name__ == "__main__":
    # 测试模型
    models = {
        'ResNet-18': create_resnet18_improved(),
        'ResNet-34': create_resnet34_improved(),
        'ResNet-50': create_resnet50_improved()
    }
    
    for name, model in models.items():
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name} Improved:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        # 测试前向传播
        x = torch.randn(1, 3, 70, 70)
        with torch.no_grad():
            output = model(x)
            print(f"  输入形状: {x.shape}")
            print(f"  输出形状: {output.shape}")
            
            # 测试特征图提取
            features = model.get_feature_maps(x)
            print(f"  特征图数量: {len(features)}")
            for feat_name, feat in features:
                print(f"    {feat_name}: {feat.shape}")
        print()