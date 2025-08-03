#!/usr/bin/env python3
"""
ShuffleNet V2 模型实现
基于 ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel shuffle operation"""
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    
    # flatten
    x = x.view(batch_size, -1, height, width)
    
    return x

class InvertedResidual(nn.Module):
    """ShuffleNet V2 Basic Unit"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        
        branch_features = out_channels // 2
        
        if self.stride > 1:
            # For stride > 1, we don't split the input
            self.branch1 = nn.Sequential(
                self.depthwise_conv(in_channels, in_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if (self.stride > 1) else branch_features,
                     branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
    
    @staticmethod
    def depthwise_conv(in_channels: int, out_channels: int, kernel_size: int,
                      stride: int = 1, padding: int = 0, bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        groups=in_channels, bias=bias)
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        
        out = channel_shuffle(out, 2)
        
        return out

class ShuffleNetV2(nn.Module):
    """ShuffleNet V2 模型"""
    
    def __init__(self, stages_repeats: List[int], stages_out_channels: List[int],
                 num_classes: int = 1000, inverted_residual=InvertedResidual):
        super().__init__()
        
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        
        # First conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(output_channels, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_shufflenetv2_x0_5(num_classes: int = 1000, **kwargs) -> ShuffleNetV2:
    """创建 ShuffleNet V2 0.5x 模型"""
    return ShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 48, 96, 192, 1024],
        num_classes=num_classes
    )

def create_shufflenetv2_x1_0(num_classes: int = 1000, **kwargs) -> ShuffleNetV2:
    """创建 ShuffleNet V2 1.0x 模型"""
    return ShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 116, 232, 464, 1024],
        num_classes=num_classes
    )

def create_shufflenetv2_x1_5(num_classes: int = 1000, **kwargs) -> ShuffleNetV2:
    """创建 ShuffleNet V2 1.5x 模型"""
    return ShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 176, 352, 704, 1024],
        num_classes=num_classes
    )

def create_shufflenetv2_x2_0(num_classes: int = 1000, **kwargs) -> ShuffleNetV2:
    """创建 ShuffleNet V2 2.0x 模型"""
    return ShuffleNetV2(
        stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 244, 488, 976, 2048],
        num_classes=num_classes
    )

if __name__ == "__main__":
    # 测试模型
    model_0_5 = create_shufflenetv2_x0_5(num_classes=2)
    model_1_0 = create_shufflenetv2_x1_0(num_classes=2)
    
    x = torch.randn(1, 3, 224, 224)
    
    print("ShuffleNet V2 0.5x:")
    print(f"参数量: {sum(p.numel() for p in model_0_5.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_0_5(x).shape}")
    
    print("\nShuffleNet V2 1.0x:")
    print(f"参数量: {sum(p.numel() for p in model_1_0.parameters()) / 1e6:.1f}M")
    print(f"输出形状: {model_1_0(x).shape}")