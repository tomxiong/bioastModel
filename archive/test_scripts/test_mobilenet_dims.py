import torch
import torch.nn as nn
from models.mobilenet_v3 import create_mobilenetv3_small, create_mobilenetv3_large

# 测试MobileNetV3的输出维度
def test_mobilenet_dims():
    # 创建模型
    model_small = create_mobilenetv3_small(pretrained=False)
    model_large = create_mobilenetv3_large(pretrained=False)
    
    # 修改第一层以接受灰度输入
    for model, name in [(model_small, 'small'), (model_large, 'large')]:
        # 找到第一个卷积层并修改
        first_conv = None
        for layer in model.features:
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Conv2d):
                        first_conv = sublayer
                        break
            elif isinstance(layer, nn.Conv2d):
                first_conv = layer
                break
            if first_conv:
                break
        
        if first_conv:
            new_first_conv = nn.Conv2d(
                1, first_conv.out_channels,
                first_conv.kernel_size, first_conv.stride,
                first_conv.padding, bias=False
            )
            
            if isinstance(model.features[0], nn.Sequential):
                model.features[0][0] = new_first_conv
            else:
                model.features[0] = new_first_conv
        
        # 移除分类器
        model.classifier = nn.Identity()
        
        # 测试输入
        x = torch.randn(2, 1, 70, 70)
        
        # 通过features提取特征
        features = x
        for i, layer in enumerate(model.features):
            features = layer(features)
            print(f"{name} - Layer {i}: {features.shape}")
        
        print(f"\n{name} final features shape: {features.shape}")
        print(f"{name} feature channels: {features.shape[1]}")
        print("-" * 50)

if __name__ == "__main__":
    test_mobilenet_dims()
