import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResnetMicro(nn.Module):
    def __init__(self, num_classes=2):
        super(ResnetMicro, self).__init__()
        self.in_planes = 64
        
        # Input validation for 70x70 images
        self.input_size = 70
        
        # Initial convolution - optimized for 70x70 input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers - reduced for micro version
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * BasicBlock.expansion, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Validate input size
        if x.shape[-2:] != (70, 70):
            raise ValueError(f"Expected input size (70, 70), got {x.shape[-2:]}")
        
        # Forward pass
        x = F.relu(self.bn1(self.conv1(x)))  # 70x70 -> 35x35
        x = self.maxpool(x)                   # 35x35 -> 18x18
        
        x = self.layer1(x)                    # 18x18 -> 18x18
        x = self.layer2(x)                    # 18x18 -> 9x9
        x = self.layer3(x)                    # 9x9 -> 5x5
        
        x = self.avgpool(x)                   # 5x5 -> 1x1
        x = torch.flatten(x, 1)               # Flatten
        x = self.dropout(x)                   # Dropout
        x = self.fc(x)                        # Classification
        
        return x

    def get_model_info(self):
        """Return model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'ResnetMicro',
            'input_size': (3, 70, 70),
            'output_size': 2,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'Micro ResNet with BasicBlocks',
            'layers': {
                'conv1': '7x7 conv, stride=2',
                'layer1': '2x BasicBlock, 64 channels',
                'layer2': '2x BasicBlock, 128 channels, stride=2',
                'layer3': '2x BasicBlock, 256 channels, stride=2',
                'classifier': 'Linear(256, 2)'
            }
        }

# Test function
def test_resnet_micro():
    """Test the ResnetMicro model"""
    print("Testing ResnetMicro model...")
    
    model = ResnetMicro(num_classes=2)
    
    # Test with correct input size
    test_input = torch.randn(4, 3, 70, 70)
    try:
        output = model(test_input)
        print(f"✅ Model test passed!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        
        # Print model info
        info = model.get_model_info()
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    # Test with wrong input size
    try:
        wrong_input = torch.randn(1, 3, 224, 224)
        model(wrong_input)
        print("❌ Model should have rejected wrong input size")
        return False
    except ValueError as e:
        print(f"✅ Model correctly rejected wrong input size: {e}")
    
    return True

if __name__ == "__main__":
    test_resnet_micro()