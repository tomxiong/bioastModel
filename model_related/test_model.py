import torch
import torch.nn as nn

# Test the problematic Conv2d call
try:
    # This should work
    conv1 = nn.Conv2d(3, 16, 1, bias=False)
    print("✓ Conv2d with correct syntax works")
    
    # Test the model import
    from models.mic_mobilenetv3 import MICMobileNetV3
    model = MICMobileNetV3(num_classes=2)
    print("✓ Model creation successful")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()