#!/usr/bin/env python3
"""
Working Training Script for mic_mobilenetv3
Using proven pattern from successful trainers
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model():
    """Train the mic_mobilenetv3 model"""
    print(f"🚀 Starting working training for mic_mobilenetv3")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading using proven pattern
    try:
        from core.real_data_loader import create_real_data_loaders
        train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32, num_workers=4)
        print(f"✅ Data loaded successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Model initialization with multiple fallback strategies
    model = None
    model_created = False
    
    try:
        # Strategy 1: Try specific model configurations
        if "mic_mobilenetv3" == "mobilenet_v3":
            from models.mobilenet_v3 import create_mobilenetv3_large
            model = create_mobilenetv3_large(num_classes=2)
            model_created = True
        elif "mic_mobilenetv3" == "efficientnet_v2":
            from models.efficientnet_v2 import EfficientNetV2
            # Try with minimal configuration
            try:
                model = EfficientNetV2(num_classes=2)
                model_created = True
            except:
                # Fallback with basic block configs
                block_configs = [
                    ['conv', 1, 1, 1, 16, 16, 1],
                    ['fused_mb', 4, 2, 1, 16, 32, 2],
                    ['fused_mb', 4, 2, 1, 32, 48, 2],
                ]
                model = EfficientNetV2(block_configs=block_configs, num_classes=2)
                model_created = True
        elif "mic_mobilenetv3" == "enhanced_airbubble_detector":
            from models.enhanced_airbubble_detector import EnhancedAirBubbleDetector
            model = EnhancedAirBubbleDetector(num_classes=2)
            model_created = True
        elif "mic_mobilenetv3" == "ghostnet":
            from models.ghostnet import GhostNet
            try:
                model = GhostNet(num_classes=2)
                model_created = True
            except:
                model = GhostNet(num_classes=2, width=1.0)
                model_created = True
        elif "mic_mobilenetv3" == "mic_mobilenetv3":
            from models.mic_mobilenetv3 import MICMobileNetV3
            model = MICMobileNetV3(num_classes=2)
            model_created = True
        elif "mic_mobilenetv3" == "micro_vit":
            from models.micro_vit import MicroViT
            model = MicroViT(num_classes=2, img_size=70, patch_size=7)
            model_created = True
        elif "mic_mobilenetv3" == "mnasnet":
            from models.mnasnet import MNASNet
            model = MNASNet(alpha=1.0, num_classes=2)
            model_created = True
        elif "mic_mobilenetv3" == "regnet":
            from models.regnet import RegNet
            model = RegNet(num_classes=2)
            model_created = True
        elif "mic_mobilenetv3" == "regnet_wrapper":
            from models.regnet_wrapper import RegNetWrapper
            model = RegNetWrapper(num_classes=2)
            model_created = True
        elif "mic_mobilenetv3" == "resnet_improved":
            from models.resnet_improved import ResNetImproved
            model = ResNetImproved(num_classes=2)
            model_created = True
        elif "mic_mobilenetv3" == "shufflenet_v2":
            from models.shufflenet_v2 import ShuffleNetV2
            model = ShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024], num_classes=2)
            model_created = True
        elif "mic_mobilenetv3" == "vit_tiny":
            from models.vit_tiny import ViTTiny
            model = ViTTiny(num_classes=2, img_size=70, patch_size=7)
            model_created = True
        
        if model_created:
            print(f"✅ Model initialized with specific configuration")
        
    except Exception as e:
        print(f"⚠️  Specific initialization failed: {e}")
        model_created = False
    
    # Strategy 2: Generic fallback
    if not model_created:
        try:
            # Import and try basic initialization
            module_name = f"models.mic_mobilenetv3"
            model_module = __import__(module_name, fromlist=[''])
            
            # Try to find the main class
            for attr_name in dir(model_module):
                attr = getattr(model_module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, torch.nn.Module) and 
                    attr != torch.nn.Module and
                    not attr_name.startswith('_')):
                    try:
                        model = attr(num_classes=2)
                        model_created = True
                        print(f"✅ Model created with generic approach: {attr_name}")
                        break
                    except:
                        continue
        except Exception as e:
            print(f"⚠️  Generic initialization failed: {e}")
    
    if not model_created or model is None:
        print(f"❌ All model initialization attempts failed")
        return False
    
    try:
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model moved to device: {param_count:.1f}M parameters")
    except Exception as e:
        print(f"❌ Failed to move model to device: {e}")
        return False
    
    # Training setup
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=30)
        print(f"✅ Training setup completed")
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        return False
    
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    
    # Training loop
    for epoch in range(30):
        try:
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    
                    # Handle multi-output models
                    if isinstance(output, dict):
                        if 'classification' in output:
                            output = output['classification']
                        elif 'logits' in output:
                            output = output['logits']
                        else:
                            output = list(output.values())[0]
                    
                    # Handle dimension mismatches
                    if output.dim() > 2:
                        output = output.view(output.size(0), -1)
                    
                    if output.size(1) != 2:
                        # Add a linear layer to map to correct output size
                        if not hasattr(model, 'final_classifier'):
                            model.final_classifier = nn.Linear(output.size(1), 2).to(device)
                        output = model.final_classifier(output)
                    
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()
                    
                    if batch_idx % 50 == 0:
                        acc = 100.*train_correct/train_total
                        print(f'Epoch {epoch+1}/30, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
                        
                except Exception as e:
                    print(f"❌ Training batch {batch_idx} failed: {e}")
                    continue
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    try:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        
                        # Handle multi-output models
                        if isinstance(output, dict):
                            if 'classification' in output:
                                output = output['classification']
                            elif 'logits' in output:
                                output = output['logits']
                            else:
                                output = list(output.values())[0]
                        
                        if output.dim() > 2:
                            output = output.view(output.size(0), -1)
                        
                        if output.size(1) != 2 and hasattr(model, 'final_classifier'):
                            output = model.final_classifier(output)
                        
                        loss = criterion(output, target)
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                    except Exception as e:
                        print(f"❌ Validation batch failed: {e}")
                        continue
            
            train_acc = 100. * train_correct / train_total if train_total > 0 else 0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            
            print(f'Epoch {epoch+1}/30: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save checkpoint
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f'checkpoints/mic_mobilenetv3_{timestamp}_best.pth'
                os.makedirs('checkpoints', exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'model_name': 'mic_mobilenetv3'
                }, checkpoint_path)
                
                print(f'✅ New best model saved: {checkpoint_path} ({best_val_acc:.2f}%)')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            scheduler.step()
            
        except Exception as e:
            print(f"❌ Epoch {epoch+1} failed: {e}")
            continue
    
    print(f"🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc > 0

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
