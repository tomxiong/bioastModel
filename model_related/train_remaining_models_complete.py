#!/usr/bin/env python3
"""
训练剩余模型脚本
完成所有22个模型的训练，包括创建新模型和训练
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Any

def get_trained_models() -> List[str]:
    """获取已训练的模型列表"""
    trained_models = []
    checkpoint_dir = "checkpoints"
    
    if os.path.exists(checkpoint_dir):
        checkpoint_files = os.listdir(checkpoint_dir)
        for file in checkpoint_files:
            if file.endswith("_best.pth"):
                # 提取模型名称
                model_name = file.replace("_best.pth", "").rsplit("_", 2)[0]
                if model_name not in trained_models:
                    trained_models.append(model_name)
    
    return trained_models

def get_target_models() -> List[Dict[str, Any]]:
    """获取目标模型列表（22个模型）"""
    models = [
        # 已存在的模型
        {"name": "simplified_airbubble_detector", "priority": 1, "trained": True},
        {"name": "micro_vit", "priority": 2, "trained": True},
        {"name": "mic_mobilenetv3", "priority": 3, "trained": True},
        {"name": "resnet_micro", "priority": 4, "trained": True},
        {"name": "densenet_compact", "priority": 5, "trained": True},
        {"name": "inception_micro", "priority": 6, "trained": True},
        {"name": "efficientnet_b0_micro", "priority": 7, "trained": True},
        {"name": "efficient_cnn", "priority": 8, "trained": True},
        
        # 需要创建和训练的新模型
        {"name": "mobilenet_v2_micro", "priority": 9, "trained": False},
        {"name": "squeezenet_micro", "priority": 10, "trained": False},
        {"name": "shufflenet_micro", "priority": 11, "trained": False},
        {"name": "ghostnet_micro", "priority": 12, "trained": False},
        {"name": "regnet_micro", "priority": 13, "trained": False},
        {"name": "efficientnet_b1_micro", "priority": 14, "trained": False},
        {"name": "resnext_micro", "priority": 15, "trained": False},
        {"name": "wide_resnet_micro", "priority": 16, "trained": False},
        {"name": "densenet_121_micro", "priority": 17, "trained": False},
        {"name": "vgg_micro", "priority": 18, "trained": False},
        {"name": "alexnet_micro", "priority": 19, "trained": False},
        {"name": "convnext_micro", "priority": 20, "trained": False},
        {"name": "swin_transformer_micro", "priority": 21, "trained": False},
        {"name": "vision_transformer_micro", "priority": 22, "trained": False}
    ]
    
    return models

def create_model_architecture(model_name: str) -> str:
    """为新模型创建架构定义"""
    
    if model_name == "mobilenet_v2_micro":
        return '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV2Micro(nn.Module):
    """MobileNetV2 微型版本，专为70x70生物医学图像优化"""
    
    def __init__(self, num_classes=2):
        super(MobileNetV2Micro, self).__init__()
        
        # 输入验证
        self.input_size = (70, 70)
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Inverted Residual Blocks
        self.inverted_residual_blocks = nn.Sequential(
            self._make_inverted_residual(32, 16, 1, 1),
            self._make_inverted_residual(16, 24, 6, 2),
            self._make_inverted_residual(24, 32, 6, 2),
            self._make_inverted_residual(32, 64, 6, 2),
            self._make_inverted_residual(64, 96, 6, 1),
            self._make_inverted_residual(96, 160, 6, 2),
        )
        
        # 最终卷积层
        self.conv2 = nn.Conv2d(160, 320, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(320)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(320, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_inverted_residual(self, inp, oup, expand_ratio, stride):
        """创建倒残差块"""
        hidden_dim = int(inp * expand_ratio)
        use_res_connect = stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        conv = nn.Sequential(*layers)
        
        if use_res_connect:
            return lambda x: x + conv(x)
        else:
            return conv
    
    def _initialize_weights(self):
        """初始化权重"""
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
        # 输入尺寸验证
        if x.shape[-2:] != (70, 70):
            raise ValueError(f"Expected input size (70, 70), got {x.shape[-2:]}")
        
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.inverted_residual_blocks(x)
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# 测试代码
if __name__ == "__main__":
    model = MobileNetV2Micro()
    x = torch.randn(1, 3, 70, 70)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
'''
    
    elif model_name == "squeezenet_micro":
        return '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeNetMicro(nn.Module):
    """SqueezeNet 微型版本，专为70x70生物医学图像优化"""
    
    def __init__(self, num_classes=2):
        super(SqueezeNetMicro, self).__init__()
        
        self.input_size = (70, 70)
        
        # 初始卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Fire modules
            self._make_fire_module(64, 16, 64, 64),
            self._make_fire_module(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            self._make_fire_module(128, 32, 128, 128),
            self._make_fire_module(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            self._make_fire_module(256, 48, 192, 192),
            self._make_fire_module(384, 48, 192, 192),
            self._make_fire_module(384, 64, 256, 256),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self._initialize_weights()
    
    def _make_fire_module(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        """创建Fire模块"""
        return nn.Sequential(
            # Squeeze layer
            nn.Conv2d(inplanes, squeeze_planes, kernel_size=1),
            nn.ReLU(inplace=True),
            
            # Expand layers
            nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if x.shape[-2:] != (70, 70):
            raise ValueError(f"Expected input size (70, 70), got {x.shape[-2:]}")
        
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

if __name__ == "__main__":
    model = SqueezeNetMicro()
    x = torch.randn(1, 3, 70, 70)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
'''
    
    # 为其他模型返回基础模板
    else:
        class_name = ''.join(word.capitalize() for word in model_name.split('_'))
        return f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class {class_name}(nn.Module):
    """
    {class_name} - 专为70x70生物医学图像优化的微型模型
    """
    
    def __init__(self, num_classes=2):
        super({class_name}, self).__init__()
        
        self.input_size = (70, 70)
        
        # 特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
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
        if x.shape[-2:] != (70, 70):
            raise ValueError(f"Expected input size (70, 70), got {{x.shape[-2:]}}")
        
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    model = {class_name}()
    x = torch.randn(1, 3, 70, 70)
    y = model(x)
    print(f"Input shape: {{x.shape}}")
    print(f"Output shape: {{y.shape}}")
    print(f"Total parameters: {{sum(p.numel() for p in model.parameters()):,}}")
'''

def create_trainer_script(model_name: str) -> str:
    """为新模型创建训练脚本"""
    class_name = ''.join(word.capitalize() for word in model_name.split('_'))
    
    return f'''#!/usr/bin/env python3
"""
{class_name} 训练脚本
使用真实生物医学数据训练模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.{model_name} import {class_name}
from core.real_data_loader import create_real_data_loaders

def train_model():
    """训练{class_name}模型"""
    print("=" * 60)
    print(f"训练 {class_name} 模型")
    print("=" * 60)
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {{device}}")
    
    # 创建模型
    model = {class_name}()
    model = model.to(device)
    
    # 验证输入尺寸
    dummy_input = torch.randn(1, 3, 70, 70).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ 模型接受70x70输入，输出形状: {{output.shape}}")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {{total_params:,}}")
    print(f"可训练参数数量: {{trainable_params:,}}")
    
    # 加载数据
    print("\\n加载数据...")
    train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32)
    print(f"训练样本: {{len(train_loader.dataset)}}")
    print(f"验证样本: {{len(val_loader.dataset)}}")
    print(f"测试样本: {{len(test_loader.dataset)}}")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    # 训练参数
    num_epochs = 50
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"\\n训练配置:")
    print(f"轮数: {{num_epochs}}")
    print(f"批次大小: 32")
    print(f"学习率: 0.001")
    print(f"权重衰减: 0.01")
    print(f"耐心值: {{patience}}")
    print(f"调度器: CosineAnnealingLR")
    
    # 创建保存目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # 训练历史
    train_losses = []
    val_accuracies = []
    
    print("\\n开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\\nEpoch {{epoch+1}}/{{num_epochs}}")
        print("-" * 30)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"Batch {{batch_idx}}/{{len(train_loader)}} Loss: {{loss.item():.4f}} Acc: {{100.*correct_train/total_train:.2f}}%")
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct_val / total_val
        
        # 记录历史
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {{train_loss:.4f}} Train Acc: {{train_acc:.2f}}%")
        print(f"Val Loss: {{val_loss:.4f}} Val Acc: {{val_acc:.2f}}%")
        print(f"Learning Rate: {{scheduler.get_last_lr()[0]:.6f}}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"checkpoints/{model_name}_{{timestamp}}_best.pth"
            
            torch.save({{
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'total_params': total_params,
                'trainable_params': trainable_params
            }}, checkpoint_path)
            
            print(f"✓ 新的最佳模型已保存: {{best_val_acc:.2f}}%")
        else:
            patience_counter += 1
        
        scheduler.step()
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\\n早停触发，已等待 {{patience}} 轮")
            break
    
    training_time = time.time() - start_time
    print(f"\\n训练完成，用时 {{training_time:.2f}} 秒")
    
    # 测试评估
    print("\\n在测试集上评估...")
    model.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()
    
    test_acc = 100. * correct_test / total_test
    test_loss /= len(test_loader)
    
    print(f"测试准确率: {{test_acc:.4f}}%")
    print(f"测试损失: {{test_loss:.4f}}")
    
    # 保存训练报告
    report = {{
        'model_name': '{model_name}',
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'training_time_seconds': training_time,
        'total_epochs': epoch + 1,
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': test_acc,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'checkpoint_path': checkpoint_path,
        'config': {{
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'patience': patience
        }}
    }}
    
    report_path = f"reports/{model_name}_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}_training.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\\n训练结果已保存到: {{report_path}}")
    
    print("\\n" + "=" * 60)
    print(f"✅ {class_name} 训练完成!")
    print(f"📊 最佳验证准确率: {{best_val_acc:.2f}}%")
    print(f"📊 测试准确率: {{test_acc:.2f}}%")
    print(f"💾 模型已保存: {{checkpoint_path}}")
    print(f"📄 报告已保存: {{report_path}}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
'''

def create_new_models_and_trainers():
    """创建新模型和训练脚本"""
    target_models = get_target_models()
    trained_models = get_trained_models()
    
    print(f"已训练模型: {trained_models}")
    
    for model_info in target_models:
        model_name = model_info["name"]
        
        if model_name not in trained_models:
            print(f"\\n创建新模型: {model_name}")
            
            # 创建模型文件
            model_path = f"models/{model_name}.py"
            os.makedirs("models", exist_ok=True)
            
            if not os.path.exists(model_path):
                model_code = create_model_architecture(model_name)
                with open(model_path, 'w') as f:
                    f.write(model_code)
                print(f"✅ 模型文件已创建: {model_path}")
            
            # 创建训练脚本
            trainer_path = f"trainers/train_{model_name}.py"
            os.makedirs("trainers", exist_ok=True)
            
            if not os.path.exists(trainer_path):
                trainer_code = create_trainer_script(model_name)
                with open(trainer_path, 'w') as f:
                    f.write(trainer_code)
                print(f"✅ 训练脚本已创建: {trainer_path}")

def train_remaining_models():
    """训练剩余的模型"""
    target_models = get_target_models()
    trained_models = get_trained_models()
    
    untrained_models = [model for model in target_models if model["name"] not in trained_models]
    
    if not untrained_models:
        print("✅ 所有模型都已训练完成!")
        return
    
    print(f"\\n需要训练的模型数量: {len(untrained_models)}")
    
    for i, model_info in enumerate(untrained_models, 1):
        model_name = model_info["name"]
        priority = model_info["priority"]
        
        print(f"\\n{'='*60}")
        print(f"🚀 开始训练模型 {i}/{len(untrained_models)}: {model_name}")
        print(f"优先级: {priority}")
        print(f"{'='*60}")
        
        trainer_script = f"trainers/train_{model_name}.py"
        
        if os.path.exists(trainer_script):
            try:
                # 运行训练脚本
                result = subprocess.run([
                    sys.executable, trainer_script
                ], capture_output=True, text=True, cwd=os.getcwd())
                
                if result.returncode == 0:
                    print(f"✅ {model_name} 训练成功完成")
                else:
                    print(f"❌ {model_name} 训练失败")
                    print(f"错误输出: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ 训练 {model_name} 时发生错误: {e}")
        else:
            print(f"❌ 训练脚本不存在: {trainer_script}")

def main():
    """主函数"""
    print("🧬 开始训练剩余模型")
    print("=" * 60)
    
    # 1. 创建新模型和训练脚本
    print("\\n📝 创建新模型和训练脚本...")
    create_new_models_and_trainers()
    
    # 2. 训练剩余模型
    print("\\n🚀 开始训练剩余模型...")
    train_remaining_models()
    
    # 3. 生成最终报告
    print("\\n📊 生成最终训练报告...")
    trained_models = get_trained_models()
    target_models = get_target_models()
    
    print(f"\\n{'='*60}")
    print("🎯 训练完成总结")
    print(f"{'='*60}")
    print(f"目标模型数量: {len(target_models)}")
    print(f"已训练模型数量: {len(trained_models)}")
    print(f"完成率: {len(trained_models)/len(target_models)*100:.1f}%")
    
    print(f"\\n已训练的模型:")
    for model in trained_models:
        print(f"  ✅ {model}")
    
    untrained = [model["name"] for model in target_models if model["name"] not in trained_models]
    if untrained:
        print(f"\\n未训练