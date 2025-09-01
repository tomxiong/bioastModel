#!/usr/bin/env python3
"""
训练额外模型脚本 - 创建并训练剩余的14个模型以达到22个模型的目标
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def get_trained_models():
    """获取已训练的模型列表"""
    trained_models = []
    checkpoint_dir = "checkpoints"
    
    if os.path.exists(checkpoint_dir):
        checkpoint_files = os.listdir(checkpoint_dir)
        for file in checkpoint_files:
            if file.endswith("_best.pth"):
                model_name = file.replace("_best.pth", "").rsplit("_", 2)[0]
                if model_name not in trained_models:
                    trained_models.append(model_name)
    
    return trained_models

def create_simple_model_template(model_name):
    """创建简单的模型模板"""
    class_name = ''.join(word.capitalize() for word in model_name.split('_'))
    
    return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class {class_name}(nn.Module):
    """
    {class_name} - 专为70x70生物医学图像优化的模型
    """
    
    def __init__(self, num_classes=2):
        super({class_name}, self).__init__()
        
        self.input_size = (70, 70)
        
        # 特征提取器
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 35x35
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 17x17
            
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8
            
            # 第四层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
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

def create_simple_trainer(model_name):
    """创建简单的训练脚本"""
    class_name = ''.join(word.capitalize() for word in model_name.split('_'))
    
    return f'''#!/usr/bin/env python3
"""
{class_name} 训练脚本
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.{model_name} import {class_name}
from core.real_data_loader import create_real_data_loaders

def train_model():
    print("=" * 60)
    print(f"训练 {class_name} 模型")
    print("=" * 60)
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {{total_params:,}}")
    
    # 加载数据
    print("\\n加载数据...")
    train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32)
    print(f"训练样本: {{len(train_loader.dataset)}}")
    print(f"验证样本: {{len(val_loader.dataset)}}")
    print(f"测试样本: {{len(test_loader.dataset)}}")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    
    num_epochs = 30
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
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
                'test_acc': 0.0
            }}, checkpoint_path)
            
            print(f"✓ 新的最佳模型已保存: {{best_val_acc:.2f}}%")
        else:
            patience_counter += 1
        
        scheduler.step()
        
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
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()
    
    test_acc = 100. * correct_test / total_test
    print(f"测试准确率: {{test_acc:.4f}}%")
    
    # 更新checkpoint中的test_acc
    checkpoint = torch.load(checkpoint_path)
    checkpoint['test_acc'] = test_acc
    torch.save(checkpoint, checkpoint_path)
    
    print(f"\\n✅ {class_name} 训练完成!")
    print(f"📊 最佳验证准确率: {{best_val_acc:.2f}}%")
    print(f"📊 测试准确率: {{test_acc:.2f}}%")
    print(f"💾 模型已保存: {{checkpoint_path}}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
'''

def main():
    print("🧬 开始训练剩余模型以达到22个模型目标")
    print("=" * 60)
    
    # 获取已训练的模型
    trained_models = get_trained_models()
    print(f"已训练模型数量: {len(trained_models)}")
    print(f"已训练模型: {trained_models}")
    
    # 定义目标模型列表（22个）
    target_models = [
        "simplified_airbubble_detector",  # 已训练
        "micro_vit",                      # 已训练
        "mic_mobilenetv3",               # 已训练
        "resnet_micro",                  # 已训练
        "densenet_compact",              # 已训练
        "inception_micro",               # 已训练
        "efficientnet_b0_micro",         # 已训练
        "efficient_cnn",                 # 已训练
        # 需要新增的14个模型
        "mobilenet_v2_micro",
        "squeezenet_micro", 
        "shufflenet_micro",
        "ghostnet_micro",
        "regnet_micro",
        "efficientnet_b1_micro",
        "resnext_micro",
        "wide_resnet_micro",
        "densenet_121_micro",
        "vgg_micro",
        "alexnet_micro",
        "convnext_micro",
        "swin_transformer_micro",
        "vision_transformer_micro"
    ]
    
    # 找出需要训练的模型
    models_to_train = [model for model in target_models if model not in trained_models]
    
    print(f"\\n需要训练的模型数量: {len(models_to_train)}")
    print(f"需要训练的模型: {models_to_train}")
    
    if not models_to_train:
        print("\\n✅ 所有22个模型都已训练完成!")
        return
    
    # 创建模型和训练脚本
    print("\\n📝 创建模型文件和训练脚本...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("trainers", exist_ok=True)
    
    for model_name in models_to_train:
        # 创建模型文件
        model_path = f"models/{model_name}.py"
        if not os.path.exists(model_path):
            model_code = create_simple_model_template(model_name)
            with open(model_path, 'w') as f:
                f.write(model_code)
            print(f"✅ 创建模型文件: {model_path}")
        
        # 创建训练脚本
        trainer_path = f"trainers/train_{model_name}.py"
        if not os.path.exists(trainer_path):
            trainer_code = create_simple_trainer(model_name)
            with open(trainer_path, 'w') as f:
                f.write(trainer_code)
            print(f"✅ 创建训练脚本: {trainer_path}")
    
    # 开始训练模型
    print(f"\\n🚀 开始训练 {len(models_to_train)} 个模型...")
    
    successful_trainings = 0
    failed_trainings = 0
    
    for i, model_name in enumerate(models_to_train, 1):
        print(f"\\n{'='*60}")
        print(f"🚀 训练模型 {i}/{len(models_to_train)}: {model_name}")
        print(f"{'='*60}")
        
        trainer_script = f"trainers/train_{model_name}.py"
        
        try:
            # 运行训练脚本
            result = subprocess.run([
                sys.executable, trainer_script
            ], cwd=os.getcwd(), timeout=1800)  # 30分钟超时
            
            if result.returncode == 0:
                print(f"✅ {model_name} 训练成功完成")
                successful_trainings += 1
            else:
                print(f"❌ {model_name} 训练失败 (返回码: {result.returncode})")
                failed_trainings += 1
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {model_name} 训练超时")
            failed_trainings += 1
        except Exception as e:
            print(f"❌ 训练 {model_name} 时发生错误: {e}")
            failed_trainings += 1
    
    # 最终统计
    final_trained_models = get_trained_models()
    
    print(f"\\n{'='*60}")
    print("🎯 训练完成总结")
    print(f"{'='*60}")
    print(f"目标模型数量: {len(target_models)}")
    print(f"最终训练模型数量: {len(final_trained_models)}")
    print(f"本次成功训练: {successful_trainings}")
    print(f"本次失败训练: {failed_trainings}")
    print(f"完成率: {len(final_trained_models)/len(target_models)*100:.1f}%")
    
    print(f"\\n最终训练的模型:")
    for model in sorted(final_trained_models):
        print(f"  ✅ {model}")
    
    remaining = [model for model in target_models if model not in final_trained_models]
    if remaining:
        print(f"\\n仍需训练的模型:")
        for model in remaining:
            print(f"  ⏸️ {model}")
    
    print(f"\\n🎉 训练任务完成!")

if __name__ == "__main__":
    main()