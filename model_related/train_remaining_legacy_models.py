#!/usr/bin/env python3
"""
训练剩余的历史模型脚本 - 使用更新后的真实数据集重新训练现有模型
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

def get_existing_model_files():
    """获取现有的模型文件列表"""
    model_files = []
    models_dir = "models"
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(".py") and file != "__init__.py":
                model_name = file.replace(".py", "")
                model_files.append(model_name)
    
    return model_files

def create_trainer_for_existing_model(model_name):
    """为现有模型创建训练脚本"""
    # 读取模型文件以获取类名
    model_file_path = f"models/{model_name}.py"
    if not os.path.exists(model_file_path):
        return None
    
    with open(model_file_path, 'r') as f:
        content = f.read()
    
    # 简单的类名提取
    import re
    class_match = re.search(r'class\s+(\w+)\s*\(', content)
    if not class_match:
        return None
    
    class_name = class_match.group(1)
    
    trainer_content = f'''#!/usr/bin/env python3
"""
{class_name} 训练脚本 - 使用真实生物医学数据集
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
    print(f"训练 {class_name} 模型 - 使用真实数据集")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {{device}}")
    
    # 创建模型
    try:
        model = {class_name}()
    except Exception as e:
        print(f"❌ 模型创建失败: {{e}}")
        return False
    
    model = model.to(device)
    
    # 验证输入尺寸
    try:
        dummy_input = torch.randn(1, 3, 70, 70).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ 模型接受70x70输入，输出形状: {{output.shape}}")
    except Exception as e:
        print(f"❌ 模型输入验证失败: {{e}}")
        return False
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {{total_params:,}}")
    
    # 加载数据
    print("\\n加载真实生物医学数据...")
    try:
        train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32)
        print(f"训练样本: {{len(train_loader.dataset)}}")
        print(f"验证样本: {{len(val_loader.dataset)}}")
        print(f"测试样本: {{len(test_loader.dataset)}}")
    except Exception as e:
        print(f"❌ 数据加载失败: {{e}}")
        return False
    
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
            try:
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
            except Exception as e:
                print(f"❌ 训练批次失败: {{e}}")
                return False
        
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
                try:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()
                except Exception as e:
                    print(f"❌ 验证批次失败: {{e}}")
                    return False
        
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
            try:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
            except Exception as e:
                print(f"❌ 测试批次失败: {{e}}")
                return False
    
    test_acc = 100. * correct_test / total_test
    print(f"测试准确率: {{test_acc:.4f}}%")
    
    # 更新checkpoint中的test_acc
    try:
        checkpoint = torch.load(checkpoint_path)
        checkpoint['test_acc'] = test_acc
        torch.save(checkpoint, checkpoint_path)
    except:
        pass
    
    print(f"\\n✅ {class_name} 训练完成!")
    print(f"📊 最佳验证准确率: {{best_val_acc:.2f}}%")
    print(f"📊 测试准确率: {{test_acc:.2f}}%")
    print(f"💾 模型已保存: {{checkpoint_path}}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
'''
    
    return trainer_content

def main():
    print("🧬 训练剩余的历史模型 - 使用真实生物医学数据集")
    print("=" * 60)
    
    # 获取已训练的模型和现有模型文件
    trained_models = get_trained_models()
    existing_models = get_existing_model_files()
    
    print(f"已训练模型数量: {len(trained_models)}")
    print(f"现有模型文件数量: {len(existing_models)}")
    
    # 找出需要重新训练的模型
    models_to_retrain = []
    
    # 特别指定的需要重新训练的模型
    priority_models = [
        "airbubble_hybrid_net",
        "coatnet", 
        "convnext_tiny"
    ]
    
    # 检查优先模型
    for model in priority_models:
        if model in existing_models:
            models_to_retrain.append(model)
            print(f"✓ 发现优先模型: {model}")
    
    # 检查其他现有但未训练的模型
    for model in existing_models:
        if model not in trained_models and model not in models_to_retrain:
            models_to_retrain.append(model)
            print(f"✓ 发现未训练模型: {model}")
    
    print(f"\\n需要(重新)训练的模型数量: {len(models_to_retrain)}")
    print(f"需要(重新)训练的模型: {models_to_retrain}")
    
    if not models_to_retrain:
        print("\\n✅ 所有现有模型都已使用真实数据集训练完成!")
        return
    
    # 创建训练脚本
    print("\\n📝 创建训练脚本...")
    os.makedirs("trainers", exist_ok=True)
    
    successful_script_creation = 0
    failed_script_creation = 0
    
    for model_name in models_to_retrain:
        trainer_path = f"trainers/train_{model_name}.py"
        
        trainer_content = create_trainer_for_existing_model(model_name)
        if trainer_content:
            with open(trainer_path, 'w') as f:
                f.write(trainer_content)
            print(f"✅ 创建训练脚本: {trainer_path}")
            successful_script_creation += 1
        else:
            print(f"❌ 无法为 {model_name} 创建训练脚本")
            failed_script_creation += 1
    
    print(f"\\n成功创建脚本: {successful_script_creation}")
    print(f"失败创建脚本: {failed_script_creation}")
    
    # 开始训练模型
    if successful_script_creation > 0:
        print(f"\\n🚀 开始训练 {successful_script_creation} 个模型...")
        
        successful_trainings = 0
        failed_trainings = 0
        
        for i, model_name in enumerate(models_to_retrain, 1):
            trainer_script = f"trainers/train_{model_name}.py"
            
            if not os.path.exists(trainer_script):
                print(f"⏸️ 跳过 {model_name} (训练脚本不存在)")
                continue
                
            print(f"\\n{'='*60}")
            print(f"🚀 训练模型 {i}/{len(models_to_retrain)}: {model_name}")
            print(f"{'='*60}")
            
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
        print("🎯 历史模型重训练完成总结")
        print(f"{'='*60}")
        print(f"目标重训练模型数量: {len(models_to_retrain)}")
        print(f"本次成功训练: {successful_trainings}")
        print(f"本次失败训练: {failed_trainings}")
        print(f"总训练模型数量: {len(final_trained_models)}")
        
        print(f"\\n本次训练的模型:")
        for model in models_to_retrain:
            if model in final_trained_models:
                print(f"  ✅ {model}")
            else:
                print(f"  ❌ {model}")
        
        print(f"\\n🎉 历史模型重训练任务完成!")

if __name__ == "__main__":
    main()