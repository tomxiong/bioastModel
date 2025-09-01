#!/usr/bin/env python3
"""
ConvnextMicro 训练脚本
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

from models.convnext_micro import ConvnextMicro
from core.real_data_loader import create_real_data_loaders

def train_model():
    print("=" * 60)
    print(f"训练 ConvnextMicro 模型")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = ConvnextMicro()
    model = model.to(device)
    
    # 验证输入尺寸
    dummy_input = torch.randn(1, 3, 70, 70).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ 模型接受70x70输入，输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32)
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")
    print(f"测试样本: {len(test_loader.dataset)}")
    
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
    
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
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
                print(f"Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f} Acc: {100.*correct_train/total_train:.2f}%")
        
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
        
        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"checkpoints/convnext_micro_{timestamp}_best.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'total_params': total_params,
                'test_acc': 0.0
            }, checkpoint_path)
            
            print(f"✓ 新的最佳模型已保存: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        scheduler.step()
        
        if patience_counter >= patience:
            print(f"\n早停触发，已等待 {patience} 轮")
            break
    
    training_time = time.time() - start_time
    print(f"\n训练完成，用时 {training_time:.2f} 秒")
    
    # 测试评估
    print("\n在测试集上评估...")
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
    print(f"测试准确率: {test_acc:.4f}%")
    
    # 更新checkpoint中的test_acc
    checkpoint = torch.load(checkpoint_path)
    checkpoint['test_acc'] = test_acc
    torch.save(checkpoint, checkpoint_path)
    
    print(f"\n✅ ConvnextMicro 训练完成!")
    print(f"📊 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"📊 测试准确率: {test_acc:.2f}%")
    print(f"💾 模型已保存: {checkpoint_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
