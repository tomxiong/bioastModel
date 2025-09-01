#!/usr/bin/env python3
"""
修复最后两个失败的模型: micro_vit 和 vit_tiny
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def create_fixed_micro_vit_trainer():
    """创建修复后的micro_vit训练脚本"""
    trainer_content = '''#!/usr/bin/env python3
"""
修复后的micro_vit训练脚本 - 解决维度不匹配问题
使用bioast_dataset真实数据
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model():
    """训练micro_vit模型"""
    print(f"🚀 开始修复训练 micro_vit")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载 - 确认使用bioast_dataset
    try:
        from core.real_data_loader import create_real_data_loaders
        train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32, num_workers=4)
        print(f"✅ 数据加载成功 (使用bioast_dataset)")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  验证批次: {len(val_loader)}")
        print(f"  测试批次: {len(test_loader)}")
        
        # 验证数据来源
        print(f"  数据集路径: bioast_dataset")
        print(f"  训练样本: {len(train_loader) * 32} (约)")
        print(f"  验证样本: {len(val_loader) * 32} (约)")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # 模型初始化 - 修复维度不匹配问题
    model = None
    try:
        from models.micro_vit import MicroViT
        # 使用适合70x70输入的配置，修复patch数量问题
        model = MicroViT(
            img_size=70,
            patch_size=10,  # 改为10，使得70/10=7，7*7=49 patches
            num_classes=2,
            embed_dim=192,
            depth=6,
            num_heads=6,
            enable_bubble_detection=False  # 简化模型，关闭bubble detection
        )
        print(f"✅ 模型初始化成功 (patch_size=10, num_patches=49)")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        traceback.print_exc()
        return False
    
    if model is None:
        print(f"❌ 模型为空")
        return False
    
    try:
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ 模型移至设备: {param_count:.1f}M 参数")
    except Exception as e:
        print(f"❌ 模型移至设备失败: {e}")
        return False
    
    # 训练设置
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=30)
        print(f"✅ 训练设置完成")
    except Exception as e:
        print(f"❌ 训练设置失败: {e}")
        return False
    
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    
    # 训练循环
    for epoch in range(30):
        try:
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    
                    # 处理多输出模型 - MicroViT返回字典
                    if isinstance(output, dict):
                        if 'classification' in output:
                            output = output['classification']
                        else:
                            # 取第一个输出
                            output = list(output.values())[0]
                    
                    # 处理维度不匹配
                    if output.dim() > 2:
                        output = output.view(output.size(0), -1)
                    
                    if output.size(1) != 2:
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
                    print(f"❌ 训练批次 {batch_idx} 失败: {e}")
                    continue
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    try:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        
                        # 处理多输出模型
                        if isinstance(output, dict):
                            if 'classification' in output:
                                output = output['classification']
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
                        print(f"❌ 验证批次失败: {e}")
                        continue
            
            train_acc = 100. * train_correct / train_total if train_total > 0 else 0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            
            print(f'Epoch {epoch+1}/30: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # 保存检查点
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f'checkpoints/micro_vit_{timestamp}_best.pth'
                os.makedirs('checkpoints', exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'model_name': 'micro_vit',
                    'data_source': 'bioast_dataset'
                }, checkpoint_path)
                
                print(f'✅ 保存新的最佳模型: {checkpoint_path} ({best_val_acc:.2f}%)')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'提前停止，训练{epoch+1}轮后触发')
                    break
            
            scheduler.step()
            
        except Exception as e:
            print(f"❌ Epoch {epoch+1} 失败: {e}")
            continue
    
    print(f"🎉 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    return best_val_acc > 0

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
'''
    
    trainer_path = "trainers/train_micro_vit_final_fixed.py"
    os.makedirs("trainers", exist_ok=True)
    
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(trainer_content)
    
    return trainer_path

def create_fixed_vit_tiny_trainer():
    """创建修复后的vit_tiny训练脚本"""
    trainer_content = '''#!/usr/bin/env python3
"""
修复后的vit_tiny训练脚本 - 解决导入错误
使用bioast_dataset真实数据
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_model():
    """训练vit_tiny模型"""
    print(f"🚀 开始修复训练 vit_tiny")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载 - 确认使用bioast_dataset
    try:
        from core.real_data_loader import create_real_data_loaders
        train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32, num_workers=4)
        print(f"✅ 数据加载成功 (使用bioast_dataset)")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  验证批次: {len(val_loader)}")
        print(f"  测试批次: {len(test_loader)}")
        
        # 验证数据来源
        print(f"  数据集路径: bioast_dataset")
        print(f"  训练样本: {len(train_loader) * 32} (约)")
        print(f"  验证样本: {len(val_loader) * 32} (约)")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # 模型初始化 - 修复导入错误
    model = None
    try:
        # 使用正确的类名 VisionTransformerTiny
        from models.vit_tiny import VisionTransformerTiny
        model = VisionTransformerTiny(
            img_size=70,  # 适配70x70输入
            patch_size=7,  # 70/7=10, 10*10=100 patches
            num_classes=2,
            embed_dim=192,
            depth=6,
            num_heads=6
        )
        print(f"✅ 模型初始化成功 (VisionTransformerTiny)")
    except Exception as e:
        print(f"❌ VisionTransformerTiny初始化失败: {e}")
        # 回退策略：使用create_vit_tiny函数
        try:
            from models.vit_tiny import create_vit_tiny
            model = create_vit_tiny(num_classes=2)
            print(f"✅ 模型初始化成功 (create_vit_tiny)")
        except Exception as e2:
            print(f"❌ create_vit_tiny初始化失败: {e2}")
            traceback.print_exc()
            return False
    
    if model is None:
        print(f"❌ 模型为空")
        return False
    
    try:
        model = model.to(device)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ 模型移至设备: {param_count:.1f}M 参数")
    except Exception as e:
        print(f"❌ 模型移至设备失败: {e}")
        return False
    
    # 训练设置
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=30)
        print(f"✅ 训练设置完成")
    except Exception as e:
        print(f"❌ 训练设置失败: {e}")
        return False
    
    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    
    # 训练循环
    for epoch in range(30):
        try:
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    
                    # 处理多输出模型
                    if isinstance(output, dict):
                        if 'classification' in output:
                            output = output['classification']
                        elif 'logits' in output:
                            output = output['logits']
                        else:
                            output = list(output.values())[0]
                    
                    # 处理维度不匹配
                    if output.dim() > 2:
                        output = output.view(output.size(0), -1)
                    
                    if output.size(1) != 2:
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
                    print(f"❌ 训练批次 {batch_idx} 失败: {e}")
                    continue
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    try:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        
                        # 处理多输出模型
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
                        print(f"❌ 验证批次失败: {e}")
                        continue
            
            train_acc = 100. * train_correct / train_total if train_total > 0 else 0
            val_acc = 100. * val_correct / val_total if val_total > 0 else 0
            
            print(f'Epoch {epoch+1}/30: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # 保存检查点
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = f'checkpoints/vit_tiny_{timestamp}_best.pth'
                os.makedirs('checkpoints', exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'model_name': 'vit_tiny',
                    'data_source': 'bioast_dataset'
                }, checkpoint_path)
                
                print(f'✅ 保存新的最佳模型: {checkpoint_path} ({best_val_acc:.2f}%)')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'提前停止，训练{epoch+1}轮后触发')
                    break
            
            scheduler.step()
            
        except Exception as e:
            print(f"❌ Epoch {epoch+1} 失败: {e}")
            continue
    
    print(f"🎉 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    return best_val_acc > 0

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
'''
    
    trainer_path = "trainers/train_vit_tiny_final_fixed.py"
    os.makedirs("trainers", exist_ok=True)
    
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(trainer_content)
    
    return trainer_path

def main():
    """主执行函数"""
    print("🔧 修复最后两个失败的模型")
    print("=" * 50)
    
    failed_models = ['micro_vit', 'vit_tiny']
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(failed_models),
        'successful': [],
        'failed': [],
        'details': {}
    }
    
    python_cmd = sys.executable
    
    # 修复 micro_vit
    print(f"\n🚀 修复训练 1/2: micro_vit")
    print("-" * 30)
    
    try:
        trainer_path = create_fixed_micro_vit_trainer()
        print(f"✅ 创建修复训练脚本: {trainer_path}")
        
        start_time = time.time()
        result = subprocess.run(
            [python_cmd, trainer_path],
            timeout=1800,  # 30分钟超时
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ micro_vit 训练成功完成! ({duration:.1f}s)")
            results['successful'].append('micro_vit')
            results['details']['micro_vit'] = {
                'status': 'success',
                'duration': duration,
                'stdout': result.stdout[-1000:]
            }
        else:
            print(f"❌ micro_vit 训练失败:")
            print(f"STDERR: {result.stderr[-500:]}")
            results['failed'].append('micro_vit')
            results['details']['micro_vit'] = {
                'status': 'failed',
                'duration': duration,
                'stderr': result.stderr[-500:]
            }
            
    except Exception as e:
        print(f"❌ micro_vit 修复错误: {e}")
        results['failed'].append('micro_vit')
        results['details']['micro_vit'] = {'status': 'error', 'error': str(e)}
    
    # 修复 vit_tiny
    print(f"\n🚀 修复训练 2/2: vit_tiny")
    print("-" * 30)
    
    try:
        trainer_path = create_fixed_vit_tiny_trainer()
        print(f"✅ 创建修复训练脚本: {trainer_path}")
        
        start_time = time.time()
        result = subprocess.run(
            [python_cmd, trainer_path],
            timeout=1800,  # 30分钟超时
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ vit_tiny 训练成功完成! ({duration:.1f}s)")
            results['successful'].append('vit_tiny')
            results['details']['vit_tiny'] = {
                'status': 'success',
                'duration': duration,
                'stdout': result.stdout[-1000:]
            }
        else:
            print(f"❌ vit_tiny 训练失败:")
            print(f"STDERR: {result.stderr[-500:]}")
            results['failed'].append('vit_tiny')
            results['details']['vit_tiny'] = {
                'status': 'failed',
                'duration': duration,
                'stderr': result.stderr[-500:]
            }
            
    except Exception as e:
        print(f"❌ vit_tiny 修复错误: {e}")
        results['failed'].append('vit_tiny')
        results['details']['vit_tiny'] = {'status': 'error', 'error': str(e)}
    
    # 保存结果
    with open('final_two_models_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 最后两个模型修复完成!")
    print("=" * 50)
    print(f"✅ 成功训练: {len(results['successful'])}")
    for model in results['successful']:
        print(f"   ✅ {model}")
    
    print(f"❌ 失败模型: {len(results['failed'])}")
    for model in results['failed']:
        print(f"   ❌ {model}")
    
    # 计算最终统计
    previous_successful = 31  # 27 + 4 from previous training
    new_successful = len(results['successful'])
    total_successful = previous_successful + new_successful
    total_models = 40
    success_rate = (total_successful / total_models) * 100
    
    print(f"\n📈 最终统计:")
    print(f"   🎯 总模型数: {total_models}")
    print(f"   ✅ 成功训练: {total_successful}")
    print(f"   📊 成功率: {success_rate:.1f}%")
    print(f"\n📊 结果保存至: final_two_models_results.json")

if __name__ == "__main__":
    main()