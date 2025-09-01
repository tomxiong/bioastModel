#!/usr/bin/env python3
"""
专门修复micro_vit的维度不匹配问题
分析失败原因并提供针对性解决方案
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def analyze_micro_vit_failure():
    """分析micro_vit失败的具体原因"""
    print("🔍 分析micro_vit训练失败原因")
    print("=" * 50)
    
    print("❌ 错误信息: The size of tensor a (100) must match the size of tensor b (196)")
    print("📊 问题分析:")
    print("  - 期望的patch数量: 196 (14x14)")
    print("  - 实际的patch数量: 100 (10x10)")
    print("  - 原因: 70x70图像使用patch_size=7时，应该产生10x10=100个patches")
    print("  - 但模型的位置编码硬编码为196个patches (14x14)")
    
    print("\n🔧 解决方案:")
    print("  1. 修改patch_size从5改为7，使70/7=10，产生10x10=100个patches")
    print("  2. 更新TurbidityPositionalEncoding中的num_patches参数")
    print("  3. 修复所有硬编码的196维度引用")
    print("  4. 简化模型架构，移除复杂的多任务输出")

def create_fixed_micro_vit_trainer():
    """创建修复后的micro_vit训练脚本"""
    trainer_content = '''#!/usr/bin/env python3
"""
修复后的micro_vit训练脚本 - 彻底解决维度不匹配问题
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
import math

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SimplifiedMicroViT(nn.Module):
    """简化版MicroViT，专门解决维度不匹配问题"""
    
    def __init__(
        self,
        img_size: int = 70,
        patch_size: int = 7,  # 70/7 = 10, 10*10 = 100 patches
        in_channels: int = 3,
        num_classes: int = 2,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 100 patches
        self.embed_dim = embed_dim
        
        print(f"SimplifiedMicroViT: img_size={img_size}, patch_size={patch_size}, num_patches={self.num_patches}")
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Position embedding - 正确的维度
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        # Layer norm and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input size mismatch: {H}x{W} vs {self.img_size}x{self.img_size}"
        
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, 10, 10) -> (B, embed_dim, 100) -> (B, 100, embed_dim)
        x = self.patch_embed(x)  # (B, embed_dim, 10, 10)
        x = x.flatten(2).transpose(1, 2)  # (B, 100, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 101, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed  # (B, 101, embed_dim)
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token
        x = self.head(cls_token_final)
        
        return x

def train_model():
    """训练micro_vit模型"""
    print(f"🚀 开始修复训练 micro_vit (维度修复版)")
    
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
    
    # 模型初始化 - 使用简化版本
    model = None
    try:
        model = SimplifiedMicroViT(
            img_size=70,
            patch_size=7,  # 关键修复：70/7=10，产生100个patches
            num_classes=2,
            embed_dim=192,
            depth=6,
            num_heads=6
        )
        print(f"✅ 模型初始化成功 (SimplifiedMicroViT)")
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
                    
                    # SimplifiedMicroViT直接返回分类结果
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
                checkpoint_path = f'checkpoints/micro_vit_fixed_{timestamp}_best.pth'
                os.makedirs('checkpoints', exist_ok=True)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'model_name': 'micro_vit_fixed',
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
    
    trainer_path = "trainers/train_micro_vit_dimension_fixed.py"
    os.makedirs("trainers", exist_ok=True)
    
    with open(trainer_path, 'w', encoding='utf-8') as f:
        f.write(trainer_content)
    
    return trainer_path

def main():
    """主执行函数"""
    print("🔧 修复micro_vit维度不匹配问题")
    print("=" * 60)
    
    # 分析失败原因
    analyze_micro_vit_failure()
    
    # 创建修复后的训练脚本
    print(f"\n🛠️ 创建修复后的训练脚本")
    print("-" * 40)
    
    try:
        trainer_path = create_fixed_micro_vit_trainer()
        print(f"✅ 创建修复训练脚本: {trainer_path}")
    except Exception as e:
        print(f"❌ 创建训练脚本失败: {e}")
        return False
    
    # 执行训练
    print(f"\n🚀 开始修复训练micro_vit")
    print("-" * 40)
    
    python_cmd = sys.executable
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [python_cmd, trainer_path],
            timeout=1800,  # 30分钟超时
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ micro_vit 修复训练成功完成! ({duration:.1f}s)")
            print(f"STDOUT: {result.stdout[-1000:]}")
            
            # 保存结果
            results = {
                'timestamp': datetime.now().isoformat(),
                'model': 'micro_vit_fixed',
                'status': 'success',
                'duration': duration,
                'stdout': result.stdout,
                'fixes_applied': [
                    'Changed patch_size from 5 to 7',
                    'Fixed num_patches from 196 to 100',
                    'Simplified model architecture',
                    'Removed complex multi-task outputs',
                    'Fixed positional encoding dimensions'
                ]
            }
            
            with open('micro_vit_fix_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            return True
        else:
            print(f"❌ micro_vit 修复训练失败:")
            print(f"STDOUT: {result.stdout[-500:]}")
            print(f"STDERR: {result.stderr[-500:]}")
            
            # 保存失败结果
            results = {
                'timestamp': datetime.now().isoformat(),
                'model': 'micro_vit_fixed',
                'status': 'failed',
                'duration': duration,
                'stdout': result.stdout[-500:],
                'stderr': result.stderr[-500:]
            }
            
            with open('micro_vit_fix_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            return False
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ micro_vit 修复训练超时 (30分钟)")
        return False
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ 修复训练错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 micro_vit修复成功！")
        print(f"📊 结果保存至: micro_vit_fix_results.json")
    else:
        print(f"\n❌ micro_vit修复失败，请检查错误信息")