#!/usr/bin/env python3
"""
MicroViT Trainer for 70x70 Biomedical Images
Transformer optimized for small images (~1.8M params)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import json
import os
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from PIL import Image
import math

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PatchEmbedding(nn.Module):
    """Patch embedding for small images"""
    def __init__(self, img_size=70, patch_size=7, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 10x10 = 100 patches
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, n_patches_h, n_patches_w)
        x = self.projection(x)  # (B, embed_dim, 10, 10)
        x = x.flatten(2)  # (B, embed_dim, 100)
        x = x.transpose(1, 2)  # (B, 100, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """Efficient multi-head attention for small sequences"""
    def __init__(self, embed_dim=192, num_heads=6, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, embed_dim=192, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, embed_dim=192, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MicroViT(nn.Module):
    """
    Micro Vision Transformer optimized for 70x70 biomedical images
    ~1.8M parameters
    """
    
    def __init__(self, img_size=70, patch_size=7, in_channels=3, num_classes=2, 
                 embed_dim=192, depth=8, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token
        x = self.head(cls_token_final)
        
        return x

class CustomDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MicroViTTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_id = "micro_vit"
        self.experiment_id = f"{self.model_id}_{self.timestamp}"
        
        # 训练配置
        self.config = {
            "batch_size": 64,
            "epochs": 50,
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "weight_decay": 0.05,
            "warmup_epochs": 5
        }
        
        # 确保目录存在
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("error_analysis", exist_ok=True)
        
        # 初始化训练记录
        self.training_record = {
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "start_time": datetime.now().isoformat(),
            "model_name": "MicroViT",
            "status": "training",
            "config": self.config,
            "paths": {
                "checkpoint": f"checkpoints/{self.experiment_id}_best.pth",
                "training_log": f"reports/{self.experiment_id}_training.json",
                "performance_report": f"reports/{self.experiment_id}_performance.html",
                "error_analysis": f"error_analysis/{self.experiment_id}_errors.html"
            },
            "metrics": {},
            "error_samples": []
        }
        
        print(f"🔧 Initializing MicroViT trainer")
        print(f"📱 Device: {self.device}")
        print(f"🆔 Experiment ID: {self.experiment_id}")
    
    def prepare_data(self):
        """准备数据加载器"""
        print("📊 Preparing data loaders...")
        
        # 数据变换 - 针对ViT优化
        train_transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 收集样本
        def collect_samples(pos_path, neg_path):
            samples = []
            # 正样本 (label=0)
            if os.path.exists(pos_path):
                for img_file in os.listdir(pos_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        samples.append((os.path.join(pos_path, img_file), 0))
            
            # 负样本 (label=1)  
            if os.path.exists(neg_path):
                for img_file in os.listdir(neg_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        samples.append((os.path.join(neg_path, img_file), 1))
            
            return samples
        
        train_samples = collect_samples('bioast_dataset/positive/train', 'bioast_dataset/negative/train')
        val_samples = collect_samples('bioast_dataset/positive/val', 'bioast_dataset/negative/val')
        test_samples = collect_samples('bioast_dataset/positive/test', 'bioast_dataset/negative/test')
        
        print(f"📊 Dataset sizes - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        # 创建数据集
        train_dataset = CustomDataset(train_samples, train_transform)
        val_dataset = CustomDataset(val_samples, val_transform)
        test_dataset = CustomDataset(test_samples, val_transform)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], 
                                shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], 
                              shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], 
                               shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self):
        """训练模型"""
        print("🚀 Starting MicroViT training...")
        
        # 创建模型
        model = MicroViT(
            img_size=70,
            patch_size=7,
            in_channels=3,
            num_classes=2,
            embed_dim=192,
            depth=8,
            num_heads=6,
            mlp_ratio=4.0,
            dropout=0.1
        ).to(self.device)
        
        # 验证输入尺寸
        test_input = torch.randn(2, 3, 70, 70).to(self.device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"✅ Model verified for 70x70 input. Output shape: {test_output.shape}")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 准备数据
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # 优化器和调度器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'], 
                               weight_decay=self.config['weight_decay'])
        
        # 学习率调度器 - 带warmup
        def lr_lambda(epoch):
            if epoch < self.config['warmup_epochs']:
                return epoch / self.config['warmup_epochs']
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - self.config['warmup_epochs']) / 
                                          (self.config['epochs'] - self.config['warmup_epochs'])))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # 训练循环
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Epoch {epoch+1}/{self.config["epochs"]}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, LR: {current_lr:.6f}')
            
            train_acc = 100. * correct / total
            avg_loss = running_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 验证阶段
            val_acc = self.evaluate_model(model, val_loader)
            val_accuracies.append(val_acc)
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_loss': avg_loss
                }, self.training_record['paths']['checkpoint'])
                print(f"💾 New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # 测试评估
        test_acc = self.evaluate_model(model, test_loader)
        
        # 更新训练记录
        self.training_record['metrics'] = {
            'best_val_accuracy': best_val_acc,
            'final_test_accuracy': test_acc,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # 错误分析
        self.analyze_errors(model, test_loader)
        
        print(f"🎉 Training completed!")
        print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
        print(f"📊 Final test accuracy: {test_acc:.2f}%")
        
        return model
    
    def evaluate_model(self, model, data_loader):
        """评估模型"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def analyze_errors(self, model, test_loader):
        """分析错误样例"""
        print("🔍 Analyzing error samples...")
        model.eval()
        error_samples = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                probabilities = torch.softmax(output, dim=1)
                
                # 收集预测结果
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # 找出错误样例
                for i in range(len(target)):
                    if predicted[i] != target[i]:
                        sample_idx = batch_idx * test_loader.batch_size + i
                        if sample_idx < len(test_loader.dataset.samples):
                            filename = test_loader.dataset.samples[sample_idx][0]
                            
                            error_samples.append({
                                'filename': filename,
                                'true_label': target[i].item(),
                                'predicted_label': predicted[i].item(),
                                'confidence': probabilities[i].max().item()
                            })
        
        self.training_record['error_samples'] = error_samples[:50]
        self.training_record['metrics']['confusion_matrix'] = confusion_matrix(all_targets, all_predictions).tolist()
        
        print(f"🔍 Found {len(error_samples)} error samples")
    
    def generate_reports(self):
        """生成报告"""
        print("📊 Generating reports...")
        
        # 生成JSON报告
        self.training_record['end_time'] = datetime.now().isoformat()
        self.training_record['status'] = 'completed'
        
        with open(self.training_record['paths']['training_log'], 'w', encoding='utf-8') as f:
            json.dump(self.training_record, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        self.generate_html_report()
        
        print(f"📊 Reports generated:")
        print(f"  - JSON: {self.training_record['paths']['training_log']}")
        print(f"  - HTML: {self.training_record['paths']['performance_report']}")
    
    def generate_html_report(self):
        """生成HTML性能报告"""
        metrics = self.training_record['metrics']
        error_samples = self.training_record['error_samples']
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Report - {self.training_record['model_name']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        .error-sample {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .error-sample.false-positive {{ background-color: #ffe6e6; }}
        .error-sample.false-negative {{ background-color: #e6f3ff; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .success {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Training Report: {self.training_record['model_name']}</h1>
            <p><strong>Experiment ID:</strong> {self.experiment_id}</p>
            <p><strong>Training Date:</strong> {self.training_record['start_time']}</p>
            <p><strong>Status:</strong> <span class="success">✅ COMPLETED</span></p>
        </div>
        
        <h2>📊 Performance Metrics</h2>
        <div class="metric-card">
            <h3>🎯 Accuracy Results</h3>
            <p><strong>Best Validation Accuracy:</strong> <span class="success">{metrics.get('best_val_accuracy', 0):.2f}%</span></p>
            <p><strong>Final Test Accuracy:</strong> <span class="success">{metrics.get('final_test_accuracy', 0):.2f}%</span></p>
        </div>
        
        <div class="metric-card">
            <h3>🏗️ Model Information</h3>
            <p><strong>Architecture:</strong> Vision Transformer (MicroViT)</p>
            <p><strong>Total Parameters:</strong> {metrics.get('total_parameters', 0):,}</p>
            <p><strong>Trainable Parameters:</strong> {metrics.get('trainable_parameters', 0):,}</p>
            <p><strong>Model Size:</strong> ~{metrics.get('total_parameters', 0)/1000000:.1f}M parameters</p>
        </div>
        
        <h2>🔍 Error Analysis</h2>
        <p><strong>Total Error Samples:</strong> {len(error_samples)}</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div class="metric-card">
                <h4>❌ False Positives</h4>
                <p>{len([e for e in error_samples if e['predicted_label'] == 1 and e['true_label'] == 0])} samples</p>
            </div>
            <div class="metric-card">
                <h4>❌ False Negatives</h4>
                <p>{len([e for e in error_samples if e['predicted_label'] == 0 and e['true_label'] == 1])} samples</p>
            </div>
        </div>
        
        <h2>📁 File Paths</h2>
        <ul>
            <li><strong>Checkpoint:</strong> {self.training_record['paths']['checkpoint']}</li>
            <li><strong>Training Log:</strong> {self.training_record['paths']['training_log']}</li>
        </ul>
    </div>
</body>
</html>
        """
        
        with open(self.training_record['paths']['performance_report'], 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def update_registry(self):
        """更新模型注册表"""
        print("📝 Updating model registry...")
        
        try:
            # 加载注册表
            with open('model_registry.json', 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # 更新训练记录
            model_info = registry['models'][self.model_id]
            model_info['training_history'].append(self.training_record)
            model_info['latest_training'] = self.training_record
            
            # 保存注册表
            registry['registry_info']['last_updated'] = datetime.now().isoformat()
            with open('model_registry.json', 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            
            print("✅ Registry updated successfully")
        except Exception as e:
            print(f"⚠️ Failed to update registry: {e}")
    
    def run_complete_training(self):
        """运行完整的训练流程"""
        try:
            print(f"🚀 Starting complete training pipeline for {self.model_id}")
            
            # 训练模型
            model = self.train_model()
            
            # 生成报告
            self.generate_reports()
            
            # 更新注册表
            self.update_registry()
            
            print(f"✅ Training pipeline completed successfully!")
            print(f"📊 Best validation accuracy: {self.training_record['metrics']['best_val_accuracy']:.2f}%")
            print(f"📊 Final test accuracy: {self.training_record['metrics']['final_test_accuracy']:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self.training_record['status'] = 'failed'
            self.training_record['error'] = str(e)
            return False

def main():
    trainer = MicroViTTrainer()
    success = trainer.run_complete_training()
    
    if success:
        print(f"🎉 MicroViT training completed successfully!")
        print(f"📊 Performance report: {trainer.training_record['paths']['performance_report']}")
    else:
        print(f"❌ MicroViT training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()