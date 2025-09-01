#!/usr/bin/env python3
"""
Simplified Air Bubble Detector Trainer
专门训练SimplifiedAirBubbleDetector模型的脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector

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

class SimplifiedAirBubbleDetectorTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_id = "simplified_airbubble_detector"
        self.experiment_id = f"{self.model_id}_{self.timestamp}"
        
        # 训练配置
        self.config = {
            "batch_size": 16,
            "epochs": 30,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "weight_decay": 1e-4
        }
#!/usr/bin/env python3
"""
Simplified Air Bubble Detector Trainer
专门训练SimplifiedAirBubbleDetector模型的脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector

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

class SimplifiedAirBubbleDetectorTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_id = "simplified_airbubble_detector"
        self.experiment_id = f"{self.model_id}_{self.timestamp}"
        
#!/usr/bin/env python3
"""
Simplified Air Bubble Detector Trainer
专门训练SimplifiedAirBubbleDetector模型的脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
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

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector

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

class SimplifiedAirBubbleDetectorTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_id = "simplified_airbubble_detector"
        self.experiment_id = f"{self.model_id}_{self.timestamp}"
        
        # 训练配置
        self.config = {
            "batch_size": 128,
            "epochs": 30,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "weight_decay": 1e-4
        }
        
        # 初始化训练记录
        self.training_record = {
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "start_time": datetime.now().isoformat(),
            "model_name": "SimplifiedAirBubbleDetector",
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
        
        print(f"🔧 Initializing trainer for {self.model_id}")
        print(f"📱 Device: {self.device}")
        print(f"🆔 Experiment ID: {self.experiment_id}")
    
    def prepare_data(self):
        """准备数据加载器"""
        print("📊 Preparing data loaders...")
        
        # 数据变换
        train_transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 检查数据集路径
        dataset_paths = {
            'train_pos': 'bioast_dataset/positive/train',
            'train_neg': 'bioast_dataset/negative/train',
            'val_pos': 'bioast_dataset/positive/val',
            'val_neg': 'bioast_dataset/negative/val',
            'test_pos': 'bioast_dataset/positive/test',
            'test_neg': 'bioast_dataset/negative/test'
        }
        
        for name, path in dataset_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset path not found: {path}")
        
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
        
        train_samples = collect_samples(dataset_paths['train_pos'], dataset_paths['train_neg'])
        val_samples = collect_samples(dataset_paths['val_pos'], dataset_paths['val_neg'])
        test_samples = collect_samples(dataset_paths['test_pos'], dataset_paths['test_neg'])
        
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
        print("🚀 Starting model training...")
        
        # 创建模型
        model = SimplifiedAirBubbleDetector(num_classes=2).to(self.device)
        
        # 验证输入尺寸
        test_input = torch.randn(1, 3, 70, 70).to(self.device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"✅ Model verified for 70x70 input. Output shape: {test_output.shape}")
        
        # 准备数据
        train_loader, val_loader, test_loader = self.prepare_data()
        
        # 优化器和调度器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'], 
                               weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])
        
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
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{self.config["epochs"]}, Batch {batch_idx}, '
                          f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
            
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
        
        self.training_record['error_samples'] = error_samples[:50]  # 限制错误样例数量
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
        .info {{ color: #17a2b8; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Training Report: {self.training_record['model_name']}</h1>
            <p><strong>Experiment ID:</strong> {self.experiment_id}</p>
            <p><strong>Training Date:</strong> {self.training_record['start_time']}</p>
            <p><strong>Status:</strong> <span class="success">✅ {self.training_record['status'].upper()}</span></p>
        </div>
        
        <h2>📊 Performance Metrics</h2>
        <div class="metric-card">
            <h3>🎯 Accuracy Results</h3>
            <p><strong>Best Validation Accuracy:</strong> <span class="success">{metrics.get('best_val_accuracy', 0):.2f}%</span></p>
            <p><strong>Final Test Accuracy:</strong> <span class="success">{metrics.get('final_test_accuracy', 0):.2f}%</span></p>
        </div>
        
        <div class="metric-card">
            <h3>🏗️ Model Information</h3>
            <p><strong>Total Parameters:</strong> {metrics.get('total_parameters', 0):,}</p>
            <p><strong>Trainable Parameters:</strong> {metrics.get('trainable_parameters', 0):,}</p>
            <p><strong>Model Size:</strong> ~{metrics.get('total_parameters', 0)/1000:.0f}K parameters</p>
        </div>
        
        <h2>⚙️ Training Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Batch Size</td><td>{self.config['batch_size']}</td></tr>
            <tr><td>Learning Rate</td><td>{self.config['learning_rate']}</td></tr>
            <tr><td>Epochs</td><td>{self.config['epochs']}</td></tr>
            <tr><td>Optimizer</td><td>{self.config['optimizer']}</td></tr>
            <tr><td>Scheduler</td><td>{self.config['scheduler']}</td></tr>
        </table>
        
        <h2>🔍 Error Analysis</h2>
        <p><strong>Total Error Samples:</strong> {len(error_samples)}</p>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div class="metric-card">
                <h4>❌ False Positives</h4>
                <p>{len([e for e in error_samples if e['predicted_label'] == 1 and e['true_label'] == 0])} samples</p>
                <small>Predicted Positive, Actually Negative</small>
            </div>
            <div class="metric-card">
                <h4>❌ False Negatives</h4>
                <p>{len([e for e in error_samples if e['predicted_label'] == 0 and e['true_label'] == 1])} samples</p>
                <small>Predicted Negative, Actually Positive</small>
            </div>
        </div>
        
        <h3>📋 Error Samples (First 10)</h3>
        """
        
        for i, error in enumerate(error_samples[:10]):
            error_type = "false-positive" if error['predicted_label'] == 1 and error['true_label'] == 0 else "false-negative"
            html_content += f"""
        <div class="error-sample {error_type}">
            <strong>Sample {i+1}:</strong> {os.path.basename(error['filename'])}<br>
            <strong>True:</strong> {'Positive' if error['true_label'] == 0 else 'Negative'} | 
            <strong>Predicted:</strong> {'Positive' if error['predicted_label'] == 0 else 'Negative'} | 
            <strong>Confidence:</strong> {error['confidence']:.4f}
        </div>
            """
        
        html_content += f"""
        
        <h2>📁 File Paths</h2>
        <ul>
            <li><strong>Checkpoint:</strong> {self.training_record['paths']['checkpoint']}</li>
            <li><strong>Training Log:</strong> {self.training_record['paths']['training_log']}</li>
            <li><strong>Performance Report:</strong> {self.training_record['paths']['performance_report']}</li>
        </ul>
        
        <div style="margin-top: 30px; padding: 15px; background-color: #e8f5e8; border-radius: 5px;">
            <h3>🎉 Training Summary</h3>
            <p>The <strong>{self.training_record['model_name']}</strong> model has been successfully trained and evaluated on 70x70 biomedical images. 
            The model achieved <strong>{metrics.get('best_val_accuracy', 0):.2f}%</strong> validation accuracy and 
            <strong>{metrics.get('final_test_accuracy', 0):.2f}%</strong> test accuracy.</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(self.training_record['paths']['performance_report'], 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def update_registry(self):
        """更新模型注册表"""
        print("📝 Updating model registry...")
        
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
            self.training_record['status'] = 'failed'
            self.training_record['error'] = str(e)
            return False

def main():
    trainer = SimplifiedAirBubbleDetectorTrainer()
    success = trainer.run_complete_training()
    
    if success:
        print(f"🎉 Training completed successfully!")
        print(f"📊 Performance report: {trainer.training_record['paths']['performance_report']}")
    else:
        print(f"❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()