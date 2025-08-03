"""
Enhanced Models Training Script for MIC Testing.

This script trains the three new enhanced models:
1. MIC_MobileNetV3 - Lightweight SE attention model
2. Micro-ViT - Micro Vision Transformer
3. AirBubble_HybridNet - Air bubble aware hybrid network

All models are specifically designed for 70x70 MIC testing images.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from training.dataset import create_data_loaders
from training.trainer import ModelTrainer
from core.config import get_experiment_path, create_experiment_structure, DATA_DIR
"""
Enhanced Models Training Script for MIC Testing.

This script trains the three new enhanced models:
1. MIC_MobileNetV3 - Lightweight SE attention model
2. Micro-ViT - Micro Vision Transformer
3. AirBubble_HybridNet - Air bubble aware hybrid network

All models are specifically designed for 70x70 MIC testing images.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

"""
Enhanced Models Training Script for MIC Testing.

This script trains the three new enhanced models:
1. MIC_MobileNetV3 - Lightweight SE attention model
2. Micro-ViT - Micro Vision Transformer
3. AirBubble_HybridNet - Air bubble aware hybrid network

All models are specifically designed for 70x70 MIC testing images.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config.training_configs import get_training_config
from torchvision import datasets, transforms
from PIL import Image
"""
Enhanced Models Training Script for MIC Testing.

This script trains the three new enhanced models:
1. MIC_MobileNetV3 - Lightweight SE attention model
2. Micro-ViT - Micro Vision Transformer
3. AirBubble_HybridNet - Air bubble aware hybrid network

All models are specifically designed for 70x70 MIC testing images.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

"""
Enhanced Models Training Script for MIC Testing.

This script trains the three new enhanced models:
1. MIC_MobileNetV3 - Lightweight SE attention model
2. Micro-ViT - Micro Vision Transformer
3. AirBubble_HybridNet - Air bubble aware hybrid network

All models are specifically designed for 70x70 MIC testing images.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.data.dataset import BioDataset
from core.training.trainer import ModelTrainer
from core.evaluation.evaluator import ModelEvaluator
from core.utils.metrics import MetricsCalculator
from core.utils.visualization import TrainingVisualizer
from core.config.training_configs import get_training_config

# Import enhanced models
from models.mic_mobilenetv3 import create_mic_mobilenetv3, MODEL_CONFIG as MIC_MOBILENET_CONFIG
from models.micro_vit import create_micro_vit, MODEL_CONFIG as MICRO_VIT_CONFIG
from models.airbubble_hybrid_net import create_airbubble_hybrid_net, MODEL_CONFIG as AIRBUBBLE_CONFIG

class EnhancedModelTrainer:
    """Trainer for enhanced MIC models."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(f"experiments/experiment_{timestamp}")
        self.model_dir = self.experiment_dir / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Enhanced Model Training: {model_name}")
        print("=" * 50)
        print(f"📱 Device: {self.device}")
        print(f"📊 Batch size: {config['batch_size']}")
        print(f"🔄 Max epochs: {config['epochs']}")
        print(f"📁 Experiment path: {self.model_dir}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def create_model(self) -> nn.Module:
        """Create the specified enhanced model."""
        print(f"🏗️ Creating {self.model_name} model...")
        
        if self.model_name == 'mic_mobilenetv3':
            model = create_mic_mobilenetv3(**MIC_MOBILENET_CONFIG['default_params'])
        elif self.model_name == 'micro_vit':
            model = create_micro_vit(**MICRO_VIT_CONFIG['default_params'])
        elif self.model_name == 'airbubble_hybrid_net':
            model = create_airbubble_hybrid_net(**AIRBUBBLE_CONFIG['default_params'])
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model_info = model.get_model_info()
        print(f"📊 Model parameters: {model_info['total_parameters']}")
        
        return model.to(self.device)
    
    def create_data_loaders(self):
        """Create data loaders."""
        print("📂 Loading datasets...")
        
        # Use existing data loader creation function
        data_loaders = create_data_loaders(
            str(DATA_DIR),
            batch_size=self.config['batch_size'],
            num_workers=0
        )
        
        self.train_loader = data_loaders['train']
        self.val_loader = data_loaders['val'] 
        self.test_loader = data_loaders['test']
        
        # Print dataset info
        print(f"加载 train 数据集: {len(self.train_loader.dataset)} 个样本")
        print(f"加载 val 数据集: {len(self.val_loader.dataset)} 个样本")
        print(f"加载 test 数据集: {len(self.test_loader.dataset)} 个样本")
    
    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler."""
        # Optimizer
        if self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        # Scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['learning_rate'] * 0.01
            )
        elif self.config['scheduler'] == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config['epochs'] // 3,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_name in ['micro_vit', 'airbubble_hybrid_net']:
                # Multi-task models
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    logits = outputs['classification']
                else:
                    logits = outputs
            else:
                # Single-task models
                logits = self.model(data)
            
            # Compute loss
            loss = self.criterion(logits, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 20 == 0:
                current_acc = 100.0 * correct / total
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} - "
                      f"Loss: {loss.item():.4f} Acc: {current_acc:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> tuple:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if self.model_name in ['micro_vit', 'airbubble_hybrid_net']:
                    # Multi-task models
                    outputs = self.model(data)
                    if isinstance(outputs, dict):
                        logits = outputs['classification']
                    else:
                        logits = outputs
                else:
                    # Single-task models
                    logits = self.model(data)
                
                # Compute loss
                loss = self.criterion(logits, target)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.model_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.model_dir / 'best_model.pth')
            torch.save(self.model.state_dict(), self.model_dir / 'best_model_weights.pth')
    
    def save_training_history(self):
        """Save training history."""
        with open(self.model_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def train(self):
        """Main training loop."""
        print("🚀 Starting training...")
        print(f"训练样本: {len(self.train_loader.dataset)} 验证样本: {len(self.val_loader.dataset)}")
        print(f"学习率: {self.config['learning_rate']} 权重衰减: {self.config['weight_decay']}")
        print(f"调度器: {self.config['scheduler']} 早停耐心: {self.config.get('patience', 10)}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config['learning_rate']
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_times'].append(epoch_time)
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Print epoch results
            print(f"Epoch {epoch + 1}/{self.config['epochs']} ({epoch_time:.1f}s) - "
                  f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} - "
                  f"LR: {current_lr:.6f}")
            
            if is_best:
                print(f"🎯 New best validation accuracy: {val_acc:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Save training history
            self.save_training_history()
            
            # Early stopping
            patience = self.config.get('patience', 10)
            if self.patience_counter >= patience:
                print(f"⏹️ Early stopping triggered after {patience} epochs without improvement")
                break
        
        total_time = time.time() - start_time
        print(f"✅ Training completed in {total_time:.1f}s")
        print(f"🏆 Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch + 1}")
        
        return self.best_val_acc

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Enhanced MIC Models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['mic_mobilenetv3', 'micro_vit', 'airbubble_hybrid_net'],
                       help='Model to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    
    args = parser.parse_args()
    
    # Get model configuration
    model_configs = {
        'mic_mobilenetv3': MIC_MOBILENET_CONFIG,
        'micro_vit': MICRO_VIT_CONFIG,
        'airbubble_hybrid_net': AIRBUBBLE_CONFIG
    }
    
    model_config = model_configs[args.model]
    training_config = model_config['training_params'].copy()
    
    # Override with command line arguments
    training_config['batch_size'] = args.batch_size
    training_config['epochs'] = args.epochs
    if args.lr is not None:
        training_config['learning_rate'] = args.lr
    
    # Add early stopping patience
    training_config['patience'] = 10
    
    # Create trainer and train
    trainer = EnhancedModelTrainer(args.model, training_config)
    
    # Setup training components
    trainer.model = trainer.create_model()
    trainer.create_data_loaders()
    trainer.create_optimizer_and_scheduler()
    
    # Train the model
    best_acc = trainer.train()
    
    print(f"\n🎉 Training completed for {args.model}")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()