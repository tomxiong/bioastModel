"""
Enhanced training script for MIC MobileNetV3 with advanced optimization strategies.

Features:
- Focal Loss for class imbalance handling
- Multi-task learning with weighted losses
- Advanced data augmentation
- Stochastic Weight Averaging (SWA)
- Exponential Moving Average (EMA)
- Gradient clipping and regularization
- Learning rate scheduling with warmup
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torchvision.transforms as transforms
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import argparse

# Simple SWA implementation (alternative to torchcontrib)
class SimpleSWA:
    """Simple Stochastic Weight Averaging implementation."""
    
    def __init__(self, optimizer, swa_start=10):
        self.optimizer = optimizer
        self.swa_start = swa_start
        self.swa_state = {}
        self.n_averaged = 0
    
    def update_swa_params(self, model):
        """Update SWA parameters."""
        if not self.swa_state:
            # Initialize SWA state
            for name, param in model.named_parameters():
                self.swa_state[name] = param.data.clone()
        else:
            # Update SWA parameters
            self.n_averaged += 1
            for name, param in model.named_parameters():
                self.swa_state[name] = (
                    self.swa_state[name] * self.n_averaged + param.data
                ) / (self.n_averaged + 1)
    
    def apply_swa_params(self, model):
        """Apply SWA parameters to model."""
        if self.swa_state:
            for name, param in model.named_parameters():
                param.data = self.swa_state[name]

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.enhanced_mic_mobilenetv3 import create_enhanced_mic_mobilenetv3, MICFocalLoss
from training.dataset import BioastDataset
from core.config.training_configs import get_training_config
from core.config.model_configs import get_model_config

class MICSpecificAugmentation:
    """MIC-specific data augmentation pipeline."""
    
    def __init__(self, p_bubble=0.3, p_turbidity=0.4, p_optical=0.2):
        self.p_bubble = p_bubble
        self.p_turbidity = p_turbidity
        self.p_optical = p_optical
        
        # Base augmentations
        self.base_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            ], p=0.5)
        ])
    
    def simulate_bubble_noise(self, image):
        """Simulate air bubble interference."""
        if np.random.random() < self.p_bubble:
            # Add circular patterns to simulate bubbles
            h, w = image.shape[-2:]
            num_bubbles = np.random.randint(1, 4)
            
            for _ in range(num_bubbles):
                center_x = np.random.randint(10, w-10)
                center_y = np.random.randint(10, h-10)
                radius = np.random.randint(3, 8)
                
                # Create circular mask
                y, x = np.ogrid[:h, :w]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                
                # Apply bubble effect (lighter center, darker ring)
                bubble_effect = np.random.uniform(0.7, 1.3)
                if len(image.shape) == 3:
                    image[:, mask] *= bubble_effect
                else:
                    image[mask] *= bubble_effect
        
        return image
    
    def add_turbidity_variation(self, image):
        """Add turbidity-like variations."""
        if np.random.random() < self.p_turbidity:
            # Add random patches of varying intensity
            h, w = image.shape[-2:]
            num_patches = np.random.randint(2, 6)
            
            for _ in range(num_patches):
                patch_size = np.random.randint(5, 15)
                x = np.random.randint(0, max(1, w - patch_size))
                y = np.random.randint(0, max(1, h - patch_size))
                
                turbidity_factor = np.random.uniform(0.8, 1.2)
                if len(image.shape) == 3:
                    image[:, y:y+patch_size, x:x+patch_size] *= turbidity_factor
                else:
                    image[y:y+patch_size, x:x+patch_size] *= turbidity_factor
        
        return image
    
    def add_optical_noise(self, image):
        """Add optical interference patterns."""
        if np.random.random() < self.p_optical:
            # Add random noise patterns
            noise_intensity = np.random.uniform(0.02, 0.08)
            noise = torch.randn_like(image) * noise_intensity
            image = torch.clamp(image + noise, 0, 1)
        
        return image
    
    def __call__(self, image):
        # Apply base transforms
        image = self.base_transforms(image)
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        
        # Apply MIC-specific augmentations
        image = self.simulate_bubble_noise(image)
        image = self.add_turbidity_variation(image)
        image = self.add_optical_noise(image)
        
        return torch.clamp(image, 0, 1)

class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class EnhancedTrainer:
    """Enhanced trainer with advanced optimization strategies."""
    
    def __init__(self, config: Dict[str, Any], model_name: str = 'enhanced_mic_mobilenetv3'):
        self.config = config
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model setup
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Data setup
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        # Advanced training components
        self.ema = None
        if config.get('ema_decay'):
            self.ema = ExponentialMovingAverage(self.model, config['ema_decay'])
        
        self.swa_model = None
        if config.get('use_swa'):
            self.swa_model = SimpleSWA(self.optimizer, swa_start=int(config['num_epochs'] * config.get('swa_start', 0.8)))
        
        # Training state
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        # Create experiment directory
        self.experiment_dir = self._create_experiment_dir()
    
    def _create_model(self):
        """Create the enhanced model."""
        model = create_enhanced_mic_mobilenetv3(
            num_classes=2,
            dropout_rate=self.config.get('dropout_rate', 0.2)
        )
        return model
    
    def _create_data_loaders(self):
        """Create enhanced data loaders with augmentation."""
        # Define transforms
        train_transform = transforms.Compose([
            MICSpecificAugmentation(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = BioastDataset('bioast_dataset', split='train', transform=train_transform)
        val_dataset = BioastDataset('bioast_dataset', split='val', transform=val_transform)
        test_dataset = BioastDataset('bioast_dataset', split='test', transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_optimizer(self):
        """Create optimizer with weight decay."""
        if self.config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 5e-4),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 5e-4)
            )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.get('scheduler') == 'cosine_with_restarts':
            T_0 = max(1, self.config['num_epochs'] // 4)  # Ensure T_0 is at least 1
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=2,
                eta_min=1e-6
            )
        else:
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.config['num_epochs']),  # Ensure T_max is at least 1
                eta_min=1e-6
            )
        
        return scheduler
    
    def _create_criterion(self):
        """Create loss function."""
        if self.config.get('focal_loss'):
            # Use Focal Loss for class imbalance
            class_weights = torch.tensor(self.config.get('class_weights', [1.0, 1.0]), dtype=torch.float)
            return MICFocalLoss(
                alpha=self.config.get('focal_alpha', 0.75),
                gamma=self.config.get('focal_gamma', 2.0),
                class_weights=class_weights.to(self.device)
            )
        else:
            # Standard Cross Entropy with label smoothing
            return nn.CrossEntropyLoss(
                label_smoothing=self.config.get('label_smoothing', 0.0)
            )
    
    def _create_experiment_dir(self):
        """Create experiment directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"experiments/{self.model_name}_{timestamp}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save configuration
        with open(f"{exp_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return exp_dir
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            
            # Calculate multi-task losses
            losses = {}
            
            # Main classification loss
            main_loss = self.criterion(outputs['classification'], target)
            losses['classification'] = main_loss
            
            # Auxiliary loss if available
            if 'aux_classification' in outputs and self.config.get('auxiliary_loss'):
                aux_loss = self.criterion(outputs['aux_classification'], target)
                losses['aux_classification'] = aux_loss * self.config.get('auxiliary_weight', 0.3)
            
            # Multi-task losses (simplified for training stability)
            total_loss_batch = main_loss
            for key, weight in self.config.get('multitask_weights', {}).items():
                if key in losses:
                    total_loss_batch += weight * losses[key]
            
            # Backward pass with gradient clipping
            total_loss_batch.backward()
            
            if self.config.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )
            
            self.optimizer.step()
            
            # Update EMA
            if self.ema:
                self.ema.update()
            
            # Statistics
            total_loss += total_loss_batch.item()
            _, predicted = outputs['classification'].max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Progress reporting
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {total_loss_batch.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs['classification'], target)
                
                total_loss += loss.item()
                _, predicted = outputs['classification'].max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting enhanced training for {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.config['num_epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            # Warmup learning rate
            if epoch <= self.config.get('warmup_epochs', 0):
                warmup_factor = self.config.get('warmup_factor', 0.1)
                lr_scale = warmup_factor + (1.0 - warmup_factor) * epoch / self.config['warmup_epochs']
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config['learning_rate'] * lr_scale
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch()
            
            # Learning rate scheduling (after warmup)
            if epoch > self.config.get('warmup_epochs', 0):
                self.scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, f"{self.experiment_dir}/best_model.pth")
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Progress report
            print(f'Epoch {epoch}/{self.config["num_epochs"]}:')
            print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  Best Val Acc: {self.best_val_acc:.2f}%')
            print('-' * 60)
        
        # Final evaluation with EMA if available
        if self.ema:
            print("\n🔍 Evaluating with EMA parameters...")
            self.ema.apply_shadow()
            ema_val_loss, ema_val_acc = self.validate_epoch()
            print(f"EMA Validation - Loss: {ema_val_loss:.4f}, Acc: {ema_val_acc:.2f}%")
            
            if ema_val_acc > self.best_val_acc:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': ema_val_acc,
                    'config': self.config,
                    'ema': True
                }, f"{self.experiment_dir}/best_ema_model.pth")
            
            self.ema.restore()
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        print(f"   Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save training history
        with open(f"{self.experiment_dir}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        return self.best_val_acc

def main():
    parser = argparse.ArgumentParser(description='Enhanced MIC MobileNetV3 Training')
    parser.add_argument('--config', type=str, default='enhanced_mic_mobilenetv3_optimized',
                        help='Training configuration name')
    parser.add_argument('--model', type=str, default='enhanced_mic_mobilenetv3',
                        help='Model name')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_training_config(args.config)
    
    # Override parameters if provided
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Create and run trainer
    trainer = EnhancedTrainer(config, args.model)
    best_acc = trainer.train()
    
    print(f"\n🎯 Final Results:")
    print(f"   Model: {args.model}")
    print(f"   Configuration: {args.config}")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()