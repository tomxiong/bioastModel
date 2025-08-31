"""
MobileNetV5 Trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from models import create_mobilenetv5
from .dataset import create_dataloaders


class MobileNetV5Trainer:
    """Trainer for MobileNetV5 models"""
    
    def __init__(self, model_name: str = 'mobilenetv5', data_dir: str = '../bioast_dataset',
                 output_dir: str = 'experiments', device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 32, num_epochs: int = 50, learning_rate: float = 0.001,
                 weight_decay: float = 1e-4, early_stopping_patience: int = 10):
        
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        
        # Create model
        self.model = create_mobilenetv5(model_name, num_classes=2, input_size=70)
        self.model.to(device)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir, batch_size=batch_size, image_size=70
        )
        
        # Setup training components for multi-task learning
        self.main_criterion = nn.CrossEntropyLoss()
        self.aux_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Loss weights for multi-task learning
        self.main_loss_weight = 1.0
        self.pore_loss_weight = 0.3
        self.colony_loss_weight = 0.3
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with multi-task learning"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass - model returns dictionary
            outputs_dict = self.model(images)
            main_outputs = outputs_dict['classification']
            
            # Calculate main classification loss
            main_loss = self.main_criterion(main_outputs, labels)
            
            # Calculate auxiliary losses if available
            total_loss = main_loss
            loss_components = {'main': main_loss.item()}
            
            # Pore detection loss
            if 'pore_classification' in outputs_dict:
                # For pore detection, we use a simple heuristic:
                # If label is 0 (negative), it's likely a pore
                pore_labels = (labels == 0).long()
                pore_loss = self.aux_criterion(outputs_dict['pore_classification'], pore_labels)
                total_loss += self.pore_loss_weight * pore_loss
                loss_components['pore'] = pore_loss.item()
            
            # Colony detection loss
            if 'colony_classification' in outputs_dict:
                # For colony detection, positive samples are colonies
                colony_labels = labels.clone()
                colony_loss = self.aux_criterion(outputs_dict['colony_classification'], colony_labels)
                total_loss += self.colony_loss_weight * colony_loss
                loss_components['colony'] = colony_loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += total_loss.item()
            _, predicted = main_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with detailed loss info
            loss_str = f'{running_loss/(batch_idx+1):.4f}'
            if len(loss_components) > 1:
                aux_loss = sum(v for k, v in loss_components.items() if k != 'main')
                loss_str += f' (aux: {aux_loss:.4f})'
            
            progress_bar.set_postfix({
                'Loss': loss_str,
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model with multi-task outputs"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass - model returns dictionary
                outputs_dict = self.model(images)
                main_outputs = outputs_dict['classification']
                
                # Calculate main classification loss (validation uses only main task)
                loss = self.main_criterion(main_outputs, labels)
                
                running_loss += loss.item()
                _, predicted = main_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'config': {
                'model_name': self.model_name,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs
            }
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.output_dir / f'{self.model_name}_latest.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.output_dir / f'{self.model_name}_best.pth')
    
    def train(self):
        """Full training loop"""
        print(f"Starting training for {self.model_name}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}")
            
            # Early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict()
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best validation accuracy: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch)
                
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save final results
        self.save_results()
        
        return self.model, self.history
    
    def save_results(self):
        """Save training results"""
        results = {
            'model_name': self.model_name,
            'best_val_acc': self.best_val_acc,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'training_history': self.history,
            'config': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'early_stopping_patience': self.early_stopping_patience
            }
        }
        
        with open(self.output_dir / f'{self.model_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {self.output_dir / f'{self.model_name}_results.json'}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MobileNetV5')
    parser.add_argument('--model', type=str, default='mobilenetv5', 
                       choices=['mobilenetv5', 'mobilenetv5_small'],
                       help='Model variant')
    parser.add_argument('--data_dir', type=str, default='../bioast_dataset',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MobileNetV5Trainer(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience
    )
    
    # Train model
    model, history = trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()