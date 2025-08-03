"""
CoAtNet training script.

Usage:
    python scripts/train_coatnet.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from models.coatnet import create_coatnet
from training.dataset import create_data_loaders
from training.trainer import ModelTrainer
from core.config import get_experiment_path, create_experiment_structure, DATA_DIR

def main():
    """Main training function for CoAtNet."""
    print("🚀 CoAtNet Training")
    print("=" * 50)
    
    # Configuration
    config = {
        'model_name': 'coatnet',
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'num_epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dropout_rate': 0.2
    }
    
    print(f"📱 Device: {config['device']}")
    print(f"📊 Batch size: {config['batch_size']}")
    print(f"🔄 Max epochs: {config['num_epochs']}")
    
    # Create experiment directory
    experiment_path = get_experiment_path('coatnet')
    create_experiment_structure(experiment_path)
    print(f"📁 Experiment path: {experiment_path}")
    
    # Setup device
    device = torch.device(config['device'])
    
    # Create model
    print("🏗️ Creating CoAtNet model...")
    model = create_coatnet(
        variant='tiny',
        num_classes=2,
        dropout_rate=config['dropout_rate']
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {num_params:,}")
    
    # Create data loaders
    print("📂 Loading datasets...")
    data_loaders = create_data_loaders(
        str(DATA_DIR),
        batch_size=config['batch_size'],
        num_workers=2
    )
    
    # Create trainer
    trainer = ModelTrainer(model, device, save_dir=str(experiment_path))
    
    # Start training
    print("🚀 Starting training...")
    history = trainer.train(
        data_loaders['train'],
        data_loaders['val'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    print("✅ Training completed!")
    print(f"📁 Results saved to: {experiment_path}")
    
    return history

if __name__ == "__main__":
    main()