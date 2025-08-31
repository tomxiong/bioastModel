"""
MobileNetV5 Training Script
Main entry point for training MobileNetV5 models
"""

import sys
import os
import argparse
from pathlib import Path
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config import MobileNetV5Config
from training import MobileNetV5Trainer
from models import create_mobilenetv5


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train MobileNetV5 for Colony Detection')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='mobilenetv5',
                       choices=['mobilenetv5', 'mobilenetv5_small'],
                       help='Model variant to train')
    
    # Training arguments
    parser.add_argument('--config', type=str, default='standard',
                       choices=['quick_test', 'standard', 'extended'],
                       help='Training configuration')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-detect)')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--num_epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay (overrides config)')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience (overrides config)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test the model implementation')
    
    args = parser.parse_args()
    
    # Setup environment
    MobileNetV5Config.setup_environment()
    
    # Set random seed
    import torch
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Test only mode
    if args.test_only:
        print("Testing MobileNetV5 implementation...")
        model = create_mobilenetv5(args.model, num_classes=2, input_size=70)
        test_input = torch.randn(1, 3, 70, 70)
        output = model(test_input)
        print(f"Model test successful!")
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return
    
    # Get configurations
    model_config = MobileNetV5Config.get_model_config(args.model)
    training_config = MobileNetV5Config.get_training_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        training_config['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        training_config['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        training_config['weight_decay'] = args.weight_decay
    if args.patience is not None:
        training_config['early_stopping_patience'] = args.patience
    
    # Set directories
    data_dir = args.data_dir if args.data_dir else str(MobileNetV5Config.get_data_dir())
    output_dir = args.output_dir if args.output_dir else str(
        MobileNetV5Config.get_mobilenetv5_experiments_dir() / f"{args.model}_{args.config}_{int(time.time())}"
    )
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Print configuration
    print("="*60)
    print("MOBILENETV5 TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Description: {model_config['description']}")
    print(f"Parameters: {model_config['params_millions']}M")
    print(f"Config: {args.config}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Epochs: {training_config['num_epochs']}")
    print(f"Learning rate: {training_config['learning_rate']}")
    print(f"Weight decay: {training_config['weight_decay']}")
    print(f"Early stopping patience: {training_config['early_stopping_patience']}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    # Check data directory
    if not Path(data_dir).exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please ensure the data directory exists with the following structure:")
        print("bioast_dataset/")
        print("  ├── positive/")
        print("  │   ├── train/")
        print("  │   ├── val/")
        print("  │   └── test/")
        print("  └── negative/")
        print("      ├── train/")
        print("      ├── val/")
        print("      └── test/")
        return
    
    # Create trainer
    trainer = MobileNetV5Trainer(
        model_name=args.model,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        **training_config
    )
    
    # Train model
    try:
        model, history = trainer.train()
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
        print(f"Model saved to: {output_dir}")
        print(f"Results saved to: {output_dir}/{args.model}_results.json")
        
    except Exception as e:
        print(f"\nERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Optionally evaluate on test set
    try:
        print("\nEvaluating on test set...")
        from evaluation import MobileNetV5Evaluator
        
        evaluator = MobileNetV5Evaluator(
            model_path=f"{output_dir}/{args.model}_best.pth",
            data_dir=data_dir,
            device=device
        )
        
        metrics = evaluator.evaluate()
        evaluator.save_results(metrics, f"{output_dir}/test_evaluation")
        
    except Exception as e:
        print(f"Warning: Test evaluation failed: {e}")


if __name__ == "__main__":
    main()