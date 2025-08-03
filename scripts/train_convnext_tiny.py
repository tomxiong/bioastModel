"""
Training script for Convnext Tiny model.

Usage:
    python scripts/train_convnext_tiny.py
    python scripts/train_convnext_tiny.py --epochs 100 --batch-size 64
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_model import main as train_main
import argparse

def parse_arguments():
    """Parse command line arguments specific to convnext_tiny."""
    parser = argparse.ArgumentParser(description='Train Convnext Tiny model')
    
    # Model-specific arguments
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                       help='Dropout rate for regularization')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./bioast_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loader workers')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Custom experiment name')
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_arguments()
    
    # Convert arguments to format expected by unified training script
    unified_args = [
        '--model', 'convnext_tiny',
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.learning_rate),
        '--weight-decay', str(args.weight_decay),
        '--data-dir', args.data_dir,
        '--num-workers', str(args.num_workers)
    ]
    
    if args.output_dir:
        unified_args.extend(['--output-dir', args.output_dir])
    
    if args.experiment_name:
        unified_args.extend(['--experiment-name', args.experiment_name])
    
    # Add model-specific parameters
    unified_args.extend([
        '--model-param', f'dropout_rate=0.2',
        '--model-param', f'pretrained=True'
    ])
    
    # Set sys.argv for the unified training script
    original_argv = sys.argv
    sys.argv = ['train_model.py'] + unified_args
    
    try:
        # Call unified training script
        train_main()
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()
