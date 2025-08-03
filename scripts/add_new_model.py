"""
Template script for adding new models to the comparison system.

This script provides a standardized workflow for:
1. Adding new model definitions
2. Creating training configurations
3. Setting up evaluation pipelines
4. Integrating with comparison system

Usage:
    python scripts/add_new_model.py --name convnext_tiny --base-config efficientnet_b0
    python scripts/add_new_model.py --name coatnet_0 --base-config resnet18_improved
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import (
    get_model_config,
    MODELS_DIR,
    EXPERIMENTS_DIR
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Add a new model to the comparison system')
    
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Name of the new model (e.g., convnext_tiny, coatnet_0)'
    )
    
    parser.add_argument(
        '--base-config',
        type=str,
        choices=['efficientnet_b0', 'resnet18_improved'],
        default='efficientnet_b0',
        help='Base configuration to copy from'
    )
    
    parser.add_argument(
        '--architecture',
        type=str,
        choices=['convnext', 'coatnet', 'vit', 'swin', 'custom'],
        default='custom',
        help='Model architecture family'
    )
    
    parser.add_argument(
        '--parameters',
        type=float,
        default=None,
        help='Estimated number of parameters in millions'
    )
    
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Brief description of the model'
    )
    
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=0.2,
        help='Default dropout rate for the model'
    )
    
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Whether to use pretrained weights by default'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be created without actually creating files'
    )
    
    return parser.parse_args()

def validate_model_name(name):
    """Validate that the model name is valid and not already used."""
    # Check naming convention
    if not name.replace('_', '').replace('-', '').isalnum():
        raise ValueError(f"Model name '{name}' contains invalid characters. Use only letters, numbers, underscores, and hyphens.")
    
    # Check if model already exists
    model_file = MODELS_DIR / f"{name}.py"
    if model_file.exists():
        raise ValueError(f"Model '{name}' already exists at {model_file}")
    
    return True

def create_model_definition_template(args):
    """Create model definition file template."""
    model_name = args.name
    architecture = args.architecture or 'custom'
    
    template_content = f'''"""
{model_name.replace('_', ' ').title()} model definition for colony detection.

This model is based on the {architecture} architecture and is designed for
binary classification of 70x70 images (colony vs no-colony).

Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class {model_name.replace('_', '').title()}(nn.Module):
    """
    {model_name.replace('_', ' ').title()} model for colony detection.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        dropout_rate (float): Dropout rate for regularization (default: 0.2)
        pretrained (bool): Whether to use pretrained weights (default: True)
    """
    
    def __init__(
        self, 
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        pretrained: bool = True,
        **kwargs
    ):
        super({model_name.replace('_', '').title()}, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # TODO: Implement model architecture
        # This is a template - replace with actual model implementation
        
        # Example backbone (replace with actual architecture)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 70, 70)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {{
            'name': '{model_name}',
            'architecture': '{architecture}',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 70, 70),
            'output_size': self.num_classes
        }}

def create_{model_name}(
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> {model_name.replace('_', '').title()}:
    """
    Create {model_name.replace('_', ' ').title()} model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        **kwargs: Additional arguments for model initialization
        
    Returns:
        {model_name.replace('_', '').title()}: Initialized model
    """
    model = {model_name.replace('_', '').title()}(
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )
    
    return model

# Model configuration for integration with training system
MODEL_CONFIG = {{
    'name': '{model_name}',
    'architecture': '{architecture}',
    'create_function': create_{model_name},
    'default_params': {{
        'num_classes': 2,
        'pretrained': True,
        'dropout_rate': 0.2
    }},
    'training_params': {{
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 50
    }},
    'estimated_parameters': {args.parameters or 'None'},
    'description': '{args.description or f"Custom {architecture} model for colony detection"}'
}}

if __name__ == "__main__":
    # Test model creation
    model = create_{model_name}()
    print(f"Created {{model.get_model_info()['name']}} with {{model.get_model_info()['total_parameters']:,}} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 70, 70)
    output = model(dummy_input)
    print(f"Output shape: {{output.shape}}")
'''
    
    return template_content

def create_training_script_template(args):
    """Create training script template for the new model."""
    model_name = args.name
    
    template_content = f'''"""
Training script for {model_name.replace('_', ' ').title()} model.

Usage:
    python scripts/train_{model_name}.py
    python scripts/train_{model_name}.py --epochs 100 --batch-size 64
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_model import main as train_main
import argparse

def parse_arguments():
    """Parse command line arguments specific to {model_name}."""
    parser = argparse.ArgumentParser(description='Train {model_name.replace("_", " ").title()} model')
    
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
        '--model', '{model_name}',
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
        '--model-param', f'dropout_rate={args.dropout_rate}',
        '--model-param', f'pretrained={args.pretrained}'
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
'''
    
    return template_content

def update_model_configs(args):
    """Update model configurations to include the new model."""
    model_name = args.name
    base_config = get_model_config(args.base_config)
    
    # Create new model config based on base config
    new_config = base_config.copy()
    new_config.update({
        'name': model_name,
        'architecture': args.architecture or 'custom',
        'parameters': (args.parameters * 1e6) if args.parameters else None,
        'description': args.description or f"Custom model based on {args.base_config}",
        'created_date': datetime.now().isoformat(),
        'base_config': args.base_config
    })
    
    return new_config

def create_evaluation_template(args):
    """Create evaluation template for the new model."""
    model_name = args.name
    
    template_content = f'''"""
Evaluation script for {model_name.replace('_', ' ').title()} model.

Usage:
    python scripts/evaluate_{model_name}.py
    python scripts/evaluate_{model_name}.py --experiment latest
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.evaluate_model import main as evaluate_main

def main():
    """Main evaluation function."""
    # Set model name for unified evaluation script
    original_argv = sys.argv
    sys.argv = ['evaluate_model.py', '--model', '{model_name}'] + sys.argv[1:]
    
    try:
        # Call unified evaluation script
        evaluate_main()
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()
'''
    
    return template_content

def create_files(args, templates):
    """Create all the necessary files for the new model."""
    model_name = args.name
    
    files_to_create = [
        (MODELS_DIR / f"{model_name}.py", templates['model_definition']),
        (Path('scripts') / f"train_{model_name}.py", templates['training_script']),
        (Path('scripts') / f"evaluate_{model_name}.py", templates['evaluation_script'])
    ]
    
    created_files = []
    
    for file_path, content in files_to_create:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        created_files.append(str(file_path))
        print(f"✅ Created: {file_path}")
    
    return created_files

def update_config_files(args, new_config):
    """Update configuration files to include the new model."""
    # Update model_configs.py
    config_file = Path('core/config/model_configs.py')
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Add new model config
        new_model_entry = f'''
# {args.name.replace('_', ' ').title()} Configuration
'{args.name}': {{
    'name': '{args.name}',
    'architecture': '{args.architecture or 'custom'}',
    'parameters': {new_config.get('parameters', 'None')},
    'description': '{new_config.get('description', '')}',
    'training_params': {{
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 50
    }},
    'created_date': '{new_config.get('created_date', '')}',
    'base_config': '{args.base_config}'
}},'''
        
        # Insert before the closing brace of MODEL_CONFIGS
        if 'MODEL_CONFIGS = {' in content:
            insertion_point = content.rfind('}')
            updated_content = content[:insertion_point] + new_model_entry + '\n' + content[insertion_point:]
            
            with open(config_file, 'w') as f:
                f.write(updated_content)
            
            print(f"✅ Updated: {config_file}")
        else:
            print(f"⚠️  Could not update {config_file} - manual update required")

def create_documentation(args, created_files):
    """Create documentation for the new model."""
    model_name = args.name
    
    doc_content = f'''# {model_name.replace('_', ' ').title()} Model

## Overview
- **Model Name**: {model_name}
- **Architecture**: {args.architecture or 'custom'}
- **Base Configuration**: {args.base_config}
- **Estimated Parameters**: {args.parameters or 'TBD'} million
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Description
{args.description or f"Custom model based on {args.base_config} configuration"}

## Files Created
'''
    
    for file_path in created_files:
        doc_content += f"- `{file_path}`\n"
    
    doc_content += f'''
## Usage

### Training
```bash
python scripts/train_{model_name}.py
python scripts/train_{model_name}.py --epochs 100 --batch-size 64
```

### Evaluation
```bash
python scripts/evaluate_{model_name}.py
python scripts/evaluate_{model_name}.py --experiment latest
```

### Comparison
```bash
python scripts/compare_models.py --models {model_name} efficientnet_b0
```

## Next Steps
1. Implement the actual model architecture in `models/{model_name}.py`
2. Test the model creation and forward pass
3. Run training with `python scripts/train_{model_name}.py`
4. Evaluate results with `python scripts/evaluate_{model_name}.py`
5. Compare with existing models using `python scripts/compare_models.py`

## Notes
- The model definition is currently a template and needs to be implemented
- Training parameters can be adjusted in the training script
- The model will be automatically included in comparison reports once trained
'''
    
    doc_file = Path('docs') / f'{model_name}_model.md'
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    print(f"✅ Created documentation: {doc_file}")
    return str(doc_file)

def main():
    """Main function."""
    args = parse_arguments()
    
    print(f"🔧 Adding New Model: {args.name}")
    print("=" * 50)
    
    try:
        # Validate model name
        validate_model_name(args.name)
        print("✅ Model name validation passed")
        
        # Create templates
        templates = {
            'model_definition': create_model_definition_template(args),
            'training_script': create_training_script_template(args),
            'evaluation_script': create_evaluation_template(args)
        }
        
        # Create new model config
        new_config = update_model_configs(args)
        
        if args.dry_run:
            print("📋 Files that would be created (Dry Run):")
            print(f"- models/{args.name}.py")
            print(f"- scripts/train_{args.name}.py")
            print(f"- scripts/evaluate_{args.name}.py")
            print(f"- docs/{args.name}_model.md")
            print("\n📋 Configuration that would be added:")
            print(json.dumps(new_config, indent=2))
            return
        
        # Create files
        created_files = create_files(args, templates)
        
        # Update configuration files
        update_config_files(args, new_config)
        
        # Create documentation
        doc_file = create_documentation(args, created_files)
        
        print(f"\n✅ Model {args.name} added successfully!")
        print(f"📁 Files created: {len(created_files) + 1}")
        print(f"📖 Documentation: {doc_file}")
        print(f"\n🚀 Next steps:")
        print(f"1. Implement the model architecture in models/{args.name}.py")
        print(f"2. Test with: python scripts/train_{args.name}.py --dry-run")
        print(f"3. Train with: python scripts/train_{args.name}.py")
        
    except Exception as e:
        print(f"❌ Failed to add model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()