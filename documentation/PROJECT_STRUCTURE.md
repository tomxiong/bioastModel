# Project Structure Summary

Generated on: 2025-08-02 20:12:47

## Core Structure

### Models (`models/`)
- `efficientnet.py` - EfficientNet-B0 implementation
- `resnet_improved.py` - Improved ResNet-18 implementation

### Training System (`training/`)
- `dataset.py` - Data loading and preprocessing
- `trainer.py` - Training loop and optimization
- `evaluator.py` - Model evaluation and metrics
- `visualizer.py` - Visualization and reporting

### Configuration (`core/config/`)
- `model_configs.py` - Model configurations
- `training_configs.py` - Training parameters
- `paths.py` - Path management

### Scripts (`scripts/`)
- `train_model.py` - Unified training script
- `evaluate_model.py` - Unified evaluation script
- `compare_models.py` - Model comparison script
- `add_new_model.py` - New model template generator
- `cleanup_project.py` - Project organization script

### Experiments (`experiments/`)
- Contains all training experiments and results
- Organized by timestamp and model name

### Reports (`reports/`)
- `individual/` - Individual model reports
- `comparisons/` - Model comparison reports

### Archive (`archive/`)
- `legacy_scripts/` - Archived legacy scripts
- `temp_files/` - Temporary files backup

## Usage Workflows

### Training a New Model
1. `python scripts/add_new_model.py --name model_name`
2. Implement model architecture
3. `python scripts/train_model.py --model model_name`

### Evaluating a Model
1. `python scripts/evaluate_model.py --model model_name`

### Comparing Models
1. `python scripts/compare_models.py --models model1 model2`

### Adding New Models
1. Use the standardized template system
2. Follow the established directory structure
3. Integrate with existing evaluation pipeline

## Best Practices
- Use unified scripts for consistency
- Follow naming conventions
- Document new models and changes
- Regular cleanup and organization
