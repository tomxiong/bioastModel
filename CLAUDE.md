# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a biomedical image analysis project focused on **70×70 pixel colony detection** in biomedical images using deep learning for binary classification:
- **Positive**: Colony present in image
- **Negative**: No colony or only air bubbles present

The project has implemented and trained **30+ different model architectures** with comprehensive performance analysis, ONNX conversion capabilities, and extensive evaluation frameworks. This is a mature codebase with extensive automation scripts and a modular architecture for training, evaluation, and deployment.

## Architecture and Structure

### Core Components

**Data Pipeline**: 
- `core/data_loader.py` and `core/real_data_loader.py` - Handle dataset loading with train/val/test splits
- `training/dataset.py` - PyTorch Dataset implementation with transforms
- `bioast_dataset/` - Main dataset directory (train/test/negative/positive structure)

**Model Definitions**: All models in `models/` directory with consistent APIs:
- `models/airbubble_hybrid_net.py` - Top performer (98.02% accuracy)
- `models/resnet_improved.py`, `models/efficientnet.py`, etc.
- Each model provides `create_<model_name>(num_classes=2)` factory function

**Training Framework**:
- `core/training_utils.py` - Common training utilities
- `training/trainer.py`, `training/evaluator.py`, `training/visualizer.py` - Modular training components
- `core/config/` - Centralized model and training configurations

**Conversion Pipeline**:
- `core/onnx_converter_base.py` and `core/enhanced_onnx_converter_base.py` - Base conversion classes
- `converters/` - Model-specific ONNX converters
- `deployment/onnx_models/` - Production ONNX models

### Configuration System

**Model Configs** (`core/config/model_configs.py`):
- `MODEL_CONFIGS` dictionary contains all model metadata
- Use `get_model_config(model_name)` to access configuration
- Includes parameters, input size, architecture type, experiment patterns

**Training Configs** (`core/config/training_configs.py`):
- `TRAINING_CONFIGS` with optimized settings per model type
- `get_model_specific_config(model_name)` returns recommended training config

## Common Development Tasks

### Environment Setup

**IMPORTANT**: This project uses a local virtual environment (`.venv`) and uv for package management.

**Environment Rules**:
- **Always use the local .venv environment**: All Python commands should be run within the `.venv` virtual environment
- **Package Installation**: Use `uv pip install <package>` instead of `pip install` for all new package installations
- **Environment Activation**: Ensure `.venv` is activated before running any Python scripts

**Virtual Environment Commands**:
```bash
# Activate virtual environment (Linux/macOS)
source .venv/bin/activate

# Install new packages (REQUIRED method)
uv pip install <package_name>

# Install from requirements
uv pip install -r requirements.txt

# Run Python scripts with virtual environment
.venv/bin/python script.py
```

### Training Models

**Single Model Training**:
```bash
# List available models
.venv/bin/python train_single_model.py --list_models

# Train specific model with custom parameters
.venv/bin/python train_single_model.py --model efficientnet_b0 --epochs 30 --batch_size 32 --lr 0.001
```

**Batch Training**:
```bash
# Train all models in sequence
.venv/bin/python scripts/auto_train_sequence.py

# Train multiple models
.venv/bin/python train_all_models.py
```

### Model Evaluation and Testing

**Batch Testing**:
```bash
# Test all trained models
.venv/bin/python scripts/batch_test_models.py

# Validate ONNX models
.venv/bin/python scripts/batch_validate_all_onnx_models.py
```

**Individual Model Analysis**:
```bash
# Generate comprehensive analysis
.venv/bin/python scripts/comprehensive_model_analysis.py

# Compare model performance
.venv/bin/python scripts/compare_models.py
```

### ONNX Conversion

**Convert Single Model**:
```bash
.venv/bin/python scripts/convert_single_model_to_onnx.py --model <model_name>
```

**Batch Conversion**:
```bash
.venv/bin/python scripts/batch_convert_models_to_onnx.py
```

### Monitoring and Analysis

**Training Progress**:
```bash
# Monitor specific model training
.venv/bin/python scripts/monitor_<model_name>_training.py

# Check training progress
.venv/bin/python scripts/check_test_progress.py
```

**Generate Reports**:
```bash
# Generate analysis reports
.venv/bin/python scripts/generate_detailed_analysis.py
.venv/bin/python scripts/generate_final_analysis.py
```

## Model Performance Hierarchy

Current best performers (from README.md):
1. **AirBubble_HybridNet**: 98.02% (CNN-Transformer hybrid)
2. **ResNet18-Improved**: 97.83% (Enhanced ResNet)
3. **EfficientNet-B0**: 97.54% (Efficient CNN)
4. **MIC_MobileNetV3**: 97.45% (Mobile-optimized)
5. **Micro-ViT**: 97.36% (Micro Vision Transformer)

All models target 70×70 input images with 2 classes (positive/negative).

## Key Patterns and Conventions

### Model Implementation Pattern
```python
def create_<model_name>(num_classes=2, **kwargs):
    """Factory function returning configured model"""
    model = ModelClass(num_classes=num_classes, **kwargs)
    return model
```

### Training Script Pattern
- Use `core/config/model_configs.py` for model metadata
- Use `core/config/training_configs.py` for training parameters
- Save checkpoints to `experiments/experiment_<timestamp>/`
- Generate training history JSON and curves

### ONNX Conversion Pattern
- Inherit from `OnnxConverterBase` or `EnhancedOnnxConverterBase`
- Implement model-specific input/output handling
- Save to `onnx_models/` or `deployment/onnx_models/`
- Include model validation post-conversion

### File Organization
- **Experiments**: `experiments/experiment_<timestamp>/` (not in git)
- **Checkpoints**: Model-specific subdirectories with `best.pth`, `latest.pth`
- **Reports**: `reports/` with comprehensive analysis files
- **Scripts**: `scripts/` for all automation and training scripts

## Dataset Structure

Expected dataset layout:
```
bioast_dataset/
├── train/
│   ├── negative/
│   └── positive/
├── val/
│   ├── negative/
│   └── positive/
└── test/
    ├── negative/
    └── positive/
```

Data loading automatically handles this structure via `BioastDataset` class.

## Development Environment

### Key Development Commands

**Environment Management**:
```bash
# Check if uv is available
which uv

# Install dependencies
uv pip install -r requirements.txt

# Run any Python script (always use virtual environment)
.venv/bin/python <script_name>.py

# For Windows, use:
.venv\Scripts\python <script_name>.py
```

**Code Quality and Testing**:
```bash
# Run linting (if available)
.venv/bin/python -m flake8 .

# Run tests (if test suite exists)
.venv/bin/python -m pytest

# Type checking (if mypy is configured)
.venv/bin/python -m mypy .
```

**Dataset Management**:
```bash
# Check dataset status
.venv/bin/python dataset_manager.py --check

# Update dataset
.venv/bin/python dataset_manager.py --update-dataset "path/to/new/dataset"

# Sync dataset (if using the sync script)
./sync_bioast_dataset.sh
```

### Important Development Notes

**Model Registry**: The project uses a model registry system (`model_registry.json`) to track trained models and their performance.

**Experiment Management**: Training experiments are stored in `experiments/experiment_<timestamp>/` directories and excluded from git.

**Checkpoints**: Model checkpoints are saved in `checkpoints/<model_name>/` with `best.pth` and `latest.pth` files.

**Reports**: Analysis reports are generated in `reports/` directory with comprehensive performance metrics.

**Chinese Comments**: The codebase contains Chinese comments and documentation - this is a bilingual project.

**GPU Support**: All training scripts automatically detect and use CUDA when available.

**Model Input Size**: All models use 70×70 input resolution (not standard 224×224) for this specific biomedical application.

## Important Notes

- Models use 70×70 input resolution (not standard 224×224)
- All models output 2 classes for binary classification
- Experiment directories are excluded from git (.gitignore)
- ONNX models are committed for deployment
- Chinese comments throughout - this is a bilingual codebase
- Comprehensive performance tracking and comparison built-in
- Use `uv` for package management, not `pip`
- Always activate the virtual environment before running scripts
- The project has extensive automation scripts in the `scripts/` directory
- Model configurations are centralized in `core/config/model_configs.py`