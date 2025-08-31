# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a biomedical image analysis project focused on **70×70 pixel colony detection** in biomedical images using deep learning for binary classification:
- **Positive**: Colony present in image
- **Negative**: No colony or only air bubbles present

The project has implemented and trained **15+ different model architectures** with comprehensive performance analysis, ONNX conversion capabilities, and extensive evaluation frameworks.

## Architecture and Structure

### Core Components

**Data Pipeline**: 
- `core/data_loader.py` and `core/real_data_loader.py` - Handle dataset loading with train/val/test splits
- `training/dataset.py` - PyTorch Dataset implementation with transforms
- `bioast_dataset/` - Main dataset directory (train/test/negative/positive structure)

**Model Definitions**: 
- **Core Models**: All models in `models/` directory with consistent APIs:
  - `models/airbubble_hybrid_net.py` - Top performer (98.02% accuracy)
  - `models/resnet_improved.py`, `models/efficientnet.py`, etc.
  - Each model provides `create_<model_name>(num_classes=2)` factory function
- **MobileNetV5 Module**: Standalone implementation in `mobilenetv5/` directory:
  - `mobilenetv5/models/mobilenetv5_model.py` - MobileNetV5 with SE attention
  - `mobilenetv5/models/mobilenetv5_small.py` - Lightweight variant (1.6M parameters)
  - Independent training pipeline without modifying existing code

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

## Development Environment Setup

**IMPORTANT**: This project uses a local virtual environment (`.venv`) and uv for package management.

**Environment Rules**:
- **Always use the local .venv environment**: All Python commands should be run within the `.venv` virtual environment
- **Package Installation**: Use `uv pip install <package>` instead of `pip install` for all new package installations
- **Environment Activation**: Ensure `.venv` is activated before running any Python scripts
- **Encoding Fix**: Set console encoding to UTF-8 to avoid Chinese text display issues
- **Code Standards**: All Python scripts should use English for console output to avoid encoding issues

**Package Management Commands**:
```bash
# Install new packages (REQUIRED method)
uv pip install <package_name>

# Install from requirements
uv pip install -r requirements.txt

# List installed packages
uv pip list

# Upgrade packages
uv pip install --upgrade <package_name>
```

**Console Encoding Setup** (Windows):
```bash
# Set UTF-8 encoding before running Python scripts
chcp 65001

# Or run Python with UTF-8 encoding
$env:PYTHONIOENCODING="utf-8"; .venv/Scripts/python your_script.py
```

## Common Development Commands

### Training Models

**Single Model Training**:
```bash
# List available models
.venv\Scripts\python train_single_model.py --list_models

# Train specific model with custom parameters
.venv\Scripts\python train_single_model.py --model efficientnet_b0 --epochs 30 --batch_size 32 --lr 0.001
```

**Batch Training**:
```bash
# Train all models in sequence
.venv\Scripts\python scripts/auto_train_sequence.py

# Train multiple models
.venv\Scripts\python train_all_models.py
```

### Model Testing and Validation

**Model Testing**:
```bash
# Batch test all trained models
.venv\Scripts\python scripts/batch_test_models.py

# Validate ONNX models
.venv\Scripts\python scripts/batch_validate_all_onnx_models.py

# Quick validation of ONNX models
.venv\Scripts\python scripts/quick_validate_all_onnx.py

# Check training progress
.venv\Scripts\python scripts/check_test_progress.py
```

**Model Analysis**:
```bash
# Analyze individual model performance
.venv\Scripts\python scripts/analyze_individual_models.py

# Generate comprehensive model analysis
.venv\Scripts\python scripts/comprehensive_model_analysis.py

# Simple model analyzer
.venv\Scripts\python scripts/simple_model_analyzer.py

# Compare model performance
.venv\Scripts\python scripts/compare_models.py
```

### ONNX Conversion

**Convert Single Model**:
```bash
.venv\Scripts\python scripts/convert_single_model_to_onnx.py --model <model_name>
```

**Batch Conversion**:
```bash
.venv\Scripts\python scripts/batch_convert_models_to_onnx.py
```

### Monitoring and Analysis

**Training Progress**:
```bash
# Monitor specific model training
.venv\Scripts\python scripts/monitor_<model_name>_training.py

# Check training progress
.venv\Scripts\python scripts/check_test_progress.py
```

**Generate Reports**:
```bash
# Generate analysis reports
.venv\Scripts\python scripts/generate_detailed_analysis.py
.venv\Scripts\python scripts/generate_final_analysis.py
```

### Dataset Management

**Dataset Operations**:
```bash
# Check dataset status
.venv\Scripts\python dataset_manager.py --check

# Update dataset
.venv\Scripts\python dataset_manager.py --update-dataset "path/to/new/dataset"

# Retrain all models
.venv\Scripts\python dataset_manager.py --retrain-all
```

### MobileNetV5 Training (Standalone Module)

**Environment Setup**:
```bash
# Activate virtual environment (required)
.venv\Scripts\activate

# Check environment
cd mobilenetv5
python check_env.py
```

**Training Commands**:
```bash
# Quick test (5 epochs)
cd mobilenetv5
python train.py --model mobilenetv5 --config quick_test --test_only

# Standard training (50 epochs)
cd mobilenetv5
python train.py --model mobilenetv5 --config standard

# Train small variant
cd mobilenetv5
python train.py --model mobilenetv5_small --config standard

# Extended training (100 epochs)
cd mobilenetv5
python train.py --model mobilenetv5 --config extended

# Custom parameters
cd mobilenetv5
python train.py --model mobilenetv5 --config standard --batch_size 16 --learning_rate 0.0005 --num_epochs 75
```

**Evaluation**:
```bash
# Evaluate trained model
cd mobilenetv5
python evaluation.py --model mobilenetv5 --checkpoint experiments/mobilenetv5/model_best.pth
```

## Build and Code Quality

**Dependencies**: Main dependencies include:
- PyTorch + torchvision
- scikit-learn
- matplotlib, seaborn
- PIL/Pillow
- ONNX, onnxruntime
- uv (package manager)
- Complete dependency list in `requirements.txt`

**Code Quality**: This project currently doesn't have formal linting/type checking configured. Consider adding:
```bash
# Install development dependencies (optional)
uv pip install black flake8 mypy pytest

# Format code (if black is installed)
black .

# Lint code (if flake8 is installed)
flake8 .

# Type check (if mypy is installed)
mypy .
```

**GPU Support**: All training scripts detect and use CUDA when available.

**Python Path**: Scripts add project root to sys.path for imports.

## Model Performance Hierarchy

Current best performers (from README.md):
1. **AirBubble_HybridNet**: 98.02% (CNN-Transformer hybrid)
2. **ResNet18-Improved**: 97.83% (Enhanced ResNet)
3. **EfficientNet-B0**: 97.54% (Efficient CNN)
4. **MIC_MobileNetV3**: 97.45% (Mobile-optimized)
5. **Micro-ViT**: 97.36% (Micro Vision Transformer)

**New Model**: MobileNetV5 module (standalone implementation):
- **MobileNetV5**: 2.8M parameters with SE attention
- **MobileNetV5 Small**: 1.6M parameters for faster inference
- Optimized for 70×70 input images with modern architecture

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
- **MobileNetV5**: `mobilenetv5/` standalone module with independent implementation

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

## Important Notes

- Models use 70×70 input resolution (not standard 224×224)
- All models output 2 classes for binary classification
- Experiment directories are excluded from git (.gitignore)
- ONNX models are committed for deployment
- Chinese comments throughout - this is a bilingual codebase
- Comprehensive performance tracking and comparison built-in
- Use uv for package management instead of pip
- Set console encoding to UTF-8 on Windows to avoid display issues