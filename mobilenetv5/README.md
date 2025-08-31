# MobileNetV5 for Colony Detection

## Overview

MobileNetV5 implementation optimized for 70×70 pixel colony detection images. This is a standalone implementation that reads from the same dataset and outputs to the experiments directory without modifying existing code.

## Features

- **MobileNetV5 Architecture**: Latest MobileNet variant with SE attention and improved blocks
- **MobileNetV5 Small**: Lightweight variant for faster inference
- **Optimized for Small Images**: Specifically designed for 70×70 input size
- **SE Attention**: Squeeze-and-Excitation blocks for better feature representation
- **Hard Swish Activation**: Modern activation function for better performance
- **Comprehensive Training**: Complete training pipeline with early stopping
- **Evaluation Metrics**: Full evaluation with medical-specific metrics

## Model Variants

### MobileNetV5
- **Parameters**: 2.8M
- **Description**: Standard MobileNetV5 with SE attention
- **Use Case**: Balanced accuracy and efficiency

### MobileNetV5 Small  
- **Parameters**: 1.6M
- **Description**: Smaller variant with reduced width
- **Use Case**: Faster inference, mobile deployment

## Directory Structure

```
mobilenetv5/
├── models/
│   ├── __init__.py
│   └── mobilenetv5_model.py      # Model implementation
├── training/
│   ├── __init__.py
│   ├── dataset.py               # Data loading
│   └── trainer.py               # Training logic
├── config.py                    # Configuration
├── evaluation.py                # Evaluation script
├── train.py                     # Main training script
└── __init__.py
```

## Environment Setup

⚠️ **重要：必须先激活项目虚拟环境**

### Windows环境
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 验证环境
python --version
pip list | grep torch
```

### Linux/Mac环境
```bash
# 激活虚拟环境
source .venv/bin/activate

# 验证环境
python --version
pip list | grep torch
```

## Quick Start (推荐)

### 使用自动化脚本 (推荐)

**Windows:**
```bash
# 在项目根目录运行
train_mobilenetv5.bat mobilenetv5 standard 32 50 0.001
```

**Linux/Mac:**
```bash
# 在项目根目录运行
./train_mobilenetv5.sh mobilenetv5 standard 32 50 0.001
```

### 手动训练步骤

#### 1. 环境检查
```bash
# 激活环境后
cd mobilenetv5
python check_env.py
```

#### 2. Quick Test
```bash
cd mobilenetv5
python train.py --model mobilenetv5 --config quick_test --test_only
```

#### 3. Standard Training
```bash
cd mobilenetv5
python train.py --model mobilenetv5 --config standard
```

#### 4. Training Small Variant
```bash
cd mobilenetv5
python train.py --model mobilenetv5_small --config standard
```

#### 5. Extended Training
```bash
cd mobilenetv5
python train.py --model mobilenetv5 --config extended --batch_size 32 --num_epochs 100
```

#### 6. Custom Parameters
```bash
cd mobilenetv5
python train.py \
    --model mobilenetv5 \
    --config standard \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --num_epochs 75 \
    --patience 15
```

## Configuration Options

### Training Configurations
- **quick_test**: 5 epochs, small batch size, fast testing
- **standard**: 50 epochs, balanced training
- **extended**: 100 epochs, longer training

### Model Options
- **mobilenetv5**: Standard variant (2.8M parameters)
- **mobilenetv5_small**: Small variant (1.6M parameters)

## Output Files

Training outputs are saved to `experiments/mobilenetv5/`:
- `model_best.pth`: Best model checkpoint
- `model_latest.pth`: Latest model checkpoint
- `model_results.json`: Training history and metrics
- `test_evaluation/`: Test set evaluation results

## Requirements

### 环境要求
项目使用与主项目相同的依赖，已包含在虚拟环境中：
- torch
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- seaborn
- pandas

### 环境检查
```bash
# 激活环境后运行完整环境检查
cd mobilenetv5
python check_env.py

# 或者手动检查
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 常见问题
1. **ModuleNotFoundError**: 确保已激活虚拟环境
2. **CUDA内存不足**: 减少batch_size参数
3. **权限问题**: 确保有写入experiments目录的权限

## Data Format

Expects the same data structure as the main project:
```
bioast_dataset/
├── positive/
│   ├── train/
│   ├── val/
│   └── test/
└── negative/
    ├── train/
    ├── val/
    └── test/
```

## Key Features

### Architecture
- **MBV5 Blocks**: Improved MobileNet blocks with SE attention
- **Hard Swish**: Modern activation function
- **Depthwise Separable Convolutions**: Efficient convolution operations
- **Residual Connections**: Skip connections for better gradient flow

### Training
- **Early Stopping**: Prevents overfitting
- **Cosine Annealing**: Learning rate scheduling
- **Data Augmentation**: Random flips, rotations, color jitter
- **AdamW Optimizer**: Modern optimizer with weight decay

### Evaluation
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, AUC
- **Medical Metrics**: Sensitivity, specificity
- **Confusion Matrix**: Detailed classification analysis
- **Model Comparison**: Performance comparison with other models

## Integration

This implementation is completely independent and does not modify existing code. It:
- Reads from the same dataset directory
- Outputs to the experiments directory
- Uses the same data format and structure
- Can be run alongside existing models