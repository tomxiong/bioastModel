# Convnext Tiny Model

## Overview
- **Model Name**: convnext_tiny
- **Architecture**: convnext
- **Base Configuration**: efficientnet_b0
- **Estimated Parameters**: 28.6 million
- **Created**: 2025-08-02 20:13:49

## Description
ConvNext-Tiny model for efficient colony detection

## Files Created
- `C:\Users\tomxiong\codebuddy\bioastModel\models\convnext_tiny.py`
- `scripts\train_convnext_tiny.py`
- `scripts\evaluate_convnext_tiny.py`

## Usage

### Training
```bash
python scripts/train_convnext_tiny.py
python scripts/train_convnext_tiny.py --epochs 100 --batch-size 64
```

### Evaluation
```bash
python scripts/evaluate_convnext_tiny.py
python scripts/evaluate_convnext_tiny.py --experiment latest
```

### Comparison
```bash
python scripts/compare_models.py --models convnext_tiny efficientnet_b0
```

## Next Steps
1. Implement the actual model architecture in `models/convnext_tiny.py`
2. Test the model creation and forward pass
3. Run training with `python scripts/train_convnext_tiny.py`
4. Evaluate results with `python scripts/evaluate_convnext_tiny.py`
5. Compare with existing models using `python scripts/compare_models.py`

## Notes
- The model definition is currently a template and needs to be implemented
- Training parameters can be adjusted in the training script
- The model will be automatically included in comparison reports once trained
