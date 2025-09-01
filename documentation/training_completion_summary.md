# BioAST Dataset Summary

This document provides a comprehensive overview of the bioast_dataset1 directory structure and content, which is used for training and evaluating biomedical image analysis models.

## Dataset Structure

The dataset is organized into two main categories:

1. **Positive Samples**: Images containing the target biomedical features
2. **Negative Samples**: Images without the target biomedical features

Each category is further divided into three subsets:

- **Train**: Used for model training
- **Validation**: Used for model validation during training
- **Test**: Used for final model evaluation

## Dataset Statistics

### Directory Structure
```
bioast_dataset1/
├── negative/
│   ├── test/
│   ├── train/
│   └── val/
└── positive/
    ├── test/
    ├── train/
    └── val/
```

### Sample Distribution

| Category | Train | Validation | Test | Total |
|----------|-------|------------|------|-------|
| Positive | ~1000+ | ~200+ | ~200+ | ~1400+ |
| Negative | ~1000+ | ~200+ | ~200+ | ~1400+ |
| **Total** | ~2000+ | ~400+ | ~400+ | ~2800+ |

## File Naming Convention

The dataset follows a consistent naming convention:

`[Batch_ID]_hole_[Hole_Number].png`

Example: `EB10000026_hole_67.png`

- **Batch_ID**: Identifier for the batch (e.g., EB10000026, EB20000012)
- **Hole_Number**: Numeric identifier for the specific hole/sample

## Batch Distribution

The dataset includes samples from multiple batches, including:

### EB1 Series:
- EB10000026
- EB10000027
- EB10000028
- EB10000029
- EB10000030
- EB10000031
- EB10000032
- EB10000033
- EB10000034
- EB10000035
- EB10000036

### EB2 Series:
- EB20000012
- EB20000013
- EB20000059
- EB20000061
- EB20000062
- EB20000063
- EB20000064
- EB20000065
- EB20000066
- EB20000067
- EB20000077
- EB20000078
- EB20000079
- EB20000080
- EB20000081
- EB20000082
- EB20000083
- EB20000084
- EB20000085
- EB20000086
- EB20000087
- EB20000088
- EB20000089
- EB20000091
- EB20000093
- EB20000094
- EB20000130
- EB20000131
- EB20000132
- EB20000133
- EB20000134
- EB20000138
- EB20000139
- EB20000140
- EB20000157
- EB20000159
- EB20000167

## Image Characteristics

- **Format**: PNG
- **Content**: Microscopy images of biological samples
- **Features**: The positive samples contain specific biomedical features of interest, while negative samples do not contain these features

## Usage in Model Training

This dataset is used for training various deep learning models for biomedical image analysis, including:

1. ResNet variants
2. EfficientNet variants
3. MobileNet variants
4. Vision Transformer (ViT) variants
5. Custom architectures like AirBubble Hybrid Net

The dataset's division into train/validation/test sets enables:
- Model training on the training set
- Hyperparameter tuning using the validation set
- Unbiased performance evaluation on the test set

## Data Preprocessing

Before being fed into the models, the images typically undergo preprocessing steps such as:
- Resizing to model-specific input dimensions
- Normalization
- Data augmentation (for training set only)

## Model Evaluation

Models trained on this dataset are evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1 Score
- Area Under ROC Curve (AUC)

The test set provides an unbiased evaluation of model performance on unseen data.