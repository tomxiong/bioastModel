# Comprehensive Model Analysis Report

Generated on: 2025-08-03 12:27:51
Total models analyzed: 8

## Executive Summary

This report presents a comprehensive analysis of 8 deep learning models trained for biomedical image classification (70x70 colony detection). The analysis includes 5 base models and 3 MIC-specialized enhanced models.

## Performance Overview

| Rank | Model | Accuracy | Parameters | Epochs | Efficiency Score |
|------|-------|----------|------------|--------|------------------|
| 1 | resnet18_improved | 0.9870 | 11,260,000 | 21 | 0.623 |
| 2 | mic_mobilenetv3 | 0.9851 | 1,138,137 | 15 | 0.857 |
| 3 | airbubble_hybrid_net | 0.9851 | 750,142 | 33 | 0.988 |
| 4 | efficientnet_b0 | 0.9833 | 1,560,000 | 16 | 0.785 |
| 5 | convnext_tiny | 0.9833 | 28,600,000 | 39 | 0.603 |
| 6 | micro_vit | 0.9833 | 3,265,402 | 39 | 0.683 |
| 7 | vit_tiny | 0.9665 | 2,720,000 | 19 | 0.692 |
| 8 | coatnet | 0.9628 | 26,042,722 | 27 | 0.592 |



## Base Models

**Best Performer**: resnet18_improved (0.9870 accuracy)

- **Average Accuracy**: 0.9766
- **Parameter Range**: 1,560,000 - 28,600,000
- **Models Count**: 5


## Enhanced Models

**Best Performer**: mic_mobilenetv3 (0.9851 accuracy)

- **Average Accuracy**: 0.9845
- **Parameter Range**: 750,142 - 3,265,402
- **Models Count**: 3


## Lightweight Models

**Best Performer**: mic_mobilenetv3 (0.9851 accuracy)

- **Average Accuracy**: 0.9807
- **Parameter Range**: 750,142 - 3,265,402
- **Models Count**: 5


## Heavy Models

**Best Performer**: resnet18_improved (0.9870 accuracy)

- **Average Accuracy**: 0.9777
- **Parameter Range**: 11,260,000 - 28,600,000
- **Models Count**: 3



## Deployment Recommendations

### 🏆 Best Overall Performance
**resnet18_improved** - 0.9870 accuracy
- Parameters: 11,260,000
- Training epochs: 21
- **Use case**: When maximum accuracy is required

### ⚡ Best Lightweight Model
**airbubble_hybrid_net** - 0.9851 accuracy
- Parameters: 750,142
- **Use case**: Mobile deployment, edge computing

### 🎯 Most Efficient Model
**airbubble_hybrid_net** - Efficiency score: 0.988
- Accuracy: 0.9851
- Parameters: 750,142
- **Use case**: Balanced performance and resource usage



## Technical Insights

### CNN Architectures
- Average accuracy: 0.9848
- Models: 5
- Best: resnet18_improved

### Transformer Architectures
- Average accuracy: 0.9749
- Models: 2
- Best: micro_vit

### MIC-Specialized Enhanced Models
- Average accuracy: 0.9845
- All models show excellent performance with specialized optimizations
- Demonstrate effectiveness of domain-specific architectural improvements



## Detailed Model Information

### resnet18_improved

- **Best Validation Accuracy**: 0.9870
- **Final Training Accuracy**: 0.9811
- **Parameters**: 11,260,000
- **Training Epochs**: 21
- **Efficiency Score**: 0.623
- **Accuracy per Million Parameters**: 0.088
- **Experiment Path**: `experiments\experiment_20250802_164948\resnet18_improved`

### mic_mobilenetv3

- **Best Validation Accuracy**: 0.9851
- **Final Training Accuracy**: 0.9776
- **Parameters**: 1,138,137
- **Training Epochs**: 15
- **Efficiency Score**: 0.857
- **Accuracy per Million Parameters**: 0.866
- **Experiment Path**: `experiments\experiment_20250803_101438\mic_mobilenetv3`

### airbubble_hybrid_net

- **Best Validation Accuracy**: 0.9851
- **Final Training Accuracy**: 0.9838
- **Parameters**: 750,142
- **Training Epochs**: 33
- **Efficiency Score**: 0.988
- **Accuracy per Million Parameters**: 1.313
- **Experiment Path**: `experiments\experiment_20250803_115344\airbubble_hybrid_net`

### efficientnet_b0

- **Best Validation Accuracy**: 0.9833
- **Final Training Accuracy**: 0.9828
- **Parameters**: 1,560,000
- **Training Epochs**: 16
- **Efficiency Score**: 0.785
- **Accuracy per Million Parameters**: 0.630
- **Experiment Path**: `experiments\experiment_20250802_140818\efficientnet_b0`

### convnext_tiny

- **Best Validation Accuracy**: 0.9833
- **Final Training Accuracy**: 0.9760
- **Parameters**: 28,600,000
- **Training Epochs**: 39
- **Efficiency Score**: 0.603
- **Accuracy per Million Parameters**: 0.034
- **Experiment Path**: `experiments\experiment_20250802_231639\convnext_tiny`

### micro_vit

- **Best Validation Accuracy**: 0.9833
- **Final Training Accuracy**: 0.9817
- **Parameters**: 3,265,402
- **Training Epochs**: 39
- **Efficiency Score**: 0.683
- **Accuracy per Million Parameters**: 0.301
- **Experiment Path**: `experiments\experiment_20250803_102845\micro_vit`

### vit_tiny

- **Best Validation Accuracy**: 0.9665
- **Final Training Accuracy**: 0.9539
- **Parameters**: 2,720,000
- **Training Epochs**: 19
- **Efficiency Score**: 0.692
- **Accuracy per Million Parameters**: 0.355
- **Experiment Path**: `experiments\experiment_20250803_020217\vit_tiny`

### coatnet

- **Best Validation Accuracy**: 0.9628
- **Final Training Accuracy**: 0.9488
- **Parameters**: 26,042,722
- **Training Epochs**: 27
- **Efficiency Score**: 0.592
- **Accuracy per Million Parameters**: 0.037
- **Experiment Path**: `experiments\experiment_20250803_032628\coatnet`

## Conclusion

The analysis reveals several key findings:

1. **MIC_MobileNetV3** emerges as the efficiency champion with 98.51% accuracy using only 1.14M parameters
2. **ConvNext-Tiny** and **Micro-ViT** tie for the highest accuracy at 98.33%
3. **Enhanced models** (MIC-specialized) show excellent performance with domain-specific optimizations
4. **Lightweight models** (< 5M parameters) achieve competitive performance suitable for deployment

The results demonstrate that specialized architectural improvements for MIC testing scenarios can achieve both high accuracy and efficiency, making them ideal for practical deployment in biomedical applications.

---
*Report generated by Comprehensive Model Analysis Script*
