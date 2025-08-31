# Enhanced MobileNetV5 External Training Guide

## 概述

本指南提供了 Enhanced MobileNetV5 模型的外部训练设置，针对8GB GPU内存进行了优化。训练脚本支持长时间运行、自动保存检查点、早停和详细的性能监控。

## 系统要求

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+
- 8GB GPU内存（推荐）
- 16GB+ 系统内存

## 环境准备

### 1. 激活虚拟环境
```bash
# Windows
.venv\Scripts\activate

# 或使用完整路径
D:\ws1\bioastModel\.venv\Scripts\activate
```

### 2. 设置编码（Windows）
```bash
chcp 65001
```

## 配置文件选择

根据GPU内存选择合适的配置文件：

### 4GB GPU 配置（推荐）
**文件**: `enhanced_mobilenetv5_config_small_gpu.json`
```json
{
  "batch_size": 8,
  "gradient_accumulation_steps": 4,
  "width_multiplier": 0.35,
  "mixed_precision": true,
  "num_workers": 0
}
```

### 8GB+ GPU 配置
**文件**: `enhanced_mobilenetv5_config.json`
```json
{
  "batch_size": 32,
  "gradient_accumulation_steps": 1,
  "width_multiplier": 0.75,
  "mixed_precision": true,
  "num_workers": 4
}
```

## 训练命令

### 基础训练
```bash
# 使用4GB GPU配置（推荐）
.venv\Scripts\python train_enhanced_mobilenetv5_external.py --config enhanced_mobilenetv5_config_small_gpu.json

# 使用8GB+ GPU配置
.venv\Scripts\python train_enhanced_mobilenetv5_external.py --config enhanced_mobilenetv5_config.json
```

### 高级选项
```bash
# 指定输出目录
.venv\Scripts\python train_enhanced_mobilenetv5_external.py --config enhanced_mobilenetv5_config_small_gpu.json --output_dir experiments/my_experiment

# 覆盖配置参数
.venv\Scripts\python train_enhanced_mobilenetv5_external.py --config enhanced_mobilenetv5_config_small_gpu.json --epochs 150 --lr 0.0005

# 启用详细日志
.venv\Scripts\python train_enhanced_mobilenetv5_external.py --config enhanced_mobilenetv5_config_small_gpu.json --verbose
```

## 监控训练进度

### 1. 查看实时输出
训练脚本会实时输出：
- 当前epoch和进度
- 训练损失和准确率
- 验证损失和准确率
- 学习率变化
- GPU内存使用情况

### 2. 检查实验目录
训练会自动创建实验目录：
```
experiments/
└── enhanced_mobilenetv5_experiment_YYYYMMDD_HHMMSS/
    ├── best_model.pth          # 最佳模型
    ├── latest_model.pth        # 最新模型
    ├── training_history.json   # 训练历史
    ├── model_config.json       # 模型配置
    ├── training_log.txt        # 训练日志
    └── checkpoints/            # 定期检查点
        ├── epoch_10.pth
        ├── epoch_20.pth
        └── ...
```

### 3. 检查训练历史
```bash
# 查看训练历史
cat experiments/enhanced_mobilenetv5_experiment_*/training_history.json

# 查看训练日志
tail -f experiments/enhanced_mobilenetv5_experiment_*/training_log.txt
```

## 恢复训练

### 从检查点恢复
```bash
# 从最佳模型恢复
.venv\Scripts\python train_enhanced_mobilenetv5_external.py --config enhanced_mobilenetv5_config_small_gpu.json --resume experiments/enhanced_mobilenetv5_experiment_*/best_model.pth

# 从特定epoch恢复
.venv\Scripts\python train_enhanced_mobilenetv5_external.py --config enhanced_mobilenetv5_config_small_gpu.json --resume experiments/enhanced_mobilenetv5_experiment_*/checkpoints/epoch_50.pth
```

### 从实验目录恢复
```bash
# 自动查找最新的实验目录
.venv\Scripts\python train_enhanced_mobilenetv5_external.py --config enhanced_mobilenetv5_config_small_gpu.json --resume auto
```

## 测试训练结果

### 使用测试脚本
```bash
# 运行测试
.venv\Scripts\python test_enhanced_mobilenetv5.py
```

### 手动测试
```bash
# 加载特定模型进行测试
.venv\Scripts\python -c "
import torch
from mobilenetv5.models.enhanced_mobilenetv5 import create_enhanced_mobilenetv5

model = create_enhanced_mobilenetv5()
checkpoint = torch.load('experiments/enhanced_mobilenetv5_experiment_*/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Model loaded with validation accuracy: {checkpoint[\"val_acc\"]:.2f}%')
"
```

## 性能优化建议

### 1. GPU内存优化
- 使用 `mixed_precision: true` 启用混合精度训练
- 调整 `batch_size` 和 `gradient_accumulation_steps`
- 设置 `num_workers: 0` 如果遇到内存问题

### 2. 训练稳定性
- 使用 `gradient_accumulation_steps` 来模拟更大的批次
- 启用早停机制避免过拟合
- 定期保存检查点

### 3. 数据加载优化
- 确保 `num_workers` 设置合理
- 检查数据路径是否正确
- 验证数据格式是否支持

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案：减少批次大小
   # 修改配置文件中的 batch_size
   ```

2. **数据加载错误**
   ```bash
   # 检查数据目录
   ls bioast_dataset/train/
   ls bioast_dataset/val/
   ls bioast_dataset/test/
   ```

3. **导入错误**
   ```bash
   # 检查Python路径
   export PYTHONPATH=$PYTHONPATH:D:\ws1\bioastModel
   ```

4. **编码问题**
   ```bash
   # 设置UTF-8编码
   chcp 65001
   $env:PYTHONIOENCODING="utf-8"
   ```

### 日志分析
```bash
# 查看错误日志
grep -i error experiments/*/training_log.txt

# 查看警告
grep -i warning experiments/*/training_log.txt

# 查看GPU内存使用
grep "GPU Memory" experiments/*/training_log.txt
```

## 预期性能

### 4GB GPU配置（推荐）
- 批次大小：8
- 梯度累积：4步
- 预计训练时间：4-6小时（100个epoch）
- GPU内存使用：~3.5GB
- 模型参数：~70K（width_multiplier=0.35）

### 8GB+ GPU配置
- 批次大小：32
- 梯度累积：1步
- 预计训练时间：1-2小时（100个epoch）
- GPU内存使用：~10GB
- 模型参数：~133K（width_multiplier=0.75）

## 结果分析

训练完成后，您可以：

1. **查看训练曲线**
   ```bash
   # 绘制训练历史
   .venv\Scripts\python -c "
   import json
   import matplotlib.pyplot as plt
   
   with open('experiments/enhanced_mobilenetv5_experiment_*/training_history.json', 'r') as f:
       history = json.load(f)
   
   plt.plot(history['train_acc'], label='Train Accuracy')
   plt.plot(history['val_acc'], label='Validation Accuracy')
   plt.legend()
   plt.show()
   "
   ```

2. **比较模型性能**
   ```bash
   # 运行测试脚本
   .venv\Scripts\python test_enhanced_mobilenetv5.py
   ```

3. **分析类别平衡**
   检查训练日志中的类别分布信息，确保模型没有偏向某一类别。

## 总结

Enhanced MobileNetV5外部训练设置提供了完整的训练解决方案，包括：
- 内存优化的配置文件
- 自动化的训练脚本
- 完善的监控和恢复机制
- 详细的故障排除指南

按照本指南，您可以在8GB GPU上成功训练Enhanced MobileNetV5模型，并获得良好的性能表现。