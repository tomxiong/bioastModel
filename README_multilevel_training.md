# 多层分类MobileNetV3训练系统

基于MobileNetV3的细菌图像四层分类模型训练系统，用于70x70灰度图像的细菌分析。

## 系统架构

### 分类层次结构
1. **microbe_type**: 微生物类型（目前只有bacteria）
2. **growth_level**: 生长水平（positive/negative）
3. **growth_pattern**: 生长模式（12种不同模式）
4. **interference_factors**: 干扰因子（4种，多标签分类）

### 模型架构
- **骨干网络**: MobileNetV3 (Small/Large)
- **输入**: 70x70灰度图像
- **输出**: 多个分类头，支持层次化分类

## 文件结构

```
bioastModel/
├── models/
│   └── multilevel_mobilenetv3.py      # 多层分类模型定义
├── training/
│   ├── multilevel_dataset.py          # 数据集和数据加载器
│   └── multilevel_trainer.py          # 训练器
├── train_multilevel_mobilenetv3.py    # 主训练脚本
├── requirements.txt                    # 依赖包
└── README_multilevel_training.md       # 本文档
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 基础训练
```bash
python train_multilevel_mobilenetv3.py
```

### 3. 自定义训练
```bash
python train_multilevel_mobilenetv3.py \
    --model_size large \
    --batch_size 64 \
    --num_epochs 150 \
    --learning_rate 0.0005 \
    --experiment_name my_experiment
```

## 参数说明

### 数据参数
- `--json_path`: JSON标注文件路径（默认: ds/images/m9e1n170.json）
- `--image_root`: 图像根目录（默认: ds/images）

### 模型参数
- `--model_size`: 模型大小，small或large（默认: small）
- `--input_channels`: 输入通道数（默认: 1，灰度图像）
- `--dropout_rate`: Dropout率（默认: 0.3）
- `--freeze_backbone`: 是否冻结骨干网络

### 训练参数
- `--batch_size`: 批次大小（默认: 32）
- `--num_epochs`: 训练轮数（默认: 100）
- `--learning_rate`: 学习率（默认: 0.001）
- `--weight_decay`: 权重衰减（默认: 0.01）

### 数据分割
- `--train_ratio`: 训练集比例（默认: 0.7）
- `--val_ratio`: 验证集比例（默认: 0.15）
- `--test_ratio`: 测试集比例（默认: 0.15）

## 数据格式

JSON标注文件格式：
```json
{
  "image_id": "unique_id",
  "image_path": "relative/path/to/image.jpg",
  "panoramic_id": "panoramic_identifier",
  "hole_number": 1,
  "features": {
    "microbe_type": "bacteria",
    "growth_level": "positive",
    "growth_pattern": "clustered",
    "interference_factors": ["pores"]
  }
}
```

## 输出结果

训练完成后，在实验目录中会生成：
- `config.json`: 训练配置
- `model_info.json`: 模型信息
- `label_info.json`: 标签映射信息
- `best_model.pth`: 最佳模型权重
- `training_history.json`: 训练历史
- `training_curves.png`: 训练曲线图
- `tensorboard/`: TensorBoard日志

## 模型评估

### 仅评估模式
```bash
python train_multilevel_mobilenetv3.py \
    --eval_only \
    --resume experiments/your_experiment/best_model.pth
```

### 评估指标
- **growth_level**: 准确率、精确率、召回率、F1分数
- **growth_pattern**: 多分类准确率、每类F1分数
- **interference_factors**: 多标签准确率、汉明损失、每标签F1分数

## 高级功能

### 1. 恢复训练
```bash
python train_multilevel_mobilenetv3.py \
    --resume experiments/your_experiment/checkpoint_epoch_50.pth
```

### 2. 自定义实验名称
```bash
python train_multilevel_mobilenetv3.py \
    --experiment_name bacteria_classification_v2
```

### 3. GPU训练
```bash
python train_multilevel_mobilenetv3.py \
    --device cuda \
    --batch_size 128
```

## 监控训练

### TensorBoard
```bash
tensorboard --logdir experiments/your_experiment/tensorboard
```

### 训练日志
实时查看训练日志：
```bash
tail -f experiments/your_experiment/training.log
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减小batch_size
   - 使用model_size='small'

2. **训练过慢**
   - 增加num_workers
   - 使用GPU训练
   - 减小图像尺寸

3. **过拟合**
   - 增加dropout_rate
   - 增加weight_decay
   - 使用数据增强

4. **欠拟合**
   - 增加模型复杂度（使用large模型）
   - 减小dropout_rate
   - 增加训练轮数

## 性能优化建议

1. **数据加载优化**
   - 使用SSD存储
   - 增加num_workers
   - 使用pin_memory=True

2. **模型优化**
   - 使用混合精度训练
   - 梯度累积
   - 学习率调度

3. **内存优化**
   - 梯度检查点
   - 模型并行
   - 数据并行

## 扩展功能

系统支持以下扩展：
- 添加新的分类任务
- 自定义损失函数权重
- 集成其他骨干网络
- 支持多尺度训练
- 添加注意力机制

## 联系信息

如有问题或建议，请查看代码注释或提交issue。