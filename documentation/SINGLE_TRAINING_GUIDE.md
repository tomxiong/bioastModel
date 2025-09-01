# 单独训练模型指南

本指南介绍如何使用新增的单独训练功能，无需训练全部模型即可训练指定的单个模型。

## 🚀 快速开始

### 1. 查看所有可用模型

```bash
python train_single_model.py --list_models
```

这将显示所有可用的模型及其参数量：

```
📋 可用模型列表:
============================================================
vit_tiny                  |    0.5M 参数
micro_vit                 |    1.8M 参数
mic_mobilenetv3           |    2.5M 参数
airbubble_hybrid_net      |    3.2M 参数
enhanced_airbubble_detector|    4.0M 参数
efficientnet_b0           |    5.3M 参数
efficientnet_b1           |    7.8M 参数
...
# 新增模型
efficientnetv2_s          |   21.5M 参数
efficientnetv2_m          |   54.1M 参数
mobilenetv3_large         |    5.4M 参数
mobilenetv3_small         |    2.9M 参数
regnet_x_400mf            |    5.2M 参数
regnet_y_400mf            |    4.3M 参数
densenet121               |    8.0M 参数
densenet169               |   14.1M 参数
shufflenetv2_x0_5         |    1.4M 参数
shufflenetv2_x1_0         |    2.3M 参数
ghostnet                  |    5.2M 参数
mnasnet_1_0               |    4.4M 参数
```

### 2. 训练单个模型

#### 基本用法

```bash
python train_single_model.py --model <模型名称>
```

#### 自定义参数

```bash
python train_single_model.py --model <模型名称> --epochs <训练轮数> --batch_size <批次大小> --lr <学习率>
```

## 📝 使用示例

### 训练轻量级模型（快速测试）

```bash
# 训练 ShuffleNet V2 0.5x (最轻量)
python train_single_model.py --model shufflenetv2_x0_5 --epochs 5

# 训练 MobileNet V3 Small
python train_single_model.py --model mobilenetv3_small --epochs 10 --batch_size 64

# 训练 GhostNet
python train_single_model.py --model ghostnet --epochs 8
```

### 训练中等规模模型

```bash
# 训练 EfficientNet V2-S
python train_single_model.py --model efficientnetv2_s --epochs 10 --batch_size 32

# 训练 RegNet X-400MF
python train_single_model.py --model regnet_x_400mf --epochs 12

# 训练 DenseNet-121
python train_single_model.py --model densenet121 --epochs 10 --batch_size 32
```

### 训练大型模型

```bash
# 训练 EfficientNet V2-M (需要更多显存)
python train_single_model.py --model efficientnetv2_m --epochs 8 --batch_size 16

# 训练 DenseNet-169
python train_single_model.py --model densenet169 --epochs 10 --batch_size 32
```

## 🎯 新增模型特点

### EfficientNet V2 系列
- **efficientnetv2_s**: 21.5M参数，平衡性能与效率
- **efficientnetv2_m**: 54.1M参数，更高精度但需要更多资源
- 特点：改进的训练策略，更快的训练速度

### MobileNet V3 系列
- **mobilenetv3_large**: 5.4M参数，移动端优化
- **mobilenetv3_small**: 2.9M参数，超轻量级
- 特点：硬件感知的神经架构搜索，高效的移动端推理

### RegNet 系列
- **regnet_x_400mf**: 5.2M参数，无SE模块
- **regnet_y_400mf**: 4.3M参数，带SE模块
- 特点：设计空间搜索得出的高效架构

### DenseNet 系列
- **densenet121**: 8.0M参数，密集连接
- **densenet169**: 14.1M参数，更深的网络
- 特点：特征重用，参数效率高

### 轻量级模型
- **shufflenetv2_x0_5**: 1.4M参数，极轻量
- **shufflenetv2_x1_0**: 2.3M参数，标准版本
- **ghostnet**: 5.2M参数，Ghost模块减少计算
- **mnasnet_1_0**: 4.4M参数，移动端神经架构搜索

## 📊 输出文件

训练完成后，会生成以下文件：

### 1. 模型检查点
```
checkpoints/{model_name}/
├── best.pth              # 最佳模型权重
├── latest.pth            # 最新模型权重
└── training_history.json # 训练历史记录
```

### 2. 训练结果
```
single_model_result_{model_name}_{timestamp}.json
```

包含：
- 模型名称
- 参数量
- 最佳验证准确率
- 训练时间

## 🔧 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--model` | 模型名称（必需） | - | `efficientnetv2_s` |
| `--epochs` | 训练轮数 | 10 | `--epochs 15` |
| `--batch_size` | 批次大小 | 64 | `--batch_size 32` |
| `--lr` | 学习率 | 0.001 | `--lr 0.0005` |
| `--list_models` | 列出所有模型 | - | `--list_models` |

## 💡 训练建议

### 根据显存选择批次大小
- **4GB显存**: batch_size=16-32
- **8GB显存**: batch_size=32-64
- **12GB+显存**: batch_size=64-128

### 根据模型大小调整参数
- **轻量级模型** (<5M参数): epochs=10-15, batch_size=64
- **中等模型** (5-20M参数): epochs=8-12, batch_size=32-64
- **大型模型** (>20M参数): epochs=5-10, batch_size=16-32

### 学习率建议
- 大多数模型: 0.001 (默认)
- 大型模型: 0.0005-0.001
- 轻量级模型: 0.001-0.002

## 🚀 批量训练新模型

如果要训练多个新增模型，可以使用示例脚本：

```bash
python example_single_training.py
```

这将依次训练几个代表性的新增模型。

## 📈 性能对比

训练完成后，可以将结果与之前的模型进行对比：

1. 查看 `gpu_performance_results_*.json` 文件中的原有模型结果
2. 比较新训练模型的 `single_model_result_*.json` 文件
3. 关注准确率、参数量和训练时间的权衡

## ❓ 常见问题

### Q: 如何选择合适的模型？
A: 
- 追求精度：EfficientNet V2系列
- 追求速度：MobileNet V3、ShuffleNet V2
- 平衡性能：RegNet、GhostNet
- 特征丰富：DenseNet

### Q: 训练时显存不足怎么办？
A: 
- 减小batch_size
- 选择更轻量的模型
- 使用梯度累积

### Q: 如何提高训练效果？
A: 
- 增加训练轮数
- 调整学习率
- 使用数据增强
- 尝试不同的优化器

---

🎉 现在你可以高效地训练和比较不同的模型架构了！