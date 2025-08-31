# MobileNetV3 多类别分类系统

基于 m13.json 数据集的多任务深度学习分类系统，支持多种生物图像分类任务。

## 系统概述

本系统实现了基于 MobileNetV3 的多类别分类器，能够处理以下分类任务：

### 支持的分类任务

1. **生长级别二分类** (`growth_level_binary`)
   - 阴性 (negative): 无菌落生长
   - 阳性 (positive): 有菌落生长
   - 样本数: 1,735 (阳性: 953, 阴性: 782)

2. **生长级别三分类** (`growth_level_ternary`)
   - 阴性 (negative): 无菌落生长
   - 阳性 (positive): 明显菌落生长
   - 弱生长 (weak_growth): 微弱菌落生长
   - 样本数: 1,824 (阳性: 953, 阴性: 782, 弱生长: 89)

3. **生长模式分类** (`growth_pattern`)
   - 清亮 (clean): 无生长或清亮
   - 聚集型 (clustered): 菌落聚集生长
   - 重度生长 (heavy_growth): 重度菌落生长
   - 小点状 (small_dots): 小点状生长
   - 样本数: 1,800

4. **干扰因素检测** (`interference_binary`)
   - 无干扰 (no_interference): 无干扰因素
   - 有干扰 (has_interference): 包含气孔、杂质等干扰
   - 样本数: 1,824 (无干扰: 583, 有干扰: 1,241)

## 数据集特性

- **图像尺寸**: 70×70 像素
- **图像格式**: PNG
- **数据来源**: ni/m13.json 标注文件
- **样本总数**: 1,824 个标注
- **图像类型**: 生物培养孔图像

## 快速开始

### 1. 测试系统

```bash
# 测试所有分类任务
.venv311/Scripts/python test_mobilenetv3_multiclass.py

# 运行演示
.venv311/Scripts/python demo_mobilenetv3_multiclass.py
```

### 2. 单任务训练

```bash
# 训练生长级别二分类
.venv311/Scripts/python train_mobilenetv3_multitask.py --task growth_level_binary --model small --epochs 30

# 训练生长模式分类
.venv311/Scripts/python train_mobilenetv3_multitask.py --task growth_pattern --model small --epochs 30 --batch_size 64

# 使用混合精度训练
.venv311/Scripts/python train_mobilenetv3_multitask.py --task growth_level_binary --model large --epochs 50 --mixed_precision
```

### 3. 批量训练

```bash
# 训练所有任务和小模型
.venv311/Scripts/python batch_train_mobilenetv3.py --tasks growth_level_binary growth_level_ternary growth_pattern interference_binary --models small --epochs 30

# 训练所有任务和所有模型
.venv311/Scripts/python batch_train_mobilenetv3.py --epochs 50 --batch_size 128 --lr 0.001
```

## 文件结构

```
├── ni_dataset.py                    # 增强的数据集类
├── train_mobilenetv3_multitask.py   # 多任务训练脚本
├── batch_train_mobilenetv3.py       # 批量训练脚本
├── demo_mobilenetv3_multiclass.py   # 演示脚本
├── test_mobilenetv3_multiclass.py   # 测试脚本
├── models/mobilenet_v3.py           # MobileNetV3 模型实现
├── ni/m13.json                      # 数据标注文件
└── experiments/                     # 实验结果目录
    ├── mobilenetv3_task_model_*/    # 单个实验结果
    └── batch_results/                # 批量训练结果
```

## 模型架构

### MobileNetV3 特性
- **轻量级**: Small 版本约 1.5M 参数
- **高效**: 优化的深度可分离卷积
- **现代架构**: 包含 SE (Squeeze-and-Excitation) 注意力机制
- **多尺度**: 支持不同的宽度乘数

### 模型版本
- **MobileNetV3-Small**: 适用于资源受限环境
- **MobileNetV3-Large**: 更高精度，更多参数

## 训练配置

### 推荐参数
```python
config = {
    'model_type': 'small',                    # 'small' or 'large'
    'classification_task': 'growth_level_binary',
    'batch_size': 128,                        # 根据GPU内存调整
    'learning_rate': 0.001,                   # 初始学习率
    'epochs': 50,                            # 训练轮数
    'weight_decay': 0.01,                    # 权重衰减
    'mixed_precision': False,                 # 混合精度训练
    'patience': 15                            # 早停耐心值
}
```

### 数据增强
训练集使用以下数据增强：
- 随机水平翻转
- 随机垂直翻转
- 随机旋转 (±15度)
- 颜色抖动 (亮度、对比度、饱和度、色调)

## 性能指标

### 推理性能
- **平均推理时间**: ~2-5ms (CPU/GPU)
- **吞吐量**: ~200-500 图像/秒
- **模型大小**: 1.5M-5.4M 参数

### 预期准确率
基于数据集特征：
- **二分类任务**: 90%+ (balanced dataset)
- **三分类任务**: 85%+ (with weak_growth class)
- **生长模式**: 80%+ (4-class classification)
- **干扰检测**: 85%+ (binary classification)

## 实验结果

训练完成后，每个实验会生成：
- **训练历史**: `train_history.json`
- **测试结果**: `test_results.json`
- **训练曲线**: `training_history.png`
- **混淆矩阵**: `confusion_matrix.png`
- **最佳模型**: `best.pth`
- **日志文件**: `training.log`

## 高级功能

### 1. 自定义分类任务
```python
from ni_dataset import NIDataset

# 创建自定义数据集
dataset = NIDataset(
    json_path='ni/m13.json',
    image_dir='ni',
    split='train',
    classification_task='growth_level_binary'
)
```

### 2. 模型推理
```python
from demo_mobilenetv3_multiclass import MobileNetV3Demo

# 创建演示实例
demo = MobileNetV3Demo(
    model_path='experiments/mobilenetv3_growth_level_binary_small_20240831_120000/best.pth',
    classification_task='growth_level_binary'
)

# 预测单张图片
result = demo.predict_image('ni/EB10000026/hole_25.png')
print(f"预测结果: {result['class_name']} (置信度: {result['confidence']:.4f})")
```

### 3. 批量预测
```python
# 批量预测
image_paths = ['ni/EB10000026/hole_25.png', 'ni/EB10000026/hole_26.png']
results = demo.batch_predict(image_paths)

for result in results:
    if 'error' not in result:
        print(f"{result['image_path']}: {result['class_name']}")
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少批次大小: `--batch_size 32`
   - 使用小模型: `--model small`

2. **训练不收敛**
   - 降低学习率: `--lr 0.0001`
   - 增加训练轮数: `--epochs 100`

3. **类别不平衡**
   - 使用加权损失函数
   - 调整数据采样策略

4. **编码问题**
   - 设置控制台编码: `chcp 65001`
   - 使用UTF-8环境变量

## 扩展开发

### 添加新的分类任务
1. 在 `ni_dataset.py` 中添加新的任务类型
2. 实现标签映射逻辑
3. 更新训练脚本

### 模型改进
- 添加注意力机制
- 实现集成学习
- 优化损失函数
- 数据增强策略

## 引用

如果您使用了本系统，请引用：
- MobileNetV3: Searching for MobileNetV3
- PyTorch: An open source machine learning framework

## 许可证

本项目遵循 MIT 许可证。