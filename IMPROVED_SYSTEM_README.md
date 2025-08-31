# MobileNetV3 多类别分类系统 - 完整版

基于 m13.json 数据集的完整多任务深度学习分类系统，确保训练集、验证集、测试集的一致性和正确分割。

## ✅ 系统特性

### 🎯 数据集分割保证
- **预分割文件**: 使用 `dataset_splits/` 目录中的预生成分割文件
- **一致性保证**: 确保训练集、验证集、测试集之间无样本重叠
- **分层分割**: 70% 训练集、15% 验证集、15% 测试集
- **类别平衡**: 每个分割中保持类别分布一致

### 📊 数据集统计

| 分类任务 | 类别数 | 总样本 | 训练集 | 验证集 | 测试集 |
|---------|-------|--------|--------|--------|--------|
| growth_level_binary | 2 | 1,735 | 1,214 (70%) | 260 (15%) | 261 (15%) |
| growth_level_ternary | 3 | 1,824 | 1,276 (70%) | 273 (15%) | 275 (15%) |
| growth_pattern | 4 | 1,800 | 1,259 (70%) | 269 (15%) | 272 (15%) |
| interference_binary | 2 | 1,824 | 1,276 (70%) | 273 (15%) | 275 (15%) |

### 🛠️ 核心文件

1. **数据集处理**:
   - `improved_ni_dataset.py` - 改进的数据集类，确保分割一致性
   - `validate_dataset_splits.py` - 数据集分割验证工具

2. **训练脚本**:
   - `train_improved_mobilenetv3.py` - 使用改进数据集的训练脚本
   - `batch_train_mobilenetv3.py` - 批量训练脚本

3. **模型和演示**:
   - `models/mobilenet_v3.py` - MobileNetV3 模型实现
   - `demo_mobilenetv3_multiclass.py` - 多类别分类演示

## 🚀 快速开始

### 1. 系统验证

```bash
# 验证数据集分割
.venv311/Scripts/python validate_dataset_splits.py

# 测试改进的数据集
.venv311/Scripts/python improved_ni_dataset.py
```

### 2. 单任务训练

```bash
# 训练生长级别二分类
.venv311/Scripts/python train_improved_mobilenetv3.py --task growth_level_binary --model small --epochs 30

# 训练生长模式分类
.venv311/Scripts/python train_improved_mobilenetv3.py --task growth_pattern --model small --epochs 30 --batch_size 64

# 使用混合精度训练
.venv311/Scripts/python train_improved_mobilenetv3.py --task growth_level_binary --model large --epochs 50 --mixed_precision
```

### 3. 批量训练

```bash
# 训练所有任务和小模型
.venv311/Scripts/python batch_train_mobilenetv3.py --tasks growth_level_binary growth_level_ternary growth_pattern interference_binary --models small --epochs 30

# 训练所有任务和所有模型
.venv311/Scripts/python batch_train_mobilenetv3.py --epochs 50 --batch_size 128 --lr 0.001
```

### 4. 演示和测试

```bash
# 运行多类别分类演示
.venv311/Scripts/python demo_mobilenetv3_multiclass.py

# 测试模型功能
.venv311/Scripts/python test_mobilenetv3_multiclass.py
```

## 📈 预期性能

### 推理性能
- **平均推理时间**: ~2-5ms (CPU/GPU)
- **吞吐量**: ~200-500 图像/秒
- **模型大小**: 1.5M-5.4M 参数

### 预期准确率
基于数据集质量和分割策略：
- **二分类任务**: 90%+ (balanced dataset)
- **三分类任务**: 85%+ (with weak_growth class)
- **生长模式**: 80%+ (4-class classification)
- **干扰检测**: 85%+ (binary classification)

## 🔧 高级功能

### 1. 自定义分类任务

```python
from improved_ni_dataset import create_improved_ni_dataloaders

# 创建自定义数据加载器
train_loader, val_loader, test_loader, class_info = create_improved_ni_dataloaders(
    json_path='ni/m13.json',
    image_dir='ni',
    batch_size=64,
    classification_task='growth_level_binary'
)
```

### 2. 模型推理

```python
from demo_mobilenetv3_multiclass import MobileNetV3Demo

# 创建演示实例
demo = MobileNetV3Demo(
    model_path='experiments/improved_mobilenetv3_growth_level_binary_small_20240831_120000/best.pth',
    classification_task='growth_level_binary'
)

# 预测单张图片
result = demo.predict_image('ni/EB10000026/hole_25.png')
print(f"预测结果: {result['class_name']} (置信度: {result['confidence']:.4f})")
```

### 3. 数据集验证

```python
from validate_dataset_splits import create_stratified_splits, validate_splits

# 创建自定义分割
splits = create_stratified_splits(samples, 'custom_task')

# 验证分割结果
validate_splits(splits, class_names, 'custom_task')
```

## 📁 文件结构

```
├── improved_ni_dataset.py              # 改进的数据集类
├── train_improved_mobilenetv3.py        # 改进的训练脚本
├── validate_dataset_splits.py          # 分割验证工具
├── demo_mobilenetv3_multiclass.py      # 演示脚本
├── test_mobilenetv3_multiclass.py      # 测试脚本
├── batch_train_mobilenetv3.py          # 批量训练脚本
├── models/mobilenet_v3.py              # MobileNetV3 模型
├── ni/m13.json                         # 数据标注文件
├── dataset_splits/                     # 预分割文件
│   ├── growth_level_binary_splits.json
│   ├── growth_level_ternary_splits.json
│   ├── growth_pattern_splits.json
│   └── interference_binary_splits.json
└── experiments/                        # 实验结果
    └── improved_mobilenetv3_*/
```

## 🎯 训练最佳实践

### 1. 数据集准备
- ✅ 使用预分割文件确保一致性
- ✅ 验证数据集无重叠
- ✅ 检查类别分布平衡
- ✅ 确认图片文件存在

### 2. 训练配置
```python
# 推荐配置
config = {
    'model_type': 'small',                    # 小模型更快
    'classification_task': 'growth_level_binary',
    'batch_size': 64,                         # 根据GPU内存调整
    'learning_rate': 0.001,                   # 标准学习率
    'epochs': 30,                            # 适中轮数
    'weight_decay': 0.01,                    # 正则化
    'mixed_precision': False,                 # 根据GPU支持
    'patience': 15                            # 早停耐心值
}
```

### 3. 监控训练
- 查看训练曲线 (`training_history.png`)
- 监控验证集准确率
- 检查类别准确率平衡
- 观察学习率衰减

## 🔍 故障排除

### 常见问题

1. **分割文件不存在**
   ```bash
   # 重新生成分割文件
   .venv311/Scripts/python validate_dataset_splits.py
   ```

2. **数据集重叠**
   ```bash
   # 验证数据集一致性
   .venv311/Scripts/python improved_ni_dataset.py
   ```

3. **内存不足**
   ```bash
   # 减少批次大小
   .venv311/Scripts/python train_improved_mobilenetv3.py --batch_size 32
   ```

4. **训练不收敛**
   ```bash
   # 调整学习率
   .venv311/Scripts/python train_improved_mobilenetv3.py --lr 0.0001
   ```

## 📊 结果分析

训练完成后，每个实验会生成：

### 输出文件
- **训练历史**: `train_history.json`
- **测试结果**: `test_results.json`
- **训练曲线**: `training_history.png`
- **混淆矩阵**: `confusion_matrix.png`
- **最佳模型**: `best.pth`
- **日志文件**: `training.log`

### 关键指标
- **验证集准确率**: 模型选择的主要指标
- **测试集准确率**: 最终性能评估
- **类别准确率**: 各类别的识别性能
- **混淆矩阵**: 错误模式分析

## 🎉 总结

这个完整的MobileNetV3多类别分类系统提供了：

1. **可靠的数据集分割**: 确保训练、验证、测试集的一致性
2. **多任务支持**: 支持多种生物图像分类任务
3. **现代训练技术**: 混合精度、早停、学习率调度
4. **完整的评估体系**: 详细的性能分析和可视化
5. **易于使用**: 简单的命令行接口和丰富的配置选项

系统现在完全可以投入使用，为各种生物图像分类任务提供高质量的深度学习解决方案。