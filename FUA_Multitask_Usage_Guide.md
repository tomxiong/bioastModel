# FUA多任务学习系统使用指南

## 概述

FUA多任务学习系统扩展了原有的二元分类功能，支持四个同时进行的分类任务：
- **生长级别 (Growth Level)**: negative, positive, weak_growth (3类)
- **生长模式 (Growth Pattern)**: clean, clustered, scattered, small_dots等 (9类)
- **干扰因素 (Interference Mapping)**: pores, debris, artifacts (3类，多标签)
- **精细分类 (Fine-grained)**: 基于以上标签的40个组合类别

## 1. 系统架构

### 1.1 核心组件

```
fua/
├── training/
│   ├── multitask_dataset.py      # 多任务数据加载器
│   └── multitask_trainer.py      # 多任务训练器
├── models/
│   └── multitask_models.py       # 多任务模型定义
├── evaluation/
│   └── multitask_evaluator.py    # 多任务评估器
└── scripts/
    ├── convert_to_multitask_format.py  # 数据格式转换
    └── train_multitask_models.py       # 训练和评估脚本
```

### 1.2 模型架构

系统提供两种多任务架构：

1. **标准多任务模型 (MultitaskBioastModel)**
   - 共享骨干网络
   - 独立的任务头部
   - 可选的注意力机制

2. **分层多任务模型 (HierarchicalMultitaskModel)**
   - 任务间有依赖关系
   - 生长模式依赖于生长级别
   - 精细分类融合所有任务信息

## 2. 数据准备

### 2.1 数据格式转换

如果现有数据是传统的目录结构（positive/negative），需要转换为多任务标注格式：

```bash
# 转换数据格式
.venv/bin/python scripts/convert_to_multitask_format.py \
    --data_root bioast_dataset \
    --output_dir bioast_dataset_multitask \
    --split_ratio 0.7 0.15 0.15 \
    --validate
```

### 2.2 标注文件格式

转换后的标注文件格式：

```json
[
    {
        "image_id": "image_000001",
        "file_path": "images/image_000001.png",
        "split": "train",
        "annotations": {
            "growth_level": "positive",
            "growth_pattern": "clustered",
            "interference_mapping": ["pores", "debris"],
            "fine_grained": "positive_clustered_pores"
        }
    }
]
```

### 2.3 数据集目录结构

```
bioast_dataset_multitask/
├── annotations/
│   └── multitask_annotations.json
├── images/
│   ├── image_000001.png
│   ├── image_000002.png
│   └── ...
└── conversion_report.txt
```

## 3. 训练多任务模型

### 3.1 基础训练

```bash
# 训练AirBubble多任务模型
.venv/bin/python scripts/train_multitask_models.py \
    --annotation_file bioast_dataset_multitask/annotations/multitask_annotations.json \
    --image_root bioast_dataset_multitask/images \
    --model_name multitask_airbubble_hybrid \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --use_amp \
    --output_dir experiments/multitask_airbubble
```

### 3.2 调整任务权重

```bash
# 调整各任务权重
.venv/bin/python scripts/train_multitask_models.py \
    --annotation_file bioast_dataset_multitask/annotations/multitask_annotations.json \
    --image_root bioast_dataset_multitask/images \
    --model_name multitask_airbubble_hybrid \
    --growth_level_weight 1.0 \
    --growth_pattern_weight 1.0 \
    --interference_weight 0.8 \
    --fine_grained_weight 0.5
```

### 3.3 可用模型

#### 标准多任务模型
- `multitask_airbubble_hybrid`: 基于AirBubble HybridNet (98.02% 准确率)
- `multitask_resnet18`: 基于ResNet18 (97.83% 准确率)
- `multitask_efficientnet`: 基于EfficientNet-B0 (97.54% 准确率)

#### MobileNetV3系列
- `multitask_mobilenetv3_large`: 基于MobileNetV3-Large，高性能版本
- `multitask_mobilenetv3_small`: 基于MobileNetV3-Small，轻量级版本
- `multitask_mic_mobilenetv3`: MIC专用MobileNetV3，带气泡检测和浊度分析
- `enhanced_multitask_mobilenetv3`: 增强版MobileNetV3，内置多任务优化

#### EfficientNetV2系列
- `multitask_efficientnet_v2_s`: 基于EfficientNetV2-S，最新架构
- `multitask_efficientnet_v2_b0`: 基于EfficientNetV2-B0，平衡版本

#### 分层模型
- `hierarchical_airbubble`: 分层多任务模型，任务间有依赖关系

### 3.4 训练配置示例

```python
config = {
    # 数据配置
    'annotation_file': 'bioast_dataset_multitask/annotations/multitask_annotations.json',
    'image_root': 'bioast_dataset_multitask/images',
    
    # 模型配置
    'model_name': 'multitask_airbubble_hybrid',
    
    # 训练参数
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'use_amp': True,
    'gradient_clip': 1.0,
    
    # 任务权重
    'task_weights': {
        'growth_level': 1.0,
        'growth_pattern': 1.0,
        'interference_mapping': 0.5,
        'fine_grained': 1.0
    }
}
```

## 4. 模型评估

### 4.1 评估训练好的模型

```bash
# 评估模式
.venv/bin/python scripts/train_multitask_models.py \
    --mode eval \
    --model_path checkpoints/multitask_airbubble_final.pth \
    --annotation_file bioast_dataset_multitask/annotations/multitask_annotations.json \
    --image_root bioast_dataset_multitask/images \
    --output_dir evaluation_results
```

### 4.2 评估指标

#### 单标签任务（growth_level, growth_pattern, fine_grained）
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1 Score)
- 混淆矩阵 (Confusion Matrix)

#### 多标签任务（interference_mapping）
- Hamming Loss
- F1 Micro/Macro
- Jaccard相似度
- 标签共现分析

### 4.3 可视化结果

评估完成后会生成以下可视化：

1. **performance_radar.png**: 多任务性能雷达图
2. **confusion_matrix_*.png**: 各任务混淆矩阵
3. **multilabel_heatmap_*.png**: 多标签指标热图
4. **class_distribution.png**: 类别分布对比
5. **roc_curve_*.png**: ROC曲线

## 5. 模型比较

### 5.1 运行多个实验

```python
from scripts.train_multitask_models import run_multitask_experiments

# 运行多个模型进行比较
results = run_multitask_experiments()
```

### 5.2 比较结果

比较结果包括：
- 综合得分对比
- 各任务F1分数对比
- 性能热图
- 详细评估报告

## 6. 高级功能

### 6.1 自定义任务权重

根据业务需求调整任务权重：

```python
# 如果更关注生长级别
task_weights = {
    'growth_level': 2.0,
    'growth_pattern': 1.0,
    'interference_mapping': 0.3,
    'fine_grained': 0.5
}

# 如果更关注干扰因素检测
task_weights = {
    'growth_level': 0.5,
    'growth_pattern': 0.5,
    'interference_mapping': 2.0,
    'fine_grained': 1.0
}
```

### 6.2 分层训练策略

1. **阶段1**: 冻结骨干网络，只训练任务头部
2. **阶段2**: 解冻骨干网络，端到端微调
3. **阶段3**: 使用不同学习率训练不同部分

```python
# 分层训练示例
model.freeze_backbone()
# 训练头部...
model.unfreeze_backbone()
# 端到端训练...
```

### 6.3 处理类别不平衡

```python
# 获取类别权重
dataset = MultitaskBioastDataset(annotation_file, image_root)
weights = dataset.get_class_weights('growth_level')

# 使用加权损失
criterion = nn.CrossEntropyLoss(weight=weights)
```

## 7. 集成到FUA系统

### 7.1 数据集版本管理

```python
from fua.dataset_iteration_manager import MultitaskDatasetVersionManager

# 创建多任务数据集版本管理器
version_manager = MultitaskDatasetVersionManager("multitask_dataset")

# 创建版本
version_info = version_manager.create_version("v1.0", "多任务初始版本")
```

### 7.2 参数优化

```python
from fua.parameter_optimizer import MultitaskParameterOptimizer

# 创建多任务参数优化器
optimizer = MultitaskParameterOptimizer("multitask_model", history_manager)

# 获取参数建议
suggestion = optimizer.suggest_parameters("adaptive")
```

## 8. 性能优化建议

### 8.1 训练优化

1. **使用混合精度训练**: `--use_amp`
2. **调整批次大小**: 根据GPU内存调整
3. **学习率调度**: 使用CosineAnnealingLR
4. **梯度裁剪**: 防止梯度爆炸

### 8.2 模型优化

1. **选择合适的骨干网络**: 根据精度和速度需求
2. **使用注意力机制**: 提高特征提取能力
3. **调整dropout率**: 防止过拟合
4. **特征维度**: 平衡模型容量和计算效率

### 8.3 数据优化

1. **数据增强**: 使用更多的数据增强策略
2. **类别平衡**: 使用加权损失或过采样
3. **标签质量**: 确保标注的准确性

## 9. 故障排除

### 9.1 常见问题

**问题1: CUDA内存不足**
```bash
# 减小批次大小
--batch_size 16

# 或使用梯度累积
```

**问题2: 训练不稳定**
```bash
# 降低学习率
--learning_rate 0.0001

# 使用梯度裁剪
--gradient_clip 0.5
```

**问题3: 某个任务性能差**
```bash
# 调整任务权重
--growth_level_weight 2.0  # 提高重要任务的权重
```

### 9.2 调试技巧

1. **检查数据加载**: 验证标注文件格式
2. **监控损失**: 观察各任务的损失变化
3. **可视化特征**: 使用t-SNE可视化特征分布
4. **错误分析**: 分析错误预测的样本

## 10. 示例代码

### 10.1 自定义训练循环

```python
from training.multitask_dataset import MultitaskBioastDataset
from training.multitask_trainer import MultitaskTrainer
from models.multitask_models import create_multitask_model

# 创建数据集
dataset = MultitaskBioastDataset(
    annotation_file="annotations.json",
    image_root="images",
    split="train",
    augment=True
)

# 创建模型
model = create_multitask_model(
    model_type="standard",
    backbone_name="airbubble_hybrid_net"
)

# 创建训练器
trainer = MultitaskTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# 开始训练
results = trainer.train()
```

### 10.2 推理示例

```python
import torch
from PIL import Image
from torchvision import transforms

# 加载模型
model = create_multitask_model("multitask_airbubble_hybrid")
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# 加载图像
image = Image.open("test.png").convert('RGB')
image = transform(image).unsqueeze(0)

# 推理
with torch.no_grad():
    outputs = model(image)

# 处理结果
for task_name, output in outputs.items():
    if task_name == "interference_mapping":
        probs = torch.sigmoid(output)
        predictions = (probs > 0.5).cpu().numpy()
    else:
        probs = torch.softmax(output, dim=1)
        predictions = probs.argmax(dim=1).cpu().numpy()
    
    print(f"{task_name}: {predictions}")
```

## 总结

FUA多任务学习系统提供了完整的端到端解决方案，从数据准备到模型训练、评估和部署。通过合理配置任务权重和选择合适的模型架构，可以有效地处理复杂的生物图像分类任务。

关键建议：
1. 从小规模实验开始，验证流程
2. 根据具体需求调整任务权重
3. 充分利用可视化工具分析结果
4. 持续监控和改进模型性能