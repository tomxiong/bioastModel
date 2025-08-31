# M16多任务MobileNetV3 ONNX模型使用指南

## 概述

M16多任务MobileNetV3模型是一个高效的生物图像分类系统，支持70×70像素图像的多任务分析，包括生长级别、生长模式、干扰因素和精细分类四个任务。

## 模型信息

- **模型名称**: M16_MultiTask_MobileNetV3
- **输入尺寸**: 3×70×70 (CHW格式)
- **文件大小**: 9.6MB
- **参数数量**: 2.51M
- **ONNX版本**: opset14
- **验证准确率**: 91.18%

## 文件清单

```
onnx_models/
├── m16_multitask_mobilenetv3.onnx          # ONNX模型文件
├── m16_multitask_metadata.json             # 模型元数据
├── m16_multitask_inference_example.py     # 推理示例代码
└── validation_report.json                  # 验证报告
```

## 安装依赖

```bash
pip install onnxruntime numpy pillow torchvision
```

## 快速开始

### 1. 基本推理

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

# 加载模型
session = ort.InferenceSession("onnx_models/m16_multitask_mobilenetv3.onnx")
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

# 预处理
transform = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# 加载图像
image = Image.open("your_image.jpg").convert('RGB')
input_tensor = transform(image).numpy()
input_tensor = np.expand_dims(input_tensor, axis=0)  # 添加batch维度

# 推理
outputs = session.run(output_names, {input_name: input_tensor})

# 处理结果
growth_level_logits = outputs[0][0]  # 生长级别 (3类)
growth_pattern_logits = outputs[1][0]  # 生长模式 (9类)
interference_logits = outputs[2][0]  # 干扰因素 (3类，多标签)
fine_grained_logits = outputs[3][0]  # 精细分类 (40类)
```

### 2. 使用推理类

```python
from m16_multitask_inference_example import M16MultitaskInference

# 初始化推理器
inference = M16MultitaskInference(
    model_path="onnx_models/m16_multitask_mobilenetv3.onnx",
    metadata_path="onnx_models/m16_multitask_metadata.json"
)

# 预测图像
results = inference.predict("your_image.jpg")

# 显示结果
print(f"生长级别: {results['growth_level']['class_name']}")
print(f"生长模式: {results['growth_pattern']['class_name']}")
print(f"干扰因素: {[cls['class_name'] for cls in results['interference_factors']['active_classes']]}")
print(f"精细分类: {results['fine_grained']['class_name']}")
```

## 输出说明

### 1. 生长级别 (Growth Level)
- **输出**: logits (3类)
- **类别**: ['negative', 'positive', 'weak_growth']
- **处理**: softmax + argmax
- **用途**: 判断是否有菌落生长

### 2. 生长模式 (Growth Pattern)
- **输出**: logits (9类)
- **类别**: ['clean', 'clustered', 'scattered', 'heavy_growth', 'small_dots', 'irregular_areas', 'light_gray', 'default_positive', 'default_weak_growth']
- **处理**: softmax + argmax
- **用途**: 识别菌落生长形态

### 3. 干扰因素 (Interference Factors)
- **输出**: logits (3类)
- **类别**: ['pores', 'debris', 'artifacts']
- **处理**: sigmoid + threshold (0.5)
- **用途**: 检测图像中的干扰因素
- **注意**: 多标签分类，可同时检测多个因素

### 4. 精细分类 (Fine Grained)
- **输出**: logits (40类)
- **类别**: 基于组合逻辑的40个精细类别
- **处理**: softmax + argmax
- **用途**: 详细的样本分类

## 预处理要求

### 图像预处理
```python
transforms.Compose([
    transforms.Resize((70, 70)),  # 调整尺寸
    transforms.ToTensor(),      # 转换为张量 [0,1]
    transforms.Normalize(       # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 输入格式
- **格式**: CHW (Channel, Height, Width)
- **数值范围**: 标准化后的浮点数
- **批量处理**: 支持动态batch size

## 性能指标

### 推理性能
- **PyTorch**: 10.35ms
- **ONNX**: 3.88ms
- **加速比**: 2.67x

### 准确率
- **综合准确率**: 90.69%
- **生长级别**: 96.32%
- **生长模式**: 95.59%
- **精细分类**: 80.15%

## 部署建议

### 1. 服务器部署
```python
import onnxruntime as ort

# 使用CUDA加速
session = ort.InferenceSession(
    "m16_multitask_mobilenetv3.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### 2. 批量处理
```python
# 批量推理
batch_input = np.stack([image1, image2, image3])  # [N, 3, 70, 70]
outputs = session.run(output_names, {input_name: batch_input})
```

### 3. 内存优化
```python
# 设置图优化级别
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession("model.onnx", sess_options=session_options)
```

## 错误处理

### 常见问题

1. **输入尺寸错误**
   ```python
   # 确保输入是 [N, 3, 70, 70]
   assert input_tensor.shape == (batch_size, 3, 70, 70)
   ```

2. **数值范围错误**
   ```python
   # 检查是否正确标准化
   assert -3 < input_tensor.mean() < 3
   ```

3. **模型加载失败**
   ```python
   # 检查文件路径
   assert Path("m16_multitask_mobilenetv3.onnx").exists()
   ```

## 示例应用

### 1. 图像分类器
```python
def classify_colony_image(image_path):
    """菌落图像分类"""
    inference = M16MultitaskInference("m16_multitask_mobilenetv3.onnx")
    results = inference.predict(image_path)
    
    # 简单分类逻辑
    if results['growth_level']['class_name'] == 'negative':
        return "阴性样本"
    elif results['growth_level']['class_name'] == 'positive':
        return f"阳性样本 - {results['growth_pattern']['class_name']}"
    else:
        return f"弱生长样本 - {results['growth_pattern']['class_name']}"
```

### 2. 质量检测器
```python
def detect_quality_issues(image_path):
    """检测图像质量问题"""
    inference = M16MultitaskInference("m16_multitask_mobilenetv3.onnx")
    results = inference.predict(image_path)
    
    issues = []
    for factor in results['interference_factors']['active_classes']:
        issues.append(factor['class_name'])
    
    return issues
```

### 3. 批量处理
```python
def batch_process_images(image_dir, output_file):
    """批量处理图像"""
    inference = M16MultitaskInference("m16_multitask_mobilenetv3.onnx")
    results = []
    
    for image_path in Path(image_dir).glob("*.jpg"):
        try:
            result = inference.predict(str(image_path))
            result['image_path'] = str(image_path)
            results.append(result)
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")
    
    # 保存结果
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
```

## 技术支持

### 验证信息
- **输出差异**: < 1e-3 (与PyTorch模型对比)
- **推理稳定性**: 100%成功率
- **兼容性**: ONNX Runtime 1.0+

### 联系方式
如有问题，请参考验证报告和示例代码。

## 更新日志

### v1.0 (2025-09-01)
- 初始版本发布
- 支持四个多任务分类
- ONNX格式转换完成
- 性能优化和验证通过