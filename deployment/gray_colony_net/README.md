# gray_colony_net C#部署指南

## 概述
本目录包含灰度菌落检测网络的C#部署文件，支持ONNX运行时推理。

## 文件说明
- `gray_colony_net.cs`: 主要的推理类
- `gray_colony_net.csproj`: 项目配置文件
- `UsageExample.cs`: 使用示例
- `gray_colony_net.onnx`: ONNX模型文件（需要从上级目录复制）

## 使用步骤

### 1. 环境要求
- .NET 6.0 或更高版本
- ONNX Runtime 1.16.0 或更高版本

### 2. 安装依赖
```bash
dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.0
```

### 3. 基本使用
```csharp
using BioAST.ColonyDetection;

// 初始化检测器
string modelPath = @"path	o\gray_colony_net.onnx";
using var detector = new gray_colony_net(modelPath);

// 加载70×70灰度图像
byte[,] grayImage = LoadYourImage();

// 运行检测
var result = detector.Predict(grayImage);

// 输出结果
Console.WriteLine($"预测类别: {result.PredictedClass}");
Console.WriteLine($"置信度: {result.Confidence:P2}");
```

## 输入输出说明

### 输入
- 图像尺寸：70×70像素
- 图像格式：灰度（单通道）
- 数值范围：0-255

### 输出
- PositiveClusterScore: 阳性聚焦型菌落得分（0-1）
- NegativePoreScore: 阴性气孔型得分（0-1）
- WeakGrowthScore: 弱生长小点型得分（0-1）
- Confidence: 预测置信度（0-1）
- PredictedClass: 预测类别名称

## 性能优化建议
1. 使用批量处理提高吞吐量
2. 考虑使用GPU加速（安装Microsoft.ML.OnnxRuntime.Gpu包）
3. 预处理图像时使用并行处理

## 注意事项
- 确保输入图像已正确缩放到70×70
- 图像应为真正的灰度图像，不是RGB转换的
- 模型对中空结构（气孔）有特殊优化
