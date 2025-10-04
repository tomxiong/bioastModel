# 🚀 BioAst ONNX C# Quick Start Guide

5分钟内开始使用 MobileNetV4 v0.11.0 ONNX 模型进行推理。

## 📦 前置要求

- ✅ .NET 6.0 SDK 或更高版本
- ✅ ONNX 模型文件: `model.onnx`
- ✅ 测试图像 (70×70 灰度或彩色图像)

## ⚡ 5分钟快速开始

### 步骤 1: 创建项目

```bash
# 进入项目目录
cd deployment/csharp_example/BioastOnnxInference

# 恢复 NuGet 包
dotnet restore
```

### 步骤 2: 复制模型文件

```bash
# 将 ONNX 模型复制到项目目录
cp ../onnx_models/mobilenetv4_v0.11.0/model.onnx ./model.onnx
```

或者在代码中指定完整路径:
```csharp
string modelPath = "../onnx_models/mobilenetv4_v0.11.0/model.onnx";
```

### 步骤 3: 运行推理

```bash
# 单张图像推理
dotnet run test_image.png
```

### 步骤 4: 查看结果

```
Model loaded successfully!
  Input: input
  Outputs: growth_level, growth_pattern, interference_factors

================================================================================
MobileNetV4 v0.11.0 Inference Results
================================================================================

[Growth Level]
  Prediction: positive (confidence: 98.54%)
  Probabilities: negative=0.0146, positive=0.9854

[Growth Pattern]
  Prediction: clustered (confidence: 89.23%)
  Top 3 probabilities:
    clustered: 0.8923
    heavy_growth: 0.0612
    even_scattered: 0.0254

[Interference Factors]
  pores: NOT DETECTED (score: 0.1234, threshold: 0.40)
  artifacts: NOT DETECTED (score: 0.0567, threshold: 0.45)
  debris: DETECTED (score: 0.6789, threshold: 0.15)
  contamination: NOT DETECTED (score: 0.0012, threshold: 0.50)

================================================================================
```

## 🔧 常用命令

### 编译项目
```bash
dotnet build
```

### 发布为独立可执行文件
```bash
# Windows
dotnet publish -c Release -r win-x64 --self-contained

# Linux
dotnet publish -c Release -r linux-x64 --self-contained

# macOS
dotnet publish -c Release -r osx-x64 --self-contained
```

### 运行测试
```bash
# 处理单张图像
dotnet run sample.png

# 批量处理目录
# 修改 Program.cs 调用 BatchInferenceExample.RunBatchInference()
dotnet run
```

## 📝 基础代码示例

### 最简单的推理代码

```csharp
using BioastOnnxInference;

// 1. 加载模型
var predictor = new BioastPredictor("model.onnx");

// 2. 运行推理
var result = predictor.Predict("image.png");

// 3. 获取结果
Console.WriteLine($"Growth Level: {result.GrowthLevel.Label}");
Console.WriteLine($"Confidence: {result.GrowthLevel.Confidence:P2}");
```

### 批量处理示例

```csharp
var predictor = new BioastPredictor("model.onnx");

foreach (var imagePath in Directory.GetFiles("images", "*.png"))
{
    var result = predictor.Predict(imagePath);

    Console.WriteLine($"{Path.GetFileName(imagePath)}: {result.GrowthLevel.Label}");
}
```

### 检测干扰因素

```csharp
var predictor = new BioastPredictor("model.onnx");
var result = predictor.Predict("sample.png");

// 检查是否有干扰
bool hasInterference = result.InterferenceFactors.Any(f => f.IsPresent);

if (hasInterference)
{
    Console.WriteLine("⚠️ 检测到干扰因素:");
    foreach (var factor in result.InterferenceFactors.Where(f => f.IsPresent))
    {
        Console.WriteLine($"  - {factor.Name} (score: {factor.Score:F4})");
    }
}
```

## 🎯 项目结构

```
BioastOnnxInference/
├── BioastOnnxInference.csproj     # 项目配置 (NuGet 依赖)
├── Program.cs                      # 主程序 (单张图像推理)
├── BatchInferenceExample.cs        # 批量处理和 CSV 导出
├── model.onnx                      # ONNX 模型文件 (需要复制)
└── README.md                       # 详细文档
```

## 📊 性能指标

基于 MobileNetV4 v0.11.0 在测试集上的性能:

| 指标 | 数值 |
|------|------|
| **总体准确率** | 94.26% |
| **Growth Level** | 98.53% |
| **Growth Pattern** | 87.31% |
| **Interference Overall** | 96.93% |
| **推理时间** (CPU) | ~1.75 ms/图像 |
| **吞吐量** | ~570 图像/秒 |
| **模型大小** | 3.69 MB |
| **参数量** | 952,201 |

## 🔍 常见问题

### Q1: "ONNX model not found" 错误

**解决方案**: 确保 `model.onnx` 在正确位置或修改代码中的路径:

```csharp
string modelPath = "/absolute/path/to/model.onnx";
```

### Q2: "Image not found" 错误

**解决方案**: 使用绝对路径或确保图像文件在当前工作目录:

```bash
dotnet run /full/path/to/image.png
```

### Q3: 推理结果不准确

**检查清单**:
- ✅ 图像是否为 70×70 像素 (会自动调整大小)
- ✅ 图像是否为灰度或彩色 (会自动转换为灰度)
- ✅ 图像归一化到 [0, 1] 范围 (自动处理)

### Q4: 性能优化建议

**提升性能**:
1. **使用 GPU** (需要 `Microsoft.ML.OnnxRuntime.Gpu` 包)
2. **批量处理** 多张图像
3. **使用单例模式** 共享模型实例
4. **启用并行处理** 利用多核 CPU

```csharp
// GPU 加速示例
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider_CUDA(0);  // 使用 GPU 0
_session = new InferenceSession(modelPath, sessionOptions);
```

## 📚 下一步

1. **基础使用**: 阅读 [README.md](README.md) 了解详细配置
2. **高级示例**: 查看 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) 学习更多用法
3. **生产部署**: 参考 REST API 和数据库集成示例
4. **性能优化**: 了解并行处理和缓存策略

## 🆘 获取帮助

- **项目文档**: [README.md](README.md)
- **使用示例**: [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
- **模型信息**: `../onnx_models/mobilenetv4_v0.11.0/model_info.json`
- **性能报告**: `/home/aaa/ws/bioastModel/V0.11.0_EVALUATION_SUMMARY.md`

---

**快速上手时间**: ~5 分钟
**模型版本**: MobileNetV4 v0.11.0
**最后更新**: 2025-10-04
