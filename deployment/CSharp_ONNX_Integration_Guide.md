# C# 项目 ONNX 多任务模型集成指南

**文档版本:** 1.0  
**创建时间:** 2025-09-19  
**适用模型:** EfficientNet-B0, ResNet-34, MobileNetV3 多任务模型

---

## 📋 目录

1. [项目概览](#项目概览)
2. [环境配置](#环境配置)
3. [模型文件准备](#模型文件准备)
4. [核心类库设计](#核心类库设计)
5. [完整示例代码](#完整示例代码)
6. [性能优化](#性能优化)
7. [错误处理](#错误处理)
8. [部署建议](#部署建议)
9. [故障排除](#故障排除)

---

## 📊 项目概览

### 模型特征
- **输入:** 70×70 灰度图像
- **输出:** 4个多任务分类结果
- **格式:** ONNX (Open Neural Network Exchange)
- **推理引擎:** ONNX Runtime

### 可用模型对比

| 模型 | 文件大小 | 验证准确率 | 推理速度 | 加速比 | 推荐场景 |
|------|----------|------------|----------|--------|----------|
| **MobileNetV3** | 15.0 MB | 90.06% | 1.04 ms | 4.7x | ⭐ 移动端/边缘设备 |
| **EfficientNet-B0** | 18.9 MB | 62.62% | 1.65 ms | 5.3x | ⭐ 最快推理 |
| **ResNet-34** | 94.3 MB | 62.82% | 3.88 ms | 2.5x | 服务器端 |

**推荐选择:** MobileNetV3 (最佳准确率和性能平衡)

---

## ⚙️ 环境配置

### 1. NuGet 包安装

在您的 C# 项目中安装以下 NuGet 包：

```xml
<!-- 在 .csproj 文件中添加 -->
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.16.3" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu" Version="1.16.3" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.0.2" />
<PackageReference Include="System.Numerics.Tensors" Version="0.1.0" />
```

### 2. 使用包管理器控制台

```powershell
Install-Package Microsoft.ML.OnnxRuntime -Version 1.16.3
Install-Package Microsoft.ML.OnnxRuntime.Gpu -Version 1.16.3
Install-Package SixLabors.ImageSharp -Version 3.0.2
Install-Package System.Numerics.Tensors -Version 0.1.0
```

### 3. 系统要求

- **.NET:** 6.0 或更高版本
- **内存:** 最少 4GB RAM
- **GPU (可选):** 支持 CUDA 的 NVIDIA GPU
- **平台:** Windows, Linux, macOS

---

## 📁 模型文件准备

### 1. 模型文件下载

确保您有以下 ONNX 模型文件：

```
models/
├── fixed_mobilenetv3_multitask.onnx          # 推荐模型
├── fixed_efficientnet_b0_multitask.onnx      # 最快推理
└── resnet34_multitask.onnx                   # 最大模型
```

### 2. 文件验证

```csharp
public static bool ValidateModelFile(string modelPath)
{
    try
    {
        var fileInfo = new FileInfo(modelPath);
        if (!fileInfo.Exists)
        {
            Console.WriteLine($"模型文件不存在: {modelPath}");
            return false;
        }
        
        // 验证文件大小
        var expectedSizes = new Dictionary<string, long>
        {
            { "fixed_mobilenetv3_multitask.onnx", 15_000_000 },      // ~15MB
            { "fixed_efficientnet_b0_multitask.onnx", 18_900_000 }, // ~18.9MB
            { "resnet34_multitask.onnx", 94_300_000 }               // ~94.3MB
        };
        
        string fileName = fileInfo.Name;
        if (expectedSizes.ContainsKey(fileName))
        {
            long expectedSize = expectedSizes[fileName];
            if (Math.Abs(fileInfo.Length - expectedSize) > expectedSize * 0.1) // 10% 容差
            {
                Console.WriteLine($"警告: 文件大小异常 {fileName}: {fileInfo.Length} bytes");
            }
        }
        
        Console.WriteLine($"✅ 模型文件验证通过: {fileName} ({fileInfo.Length / 1024 / 1024}MB)");
        return true;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"模型文件验证失败: {ex.Message}");
        return false;
    }
}
```

---

## 🏗️ 核心类库设计

### 1. 数据结构定义

```csharp
using System;
using System.Collections.Generic;

/// <summary>
/// 多任务分类结果
/// </summary>
public class MultitaskResult
{
    /// <summary>
    /// 生长水平分类 (0: 阴性, 1: 阳性)
    /// </summary>
    public GrowthLevelResult GrowthLevel { get; set; }
    
    /// <summary>
    /// 生长模式分类 (12种模式)
    /// </summary>
    public GrowthPatternResult GrowthPattern { get; set; }
    
    /// <summary>
    /// 干扰因素检测 (多标签)
    /// </summary>
    public InterferenceFactorsResult InterferenceFactors { get; set; }
    
    /// <summary>
    /// 微生物类型分类 (4种类型)
    /// </summary>
    public MicrobeTypeResult MicrobeType { get; set; }
    
    /// <summary>
    /// 推理耗时 (毫秒)
    /// </summary>
    public double InferenceTimeMs { get; set; }
    
    /// <summary>
    /// 整体置信度
    /// </summary>
    public double OverallConfidence => 
        (GrowthLevel.Confidence + GrowthPattern.Confidence + 
         MicrobeType.Confidence + InterferenceFactors.AverageConfidence) / 4.0;
}

/// <summary>
/// 生长水平结果
/// </summary>
public class GrowthLevelResult
{
    public enum Level { Negative = 0, Positive = 1 }
    
    public Level PredictedLevel { get; set; }
    public double Confidence { get; set; }
    public double[] Probabilities { get; set; } // [negative_prob, positive_prob]
    
    public override string ToString() => 
        $"{PredictedLevel} (置信度: {Confidence:F3})";
}

/// <summary>
/// 生长模式结果
/// </summary>
public class GrowthPatternResult
{
    public enum Pattern
    {
        Clean = 0, Clustered = 1, WeakScattered = 2, HeavyGrowth = 3,
        Focal = 4, Diffuse = 5, Patchy = 6, Confluent = 7,
        Discrete = 8, Uniform = 9, Sparse = 10, Dense = 11
    }
    
    public Pattern PredictedPattern { get; set; }
    public double Confidence { get; set; }
    public double[] Probabilities { get; set; } // 12个类别的概率
    
    /// <summary>
    /// 获取前N个最可能的模式
    /// </summary>
    public List<(Pattern pattern, double probability)> GetTopPredictions(int topN = 3)
    {
        var results = new List<(Pattern, double)>();
        for (int i = 0; i < Probabilities.Length; i++)
        {
            results.Add(((Pattern)i, Probabilities[i]));
        }
        return results.OrderByDescending(x => x.Item2).Take(topN).ToList();
    }
    
    public override string ToString() => 
        $"{PredictedPattern} (置信度: {Confidence:F3})";
}

/// <summary>
/// 干扰因素结果 (多标签分类)
/// </summary>
public class InterferenceFactorsResult
{
    public enum Factor { Pores = 0, Debris = 1, Artifacts = 2, Contamination = 3 }
    
    public Dictionary<Factor, bool> DetectedFactors { get; set; }
    public Dictionary<Factor, double> Confidences { get; set; }
    
    public double AverageConfidence => Confidences.Values.Average();
    
    /// <summary>
    /// 获取检测到的干扰因素列表
    /// </summary>
    public List<Factor> GetDetectedFactors() => 
        DetectedFactors.Where(kvp => kvp.Value).Select(kvp => kvp.Key).ToList();
    
    public override string ToString()
    {
        var detected = GetDetectedFactors();
        return detected.Any() 
            ? $"检测到: {string.Join(", ", detected)} (平均置信度: {AverageConfidence:F3})"
            : "无干扰因素";
    }
}

/// <summary>
/// 微生物类型结果
/// </summary>
public class MicrobeTypeResult
{
    public enum Type { TypeA = 0, TypeB = 1, TypeC = 2, TypeD = 3 }
    
    public Type PredictedType { get; set; }
    public double Confidence { get; set; }
    public double[] Probabilities { get; set; } // 4个类别的概率
    
    public override string ToString() => 
        $"{PredictedType} (置信度: {Confidence:F3})";
}
```

### 2. 主推理类

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

/// <summary>
/// 多任务 ONNX 模型推理器
/// </summary>
public class MultitaskOnnxPredictor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly SessionOptions _sessionOptions;
    private bool _disposed = false;
    
    // 任务输出名称映射
    private static readonly Dictionary<string, string> TaskOutputMapping = new()
    {
        { "growth_level", "growth_level" },
        { "growth_pattern", "growth_pattern" },
        { "interference_factors", "interference_factors" },
        { "microbe_type", "microbe_type" }
    };
    
    /// <summary>
    /// 构造函数
    /// </summary>
    /// <param name="modelPath">ONNX模型文件路径</param>
    /// <param name="useGpu">是否使用GPU加速</param>
    /// <param name="deviceId">GPU设备ID (默认0)</param>
    public MultitaskOnnxPredictor(string modelPath, bool useGpu = false, int deviceId = 0)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"模型文件不存在: {modelPath}");
            
        _sessionOptions = new SessionOptions();
        
        // GPU 配置
        if (useGpu)
        {
            try
            {
                _sessionOptions.AppendExecutionProvider_CUDA(deviceId);
                Console.WriteLine($"✅ 启用GPU加速 (设备ID: {deviceId})");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ GPU加速启用失败，回退到CPU: {ex.Message}");
            }
        }
        
        // 性能优化设置
        _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        _sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        
        // 创建推理会话
        _session = new InferenceSession(modelPath, _sessionOptions);
        
        // 获取输入输出信息
        _inputNames = _session.InputMetadata.Keys.ToArray();
        _outputNames = _session.OutputMetadata.Keys.ToArray();
        
        Console.WriteLine($"✅ 模型加载成功: {Path.GetFileName(modelPath)}");
        Console.WriteLine($"   输入: {string.Join(", ", _inputNames)}");
        Console.WriteLine($"   输出: {string.Join(", ", _outputNames)}");
        
        // 验证输入输出
        ValidateModelSignature();
    }
    
    /// <summary>
    /// 验证模型签名
    /// </summary>
    private void ValidateModelSignature()
    {
        // 验证输入
        if (_inputNames.Length != 1 || _inputNames[0] != "input")
        {
            throw new InvalidOperationException($"期望输入名称为 'input'，实际: {string.Join(", ", _inputNames)}");
        }
        
        var inputMeta = _session.InputMetadata["input"];
        var expectedShape = new int[] { 1, 1, 70, 70 }; // [batch, channels, height, width]
        
        // 验证输出
        var expectedOutputs = TaskOutputMapping.Values.ToHashSet();
        var actualOutputs = _outputNames.ToHashSet();
        
        if (!expectedOutputs.IsSubsetOf(actualOutputs))
        {
            var missing = expectedOutputs.Except(actualOutputs);
            Console.WriteLine($"⚠️ 缺少预期输出: {string.Join(", ", missing)}");
        }
        
        Console.WriteLine("✅ 模型签名验证通过");
    }
    
    /// <summary>
    /// 预处理图像
    /// </summary>
    /// <param name="imagePath">图像文件路径</param>
    /// <returns>预处理后的张量</returns>
    public DenseTensor<float> PreprocessImage(string imagePath)
    {
        if (!File.Exists(imagePath))
            throw new FileNotFoundException($"图像文件不存在: {imagePath}");
            
        return PreprocessImageFromStream(File.OpenRead(imagePath));
    }
    
    /// <summary>
    /// 从字节数组预处理图像
    /// </summary>
    public DenseTensor<float> PreprocessImage(byte[] imageBytes)
    {
        return PreprocessImageFromStream(new MemoryStream(imageBytes));
    }
    
    /// <summary>
    /// 从流预处理图像
    /// </summary>
    private DenseTensor<float> PreprocessImageFromStream(Stream imageStream)
    {
        using (imageStream)
        using (var image = Image.Load<L8>(imageStream)) // L8 = 8位灰度
        {
            // 调整大小到 70x70
            image.Mutate(x => x.Resize(70, 70));
            
            // 创建张量 [1, 1, 70, 70]
            var tensor = new DenseTensor<float>(new[] { 1, 1, 70, 70 });
            
            // 转换像素值并归一化到 [0, 1]
            for (int y = 0; y < 70; y++)
            {
                for (int x = 0; x < 70; x++)
                {
                    var pixel = image[x, y];
                    tensor[0, 0, y, x] = pixel.PackedValue / 255.0f;
                }
            }
            
            return tensor;
        }
    }
    
    /// <summary>
    /// 执行预测
    /// </summary>
    /// <param name="imagePath">图像文件路径</param>
    /// <returns>多任务预测结果</returns>
    public MultitaskResult Predict(string imagePath)
    {
        var inputTensor = PreprocessImage(imagePath);
        return Predict(inputTensor);
    }
    
    /// <summary>
    /// 执行预测 (从字节数组)
    /// </summary>
    public MultitaskResult Predict(byte[] imageBytes)
    {
        var inputTensor = PreprocessImage(imageBytes);
        return Predict(inputTensor);
    }
    
    /// <summary>
    /// 执行预测 (核心方法)
    /// </summary>
    private MultitaskResult Predict(DenseTensor<float> inputTensor)
    {
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            // 创建输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };
            
            // 执行推理
            using var results = _session.Run(inputs);
            stopwatch.Stop();
            
            // 解析结果
            var result = new MultitaskResult
            {
                InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds
            };
            
            // 解析各任务结果
            foreach (var output in results)
            {
                switch (output.Name)
                {
                    case "growth_level":
                        result.GrowthLevel = ParseGrowthLevel(output.AsTensor<float>());
                        break;
                    case "growth_pattern":
                        result.GrowthPattern = ParseGrowthPattern(output.AsTensor<float>());
                        break;
                    case "interference_factors":
                        result.InterferenceFactors = ParseInterferenceFactors(output.AsTensor<float>());
                        break;
                    case "microbe_type":
                        result.MicrobeType = ParseMicrobeType(output.AsTensor<float>());
                        break;
                }
            }
            
            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            throw new InvalidOperationException($"推理失败: {ex.Message}", ex);
        }
    }
    
    /// <summary>
    /// 解析生长水平结果
    /// </summary>
    private GrowthLevelResult ParseGrowthLevel(Tensor<float> tensor)
    {
        var probabilities = Softmax(tensor.ToArray());
        var predictedClass = Array.IndexOf(probabilities, probabilities.Max());
        
        return new GrowthLevelResult
        {
            PredictedLevel = (GrowthLevelResult.Level)predictedClass,
            Confidence = probabilities[predictedClass],
            Probabilities = probabilities.Select(p => (double)p).ToArray()
        };
    }
    
    /// <summary>
    /// 解析生长模式结果
    /// </summary>
    private GrowthPatternResult ParseGrowthPattern(Tensor<float> tensor)
    {
        var probabilities = Softmax(tensor.ToArray());
        var predictedClass = Array.IndexOf(probabilities, probabilities.Max());
        
        return new GrowthPatternResult
        {
            PredictedPattern = (GrowthPatternResult.Pattern)predictedClass,
            Confidence = probabilities[predictedClass],
            Probabilities = probabilities.Select(p => (double)p).ToArray()
        };
    }
    
    /// <summary>
    /// 解析干扰因素结果 (多标签)
    /// </summary>
    private InterferenceFactorsResult ParseInterferenceFactors(Tensor<float> tensor)
    {
        var logits = tensor.ToArray();
        var probabilities = logits.Select(Sigmoid).ToArray();
        
        var detectedFactors = new Dictionary<InterferenceFactorsResult.Factor, bool>();
        var confidences = new Dictionary<InterferenceFactorsResult.Factor, double>();
        
        for (int i = 0; i < probabilities.Length; i++)
        {
            var factor = (InterferenceFactorsResult.Factor)i;
            var probability = probabilities[i];
            
            detectedFactors[factor] = probability > 0.5;
            confidences[factor] = probability;
        }
        
        return new InterferenceFactorsResult
        {
            DetectedFactors = detectedFactors,
            Confidences = confidences
        };
    }
    
    /// <summary>
    /// 解析微生物类型结果
    /// </summary>
    private MicrobeTypeResult ParseMicrobeType(Tensor<float> tensor)
    {
        var probabilities = Softmax(tensor.ToArray());
        var predictedClass = Array.IndexOf(probabilities, probabilities.Max());
        
        return new MicrobeTypeResult
        {
            PredictedType = (MicrobeTypeResult.Type)predictedClass,
            Confidence = probabilities[predictedClass],
            Probabilities = probabilities.Select(p => (double)p).ToArray()
        };
    }
    
    /// <summary>
    /// Softmax 激活函数
    /// </summary>
    private float[] Softmax(float[] values)
    {
        var maxVal = values.Max();
        var exp = values.Select(v => (float)Math.Exp(v - maxVal)).ToArray();
        var sum = exp.Sum();
        return exp.Select(e => e / sum).ToArray();
    }
    
    /// <summary>
    /// Sigmoid 激活函数
    /// </summary>
    private double Sigmoid(float value) => 1.0 / (1.0 + Math.Exp(-value));
    
    /// <summary>
    /// 批量预测
    /// </summary>
    /// <param name="imagePaths">图像路径列表</param>
    /// <param name="progressCallback">进度回调</param>
    /// <returns>预测结果列表</returns>
    public List<(string imagePath, MultitaskResult result)> PredictBatch(
        IEnumerable<string> imagePaths, 
        Action<int, int>? progressCallback = null)
    {
        var results = new List<(string, MultitaskResult)>();
        var imageList = imagePaths.ToList();
        
        for (int i = 0; i < imageList.Count; i++)
        {
            try
            {
                var result = Predict(imageList[i]);
                results.Add((imageList[i], result));
                progressCallback?.Invoke(i + 1, imageList.Count);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ 预测失败 {imageList[i]}: {ex.Message}");
                progressCallback?.Invoke(i + 1, imageList.Count);
            }
        }
        
        return results;
    }
    
    /// <summary>
    /// 获取模型信息
    /// </summary>
    public void PrintModelInfo()
    {
        Console.WriteLine("=== 模型信息 ===");
        Console.WriteLine($"输入数量: {_session.InputMetadata.Count}");
        
        foreach (var input in _session.InputMetadata)
        {
            Console.WriteLine($"  - {input.Key}: {string.Join("x", input.Value.Dimensions)}");
        }
        
        Console.WriteLine($"输出数量: {_session.OutputMetadata.Count}");
        foreach (var output in _session.OutputMetadata)
        {
            Console.WriteLine($"  - {output.Key}: {string.Join("x", output.Value.Dimensions)}");
        }
        
        Console.WriteLine($"执行提供程序: {string.Join(", ", _session.GetProviders())}");
    }
    
    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _sessionOptions?.Dispose();
            _disposed = true;
        }
    }
}
```

---

## 💻 完整示例代码

### 1. 基础使用示例

```csharp
using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            // 基础使用示例
            BasicUsageExample();
            
            // 批量处理示例
            BatchProcessingExample();
            
            // 性能测试示例
            PerformanceTestExample();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"程序执行失败: {ex.Message}");
        }
    }
    
    /// <summary>
    /// 基础使用示例
    /// </summary>
    static void BasicUsageExample()
    {
        Console.WriteLine("=== 基础使用示例 ===");
        
        // 1. 验证模型文件
        string modelPath = "models/fixed_mobilenetv3_multitask.onnx";
        if (!MultitaskOnnxPredictor.ValidateModelFile(modelPath))
            return;
        
        // 2. 创建预测器 (启用GPU加速)
        using var predictor = new MultitaskOnnxPredictor(modelPath, useGpu: true);
        
        // 3. 打印模型信息
        predictor.PrintModelInfo();
        
        // 4. 执行单张图像预测
        string imagePath = "test_images/sample_001.png";
        if (File.Exists(imagePath))
        {
            var result = predictor.Predict(imagePath);
            PrintDetailedResult(imagePath, result);
        }
        else
        {
            Console.WriteLine($"测试图像不存在: {imagePath}");
        }
    }
    
    /// <summary>
    /// 批量处理示例
    /// </summary>
    static void BatchProcessingExample()
    {
        Console.WriteLine("\n=== 批量处理示例 ===");
        
        using var predictor = new MultitaskOnnxPredictor(
            "models/fixed_mobilenetv3_multitask.onnx", 
            useGpu: true);
        
        // 获取测试图像列表
        var testImages = Directory.GetFiles("test_images", "*.png")
                                 .Take(10) // 处理前10张图像
                                 .ToList();
        
        if (!testImages.Any())
        {
            Console.WriteLine("未找到测试图像");
            return;
        }
        
        Console.WriteLine($"开始批量处理 {testImages.Count} 张图像...");
        
        // 执行批量预测
        var results = predictor.PredictBatch(testImages, (current, total) =>
        {
            Console.WriteLine($"进度: {current}/{total} ({current * 100.0 / total:F1}%)");
        });
        
        // 统计结果
        GenerateBatchReport(results);
    }
    
    /// <summary>
    /// 性能测试示例
    /// </summary>
    static void PerformanceTestExample()
    {
        Console.WriteLine("\n=== 性能测试示例 ===");
        
        var models = new[]
        {
            ("MobileNetV3", "models/fixed_mobilenetv3_multitask.onnx"),
            ("EfficientNet-B0", "models/fixed_efficientnet_b0_multitask.onnx"),
            ("ResNet-34", "models/resnet34_multitask.onnx")
        };
        
        string testImage = "test_images/sample_001.png";
        if (!File.Exists(testImage))
        {
            Console.WriteLine("性能测试需要测试图像");
            return;
        }
        
        foreach (var (modelName, modelPath) in models)
        {
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"跳过不存在的模型: {modelName}");
                continue;
            }
            
            Console.WriteLine($"\n测试模型: {modelName}");
            
            using var predictor = new MultitaskOnnxPredictor(modelPath, useGpu: true);
            
            // 预热
            Console.WriteLine("预热中...");
            for (int i = 0; i < 5; i++)
            {
                predictor.Predict(testImage);
            }
            
            // 性能测试
            var times = new List<double>();
            const int iterations = 100;
            
            Console.WriteLine($"执行 {iterations} 次推理...");
            for (int i = 0; i < iterations; i++)
            {
                var result = predictor.Predict(testImage);
                times.Add(result.InferenceTimeMs);
            }
            
            // 统计结果
            Console.WriteLine($"平均推理时间: {times.Average():F2} ms");
            Console.WriteLine($"最快推理时间: {times.Min():F2} ms");
            Console.WriteLine($"最慢推理时间: {times.Max():F2} ms");
            Console.WriteLine($"标准差: {CalculateStandardDeviation(times):F2} ms");
        }
    }
    
    /// <summary>
    /// 打印详细结果
    /// </summary>
    static void PrintDetailedResult(string imagePath, MultitaskResult result)
    {
        Console.WriteLine($"\n=== 预测结果: {Path.GetFileName(imagePath)} ===");
        Console.WriteLine($"推理时间: {result.InferenceTimeMs:F2} ms");
        Console.WriteLine($"整体置信度: {result.OverallConfidence:F3}");
        Console.WriteLine();
        
        Console.WriteLine($"🔬 生长水平: {result.GrowthLevel}");
        Console.WriteLine($"🧬 生长模式: {result.GrowthPattern}");
        Console.WriteLine($"⚠️  干扰因素: {result.InterferenceFactors}");
        Console.WriteLine($"🦠 微生物类型: {result.MicrobeType}");
        
        // 显示详细概率
        Console.WriteLine("\n详细概率分布:");
        
        Console.WriteLine("生长模式 Top 3:");
        var topPatterns = result.GrowthPattern.GetTopPredictions(3);
        foreach (var (pattern, prob) in topPatterns)
        {
            Console.WriteLine($"  {pattern}: {prob:F3}");
        }
        
        Console.WriteLine("干扰因素详情:");
        foreach (var (factor, confidence) in result.InterferenceFactors.Confidences)
        {
            var status = result.InterferenceFactors.DetectedFactors[factor] ? "✓" : "✗";
            Console.WriteLine($"  {status} {factor}: {confidence:F3}");
        }
    }
    
    /// <summary>
    /// 生成批量处理报告
    /// </summary>
    static void GenerateBatchReport(List<(string imagePath, MultitaskResult result)> results)
    {
        Console.WriteLine("\n=== 批量处理报告 ===");
        Console.WriteLine($"处理图像数: {results.Count}");
        
        if (!results.Any()) return;
        
        var times = results.Select(r => r.result.InferenceTimeMs).ToList();
        var confidences = results.Select(r => r.result.OverallConfidence).ToList();
        
        Console.WriteLine($"平均推理时间: {times.Average():F2} ms");
        Console.WriteLine($"平均置信度: {confidences.Average():F3}");
        
        // 统计各任务预测分布
        var growthLevels = results.Select(r => r.result.GrowthLevel.PredictedLevel).ToList();
        var growthPatterns = results.Select(r => r.result.GrowthPattern.PredictedPattern).ToList();
        var microbeTypes = results.Select(r => r.result.MicrobeType.PredictedType).ToList();
        
        Console.WriteLine("\n预测分布统计:");
        Console.WriteLine("生长水平:");
        foreach (var group in growthLevels.GroupBy(x => x))
        {
            Console.WriteLine($"  {group.Key}: {group.Count()} ({group.Count() * 100.0 / results.Count:F1}%)");
        }
        
        Console.WriteLine("微生物类型:");
        foreach (var group in microbeTypes.GroupBy(x => x))
        {
            Console.WriteLine($"  {group.Key}: {group.Count()} ({group.Count() * 100.0 / results.Count:F1}%)");
        }
        
        // 检测到干扰因素的图像数
        var imagesWithInterference = results.Count(r => r.result.InterferenceFactors.GetDetectedFactors().Any());
        Console.WriteLine($"\n检测到干扰因素的图像: {imagesWithInterference} ({imagesWithInterference * 100.0 / results.Count:F1}%)");
    }
    
    /// <summary>
    /// 计算标准差
    /// </summary>
    static double CalculateStandardDeviation(IEnumerable<double> values)
    {
        var mean = values.Average();
        var squaredDeviations = values.Select(x => Math.Pow(x - mean, 2));
        return Math.Sqrt(squaredDeviations.Average());
    }
}
```

### 2. Web API 集成示例

```csharp
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Http;
using System.ComponentModel.DataAnnotations;

[ApiController]
[Route("api/[controller]")]
public class MultitaskPredictionController : ControllerBase
{
    private readonly MultitaskOnnxPredictor _predictor;
    private readonly ILogger<MultitaskPredictionController> _logger;
    
    public MultitaskPredictionController(
        MultitaskOnnxPredictor predictor,
        ILogger<MultitaskPredictionController> logger)
    {
        _predictor = predictor;
        _logger = logger;
    }
    
    /// <summary>
    /// 单张图像预测
    /// </summary>
    [HttpPost("predict")]
    public async Task<ActionResult<MultitaskApiResponse>> PredictImage(IFormFile image)
    {
        try
        {
            // 验证输入
            if (image == null || image.Length == 0)
                return BadRequest("请上传有效的图像文件");
            
            // 验证文件类型
            var allowedTypes = new[] { "image/png", "image/jpeg", "image/jpg" };
            if (!allowedTypes.Contains(image.ContentType.ToLower()))
                return BadRequest("仅支持 PNG 和 JPEG 格式");
            
            // 验证文件大小 (限制10MB)
            if (image.Length > 10 * 1024 * 1024)
                return BadRequest("图像文件大小不能超过10MB");
            
            // 读取图像数据
            using var stream = new MemoryStream();
            await image.CopyToAsync(stream);
            var imageBytes = stream.ToArray();
            
            // 执行预测
            var result = _predictor.Predict(imageBytes);
            
            _logger.LogInformation($"预测完成: {image.FileName}, 耗时: {result.InferenceTimeMs:F2}ms");
            
            return Ok(new MultitaskApiResponse
            {
                Success = true,
                Result = result,
                Message = "预测成功"
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "预测过程中发生错误");
            return StatusCode(500, new MultitaskApiResponse
            {
                Success = false,
                Message = "预测失败: " + ex.Message
            });
        }
    }
    
    /// <summary>
    /// 获取模型信息
    /// </summary>
    [HttpGet("model-info")]
    public ActionResult<object> GetModelInfo()
    {
        return Ok(new
        {
            InputSize = "70x70 grayscale",
            Tasks = new[]
            {
                "Growth Level (2 classes)",
                "Growth Pattern (12 classes)", 
                "Interference Factors (4 classes, multi-label)",
                "Microbe Type (4 classes)"
            },
            SupportedFormats = new[] { "PNG", "JPEG" },
            MaxFileSize = "10MB"
        });
    }
}

/// <summary>
/// API 响应模型
/// </summary>
public class MultitaskApiResponse
{
    public bool Success { get; set; }
    public MultitaskResult? Result { get; set; }
    public string Message { get; set; } = string.Empty;
}

/// <summary>
/// 服务注册 (在 Program.cs 或 Startup.cs 中)
/// </summary>
public static class ServiceRegistration
{
    public static IServiceCollection AddMultitaskPredictor(
        this IServiceCollection services, 
        string modelPath, 
        bool useGpu = false)
    {
        services.AddSingleton(provider => 
            new MultitaskOnnxPredictor(modelPath, useGpu));
        
        return services;
    }
}
```

---

## 🚀 性能优化

### 1. 内存管理优化

```csharp
/// <summary>
/// 内存优化的预测器包装类
/// </summary>
public class OptimizedMultitaskPredictor : IDisposable
{
    private readonly MultitaskOnnxPredictor _predictor;
    private readonly object _lock = new object();
    private readonly LRUCache<string, DenseTensor<float>> _tensorCache;
    
    public OptimizedMultitaskPredictor(string modelPath, bool useGpu = false, int cacheSize = 100)
    {
        _predictor = new MultitaskOnnxPredictor(modelPath, useGpu);
        _tensorCache = new LRUCache<string, DenseTensor<float>>(cacheSize);
    }
    
    /// <summary>
    /// 带缓存的预测
    /// </summary>
    public MultitaskResult PredictWithCache(string imagePath)
    {
        lock (_lock)
        {
            var cacheKey = $"{imagePath}_{new FileInfo(imagePath).LastWriteTime.Ticks}";
            
            if (!_tensorCache.TryGet(cacheKey, out var cachedTensor))
            {
                cachedTensor = _predictor.PreprocessImage(imagePath);
                _tensorCache.Set(cacheKey, cachedTensor);
            }
            
            return _predictor.Predict(cachedTensor);
        }
    }
    
    public void Dispose()
    {
        _predictor?.Dispose();
        _tensorCache?.Dispose();
    }
}

/// <summary>
/// 简单的 LRU 缓存实现
/// </summary>
public class LRUCache<TKey, TValue> : IDisposable where TValue : IDisposable
{
    private readonly int _capacity;
    private readonly Dictionary<TKey, LinkedListNode<(TKey key, TValue value)>> _map;
    private readonly LinkedList<(TKey key, TValue value)> _list;
    
    public LRUCache(int capacity)
    {
        _capacity = capacity;
        _map = new Dictionary<TKey, LinkedListNode<(TKey, TValue)>>();
        _list = new LinkedList<(TKey, TValue)>();
    }
    
    public bool TryGet(TKey key, out TValue value)
    {
        if (_map.TryGetValue(key, out var node))
        {
            // 移动到前端
            _list.Remove(node);
            _list.AddFirst(node);
            value = node.Value.value;
            return true;
        }
        
        value = default(TValue);
        return false;
    }
    
    public void Set(TKey key, TValue value)
    {
        if (_map.TryGetValue(key, out var existingNode))
        {
            // 更新现有项
            existingNode.Value = (key, value);
            _list.Remove(existingNode);
            _list.AddFirst(existingNode);
        }
        else
        {
            // 添加新项
            if (_list.Count >= _capacity)
            {
                // 移除最旧的项
                var lastNode = _list.Last;
                _list.RemoveLast();
                _map.Remove(lastNode.Value.key);
                lastNode.Value.value?.Dispose();
            }
            
            var newNode = new LinkedListNode<(TKey, TValue)>((key, value));
            _list.AddFirst(newNode);
            _map[key] = newNode;
        }
    }
    
    public void Dispose()
    {
        foreach (var item in _list)
        {
            item.value?.Dispose();
        }
        _list.Clear();
        _map.Clear();
    }
}
```

### 2. 并行处理优化

```csharp
/// <summary>
/// 并行预测器
/// </summary>
public class ParallelMultitaskPredictor : IDisposable
{
    private readonly MultitaskOnnxPredictor[] _predictors;
    private readonly SemaphoreSlim _semaphore;
    private int _currentIndex = 0;
    
    public ParallelMultitaskPredictor(string modelPath, int instances = Environment.ProcessorCount, bool useGpu = false)
    {
        _predictors = new MultitaskOnnxPredictor[instances];
        _semaphore = new SemaphoreSlim(instances, instances);
        
        for (int i = 0; i < instances; i++)
        {
            _predictors[i] = new MultitaskOnnxPredictor(modelPath, useGpu, i % 2); // 轮换GPU设备
        }
    }
    
    /// <summary>
    /// 并行批量预测
    /// </summary>
    public async Task<List<(string imagePath, MultitaskResult result)>> PredictBatchParallel(
        IEnumerable<string> imagePaths,
        int maxDegreeOfParallelism = -1)
    {
        var options = new ParallelOptions();
        if (maxDegreeOfParallelism > 0)
            options.MaxDegreeOfParallelism = maxDegreeOfParallelism;
        
        var imageList = imagePaths.ToList();
        var results = new ConcurrentBag<(string, MultitaskResult)>();
        
        await Parallel.ForEachAsync(imageList, options, async (imagePath, ct) =>
        {
            await _semaphore.WaitAsync(ct);
            try
            {
                var predictorIndex = Interlocked.Increment(ref _currentIndex) % _predictors.Length;
                var predictor = _predictors[predictorIndex];
                
                var result = predictor.Predict(imagePath);
                results.Add((imagePath, result));
            }
            finally
            {
                _semaphore.Release();
            }
        });
        
        return results.ToList();
    }
    
    public void Dispose()
    {
        foreach (var predictor in _predictors)
        {
            predictor?.Dispose();
        }
        _semaphore?.Dispose();
    }
}
```

---

## ⚠️ 错误处理

### 1. 自定义异常类

```csharp
/// <summary>
/// 多任务预测异常
/// </summary>
public class MultitaskPredictionException : Exception
{
    public string? ModelPath { get; }
    public string? ImagePath { get; }
    public TimeSpan? InferenceTime { get; }
    
    public MultitaskPredictionException(string message) : base(message) { }
    
    public MultitaskPredictionException(string message, Exception innerException) 
        : base(message, innerException) { }
    
    public MultitaskPredictionException(string message, string modelPath, string imagePath)
        : base(message)
    {
        ModelPath = modelPath;
        ImagePath = imagePath;
    }
}

/// <summary>
/// 模型加载异常
/// </summary>
public class ModelLoadException : Exception
{
    public string ModelPath { get; }
    
    public ModelLoadException(string modelPath, string message) : base(message)
    {
        ModelPath = modelPath;
    }
    
    public ModelLoadException(string modelPath, string message, Exception innerException) 
        : base(message, innerException)
    {
        ModelPath = modelPath;
    }
}
```

### 2. 健壮的预测器

```csharp
/// <summary>
/// 带错误处理的预测器
/// </summary>
public class RobustMultitaskPredictor : IDisposable
{
    private readonly MultitaskOnnxPredictor _predictor;
    private readonly ILogger? _logger;
    private readonly RetryPolicy _retryPolicy;
    
    public RobustMultitaskPredictor(
        string modelPath, 
        bool useGpu = false, 
        ILogger? logger = null,
        RetryPolicy? retryPolicy = null)
    {
        try
        {
            _predictor = new MultitaskOnnxPredictor(modelPath, useGpu);
            _logger = logger;
            _retryPolicy = retryPolicy ?? new RetryPolicy(maxRetries: 3, delayMs: 100);
        }
        catch (Exception ex)
        {
            throw new ModelLoadException(modelPath, $"模型加载失败: {ex.Message}", ex);
        }
    }
    
    /// <summary>
    /// 安全预测
    /// </summary>
    public async Task<MultitaskResult?> TryPredictAsync(string imagePath)
    {
        return await _retryPolicy.ExecuteAsync(async () =>
        {
            try
            {
                // 验证输入
                if (string.IsNullOrWhiteSpace(imagePath))
                    throw new ArgumentException("图像路径不能为空");
                
                if (!File.Exists(imagePath))
                    throw new FileNotFoundException($"图像文件不存在: {imagePath}");
                
                // 验证文件格式
                var extension = Path.GetExtension(imagePath).ToLowerInvariant();
                var supportedExtensions = new[] { ".png", ".jpg", ".jpeg" };
                if (!supportedExtensions.Contains(extension))
                    throw new NotSupportedException($"不支持的文件格式: {extension}");
                
                // 验证文件大小
                var fileInfo = new FileInfo(imagePath);
                if (fileInfo.Length == 0)
                    throw new InvalidDataException("图像文件为空");
                
                if (fileInfo.Length > 50 * 1024 * 1024) // 50MB 限制
                    throw new InvalidDataException("图像文件过大 (>50MB)");
                
                // 执行预测
                var result = _predictor.Predict(imagePath);
                
                // 验证结果
                ValidateResult(result);
                
                _logger?.LogInformation($"预测成功: {Path.GetFileName(imagePath)}, 耗时: {result.InferenceTimeMs:F2}ms");
                
                return result;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, $"预测失败: {imagePath}");
                throw new MultitaskPredictionException($"预测失败: {ex.Message}", modelPath: "", imagePath);
            }
        });
    }
    
    /// <summary>
    /// 验证预测结果
    /// </summary>
    private void ValidateResult(MultitaskResult result)
    {
        if (result == null)
            throw new InvalidOperationException("预测结果为空");
        
        if (result.InferenceTimeMs < 0 || result.InferenceTimeMs > 60000) // 超过1分钟
            throw new InvalidOperationException($"推理时间异常: {result.InferenceTimeMs}ms");
        
        if (result.OverallConfidence < 0 || result.OverallConfidence > 1)
            throw new InvalidOperationException($"置信度超出范围: {result.OverallConfidence}");
    }
    
    public void Dispose()
    {
        _predictor?.Dispose();
    }
}

/// <summary>
/// 重试策略
/// </summary>
public class RetryPolicy
{
    public int MaxRetries { get; }
    public int DelayMs { get; }
    
    public RetryPolicy(int maxRetries = 3, int delayMs = 100)
    {
        MaxRetries = maxRetries;
        DelayMs = delayMs;
    }
    
    public async Task<T> ExecuteAsync<T>(Func<Task<T>> operation)
    {
        var lastException = default(Exception);
        
        for (int attempt = 0; attempt <= MaxRetries; attempt++)
        {
            try
            {
                return await operation();
            }
            catch (Exception ex)
            {
                lastException = ex;
                
                if (attempt == MaxRetries)
                    break;
                
                await Task.Delay(DelayMs * (int)Math.Pow(2, attempt)); // 指数退避
            }
        }
        
        throw new InvalidOperationException($"操作在 {MaxRetries + 1} 次尝试后失败", lastException);
    }
}
```

---

## 🚀 部署建议

### 1. Docker 容器化

**Dockerfile:**

```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

# 安装 ONNX Runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["MultitaskPrediction.csproj", "."]
RUN dotnet restore "./MultitaskPrediction.csproj"
COPY . .
WORKDIR "/src/."
RUN dotnet build "MultitaskPrediction.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "MultitaskPrediction.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .

# 复制模型文件
COPY models/ ./models/

ENTRYPOINT ["dotnet", "MultitaskPrediction.dll"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  multitask-prediction:
    build: .
    ports:
      - "8080:80"
    environment:
      - ASPNETCORE_ENVIRONMENT=Production
      - MODEL_PATH=/app/models/fixed_mobilenetv3_multitask.onnx
      - USE_GPU=false
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 2. 云部署配置

**Azure Container Instances:**

```bash
# 创建资源组
az group create --name multitask-rg --location eastus

# 部署容器
az container create \
  --resource-group multitask-rg \
  --name multitask-prediction \
  --image your-registry/multitask-prediction:latest \
  --cpu 4 \
  --memory 8 \
  --ports 80 \
  --environment-variables MODEL_PATH=/app/models/fixed_mobilenetv3_multitask.onnx USE_GPU=false
```

### 3. 生产环境配置

**appsettings.Production.json:**

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "MultitaskPredictor": {
    "ModelPath": "models/fixed_mobilenetv3_multitask.onnx",
    "UseGpu": false,
    "MaxConcurrentRequests": 10,
    "EnableCaching": true,
    "CacheSize": 100,
    "RequestTimeoutMs": 30000
  },
  "RateLimiting": {
    "EnableRateLimiting": true,
    "RequestsPerMinute": 100,
    "BurstSize": 20
  }
}
```

---

## 🔧 故障排除

### 1. 常见错误和解决方案

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ModelLoadException` | 模型文件损坏或不兼容 | 重新下载模型文件，验证文件完整性 |
| `OutOfMemoryException` | 内存不足 | 减少批处理大小，启用图像缓存 |
| `CUDA_ERROR` | GPU配置问题 | 检查CUDA版本，降级到CPU模式 |
| `InvalidImageFormatException` | 不支持的图像格式 | 转换为PNG或JPEG格式 |
| `DimensionMismatchException` | 图像尺寸错误 | 确保图像是70×70像素 |

### 2. 调试工具

```csharp
/// <summary>
/// 诊断工具类
/// </summary>
public static class DiagnosticTools
{
    /// <summary>
    /// 系统环境检查
    /// </summary>
    public static void CheckSystemEnvironment()
    {
        Console.WriteLine("=== 系统环境检查 ===");
        
        // .NET 版本
        Console.WriteLine($".NET 版本: {Environment.Version}");
        
        // 操作系统
        Console.WriteLine($"操作系统: {Environment.OSVersion}");
        
        // 可用内存
        var totalMemory = GC.GetTotalMemory(false);
        Console.WriteLine($"当前内存使用: {totalMemory / 1024 / 1024} MB");
        
        // CPU 核心数
        Console.WriteLine($"CPU 核心数: {Environment.ProcessorCount}");
        
        // ONNX Runtime 提供程序
        try
        {
            using var session = new InferenceSession(Array.Empty<byte>());
            Console.WriteLine($"ONNX Runtime 提供程序: {string.Join(", ", session.GetProviders())}");
        }
        catch
        {
            Console.WriteLine("ONNX Runtime 检查失败");
        }
    }
    
    /// <summary>
    /// 模型兼容性检查
    /// </summary>
    public static bool CheckModelCompatibility(string modelPath)
    {
        try
        {
            using var session = new InferenceSession(modelPath);
            
            var inputMeta = session.InputMetadata.First();
            var expectedDimensions = new long[] { 1, 1, 70, 70 };
            
            Console.WriteLine($"模型输入: {inputMeta.Key}");
            Console.WriteLine($"输入维度: [{string.Join(", ", inputMeta.Value.Dimensions)}]");
            Console.WriteLine($"期望维度: [{string.Join(", ", expectedDimensions)}]");
            
            // 检查维度兼容性
            if (inputMeta.Value.Dimensions.Length != expectedDimensions.Length)
            {
                Console.WriteLine("❌ 输入维度数量不匹配");
                return false;
            }
            
            for (int i = 0; i < expectedDimensions.Length; i++)
            {
                if (inputMeta.Value.Dimensions[i] != expectedDimensions[i] && 
                    inputMeta.Value.Dimensions[i] != -1) // -1 表示动态维度
                {
                    Console.WriteLine($"❌ 维度 {i} 不匹配: 期望 {expectedDimensions[i]}, 实际 {inputMeta.Value.Dimensions[i]}");
                    return false;
                }
            }
            
            Console.WriteLine("✅ 模型兼容性检查通过");
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ 模型兼容性检查失败: {ex.Message}");
            return false;
        }
    }
    
    /// <summary>
    /// 性能基准测试
    /// </summary>
    public static void RunPerformanceBenchmark(MultitaskOnnxPredictor predictor, string testImagePath)
    {
        Console.WriteLine("=== 性能基准测试 ===");
        
        if (!File.Exists(testImagePath))
        {
            Console.WriteLine($"测试图像不存在: {testImagePath}");
            return;
        }
        
        // 预热
        Console.WriteLine("预热中...");
        for (int i = 0; i < 10; i++)
        {
            predictor.Predict(testImagePath);
        }
        
        // 基准测试
        var times = new List<double>();
        const int iterations = 50;
        
        Console.WriteLine($"执行 {iterations} 次基准测试...");
        
        for (int i = 0; i < iterations; i++)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = predictor.Predict(testImagePath);
            stopwatch.Stop();
            
            times.Add(stopwatch.Elapsed.TotalMilliseconds);
        }
        
        // 统计结果
        Console.WriteLine($"平均推理时间: {times.Average():F2} ms");
        Console.WriteLine($"中位数推理时间: {times.OrderBy(x => x).Skip(times.Count / 2).First():F2} ms");
        Console.WriteLine($"95百分位推理时间: {times.OrderBy(x => x).Skip((int)(times.Count * 0.95)).First():F2} ms");
        Console.WriteLine($"标准差: {CalculateStandardDeviation(times):F2} ms");
        
        // 内存使用情况
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        
        var memoryAfter = GC.GetTotalMemory(false);
        Console.WriteLine($"测试后内存使用: {memoryAfter / 1024 / 1024} MB");
    }
    
    private static double CalculateStandardDeviation(IEnumerable<double> values)
    {
        var mean = values.Average();
        var squaredDeviations = values.Select(x => Math.Pow(x - mean, 2));
        return Math.Sqrt(squaredDeviations.Average());
    }
}
```

### 3. 日志配置

**NLog.config:**

```xml
<?xml version="1.0" encoding="utf-8" ?>
<nlog xmlns="http://www.nlog-project.org/schemas/NLog.xsd"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  
  <targets>
    <target xsi:type="File" name="allfile" 
            fileName="logs/multitask-${shortdate}.log"
            layout="${longdate} ${uppercase:${level}} ${logger} ${message} ${exception:format=tostring}" />
            
    <target xsi:type="File" name="errorfile" 
            fileName="logs/errors-${shortdate}.log"
            layout="${longdate} ${uppercase:${level}} ${logger} ${message} ${exception:format=tostring}" />
            
    <target xsi:type="Console" name="console"
            layout="${time} [${uppercase:${level}}] ${message}" />
  </targets>

  <rules>
    <logger name="*" minlevel="Trace" writeTo="allfile" />
    <logger name="*" minlevel="Error" writeTo="errorfile" />
    <logger name="*" minlevel="Info" writeTo="console" />
  </rules>
</nlog>
```

---

## 📚 附录

### A. 模型输出类别映射

**生长模式 (Growth Pattern) 映射:**

```csharp
public static readonly Dictionary<int, string> GrowthPatternMapping = new()
{
    { 0, "clean" },           // 清洁
    { 1, "clustered" },       // 聚集
    { 2, "weak_scattered" },  // 弱散布
    { 3, "heavy_growth" },    // 重度生长
    { 4, "focal" },           // 局灶性
    { 5, "diffuse" },         // 弥漫性
    { 6, "patchy" },          // 斑片状
    { 7, "confluent" },       // 融合性
    { 8, "discrete" },        // 离散性
    { 9, "uniform" },         // 均匀性
    { 10, "sparse" },         // 稀疏
    { 11, "dense" }           // 密集
};
```

### B. 版本兼容性

| 组件 | 最低版本 | 推荐版本 | 备注 |
|------|----------|----------|------|
| .NET | 6.0 | 8.0 | LTS版本 |
| ONNX Runtime | 1.14.0 | 1.16.3 | 最新稳定版 |
| ImageSharp | 2.1.0 | 3.0.2 | 图像处理 |
| System.Numerics.Tensors | 0.1.0 | 0.1.0 | 张量操作 |

### C. 联系信息

- **项目维护:** bioastModel Team
- **技术支持:** support@bioastmodel.com
- **文档更新:** 2025-09-19

---

*本文档将随着模型更新和功能改进持续维护更新。如有问题或建议，请通过 GitHub Issues 反馈。*