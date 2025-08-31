# M16多任务MobileNetV3 .NET调用指南

## 概述

本文档详细介绍如何在.NET环境中调用M16多任务MobileNetV3 ONNX模型，包括图像预处理、模型推理和结果解析。

## 环境要求

### .NET版本
- .NET 6.0 或更高版本
- .NET Standard 2.0+

### NuGet包
```bash
dotnet add package Microsoft.ML.OnnxRuntime
dotnet add package System.Drawing.Common
dotnet add package SixLabors.ImageSharp
```

## 项目设置

### 1. 创建.NET项目

```bash
# 创建控制台应用
dotnet new console -n M16MultitaskDemo
cd M16MultitaskDemo

# 添加NuGet包
dotnet add package Microsoft.ML.OnnxRuntime
dotnet add package SixLabors.ImageSharp
```

### 2. 项目文件 (M16MultitaskDemo.csproj)

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.18.0" />
    <PackageReference Include="SixLabors.ImageSharp" Version="3.1.2" />
  </ItemGroup>

</Project>
```

## 完整实现代码

### 1. M16多任务推理类 (M16MultitaskInference.cs)

```csharp
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace M16MultitaskDemo
{
    public class M16MultitaskInference : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string[] _growthLevelClasses;
        private readonly string[] _growthPatternClasses;
        private readonly string[] _interferenceClasses;
        
        // 预处理参数
        private readonly float[] _mean = new float[] { 0.485f, 0.456, 0.406f };
        private readonly float[] _std = new float[] { 0.229f, 0.224, 0.225f };
        private readonly int _imageSize = 70;

        public M16MultitaskInference(string modelPath)
        {
            // 加载ONNX模型
            var sessionOptions = new SessionOptions();
            _session = new InferenceSession(modelPath, sessionOptions);

            // 类别定义
            _growthLevelClasses = new[] { "negative", "positive", "weak_growth" };
            _growthPatternClasses = new[] 
            { 
                "clean", "clustered", "scattered", "heavy_growth", "small_dots", 
                "irregular_areas", "light_gray", "default_positive", "default_weak_growth" 
            };
            _interferenceClasses = new[] { "pores", "debris", "artifacts" };
        }

        /// <summary>
        /// 预处理图像
        /// </summary>
        /// <param name="imagePath">图像路径</param>
        /// <returns>预处理后的张量</returns>
        public DenseTensor<float> PreprocessImage(string imagePath)
        {
            using var image = Image.Load<Rgb24>(imagePath);
            
            // 调整大小
            image.Mutate(x => x.Resize(_imageSize, _imageSize));
            
            // 转换为浮点数数组并归一化
            var tensor = new DenseTensor<float>(new[] { 1, 3, _imageSize, _imageSize });
            
            for (int y = 0; y < _imageSize; y++)
            {
                for (int x = 0; x < _imageSize; x++)
                {
                    var pixel = image[x, y];
                    
                    // 归一化: (pixel/255 - mean) / std
                    tensor[0, 0, y, x] = ((pixel.R / 255.0f) - _mean[0]) / _std[0];
                    tensor[0, 1, y, x] = ((pixel.G / 255.0f) - _mean[1]) / _std[1];
                    tensor[0, 2, y, x] = ((pixel.B / 255.0f) - _mean[2]) / _std[2];
                }
            }
            
            return tensor;
        }

        /// <summary>
        /// 执行推理
        /// </summary>
        /// <param name="inputTensor">输入张量</param>
        /// <returns>推理结果</returns>
        public M16PredictionResult Predict(DenseTensor<float> inputTensor)
        {
            // 准备输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // 执行推理
            using var results = _session.Run(inputs);

            // 解析输出
            var growthLevelOutput = results.First(x => x.Name == "growth_level").AsTensor<float>();
            var growthPatternOutput = results.First(x => x.Name == "growth_pattern").AsTensor<float>();
            var interferenceOutput = results.First(x => x.Name == "interference_factors").AsTensor<float>();
            var fineGrainedOutput = results.First(x => x.Name == "fine_grained").AsTensor<float>();

            return new M16PredictionResult
            {
                GrowthLevel = ProcessClassification(growthLevelOutput, _growthLevelClasses),
                GrowthPattern = ProcessClassification(growthPatternOutput, _growthPatternClasses),
                InterferenceFactors = ProcessMultilabel(interferenceOutput, _interferenceClasses),
                FineGrained = ProcessClassification(fineGrainedOutput, Enumerable.Range(0, 40).Select(i => $"class_{i}").ToArray())
            };
        }

        /// <summary>
        /// 处理分类输出
        /// </summary>
        private ClassificationResult ProcessClassification(DenseTensor<float> logits, string[] classes)
        {
            var probs = Softmax(logits);
            var maxProb = probs.Max();
            var predictedClass = Array.IndexOf(probs, maxProb);

            return new ClassificationResult
            {
                ClassId = predictedClass,
                ClassName = predictedClass < classes.Length ? classes[predictedClass] : $"class_{predictedClass}",
                Confidence = maxProb,
                Probabilities = probs
            };
        }

        /// <summary>
        /// 处理多标签输出
        /// </summary>
        private MultilabelResult ProcessMultilabel(DenseTensor<float> logits, string[] classes)
        {
            var probs = Sigmoid(logits);
            var activeClasses = new List<ActiveClass>();

            for (int i = 0; i < probs.Length; i++)
            {
                if (probs[i] > 0.5f)
                {
                    activeClasses.Add(new ActiveClass
                    {
                        ClassId = i,
                        ClassName = i < classes.Length ? classes[i] : $"class_{i}",
                        Confidence = probs[i]
                    });
                }
            }

            return new MultilabelResult
            {
                ActiveClasses = activeClasses,
                Probabilities = probs
            };
        }

        /// <summary>
        /// Softmax函数
        /// </summary>
        private float[] Softmax(DenseTensor<float> logits)
        {
            var max = logits.Max();
            var exp = logits.Select(x => MathF.Exp(x - max)).ToArray();
            var sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        /// <summary>
        /// Sigmoid函数
        /// </summary>
        private float[] Sigmoid(DenseTensor<float> logits)
        {
            return logits.Select(x => 1.0f / (1.0f + MathF.Exp(-x))).ToArray();
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    /// <summary>
    /// 预测结果
    /// </summary>
    public class M16PredictionResult
    {
        public ClassificationResult GrowthLevel { get; set; } = null!;
        public ClassificationResult GrowthPattern { get; set; } = null!;
        public MultilabelResult InterferenceFactors { get; set; } = null!;
        public ClassificationResult FineGrained { get; set; } = null!;
    }

    /// <summary>
    /// 分类结果
    /// </summary>
    public class ClassificationResult
    {
        public int ClassId { get; set; }
        public string ClassName { get; set; } = null!;
        public float Confidence { get; set; }
        public float[] Probabilities { get; set; } = null!;
    }

    /// <summary>
    /// 多标签结果
    /// </summary>
    public class MultilabelResult
    {
        public List<ActiveClass> ActiveClasses { get; set; } = new();
        public float[] Probabilities { get; set; } = null!;
    }

    /// <summary>
    /// 激活类别
    /// </summary>
    public class ActiveClass
    {
        public int ClassId { get; set; }
        public string ClassName { get; set; } = null!;
        public float Confidence { get; set; }
    }
}
```

### 2. 主程序 (Program.cs)

```csharp
using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace M16MultitaskDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("M16多任务MobileNetV3 .NET演示程序");
            Console.WriteLine("=====================================");

            // 检查模型文件
            string modelPath = "onnx_models/m16_multitask_mobilenetv3.onnx";
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"错误: 模型文件不存在: {modelPath}");
                Console.WriteLine("请确保模型文件在正确位置");
                return;
            }

            // 初始化推理器
            using var inference = new M16MultitaskInference(modelPath);
            Console.WriteLine("✓ 模型加载成功");

            // 显示模型信息
            Console.WriteLine($"✓ 输入尺寸: 3x70x70");
            Console.WriteLine($"✓ 输出任务: 4个");
            Console.WriteLine($"✓ 生长级别: 3类");
            Console.WriteLine($"✓ 生长模式: 9类");
            Console.WriteLine($"✓ 干扰因素: 3类");
            Console.WriteLine($"✓ 精细分类: 40类");

            Console.WriteLine("\n=====================================");
            Console.WriteLine("使用方法:");
            Console.WriteLine("1. 输入图像路径进行预测");
            Console.WriteLine("2. 输入 'quit' 退出程序");
            Console.WriteLine("=====================================");

            // 交互式预测
            while (true)
            {
                try
                {
                    Console.Write("\n请输入图像路径: ");
                    string? imagePath = Console.ReadLine();

                    if (string.IsNullOrEmpty(imagePath))
                        continue;

                    if (imagePath.ToLower() == "quit")
                        break;

                    if (!File.Exists(imagePath))
                    {
                        Console.WriteLine($"错误: 文件不存在: {imagePath}");
                        continue;
                    }

                    // 预测
                    Console.WriteLine($"\n正在分析图像: {imagePath}");
                    var result = PredictImage(inference, imagePath);
                    PrintResults(result);

                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\n错误: {ex.Message}");
                }
            }

            Console.WriteLine("\n感谢使用M16多任务分类系统!");
        }

        static M16PredictionResult PredictImage(M16MultitaskInference inference, string imagePath)
        {
            // 预处理
            var inputTensor = inference.PreprocessImage(imagePath);

            // 推理
            return inference.Predict(inputTensor);
        }

        static void PrintResults(M16PredictionResult result)
        {
            Console.WriteLine("\n=== M16多任务分类结果 ===");

            // 生长级别
            Console.WriteLine($"生长级别: {result.GrowthLevel.ClassName} (置信度: {result.GrowthLevel.Confidence:F3})");

            // 生长模式
            Console.WriteLine($"生长模式: {result.GrowthPattern.ClassName} (置信度: {result.GrowthPattern.Confidence:F3})");

            // 干扰因素
            if (result.InterferenceFactors.ActiveClasses.Count > 0)
            {
                var factors = string.Join(", ", result.InterferenceFactors.ActiveClasses.Select(c => c.ClassName));
                Console.WriteLine($"干扰因素: {factors}");
            }
            else
            {
                Console.WriteLine("干扰因素: 无");
            }

            // 精细分类
            Console.WriteLine($"精细分类ID: {result.FineGrained.ClassId} (置信度: {result.FineGrained.Confidence:F3})");

            // 简单解释
            Console.WriteLine("\n=== 简单解释 ===");
            if (result.GrowthLevel.ClassName == "negative")
            {
                Console.WriteLine("✓ 未检测到菌落生长");
            }
            else if (result.GrowthLevel.ClassName == "positive")
            {
                Console.WriteLine($"✓ 检测到菌落生长，形态为{result.GrowthPattern.ClassName}");
            }
            else
            {
                Console.WriteLine($"✓ 检测到微弱生长，形态为{result.GrowthPattern.ClassName}");
            }

            if (result.InterferenceFactors.ActiveClasses.Count > 0)
            {
                var factors = string.Join(", ", result.InterferenceFactors.ActiveClasses.Select(c => c.ClassName));
                Console.WriteLine($"⚠️  图像中存在干扰因素: {factors}");
            }
        }
    }
}
```

## 预处理详解

### 1. 图像加载和调整大小

```csharp
using var image = Image.Load<Rgb24>(imagePath);
image.Mutate(x => x.Resize(70, 70));
```

### 2. 归一化处理

```csharp
// 转换为CHW格式并归一化
tensor[0, 0, y, x] = ((pixel.R / 255.0f) - mean[0]) / std[0];  // Red通道
tensor[0, 1, y, x] = ((pixel.G / 255.0f) - mean[1]) / std[1];  // Green通道
tensor[0, 2, y, x] = ((pixel.B / 255.0f) - mean[2]) / std[2];  // Blue通道
```

### 3. 预处理参数

- **输入尺寸**: 70×70像素
- **颜色格式**: RGB
- **数值范围**: [0, 1] → 标准化
- **通道顺序**: CHW (Channel, Height, Width)
- **均值**: [0.485, 0.456, 0.406]
- **标准差**: [0.229, 0.224, 0.225]

## 批量处理示例

### 1. 批量推理类 (BatchInference.cs)

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace M16MultitaskDemo
{
    public class BatchInference : IDisposable
    {
        private readonly M16MultitaskInference _inference;

        public BatchInference(string modelPath)
        {
            _inference = new M16MultitaskInference(modelPath);
        }

        /// <summary>
        /// 批量预测
        /// </summary>
        public List<BatchResult> PredictBatch(List<string> imagePaths)
        {
            var results = new List<BatchResult>();

            foreach (var imagePath in imagePaths)
            {
                try
                {
                    var inputTensor = _inference.PreprocessImage(imagePath);
                    var prediction = _inference.Predict(inputTensor);

                    results.Add(new BatchResult
                    {
                        ImagePath = imagePath,
                        Prediction = prediction,
                        Success = true,
                        ErrorMessage = null
                    });
                }
                catch (Exception ex)
                {
                    results.Add(new BatchResult
                    {
                        ImagePath = imagePath,
                        Prediction = null,
                        Success = false,
                        ErrorMessage = ex.Message
                    });
                }
            }

            return results;
        }

        /// <summary>
        /// 保存批量结果到JSON
        /// </summary>
        public void SaveResultsToJson(List<BatchResult> results, string outputPath)
        {
            var json = System.Text.Json.JsonSerializer.Serialize(results, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
            });

            File.WriteAllText(outputPath, json);
        }

        public void Dispose()
        {
            _inference?.Dispose();
        }
    }

    public class BatchResult
    {
        public string ImagePath { get; set; } = null!;
        public M16PredictionResult? Prediction { get; set; }
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
    }
}
```

### 2. 批量处理示例

```csharp
// 批量处理示例
var imageFolder = "path/to/images";
var imageFiles = Directory.GetFiles(imageFolder, "*.jpg").ToList();

using var batchInference = new BatchInference("onnx_models/m16_multitask_mobilenetv3.onnx");
var results = batchInference.PredictBatch(imageFiles);

// 保存结果
batchInference.SaveResultsToJson(results, "batch_results.json");

// 显示统计信息
var successCount = results.Count(r => r.Success);
Console.WriteLine($"处理完成: {successCount}/{results.Count} 成功");
```

## ASP.NET Core集成示例

### 1. 控制器 (M16Controller.cs)

```csharp
using Microsoft.AspNetCore.Mvc;
using System.IO;
using System.Threading.Tasks;

namespace M16WebApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class M16Controller : ControllerBase
    {
        private readonly M16MultitaskInference _inference;

        public M16Controller()
        {
            _inference = new M16MultitaskInference("onnx_models/m16_multitask_mobilenetv3.onnx");
        }

        [HttpPost("predict")]
        public async Task<IActionResult> Predict(IFormFile file)
        {
            if (file == null || file.Length == 0)
            {
                return BadRequest("请上传图像文件");
            }

            try
            {
                // 保存临时文件
                var tempPath = Path.GetTempFileName();
                using (var stream = new FileStream(tempPath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }

                // 预测
                var inputTensor = _inference.PreprocessImage(tempPath);
                var result = _inference.Predict(inputTensor);

                // 删除临时文件
                System.IO.File.Delete(tempPath);

                return Ok(result);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"预测失败: {ex.Message}");
            }
        }

        [HttpPost("predict-base64")]
        public IActionResult PredictBase64([FromBody] Base64ImageRequest request)
        {
            try
            {
                // 解码Base64
                var bytes = Convert.FromBase64String(request.ImageData);
                
                // 保存临时文件
                var tempPath = Path.GetTempFileName();
                System.IO.File.WriteAllBytes(tempPath, bytes);

                // 预测
                var inputTensor = _inference.PreprocessImage(tempPath);
                var result = _inference.Predict(inputTensor);

                // 删除临时文件
                System.IO.File.Delete(tempPath);

                return Ok(result);
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"预测失败: {ex.Message}");
            }
        }
    }

    public class Base64ImageRequest
    {
        public string ImageData { get; set; } = null!;
    }
}
```

## 性能优化建议

### 1. 内存管理
```csharp
// 使用using语句确保资源释放
using var inference = new M16MultitaskInference(modelPath);
```

### 2. 批量处理
```csharp
// 批量处理比单张处理更高效
var batchResults = batchInference.PredictBatch(imagePaths);
```

### 3. 并行处理
```csharp
// 使用Parallel.ForEach进行并行处理
Parallel.ForEach(imagePaths, imagePath =>
{
    var result = PredictImage(imagePath);
});
```

### 4. 模型缓存
```csharp
// 在Web应用中缓存模型实例
public class M16Service
{
    private readonly M16MultitaskInference _inference;
    
    public M16Service()
    {
        _inference = new M16MultitaskInference("model.onnx");
    }
    
    public M16PredictionResult Predict(string imagePath)
    {
        return _inference.Predict(_inference.PreprocessImage(imagePath));
    }
}
```

## 错误处理

### 1. 常见错误处理

```csharp
try
{
    var result = inference.Predict(inputTensor);
}
catch (OnnxRuntimeException ex)
{
    Console.WriteLine($"ONNX运行时错误: {ex.Message}");
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"文件未找到: {ex.Message}");
}
catch (Exception ex)
{
    Console.WriteLine($"未知错误: {ex.Message}");
}
```

### 2. 输入验证

```csharp
public bool ValidateImage(string imagePath)
{
    if (!File.Exists(imagePath))
        return false;
    
    try
    {
        using var image = Image.Load<Rgb24>(imagePath);
        return true;
    }
    catch
    {
        return false;
    }
}
```

## 部署建议

### 1. 服务器部署
- 使用.NET 6/8长期支持版本
- 配置足够的内存
- 考虑使用GPU加速的ONNX Runtime

### 2. 容器化部署
```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
COPY . .
ENTRYPOINT ["dotnet", "M16MultitaskDemo.dll"]
```

### 3. 云服务部署
- Azure Functions
- AWS Lambda
- Google Cloud Functions

这个.NET实现提供了完整的M16多任务模型调用方案，包括预处理、推理、结果解析和多种部署方式。