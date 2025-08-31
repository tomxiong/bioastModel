# EfficientNetV2-S ONNX模型在C#项目中的使用指南

## 概述
本指南详细介绍如何在C#项目中使用EfficientNetV2-S ONNX模型进行生物抗菌素敏感性测试图像分类。

## 环境要求

### NuGet包依赖
```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.16.3" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.0.2" />
<PackageReference Include="System.Numerics.Tensors" Version="8.0.0" />
```

### 安装命令
```bash
Install-Package Microsoft.ML.OnnxRuntime
Install-Package SixLabors.ImageSharp
Install-Package System.Numerics.Tensors
```

## 模型信息
- **模型文件**: `efficientnet_v2_s.onnx`
- **输入尺寸**: 1×3×70×70 (批次×通道×高×宽)
- **输出尺寸**: 1×2 (二分类概率)
- **类别**: 0=Benign(阴性), 1=Malignant(阳性)
- **准确率**: 98.86%

## 完整C#实现代码

### 1. 模型推理类

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;

public class BioASTClassifier : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;
    
    // ImageNet标准化参数
    private readonly float[] _mean = { 0.485f, 0.456f, 0.406f };
    private readonly float[] _std = { 0.229f, 0.224f, 0.225f };
    
    public BioASTClassifier(string modelPath)
    {
        // 创建ONNX Runtime会话
        _session = new InferenceSession(modelPath);
        
        // 获取输入输出名称
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
        
        Console.WriteLine($"模型加载成功");
        Console.WriteLine($"输入名称: {_inputName}");
        Console.WriteLine($"输出名称: {_outputName}");
    }
    
    /// <summary>
    /// 预测单张图像
    /// </summary>
    /// <param name="imagePath">图像文件路径</param>
    /// <returns>预测结果</returns>
    public PredictionResult Predict(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        return Predict(image);
    }
    
    /// <summary>
    /// 预测图像对象
    /// </summary>
    /// <param name="image">图像对象</param>
    /// <returns>预测结果</returns>
    public PredictionResult Predict(Image<Rgb24> image)
    {
        // 1. 图像预处理
        var tensor = PreprocessImage(image);
        
        // 2. 创建输入
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, tensor)
        };
        
        // 3. 运行推理
        using var results = _session.Run(inputs);
        
        // 4. 处理输出
        var output = results.First().AsEnumerable<float>().ToArray();
        
        // 5. 应用Softmax获取概率
        var probabilities = Softmax(output);
        
        // 6. 获取预测类别
        var predictedClass = probabilities[0] > probabilities[1] ? 0 : 1;
        var confidence = Math.Max(probabilities[0], probabilities[1]);
        
        return new PredictionResult
        {
            PredictedClass = predictedClass,
            ClassName = predictedClass == 0 ? "Benign" : "Malignant",
            Confidence = confidence,
            BenignProbability = probabilities[0],
            MalignantProbability = probabilities[1]
        };
    }
    
    /// <summary>
    /// 图像预处理 - 关键步骤！
    /// </summary>
    /// <param name="image">输入图像</param>
    /// <returns>预处理后的张量</returns>
    private DenseTensor<float> PreprocessImage(Image<Rgb24> image)
    {
        // 1. 调整图像尺寸到70x70
        image.Mutate(x => x.Resize(70, 70));
        
        // 2. 创建张量 [1, 3, 70, 70]
        var tensor = new DenseTensor<float>(new[] { 1, 3, 70, 70 });
        
        // 3. 转换像素值并标准化
        for (int y = 0; y < 70; y++)
        {
            for (int x = 0; x < 70; x++)
            {
                var pixel = image[x, y];
                
                // 转换到[0,1]范围
                float r = pixel.R / 255.0f;
                float g = pixel.G / 255.0f;
                float b = pixel.B / 255.0f;
                
                // ImageNet标准化
                tensor[0, 0, y, x] = (r - _mean[0]) / _std[0]; // Red channel
                tensor[0, 1, y, x] = (g - _mean[1]) / _std[1]; // Green channel
                tensor[0, 2, y, x] = (b - _mean[2]) / _std[2]; // Blue channel
            }
        }
        
        return tensor;
    }
    
    /// <summary>
    /// Softmax函数
    /// </summary>
    /// <param name="values">输入值</param>
    /// <returns>概率分布</returns>
    private float[] Softmax(float[] values)
    {
        var max = values.Max();
        var exp = values.Select(v => Math.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(e => (float)(e / sum)).ToArray();
    }
    
    public void Dispose()
    {
        _session?.Dispose();
    }
}

/// <summary>
/// 预测结果类
/// </summary>
public class PredictionResult
{
    public int PredictedClass { get; set; }
    public string ClassName { get; set; }
    public float Confidence { get; set; }
    public float BenignProbability { get; set; }
    public float MalignantProbability { get; set; }
    
    public override string ToString()
    {
        return $"预测: {ClassName} (置信度: {Confidence:P2}, Benign: {BenignProbability:P2}, Malignant: {MalignantProbability:P2})";
    }
}
```

### 2. 使用示例

```csharp
class Program
{
    static void Main(string[] args)
    {
        // 模型文件路径
        string modelPath = "path/to/efficientnet_v2_s.onnx";
        
        // 创建分类器
        using var classifier = new BioASTClassifier(modelPath);
        
        // 单张图像预测
        string imagePath = "path/to/test_image.png";
        var result = classifier.Predict(imagePath);
        
        Console.WriteLine($"预测结果: {result}");
        
        // 批量预测示例
        string[] imageFiles = Directory.GetFiles("path/to/images", "*.png");
        
        foreach (var imageFile in imageFiles)
        {
            var prediction = classifier.Predict(imageFile);
            Console.WriteLine($"{Path.GetFileName(imageFile)}: {prediction.ClassName} ({prediction.Confidence:P2})");
        }
    }
}
```

### 3. 异步批量处理

```csharp
public async Task<List<PredictionResult>> PredictBatchAsync(string[] imagePaths)
{
    var tasks = imagePaths.Select(async imagePath =>
    {
        return await Task.Run(() => 
        {
            var result = Predict(imagePath);
            return new { ImagePath = imagePath, Result = result };
        });
    });
    
    var results = await Task.WhenAll(tasks);
    return results.Select(r => r.Result).ToList();
}
```

## 图像预处理详解

### 必须的预处理步骤

1. **尺寸调整**: 图像必须调整为70×70像素
2. **像素值归一化**: 将[0,255]范围转换为[0,1]
3. **ImageNet标准化**: 使用均值[0.485, 0.456, 0.406]和标准差[0.229, 0.224, 0.225]
4. **通道顺序**: RGB格式，通道维度在第二位[batch, channels, height, width]

### 预处理公式
```
normalized_pixel = (pixel_value / 255.0 - mean) / std
```

### 支持的图像格式
- PNG (推荐)
- JPEG
- BMP
- TIFF

## 性能优化建议

### 1. 模型加载优化
```csharp
// 使用GPU加速（如果可用）
var options = new SessionOptions();
options.AppendExecutionProvider_CUDA(0); // GPU设备ID
var session = new InferenceSession(modelPath, options);
```

### 2. 内存管理
```csharp
// 重用张量对象
private readonly DenseTensor<float> _reusableTensor = new DenseTensor<float>(new[] { 1, 3, 70, 70 });
```

### 3. 批量处理
```csharp
// 批量推理可以提高吞吐量
public PredictionResult[] PredictBatch(Image<Rgb24>[] images)
{
    var batchSize = images.Length;
    var tensor = new DenseTensor<float>(new[] { batchSize, 3, 70, 70 });
    
    // 预处理所有图像到一个批次张量中
    for (int i = 0; i < batchSize; i++)
    {
        PreprocessImageToBatch(images[i], tensor, i);
    }
    
    // 批量推理
    // ...
}
```

## 错误处理

```csharp
try
{
    var result = classifier.Predict(imagePath);
    Console.WriteLine($"预测成功: {result}");
}
catch (FileNotFoundException)
{
    Console.WriteLine("图像文件不存在");
}
catch (OnnxRuntimeException ex)
{
    Console.WriteLine($"ONNX推理错误: {ex.Message}");
}
catch (Exception ex)
{
    Console.WriteLine($"未知错误: {ex.Message}");
}
```

## 部署注意事项

1. **模型文件**: 确保`efficientnet_v2_s.onnx`文件包含在部署包中
2. **依赖项**: 确保目标机器安装了Visual C++ Redistributable
3. **权限**: 确保应用程序有读取模型文件和图像文件的权限
4. **内存**: 模型大小约80MB，确保有足够内存

## 常见问题

### Q: 预测结果不准确？
A: 检查图像预处理步骤，特别是标准化参数是否正确。

### Q: 推理速度慢？
A: 考虑使用GPU加速或批量处理。

### Q: 内存占用高？
A: 重用张量对象，及时释放图像资源。

### Q: 支持其他图像尺寸？
A: 模型训练时使用70×70，其他尺寸需要重新训练。

## 总结

EfficientNetV2-S ONNX模型在C#中的使用关键点：
1. 正确的图像预处理（70×70尺寸 + ImageNet标准化）
2. 合适的NuGet包依赖
3. 适当的错误处理和资源管理
4. 根据需求选择合适的推理方式（单张/批量/异步）

模型已在验证集上达到98.86%的准确率，适合生产环境使用。