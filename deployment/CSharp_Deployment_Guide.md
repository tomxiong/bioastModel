# C# 项目部署指南

## 📋 概述

本指南详细说明如何在C#项目中部署和使用转换后的ONNX模型，用于MIC（最小抑菌浓度）测试中的气泡检测。

## 🎯 模型性能排行

根据性能验证报告，推荐使用以下模型（按准确率排序）：

| 排名 | 模型名称 | 准确率 | 文件大小 | 推荐用途 |
|------|----------|--------|----------|----------|
| 🏆 1 | simplified_airbubble_detector | 100.00% | ~0.5MB | **生产环境首选** |
| 🥈 2 | efficientnet_b0 | 98.14% | ~6MB | 高精度备选方案 |
| 🥉 3 | resnet18_improved | 97.83% | ~43MB | 传统CNN方案 |
| 4 | coatnet | 91.30% | ~34MB | 混合架构 |
| 5 | convnext_tiny | 89.70% | ~109MB | 现代CNN |

## 🛠️ 环境准备

### 1. 安装必要的NuGet包

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.16.3" />
<PackageReference Include="Microsoft.ML.OnnxRuntime.Managed" Version="1.16.3" />
<PackageReference Include="SixLabors.ImageSharp" Version="3.0.2" />
<PackageReference Include="System.Numerics.Tensors" Version="8.0.0" />
```

### 2. 项目结构

```
YourProject/
├── Models/
│   ├── simplified_airbubble_detector.onnx    # 推荐主模型
│   ├── efficientnet_b0.onnx                  # 备选模型
│   └── model_info.json                       # 模型信息文件
├── Services/
│   ├── AirBubbleDetectionService.cs          # 检测服务
│   └── ImagePreprocessor.cs                  # 图像预处理
└── Models/
    ├── DetectionResult.cs                     # 结果模型
    └── ModelInfo.cs                          # 模型信息模型
```

## 💻 代码实现

### 1. 检测结果模型

```csharp
// Models/DetectionResult.cs
public class DetectionResult
{
    public bool HasAirBubble { get; set; }
    public float Confidence { get; set; }
    public float[] Probabilities { get; set; }
    public string ModelUsed { get; set; }
    public TimeSpan InferenceTime { get; set; }
}

// Models/ModelInfo.cs
public class ModelInfo
{
    public string Name { get; set; }
    public string Description { get; set; }
    public int Priority { get; set; }
    public string OnnxFile { get; set; }
    public double FileSizeMb { get; set; }
    public int[] InputShape { get; set; }
    public int[] OutputShape { get; set; }
}
```

### 2. 图像预处理器

```csharp
// Services/ImagePreprocessor.cs
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public class ImagePreprocessor
{
    private const int TARGET_WIDTH = 70;
    private const int TARGET_HEIGHT = 70;
    
    public float[] PreprocessImage(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        return PreprocessImage(image);
    }
    
    public float[] PreprocessImage(Image<Rgb24> image)
    {
        // 调整图像大小到70x70
        image.Mutate(x => x.Resize(TARGET_WIDTH, TARGET_HEIGHT));
        
        // 转换为张量格式 [C, H, W]
        var tensor = new float[3 * TARGET_HEIGHT * TARGET_WIDTH];
        
        for (int y = 0; y < TARGET_HEIGHT; y++)
        {
            for (int x = 0; x < TARGET_WIDTH; x++)
            {
                var pixel = image[x, y];
                
                // 归一化到 [0, 1] 范围
                tensor[0 * TARGET_HEIGHT * TARGET_WIDTH + y * TARGET_WIDTH + x] = pixel.R / 255.0f;
                tensor[1 * TARGET_HEIGHT * TARGET_WIDTH + y * TARGET_WIDTH + x] = pixel.G / 255.0f;
                tensor[2 * TARGET_HEIGHT * TARGET_WIDTH + y * TARGET_WIDTH + x] = pixel.B / 255.0f;
            }
        }
        
        return tensor;
    }
    
    public float[] PreprocessImageFromBytes(byte[] imageBytes)
    {
        using var image = Image.Load<Rgb24>(imageBytes);
        return PreprocessImage(image);
    }
}
```

### 3. 气泡检测服务

```csharp
// Services/AirBubbleDetectionService.cs
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;

public class AirBubbleDetectionService : IDisposable
{
    private readonly InferenceSession _session;
    private readonly ImagePreprocessor _preprocessor;
    private readonly string _modelName;
    private bool _disposed = false;
    
    public AirBubbleDetectionService(string modelPath, string modelName = "unknown")
    {
        var sessionOptions = new SessionOptions
        {
            EnableCpuMemArena = false,
            EnableMemoryPattern = false,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        };
        
        _session = new InferenceSession(modelPath, sessionOptions);
        _preprocessor = new ImagePreprocessor();
        _modelName = modelName;
    }
    
    public DetectionResult DetectAirBubble(string imagePath)
    {
        var startTime = DateTime.UtcNow;
        
        try
        {
            // 预处理图像
            var inputTensor = _preprocessor.PreprocessImage(imagePath);
            
            // 创建输入张量
            var inputDimensions = new int[] { 1, 3, 70, 70 };
            var tensor = new DenseTensor<float>(inputTensor, inputDimensions);
            
            // 创建输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };
            
            // 运行推理
            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            
            // 应用Softmax
            var probabilities = Softmax(output);
            
            var inferenceTime = DateTime.UtcNow - startTime;
            
            return new DetectionResult
            {
                HasAirBubble = probabilities[1] > 0.5f,
                Confidence = Math.Max(probabilities[0], probabilities[1]),
                Probabilities = probabilities,
                ModelUsed = _modelName,
                InferenceTime = inferenceTime
            };
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"气泡检测失败: {ex.Message}", ex);
        }
    }
    
    public DetectionResult DetectAirBubble(byte[] imageBytes)
    {
        var startTime = DateTime.UtcNow;
        
        try
        {
            var inputTensor = _preprocessor.PreprocessImageFromBytes(imageBytes);
            var inputDimensions = new int[] { 1, 3, 70, 70 };
            var tensor = new DenseTensor<float>(inputTensor, inputDimensions);
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };
            
            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            var probabilities = Softmax(output);
            
            var inferenceTime = DateTime.UtcNow - startTime;
            
            return new DetectionResult
            {
                HasAirBubble = probabilities[1] > 0.5f,
                Confidence = Math.Max(probabilities[0], probabilities[1]),
                Probabilities = probabilities,
                ModelUsed = _modelName,
                InferenceTime = inferenceTime
            };
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"气泡检测失败: {ex.Message}", ex);
        }
    }
    
    private float[] Softmax(float[] values)
    {
        var max = values.Max();
        var exp = values.Select(v => Math.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(e => (float)(e / sum)).ToArray();
    }
    
    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _disposed = true;
        }
    }
}
```

### 4. 模型管理器

```csharp
// Services/ModelManager.cs
public class ModelManager
{
    private readonly Dictionary<string, AirBubbleDetectionService> _models;
    private readonly List<ModelInfo> _modelInfos;
    
    public ModelManager(string modelsDirectory)
    {
        _models = new Dictionary<string, AirBubbleDetectionService>();
        _modelInfos = LoadModelInfos(modelsDirectory);
        
        // 加载所有可用模型
        foreach (var modelInfo in _modelInfos)
        {
            var modelPath = Path.Combine(modelsDirectory, modelInfo.OnnxFile);
            if (File.Exists(modelPath))
            {
                _models[modelInfo.Name] = new AirBubbleDetectionService(modelPath, modelInfo.Name);
            }
        }
    }
    
    public AirBubbleDetectionService GetBestModel()
    {
        // 返回优先级最高的可用模型
        var bestModel = _modelInfos
            .Where(m => _models.ContainsKey(m.Name))
            .OrderBy(m => m.Priority)
            .FirstOrDefault();
            
        return bestModel != null ? _models[bestModel.Name] : null;
    }
    
    public AirBubbleDetectionService GetModel(string modelName)
    {
        return _models.TryGetValue(modelName, out var model) ? model : null;
    }
    
    public List<string> GetAvailableModels()
    {
        return _models.Keys.ToList();
    }
    
    private List<ModelInfo> LoadModelInfos(string modelsDirectory)
    {
        var infoPath = Path.Combine(modelsDirectory, "model_info.json");
        if (!File.Exists(infoPath))
        {
            return new List<ModelInfo>();
        }
        
        var json = File.ReadAllText(infoPath);
        var data = JsonSerializer.Deserialize<JsonElement>(json);
        
        return data.GetProperty("models")
            .EnumerateArray()
            .Select(m => new ModelInfo
            {
                Name = m.GetProperty("name").GetString(),
                Description = m.GetProperty("description").GetString(),
                Priority = m.GetProperty("priority").GetInt32(),
                OnnxFile = m.GetProperty("onnx_file").GetString(),
                FileSizeMb = m.GetProperty("file_size_mb").GetDouble(),
                InputShape = m.GetProperty("input_shape").EnumerateArray().Select(x => x.GetInt32()).ToArray(),
                OutputShape = m.GetProperty("output_shape").EnumerateArray().Select(x => x.GetInt32()).ToArray()
            })
            .ToList();
    }
    
    public void Dispose()
    {
        foreach (var model in _models.Values)
        {
            model.Dispose();
        }
        _models.Clear();
    }
}
```

## 🚀 使用示例

### 1. 基本使用

```csharp
// 初始化模型管理器
var modelManager = new ModelManager("./Models");

// 获取最佳模型（推荐使用简化气泡检测器）
using var detector = modelManager.GetBestModel();

// 检测图像中的气泡
var result = detector.DetectAirBubble("path/to/mic_test_image.jpg");

Console.WriteLine($"检测结果: {(result.HasAirBubble ? "有气泡" : "无气泡")}");
Console.WriteLine($"置信度: {result.Confidence:P2}");
Console.WriteLine($"使用模型: {result.ModelUsed}");
Console.WriteLine($"推理时间: {result.InferenceTime.TotalMilliseconds:F2}ms");
```

### 2. 批量处理

```csharp
public async Task<List<DetectionResult>> ProcessMICPlate(string[] imagePaths)
{
    var modelManager = new ModelManager("./Models");
    using var detector = modelManager.GetBestModel();
    
    var results = new List<DetectionResult>();
    
    foreach (var imagePath in imagePaths)
    {
        try
        {
            var result = detector.DetectAirBubble(imagePath);
            results.Add(result);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"处理图像失败 {imagePath}: {ex.Message}");
        }
    }
    
    return results;
}
```

### 3. Web API 集成

```csharp
[ApiController]
[Route("api/[controller]")]
public class AirBubbleDetectionController : ControllerBase
{
    private readonly ModelManager _modelManager;
    
    public AirBubbleDetectionController(ModelManager modelManager)
    {
        _modelManager = modelManager;
    }
    
    [HttpPost("detect")]
    public async Task<IActionResult> DetectAirBubble(IFormFile image)
    {
        if (image == null || image.Length == 0)
        {
            return BadRequest("请提供有效的图像文件");
        }
        
        try
        {
            using var detector = _modelManager.GetBestModel();
            using var memoryStream = new MemoryStream();
            await image.CopyToAsync(memoryStream);
            
            var result = detector.DetectAirBubble(memoryStream.ToArray());
            
            return Ok(new
            {
                hasAirBubble = result.HasAirBubble,
                confidence = result.Confidence,
                probabilities = result.Probabilities,
                modelUsed = result.ModelUsed,
                inferenceTimeMs = result.InferenceTime.TotalMilliseconds
            });
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"检测失败: {ex.Message}");
        }
    }
    
    [HttpGet("models")]
    public IActionResult GetAvailableModels()
    {
        var models = _modelManager.GetAvailableModels();
        return Ok(models);
    }
}
```

## ⚡ 性能优化建议

### 1. 模型选择策略

- **生产环境**: 优先使用 `simplified_airbubble_detector`（100%准确率，最小文件）
- **高精度需求**: 使用 `efficientnet_b0`（98.14%准确率，平衡性能）
- **资源受限**: 使用 `mic_mobilenetv3`（85.20%准确率，移动优化）

### 2. 性能优化

```csharp
// 使用单例模式避免重复加载模型
services.AddSingleton<ModelManager>(provider => 
    new ModelManager("./Models"));

// 启用模型预热
services.AddHostedService<ModelWarmupService>();
```

### 3. 内存管理

```csharp
// 及时释放资源
using var detector = modelManager.GetBestModel();
var result = detector.DetectAirBubble(imagePath);
// detector 会自动释放

// 批量处理时复用检测器
using var detector = modelManager.GetBestModel();
foreach (var imagePath in imagePaths)
{
    var result = detector.DetectAirBubble(imagePath);
    // 处理结果...
}
```

## 🔧 部署步骤

### 1. 转换模型为ONNX格式

```bash
# 在Python环境中运行转换脚本
cd /path/to/bioastModel
python scripts/convert_models_to_onnx.py
```

转换完成后，会在 `deployment/onnx_models/` 目录下生成：
- `simplified_airbubble_detector.onnx` - 推荐主模型
- `efficientnet_b0.onnx` - 备选模型
- `resnet18_improved.onnx` - 高性能模型
- `model_info.json` - 模型信息文件

### 2. 复制模型文件到C#项目

```
YourCSharpProject/
├── Models/
│   ├── simplified_airbubble_detector.onnx
│   ├── efficientnet_b0.onnx
│   ├── resnet18_improved.onnx
│   └── model_info.json
```

### 3. 配置依赖注入（ASP.NET Core）

```csharp
// Program.cs 或 Startup.cs
public void ConfigureServices(IServiceCollection services)
{
    // 注册模型管理器
    services.AddSingleton<ModelManager>(provider => 
        new ModelManager(Path.Combine(Directory.GetCurrentDirectory(), "Models")));
    
    // 注册图像预处理器
    services.AddScoped<ImagePreprocessor>();
    
    // 配置CORS（如果需要）
    services.AddCors(options =>
    {
        options.AddPolicy("AllowAll", builder =>
        {
            builder.AllowAnyOrigin()
                   .AllowAnyMethod()
                   .AllowAnyHeader();
        });
    });
}
```

## 📊 性能基准测试

### 推荐模型性能对比

| 模型 | 准确率 | 推理时间 | 内存占用 | 文件大小 | 推荐场景 |
|------|--------|----------|----------|----------|----------|
| Simplified AirBubble | 100.00% | ~5ms | ~50MB | ~0.5MB | **生产首选** |
| EfficientNet-B0 | 98.14% | ~15ms | ~200MB | ~6MB | 高精度需求 |
| ResNet18-Improved | 97.83% | ~25ms | ~400MB | ~43MB | 传统方案 |

### 性能测试代码

```csharp
public class PerformanceBenchmark
{
    public async Task<BenchmarkResult> RunBenchmark(string modelPath, string[] testImages)
    {
        var results = new List<double>();
        using var detector = new AirBubbleDetectionService(modelPath);
        
        // 预热
        detector.DetectAirBubble(testImages[0]);
        
        // 性能测试
        foreach (var imagePath in testImages)
        {
            var startTime = DateTime.UtcNow;
            var result = detector.DetectAirBubble(imagePath);
            var endTime = DateTime.UtcNow;
            
            results.Add((endTime - startTime).TotalMilliseconds);
        }
        
        return new BenchmarkResult
        {
            AverageInferenceTime = results.Average(),
            MinInferenceTime = results.Min(),
            MaxInferenceTime = results.Max(),
            TotalImages = testImages.Length
        };
    }
}
```

## 🛡️ 错误处理和日志

### 1. 异常处理

```csharp
public class SafeAirBubbleDetectionService
{
    private readonly AirBubbleDetectionService _detector;
    private readonly ILogger<SafeAirBubbleDetectionService> _logger;
    
    public SafeAirBubbleDetectionService(
        AirBubbleDetectionService detector,
        ILogger<SafeAirBubbleDetectionService> logger)
    {
        _detector = detector;
        _logger = logger;
    }
    
    public async Task<DetectionResult> SafeDetectAirBubble(string imagePath)
    {
        try
        {
            _logger.LogInformation($"开始检测图像: {imagePath}");
            
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"图像文件不存在: {imagePath}");
            }
            
            var result = _detector.DetectAirBubble(imagePath);
            
            _logger.LogInformation($"检测完成: {result.HasAirBubble}, 置信度: {result.Confidence:P2}");
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"气泡检测失败: {imagePath}");
            
            return new DetectionResult
            {
                HasAirBubble = false,
                Confidence = 0.0f,
                Probabilities = new float[] { 1.0f, 0.0f },
                ModelUsed = "error",
                InferenceTime = TimeSpan.Zero
            };
        }
    }
}
```

### 2. 健康检查

```csharp
public class ModelHealthCheck : IHealthCheck
{
    private readonly ModelManager _modelManager;
    
    public ModelHealthCheck(ModelManager modelManager)
    {
        _modelManager = modelManager;
    }
    
    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var bestModel = _modelManager.GetBestModel();
            if (bestModel == null)
            {
                return Task.FromResult(HealthCheckResult.Unhealthy("没有可用的模型"));
            }
            
            // 创建测试图像（70x70的随机图像）
            var testImage = CreateTestImage();
            var result = bestModel.DetectAirBubble(testImage);
            
            if (result.InferenceTime.TotalSeconds > 1.0)
            {
                return Task.FromResult(HealthCheckResult.Degraded("模型推理时间过长"));
            }
            
            return Task.FromResult(HealthCheckResult.Healthy("模型运行正常"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(HealthCheckResult.Unhealthy($"模型健康检查失败: {ex.Message}"));
        }
    }
    
    private byte[] CreateTestImage()
    {
        using var image = new Image<Rgb24>(70, 70);
        image.Mutate(x => x.Fill(Color.White));
        
        using var stream = new MemoryStream();
        image.SaveAsJpeg(stream);
        return stream.ToArray();
    }
}
```

## 🔄 模型更新和版本管理

### 1. 模型版本控制

```csharp
public class ModelVersionManager
{
    private readonly string _modelsDirectory;
    private readonly ILogger<ModelVersionManager> _logger;
    
    public ModelVersionManager(string modelsDirectory, ILogger<ModelVersionManager> logger)
    {
        _modelsDirectory = modelsDirectory;
        _logger = logger;
    }
    
    public async Task<bool> UpdateModel(string modelName, Stream modelStream, string version)
    {
        try
        {
            var backupPath = Path.Combine(_modelsDirectory, $"{modelName}_backup_{DateTime.Now:yyyyMMdd_HHmmss}.onnx");
            var currentPath = Path.Combine(_modelsDirectory, $"{modelName}.onnx");
            var newPath = Path.Combine(_modelsDirectory, $"{modelName}_v{version}.onnx");
            
            // 备份当前模型
            if (File.Exists(currentPath))
            {
                File.Copy(currentPath, backupPath);
                _logger.LogInformation($"已备份模型: {backupPath}");
            }
            
            // 保存新模型
            using var fileStream = File.Create(newPath);
            await modelStream.CopyToAsync(fileStream);
            
            // 验证新模型
            if (await ValidateModel(newPath))
            {
                // 替换当前模型
                if (File.Exists(currentPath))
                {
                    File.Delete(currentPath);
                }
                File.Move(newPath, currentPath);
                
                _logger.LogInformation($"模型更新成功: {modelName} -> v{version}");
                return true;
            }
            else
            {
                File.Delete(newPath);
                _logger.LogError($"新模型验证失败: {modelName} v{version}");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"模型更新失败: {modelName}");
            return false;
        }
    }
    
    private async Task<bool> ValidateModel(string modelPath)
    {
        try
        {
            using var detector = new AirBubbleDetectionService(modelPath);
            var testImage = CreateTestImage();
            var result = detector.DetectAirBubble(testImage);
            return result != null && result.Probabilities?.Length == 2;
        }
        catch
        {
            return false;
        }
    }
}
```

## 📱 移动端部署（Xamarin/MAUI）

### 1. 平台特定配置

```csharp
// Platforms/Android/MainActivity.cs
[Activity(Theme = "@style/Maui.SplashTheme", MainLauncher = true)]
public class MainActivity : MauiAppCompatActivity
{
    protected override void onCreate(Bundle savedInstanceState)
    {
        base.onCreate(savedInstanceState);
        
        // 复制模型文件到应用目录
        CopyModelFiles();
    }
    
    private void CopyModelFiles()
    {
        var modelFiles = new[] { "simplified_airbubble_detector.onnx", "model_info.json" };
        var assetsPath = Path.Combine(FileSystem.Current.CacheDirectory, "Models");
        Directory.CreateDirectory(assetsPath);
        
        foreach (var modelFile in modelFiles)
        {
            using var assetStream = Assets.Open($"Models/{modelFile}");
            var targetPath = Path.Combine(assetsPath, modelFile);
            using var fileStream = File.Create(targetPath);
            assetStream.CopyTo(fileStream);
        }
    }
}
```

### 2. 跨平台服务

```csharp
public interface IPlatformModelService
{
    string GetModelsDirectory();
    Task<bool> IsModelAvailable(string modelName);
}

// Platforms/Android/PlatformModelService.cs
public class PlatformModelService : IPlatformModelService
{
    public string GetModelsDirectory()
    {
        return Path.Combine(FileSystem.Current.CacheDirectory, "Models");
    }
    
    public async Task<bool> IsModelAvailable(string modelName)
    {
        var modelPath = Path.Combine(GetModelsDirectory(), $"{modelName}.onnx");
        return File.Exists(modelPath);
    }
}
```

## 🚀 生产部署清单

### 部署前检查

- [ ] 所有ONNX模型文件已转换并测试
- [ ] model_info.json 文件已生成
- [ ] NuGet包依赖已安装
- [ ] 性能基准测试已完成
- [ ] 错误处理和日志已配置
- [ ] 健康检查已实现
- [ ] 内存管理已优化

### 推荐部署配置

```json
{
  "ModelSettings": {
    "DefaultModel": "simplified_airbubble_detector",
    "FallbackModel": "efficientnet_b0",
    "MaxInferenceTime": "00:00:01",
    "EnableModelCaching": true,
    "EnablePerformanceLogging": true
  },
  "Logging": {
    "LogLevel": {
      "AirBubbleDetection": "Information",
      "ModelManager": "Information"
    }
  }
}
```

## 📞 技术支持

### 常见问题

1. **Q: 模型加载失败**
   - A: 检查ONNX文件路径和权限，确保Microsoft.ML.OnnxRuntime版本兼容

2. **Q: 推理速度慢**
   - A: 使用simplified_airbubble_detector模型，启用模型缓存

3. **Q: 内存占用过高**
   - A: 及时释放DetectionService，避免同时加载多个模型

4. **Q: 准确率不符合预期**
   - A: 检查图像预处理是否正确，确保输入尺寸为70x70

### 联系方式

- 技术文档: `reports/optimized_model_performance_validation_report.html`
- 模型性能: 简化气泡检测器达到100%准确率
- 部署支持: 参考本指南完整实现

---

**部署指南版本**: 1.0  
**最后更新**: 2025-08-03  
**兼容模型**: 所有10个ONNX模型  
**推荐模型**: simplified_airbubble_detector.onnx
