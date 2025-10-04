# C# ONNX 集成指南

MobileNetV4 v0.11.0 ONNX 模型的完整 C# 集成解决方案。

## 📋 目录

1. [概述](#概述)
2. [项目结构](#项目结构)
3. [快速开始](#快速开始)
4. [核心功能](#核心功能)
5. [性能指标](#性能指标)
6. [集成示例](#集成示例)
7. [最佳实践](#最佳实践)

---

## 概述

### 模型信息

- **模型名称**: MobileNetV4 v0.11.0
- **架构**: Universal Inverted Bottleneck + SE/ECA Attention
- **输入尺寸**: 70×70 像素 (单通道灰度图)
- **参数量**: 952,201
- **模型大小**: 3.69 MB (ONNX)
- **OPSET 版本**: 14

### 任务定义

| 任务 | 类型 | 输出 | 激活函数 |
|------|------|------|----------|
| **Growth Level** | 二分类 | 2 类 (negative, positive) | Sigmoid |
| **Growth Pattern** | 多分类 | 10 类 | Softmax |
| **Interference Factors** | 多标签 | 4 类 (pores, artifacts, debris, contamination) | Sigmoid |

### 性能指标 (测试集)

| 指标 | 准确率 |
|------|--------|
| **总体准确率** | **94.26%** |
| Growth Level | 98.53% |
| Growth Pattern | 87.31% |
| Interference Overall | 96.93% |

**推理性能** (CPU):
- 单图像推理: ~1.75 ms
- 吞吐量: ~570 图像/秒 (batch=1)
- 批量推理: ~1.00 ms/图像 (batch=4)

---

## 项目结构

```
deployment/
├── onnx_models/
│   └── mobilenetv4_v0.11.0/
│       ├── model.onnx                    # ONNX 模型文件 (3.69 MB)
│       └── model_info.json               # 模型元数据
│
└── csharp_example/
    ├── BioastOnnxInference/              # C# 控制台项目
    │   ├── BioastOnnxInference.csproj    # 项目配置
    │   ├── Program.cs                     # 主程序 (单图像推理)
    │   └── BatchInferenceExample.cs       # 批量处理示例
    │
    ├── QUICKSTART.md                      # 5分钟快速入门
    ├── README.md                          # 详细项目文档
    └── USAGE_EXAMPLES.md                  # 13个实用示例
```

---

## 快速开始

### 前置要求

```bash
# 检查 .NET 版本
dotnet --version
# 需要: >= 6.0

# 安装 .NET 6.0 SDK (如未安装)
# Windows: https://dotnet.microsoft.com/download/dotnet/6.0
# Linux: sudo apt install dotnet-sdk-6.0
# macOS: brew install dotnet-sdk
```

### 3步快速运行

```bash
# 1. 进入项目目录
cd deployment/csharp_example/BioastOnnxInference

# 2. 恢复 NuGet 包
dotnet restore

# 3. 运行推理
dotnet run /path/to/your/image.png
```

**详细快速入门**: 参阅 [QUICKSTART.md](csharp_example/QUICKSTART.md)

---

## 核心功能

### 1. BioastPredictor 类

主要推理引擎,封装了 ONNX Runtime 和所有后处理逻辑。

#### 初始化

```csharp
using BioastOnnxInference;

var predictor = new BioastPredictor("model.onnx");
```

#### 单图像推理

```csharp
var result = predictor.Predict("colony_image.png");

// 访问结果
Console.WriteLine($"Growth Level: {result.GrowthLevel.Label}");
Console.WriteLine($"Confidence: {result.GrowthLevel.Confidence:P2}");
Console.WriteLine($"Pattern: {result.GrowthPattern.Label}");

// 检查干扰因素
foreach (var factor in result.InterferenceFactors.Where(f => f.IsPresent))
{
    Console.WriteLine($"Detected: {factor.Name} (score: {factor.Score:F4})");
}
```

### 2. 批量处理

```csharp
using BioastOnnxInference;

// 处理整个目录
BatchInferenceExample.RunBatchInference(
    modelPath: "model.onnx",
    imagesDirectory: "images/"
);
```

**输出包括**:
- 每张图像的详细结果
- 处理统计 (总数、成功、错误)
- 分布统计 (Growth Level、Pattern、Interference)
- 性能指标 (平均推理时间、吞吐量)

### 3. CSV 导出

```csharp
CsvExportExample.ExportToCsv(
    modelPath: "model.onnx",
    imagesDirectory: "production_images/",
    outputCsvPath: "results.csv"
);
```

**CSV 格式**:
```csv
filename,growth_level,growth_level_confidence,growth_pattern,growth_pattern_confidence,pores,artifacts,debris,contamination
sample_001.png,positive,0.9854,clustered,0.8923,0.1234,0.0567,0.6789,0.0012
```

---

## 性能指标

### ONNX vs PyTorch 对比

| 指标 | PyTorch | ONNX | 提升 |
|------|---------|------|------|
| **模型大小** | 11.16 MB | 3.69 MB | -66.9% |
| **单图推理** (batch=1) | 7.89 ms | 1.75 ms | **4.50x** ⭐ |
| **批量推理** (batch=4) | 13.85 ms | 3.99 ms | **3.47x** |
| **吞吐量** (batch=1) | 127 img/s | 570 img/s | **4.50x** |
| **精度损失** | - | < 1e-6 | 几乎无损 ✅ |

### 推理时间基准测试

基于 Intel CPU (无 GPU 加速):

| Batch Size | 平均时间 (ms) | 吞吐量 (img/s) |
|------------|---------------|----------------|
| 1 | 1.75 | 570 |
| 4 | 1.00 (per image) | 1000 |
| 16 | 0.63 (per image) | 1587 |
| 32 | 0.58 (per image) | 1724 |

**推荐配置**:
- **实时推理**: batch=1 (1.75 ms, 570 img/s)
- **批量处理**: batch=4-16 (最佳吞吐量)

---

## 集成示例

### 1. ASP.NET Core Web API

```csharp
[ApiController]
[Route("api/colony-detection")]
public class ColonyDetectionController : ControllerBase
{
    private static readonly BioastPredictor _predictor = new("model.onnx");

    [HttpPost("analyze")]
    public IActionResult AnalyzeImage([FromForm] IFormFile image)
    {
        var tempPath = Path.GetTempFileName();

        try
        {
            using (var stream = new FileStream(tempPath, FileMode.Create))
            {
                image.CopyTo(stream);
            }

            var result = _predictor.Predict(tempPath);

            return Ok(new
            {
                growthLevel = result.GrowthLevel.Label,
                confidence = result.GrowthLevel.Confidence,
                pattern = result.GrowthPattern.Label,
                interference = result.InterferenceFactors
                    .Where(f => f.IsPresent)
                    .Select(f => f.Name)
            });
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }
}
```

**API 调用**:
```bash
curl -X POST http://localhost:5000/api/colony-detection/analyze \
  -F "image=@sample.png"
```

**响应**:
```json
{
  "growthLevel": "positive",
  "confidence": 0.9854,
  "pattern": "clustered",
  "interference": ["debris"]
}
```

### 2. WPF 桌面应用

```csharp
public partial class MainWindow : Window
{
    private readonly BioastPredictor _predictor;

    public MainWindow()
    {
        InitializeComponent();
        _predictor = new BioastPredictor("model.onnx");
    }

    private void AnalyzeButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp"
        };

        if (dialog.ShowDialog() == true)
        {
            var result = _predictor.Predict(dialog.FileName);

            ResultTextBlock.Text = $"Growth Level: {result.GrowthLevel.Label}\n" +
                                   $"Confidence: {result.GrowthLevel.Confidence:P2}\n" +
                                   $"Pattern: {result.GrowthPattern.Label}";

            // 显示图像
            ImageControl.Source = new BitmapImage(new Uri(dialog.FileName));
        }
    }
}
```

### 3. 控制台批量处理

```csharp
class Program
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: BioastBatch <images_directory>");
            return;
        }

        string imagesDir = args[0];
        var predictor = new BioastPredictor("model.onnx");

        var imageFiles = Directory.GetFiles(imagesDir, "*.png");
        Console.WriteLine($"Processing {imageFiles.Length} images...\n");

        int positiveCount = 0;

        foreach (var imagePath in imageFiles)
        {
            var result = predictor.Predict(imagePath);

            if (result.GrowthLevel.Label == "positive")
            {
                positiveCount++;
                Console.WriteLine($"✅ {Path.GetFileName(imagePath)}: {result.GrowthPattern.Label}");
            }
        }

        Console.WriteLine($"\nSummary: {positiveCount}/{imageFiles.Length} positive samples");
    }
}
```

### 4. Windows Service

```csharp
public class ColonyDetectionService : BackgroundService
{
    private readonly BioastPredictor _predictor;
    private readonly string _watchFolder;

    public ColonyDetectionService(IConfiguration config)
    {
        _predictor = new BioastPredictor(config["ModelPath"]);
        _watchFolder = config["WatchFolder"];
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var watcher = new FileSystemWatcher(_watchFolder)
        {
            Filter = "*.png",
            NotifyFilter = NotifyFilters.FileName
        };

        watcher.Created += async (sender, e) =>
        {
            await Task.Delay(500); // 等待文件写入完成

            try
            {
                var result = _predictor.Predict(e.FullPath);

                // 保存结果
                var resultPath = Path.ChangeExtension(e.FullPath, ".json");
                var json = JsonSerializer.Serialize(result);
                await File.WriteAllTextAsync(resultPath, json);

                Console.WriteLine($"Processed: {e.Name} -> {result.GrowthLevel.Label}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing {e.Name}: {ex.Message}");
            }
        };

        watcher.EnableRaisingEvents = true;

        await Task.Delay(Timeout.Infinite, stoppingToken);
    }
}
```

---

## 最佳实践

### 1. 模型加载优化

**❌ 不推荐**: 每次推理重新加载模型
```csharp
// 性能差: 每次创建新实例
var result = new BioastPredictor("model.onnx").Predict("image.png");
```

**✅ 推荐**: 使用单例模式
```csharp
// 性能优: 共享模型实例
public sealed class PredictorSingleton
{
    private static readonly Lazy<BioastPredictor> _instance =
        new(() => new BioastPredictor("model.onnx"));

    public static BioastPredictor Instance => _instance.Value;
}

// 使用
var result = PredictorSingleton.Instance.Predict("image.png");
```

### 2. 错误处理

```csharp
public (bool Success, PredictionResult? Result, string? Error) TryPredict(string imagePath)
{
    try
    {
        if (!File.Exists(imagePath))
            return (false, null, "File not found");

        var result = _predictor.Predict(imagePath);

        // 验证置信度
        if (result.GrowthLevel.Confidence < 0.5f)
            _logger.LogWarning($"Low confidence: {result.GrowthLevel.Confidence:P2}");

        return (true, result, null);
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, $"Prediction failed for {imagePath}");
        return (false, null, ex.Message);
    }
}
```

### 3. 并行处理

```csharp
public async Task<List<PredictionResult>> ProcessInParallelAsync(string[] imagePaths)
{
    var results = new ConcurrentBag<PredictionResult>();

    await Parallel.ForEachAsync(imagePaths, async (path, token) =>
    {
        // 每个线程独立的 predictor 实例
        var predictor = new BioastPredictor("model.onnx");
        var result = predictor.Predict(path);
        results.Add(result);
    });

    return results.ToList();
}
```

### 4. 性能监控

```csharp
var stopwatch = Stopwatch.StartNew();
var result = predictor.Predict("image.png");
stopwatch.Stop();

if (stopwatch.ElapsedMilliseconds > 10)
{
    _logger.LogWarning($"Slow inference: {stopwatch.ElapsedMilliseconds} ms");
}
```

### 5. GPU 加速 (可选)

```csharp
// 安装: dotnet add package Microsoft.ML.OnnxRuntime.Gpu

public BioastPredictor(string modelPath, bool useGpu = false)
{
    var sessionOptions = new SessionOptions();
    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

    if (useGpu)
    {
        sessionOptions.AppendExecutionProvider_CUDA(0); // GPU 0
    }

    _session = new InferenceSession(modelPath, sessionOptions);
}
```

---

## 完整文档索引

1. **[QUICKSTART.md](csharp_example/QUICKSTART.md)** - 5分钟快速入门
2. **[README.md](csharp_example/README.md)** - 项目详细文档
3. **[USAGE_EXAMPLES.md](csharp_example/USAGE_EXAMPLES.md)** - 13个实用代码示例
4. **本文档** - 集成指南和最佳实践

### 代码文件

- [Program.cs](csharp_example/BioastOnnxInference/Program.cs) - 主程序 (单图推理)
- [BatchInferenceExample.cs](csharp_example/BioastOnnxInference/BatchInferenceExample.cs) - 批量处理和CSV导出
- [BioastOnnxInference.csproj](csharp_example/BioastOnnxInference/BioastOnnxInference.csproj) - 项目配置

### 模型文件

- [model.onnx](onnx_models/mobilenetv4_v0.11.0/model.onnx) - ONNX 模型 (3.69 MB)
- [model_info.json](onnx_models/mobilenetv4_v0.11.0/model_info.json) - 模型元数据

---

## 技术支持

### 常见问题

参阅 [QUICKSTART.md#常见问题](csharp_example/QUICKSTART.md#常见问题)

### 性能优化

参阅 [USAGE_EXAMPLES.md#性能优化](csharp_example/USAGE_EXAMPLES.md#性能优化)

### 错误处理

参阅 [USAGE_EXAMPLES.md#错误处理](csharp_example/USAGE_EXAMPLES.md#错误处理)

---

**模型版本**: MobileNetV4 v0.11.0
**文档版本**: 1.0
**最后更新**: 2025-10-04
