# BioAst ONNX C# Usage Examples

Complete usage guide with practical examples for integrating MobileNetV4 v0.11.0 ONNX model into C# applications.

## 📋 Table of Contents

1. [Basic Single Image Inference](#basic-single-image-inference)
2. [Batch Processing](#batch-processing)
3. [CSV Export](#csv-export)
4. [Production Integration](#production-integration)
5. [Performance Optimization](#performance-optimization)
6. [Error Handling](#error-handling)

---

## Basic Single Image Inference

### Example 1: Simple Prediction

```csharp
using BioastOnnxInference;

class Program
{
    static void Main(string[] args)
    {
        // Initialize predictor
        var predictor = new BioastPredictor("model.onnx");

        // Run inference
        var result = predictor.Predict("colony_sample.png");

        // Check growth level
        if (result.GrowthLevel.Label == "positive")
        {
            Console.WriteLine($"Colony detected! Pattern: {result.GrowthPattern.Label}");
        }
        else
        {
            Console.WriteLine("No colony detected");
        }
    }
}
```

### Example 2: Detailed Output

```csharp
var predictor = new BioastPredictor("model.onnx");
var result = predictor.Predict("test_image.png");

// Growth Level
Console.WriteLine($"Growth Level: {result.GrowthLevel.Label}");
Console.WriteLine($"Confidence: {result.GrowthLevel.Confidence:P2}");

// Growth Pattern with top predictions
Console.WriteLine("\nTop 3 Growth Patterns:");
var sortedProbs = result.GrowthPattern.Probabilities
    .Select((prob, idx) => new { Index = idx, Prob = prob })
    .OrderByDescending(x => x.Prob)
    .Take(3);

foreach (var item in sortedProbs)
{
    string label = LabelMappings["growth_pattern"][item.Index];
    Console.WriteLine($"  {label}: {item.Prob:P2}");
}

// Interference Factors
Console.WriteLine("\nInterference Factors:");
foreach (var factor in result.InterferenceFactors)
{
    Console.WriteLine($"  {factor.Name}: {factor.Score:F4} ({(factor.IsPresent ? "DETECTED" : "NOT DETECTED")})");
}
```

### Example 3: Quality Control Check

```csharp
var predictor = new BioastPredictor("model.onnx");
var result = predictor.Predict("sample.png");

// Check for quality issues
bool hasInterference = result.InterferenceFactors.Any(f => f.IsPresent);

if (hasInterference)
{
    var detectedFactors = result.InterferenceFactors
        .Where(f => f.IsPresent)
        .Select(f => f.Name);

    Console.WriteLine($"⚠️ Quality issues detected: {string.Join(", ", detectedFactors)}");
    Console.WriteLine("Sample may need manual review");
}
else
{
    Console.WriteLine("✅ No interference detected - sample is clean");
}
```

---

## Batch Processing

### Example 4: Process Directory

```csharp
using BioastOnnxInference;

class Program
{
    static void Main()
    {
        string modelPath = "model.onnx";
        string imagesDir = "test_images/";

        BatchInferenceExample.RunBatchInference(modelPath, imagesDir);
    }
}
```

**Output**:
```
=== Batch Inference Example ===

Model loaded successfully!
Found 150 images in test_images/

[1/150] sample_001.png
  Growth Level: positive (98.5%)
  Growth Pattern: clustered (89.2%)
  Interference: debris
  Inference time: 1.75 ms

[2/150] sample_002.png
  Growth Level: negative (95.3%)
  Growth Pattern: negative (92.1%)
  Interference: None detected
  Inference time: 1.68 ms

...

================================================================================
Batch Processing Summary
================================================================================

Processing Statistics:
  Total images processed: 150
  Successful: 150
  Errors: 0
  Total time: 5234.56 ms (5.23 seconds)
  Average inference time: 1.74 ms/image
  Throughput: 28.7 images/second

Growth Level Distribution:
  Positive: 89 (59.3%)
  Negative: 61 (40.7%)

Growth Pattern Distribution:
  clustered: 45 (30.0%)
  weak_scattered: 28 (18.7%)
  negative: 61 (40.7%)
  ...
```

### Example 5: Parallel Processing

```csharp
using System.Collections.Concurrent;
using System.Threading.Tasks;

public class ParallelBatchProcessor
{
    public static async Task ProcessImagesInParallel(string modelPath, string[] imagePaths)
    {
        var results = new ConcurrentBag<(string Path, PredictionResult Result)>();

        await Parallel.ForEachAsync(imagePaths, new ParallelOptions
        {
            MaxDegreeOfParallelism = Environment.ProcessorCount
        },
        async (imagePath, cancellationToken) =>
        {
            // Each thread gets its own predictor instance
            var predictor = new BioastPredictor(modelPath);
            var result = predictor.Predict(imagePath);
            results.Add((imagePath, result));
        });

        Console.WriteLine($"Processed {results.Count} images in parallel");

        // Analyze results
        int positiveCount = results.Count(r => r.Result.GrowthLevel.Label == "positive");
        Console.WriteLine($"Positive samples: {positiveCount}/{results.Count}");
    }
}
```

---

## CSV Export

### Example 6: Export Results to CSV

```csharp
using BioastOnnxInference;

class Program
{
    static void Main()
    {
        string modelPath = "model.onnx";
        string imagesDir = "production_images/";
        string outputCsv = "results.csv";

        CsvExportExample.ExportToCsv(modelPath, imagesDir, outputCsv);
    }
}
```

**Generated CSV** (`results.csv`):
```csv
filename,growth_level,growth_level_confidence,growth_pattern,growth_pattern_confidence,pores,artifacts,debris,contamination
sample_001.png,positive,0.9854,clustered,0.8923,0.1234,0.0567,0.6789,0.0012
sample_002.png,negative,0.9532,negative,0.9215,0.0234,0.0123,0.0089,0.0005
sample_003.png,positive,0.9921,heavy_growth,0.9456,0.4532,0.1234,0.2345,0.0023
...
```

### Example 7: Custom CSV with Filtering

```csharp
public static void ExportPositiveSamplesOnly(string modelPath, string imagesDir, string outputCsv)
{
    var predictor = new BioastPredictor(modelPath);
    var imageFiles = Directory.GetFiles(imagesDir, "*.png");

    using var writer = new StreamWriter(outputCsv);
    writer.WriteLine("filename,pattern,confidence,has_interference");

    foreach (var imagePath in imageFiles)
    {
        var result = predictor.Predict(imagePath);

        // Only export positive samples
        if (result.GrowthLevel.Label == "positive")
        {
            bool hasInterference = result.InterferenceFactors.Any(f => f.IsPresent);

            writer.WriteLine(
                $"{Path.GetFileName(imagePath)}," +
                $"{result.GrowthPattern.Label}," +
                $"{result.GrowthPattern.Confidence:F4}," +
                $"{hasInterference}"
            );
        }
    }
}
```

---

## Production Integration

### Example 8: REST API Integration

```csharp
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("api/[controller]")]
public class ColonyDetectionController : ControllerBase
{
    private readonly BioastPredictor _predictor;

    public ColonyDetectionController()
    {
        _predictor = new BioastPredictor("model.onnx");
    }

    [HttpPost("analyze")]
    public IActionResult AnalyzeImage([FromForm] IFormFile imageFile)
    {
        if (imageFile == null || imageFile.Length == 0)
            return BadRequest("No image provided");

        // Save uploaded file temporarily
        var tempPath = Path.GetTempFileName();
        using (var stream = new FileStream(tempPath, FileMode.Create))
        {
            imageFile.CopyTo(stream);
        }

        try
        {
            // Run inference
            var result = _predictor.Predict(tempPath);

            // Build response
            var response = new
            {
                growthLevel = new
                {
                    label = result.GrowthLevel.Label,
                    confidence = result.GrowthLevel.Confidence
                },
                growthPattern = new
                {
                    label = result.GrowthPattern.Label,
                    confidence = result.GrowthPattern.Confidence
                },
                interferenceFactors = result.InterferenceFactors.Select(f => new
                {
                    name = f.Name,
                    score = f.Score,
                    detected = f.IsPresent
                })
            };

            return Ok(response);
        }
        finally
        {
            // Cleanup
            if (System.IO.File.Exists(tempPath))
                System.IO.File.Delete(tempPath);
        }
    }
}
```

### Example 9: Database Integration

```csharp
using System.Data.SqlClient;

public class InferenceResultRepository
{
    private readonly string _connectionString;
    private readonly BioastPredictor _predictor;

    public InferenceResultRepository(string connectionString, string modelPath)
    {
        _connectionString = connectionString;
        _predictor = new BioastPredictor(modelPath);
    }

    public void ProcessAndSaveResult(string imagePath, string sampleId)
    {
        var result = _predictor.Predict(imagePath);

        using var connection = new SqlConnection(_connectionString);
        connection.Open();

        var command = new SqlCommand(@"
            INSERT INTO InferenceResults
            (SampleId, ImagePath, GrowthLevel, GrowthLevelConfidence,
             GrowthPattern, GrowthPatternConfidence, ProcessedAt)
            VALUES
            (@SampleId, @ImagePath, @GrowthLevel, @GrowthLevelConfidence,
             @GrowthPattern, @GrowthPatternConfidence, @ProcessedAt)
        ", connection);

        command.Parameters.AddWithValue("@SampleId", sampleId);
        command.Parameters.AddWithValue("@ImagePath", imagePath);
        command.Parameters.AddWithValue("@GrowthLevel", result.GrowthLevel.Label);
        command.Parameters.AddWithValue("@GrowthLevelConfidence", result.GrowthLevel.Confidence);
        command.Parameters.AddWithValue("@GrowthPattern", result.GrowthPattern.Label);
        command.Parameters.AddWithValue("@GrowthPatternConfidence", result.GrowthPattern.Confidence);
        command.Parameters.AddWithValue("@ProcessedAt", DateTime.UtcNow);

        command.ExecuteNonQuery();

        // Save interference factors
        SaveInterferenceFactors(connection, sampleId, result.InterferenceFactors);
    }

    private void SaveInterferenceFactors(SqlConnection connection, string sampleId,
        List<InterferenceFactorOutput> factors)
    {
        foreach (var factor in factors.Where(f => f.IsPresent))
        {
            var command = new SqlCommand(@"
                INSERT INTO InterferenceFactors (SampleId, FactorName, Score)
                VALUES (@SampleId, @FactorName, @Score)
            ", connection);

            command.Parameters.AddWithValue("@SampleId", sampleId);
            command.Parameters.AddWithValue("@FactorName", factor.Name);
            command.Parameters.AddWithValue("@Score", factor.Score);

            command.ExecuteNonQuery();
        }
    }
}
```

---

## Performance Optimization

### Example 10: Singleton Pattern for Model Sharing

```csharp
public sealed class BioastPredictorSingleton
{
    private static readonly Lazy<BioastPredictor> _instance =
        new Lazy<BioastPredictor>(() => new BioastPredictor("model.onnx"));

    public static BioastPredictor Instance => _instance.Value;

    private BioastPredictorSingleton() { }
}

// Usage
var result = BioastPredictorSingleton.Instance.Predict("image.png");
```

### Example 11: Image Caching

```csharp
using System.Collections.Concurrent;

public class CachingPredictor
{
    private readonly BioastPredictor _predictor;
    private readonly ConcurrentDictionary<string, PredictionResult> _cache;

    public CachingPredictor(string modelPath)
    {
        _predictor = new BioastPredictor(modelPath);
        _cache = new ConcurrentDictionary<string, PredictionResult>();
    }

    public PredictionResult PredictWithCache(string imagePath)
    {
        // Use file hash as cache key
        string cacheKey = GetFileHash(imagePath);

        return _cache.GetOrAdd(cacheKey, _ => _predictor.Predict(imagePath));
    }

    private string GetFileHash(string filePath)
    {
        using var md5 = System.Security.Cryptography.MD5.Create();
        using var stream = File.OpenRead(filePath);
        var hash = md5.ComputeHash(stream);
        return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
    }
}
```

---

## Error Handling

### Example 12: Robust Error Handling

```csharp
public class SafePredictor
{
    private readonly BioastPredictor _predictor;
    private readonly ILogger _logger;

    public SafePredictor(string modelPath, ILogger logger)
    {
        _predictor = new BioastPredictor(modelPath);
        _logger = logger;
    }

    public (bool Success, PredictionResult? Result, string? Error) TryPredict(string imagePath)
    {
        try
        {
            // Validate input
            if (!File.Exists(imagePath))
            {
                return (false, null, $"Image file not found: {imagePath}");
            }

            var fileInfo = new FileInfo(imagePath);
            if (fileInfo.Length == 0)
            {
                return (false, null, "Image file is empty");
            }

            // Run inference
            var result = _predictor.Predict(imagePath);

            // Validate output
            if (result.GrowthLevel.Confidence < 0.5f)
            {
                _logger.LogWarning($"Low confidence prediction for {imagePath}: {result.GrowthLevel.Confidence:P2}");
            }

            return (true, result, null);
        }
        catch (OutOfMemoryException ex)
        {
            _logger.LogError(ex, $"Out of memory processing {imagePath}");
            return (false, null, "Insufficient memory");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error processing {imagePath}");
            return (false, null, ex.Message);
        }
    }
}

// Usage
var safePredictor = new SafePredictor("model.onnx", logger);

foreach (var imagePath in imageFiles)
{
    var (success, result, error) = safePredictor.TryPredict(imagePath);

    if (success && result != null)
    {
        Console.WriteLine($"✅ {imagePath}: {result.GrowthLevel.Label}");
    }
    else
    {
        Console.WriteLine($"❌ {imagePath}: {error}");
    }
}
```

### Example 13: Retry Logic

```csharp
using Polly;

public class ResilientPredictor
{
    private readonly BioastPredictor _predictor;
    private readonly IAsyncPolicy _retryPolicy;

    public ResilientPredictor(string modelPath)
    {
        _predictor = new BioastPredictor(modelPath);

        _retryPolicy = Policy
            .Handle<Exception>()
            .WaitAndRetryAsync(
                retryCount: 3,
                sleepDurationProvider: attempt => TimeSpan.FromMilliseconds(100 * Math.Pow(2, attempt)),
                onRetry: (exception, timeSpan, retryCount, context) =>
                {
                    Console.WriteLine($"Retry {retryCount} after {timeSpan.TotalMilliseconds}ms due to: {exception.Message}");
                }
            );
    }

    public async Task<PredictionResult> PredictWithRetryAsync(string imagePath)
    {
        return await _retryPolicy.ExecuteAsync(() =>
        {
            return Task.FromResult(_predictor.Predict(imagePath));
        });
    }
}
```

---

## 🎯 Best Practices Summary

1. **Model Loading**: Load model once and reuse (singleton pattern)
2. **Batch Processing**: Process multiple images to amortize model loading cost
3. **Error Handling**: Always validate inputs and handle exceptions gracefully
4. **Performance**: Use parallel processing for large batches
5. **Production**: Implement caching, retry logic, and comprehensive logging
6. **Quality Control**: Check confidence scores and interference factors
7. **Data Export**: Save results to CSV/database for analysis

---

## 📚 Additional Resources

- **Main README**: [README.md](README.md)
- **Model Info**: [model_info.json](../model_info.json)
- **Performance Report**: `/home/aaa/ws/bioastModel/V0.11.0_EVALUATION_SUMMARY.md`

---

**Model Version**: MobileNetV4 v0.11.0
**Last Updated**: 2025-10-04
