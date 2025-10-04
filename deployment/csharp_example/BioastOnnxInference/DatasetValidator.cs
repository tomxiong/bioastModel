using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Diagnostics;

namespace BioastOnnxInference
{
    /// <summary>
    /// 数据集标注信息
    /// </summary>
    public class DatasetAnnotations
    {
        [JsonPropertyName("images")]
        public Dictionary<string, ImageAnnotation> Images { get; set; } = new();
    }

    public class ImageAnnotation
    {
        [JsonPropertyName("growth_level")]
        public string GrowthLevel { get; set; } = string.Empty;

        [JsonPropertyName("growth_pattern")]
        public string GrowthPattern { get; set; } = string.Empty;

        [JsonPropertyName("interference_factors")]
        public Dictionary<string, bool> InterferenceFactors { get; set; } = new();
    }

    /// <summary>
    /// 数据集划分信息
    /// </summary>
    public class DatasetSplit
    {
        [JsonPropertyName("test")]
        public List<string> Test { get; set; } = new();
    }

    /// <summary>
    /// 验证结果统计
    /// </summary>
    public class ValidationResult
    {
        public int TotalSamples { get; set; }
        public int GrowthLevelCorrect { get; set; }
        public int GrowthPatternCorrect { get; set; }
        public Dictionary<string, int> InterferenceCorrect { get; set; } = new();
        public Dictionary<string, int> InterferenceTotal { get; set; } = new();

        // 混淆矩阵 (Growth Level)
        public int TrueNegative { get; set; }
        public int TruePositive { get; set; }
        public int FalseNegative { get; set; }
        public int FalsePositive { get; set; }

        // 错误样本
        public List<ErrorSample> ErrorSamples { get; set; } = new();

        // 干扰因素详细统计
        public Dictionary<string, InterferenceStats> InterferenceStats { get; set; } = new();
    }

    public class ErrorSample
    {
        public string ImagePath { get; set; } = string.Empty;
        public string ErrorType { get; set; } = string.Empty;
        public string Expected { get; set; } = string.Empty;
        public string Predicted { get; set; } = string.Empty;
        public float Confidence { get; set; }
    }

    public class InterferenceStats
    {
        public int TruePositive { get; set; }
        public int TrueNegative { get; set; }
        public int FalsePositive { get; set; }
        public int FalseNegative { get; set; }

        public double Precision => TruePositive + FalsePositive > 0
            ? (double)TruePositive / (TruePositive + FalsePositive)
            : 0;

        public double Recall => TruePositive + FalseNegative > 0
            ? (double)TruePositive / (TruePositive + FalseNegative)
            : 0;

        public double F1Score => Precision + Recall > 0
            ? 2 * (Precision * Recall) / (Precision + Recall)
            : 0;

        public double Accuracy => TruePositive + TrueNegative + FalsePositive + FalseNegative > 0
            ? (double)(TruePositive + TrueNegative) / (TruePositive + TrueNegative + FalsePositive + FalseNegative)
            : 0;
    }

    /// <summary>
    /// 数据集验证器
    /// </summary>
    public class DatasetValidator
    {
        private readonly BioastPredictor _predictor;
        private readonly string _dataRoot;
        private readonly string _annotationsFile;
        private readonly string _splitFile;

        public DatasetValidator(
            string modelPath,
            string dataRoot = "ds/images",
            string annotationsFile = "ds/images/m9e1n170_cleaned_round2.json",
            string splitFile = "ds/images/dataset_split_seed44.json")
        {
            _predictor = new BioastPredictor(modelPath);
            _dataRoot = dataRoot;
            _annotationsFile = annotationsFile;
            _splitFile = splitFile;
        }

        public ValidationResult ValidateTestSet()
        {
            Console.WriteLine("=== 数据集验证开始 ===\n");

            // 1. 加载标注
            Console.WriteLine("[1/4] 加载数据集标注...");
            var annotations = LoadAnnotations(_annotationsFile);
            Console.WriteLine($"  加载了 {annotations.Images.Count} 个图像标注");

            // 2. 加载测试集划分
            Console.WriteLine("\n[2/4] 加载测试集划分...");
            var testImages = LoadTestSplit(_splitFile);
            Console.WriteLine($"  测试集包含 {testImages.Count} 个样本");

            // 3. 运行推理
            Console.WriteLine("\n[3/4] 开始批量推理...");
            var result = new ValidationResult
            {
                TotalSamples = testImages.Count
            };

            // 初始化干扰因素统计
            var interferenceFactors = new[] { "pores", "artifacts", "debris", "contamination" };
            foreach (var factor in interferenceFactors)
            {
                result.InterferenceCorrect[factor] = 0;
                result.InterferenceTotal[factor] = 0;
                result.InterferenceStats[factor] = new InterferenceStats();
            }

            var stopwatch = Stopwatch.StartNew();
            int processed = 0;

            foreach (var imageName in testImages)
            {
                if (!annotations.Images.TryGetValue(imageName, out var annotation))
                {
                    Console.WriteLine($"  警告: 找不到图像标注 {imageName}");
                    continue;
                }

                // 构建图像路径
                var parts = imageName.Split('/');
                var imagePath = Path.Combine(_dataRoot, parts[0], parts[1]);

                if (!File.Exists(imagePath))
                {
                    Console.WriteLine($"  警告: 找不到图像文件 {imagePath}");
                    continue;
                }

                try
                {
                    // 运行推理
                    var prediction = _predictor.Predict(imagePath);

                    // 验证 Growth Level
                    if (prediction.GrowthLevel.Label == annotation.GrowthLevel)
                    {
                        result.GrowthLevelCorrect++;

                        if (annotation.GrowthLevel == "negative")
                            result.TrueNegative++;
                        else
                            result.TruePositive++;
                    }
                    else
                    {
                        if (annotation.GrowthLevel == "negative")
                        {
                            result.FalsePositive++;
                        }
                        else
                        {
                            result.FalseNegative++;
                        }

                        result.ErrorSamples.Add(new ErrorSample
                        {
                            ImagePath = imageName,
                            ErrorType = "Growth Level",
                            Expected = annotation.GrowthLevel,
                            Predicted = prediction.GrowthLevel.Label,
                            Confidence = prediction.GrowthLevel.Confidence
                        });
                    }

                    // 验证 Growth Pattern
                    if (prediction.GrowthPattern.Label == annotation.GrowthPattern)
                    {
                        result.GrowthPatternCorrect++;
                    }
                    else
                    {
                        result.ErrorSamples.Add(new ErrorSample
                        {
                            ImagePath = imageName,
                            ErrorType = "Growth Pattern",
                            Expected = annotation.GrowthPattern,
                            Predicted = prediction.GrowthPattern.Label,
                            Confidence = prediction.GrowthPattern.Confidence
                        });
                    }

                    // 验证 Interference Factors
                    foreach (var factor in prediction.InterferenceFactors)
                    {
                        if (annotation.InterferenceFactors.TryGetValue(factor.Name, out var expected))
                        {
                            result.InterferenceTotal[factor.Name]++;

                            var stats = result.InterferenceStats[factor.Name];

                            if (factor.IsPresent == expected)
                            {
                                result.InterferenceCorrect[factor.Name]++;

                                if (factor.IsPresent)
                                    stats.TruePositive++;
                                else
                                    stats.TrueNegative++;
                            }
                            else
                            {
                                if (factor.IsPresent && !expected)
                                {
                                    stats.FalsePositive++;

                                    result.ErrorSamples.Add(new ErrorSample
                                    {
                                        ImagePath = imageName,
                                        ErrorType = $"Interference - {factor.Name} (FP)",
                                        Expected = "false",
                                        Predicted = "true",
                                        Confidence = factor.Score
                                    });
                                }
                                else if (!factor.IsPresent && expected)
                                {
                                    stats.FalseNegative++;

                                    result.ErrorSamples.Add(new ErrorSample
                                    {
                                        ImagePath = imageName,
                                        ErrorType = $"Interference - {factor.Name} (FN)",
                                        Expected = "true",
                                        Predicted = "false",
                                        Confidence = factor.Score
                                    });
                                }
                            }
                        }
                    }

                    processed++;

                    if (processed % 100 == 0)
                    {
                        Console.WriteLine($"  已处理: {processed}/{testImages.Count}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  错误: 处理 {imageName} 时出错: {ex.Message}");
                }
            }

            stopwatch.Stop();

            Console.WriteLine($"\n  完成! 处理了 {processed} 个样本");
            Console.WriteLine($"  总耗时: {stopwatch.Elapsed.TotalSeconds:F2} 秒");
            Console.WriteLine($"  平均推理时间: {stopwatch.Elapsed.TotalMilliseconds / processed:F2} ms/图像");

            return result;
        }

        private DatasetAnnotations LoadAnnotations(string filePath)
        {
            var json = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<DatasetAnnotations>(json)
                ?? throw new Exception("Failed to load annotations");
        }

        private List<string> LoadTestSplit(string filePath)
        {
            var json = File.ReadAllText(filePath);
            var split = JsonSerializer.Deserialize<DatasetSplit>(json)
                ?? throw new Exception("Failed to load dataset split");
            return split.Test;
        }

        public static void PrintResults(ValidationResult result)
        {
            Console.WriteLine("\n" + new string('=', 80));
            Console.WriteLine("验证结果汇总");
            Console.WriteLine(new string('=', 80));

            // 总体准确率
            Console.WriteLine("\n[总体性能]");
            var growthLevelAcc = 100.0 * result.GrowthLevelCorrect / result.TotalSamples;
            var growthPatternAcc = 100.0 * result.GrowthPatternCorrect / result.TotalSamples;

            Console.WriteLine($"  Growth Level 准确率: {growthLevelAcc:F2}% ({result.GrowthLevelCorrect}/{result.TotalSamples})");
            Console.WriteLine($"  Growth Pattern 准确率: {growthPatternAcc:F2}% ({result.GrowthPatternCorrect}/{result.TotalSamples})");

            // Interference Factors
            Console.WriteLine("\n[Interference Factors 准确率]");
            double interferenceOverallAcc = 0;
            int factorCount = 0;

            foreach (var (factor, stats) in result.InterferenceStats.OrderBy(kv => kv.Key))
            {
                Console.WriteLine($"  {factor}:");
                Console.WriteLine($"    准确率: {stats.Accuracy:P2}");
                Console.WriteLine($"    精确率: {stats.Precision:P2}");
                Console.WriteLine($"    召回率: {stats.Recall:P2}");
                Console.WriteLine($"    F1分数: {stats.F1Score:P2}");
                Console.WriteLine($"    TP={stats.TruePositive}, FP={stats.FalsePositive}, FN={stats.FalseNegative}, TN={stats.TrueNegative}");

                interferenceOverallAcc += stats.Accuracy;
                factorCount++;
            }

            if (factorCount > 0)
            {
                interferenceOverallAcc /= factorCount;
                Console.WriteLine($"\n  Interference Overall 准确率: {interferenceOverallAcc:P2}");
            }

            // 总准确率
            var totalAccuracy = (growthLevelAcc + growthPatternAcc + interferenceOverallAcc * 100) / 3;
            Console.WriteLine($"\n[总准确率] {totalAccuracy:F2}%");

            // Growth Level 混淆矩阵
            Console.WriteLine("\n[Growth Level 混淆矩阵]");
            Console.WriteLine($"              Predicted Negative  Predicted Positive");
            Console.WriteLine($"  Actual Negative:     {result.TrueNegative,-6}            {result.FalsePositive,-6}");
            Console.WriteLine($"  Actual Positive:     {result.FalseNegative,-6}            {result.TruePositive,-6}");

            // 错误样本统计
            Console.WriteLine("\n[错误样本统计]");
            var errorsByType = result.ErrorSamples
                .GroupBy(e => e.ErrorType)
                .OrderByDescending(g => g.Count());

            foreach (var group in errorsByType)
            {
                Console.WriteLine($"  {group.Key}: {group.Count()} 个错误");
            }

            // 显示前10个错误样本
            Console.WriteLine("\n[前10个错误样本]");
            foreach (var error in result.ErrorSamples.Take(10))
            {
                Console.WriteLine($"  {error.ImagePath}");
                Console.WriteLine($"    类型: {error.ErrorType}");
                Console.WriteLine($"    期望: {error.Expected}, 预测: {error.Predicted} (置信度: {error.Confidence:P2})");
            }

            if (result.ErrorSamples.Count > 10)
            {
                Console.WriteLine($"  ... 还有 {result.ErrorSamples.Count - 10} 个错误样本");
            }

            Console.WriteLine("\n" + new string('=', 80));
        }

        public static void ExportResults(ValidationResult result, string outputPath)
        {
            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };

            var json = JsonSerializer.Serialize(result, options);
            File.WriteAllText(outputPath, json);

            Console.WriteLine($"\n验证结果已导出到: {outputPath}");
        }
    }
}
