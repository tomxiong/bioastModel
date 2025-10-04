using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Diagnostics;

namespace BioastOnnxInference
{
    /// <summary>
    /// Example demonstrating batch inference for multiple images
    /// </summary>
    public class BatchInferenceExample
    {
        public static void RunBatchInference(string modelPath, string imagesDirectory)
        {
            Console.WriteLine("=== Batch Inference Example ===\n");

            // Initialize predictor
            var predictor = new BioastPredictor(modelPath);

            // Get all image files
            var imageExtensions = new[] { ".png", ".jpg", ".jpeg", ".bmp" };
            var imageFiles = Directory.GetFiles(imagesDirectory)
                .Where(f => imageExtensions.Contains(Path.GetExtension(f).ToLower()))
                .ToArray();

            Console.WriteLine($"Found {imageFiles.Length} images in {imagesDirectory}\n");

            if (imageFiles.Length == 0)
            {
                Console.WriteLine("No images found. Please provide a directory with image files.");
                return;
            }

            // Statistics
            var stats = new BatchStatistics();
            var stopwatch = Stopwatch.StartNew();

            // Process each image
            foreach (var imagePath in imageFiles)
            {
                try
                {
                    var imageStopwatch = Stopwatch.StartNew();
                    var result = predictor.Predict(imagePath);
                    imageStopwatch.Stop();

                    // Update statistics
                    stats.TotalImages++;
                    stats.TotalInferenceTime += imageStopwatch.Elapsed.TotalMilliseconds;

                    if (result.GrowthLevel.Label == "positive")
                        stats.PositiveCount++;
                    else
                        stats.NegativeCount++;

                    stats.PatternCounts[result.GrowthPattern.Label] =
                        stats.PatternCounts.GetValueOrDefault(result.GrowthPattern.Label, 0) + 1;

                    foreach (var factor in result.InterferenceFactors.Where(f => f.IsPresent))
                    {
                        stats.InterferenceCounts[factor.Name] =
                            stats.InterferenceCounts.GetValueOrDefault(factor.Name, 0) + 1;
                    }

                    // Print result
                    Console.WriteLine($"[{stats.TotalImages}/{imageFiles.Length}] {Path.GetFileName(imagePath)}");
                    Console.WriteLine($"  Growth Level: {result.GrowthLevel.Label} ({result.GrowthLevel.Confidence:P1})");
                    Console.WriteLine($"  Growth Pattern: {result.GrowthPattern.Label} ({result.GrowthPattern.Confidence:P1})");

                    var detectedInterferences = result.InterferenceFactors
                        .Where(f => f.IsPresent)
                        .Select(f => f.Name)
                        .ToArray();

                    if (detectedInterferences.Any())
                    {
                        Console.WriteLine($"  Interference: {string.Join(", ", detectedInterferences)}");
                    }
                    else
                    {
                        Console.WriteLine($"  Interference: None detected");
                    }

                    Console.WriteLine($"  Inference time: {imageStopwatch.Elapsed.TotalMilliseconds:F2} ms\n");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing {Path.GetFileName(imagePath)}: {ex.Message}\n");
                    stats.ErrorCount++;
                }
            }

            stopwatch.Stop();

            // Print summary
            PrintSummary(stats, stopwatch.Elapsed.TotalMilliseconds);
        }

        private static void PrintSummary(BatchStatistics stats, double totalTime)
        {
            Console.WriteLine("\n" + new string('=', 80));
            Console.WriteLine("Batch Processing Summary");
            Console.WriteLine(new string('=', 80));

            Console.WriteLine($"\nProcessing Statistics:");
            Console.WriteLine($"  Total images processed: {stats.TotalImages}");
            Console.WriteLine($"  Successful: {stats.TotalImages - stats.ErrorCount}");
            Console.WriteLine($"  Errors: {stats.ErrorCount}");
            Console.WriteLine($"  Total time: {totalTime:F2} ms ({totalTime / 1000:F2} seconds)");
            Console.WriteLine($"  Average inference time: {stats.TotalInferenceTime / stats.TotalImages:F2} ms/image");
            Console.WriteLine($"  Throughput: {stats.TotalImages / (totalTime / 1000):F1} images/second");

            Console.WriteLine($"\nGrowth Level Distribution:");
            Console.WriteLine($"  Positive: {stats.PositiveCount} ({100.0 * stats.PositiveCount / stats.TotalImages:F1}%)");
            Console.WriteLine($"  Negative: {stats.NegativeCount} ({100.0 * stats.NegativeCount / stats.TotalImages:F1}%)");

            if (stats.PatternCounts.Any())
            {
                Console.WriteLine($"\nGrowth Pattern Distribution:");
                foreach (var (pattern, count) in stats.PatternCounts.OrderByDescending(kv => kv.Value))
                {
                    Console.WriteLine($"  {pattern}: {count} ({100.0 * count / stats.TotalImages:F1}%)");
                }
            }

            if (stats.InterferenceCounts.Any())
            {
                Console.WriteLine($"\nInterference Factors Detected:");
                foreach (var (factor, count) in stats.InterferenceCounts.OrderByDescending(kv => kv.Value))
                {
                    Console.WriteLine($"  {factor}: {count} ({100.0 * count / stats.TotalImages:F1}%)");
                }
            }

            Console.WriteLine("\n" + new string('=', 80));
        }

        private class BatchStatistics
        {
            public int TotalImages { get; set; }
            public int PositiveCount { get; set; }
            public int NegativeCount { get; set; }
            public int ErrorCount { get; set; }
            public double TotalInferenceTime { get; set; }
            public Dictionary<string, int> PatternCounts { get; set; } = new();
            public Dictionary<string, int> InterferenceCounts { get; set; } = new();
        }
    }

    /// <summary>
    /// Example demonstrating CSV export of results
    /// </summary>
    public class CsvExportExample
    {
        public static void ExportToCsv(string modelPath, string imagesDirectory, string outputCsvPath)
        {
            Console.WriteLine("=== CSV Export Example ===\n");

            var predictor = new BioastPredictor(modelPath);

            var imageExtensions = new[] { ".png", ".jpg", ".jpeg", ".bmp" };
            var imageFiles = Directory.GetFiles(imagesDirectory)
                .Where(f => imageExtensions.Contains(Path.GetExtension(f).ToLower()))
                .ToArray();

            Console.WriteLine($"Processing {imageFiles.Length} images...\n");

            using var writer = new StreamWriter(outputCsvPath);

            // Write header
            writer.WriteLine("filename,growth_level,growth_level_confidence,growth_pattern,growth_pattern_confidence,pores,artifacts,debris,contamination");

            // Process each image
            int processed = 0;
            foreach (var imagePath in imageFiles)
            {
                try
                {
                    var result = predictor.Predict(imagePath);
                    processed++;

                    var poresScore = result.InterferenceFactors.First(f => f.Name == "pores").Score;
                    var artifactsScore = result.InterferenceFactors.First(f => f.Name == "artifacts").Score;
                    var debrisScore = result.InterferenceFactors.First(f => f.Name == "debris").Score;
                    var contaminationScore = result.InterferenceFactors.First(f => f.Name == "contamination").Score;

                    writer.WriteLine(
                        $"{Path.GetFileName(imagePath)}," +
                        $"{result.GrowthLevel.Label}," +
                        $"{result.GrowthLevel.Confidence:F4}," +
                        $"{result.GrowthPattern.Label}," +
                        $"{result.GrowthPattern.Confidence:F4}," +
                        $"{poresScore:F4}," +
                        $"{artifactsScore:F4}," +
                        $"{debrisScore:F4}," +
                        $"{contaminationScore:F4}"
                    );

                    if (processed % 100 == 0)
                    {
                        Console.WriteLine($"Processed {processed}/{imageFiles.Length} images...");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing {Path.GetFileName(imagePath)}: {ex.Message}");
                }
            }

            Console.WriteLine($"\nResults exported to: {outputCsvPath}");
            Console.WriteLine($"Successfully processed: {processed}/{imageFiles.Length} images");
        }
    }
}
