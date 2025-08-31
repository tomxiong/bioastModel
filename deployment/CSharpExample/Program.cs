using BioASTClassification;
using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace BioASTClassification
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== EfficientNetV2-S BioAST分类器演示 ===");
            Console.WriteLine();
            
            // 模型文件路径
            string modelPath = "efficientnet_v2_s.onnx";
            
            // 检查模型文件是否存在
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"错误: 模型文件不存在: {modelPath}");
                Console.WriteLine("请将 efficientnet_v2_s.onnx 文件复制到程序目录中。");
                Console.WriteLine("按任意键退出...");
                Console.ReadKey();
                return;
            }
            
            try
            {
                // 创建分类器
                using var classifier = new BioASTClassifier(modelPath);
                Console.WriteLine();
                
                // 显示菜单
                while (true)
                {
                    Console.WriteLine("请选择操作:");
                    Console.WriteLine("1. 单张图像预测");
                    Console.WriteLine("2. 批量图像预测");
                    Console.WriteLine("3. 异步批量预测");
                    Console.WriteLine("4. 性能测试");
                    Console.WriteLine("0. 退出");
                    Console.Write("请输入选择 (0-4): ");
                    
                    var choice = Console.ReadLine();
                    Console.WriteLine();
                    
                    switch (choice)
                    {
                        case "1":
                            await SingleImagePrediction(classifier);
                            break;
                        case "2":
                            await BatchPrediction(classifier);
                            break;
                        case "3":
                            await AsyncBatchPrediction(classifier);
                            break;
                        case "4":
                            await PerformanceTest(classifier);
                            break;
                        case "0":
                            Console.WriteLine("程序退出。");
                            return;
                        default:
                            Console.WriteLine("无效选择，请重新输入。");
                            break;
                    }
                    
                    Console.WriteLine();
                    Console.WriteLine("按任意键继续...");
                    Console.ReadKey();
                    Console.Clear();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"程序运行错误: {ex.Message}");
                Console.WriteLine("按任意键退出...");
                Console.ReadKey();
            }
        }
        
        /// <summary>
        /// 单张图像预测
        /// </summary>
        static async Task SingleImagePrediction(BioASTClassifier classifier)
        {
            Console.Write("请输入图像文件路径: ");
            var imagePath = Console.ReadLine()?.Trim('"'); // 去除可能的引号
            
            if (string.IsNullOrEmpty(imagePath))
            {
                Console.WriteLine("路径不能为空。");
                return;
            }
            
            if (!File.Exists(imagePath))
            {
                Console.WriteLine($"文件不存在: {imagePath}");
                return;
            }
            
            try
            {
                Console.WriteLine("正在预测...");
                var result = classifier.Predict(imagePath);
                
                Console.WriteLine("=== 预测结果 ===");
                Console.WriteLine($"文件: {Path.GetFileName(imagePath)}");
                Console.WriteLine($"预测类别: {result.ClassName}");
                Console.WriteLine($"置信度: {result.Confidence:P2}");
                Console.WriteLine($"Benign概率: {result.BenignProbability:P2}");
                Console.WriteLine($"Malignant概率: {result.MalignantProbability:P2}");
                Console.WriteLine($"推理时间: {result.InferenceTimeMs:F2} ms");
                
                // 医学建议
                if (result.IsMalignant && result.Confidence > 0.8f)
                {
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("⚠️  高置信度阳性结果，建议进一步检查！");
                    Console.ResetColor();
                }
                else if (result.IsBenign && result.Confidence > 0.9f)
                {
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("✅ 高置信度阴性结果。");
                    Console.ResetColor();
                }
                else
                {
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("⚠️  置信度较低，建议人工复核。");
                    Console.ResetColor();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"预测失败: {ex.Message}");
            }
        }
        
        /// <summary>
        /// 批量预测
        /// </summary>
        static async Task BatchPrediction(BioASTClassifier classifier)
        {
            Console.Write("请输入图像文件夹路径: ");
            var folderPath = Console.ReadLine()?.Trim('"');
            
            if (string.IsNullOrEmpty(folderPath) || !Directory.Exists(folderPath))
            {
                Console.WriteLine("文件夹不存在。");
                return;
            }
            
            // 支持的图像格式
            var supportedExtensions = new[] { ".png", ".jpg", ".jpeg", ".bmp", ".tiff" };
            var imageFiles = Directory.GetFiles(folderPath)
                .Where(f => supportedExtensions.Contains(Path.GetExtension(f).ToLower()))
                .ToArray();
            
            if (imageFiles.Length == 0)
            {
                Console.WriteLine("文件夹中没有找到支持的图像文件。");
                return;
            }
            
            Console.WriteLine($"找到 {imageFiles.Length} 个图像文件，开始批量预测...");
            
            var startTime = DateTime.UtcNow;
            var results = classifier.PredictBatch(imageFiles);
            var totalTime = DateTime.UtcNow - startTime;
            
            Console.WriteLine();
            Console.WriteLine("=== 批量预测结果 ===");
            
            int benignCount = 0, malignantCount = 0, errorCount = 0;
            
            foreach (var (imagePath, result) in results)
            {
                var fileName = Path.GetFileName(imagePath);
                
                if (result.PredictedClass == -1)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"{fileName}: 预测失败");
                    Console.ResetColor();
                    errorCount++;
                }
                else
                {
                    var color = result.IsMalignant ? ConsoleColor.Red : ConsoleColor.Green;
                    Console.ForegroundColor = color;
                    Console.WriteLine($"{fileName}: {result.ClassName} ({result.Confidence:P1})");
                    Console.ResetColor();
                    
                    if (result.IsBenign) benignCount++;
                    else malignantCount++;
                }
            }
            
            Console.WriteLine();
            Console.WriteLine("=== 统计信息 ===");
            Console.WriteLine($"总文件数: {imageFiles.Length}");
            Console.WriteLine($"Benign: {benignCount}");
            Console.WriteLine($"Malignant: {malignantCount}");
            Console.WriteLine($"错误: {errorCount}");
            Console.WriteLine($"总耗时: {totalTime.TotalSeconds:F2} 秒");
            Console.WriteLine($"平均耗时: {totalTime.TotalMilliseconds / imageFiles.Length:F2} ms/图像");
        }
        
        /// <summary>
        /// 异步批量预测
        /// </summary>
        static async Task AsyncBatchPrediction(BioASTClassifier classifier)
        {
            Console.Write("请输入图像文件夹路径: ");
            var folderPath = Console.ReadLine()?.Trim('"');
            
            if (string.IsNullOrEmpty(folderPath) || !Directory.Exists(folderPath))
            {
                Console.WriteLine("文件夹不存在。");
                return;
            }
            
            var supportedExtensions = new[] { ".png", ".jpg", ".jpeg", ".bmp", ".tiff" };
            var imageFiles = Directory.GetFiles(folderPath)
                .Where(f => supportedExtensions.Contains(Path.GetExtension(f).ToLower()))
                .ToArray();
            
            if (imageFiles.Length == 0)
            {
                Console.WriteLine("文件夹中没有找到支持的图像文件。");
                return;
            }
            
            Console.WriteLine($"找到 {imageFiles.Length} 个图像文件，开始异步批量预测...");
            
            var startTime = DateTime.UtcNow;
            var results = await classifier.PredictBatchAsync(imageFiles);
            var totalTime = DateTime.UtcNow - startTime;
            
            Console.WriteLine();
            Console.WriteLine("=== 异步批量预测结果 ===");
            
            int benignCount = 0, malignantCount = 0, errorCount = 0;
            
            foreach (var (imagePath, result) in results)
            {
                var fileName = Path.GetFileName(imagePath);
                
                if (result.PredictedClass == -1)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"{fileName}: 预测失败");
                    Console.ResetColor();
                    errorCount++;
                }
                else
                {
                    var color = result.IsMalignant ? ConsoleColor.Red : ConsoleColor.Green;
                    Console.ForegroundColor = color;
                    Console.WriteLine($"{fileName}: {result.ClassName} ({result.Confidence:P1})");
                    Console.ResetColor();
                    
                    if (result.IsBenign) benignCount++;
                    else malignantCount++;
                }
            }
            
            Console.WriteLine();
            Console.WriteLine("=== 统计信息 ===");
            Console.WriteLine($"总文件数: {imageFiles.Length}");
            Console.WriteLine($"Benign: {benignCount}");
            Console.WriteLine($"Malignant: {malignantCount}");
            Console.WriteLine($"错误: {errorCount}");
            Console.WriteLine($"总耗时: {totalTime.TotalSeconds:F2} 秒 (异步)");
            Console.WriteLine($"平均耗时: {totalTime.TotalMilliseconds / imageFiles.Length:F2} ms/图像");
        }
        
        /// <summary>
        /// 性能测试
        /// </summary>
        static async Task PerformanceTest(BioASTClassifier classifier)
        {
            Console.Write("请输入测试图像文件路径: ");
            var imagePath = Console.ReadLine()?.Trim('"');
            
            if (string.IsNullOrEmpty(imagePath) || !File.Exists(imagePath))
            {
                Console.WriteLine("文件不存在。");
                return;
            }
            
            Console.Write("请输入测试次数 (默认100): ");
            var input = Console.ReadLine();
            int testCount = 100;
            
            if (!string.IsNullOrEmpty(input) && int.TryParse(input, out var count) && count > 0)
            {
                testCount = count;
            }
            
            Console.WriteLine($"开始性能测试，预测 {testCount} 次...");
            
            var times = new List<double>();
            var results = new List<PredictionResult>();
            
            // 预热
            Console.WriteLine("预热中...");
            for (int i = 0; i < 5; i++)
            {
                classifier.Predict(imagePath);
            }
            
            // 正式测试
            Console.WriteLine("正式测试中...");
            var totalStartTime = DateTime.UtcNow;
            
            for (int i = 0; i < testCount; i++)
            {
                var result = classifier.Predict(imagePath);
                times.Add(result.InferenceTimeMs);
                results.Add(result);
                
                if ((i + 1) % 10 == 0)
                {
                    Console.Write($"\r进度: {i + 1}/{testCount}");
                }
            }
            
            var totalTime = DateTime.UtcNow - totalStartTime;
            Console.WriteLine();
            
            // 统计结果
            var avgTime = times.Average();
            var minTime = times.Min();
            var maxTime = times.Max();
            var medianTime = times.OrderBy(x => x).Skip(times.Count / 2).First();
            
            // 检查结果一致性
            var firstResult = results.First();
            var consistentResults = results.All(r => r.PredictedClass == firstResult.PredictedClass);
            
            Console.WriteLine();
            Console.WriteLine("=== 性能测试结果 ===");
            Console.WriteLine($"测试文件: {Path.GetFileName(imagePath)}");
            Console.WriteLine($"测试次数: {testCount}");
            Console.WriteLine($"预测结果: {firstResult.ClassName} ({firstResult.Confidence:P2})");
            Console.WriteLine($"结果一致性: {(consistentResults ? "✅ 一致" : "❌ 不一致")}");
            Console.WriteLine();
            Console.WriteLine("推理时间统计:");
            Console.WriteLine($"  平均时间: {avgTime:F2} ms");
            Console.WriteLine($"  最短时间: {minTime:F2} ms");
            Console.WriteLine($"  最长时间: {maxTime:F2} ms");
            Console.WriteLine($"  中位数时间: {medianTime:F2} ms");
            Console.WriteLine();
            Console.WriteLine($"总耗时: {totalTime.TotalSeconds:F2} 秒");
            Console.WriteLine($"吞吐量: {testCount / totalTime.TotalSeconds:F2} 图像/秒");
        }
    }
}