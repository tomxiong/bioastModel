using System;
using System.IO;
using System.Text.Json;

namespace M16MultitaskDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("M16多任务MobileNetV3 .NET演示程序");
            Console.WriteLine("=====================================");

            // 检查模型文件
            string modelPath = "../onnx_models/m16_multitask_mobilenetv3.onnx";
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"错误: 模型文件不存在: {modelPath}");
                Console.WriteLine("请确保模型文件在正确位置");
                Console.WriteLine("当前工作目录: " + Directory.GetCurrentDirectory());
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
            Console.WriteLine("3. 输入 'test' 运行测试");
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

                    if (imagePath.ToLower() == "test")
                    {
                        RunTest(inference);
                        continue;
                    }

                    if (!File.Exists(imagePath))
                    {
                        Console.WriteLine($"错误: 文件不存在: {imagePath}");
                        continue;
                    }

                    // 预测
                    Console.WriteLine($"\n正在分析图像: {imagePath}");
                    var result = PredictImage(inference, imagePath);
                    PrintResults(result);

                    // 保存结果到JSON
                    string jsonResult = JsonSerializer.Serialize(result, new JsonSerializerOptions
                    {
                        WriteIndented = true,
                        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                    });
                    
                    string outputPath = Path.Combine(
                        Path.GetDirectoryName(imagePath) ?? "",
                        Path.GetFileNameWithoutExtension(imagePath) + "_result.json");
                    
                    File.WriteAllText(outputPath, jsonResult);
                    Console.WriteLine($"\n结果已保存到: {outputPath}");

                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\n错误: {ex.Message}");
                    Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
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

        static void RunTest(M16MultitaskInference inference)
        {
            Console.WriteLine("\n=== 运行测试 ===");
            
            // 创建测试图像 (70x70 纯色图像)
            string testImagePath = "test_image.jpg";
            using (var image = new SixLabors.ImageSharp.Image<Rgb24>(70, 70))
            {
                // 创建一个简单的测试图像
                for (int y = 0; y < 70; y++)
                {
                    for (int x = 0; x < 70; x++)
                    {
                        image[x, y] = new Rgb24(
                            (byte)((x / 70.0) * 255),  // Red gradient
                            (byte)((y / 70.0) * 255),  // Green gradient
                            (byte)128                // Blue constant
                        );
                    }
                }
                image.Save(testImagePath);
            }

            Console.WriteLine($"创建测试图像: {testImagePath}");

            // 运行预测
            var result = PredictImage(inference, testImagePath);
            PrintResults(result);

            // 清理测试文件
            if (File.Exists(testImagePath))
            {
                File.Delete(testImagePath);
                Console.WriteLine($"删除测试图像: {testImagePath}");
            }
        }
    }
}