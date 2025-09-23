using System;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;

namespace BioastMicClassification
{
    /// <summary>
    /// C#使用示例程序
    /// </summary>
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🚀 核心边界优化多任务MIC MobileNetV3 C#推理示例");
            Console.WriteLine("Core Boundary Optimization Multitask MIC MobileNetV3 C# Inference Example");
            Console.WriteLine("=" * 80);

            // 模型文件路径
            string modelPath = "multitask_mic_mobilenetv3.onnx";
            string configDir = "./";
            
            // 测试图片路径
            string testImagePath = "test_image.png";

            try
            {
                // 创建分类器
                using var classifier = new MultitaskMicClassifier(modelPath, configDir);
                
                // 示例1：从文件路径推理
                if (File.Exists(testImagePath))
                {
                    Console.WriteLine($"\n📷 分析图片: {testImagePath}");
                    var result = await classifier.ClassifyAsync(testImagePath);
                    PrintResults(result);
                }
                
                // 示例2：从Bitmap推理
                Console.WriteLine("\n🎨 创建测试图片进行推理...");
                using var testBitmap = CreateTestImage();
                var testResult = await classifier.ClassifyAsync(testBitmap);
                PrintResults(testResult);
                
                // 示例3：批量处理
                string[] imagePaths = { "image1.png", "image2.png", "image3.png" };
                await BatchProcessImages(classifier, imagePaths);
                
                Console.WriteLine("\n✅ 推理完成！");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 错误: {ex.Message}");
                Console.WriteLine($"详细信息: {ex}");
            }
        }

        /// <summary>
        /// 打印分类结果
        /// </summary>
        static void PrintResults(ClassificationResult result)
        {
            Console.WriteLine("\n📊 分类结果:");
            
            // 主分类结果
            Console.WriteLine($"🎯 主分类: {result.Classification.PredictedClass}");
            Console.WriteLine($"   置信度: {result.Classification.Confidence:F4}");
            Console.WriteLine($"   阴性概率: {result.Classification.NegativeProbability:F4}");
            Console.WriteLine($"   阳性概率: {result.Classification.PositiveProbability:F4}");
            
            // 生长模式
            Console.WriteLine($"\n🌱 生长模式: {result.GrowthPattern.PredictedPattern}");
            Console.WriteLine($"   置信度: {result.GrowthPattern.Confidence:F4}");
            
            // 显示前3个最可能的生长模式
            var topPatterns = result.GrowthPattern.AllProbabilities
                .OrderByDescending(kvp => kvp.Value)
                .Take(3);
            Console.WriteLine("   前3个模式:");
            foreach (var pattern in topPatterns)
            {
                Console.WriteLine($"     {pattern.Key}: {pattern.Value:F4}");
            }
            
            // 干扰因素
            Console.WriteLine($"\n🔍 干扰因素检测:");
            Console.WriteLine($"   检测到的因素: {string.Join(", ", result.InterferenceFactors.DetectedFactors)}");
            Console.WriteLine($"   是否有气孔: {(result.InterferenceFactors.HasPores ? "是" : "否")}");
            
            Console.WriteLine("   所有因素概率:");
            foreach (var factor in result.InterferenceFactors.AllProbabilities)
            {
                Console.WriteLine($"     {factor.Key}: {factor.Value:F4}");
            }
        }

        /// <summary>
        /// 创建测试图片
        /// </summary>
        static Bitmap CreateTestImage()
        {
            var bitmap = new Bitmap(70, 70);
            using var g = Graphics.FromImage(bitmap);
            
            // 创建一个简单的测试图案
            g.Clear(Color.LightGray);
            g.FillEllipse(Brushes.DarkGray, 20, 20, 30, 30);
            g.FillEllipse(Brushes.Black, 25, 25, 5, 5); // 模拟气孔
            
            return bitmap;
        }

        /// <summary>
        /// 批量处理图片
        /// </summary>
        static async Task BatchProcessImages(MultitaskMicClassifier classifier, string[] imagePaths)
        {
            Console.WriteLine("\n📁 批量处理示例:");
            
            var tasks = new List<Task<(string path, ClassificationResult result)>>();
            
            foreach (var imagePath in imagePaths)
            {
                if (File.Exists(imagePath))
                {
                    tasks.Add(ProcessSingleImage(classifier, imagePath));
                }
                else
                {
                    Console.WriteLine($"⚠️ 文件不存在: {imagePath}");
                }
            }
            
            if (tasks.Count > 0)
            {
                var results = await Task.WhenAll(tasks);
                
                Console.WriteLine($"\n📈 批量处理结果 (共{results.Length}个图片):");
                foreach (var (path, result) in results)
                {
                    Console.WriteLine($"   {Path.GetFileName(path)}: {result.Classification.PredictedClass} ({result.Classification.Confidence:F3})");
                }
            }
            else
            {
                Console.WriteLine("   没有找到有效的图片文件");
            }
        }

        /// <summary>
        /// 处理单个图片
        /// </summary>
        static async Task<(string path, ClassificationResult result)> ProcessSingleImage(
            MultitaskMicClassifier classifier, string imagePath)
        {
            var result = await classifier.ClassifyAsync(imagePath);
            return (imagePath, result);
        }
    }

    /// <summary>
    /// 高级使用示例
    /// </summary>
    public class AdvancedUsageExample
    {
        /// <summary>
        /// 实时处理示例
        /// </summary>
        public static async Task RealTimeProcessingExample()
        {
            using var classifier = new MultitaskMicClassifier("multitask_mic_mobilenetv3.onnx");
            
            // 模拟实时图像流处理
            for (int i = 0; i < 10; i++)
            {
                using var testImage = CreateRandomTestImage();
                var result = await classifier.ClassifyAsync(testImage);
                
                Console.WriteLine($"Frame {i + 1}: {result.Classification.PredictedClass} " +
                                $"(Confidence: {result.Classification.Confidence:F3})");
                
                // 模拟处理间隔
                await Task.Delay(100);
            }
        }

        /// <summary>
        /// 基于条件的分类示例
        /// </summary>
        public static async Task ConditionalClassificationExample(string imagePath)
        {
            using var classifier = new MultitaskMicClassifier("multitask_mic_mobilenetv3.onnx");
            var result = await classifier.ClassifyAsync(imagePath);
            
            // 基于干扰因素调整判断
            if (result.InterferenceFactors.HasPores && 
                result.Classification.PredictedClass == "positive" &&
                result.Classification.Confidence < 0.8)
            {
                Console.WriteLine("⚠️ 检测到气孔且置信度较低，建议人工复核");
            }
            
            // 基于生长模式的特殊处理
            if (result.GrowthPattern.PredictedPattern == "weak_scattered_pos" &&
                result.Classification.PredictedClass == "negative")
            {
                Console.WriteLine("🔍 检测到弱分散模式但判定为阴性，可能是边界样本");
            }
        }

        /// <summary>
        /// 创建随机测试图片
        /// </summary>
        private static Bitmap CreateRandomTestImage()
        {
            var random = new Random();
            var bitmap = new Bitmap(70, 70);
            using var g = Graphics.FromImage(bitmap);
            
            g.Clear(Color.FromArgb(random.Next(200, 255), random.Next(200, 255), random.Next(200, 255)));
            
            // 随机添加一些特征
            if (random.NextDouble() > 0.5)
            {
                g.FillEllipse(Brushes.Gray, 
                    random.Next(20), random.Next(20), 
                    random.Next(10, 30), random.Next(10, 30));
            }
            
            return bitmap;
        }
    }
}