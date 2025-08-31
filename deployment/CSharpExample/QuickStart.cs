using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BioASTClassification
{
    /// <summary>
    /// 快速开始示例 - 最简单的使用方式
    /// </summary>
    public class QuickStartExample
    {
        public static void Main()
        {
            // 1. 创建分类器
            using var classifier = new SimpleBioASTClassifier("efficientnet_v2_s.onnx");
            
            // 2. 预测单张图像
            var result = classifier.Predict("path/to/your/image.png");
            
            // 3. 输出结果
            Console.WriteLine($"预测结果: {result.ClassName}");
            Console.WriteLine($"置信度: {result.Confidence:P2}");
            
            // 4. 根据结果做决策
            if (result.IsMalignant && result.Confidence > 0.8f)
            {
                Console.WriteLine("⚠️ 检测到高风险，建议进一步检查！");
            }
            else if (result.IsBenign && result.Confidence > 0.9f)
            {
                Console.WriteLine("✅ 结果正常。");
            }
            else
            {
                Console.WriteLine("⚠️ 置信度较低，建议人工复核。");
            }
        }
    }
    
    /// <summary>
    /// 简化版分类器 - 仅包含核心功能
    /// </summary>
    public class SimpleBioASTClassifier : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        
        // ImageNet标准化参数
        private readonly float[] _mean = { 0.485f, 0.456f, 0.406f };
        private readonly float[] _std = { 0.229f, 0.224f, 0.225f };
        
        public SimpleBioASTClassifier(string modelPath)
        {
            _session = new InferenceSession(modelPath);
            _inputName = _session.InputMetadata.Keys.First();
        }
        
        public SimpleResult Predict(string imagePath)
        {
            // 加载和预处理图像
            using var image = Image.Load<Rgb24>(imagePath);
            image.Mutate(x => x.Resize(70, 70));
            
            // 创建输入张量
            var tensor = new DenseTensor<float>(new[] { 1, 3, 70, 70 });
            
            // 图像预处理
            for (int y = 0; y < 70; y++)
            {
                for (int x = 0; x < 70; x++)
                {
                    var pixel = image[x, y];
                    
                    // 归一化和标准化
                    tensor[0, 0, y, x] = (pixel.R / 255.0f - _mean[0]) / _std[0];
                    tensor[0, 1, y, x] = (pixel.G / 255.0f - _mean[1]) / _std[1];
                    tensor[0, 2, y, x] = (pixel.B / 255.0f - _mean[2]) / _std[2];
                }
            }
            
            // 运行推理
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();
            
            // 计算概率
            var exp0 = Math.Exp(output[0]);
            var exp1 = Math.Exp(output[1]);
            var sum = exp0 + exp1;
            
            var benignProb = (float)(exp0 / sum);
            var malignantProb = (float)(exp1 / sum);
            
            var predictedClass = benignProb > malignantProb ? 0 : 1;
            var confidence = Math.Max(benignProb, malignantProb);
            
            return new SimpleResult
            {
                PredictedClass = predictedClass,
                ClassName = predictedClass == 0 ? "Benign" : "Malignant",
                Confidence = confidence,
                BenignProbability = benignProb,
                MalignantProbability = malignantProb,
                IsBenign = predictedClass == 0,
                IsMalignant = predictedClass == 1
            };
        }
        
        public void Dispose() => _session?.Dispose();
    }
    
    /// <summary>
    /// 简化的预测结果
    /// </summary>
    public class SimpleResult
    {
        public int PredictedClass { get; set; }
        public string ClassName { get; set; } = string.Empty;
        public float Confidence { get; set; }
        public float BenignProbability { get; set; }
        public float MalignantProbability { get; set; }
        public bool IsBenign { get; set; }
        public bool IsMalignant { get; set; }
    }
}