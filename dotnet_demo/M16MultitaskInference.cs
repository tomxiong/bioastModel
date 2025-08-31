using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace M16MultitaskDemo
{
    public class M16MultitaskInference : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string[] _growthLevelClasses;
        private readonly string[] _growthPatternClasses;
        private readonly string[] _interferenceClasses;
        
        // 预处理参数
        private readonly float[] _mean = new float[] { 0.485f, 0.456f, 0.406f };
        private readonly float[] _std = new float[] { 0.229f, 0.224, 0.225f };
        private readonly int _imageSize = 70;

        public M16MultitaskInference(string modelPath)
        {
            // 加载ONNX模型
            var sessionOptions = new SessionOptions();
            _session = new InferenceSession(modelPath, sessionOptions);

            // 类别定义
            _growthLevelClasses = new[] { "negative", "positive", "weak_growth" };
            _growthPatternClasses = new[] 
            { 
                "clean", "clustered", "scattered", "heavy_growth", "small_dots", 
                "irregular_areas", "light_gray", "default_positive", "default_weak_growth" 
            };
            _interferenceClasses = new[] { "pores", "debris", "artifacts" };
        }

        /// <summary>
        /// 预处理图像
        /// </summary>
        /// <param name="imagePath">图像路径</param>
        /// <returns>预处理后的张量</returns>
        public DenseTensor<float> PreprocessImage(string imagePath)
        {
            using var image = Image.Load<Rgb24>(imagePath);
            
            // 调整大小
            image.Mutate(x => x.Resize(_imageSize, _imageSize));
            
            // 转换为浮点数数组并归一化
            var tensor = new DenseTensor<float>(new[] { 1, 3, _imageSize, _imageSize });
            
            for (int y = 0; y < _imageSize; y++)
            {
                for (int x = 0; x < _imageSize; x++)
                {
                    var pixel = image[x, y];
                    
                    // 归一化: (pixel/255 - mean) / std
                    tensor[0, 0, y, x] = ((pixel.R / 255.0f) - _mean[0]) / _std[0];
                    tensor[0, 1, y, x] = ((pixel.G / 255.0f) - _mean[1]) / _std[1];
                    tensor[0, 2, y, x] = ((pixel.B / 255.0f) - _mean[2]) / _std[2];
                }
            }
            
            return tensor;
        }

        /// <summary>
        /// 执行推理
        /// </summary>
        /// <param name="inputTensor">输入张量</param>
        /// <returns>推理结果</returns>
        public M16PredictionResult Predict(DenseTensor<float> inputTensor)
        {
            // 准备输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // 执行推理
            using var results = _session.Run(inputs);

            // 解析输出
            var growthLevelOutput = results.First(x => x.Name == "growth_level").AsTensor<float>();
            var growthPatternOutput = results.First(x => x.Name == "growth_pattern").AsTensor<float>();
            var interferenceOutput = results.First(x => x.Name == "interference_factors").AsTensor<float>();
            var fineGrainedOutput = results.First(x => x.Name == "fine_grained").AsTensor<float>();

            return new M16PredictionResult
            {
                GrowthLevel = ProcessClassification(growthLevelOutput, _growthLevelClasses),
                GrowthPattern = ProcessClassification(growthPatternOutput, _growthPatternClasses),
                InterferenceFactors = ProcessMultilabel(interferenceOutput, _interferenceClasses),
                FineGrained = ProcessClassification(fineGrainedOutput, Enumerable.Range(0, 40).Select(i => $"class_{i}").ToArray())
            };
        }

        /// <summary>
        /// 处理分类输出
        /// </summary>
        private ClassificationResult ProcessClassification(DenseTensor<float> logits, string[] classes)
        {
            var probs = Softmax(logits);
            var maxProb = probs.Max();
            var predictedClass = Array.IndexOf(probs, maxProb);

            return new ClassificationResult
            {
                ClassId = predictedClass,
                ClassName = predictedClass < classes.Length ? classes[predictedClass] : $"class_{predictedClass}",
                Confidence = maxProb,
                Probabilities = probs
            };
        }

        /// <summary>
        /// 处理多标签输出
        /// </summary>
        private MultilabelResult ProcessMultilabel(DenseTensor<float> logits, string[] classes)
        {
            var probs = Sigmoid(logits);
            var activeClasses = new List<ActiveClass>();

            for (int i = 0; i < probs.Length; i++)
            {
                if (probs[i] > 0.5f)
                {
                    activeClasses.Add(new ActiveClass
                    {
                        ClassId = i,
                        ClassName = i < classes.Length ? classes[i] : $"class_{i}",
                        Confidence = probs[i]
                    });
                }
            }

            return new MultilabelResult
            {
                ActiveClasses = activeClasses,
                Probabilities = probs
            };
        }

        /// <summary>
        /// Softmax函数
        /// </summary>
        private float[] Softmax(DenseTensor<float> logits)
        {
            var max = logits.Max();
            var exp = logits.Select(x => MathF.Exp(x - max)).ToArray();
            var sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        /// <summary>
        /// Sigmoid函数
        /// </summary>
        private float[] Sigmoid(DenseTensor<float> logits)
        {
            return logits.Select(x => 1.0f / (1.0f + MathF.Exp(-x))).ToArray();
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    /// <summary>
    /// 预测结果
    /// </summary>
    public class M16PredictionResult
    {
        public ClassificationResult GrowthLevel { get; set; } = null!;
        public ClassificationResult GrowthPattern { get; set; } = null!;
        public MultilabelResult InterferenceFactors { get; set; } = null!;
        public ClassificationResult FineGrained { get; set; } = null!;
    }

    /// <summary>
    /// 分类结果
    /// </summary>
    public class ClassificationResult
    {
        public int ClassId { get; set; }
        public string ClassName { get; set; } = null!;
        public float Confidence { get; set; }
        public float[] Probabilities { get; set; } = null!;
    }

    /// <summary>
    /// 多标签结果
    /// </summary>
    public class MultilabelResult
    {
        public List<ActiveClass> ActiveClasses { get; set; } = new();
        public float[] Probabilities { get; set; } = null!;
    }

    /// <summary>
    /// 激活类别
    /// </summary>
    public class ActiveClass
    {
        public int ClassId { get; set; }
        public string ClassName { get; set; } = null!;
        public float Confidence { get; set; }
    }
}