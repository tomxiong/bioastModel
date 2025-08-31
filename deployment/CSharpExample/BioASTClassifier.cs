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
    /// EfficientNetV2-S ONNX模型的生物抗菌素敏感性测试分类器
    /// </summary>
    public class BioASTClassifier : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly string _outputName;
        
        // ImageNet标准化参数
        private readonly float[] _mean = { 0.485f, 0.456f, 0.406f };
        private readonly float[] _std = { 0.229f, 0.224f, 0.225f };
        
        /// <summary>
        /// 初始化分类器
        /// </summary>
        /// <param name="modelPath">ONNX模型文件路径</param>
        public BioASTClassifier(string modelPath)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"模型文件不存在: {modelPath}");
            
            try
            {
                // 创建ONNX Runtime会话
                _session = new InferenceSession(modelPath);
                
                // 获取输入输出名称
                _inputName = _session.InputMetadata.Keys.First();
                _outputName = _session.OutputMetadata.Keys.First();
                
                Console.WriteLine($"模型加载成功: {Path.GetFileName(modelPath)}");
                Console.WriteLine($"输入名称: {_inputName}");
                Console.WriteLine($"输出名称: {_outputName}");
                
                // 打印模型信息
                var inputMeta = _session.InputMetadata[_inputName];
                Console.WriteLine($"输入维度: [{string.Join(", ", inputMeta.Dimensions)}]");
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"模型加载失败: {ex.Message}", ex);
            }
        }
        
        /// <summary>
        /// 预测单张图像
        /// </summary>
        /// <param name="imagePath">图像文件路径</param>
        /// <returns>预测结果</returns>
        public PredictionResult Predict(string imagePath)
        {
            if (!File.Exists(imagePath))
                throw new FileNotFoundException($"图像文件不存在: {imagePath}");
            
            try
            {
                using var image = Image.Load<Rgb24>(imagePath);
                return Predict(image);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"图像预测失败: {ex.Message}", ex);
            }
        }
        
        /// <summary>
        /// 预测图像对象
        /// </summary>
        /// <param name="image">图像对象</param>
        /// <returns>预测结果</returns>
        public PredictionResult Predict(Image<Rgb24> image)
        {
            var startTime = DateTime.UtcNow;
            
            try
            {
                // 1. 图像预处理
                var tensor = PreprocessImage(image);
                
                // 2. 创建输入
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_inputName, tensor)
                };
                
                // 3. 运行推理
                using var results = _session.Run(inputs);
                
                // 4. 处理输出
                var output = results.First().AsEnumerable<float>().ToArray();
                
                // 5. 应用Softmax获取概率
                var probabilities = Softmax(output);
                
                // 6. 获取预测类别
                var predictedClass = probabilities[0] > probabilities[1] ? 0 : 1;
                var confidence = Math.Max(probabilities[0], probabilities[1]);
                
                var inferenceTime = DateTime.UtcNow - startTime;
                
                return new PredictionResult
                {
                    PredictedClass = predictedClass,
                    ClassName = predictedClass == 0 ? "Benign" : "Malignant",
                    Confidence = confidence,
                    BenignProbability = probabilities[0],
                    MalignantProbability = probabilities[1],
                    InferenceTimeMs = inferenceTime.TotalMilliseconds
                };
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"模型推理失败: {ex.Message}", ex);
            }
        }
        
        /// <summary>
        /// 批量预测
        /// </summary>
        /// <param name="imagePaths">图像文件路径数组</param>
        /// <returns>预测结果列表</returns>
        public List<(string ImagePath, PredictionResult Result)> PredictBatch(string[] imagePaths)
        {
            var results = new List<(string, PredictionResult)>();
            
            foreach (var imagePath in imagePaths)
            {
                try
                {
                    var result = Predict(imagePath);
                    results.Add((imagePath, result));
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"预测失败 {imagePath}: {ex.Message}");
                    // 添加失败结果
                    results.Add((imagePath, new PredictionResult
                    {
                        PredictedClass = -1,
                        ClassName = "Error",
                        Confidence = 0,
                        BenignProbability = 0,
                        MalignantProbability = 0,
                        InferenceTimeMs = 0
                    }));
                }
            }
            
            return results;
        }
        
        /// <summary>
        /// 异步批量预测
        /// </summary>
        /// <param name="imagePaths">图像文件路径数组</param>
        /// <returns>预测结果列表</returns>
        public async Task<List<(string ImagePath, PredictionResult Result)>> PredictBatchAsync(string[] imagePaths)
        {
            var tasks = imagePaths.Select(async imagePath =>
            {
                return await Task.Run(() => 
                {
                    try
                    {
                        var result = Predict(imagePath);
                        return (imagePath, result);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"异步预测失败 {imagePath}: {ex.Message}");
                        return (imagePath, new PredictionResult
                        {
                            PredictedClass = -1,
                            ClassName = "Error",
                            Confidence = 0,
                            BenignProbability = 0,
                            MalignantProbability = 0,
                            InferenceTimeMs = 0
                        });
                    }
                });
            });
            
            var results = await Task.WhenAll(tasks);
            return results.ToList();
        }
        
        /// <summary>
        /// 图像预处理 - 关键步骤！
        /// </summary>
        /// <param name="image">输入图像</param>
        /// <returns>预处理后的张量</returns>
        private DenseTensor<float> PreprocessImage(Image<Rgb24> image)
        {
            // 1. 调整图像尺寸到70x70
            image.Mutate(x => x.Resize(70, 70));
            
            // 2. 创建张量 [1, 3, 70, 70]
            var tensor = new DenseTensor<float>(new[] { 1, 3, 70, 70 });
            
            // 3. 转换像素值并标准化
            for (int y = 0; y < 70; y++)
            {
                for (int x = 0; x < 70; x++)
                {
                    var pixel = image[x, y];
                    
                    // 转换到[0,1]范围
                    float r = pixel.R / 255.0f;
                    float g = pixel.G / 255.0f;
                    float b = pixel.B / 255.0f;
                    
                    // ImageNet标准化
                    tensor[0, 0, y, x] = (r - _mean[0]) / _std[0]; // Red channel
                    tensor[0, 1, y, x] = (g - _mean[1]) / _std[1]; // Green channel
                    tensor[0, 2, y, x] = (b - _mean[2]) / _std[2]; // Blue channel
                }
            }
            
            return tensor;
        }
        
        /// <summary>
        /// Softmax函数
        /// </summary>
        /// <param name="values">输入值</param>
        /// <returns>概率分布</returns>
        private float[] Softmax(float[] values)
        {
            var max = values.Max();
            var exp = values.Select(v => Math.Exp(v - max)).ToArray();
            var sum = exp.Sum();
            return exp.Select(e => (float)(e / sum)).ToArray();
        }
        
        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            _session?.Dispose();
        }
    }
    
    /// <summary>
    /// 预测结果类
    /// </summary>
    public class PredictionResult
    {
        /// <summary>
        /// 预测类别 (0=Benign, 1=Malignant)
        /// </summary>
        public int PredictedClass { get; set; }
        
        /// <summary>
        /// 类别名称
        /// </summary>
        public string ClassName { get; set; } = string.Empty;
        
        /// <summary>
        /// 置信度 (0-1)
        /// </summary>
        public float Confidence { get; set; }
        
        /// <summary>
        /// Benign概率
        /// </summary>
        public float BenignProbability { get; set; }
        
        /// <summary>
        /// Malignant概率
        /// </summary>
        public float MalignantProbability { get; set; }
        
        /// <summary>
        /// 推理时间(毫秒)
        /// </summary>
        public double InferenceTimeMs { get; set; }
        
        /// <summary>
        /// 是否为阳性结果
        /// </summary>
        public bool IsMalignant => PredictedClass == 1;
        
        /// <summary>
        /// 是否为阴性结果
        /// </summary>
        public bool IsBenign => PredictedClass == 0;
        
        public override string ToString()
        {
            return $"预测: {ClassName} (置信度: {Confidence:P2}, Benign: {BenignProbability:P2}, Malignant: {MalignantProbability:P2}, 推理时间: {InferenceTimeMs:F2}ms)";
        }
    }
}