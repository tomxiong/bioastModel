using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Runtime.InteropServices;

namespace BioastMicClassification
{
    /// <summary>
    /// 核心边界优化多任务MIC MobileNetV3模型推理类
    /// Core Boundary Optimization Multitask MIC MobileNetV3 Inference
    /// </summary>
    public class MultitaskMicClassifier : IDisposable
    {
        private InferenceSession _session;
        private readonly string _modelPath;
        private readonly ModelInfo _modelInfo;
        private readonly LabelMappings _labelMappings;
        private bool _disposed = false;

        // 模型输入输出名称
        private const string INPUT_NAME = "image";
        private const string OUTPUT_CLASSIFICATION = "classification";
        private const string OUTPUT_GROWTH_PATTERN = "growth_pattern";
        private const string OUTPUT_INTERFERENCE = "interference_factors";

        // 预处理参数
        private const int INPUT_SIZE = 70;
        private const float MEAN = 0.485f;
        private const float STD = 0.229f;

        public MultitaskMicClassifier(string modelPath, string configDir = null)
        {
            _modelPath = modelPath;
            
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"ONNX模型文件不存在: {modelPath}");
            }

            // 加载模型配置
            string configPath = configDir ?? Path.GetDirectoryName(modelPath);
            _modelInfo = LoadModelInfo(Path.Combine(configPath, "model_info.json"));
            _labelMappings = LoadLabelMappings(Path.Combine(configPath, "label_mappings.json"));

            // 初始化ONNX Runtime
            InitializeOnnxSession();
        }

        /// <summary>
        /// 初始化ONNX推理会话
        /// </summary>
        private void InitializeOnnxSession()
        {
            try
            {
                var options = new SessionOptions
                {
                    EnableCpuMemArena = true,
                    EnableMemoryPattern = true,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                };

                // 如果有GPU，启用GPU推理
                if (HasCuda())
                {
                    options.AppendExecutionProvider_CUDA(0);
                    Console.WriteLine("✅ 使用GPU推理");
                }
                else
                {
                    Console.WriteLine("⚠️ 使用CPU推理");
                }

                _session = new InferenceSession(_modelPath, options);
                Console.WriteLine($"✅ ONNX模型加载成功: {_modelPath}");
                
                // 打印模型信息
                PrintModelInfo();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"ONNX模型初始化失败: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// 检查是否支持CUDA
        /// </summary>
        private bool HasCuda()
        {
            try
            {
                return OrtEnv.Instance().GetAvailableProviders().Contains("CUDAExecutionProvider");
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// 打印模型信息
        /// </summary>
        private void PrintModelInfo()
        {
            Console.WriteLine($"📊 模型信息:");
            Console.WriteLine($"   名称: {_modelInfo.ModelName}");
            Console.WriteLine($"   版本: {_modelInfo.Version}");
            Console.WriteLine($"   准确率: {_modelInfo.Performance.Accuracy}");
            Console.WriteLine($"   输入尺寸: {INPUT_SIZE}x{INPUT_SIZE} (灰度图)");
            Console.WriteLine($"   生长模式类别: {_labelMappings.GrowthPattern.Count}");
            Console.WriteLine($"   干扰因素类别: {_labelMappings.InterferenceFactors.Count}");
        }

        /// <summary>
        /// 从图片文件进行推理
        /// </summary>
        public async Task<ClassificationResult> ClassifyAsync(string imagePath)
        {
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"图片文件不存在: {imagePath}");
            }

            using var bitmap = new Bitmap(imagePath);
            return await ClassifyAsync(bitmap);
        }

        /// <summary>
        /// 从Bitmap进行推理
        /// </summary>
        public async Task<ClassificationResult> ClassifyAsync(Bitmap image)
        {
            return await Task.Run(() => Classify(image));
        }

        /// <summary>
        /// 同步推理方法
        /// </summary>
        public ClassificationResult Classify(Bitmap image)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(MultitaskMicClassifier));
            }

            try
            {
                // 预处理图像
                var inputTensor = PreprocessImage(image);

                // 创建输入
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(INPUT_NAME, inputTensor)
                };

                // 运行推理
                using var outputs = _session.Run(inputs);
                
                // 解析输出
                return ParseOutputs(outputs);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"推理过程发生错误: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// 图像预处理
        /// </summary>
        private DenseTensor<float> PreprocessImage(Bitmap image)
        {
            // 转换为灰度图并调整大小
            using var resized = ResizeToGrayscale(image, INPUT_SIZE, INPUT_SIZE);
            
            // 创建输入张量 [1, 1, 70, 70]
            var tensor = new DenseTensor<float>(new[] { 1, 1, INPUT_SIZE, INPUT_SIZE });
            
            // 锁定位图数据
            var bitmapData = resized.LockBits(
                new Rectangle(0, 0, INPUT_SIZE, INPUT_SIZE),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);

            try
            {
                unsafe
                {
                    byte* ptr = (byte*)bitmapData.Scan0.ToPointer();
                    int stride = bitmapData.Stride;

                    for (int y = 0; y < INPUT_SIZE; y++)
                    {
                        for (int x = 0; x < INPUT_SIZE; x++)
                        {
                            // 获取像素值 (BGR格式)
                            byte* pixel = ptr + y * stride + x * 3;
                            byte b = pixel[0];
                            byte g = pixel[1];
                            byte r = pixel[2];

                            // 转换为灰度值
                            float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
                            
                            // 标准化
                            float normalized = (gray - MEAN) / STD;
                            
                            // 设置张量值 [batch, channel, height, width]
                            tensor[0, 0, y, x] = normalized;
                        }
                    }
                }
            }
            finally
            {
                resized.UnlockBits(bitmapData);
            }

            return tensor;
        }

        /// <summary>
        /// 调整图像大小并转换为灰度图
        /// </summary>
        private Bitmap ResizeToGrayscale(Bitmap original, int width, int height)
        {
            var resized = new Bitmap(width, height);
            using var graphics = Graphics.FromImage(resized);
            
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBilinear;
            graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            
            graphics.DrawImage(original, 0, 0, width, height);
            
            return resized;
        }

        /// <summary>
        /// 解析模型输出
        /// </summary>
        private ClassificationResult ParseOutputs(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
        {
            var result = new ClassificationResult();

            foreach (var output in outputs)
            {
                switch (output.Name)
                {
                    case OUTPUT_CLASSIFICATION:
                        var classificationTensor = output.AsTensor<float>();
                        var classProbs = classificationTensor.ToArray();
                        result.Classification = new ClassificationOutput
                        {
                            NegativeProbability = classProbs[0],
                            PositiveProbability = classProbs[1],
                            PredictedClass = classProbs[1] > classProbs[0] ? "positive" : "negative",
                            Confidence = Math.Max(classProbs[0], classProbs[1])
                        };
                        break;

                    case OUTPUT_GROWTH_PATTERN:
                        var growthTensor = output.AsTensor<float>();
                        var growthProbs = growthTensor.ToArray();
                        result.GrowthPattern = ParseGrowthPattern(growthProbs);
                        break;

                    case OUTPUT_INTERFERENCE:
                        var interferenceTensor = output.AsTensor<float>();
                        var interferenceProbs = interferenceTensor.ToArray();
                        result.InterferenceFactors = ParseInterferenceFactors(interferenceProbs);
                        break;
                }
            }

            return result;
        }

        /// <summary>
        /// 解析生长模式输出
        /// </summary>
        private GrowthPatternOutput ParseGrowthPattern(float[] probabilities)
        {
            var patterns = _labelMappings.GrowthPattern.ToList();
            var maxIndex = Array.IndexOf(probabilities, probabilities.Max());
            
            return new GrowthPatternOutput
            {
                PredictedPattern = patterns[maxIndex].Key,
                Confidence = probabilities[maxIndex],
                AllProbabilities = patterns.ToDictionary(
                    kvp => kvp.Key, 
                    kvp => probabilities[kvp.Value]
                )
            };
        }

        /// <summary>
        /// 解析干扰因素输出
        /// </summary>
        private InterferenceFactorsOutput ParseInterferenceFactors(float[] probabilities)
        {
            var factors = _labelMappings.InterferenceFactors.ToList();
            var detectedFactors = new List<string>();
            var allProbabilities = new Dictionary<string, float>();

            for (int i = 0; i < factors.Count && i < probabilities.Length; i++)
            {
                var factor = factors[i].Key;
                var prob = probabilities[i];
                allProbabilities[factor] = prob;
                
                // 使用0.5作为阈值判断是否存在该干扰因素
                if (prob > 0.5f)
                {
                    detectedFactors.Add(factor);
                }
            }

            return new InterferenceFactorsOutput
            {
                DetectedFactors = detectedFactors,
                AllProbabilities = allProbabilities,
                HasPores = allProbabilities.GetValueOrDefault("pores", 0) > 0.5f
            };
        }

        /// <summary>
        /// 加载模型信息
        /// </summary>
        private ModelInfo LoadModelInfo(string path)
        {
            if (!File.Exists(path))
            {
                Console.WriteLine($"⚠️ 模型信息文件不存在: {path}，使用默认配置");
                return new ModelInfo { ModelName = "MultitaskMIC_MobileNetV3", Version = "1.0" };
            }

            var json = File.ReadAllText(path);
            return JsonConvert.DeserializeObject<ModelInfo>(json);
        }

        /// <summary>
        /// 加载标签映射
        /// </summary>
        private LabelMappings LoadLabelMappings(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"标签映射文件不存在: {path}");
            }

            var json = File.ReadAllText(path);
            return JsonConvert.DeserializeObject<LabelMappings>(json);
        }

        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
            }
        }
    }

    #region Data Models

    /// <summary>
    /// 分类结果
    /// </summary>
    public class ClassificationResult
    {
        public ClassificationOutput Classification { get; set; }
        public GrowthPatternOutput GrowthPattern { get; set; }
        public InterferenceFactorsOutput InterferenceFactors { get; set; }
    }

    /// <summary>
    /// 主分类输出
    /// </summary>
    public class ClassificationOutput
    {
        public float NegativeProbability { get; set; }
        public float PositiveProbability { get; set; }
        public string PredictedClass { get; set; }
        public float Confidence { get; set; }
    }

    /// <summary>
    /// 生长模式输出
    /// </summary>
    public class GrowthPatternOutput
    {
        public string PredictedPattern { get; set; }
        public float Confidence { get; set; }
        public Dictionary<string, float> AllProbabilities { get; set; }
    }

    /// <summary>
    /// 干扰因素输出
    /// </summary>
    public class InterferenceFactorsOutput
    {
        public List<string> DetectedFactors { get; set; }
        public Dictionary<string, float> AllProbabilities { get; set; }
        public bool HasPores { get; set; }
    }

    /// <summary>
    /// 模型信息
    /// </summary>
    public class ModelInfo
    {
        [JsonProperty("model_name")]
        public string ModelName { get; set; }
        
        [JsonProperty("version")]
        public string Version { get; set; }
        
        [JsonProperty("description")]
        public string Description { get; set; }
        
        [JsonProperty("performance")]
        public PerformanceInfo Performance { get; set; }
    }

    /// <summary>
    /// 性能信息
    /// </summary>
    public class PerformanceInfo
    {
        [JsonProperty("accuracy")]
        public string Accuracy { get; set; }
        
        [JsonProperty("optimized_for")]
        public string OptimizedFor { get; set; }
    }

    /// <summary>
    /// 标签映射
    /// </summary>
    public class LabelMappings
    {
        [JsonProperty("growth_pattern")]
        public Dictionary<string, int> GrowthPattern { get; set; }
        
        [JsonProperty("interference_factors")]
        public Dictionary<string, int> InterferenceFactors { get; set; }
        
        [JsonProperty("microbe_type")]
        public Dictionary<string, int> MicrobeType { get; set; }
    }

    #endregion
}