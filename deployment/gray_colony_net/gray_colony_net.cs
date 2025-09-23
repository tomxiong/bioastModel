using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace BioAST.ColonyDetection
{
    /// <summary>
    /// 灰度菌落检测网络ONNX推理类
    /// 支持70×70灰度图像的三分类：
    /// - 阳性聚焦型菌落
    /// - 阴性气孔型（中空不规则边缘）
    /// - 弱生长小点气孔型
    /// </summary>
    public class gray_colony_net
    {
        private readonly InferenceSession _session;
        private readonly string[] _outputNames;
        
        /// <summary>
        /// 模型输入尺寸
        /// </summary>
        public const int InputWidth = 70;
        public const int InputHeight = 70;
        
        /// <summary>
        /// 初始化模型
        /// </summary>
        /// <param name="modelPath">ONNX模型文件路径</param>
        public gray_colony_net(string modelPath)
        {
            var options = new SessionOptions();
            // 根据需要选择执行提供程序
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            
            _session = new InferenceSession(modelPath, options);
            _outputNames = _session.OutputNames.ToArray();
        }
        
        /// <summary>
        /// 预处理灰度图像
        /// </summary>
        /// <param name="grayImage">灰度图像数据（70×70）</param>
        /// <returns>归一化的输入张量</returns>
        private DenseTensor<float> Preprocess(byte[,] grayImage)
        {
            if (grayImage.GetLength(0) != InputHeight || grayImage.GetLength(1) != InputWidth)
            {
                throw new ArgumentException($"图像尺寸必须是 {InputHeight}x{InputWidth}");
            }
            
            // 归一化到[0, 1]
            var input = new DenseTensor<float>(new[] { 1, 1, InputHeight, InputWidth });
            
            for (int y = 0; y < InputHeight; y++)
            {
                for (int x = 0; x < InputWidth; x++)
                {
                    input[0, 0, y, x] = grayImage[y, x] / 255.0f;
                }
            }
            
            return input;
        }
        
        /// <summary>
        /// 运行推理
        /// </summary>
        /// <param name="grayImage">灰度图像</param>
        /// <returns>分类结果</returns>
        public ColonyDetectionResult Predict(byte[,] grayImage)
        {
            // 预处理
            var input = Preprocess(grayImage);
            
            // 准备输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", input)
            };
            
            // 运行推理
            using var results = _session.Run(inputs);
            
            // 解析输出
            var positiveScore = results.First(x => x.Name == "positive_cluster").AsEnumerable<float>().First();
            var negativeScore = results.First(x => x.Name == "negative_pore").AsEnumerable<float>().First();
            var weakScore = results.First(x => x.Name == "weak_growth").AsEnumerable<float>().First();
            var confidence = results.First(x => x.Name == "confidence").AsEnumerable<float>().First();
            
            // 确定预测类别
            var maxScore = Math.Max(positiveScore, Math.Max(negativeScore, weakScore));
            string predictedClass;
            if (maxScore == positiveScore)
                predictedClass = "阳性聚焦型";
            else if (maxScore == negativeScore)
                predictedClass = "阴性气孔型";
            else
                predictedClass = "弱生长小点型";
            
            return new ColonyDetectionResult
            {
                PositiveClusterScore = positiveScore,
                NegativePoreScore = negativeScore,
                WeakGrowthScore = weakScore,
                Confidence = confidence,
                PredictedClass = predictedClass,
                MaxScore = maxScore
            };
        }
        
        /// <summary>
        /// 批量预测
        /// </summary>
        /// <param name="images">图像列表</param>
        /// <returns>预测结果列表</returns>
        public List<ColonyDetectionResult> PredictBatch(List<byte[,]> images)
        {
            return images.Select(Predict).ToList();
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
    /// 菌落检测结果
    /// </summary>
    public class ColonyDetectionResult
    {
        /// <summary>
        /// 阳性聚焦型菌落得分
        /// </summary>
        public float PositiveClusterScore { get; set; }
        
        /// <summary>
        /// 阴性气孔型得分
        /// </summary>
        public float NegativePoreScore { get; set; }
        
        /// <summary>
        /// 弱生长小点型得分
        /// </summary>
        public float WeakGrowthScore { get; set; }
        
        /// <summary>
        /// 预测置信度
        /// </summary>
        public float Confidence { get; set; }
        
        /// <summary>
        /// 预测类别
        /// </summary>
        public string PredictedClass { get; set; }
        
        /// <summary>
        /// 最高得分
        /// </summary>
        public float MaxScore { get; set; }
        
        /// <summary>
        /// 获取格式化的结果字符串
        /// </summary>
        public override string ToString()
        {
            return $"预测: {PredictedClass} (置信度: {Confidence:P2})
" +
                   $"  阳性聚焦型: {PositiveClusterScore:P2}
" +
                   $"  阴性气孔型: {NegativePoreScore:P2}
" +
                   $"  弱生长小点型: {WeakGrowthScore:P2}";
        }
    }
}
