using System;
using System.IO;
using BioAST.ColonyDetection;

namespace BioAST.Examples
{
    class Program
    {
        static void Main(string[] args)
        {
            // 初始化模型
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "gray_colony_net.onnx");
            using var detector = new gray_colony_net(modelPath);
            
            // 示例：加载70×70灰度图像
            var grayImage = LoadGrayImage("sample_colony.png");
            
            // 运行检测
            var result = detector.Predict(grayImage);
            
            // 输出结果
            Console.WriteLine("菌落检测结果:");
            Console.WriteLine(result);
            
            // 判断结果
            if (result.PredictedClass == "阳性聚焦型" && result.Confidence > 0.8)
            {
                Console.WriteLine("检测到阳性菌落，建议进一步培养观察");
            }
            else if (result.PredictedClass == "阴性气孔型")
            {
                Console.WriteLine("检测到气孔，可忽略");
            }
            else if (result.PredictedClass == "弱生长小点型")
            {
                Console.WriteLine("检测到弱生长，建议延长培养时间");
            }
        }
        
        static byte[,] LoadGrayImage(string imagePath)
        {
            // 这里实现图像加载和转换为70×70灰度的逻辑
            // 实际应用中需要根据具体需求实现
            var image = new byte[70, 70];
            // ... 加载和处理图像
            return image;
        }
    }
}
