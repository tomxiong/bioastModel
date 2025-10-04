using System;

namespace BioastOnnxInference
{
    /// <summary>
    /// 数据集验证程序
    /// 用于验证 ONNX 模型在测试集上的性能
    /// </summary>
    public class ValidateProgram
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("BioAst ONNX Model - Dataset Validation Tool");
            Console.WriteLine("============================================\n");

            // 配置参数
            string modelPath = args.Length > 0 ? args[0] : "../../../model.onnx";
            string dataRoot = "../../../../../../ds/images";  // 相对于 bin/Debug/net6.0
            string annotationsFile = "../../../../../../ds/images/m9e1n170_cleaned_round2.json";
            string splitFile = "../../../../../../ds/images/dataset_split_seed44.json";
            string outputPath = "validation_results.json";

            Console.WriteLine($"模型路径: {modelPath}");
            Console.WriteLine($"数据根目录: {dataRoot}");
            Console.WriteLine($"标注文件: {annotationsFile}");
            Console.WriteLine($"划分文件: {splitFile}");
            Console.WriteLine();

            try
            {
                // 创建验证器
                var validator = new DatasetValidator(
                    modelPath: modelPath,
                    dataRoot: dataRoot,
                    annotationsFile: annotationsFile,
                    splitFile: splitFile
                );

                // 运行验证
                var result = validator.ValidateTestSet();

                // 打印结果
                DatasetValidator.PrintResults(result);

                // 导出结果
                DatasetValidator.ExportResults(result, outputPath);

                Console.WriteLine("\n验证完成!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n错误: {ex.Message}");
                Console.WriteLine($"堆栈跟踪: {ex.StackTrace}");
                Environment.Exit(1);
            }
        }
    }
}
