#!/usr/bin/env python3
"""
灰度菌落网络ONNX部署工具
包含模型导出、验证和C#部署示例
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import json
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gray_colony_net import create_gray_colony_net, export_to_onnx


class ONNXModelValidator:
    """ONNX模型验证器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ort_session = None
        
    def load_model(self):
        """加载ONNX模型"""
        try:
            self.ort_session = ort.InferenceSession(self.model_path)
            print(f"✓ 成功加载ONNX模型: {self.model_path}")
            
            # 打印模型信息
            input_info = self.ort_session.get_inputs()[0]
            output_info = self.ort_session.get_outputs()
            
            print(f"  输入: {input_info.name} - {input_info.shape} - {input_info.type}")
            for output in output_info:
                print(f"  输出: {output.name} - {output.shape} - {output.type}")
                
            return True
        except Exception as e:
            print(f"✗ 加载ONNX模型失败: {e}")
            return False
    
    def validate_with_pytorch(self, pytorch_model, test_input: torch.Tensor):
        """与PyTorch模型对比验证"""
        if not self.ort_session:
            if not self.load_model():
                return False
        
        print("\n=== ONNX模型验证 ===")
        
        # PyTorch推理
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_outputs = pytorch_model.get_onnx_compatible_output(test_input)
        
        # ONNX推理
        ort_inputs = {self.ort_session.get_inputs()[0].name: test_input.numpy()}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        
        # 获取输出名称
        output_names = [output.name for output in self.ort_session.get_outputs()]
        
        # 比较结果
        print("\n输出对比:")
        max_diff = 0
        for i, (name, torch_out) in enumerate(pytorch_outputs.items()):
            onnx_out = torch.tensor(ort_outputs[i])
            diff = torch.abs(torch_out - onnx_out).max().item()
            max_diff = max(max_diff, diff)
            
            print(f"  {name}:")
            print(f"    PyTorch shape: {torch_out.shape}")
            print(f"    ONNX shape: {onnx_out.shape}")
            print(f"    最大差异: {diff:.6f}")
        
        print(f"\n最大差异: {max_diff:.6f}")
        if max_diff < 1e-3:
            print("✓ ONNX模型验证通过")
            return True
        else:
            print("✗ ONNX模型验证失败")
            return False


class ONNXDeploymentGenerator:
    """ONNX部署代码生成器"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model_name = self.model_path.stem
        
    def generate_csharp_wrapper(self, output_dir: Path):
        """生成C#包装器"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 主类文件
        cs_file = output_dir / f"{self.model_name}.cs"
        
        cs_code = f"""using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace BioAST.ColonyDetection
{{
    /// <summary>
    /// 灰度菌落检测网络ONNX推理类
    /// 支持70×70灰度图像的三分类：
    /// - 阳性聚焦型菌落
    /// - 阴性气孔型（中空不规则边缘）
    /// - 弱生长小点气孔型
    /// </summary>
    public class {self.model_name}
    {{
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
        public {self.model_name}(string modelPath)
        {{
            var options = new SessionOptions();
            // 根据需要选择执行提供程序
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            
            _session = new InferenceSession(modelPath, options);
            _outputNames = _session.OutputNames.ToArray();
        }}
        
        /// <summary>
        /// 预处理灰度图像
        /// </summary>
        /// <param name="grayImage">灰度图像数据（70×70）</param>
        /// <returns>归一化的输入张量</returns>
        private DenseTensor<float> Preprocess(byte[,] grayImage)
        {{
            if (grayImage.GetLength(0) != InputHeight || grayImage.GetLength(1) != InputWidth)
            {{
                throw new ArgumentException($"图像尺寸必须是 {{InputHeight}}x{{InputWidth}}");
            }}
            
            // 归一化到[0, 1]
            var input = new DenseTensor<float>(new[] {{ 1, 1, InputHeight, InputWidth }});
            
            for (int y = 0; y < InputHeight; y++)
            {{
                for (int x = 0; x < InputWidth; x++)
                {{
                    input[0, 0, y, x] = grayImage[y, x] / 255.0f;
                }}
            }}
            
            return input;
        }}
        
        /// <summary>
        /// 运行推理
        /// </summary>
        /// <param name="grayImage">灰度图像</param>
        /// <returns>分类结果</returns>
        public ColonyDetectionResult Predict(byte[,] grayImage)
        {{
            // 预处理
            var input = Preprocess(grayImage);
            
            // 准备输入
            var inputs = new List<NamedOnnxValue>
            {{
                NamedOnnxValue.CreateFromTensor("input", input)
            }};
            
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
            {{
                PositiveClusterScore = positiveScore,
                NegativePoreScore = negativeScore,
                WeakGrowthScore = weakScore,
                Confidence = confidence,
                PredictedClass = predictedClass,
                MaxScore = maxScore
            }};
        }}
        
        /// <summary>
        /// 批量预测
        /// </summary>
        /// <param name="images">图像列表</param>
        /// <returns>预测结果列表</returns>
        public List<ColonyDetectionResult> PredictBatch(List<byte[,]> images)
        {{
            return images.Select(Predict).ToList();
        }}
        
        /// <summary>
        /// 释放资源
        /// </summary>
        public void Dispose()
        {{
            _session?.Dispose();
        }}
    }}
    
    /// <summary>
    /// 菌落检测结果
    /// </summary>
    public class ColonyDetectionResult
    {{
        /// <summary>
        /// 阳性聚焦型菌落得分
        /// </summary>
        public float PositiveClusterScore {{ get; set; }}
        
        /// <summary>
        /// 阴性气孔型得分
        /// </summary>
        public float NegativePoreScore {{ get; set; }}
        
        /// <summary>
        /// 弱生长小点型得分
        /// </summary>
        public float WeakGrowthScore {{ get; set; }}
        
        /// <summary>
        /// 预测置信度
        /// </summary>
        public float Confidence {{ get; set; }}
        
        /// <summary>
        /// 预测类别
        /// </summary>
        public string PredictedClass {{ get; set; }}
        
        /// <summary>
        /// 最高得分
        /// </summary>
        public float MaxScore {{ get; set; }}
        
        /// <summary>
        /// 获取格式化的结果字符串
        /// </summary>
        public override string ToString()
        {{
            return $"预测: {{PredictedClass}} (置信度: {{Confidence:P2}})\n" +
                   $"  阳性聚焦型: {{PositiveClusterScore:P2}}\n" +
                   $"  阴性气孔型: {{NegativePoreScore:P2}}\n" +
                   $"  弱生长小点型: {{WeakGrowthScore:P2}}";
        }}
    }}
}}
"""
        
        with open(cs_file, 'w', encoding='utf-8') as f:
            f.write(cs_code)
        
        print(f"✓ C#包装器已生成: {cs_file}")
        
        # 项目文件
        proj_file = output_dir / f"{self.model_name}.csproj"
        proj_content = f"""<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>net6.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.16.0" />
  </ItemGroup>

</Project>
"""
        
        with open(proj_file, 'w', encoding='utf-8') as f:
            f.write(proj_content)
        
        print(f"✓ 项目文件已生成: {proj_file}")
        
        # 使用示例
        example_file = output_dir / "UsageExample.cs"
        example_content = f"""using System;
using System.IO;
using BioAST.ColonyDetection;

namespace BioAST.Examples
{{
    class Program
    {{
        static void Main(string[] args)
        {{
            // 初始化模型
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "{self.model_name}.onnx");
            using var detector = new {self.model_name}(modelPath);
            
            // 示例：加载70×70灰度图像
            var grayImage = LoadGrayImage("sample_colony.png");
            
            // 运行检测
            var result = detector.Predict(grayImage);
            
            // 输出结果
            Console.WriteLine("菌落检测结果:");
            Console.WriteLine(result);
            
            // 判断结果
            if (result.PredictedClass == "阳性聚焦型" && result.Confidence > 0.8)
            {{
                Console.WriteLine("检测到阳性菌落，建议进一步培养观察");
            }}
            else if (result.PredictedClass == "阴性气孔型")
            {{
                Console.WriteLine("检测到气孔，可忽略");
            }}
            else if (result.PredictedClass == "弱生长小点型")
            {{
                Console.WriteLine("检测到弱生长，建议延长培养时间");
            }}
        }}
        
        static byte[,] LoadGrayImage(string imagePath)
        {{
            // 这里实现图像加载和转换为70×70灰度的逻辑
            // 实际应用中需要根据具体需求实现
            var image = new byte[70, 70];
            // ... 加载和处理图像
            return image;
        }}
    }}
}}
"""
        
        with open(example_file, 'w', encoding='utf-8') as f:
            f.write(example_content)
        
        print(f"✓ 使用示例已生成: {example_file}")
        
        # README
        readme_file = output_dir / "README.md"
        readme_content = f"""# {self.model_name} C#部署指南

## 概述
本目录包含灰度菌落检测网络的C#部署文件，支持ONNX运行时推理。

## 文件说明
- `{self.model_name}.cs`: 主要的推理类
- `{self.model_name}.csproj`: 项目配置文件
- `UsageExample.cs`: 使用示例
- `{self.model_name}.onnx`: ONNX模型文件（需要从上级目录复制）

## 使用步骤

### 1. 环境要求
- .NET 6.0 或更高版本
- ONNX Runtime 1.16.0 或更高版本

### 2. 安装依赖
```bash
dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.0
```

### 3. 基本使用
```csharp
using BioAST.ColonyDetection;

// 初始化检测器
string modelPath = @"path\to\{self.model_name}.onnx";
using var detector = new {self.model_name}(modelPath);

// 加载70×70灰度图像
byte[,] grayImage = LoadYourImage();

// 运行检测
var result = detector.Predict(grayImage);

// 输出结果
Console.WriteLine($"预测类别: {{result.PredictedClass}}");
Console.WriteLine($"置信度: {{result.Confidence:P2}}");
```

## 输入输出说明

### 输入
- 图像尺寸：70×70像素
- 图像格式：灰度（单通道）
- 数值范围：0-255

### 输出
- PositiveClusterScore: 阳性聚焦型菌落得分（0-1）
- NegativePoreScore: 阴性气孔型得分（0-1）
- WeakGrowthScore: 弱生长小点型得分（0-1）
- Confidence: 预测置信度（0-1）
- PredictedClass: 预测类别名称

## 性能优化建议
1. 使用批量处理提高吞吐量
2. 考虑使用GPU加速（安装Microsoft.ML.OnnxRuntime.Gpu包）
3. 预处理图像时使用并行处理

## 注意事项
- 确保输入图像已正确缩放到70×70
- 图像应为真正的灰度图像，不是RGB转换的
- 模型对中空结构（气孔）有特殊优化
"""
        
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✓ README已生成: {readme_file}")


def main():
    """主函数：导出并验证ONNX模型"""
    print("=== 灰度菌落网络ONNX部署 ===")
    
    # 1. 创建模型
    print("\n1. 创建PyTorch模型...")
    model = create_gray_colony_net(num_classes=3, model_size='base')
    model_info = model.get_model_info()
    print(f"   模型: {model_info['model_name']}")
    print(f"   参数量: {model_info['total_parameters']:,}")
    
    # 2. 生成测试数据
    print("\n2. 准备测试数据...")
    test_input = torch.randn(1, 1, 70, 70)
    
    # 3. 导出ONNX
    print("\n3. 导出ONNX模型...")
    model_path = "gray_colony_net.onnx"
    export_to_onnx(model, model_path)
    
    # 4. 验证ONNX模型
    print("\n4. 验证ONNX模型...")
    validator = ONNXModelValidator(model_path)
    if validator.validate_with_pytorch(model, test_input):
        print("✓ ONNX模型验证成功")
    else:
        print("✗ ONNX模型验证失败")
        return
    
    # 5. 生成部署代码
    print("\n5. 生成部署代码...")
    output_dir = Path("deployment/gray_colony_net")
    generator = ONNXDeploymentGenerator(model_path)
    generator.generate_csharp_wrapper(output_dir)
    
    # 6. 复制ONNX模型到部署目录
    import shutil
    deployment_model_path = output_dir / model_path
    shutil.copy2(model_path, deployment_model_path)
    print(f"✓ 模型已复制到: {deployment_model_path}")
    
    # 7. 生成配置文件
    config = {
        "model_info": model_info,
        "input_requirements": {
            "width": 70,
            "height": 70,
            "channels": 1,
            "format": "grayscale",
            "value_range": [0, 255]
        },
        "output_classes": [
            {"name": "positive_cluster", "description": "阳性聚焦型菌落"},
            {"name": "negative_pore", "description": "阴性气孔型（中空不规则边缘）"},
            {"name": "weak_growth", "description": "弱生长小点气孔型"}
        ],
        "preprocessing": {
            "normalization": "divide by 255",
            "target_size": [70, 70]
        },
        "deployment": {
            "onnx_runtime": "1.16.0+",
            "supported_platforms": ["Windows", "Linux", "macOS", "Android", "iOS"],
            "recommended_batch_size": 1
        }
    }
    
    config_path = output_dir / "model_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ 配置文件已生成: {config_path}")
    
    print(f"\n=== 完成 ===")
    print(f"所有文件已生成到: {output_dir.absolute()}")
    print(f"\n部署包包含:")
    print(f"  - ONNX模型文件")
    print(f"  - C#包装器类")
    print(f"  - 项目配置文件")
    print(f"  - 使用示例代码")
    print(f"  - 部署文档")
    print(f"  - 模型配置文件")


if __name__ == "__main__":
    main()