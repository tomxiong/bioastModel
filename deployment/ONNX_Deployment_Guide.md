# ONNX 模型部署指南

本指南提供了如何在不同平台上部署和使用我们转换的 ONNX 模型的详细说明。

## 目录

1. [模型概述](#模型概述)
2. [Python 部署](#python-部署)
3. [C# 部署](#c-部署)
4. [JavaScript 部署](#javascript-部署)
5. [移动端部署](#移动端部署)
   - [Android](#android)
   - [iOS](#ios)
6. [性能优化建议](#性能优化建议)
7. [常见问题解答](#常见问题解答)

## 模型概述

我们已经将以下模型转换为 ONNX 格式：

| 模型名称 | 大小 | 输出形状 | 主要用途 |
|---------|------|---------|---------|
| airbubble_hybrid_net | 0.39 MB | (1, 2) | 气泡检测与分类 |
| simplified_airbubble_detector | 0.53 MB | (1, 2) | 简化版气泡检测 |
| enhanced_airbubble_detector | 2.89 MB | (1, 2) | 增强版气泡检测 |
| mic_mobilenetv3 | 4.34 MB | (1, 1, 3, 3) | 微生物菌落识别 |
| efficientnet_b0 | 5.93 MB | (1, 2) | 高效菌落分类 |
| micro_vit | 8.08 MB | (1, 196, 1) | 微型视觉转换器 |
| vit_tiny | 10.43 MB | (1, 2) | 轻量级视觉转换器 |
| resnet18_improved | 42.98 MB | (1, 2) | 改进版残差网络 |
| coatnet | 99.41 MB | (1, 2) | 卷积注意力混合网络 |
| convnext_tiny | 106.22 MB | (1, 2) | 下一代卷积网络 |

所有模型都接受 RGB 图像作为输入，输入尺寸为 `(1, 3, 70, 70)`，表示批次大小为 1，3 个颜色通道，图像高度和宽度均为 70 像素。

## Python 部署

### 安装依赖

```bash
pip install onnxruntime numpy pillow
```

### 示例代码

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(70, 70)):
    """预处理图像"""
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    
    # 调整大小
    img = img.resize(target_size)
    
    # 转换为 numpy 数组并归一化
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # 转换为 NCHW 格式 (批次大小, 通道数, 高度, 宽度)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def run_inference(model_path, image_path):
    """运行推理"""
    # 加载 ONNX 模型
    session = ort.InferenceSession(model_path)
    
    # 预处理图像
    input_data = preprocess_image(image_path)
    
    # 获取输入名称
    input_name = session.get_inputs()[0].name
    
    # 运行推理
    outputs = session.run(None, {input_name: input_data})
    
    return outputs[0]

# 示例使用
model_path = "deployment/onnx_models/efficientnet_b0.onnx"
image_path = "path/to/your/image.jpg"
result = run_inference(model_path, image_path)
print(f"推理结果: {result}")
print(f"预测类别: {np.argmax(result)}")
```

### 批量处理

```python
import os
import glob
import pandas as pd

def batch_inference(model_path, image_dir, output_csv):
    """批量处理目录中的所有图像"""
    # 加载模型
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # 获取所有图像文件
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                 glob.glob(os.path.join(image_dir, "*.png"))
    
    results = []
    
    # 处理每个图像
    for image_path in image_paths:
        # 预处理图像
        input_data = preprocess_image(image_path)
        
        # 运行推理
        outputs = session.run(None, {input_name: input_data})
        
        # 保存结果
        results.append({
            "image": os.path.basename(image_path),
            "prediction": np.argmax(outputs[0]),
            "confidence": np.max(outputs[0]),
            "raw_output": outputs[0].tolist()
        })
    
    # 保存为 CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"结果已保存到 {output_csv}")

# 示例使用
model_path = "deployment/onnx_models/efficientnet_b0.onnx"
image_dir = "path/to/your/images"
output_csv = "results.csv"
batch_inference(model_path, image_dir, output_csv)
```

## C# 部署

### 安装依赖

在 Visual Studio 中，通过 NuGet 包管理器安装以下包：
- Microsoft.ML.OnnxRuntime
- SixLabors.ImageSharp

### 示例代码

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxModelInference
{
    class Program
    {
        static void Main(string[] args)
        {
            // 模型路径
            string modelPath = @"deployment\onnx_models\efficientnet_b0.onnx";
            
            // 图像路径
            string imagePath = @"path\to\your\image.jpg";
            
            // 运行推理
            float[] result = RunInference(modelPath, imagePath);
            
            // 输出结果
            Console.WriteLine($"推理结果: [{string.Join(", ", result)}]");
            Console.WriteLine($"预测类别: {Array.IndexOf(result, result.Max())}");
        }
        
        static float[] RunInference(string modelPath, string imagePath)
        {
            // 加载并预处理图像
            var tensor = PreprocessImage(imagePath);
            
            // 创建输入
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };
            
            // 创建推理会话
            using var session = new InferenceSession(modelPath);
            
            // 运行推理
            using var results = session.Run(inputs);
            
            // 获取输出
            var output = results.First().AsTensor<float>();
            
            return output.ToArray();
        }
        
        static DenseTensor<float> PreprocessImage(string imagePath)
        {
            // 加载图像
            using var image = Image.Load<Rgb24>(imagePath);
            
            // 调整大小
            image.Mutate(x => x.Resize(70, 70));
            
            // 创建输入张量
            var tensor = new DenseTensor<float>(new[] { 1, 3, 70, 70 });
            
            // 填充张量数据
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    var pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        tensor[0, 0, y, x] = pixelRow[x].R / 255.0f;
                        tensor[0, 1, y, x] = pixelRow[x].G / 255.0f;
                        tensor[0, 2, y, x] = pixelRow[x].B / 255.0f;
                    }
                }
            });
            
            return tensor;
        }
    }
}
```

## JavaScript 部署

### 安装依赖

```bash
npm install onnxruntime-web
```

### HTML 和 JavaScript 示例

```html
<!DOCTYPE html>
<html>
<head>
    <title>ONNX 模型推理</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
</head>
<body>
    <h1>ONNX 模型推理示例</h1>
    
    <div>
        <input type="file" id="imageUpload" accept="image/*">
        <button id="runButton">运行推理</button>
    </div>
    
    <div>
        <h3>预览：</h3>
        <canvas id="preview" width="70" height="70"></canvas>
    </div>
    
    <div>
        <h3>结果：</h3>
        <pre id="result"></pre>
    </div>
    
    <script>
        // 设置 ONNX Runtime Web 的 wasm 路径
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        // 获取 DOM 元素
        const imageUpload = document.getElementById('imageUpload');
        const runButton = document.getElementById('runButton');
        const preview = document.getElementById('preview');
        const result = document.getElementById('result');
        const ctx = preview.getContext('2d');
        
        // 预处理图像
        function preprocessImage(img) {
            // 绘制图像到 canvas
            ctx.drawImage(img, 0, 0, 70, 70);
            
            // 获取图像数据
            const imageData = ctx.getImageData(0, 0, 70, 70);
            const { data } = imageData;
            
            // 创建输入张量 (NCHW 格式)
            const tensor = new Float32Array(1 * 3 * 70 * 70);
            
            // 填充张量数据
            for (let y = 0; y < 70; y++) {
                for (let x = 0; x < 70; x++) {
                    const pixelIndex = (y * 70 + x) * 4;
                    tensor[0 * 70 * 70 + y * 70 + x] = data[pixelIndex] / 255.0;     // R
                    tensor[1 * 70 * 70 + y * 70 + x] = data[pixelIndex + 1] / 255.0; // G
                    tensor[2 * 70 * 70 + y * 70 + x] = data[pixelIndex + 2] / 255.0; // B
                }
            }
            
            return tensor;
        }
        
        // 运行推理
        async function runInference(tensor) {
            try {
                // 加载模型
                const session = await ort.InferenceSession.create('model/simplified_airbubble_detector.onnx');
                
                // 创建输入
                const input = new ort.Tensor('float32', tensor, [1, 3, 70, 70]);
                
                // 运行推理
                const outputs = await session.run({ input });
                
                // 获取输出
                const outputData = outputs.output.data;
                
                // 显示结果
                result.textContent = `原始输出: [${Array.from(outputData).join(', ')}]\n`;
                result.textContent += `预测类别: ${outputData[0] > outputData[1] ? 0 : 1}`;
            } catch (error) {
                result.textContent = `错误: ${error.message}`;
                console.error(error);
            }
        }
        
        // 处理图像上传
        imageUpload.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                const img = new Image();
                img.onload = () => {
                    ctx.clearRect(0, 0, 70, 70);
                    ctx.drawImage(img, 0, 0, 70, 70);
                };
                img.src = URL.createObjectURL(file);
            }
        });
        
        // 处理运行按钮点击
        runButton.addEventListener('click', async () => {
            if (imageUpload.files.length > 0) {
                const file = imageUpload.files[0];
                const img = new Image();
                img.onload = () => {
                    const tensor = preprocessImage(img);
                    runInference(tensor);
                };
                img.src = URL.createObjectURL(file);
            } else {
                result.textContent = '请先上传图像';
            }
        });
    </script>
</body>
</html>
```

## 移动端部署

### Android

#### 安装依赖

在 `build.gradle` 文件中添加：

```gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.12.1'
}
```

#### 示例代码

```java
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try {
            // 初始化 ONNX Runtime
            ortEnvironment = OrtEnvironment.getEnvironment();
            
            // 从 assets 复制模型到内部存储
            File modelFile = new File(getFilesDir(), "model.onnx");
            copyAssetToFile("simplified_airbubble_detector.onnx", modelFile);
            
            // 创建会话
            ortSession = ortEnvironment.createSession(modelFile.getAbsolutePath());
            
            // 加载示例图像
            Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("sample.jpg"));
            
            // 预处理图像
            FloatBuffer inputBuffer = preprocessImage(bitmap);
            
            // 运行推理
            float[] result = runInference(inputBuffer);
            
            // 处理结果
            int predictedClass = result[0] > result[1] ? 0 : 1;
            System.out.println("预测类别: " + predictedClass);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private FloatBuffer preprocessImage(Bitmap bitmap) {
        // 调整大小
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 70, 70, true);
        
        // 创建输入缓冲区
        FloatBuffer buffer = FloatBuffer.allocate(3 * 70 * 70);
        
        // 填充数据
        int[] pixels = new int[70 * 70];
        resizedBitmap.getPixels(pixels, 0, 70, 0, 0, 70, 70);
        
        for (int y = 0; y < 70; y++) {
            for (int x = 0; x < 70; x++) {
                int pixel = pixels[y * 70 + x];
                buffer.put(0 * 70 * 70 + y * 70 + x, ((pixel >> 16) & 0xFF) / 255.0f); // R
                buffer.put(1 * 70 * 70 + y * 70 + x, ((pixel >> 8) & 0xFF) / 255.0f);  // G
                buffer.put(2 * 70 * 70 + y * 70 + x, (pixel & 0xFF) / 255.0f);         // B
            }
        }
        
        return buffer;
    }

    private float[] runInference(FloatBuffer inputBuffer) throws Exception {
        // 创建输入张量
        OnnxTensor inputTensor = OnnxTensor.createTensor(
                ortEnvironment,
                inputBuffer,
                new long[]{1, 3, 70, 70}
        );
        
        // 运行推理
        Map<String, OnnxTensor> inputs = Collections.singletonMap("input", inputTensor);
        OrtSession.Result result = ortSession.run(inputs);
        
        // 获取输出
        float[][] output = (float[][]) result.get(0).getValue();
        
        return output[0];
    }

    private void copyAssetToFile(String assetName, File outFile) throws IOException {
        try (InputStream in = getAssets().open(assetName);
             FileOutputStream out = new FileOutputStream(outFile)) {
            byte[] buffer = new byte[1024];
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            ortSession.close();
            ortEnvironment.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### iOS

#### 安装依赖

在 `Podfile` 中添加：

```ruby
pod 'onnxruntime-objc'
```

#### 示例代码

```swift
import UIKit
import onnxruntime_objc

class ViewController: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        do {
            // 加载模型
            guard let modelPath = Bundle.main.path(forResource: "simplified_airbubble_detector", ofType: "onnx") else {
                print("模型文件未找到")
                return
            }
            
            // 创建会话
            let session = try ORTSession(path: modelPath)
            
            // 加载示例图像
            guard let image = UIImage(named: "sample.jpg") else {
                print("图像未找到")
                return
            }
            
            // 预处理图像
            let inputData = preprocessImage(image)
            
            // 创建输入
            let inputShape: [NSNumber] = [1, 3, 70, 70]
            let inputTensor = try ORTValue(tensorData: inputData, elementType: .float, shape: inputShape)
            
            // 运行推理
            let inputs = ["input": inputTensor]
            let outputs = try session.run(withInputs: inputs, outputNames: ["output"], runOptions: nil)
            
            // 处理结果
            if let outputTensor = outputs["output"] {
                let outputData = try outputTensor.tensorData() as Data
                let result = outputData.withUnsafeBytes { (pointer: UnsafeRawBufferPointer) -> [Float] in
                    let floatBuffer = pointer.bindMemory(to: Float.self)
                    return Array(floatBuffer)
                }
                
                let predictedClass = result[0] > result[1] ? 0 : 1
                print("预测类别: \(predictedClass)")
            }
            
        } catch {
            print("错误: \(error)")
        }
    }
    
    func preprocessImage(_ image: UIImage) -> Data {
        // 调整大小
        let size = CGSize(width: 70, height: 70)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        
        // 获取像素数据
        let cgImage = resizedImage.cgImage!
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerRow = width * 4
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: rgbColorSpace, bitmapInfo: bitmapInfo.rawValue)!
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        let pixelData = context.data!
        
        // 创建输入数据
        var inputData = Data(count: 3 * 70 * 70 * MemoryLayout<Float>.size)
        inputData.withUnsafeMutableBytes { (inputBuffer: UnsafeMutableRawBufferPointer) -> Void in
            let floatBuffer = inputBuffer.bindMemory(to: Float.self)
            let pixels = pixelData.bindMemory(to: UInt8.self)
            
            for y in 0..<height {
                for x in 0..<width {
                    let pixelIndex = (y * width + x) * 4
                    let r = Float(pixels[pixelIndex]) / 255.0
                    let g = Float(pixels[pixelIndex + 1]) / 255.0
                    let b = Float(pixels[pixelIndex + 2]) / 255.0
                    
                    floatBuffer[0 * height * width + y * width + x] = r
                    floatBuffer[1 * height * width + y * width + x] = g
                    floatBuffer[2 * height * width + y * width + x] = b
                }
            }
        }
        
        return inputData
    }
}
```

## 性能优化建议

1. **模型量化**
   - 使用 ONNX Runtime 的量化工具将模型从 FP32 转换为 INT8，可以显著减小模型大小并提高推理速度
   - 示例代码：
     ```python
     import onnx
     from onnxruntime.quantization import quantize_dynamic, QuantType
     
     # 加载模型
     model_path = "deployment/onnx_models/convnext_tiny.onnx"
     quantized_model_path = "deployment/onnx_models/convnext_tiny_quantized.onnx"
     
     # 量化模型
     quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QInt8)
     ```

2. **批量处理**
   - 当需要处理多个图像时，使用批量处理而不是逐个处理
   - 这可以更好地利用硬件资源，提高吞吐量

3. **使用 GPU 加速**
   - 在支持 GPU 的平台上，使用 ONNX Runtime 的 GPU 执行提供程序
   - 示例代码：
     ```python
     import onnxruntime as ort
     
     # 创建 GPU 会话
     session_options = ort.SessionOptions()
     session = ort.InferenceSession(
         "model.onnx",
         sess_options=session_options,
         providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
     )
     ```

4. **模型剪枝**
   - 对于较大的模型（如 `convnext_tiny` 和 `coatnet`），可以考虑使用模型剪枝技术减小模型大小
   - 这需要在转换为 ONNX 之前在 PyTorch 中进行

5. **输入优化**
   - 预处理图像时，尽可能在 CPU 上进行，以减少数据传输开销
   - 考虑使用内存映射文件进行大批量数据处理

## 常见问题解答

### Q: 如何选择合适的模型？

A: 根据您的应用场景和资源限制选择合适的模型：
- 对于资源受限的设备（如移动设备），推荐使用较小的模型，如 `airbubble_hybrid_net`、`simplified_airbubble_detector` 或 `mic_mobilenetv3`
- 对于需要高精度的场景，推荐使用较大的模型，如 `resnet18_improved`、`coatnet` 或 `convnext_tiny`
- 对于平衡性能和资源的场景，推荐使用中等大小的模型，如 `efficientnet_b0`、`micro_vit` 或 `vit_tiny`

### Q: 模型输出如何解释？

A: 大多数模型的输出是一个长度为 2 的数组，表示两个类别的概率或分数。通常，索引 0 表示阴性（无菌落），索引 1 表示阳性（有菌落）。使用 `argmax` 函数找出得分最高的类别。

### Q: 如何处理不同尺寸的输入图像？

A: 所有模型都期望输入尺寸为 70x70 像素的 RGB 图像。如果您的图像尺寸不同，需要在预处理阶段将其调整为 70x70 像素。

### Q: 如何提高模型在特定场景下的性能？

A: 如果模型在特定场景下表现不佳，可以考虑以下方法：
1. 使用更适合该场景的模型
2. 在原始 PyTorch 模型上使用特定场景的数据进行微调，然后再转换为 ONNX
3. 调整预处理步骤，例如改变图像归一化方法或增加数据增强

### Q: 如何处理 ONNX 模型加载错误？

A: 常见的 ONNX 模型加载错误包括：
1. 版本不兼容：确保您使用的 ONNX Runtime 版本与模型兼容
2. 内存不足：对于较大的模型，确保设备有足够的内存
3. 操作不支持：某些平台可能不支持模型中使用的某些操作，可以尝试使用 ONNX 的简化工具来解决这个问题