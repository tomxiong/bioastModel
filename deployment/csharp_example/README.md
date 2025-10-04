# BioAst ONNX Inference - C# Example

Complete C# console application demonstrating how to use the MobileNetV4 v0.11.0 ONNX model for inference.

## 🎯 Features

- ✅ Load MobileNetV4 v0.11.0 ONNX model
- ✅ Preprocess 70×70 grayscale images
- ✅ Multi-task inference (Growth Level + Growth Pattern + Interference Factors)
- ✅ Optimized thresholds for interference detection
- ✅ Detailed probability outputs

## 📋 Prerequisites

- **.NET 6.0 SDK** or higher
- ONNX model file: `model.onnx` (MobileNetV4 v0.11.0)

## 🛠️ Installation

### 1. Clone or Copy the Project

```bash
cd deployment/csharp_example/BioastOnnxInference
```

### 2. Restore NuGet Packages

```bash
dotnet restore
```

The project uses the following NuGet packages:
- **Microsoft.ML.OnnxRuntime** (v1.16.3) - ONNX inference engine
- **SixLabors.ImageSharp** (v3.0.2) - Image processing

## 🚀 Usage

### Build the Project

```bash
dotnet build
```

### Run Inference

```bash
dotnet run <path_to_image>
```

**Example**:
```bash
dotnet run sample_colony.png
```

### Expected Output

```
Model loaded successfully!
  Input: input
  Outputs: growth_level, growth_pattern, interference_factors

================================================================================
MobileNetV4 v0.11.0 Inference Results
================================================================================

[Growth Level]
  Prediction: positive (confidence: 98.54%)
  Probabilities: negative=0.0146, positive=0.9854

[Growth Pattern]
  Prediction: clustered (confidence: 89.23%)
  Top 3 probabilities:
    clustered: 0.8923
    heavy_growth: 0.0612
    even_scattered: 0.0254

[Interference Factors]
  pores: NOT DETECTED (score: 0.1234, threshold: 0.40)
  artifacts: NOT DETECTED (score: 0.0567, threshold: 0.45)
  debris: DETECTED (score: 0.6789, threshold: 0.15)
  contamination: NOT DETECTED (score: 0.0012, threshold: 0.50)

================================================================================
```

## 📁 Project Structure

```
BioastOnnxInference/
├── BioastOnnxInference.csproj    # Project file with NuGet dependencies
├── Program.cs                     # Main application code
└── README.md                      # This file
```

## 🔧 Key Components

### BioastPredictor Class

Main inference class with the following methods:

- `PreprocessImage(string imagePath)` - Loads and preprocesses image to 70×70 grayscale
- `Predict(string imagePath)` - Runs inference and returns results
- `ProcessGrowthLevel(float[] logits)` - Applies sigmoid for binary classification
- `ProcessGrowthPattern(float[] logits)` - Applies softmax for 10-class classification
- `ProcessInterferenceFactors(float[] logits)` - Applies sigmoid with optimized thresholds

### Image Preprocessing

1. **Load Image**: Supports PNG, JPEG, BMP formats
2. **Resize**: 70×70 pixels
3. **Grayscale Conversion**: RGB → Gray (0.299R + 0.587G + 0.114B)
4. **Normalization**: [0, 255] → [0.0, 1.0]
5. **Tensor Format**: [1, 1, 70, 70] (batch=1, channels=1, height=70, width=70)

### Output Processing

#### Growth Level (Binary Classification)
- **Labels**: `negative`, `positive`
- **Activation**: Sigmoid
- **Threshold**: 0.5

#### Growth Pattern (10-Class Classification)
- **Labels**: `center_dots`, `clean`, `clustered`, `even_scattered`, `heavy_growth`, `negative`, `weak_scattered`, `weak_scattered_neg`, `weak_scattered_pos`, `unclear`
- **Activation**: Softmax
- **Selection**: argmax

#### Interference Factors (Multi-Label Classification)
- **Labels**: `pores`, `artifacts`, `debris`, `contamination`
- **Activation**: Sigmoid
- **Optimized Thresholds**:
  - pores: 0.40
  - artifacts: 0.45
  - debris: 0.15
  - contamination: 0.50

## 🎨 Customization

### Adjust Model Path

In `Program.cs`, modify the model path:

```csharp
string modelPath = "path/to/your/model.onnx";
```

### Change Thresholds

Modify the `OptimalThresholds` dictionary in `Program.cs`:

```csharp
private static readonly Dictionary<string, float> OptimalThresholds = new()
{
    { "pores", 0.40f },        // Adjust as needed
    { "artifacts", 0.45f },
    { "debris", 0.15f },
    { "contamination", 0.50f }
};
```

### Enable GPU Acceleration (Optional)

Install the CUDA provider:

```bash
dotnet add package Microsoft.ML.OnnxRuntime.Gpu
```

Update session options in `BioastPredictor` constructor:

```csharp
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider_CUDA(0);  // Use GPU 0
sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
```

## 📊 Performance

Based on ONNX benchmarking results:

- **Inference Time**: ~1.75 ms per image (CPU)
- **Throughput**: ~570 images/second (batch=1)
- **Model Size**: 3.69 MB
- **Precision**: Max difference vs PyTorch < 1e-6

## 🧪 Testing

### Create a Test Script

```csharp
// Example test with multiple images
var predictor = new BioastPredictor("model.onnx");

foreach (var imagePath in Directory.GetFiles("test_images", "*.png"))
{
    Console.WriteLine($"\nProcessing: {imagePath}");
    var result = predictor.Predict(imagePath);
    Console.WriteLine($"Growth Level: {result.GrowthLevel.Label}");
    Console.WriteLine($"Growth Pattern: {result.GrowthPattern.Label}");
}
```

## 🐛 Troubleshooting

### "ONNX model not found"
- Ensure `model.onnx` is in the correct path
- Check the `modelPath` variable in `Program.cs`

### "Image not found"
- Verify the image path is correct
- Ensure the image format is supported (PNG, JPEG, BMP)

### Low inference accuracy
- Verify input preprocessing matches training (grayscale, 70×70, [0,1] normalization)
- Check that label mappings match your dataset
- Validate thresholds are appropriate for your use case

### Out of memory errors
- Process images one at a time (batch=1)
- Use smaller image batches
- Consider enabling GPU acceleration

## 📚 Additional Resources

- **Model Info**: `../model_info.json`
- **Performance Analysis**: `/home/aaa/ws/bioastModel/V0.11.0_EVALUATION_SUMMARY.md`
- **ONNX Runtime Docs**: https://onnxruntime.ai/docs/
- **ImageSharp Docs**: https://docs.sixlabors.com/

## 📝 License

This example code is provided as-is for demonstration purposes.

---

**Model Version**: MobileNetV4 v0.11.0
**Architecture**: Universal Inverted Bottleneck + SE/ECA Attention
**Parameters**: 952,201
**Test Accuracy**: 94.26% (Total), 98.53% (Growth Level), 87.31% (Growth Pattern), 96.93% (Interference)
