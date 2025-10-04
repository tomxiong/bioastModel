using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace BioastOnnxInference
{
    public class Program
    {
        // Label mappings based on training dataset
        private static readonly Dictionary<string, string[]> LabelMappings = new()
        {
            { "growth_level", new[] { "negative", "positive" } },
            {
                "growth_pattern",
                new[] {
                    "center_dots", "clean", "clustered", "even_scattered",
                    "heavy_growth", "negative", "weak_scattered",
                    "weak_scattered_neg", "weak_scattered_pos", "unclear"
                }
            },
            { "interference_factors", new[] { "pores", "artifacts", "debris", "contamination" } }
        };

        // Optimized thresholds from v0.11.0 evaluation
        private static readonly Dictionary<string, float> OptimalThresholds = new()
        {
            { "pores", 0.40f },
            { "artifacts", 0.45f },
            { "debris", 0.15f },
            { "contamination", 0.50f }
        };

        public static void Main(string[] args)
        {
            // Check if image path provided
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: BioastOnnxInference <path_to_image>");
                Console.WriteLine("Example: BioastOnnxInference sample_image.png");
                return;
            }

            string imagePath = args[0];
            string modelPath = "../../../model.onnx"; // Adjust path to your model location

            try
            {
                // Run inference
                var predictor = new BioastPredictor(modelPath);
                var result = predictor.Predict(imagePath);

                // Display results
                Console.WriteLine("\n" + new string('=', 80));
                Console.WriteLine("MobileNetV4 v0.11.0 Inference Results");
                Console.WriteLine(new string('=', 80));

                Console.WriteLine("\n[Growth Level]");
                Console.WriteLine($"  Prediction: {result.GrowthLevel.Label} (confidence: {result.GrowthLevel.Confidence:P2})");
                Console.WriteLine($"  Probabilities: negative={result.GrowthLevel.Probabilities[0]:F4}, positive={result.GrowthLevel.Probabilities[1]:F4}");

                Console.WriteLine("\n[Growth Pattern]");
                Console.WriteLine($"  Prediction: {result.GrowthPattern.Label} (confidence: {result.GrowthPattern.Confidence:P2})");
                Console.WriteLine("  Top 3 probabilities:");
                var top3 = result.GrowthPattern.Probabilities
                    .Select((prob, idx) => (Label: LabelMappings["growth_pattern"][idx], Prob: prob))
                    .OrderByDescending(x => x.Prob)
                    .Take(3);
                foreach (var (label, prob) in top3)
                {
                    Console.WriteLine($"    {label}: {prob:F4}");
                }

                Console.WriteLine("\n[Interference Factors]");
                foreach (var factor in result.InterferenceFactors)
                {
                    string status = factor.IsPresent ? "DETECTED" : "NOT DETECTED";
                    Console.WriteLine($"  {factor.Name}: {status} (score: {factor.Score:F4}, threshold: {factor.Threshold:F2})");
                }

                Console.WriteLine("\n" + new string('=', 80));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error during inference: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }

    public class BioastPredictor
    {
        private readonly InferenceSession _session;

        public BioastPredictor(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"ONNX model not found at: {modelPath}");
            }

            // Create inference session
            var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _session = new InferenceSession(modelPath, sessionOptions);

            Console.WriteLine("Model loaded successfully!");
            Console.WriteLine($"  Input: {_session.InputMetadata.First().Key}");
            Console.WriteLine($"  Outputs: {string.Join(", ", _session.OutputMetadata.Keys)}");
        }

        public PredictionResult Predict(string imagePath)
        {
            // Load and preprocess image
            var inputTensor = PreprocessImage(imagePath);

            // Prepare input
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // Run inference
            using var results = _session.Run(inputs);

            // Parse outputs
            var growthLevelOutput = results.First(r => r.Name == "growth_level").AsEnumerable<float>().ToArray();
            var growthPatternOutput = results.First(r => r.Name == "growth_pattern").AsEnumerable<float>().ToArray();
            var interferenceOutput = results.First(r => r.Name == "interference_factors").AsEnumerable<float>().ToArray();

            return new PredictionResult
            {
                GrowthLevel = ProcessGrowthLevel(growthLevelOutput),
                GrowthPattern = ProcessGrowthPattern(growthPatternOutput),
                InterferenceFactors = ProcessInterferenceFactors(interferenceOutput)
            };
        }

        private Tensor<float> PreprocessImage(string imagePath)
        {
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"Image not found at: {imagePath}");
            }

            // Load image and convert to grayscale
            using var image = Image.Load<Rgb24>(imagePath);

            // Resize to 70x70
            image.Mutate(x => x.Resize(70, 70));

            // Convert to grayscale and normalize to [0, 1]
            var tensor = new DenseTensor<float>(new[] { 1, 1, 70, 70 });

            for (int y = 0; y < 70; y++)
            {
                for (int x = 0; x < 70; x++)
                {
                    var pixel = image[x, y];
                    // Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
                    float gray = (0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B) / 255.0f;
                    tensor[0, 0, y, x] = gray;
                }
            }

            return tensor;
        }

        private ClassificationOutput ProcessGrowthLevel(float[] logits)
        {
            // Apply sigmoid to convert logits to probabilities
            var probs = Sigmoid(logits);

            int predictedClass = probs[1] > 0.5f ? 1 : 0;

            return new ClassificationOutput
            {
                Label = Program.LabelMappings["growth_level"][predictedClass],
                Confidence = probs[predictedClass],
                Probabilities = probs
            };
        }

        private ClassificationOutput ProcessGrowthPattern(float[] logits)
        {
            // Apply softmax to convert logits to probabilities
            var probs = Softmax(logits);

            int predictedClass = Array.IndexOf(probs, probs.Max());

            return new ClassificationOutput
            {
                Label = Program.LabelMappings["growth_pattern"][predictedClass],
                Confidence = probs[predictedClass],
                Probabilities = probs
            };
        }

        private List<InterferenceFactorOutput> ProcessInterferenceFactors(float[] logits)
        {
            // Apply sigmoid for multi-label classification
            var probs = Sigmoid(logits);

            var results = new List<InterferenceFactorOutput>();
            var factorNames = Program.LabelMappings["interference_factors"];

            for (int i = 0; i < factorNames.Length; i++)
            {
                string factorName = factorNames[i];
                float threshold = Program.OptimalThresholds[factorName];

                results.Add(new InterferenceFactorOutput
                {
                    Name = factorName,
                    Score = probs[i],
                    Threshold = threshold,
                    IsPresent = probs[i] >= threshold
                });
            }

            return results;
        }

        private float[] Sigmoid(float[] x)
        {
            return x.Select(v => 1.0f / (1.0f + MathF.Exp(-v))).ToArray();
        }

        private float[] Softmax(float[] x)
        {
            var exps = x.Select(v => MathF.Exp(v - x.Max())).ToArray();
            var sum = exps.Sum();
            return exps.Select(v => v / sum).ToArray();
        }
    }

    public class PredictionResult
    {
        public ClassificationOutput GrowthLevel { get; set; } = null!;
        public ClassificationOutput GrowthPattern { get; set; } = null!;
        public List<InterferenceFactorOutput> InterferenceFactors { get; set; } = null!;
    }

    public class ClassificationOutput
    {
        public string Label { get; set; } = string.Empty;
        public float Confidence { get; set; }
        public float[] Probabilities { get; set; } = Array.Empty<float>();
    }

    public class InterferenceFactorOutput
    {
        public string Name { get; set; } = string.Empty;
        public float Score { get; set; }
        public float Threshold { get; set; }
        public bool IsPresent { get; set; }
    }
}
