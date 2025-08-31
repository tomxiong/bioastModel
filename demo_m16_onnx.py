#!/usr/bin/env python3
"""
M16多任务ONNX模型使用演示
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import json
from pathlib import Path

class SimpleM16Inference:
    """简化的M16多任务推理类"""
    
    def __init__(self, model_path: str):
        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 类别定义
        self.growth_level_classes = ['negative', 'positive', 'weak_growth']
        self.growth_pattern_classes = ['clean', 'clustered', 'scattered', 'heavy_growth', 
                                      'small_dots', 'irregular_areas', 'light_gray', 
                                      'default_positive', 'default_weak_growth']
        self.interference_classes = ['pores', 'debris', 'artifacts']
    
    def predict(self, image_path: str):
        """预测图像"""
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).numpy()
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 处理结果
        results = {
            'growth_level': self._process_classification(outputs[0][0], self.growth_level_classes),
            'growth_pattern': self._process_classification(outputs[1][0], self.growth_pattern_classes),
            'interference_factors': self._process_multilabel(outputs[2][0], self.interference_classes),
            'fine_grained': self._process_classification(outputs[3][0], list(range(40)))
        }
        
        return results
    
    def _process_classification(self, logits, classes):
        """处理分类输出"""
        probs = self.softmax(logits)
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        return {
            'class_id': pred_class,
            'class_name': classes[pred_class] if pred_class < len(classes) else f"class_{pred_class}",
            'confidence': confidence
        }
    
    def _process_multilabel(self, logits, classes):
        """处理多标签输出"""
        probs = self.sigmoid(logits)
        predictions = (probs > 0.5).astype(int)
        
        active_classes = []
        for i, pred in enumerate(predictions):
            if pred == 1 and i < len(classes):
                active_classes.append({
                    'class_id': i,
                    'class_name': classes[i],
                    'confidence': float(probs[i])
                })
        
        return {'active_classes': active_classes}
    
    def softmax(self, x):
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def sigmoid(self, x):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-x))

def print_results(results):
    """Print results"""
    print("=== M16 Multi-task Classification Results ===")
    
    gl = results['growth_level']
    print(f"Growth level: {gl['class_name']} (confidence: {gl['confidence']:.3f})")
    
    gp = results['growth_pattern']
    print(f"Growth pattern: {gp['class_name']} (confidence: {gp['confidence']:.3f})")
    
    if_factors = results['interference_factors']['active_classes']
    if if_factors:
        factor_names = [f['class_name'] for f in if_factors]
        print(f"Interference factors: {', '.join(factor_names)}")
    else:
        print("Interference factors: None")
    
    fg = results['fine_grained']
    print(f"Fine-grained ID: {fg['class_id']} (confidence: {fg['confidence']:.3f})")
    
    # Simple explanation
    print("\n=== Simple Explanation ===")
    if gl['class_name'] == 'negative':
        print("No colony growth detected")
    elif gl['class_name'] == 'positive':
        print(f"Colony growth detected, pattern: {gp['class_name']}")
    else:
        print(f"Weak growth detected, pattern: {gp['class_name']}")
    
    if if_factors:
        print(f"Warning: Interference factors present: {', '.join(factor_names)}")

def main():
    """Main function"""
    print("M16 Multi-task ONNX Model Demo")
    print("=" * 50)
    
    # Check model file
    model_path = "onnx_models/m16_multitask_mobilenetv3.onnx"
    if not Path(model_path).exists():
        print(f"Error: Model file does not exist: {model_path}")
        return
    
    # 初始化推理器
    try:
        inference = SimpleM16Inference(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # 显示模型信息
    print(f"Input size: {inference.session.get_inputs()[0].shape}")
    print(f"Output tasks: {len(inference.output_names)}")
    print(f"Supported classes:")
    print(f"  - Growth levels: {len(inference.growth_level_classes)} classes")
    print(f"  - Growth patterns: {len(inference.growth_pattern_classes)} classes")
    print(f"  - Interference factors: {len(inference.interference_classes)} classes")
    print(f"  - Fine-grained: 40 classes")
    
    print("\n" + "=" * 50)
    print("Usage:")
    print("1. Place image files in the project directory")
    print("2. Enter image path to predict")
    print("3. View detailed classification results")
    print("=" * 50)
    
    # Interactive prediction
    while True:
        try:
            image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
            
            if image_path.lower() == 'quit':
                break
            
            if not image_path:
                continue
            
            if not Path(image_path).exists():
                print(f"Error: File does not exist: {image_path}")
                continue
            
            # Predict
            print(f"\nAnalyzing image: {image_path}")
            results = inference.predict(image_path)
            print_results(results)
            
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    print("\nThank you for using M16 multi-task classification system!")

if __name__ == "__main__":
    main()