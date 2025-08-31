#!/usr/bin/env python3
"""
m16多任务MobileNetV3模型ONNX转换脚本
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional

# 导入模型和数据集
from models.enhanced_multitask_mobilenetv3 import create_enhanced_multitask_mobilenetv3, get_class_definitions
from enhanced_multitask_ni_dataset import EnhancedMultiTaskNIDataset

class M16MultitaskONNXConverter:
    """m16多任务模型ONNX转换器"""
    
    def __init__(self, model_path: str, output_dir: str = "onnx_models"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 加载模型配置
        self.model_config = self._load_model_config()
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载模型权重
        self._load_model_weights()
        
    def _load_model_config(self) -> Dict:
        """加载模型配置"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        return checkpoint.get('config', {
            'growth_level_classes': 3,
            'growth_pattern_classes': 9,
            'interference_classes': 3,
            'fine_grained_classes': 40,
            'width_mult': 1.0,
            'dropout_rate': 0.2
        })
    
    def _create_model(self):
        """创建模型"""
        model = create_enhanced_multitask_mobilenetv3(
            growth_level_classes=self.model_config.get('growth_level_classes', 3),
            growth_pattern_classes=self.model_config.get('growth_pattern_classes', 9),
            interference_classes=self.model_config.get('interference_classes', 3),
            fine_grained_classes=self.model_config.get('fine_grained_classes', 40),
            width_mult=self.model_config.get('width_mult', 1.0),
            dropout_rate=self.model_config.get('dropout_rate', 0.2)
        )
        model.eval()
        return model
    
    def _load_model_weights(self):
        """加载模型权重"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"模型权重加载成功: {self.model_path}")
    
    def convert_to_onnx(self, 
                      input_shape: Tuple[int, int, int, int] = (1, 3, 70, 70),
                      opset_version: int = 14) -> str:
        """转换为ONNX格式"""
        
        self.logger.info("开始ONNX转换...")
        
        # 创建示例输入
        dummy_input = torch.randn(*input_shape)
        
        # 定义输出路径
        output_path = self.output_dir / "m16_multitask_mobilenetv3.onnx"
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=[
                'growth_level',
                'growth_pattern', 
                'interference_factors',
                'fine_grained'
            ],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'growth_level': {0: 'batch_size'},
                'growth_pattern': {0: 'batch_size'},
                'interference_factors': {0: 'batch_size'},
                'fine_grained': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"ONNX模型已保存到: {output_path}")
        
        # 验证ONNX模型
        self._validate_onnx_model(str(output_path), dummy_input)
        
        return str(output_path)
    
    def _validate_onnx_model(self, onnx_path: str, test_input: torch.Tensor):
        """验证ONNX模型"""
        
        self.logger.info("验证ONNX模型...")
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 创建ONNX Runtime会话
        ort_session = ort.InferenceSession(onnx_path)
        
        # 获取输入输出信息
        input_name = ort_session.get_inputs()[0].name
        output_names = [output.name for output in ort_session.get_outputs()]
        
        # PyTorch推理
        with torch.no_grad():
            pytorch_outputs = self.model(test_input)
        
        # ONNX推理
        onnx_outputs = ort_session.run(
            output_names,
            {input_name: test_input.numpy()}
        )
        
        # 比较结果
        self._compare_outputs(pytorch_outputs, onnx_outputs)
        
        self.logger.info("✅ ONNX模型验证通过")
    
    def _compare_outputs(self, pytorch_outputs: Dict, onnx_outputs: List, 
                        tolerance: float = 1e-3):
        """比较PyTorch和ONNX输出"""
        
        output_names = ['growth_level', 'growth_pattern', 'interference_factors', 'fine_grained']
        
        for i, (name, pt_output) in enumerate(zip(output_names, pytorch_outputs.values())):
            onnx_output = onnx_outputs[i]
            
            # 计算差异
            diff = np.abs(pt_output.numpy() - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            self.logger.info(f"{name}: 最大差异={max_diff:.6f}, 平均差异={mean_diff:.6f}")
            
            if max_diff > tolerance:
                self.logger.warning(f"⚠️  {name} 输出差异超过容差")
            else:
                self.logger.info(f"✅ {name} 输出一致")
    
    def save_metadata(self, onnx_path: str):
        """保存模型元数据"""
        
        metadata = {
            'model_name': 'M16_MultiTask_MobileNetV3',
            'version': '1.0',
            'input_size': [3, 70, 70],
            'input_format': 'CHW',
            'output_format': {
                'growth_level': 'logits (3 classes)',
                'growth_pattern': 'logits (9 classes)', 
                'interference_factors': 'logits (3 classes, multi-label)',
                'fine_grained': 'logits (40 classes)'
            },
            'preprocessing': {
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'range': [0, 1]
            },
            'classes': get_class_definitions(),
            'model_config': self.model_config,
            'performance': {
                'validation_accuracy': '90.69%',
                'best_epoch': 59,
                'training_epochs': 100
            },
            'usage': {
                'growth_level': '菌落生长级别分类 (negative/positive/weak_growth)',
                'growth_pattern': '菌落生长模式分类 (9种模式)',
                'interference_factors': '干扰因素检测 (pores/debris/artifacts)',
                'fine_grained': '精细分类 (40种组合类别)'
            }
        }
        
        metadata_path = self.output_dir / "m16_multitask_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"元数据已保存到: {metadata_path}")
    
    def create_inference_example(self, onnx_path: str):
        """创建推理示例代码"""
        
        example_code = '''#!/usr/bin/env python3
"""
m16多任务模型ONNX推理示例
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import json
from pathlib import Path

class M16MultitaskInference:
    """m16多任务模型推理类"""
    
    def __init__(self, model_path: str, metadata_path: str = None):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 加载元数据
        self.metadata = self._load_metadata(metadata_path)
        
        # 定义预处理
        self.transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 类别定义
        self.classes = self.metadata.get('classes', {})
        
    def _load_metadata(self, metadata_path: str) -> dict:
        """加载元数据"""
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image)
        return tensor.numpy()
    
    def predict(self, image_path: str) -> dict:
        """预测单张图像"""
        # 预处理
        input_data = self.preprocess_image(image_path)
        input_data = np.expand_dims(input_data, axis=0)  # 添加batch维度
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # 处理输出
        results = {
            'growth_level': self._process_classification(outputs[0][0], 'growth_level'),
            'growth_pattern': self._process_classification(outputs[1][0], 'growth_pattern'),
            'interference_factors': self._process_multilabel(outputs[2][0], 'interference_factors'),
            'fine_grained': self._process_classification(outputs[3][0], 'fine_grained')
        }
        
        return results
    
    def _process_classification(self, logits: np.ndarray, task_name: str) -> dict:
        """处理分类输出"""
        probs = softmax(logits)
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        classes = self.classes.get(task_name, {}).get('classes', [])
        class_name = classes[pred_class] if pred_class < len(classes) else f"class_{pred_class}"
        
        return {
            'class_id': pred_class,
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': probs.tolist()
        }
    
    def _process_multilabel(self, logits: np.ndarray, task_name: str) -> dict:
        """处理多标签输出"""
        probs = sigmoid(logits)
        predictions = (probs > 0.5).astype(int)
        
        classes = self.classes.get(task_name, {}).get('classes', [])
        active_classes = []
        
        for i, pred in enumerate(predictions):
            if pred == 1 and i < len(classes):
                active_classes.append({
                    'class_id': i,
                    'class_name': classes[i],
                    'confidence': float(probs[i])
                })
        
        return {
            'active_classes': active_classes,
            'probabilities': probs.tolist()
        }

def softmax(x):
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def main():
    """主函数示例"""
    # 初始化推理器
    model_path = "onnx_models/m16_multitask_mobilenetv3.onnx"
    metadata_path = "onnx_models/m16_multitask_metadata.json"
    
    inference = M16MultitaskInference(model_path, metadata_path)
    
    # 预测图像
    image_path = "path/to/your/image.jpg"
    results = inference.predict(image_path)
    
    # 打印结果
    print("=== m16多任务分类结果 ===")
    print(f"生长级别: {results['growth_level']['class_name']} (置信度: {results['growth_level']['confidence']:.3f})")
    print(f"生长模式: {results['growth_pattern']['class_name']} (置信度: {results['growth_pattern']['confidence']:.3f})")
    print(f"干扰因素: {[cls['class_name'] for cls in results['interference_factors']['active_classes']]}")
    print(f"精细分类: {results['fine_grained']['class_name']} (置信度: {results['fine_grained']['confidence']:.3f})")

if __name__ == "__main__":
    main()
'''
        
        example_path = self.output_dir / "m16_multitask_inference_example.py"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)
        
        self.logger.info(f"推理示例已保存到: {example_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='m16多任务模型ONNX转换')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--output_dir', type=str, default='onnx_models', help='输出目录')
    parser.add_argument('--input_shape', type=int, nargs=4, default=[1, 3, 70, 70], help='输入形状')
    parser.add_argument('--opset_version', type=int, default=14, help='ONNX算子版本')
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = M16MultitaskONNXConverter(args.model_path, args.output_dir)
    
    # 转换为ONNX
    onnx_path = converter.convert_to_onnx(
        input_shape=tuple(args.input_shape),
        opset_version=args.opset_version
    )
    
    # 保存元数据
    converter.save_metadata(onnx_path)
    
    # 创建推理示例
    converter.create_inference_example(onnx_path)
    
    print(f"\\n✅ ONNX转换完成!")
    print(f"模型文件: {onnx_path}")
    print(f"元数据文件: {converter.output_dir / 'm16_multitask_metadata.json'}")
    print(f"推理示例: {converter.output_dir / 'm16_multitask_inference_example.py'}")

if __name__ == "__main__":
    main()