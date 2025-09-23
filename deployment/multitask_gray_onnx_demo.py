#!/usr/bin/env python3
"""
多任务灰度菌落检测网络ONNX部署示例
展示如何在生产环境中使用转换后的ONNX模型
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import Dict, List, Tuple, Any
import json
import time

class MultitaskGrayColonyONNX:
    """ONNX格式的多任务灰度菌落检测模型"""
    
    def __init__(self, model_path: str):
        """
        初始化ONNX模型
        
        Args:
            model_path: ONNX模型文件路径
        """
        # 创建ONNX运行时会话
        self.session = ort.InferenceSession(model_path)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 类别名称定义
        self.growth_level_names = ['negative', 'positive', 'weak_growth']
        self.growth_pattern_names = [
            'clean', 'clustered', 'scattered', 'small_dots',
            'ring_shaped', 'irregular', 'mixed', 'sparse', 'dense'
        ]
        self.interference_names = ['pores', 'debris', 'artifacts', 'contamination']
        self.fine_grained_names = [
            'positive_cluster_no_pores',
            'positive_cluster_with_pores',
            'positive_cluster_overlapping_pores',
            'negative_clean_no_pores',
            'negative_clean_with_pores',
            'weak_growth_center_no_pores',
            'weak_growth_center_with_pores',
            'weak_growth_center_overlapping_pores',
            'weak_growth_scattered_no_pores',
            'weak_growth_scattered_with_pores',
            'weak_growth_scattered_overlapping_pores',
            'with_debris',
            'with_artifacts',
            'contaminated',
            'other'
        ]
        
        # 阈值设置
        self.thresholds = {
            'interference': 0.5,
            'pore_confidence': 0.5,
            'bg_confidence': 0.5
        }
        
        print(f"模型加载成功: {model_path}")
        print(f"输入形状: {self.input_shape}")
        print(f"输出任务: {len(self.output_names)} 个")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 输入图像 (H, W) 或 (H, W, C)
            
        Returns:
            预处理后的图像 (1, 1, 70, 70)
        """
        # 转换为灰度
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        
        # 调整大小到70x70
        if image.shape != (70, 70):
            image = cv2.resize(image, (70, 70), interpolation=cv2.INTER_AREA)
        
        # 归一化到0-1
        image = image.astype(np.float32) / 255.0
        
        # 标准化 (使用ImageNet均值和标准差)
        mean = 0.449
        std = 0.226
        image = (image - mean) / std
        
        # 添加批次和通道维度
        image = np.expand_dims(image, axis=0)  # (1, 70, 70)
        image = np.expand_dims(image, axis=0)  # (1, 1, 70, 70)
        
        return image
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            image: 输入图像
            
        Returns:
            包含所有预测结果的字典
        """
        # 预处理
        input_tensor = self.preprocess_image(image)
        
        # 推理
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        
        # 解析输出
        growth_level = outputs[0][0]  # (3,)
        growth_pattern = outputs[1][0]  # (9,)
        interference = outputs[2][0]  # (4,)
        fine_grained = outputs[3][0]  # (15,)
        pore_confidence = outputs[4][0][0]  # 标量
        bg_confidence = outputs[5][0][0]  # 标量
        
        # 处理各任务预测
        results = {
            # 原始输出
            'raw_outputs': {
                'growth_level': growth_level.tolist(),
                'growth_pattern': growth_pattern.tolist(),
                'interference': interference.tolist(),
                'fine_grained': fine_grained.tolist(),
                'pore_confidence': float(pore_confidence),
                'bg_confidence': float(bg_confidence)
            },
            
            # 解析后的预测
            'predictions': {
                'growth_level': self._parse_growth_level(growth_level),
                'growth_pattern': self._parse_growth_pattern(growth_pattern),
                'interference_mapping': self._parse_interference(interference),
                'fine_grained': self._parse_fine_grained(fine_grained),
                'auxiliary': {
                    'pore_confidence': float(pore_confidence),
                    'bg_confidence': float(bg_confidence),
                    'has_pores': pore_confidence > self.thresholds['pore_confidence']
                }
            },
            
            # 元信息
            'metadata': {
                'inference_time_ms': inference_time * 1000,
                'input_size': image.shape,
                'model_type': 'multitask_gray_colony_onnx'
            }
        }
        
        return results
    
    def _parse_growth_level(self, logits: np.ndarray) -> Dict[str, Any]:
        """解析生长级别预测"""
        probs = self._softmax(logits)
        idx = np.argmax(probs)
        
        return {
            'class': self.growth_level_names[idx],
            'confidence': float(probs[idx]),
            'probabilities': {name: float(prob) for name, prob in zip(self.growth_level_names, probs)}
        }
    
    def _parse_growth_pattern(self, logits: np.ndarray) -> Dict[str, Any]:
        """解析生长模式预测"""
        probs = self._softmax(logits)
        idx = np.argmax(probs)
        
        return {
            'class': self.growth_pattern_names[idx],
            'confidence': float(probs[idx]),
            'probabilities': {name: float(prob) for name, prob in zip(self.growth_pattern_names, probs)}
        }
    
    def _parse_interference(self, logits: np.ndarray) -> Dict[str, Any]:
        """解析干扰因素预测（多标签）"""
        probs = self._sigmoid(logits)
        labels = [self.interference_names[i] for i, prob in enumerate(probs) 
                 if prob > self.thresholds['interference']]
        
        return {
            'labels': labels,
            'probabilities': {name: float(prob) for name, prob in zip(self.interference_names, probs)}
        }
    
    def _parse_fine_grained(self, logits: np.ndarray) -> Dict[str, Any]:
        """解析精细分类预测"""
        probs = self._softmax(logits)
        idx = np.argmax(probs)
        
        return {
            'class': self.fine_grained_names[idx],
            'confidence': float(probs[idx]),
            'probabilities': {name: float(prob) for name, prob in zip(self.fine_grained_names, probs)}
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-x))
    
    def batch_predict(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            images: 图像列表
            
        Returns:
            预测结果列表
        """
        # 批量预处理
        batch_input = np.stack([self.preprocess_image(img) for img in images], axis=0)
        batch_input = batch_input.squeeze(1)  # 移除多余的通道维度
        
        # 批量推理
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: batch_input})
        inference_time = time.time() - start_time
        
        # 处理每个样本的输出
        batch_size = len(images)
        results = []
        
        for i in range(batch_size):
            growth_level = outputs[0][i]
            growth_pattern = outputs[1][i]
            interference = outputs[2][i]
            fine_grained = outputs[3][i]
            pore_confidence = outputs[4][i][0]
            bg_confidence = outputs[5][i][0]
            
            result = {
                'predictions': {
                    'growth_level': self._parse_growth_level(growth_level),
                    'growth_pattern': self._parse_growth_pattern(growth_pattern),
                    'interference_mapping': self._parse_interference(interference),
                    'fine_grained': self._parse_fine_grained(fine_grained),
                    'auxiliary': {
                        'pore_confidence': float(pore_confidence),
                        'bg_confidence': float(bg_confidence),
                        'has_pores': pore_confidence > self.thresholds['pore_confidence']
                    }
                },
                'metadata': {
                    'inference_time_ms': inference_time * 1000 / batch_size,
                    'sample_index': i
                }
            }
            
            results.append(result)
        
        return results


def demo():
    """演示ONNX模型的使用"""
    print("=== 多任务灰度菌落检测网络ONNX部署演示 ===")
    
    # 创建模型实例
    model_path = "multitask_gray_colony_net.onnx"
    try:
        detector = MultitaskGrayColonyONNX(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保已运行 test_multitask_gray_onnx.py 生成ONNX模型")
        return
    
    # 创建测试图像
    print("\n--- 创建测试图像 ---")
    test_images = []
    
    # 测试图像1: 模拟阳性菌落
    img1 = np.random.rand(70, 70) * 0.3 + 0.4  # 中等灰度
    # 添加一些菌落样结构
    cv2.circle(img1, (35, 35), 15, 0.8, -1)
    cv2.circle(img1, (25, 25), 3, 0.2, -1)  # 气孔
    test_images.append(img1)
    
    # 测试图像2: 模拟阴性样本
    img2 = np.random.rand(70, 70) * 0.2 + 0.1  # 较暗
    # 添加一些气孔
    for _ in range(5):
        x, y = np.random.randint(10, 60, 2)
        cv2.circle(img2, (x, y), 2, 0.9, -1)
    test_images.append(img2)
    
    print(f"创建了 {len(test_images)} 个测试图像")
    
    # 单张预测
    print("\n--- 单张预测演示 ---")
    result = detector.predict(test_images[0])
    
    print("预测结果:")
    pred = result['predictions']
    print(f"  生长级别: {pred['growth_level']['class']} (置信度: {pred['growth_level']['confidence']:.3f})")
    print(f"  生长模式: {pred['growth_pattern']['class']} (置信度: {pred['growth_pattern']['confidence']:.3f})")
    print(f"  干扰因素: {pred['interference_mapping']['labels']}")
    print(f"  精细分类: {pred['fine_grained']['class']}")
    print(f"  气孔置信度: {pred['auxiliary']['pore_confidence']:.3f}")
    print(f"  推理时间: {result['metadata']['inference_time_ms']:.2f} ms")
    
    # 批量预测
    print("\n--- 批量预测演示 ---")
    batch_results = detector.batch_predict(test_images)
    
    for i, result in enumerate(batch_results):
        pred = result['predictions']
        print(f"\n图像 {i+1}:")
        print(f"  生长级别: {pred['growth_level']['class']}")
        print(f"  生长模式: {pred['growth_pattern']['class']}")
        print(f"  精细分类: {pred['fine_grained']['class']}")
        print(f"  推理时间: {result['metadata']['inference_time_ms']:.2f} ms")
    
    # 性能测试
    print("\n--- 性能测试 ---")
    num_tests = 100
    test_batch = [test_images[0] for _ in range(num_tests)]
    
    start_time = time.time()
    batch_results = detector.batch_predict(test_batch)
    total_time = time.time() - start_time
    
    avg_time = total_time / num_tests * 1000
    fps = num_tests / total_time
    
    print(f"平均推理时间: {avg_time:.2f} ms/张")
    print(f"吞吐量: {fps:.1f} FPS")
    
    # 保存结果示例
    print("\n--- 保存结果示例 ---")
    sample_result = detector.predict(test_images[0])
    
    # 保存为JSON
    with open('prediction_result_sample.json', 'w', encoding='utf-8') as f:
        json.dump(sample_result, f, indent=2, ensure_ascii=False)
    
    print("预测结果已保存到: prediction_result_sample.json")
    
    print(f"\n✓ ONNX部署演示完成！")


if __name__ == "__main__":
    demo()