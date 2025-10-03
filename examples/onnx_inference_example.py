#!/usr/bin/env python3
"""
MobileNetV4 v1.1 ONNX推理示例
示例代码展示如何使用ONNX模型进行细菌图像分类推理
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class MobileNetV4Classifier:
    """
    MobileNetV4分类器 - ONNX版本

    功能: 对70×70像素的细菌图像进行三级分类
    - Growth Level: 阴性/阳性 (2类)
    - Growth Pattern: 生长模式 (10类)
    - Interference Factors: 干扰因素 (4类,多标签)
    """

    # 类别标签定义
    GROWTH_LEVEL_LABELS = ['negative', 'positive']

    GROWTH_PATTERN_LABELS = [
        'clean', 'clustered', 'weak_scattered', 'scattered',
        'heavy_growth', 'partial_growth', 'edge_growth',
        'center_growth', 'litter_center_dots', 'unknown'
    ]

    INTERFERENCE_LABELS = [
        'pores', 'artifacts', 'debris', 'contamination'
    ]

    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        初始化分类器

        Args:
            model_path: ONNX模型文件路径
            use_gpu: 是否使用GPU加速
        """
        self.model_path = model_path

        # 配置推理提供者
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        # 创建推理会话
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"✓ 模型加载成功: {model_path}")
        print(f"  - 推理提供者: {self.session.get_providers()[0]}")
        print(f"  - 输入: {self.input_name}")
        print(f"  - 输出: {self.output_names}")

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        图像预处理

        Args:
            image_path: 图像文件路径

        Returns:
            预处理后的图像张量 [1, 1, 70, 70]
        """
        # 读取图像为灰度图
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 调整大小到70x70
        image = cv2.resize(image, (70, 70))

        # 归一化到[-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5

        # 添加batch和channel维度: [H, W] -> [1, 1, H, W]
        image = image[np.newaxis, np.newaxis, :, :]

        return image

    def predict(self, image_path: str, return_probs: bool = False):
        """
        对单张图像进行预测

        Args:
            image_path: 图像文件路径
            return_probs: 是否返回概率值

        Returns:
            预测结果字典
        """
        # 预处理
        input_data = self.preprocess(image_path)

        # 推理
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_data}
        )

        # 解析输出
        growth_level_logits = outputs[0][0]  # [2]
        growth_pattern_logits = outputs[1][0]  # [10]
        interference_logits = outputs[2][0]  # [4]

        # Softmax + 预测
        growth_level_probs = self._softmax(growth_level_logits)
        growth_level_pred = int(np.argmax(growth_level_probs))

        growth_pattern_probs = self._softmax(growth_pattern_logits)
        growth_pattern_pred = int(np.argmax(growth_pattern_probs))

        # Sigmoid + 多标签预测
        interference_probs = self._sigmoid(interference_logits)
        interference_preds = (interference_probs > 0.5).astype(int)

        # 构建结果
        result = {
            'growth_level': {
                'label': self.GROWTH_LEVEL_LABELS[growth_level_pred],
                'confidence': float(growth_level_probs[growth_level_pred])
            },
            'growth_pattern': {
                'label': self.GROWTH_PATTERN_LABELS[growth_pattern_pred],
                'confidence': float(growth_pattern_probs[growth_pattern_pred])
            },
            'interference_factors': {
                'labels': [
                    self.INTERFERENCE_LABELS[i]
                    for i in range(len(interference_preds))
                    if interference_preds[i] == 1
                ],
                'probabilities': {
                    self.INTERFERENCE_LABELS[i]: float(interference_probs[i])
                    for i in range(len(interference_probs))
                }
            }
        }

        # 可选：返回完整概率分布
        if return_probs:
            result['probabilities'] = {
                'growth_level': {
                    self.GROWTH_LEVEL_LABELS[i]: float(growth_level_probs[i])
                    for i in range(len(growth_level_probs))
                },
                'growth_pattern': {
                    self.GROWTH_PATTERN_LABELS[i]: float(growth_pattern_probs[i])
                    for i in range(len(growth_pattern_probs))
                }
            }

        return result

    def batch_predict(self, image_paths: list, batch_size: int = 32):
        """
        批量预测

        Args:
            image_paths: 图像路径列表
            batch_size: 批量大小

        Returns:
            预测结果列表
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # 预处理批量图像
            batch_data = np.concatenate([
                self.preprocess(path) for path in batch_paths
            ], axis=0)

            # 批量推理
            outputs = self.session.run(
                self.output_names,
                {self.input_name: batch_data}
            )

            # 解析每个样本的结果
            for j in range(len(batch_paths)):
                growth_level_logits = outputs[0][j]
                growth_pattern_logits = outputs[1][j]
                interference_logits = outputs[2][j]

                growth_level_probs = self._softmax(growth_level_logits)
                growth_level_pred = int(np.argmax(growth_level_probs))

                growth_pattern_probs = self._softmax(growth_pattern_logits)
                growth_pattern_pred = int(np.argmax(growth_pattern_probs))

                interference_probs = self._sigmoid(interference_logits)
                interference_preds = (interference_probs > 0.5).astype(int)

                result = {
                    'image_path': batch_paths[j],
                    'growth_level': {
                        'label': self.GROWTH_LEVEL_LABELS[growth_level_pred],
                        'confidence': float(growth_level_probs[growth_level_pred])
                    },
                    'growth_pattern': {
                        'label': self.GROWTH_PATTERN_LABELS[growth_pattern_pred],
                        'confidence': float(growth_pattern_probs[growth_pattern_pred])
                    },
                    'interference_factors': {
                        'labels': [
                            self.INTERFERENCE_LABELS[k]
                            for k in range(len(interference_preds))
                            if interference_preds[k] == 1
                        ]
                    }
                }

                results.append(result)

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-x))


def example_single_image():
    """示例1: 单张图像推理"""
    print("=" * 80)
    print("示例1: 单张图像推理")
    print("=" * 80)

    # 初始化分类器
    model_path = "deployment/onnx_models/mobilenetv4_v1.1.onnx"
    classifier = MobileNetV4Classifier(model_path, use_gpu=True)

    # 单张图像推理（替换为实际图像路径）
    image_path = "path/to/your/image.png"

    # 检查图像是否存在
    if not Path(image_path).exists():
        print(f"\n⚠ 图像不存在: {image_path}")
        print("请替换为实际的图像路径")
        return

    # 预测
    result = classifier.predict(image_path, return_probs=True)

    # 输出结果
    print(f"\n图像: {image_path}")
    print(f"\n预测结果:")
    print(f"  Growth Level: {result['growth_level']['label']}")
    print(f"  - 置信度: {result['growth_level']['confidence']:.2%}")

    print(f"\n  Growth Pattern: {result['growth_pattern']['label']}")
    print(f"  - 置信度: {result['growth_pattern']['confidence']:.2%}")

    print(f"\n  Interference Factors: {', '.join(result['interference_factors']['labels']) or 'None'}")
    print(f"  - 概率分布:")
    for label, prob in result['interference_factors']['probabilities'].items():
        print(f"    - {label}: {prob:.2%}")


def example_batch_inference():
    """示例2: 批量推理"""
    print("\n" + "=" * 80)
    print("示例2: 批量推理")
    print("=" * 80)

    # 初始化分类器
    model_path = "deployment/onnx_models/mobilenetv4_v1.1.onnx"
    classifier = MobileNetV4Classifier(model_path, use_gpu=True)

    # 批量图像路径（替换为实际路径）
    image_paths = [
        "path/to/image1.png",
        "path/to/image2.png",
        "path/to/image3.png"
    ]

    # 检查图像是否存在
    existing_paths = [p for p in image_paths if Path(p).exists()]

    if not existing_paths:
        print(f"\n⚠ 没有找到有效的图像")
        print("请替换为实际的图像路径")
        return

    # 批量预测
    results = classifier.batch_predict(existing_paths, batch_size=32)

    # 输出结果
    print(f"\n批量推理完成，共处理 {len(results)} 张图像")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {Path(result['image_path']).name}")
        print(f"   - Growth Level: {result['growth_level']['label']} ({result['growth_level']['confidence']:.2%})")
        print(f"   - Growth Pattern: {result['growth_pattern']['label']} ({result['growth_pattern']['confidence']:.2%})")
        print(f"   - Interference: {', '.join(result['interference_factors']['labels']) or 'None'}")


def example_performance_benchmark():
    """示例3: 性能基准测试"""
    print("\n" + "=" * 80)
    print("示例3: 性能基准测试")
    print("=" * 80)

    import time

    # 初始化分类器
    model_path = "deployment/onnx_models/mobilenetv4_v1.1.onnx"
    classifier = MobileNetV4Classifier(model_path, use_gpu=True)

    # 创建随机测试数据
    num_images = 100
    test_images = [np.random.randint(0, 255, (70, 70), dtype=np.uint8) for _ in range(num_images)]

    # 保存临时图像
    import tempfile
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    for i, img in enumerate(test_images):
        path = f"{temp_dir}/test_{i}.png"
        cv2.imwrite(path, img)
        image_paths.append(path)

    # 预热
    print("\n预热中...")
    classifier.predict(image_paths[0])

    # 基准测试
    print(f"\n处理 {num_images} 张图像...")
    start_time = time.time()
    results = classifier.batch_predict(image_paths, batch_size=32)
    end_time = time.time()

    # 计算性能
    total_time = end_time - start_time
    avg_time = total_time / num_images
    fps = num_images / total_time

    print(f"\n性能统计:")
    print(f"  - 总时间: {total_time:.2f}秒")
    print(f"  - 平均时间: {avg_time*1000:.2f}毫秒/张")
    print(f"  - 吞吐量: {fps:.2f} FPS")

    # 清理
    import shutil
    shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # 运行示例
    print("\n" + "=" * 80)
    print("MobileNetV4 v1.1 ONNX推理示例")
    print("=" * 80)

    try:
        # 示例1: 单张图像
        example_single_image()

        # 示例2: 批量推理
        # example_batch_inference()

        # 示例3: 性能测试
        example_performance_benchmark()

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保ONNX模型文件存在: deployment/onnx_models/mobilenetv4_v1.1.onnx")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
