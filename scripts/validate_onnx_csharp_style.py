#!/usr/bin/env python3
"""
使用 ONNX 模型验证测试集 (C# 风格实现)
模拟 C# DatasetValidator 的行为,用于对比验证
"""

import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
from PIL import Image
import sys
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CSharpStyleValidator:
    """模拟 C# DatasetValidator 的验证逻辑"""

    def __init__(
        self,
        model_path: str,
        data_root: str = "ds/images",
        annotations_file: str = "ds/images/m9e1n170_cleaned_round2.json",
        split_file: str = "ds/images/dataset_split_seed44.json"
    ):
        self.model_path = model_path
        self.data_root = Path(data_root)
        self.annotations_file = Path(annotations_file)
        self.split_file = Path(split_file)

        # 加载 ONNX 模型
        print("加载 ONNX 模型...")
        self.session = ort.InferenceSession(model_path)
        print(f"  模型加载成功: {model_path}")

        # 优化的阈值 (与 C# 代码一致)
        self.optimal_thresholds = {
            'pores': 0.40,
            'artifacts': 0.45,
            'debris': 0.15,
            'contamination': 0.50
        }

        # Label映射
        self.label_mappings = {
            'growth_level': ['negative', 'positive'],
            'growth_pattern': [
                'center_dots', 'clean', 'clustered', 'even_scattered',
                'heavy_growth', 'negative', 'weak_scattered',
                'weak_scattered_neg', 'weak_scattered_pos', 'unclear'
            ],
            'interference_factors': ['pores', 'artifacts', 'debris', 'contamination']
        }

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """预处理图像 (模拟 C# PreprocessImage)"""
        # 加载图像
        img = Image.open(image_path)

        # 调整大小到 70x70
        img = img.resize((70, 70))

        # 转换为灰度
        if img.mode != 'L':
            img = img.convert('L')

        # 转换为 numpy 数组并归一化
        img_array = np.array(img).astype(np.float32) / 255.0

        # 构建张量 [1, 1, 70, 70]
        tensor = img_array.reshape(1, 1, 70, 70)

        return tensor

    def sigmoid(self, x):
        """Sigmoid 激活函数"""
        return 1.0 / (1.0 + np.exp(-x))

    def softmax(self, x):
        """Softmax 激活函数"""
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def predict(self, image_path: Path):
        """推理单张图像 (模拟 C# Predict)"""
        # 预处理
        input_tensor = self.preprocess_image(image_path)

        # 运行推理
        outputs = self.session.run(None, {'input': input_tensor})

        # 解析输出
        growth_level_logits = outputs[0][0]  # [2]
        growth_pattern_logits = outputs[1][0]  # [10]
        interference_logits = outputs[2][0]  # [4]

        # Growth Level (Sigmoid)
        growth_level_probs = self.sigmoid(growth_level_logits)
        growth_level_pred = 1 if growth_level_probs[1] > 0.5 else 0
        growth_level_label = self.label_mappings['growth_level'][growth_level_pred]

        # Growth Pattern (Softmax)
        growth_pattern_probs = self.softmax(growth_pattern_logits)
        growth_pattern_pred = np.argmax(growth_pattern_probs)
        growth_pattern_label = self.label_mappings['growth_pattern'][growth_pattern_pred]

        # Interference Factors (Sigmoid + Threshold)
        interference_probs = self.sigmoid(interference_logits)
        interference_predictions = {}

        for i, factor_name in enumerate(self.label_mappings['interference_factors']):
            threshold = self.optimal_thresholds[factor_name]
            interference_predictions[factor_name] = {
                'score': float(interference_probs[i]),
                'is_present': interference_probs[i] >= threshold,
                'threshold': threshold
            }

        return {
            'growth_level': {
                'label': growth_level_label,
                'confidence': float(growth_level_probs[growth_level_pred]),
                'probabilities': growth_level_probs.tolist()
            },
            'growth_pattern': {
                'label': growth_pattern_label,
                'confidence': float(growth_pattern_probs[growth_pattern_pred]),
                'probabilities': growth_pattern_probs.tolist()
            },
            'interference_factors': interference_predictions
        }

    def validate_test_set(self):
        """验证测试集 (模拟 C# ValidateTestSet)"""
        print("\n=== 数据集验证开始 ===\n")

        # 1. 加载标注
        print("[1/4] 加载数据集标注...")
        with open(self.annotations_file, 'r') as f:
            annotations_data = json.load(f)
        annotations = annotations_data['annotations']  # 使用 'annotations' 键
        print(f"  加载了 {len(annotations)} 个图像标注")

        # 2. 加载测试集划分
        print("\n[2/4] 加载测试集划分...")
        with open(self.split_file, 'r') as f:
            split_data = json.load(f)
        test_images = split_data['splits']['test']  # 使用 splits.test 键
        print(f"  测试集包含 {len(test_images)} 个样本")

        # 3. 初始化结果统计
        results = {
            'total_samples': len(test_images),
            'growth_level_correct': 0,
            'growth_pattern_correct': 0,
            'interference_correct': defaultdict(int),
            'interference_total': defaultdict(int),
            'true_negative': 0,
            'true_positive': 0,
            'false_negative': 0,
            'false_positive': 0,
            'interference_stats': defaultdict(lambda: {
                'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0
            }),
            'error_samples': []
        }

        # 4. 运行推理
        print("\n[3/4] 开始批量推理...")
        processed = 0

        for image_name in test_images:
            if image_name not in annotations:
                print(f"  警告: 找不到图像标注 {image_name}")
                continue

            annotation = annotations[image_name]

            # 构建图像路径
            parts = image_name.split('/')
            image_path = self.data_root / parts[0] / parts[1]

            if not image_path.exists():
                print(f"  警告: 找不到图像文件 {image_path}")
                continue

            try:
                # 运行推理
                prediction = self.predict(image_path)

                # 验证 Growth Level
                expected_level = annotation['growth_level']
                predicted_level = prediction['growth_level']['label']

                if predicted_level == expected_level:
                    results['growth_level_correct'] += 1
                    if expected_level == 'negative':
                        results['true_negative'] += 1
                    else:
                        results['true_positive'] += 1
                else:
                    if expected_level == 'negative':
                        results['false_positive'] += 1
                    else:
                        results['false_negative'] += 1

                    results['error_samples'].append({
                        'image_path': image_name,
                        'error_type': 'Growth Level',
                        'expected': expected_level,
                        'predicted': predicted_level,
                        'confidence': prediction['growth_level']['confidence']
                    })

                # 验证 Growth Pattern
                expected_pattern = annotation['growth_pattern']
                predicted_pattern = prediction['growth_pattern']['label']

                if predicted_pattern == expected_pattern:
                    results['growth_pattern_correct'] += 1
                else:
                    results['error_samples'].append({
                        'image_path': image_name,
                        'error_type': 'Growth Pattern',
                        'expected': expected_pattern,
                        'predicted': predicted_pattern,
                        'confidence': prediction['growth_pattern']['confidence']
                    })

                # 验证 Interference Factors
                for factor_name, factor_pred in prediction['interference_factors'].items():
                    if factor_name in annotation['interference_factors']:
                        expected = annotation['interference_factors'][factor_name]
                        predicted = factor_pred['is_present']

                        results['interference_total'][factor_name] += 1
                        stats = results['interference_stats'][factor_name]

                        if predicted == expected:
                            results['interference_correct'][factor_name] += 1
                            if predicted:
                                stats['tp'] += 1
                            else:
                                stats['tn'] += 1
                        else:
                            if predicted and not expected:
                                stats['fp'] += 1
                                results['error_samples'].append({
                                    'image_path': image_name,
                                    'error_type': f'Interference - {factor_name} (FP)',
                                    'expected': 'false',
                                    'predicted': 'true',
                                    'confidence': factor_pred['score']
                                })
                            elif not predicted and expected:
                                stats['fn'] += 1
                                results['error_samples'].append({
                                    'image_path': image_name,
                                    'error_type': f'Interference - {factor_name} (FN)',
                                    'expected': 'true',
                                    'predicted': 'false',
                                    'confidence': factor_pred['score']
                                })

                processed += 1

                if processed % 100 == 0:
                    print(f"  已处理: {processed}/{len(test_images)}")

            except Exception as e:
                print(f"  错误: 处理 {image_name} 时出错: {e}")

        print(f"\n  完成! 处理了 {processed} 个样本")

        return results

    @staticmethod
    def print_results(results):
        """打印结果 (模拟 C# PrintResults)"""
        print("\n" + "=" * 80)
        print("验证结果汇总")
        print("=" * 80)

        total = results['total_samples']

        # 总体性能
        print("\n[总体性能]")
        growth_level_acc = 100.0 * results['growth_level_correct'] / total
        growth_pattern_acc = 100.0 * results['growth_pattern_correct'] / total

        print(f"  Growth Level 准确率: {growth_level_acc:.2f}% ({results['growth_level_correct']}/{total})")
        print(f"  Growth Pattern 准确率: {growth_pattern_acc:.2f}% ({results['growth_pattern_correct']}/{total})")

        # Interference Factors
        print("\n[Interference Factors 准确率]")
        interference_overall_acc = 0
        factor_count = 0

        for factor_name in sorted(results['interference_stats'].keys()):
            stats = results['interference_stats'][factor_name]
            tp, tn, fp, fn = stats['tp'], stats['tn'], stats['fp'], stats['fn']

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  {factor_name}:")
            print(f"    准确率: {accuracy:.2%}")
            print(f"    精确率: {precision:.2%}")
            print(f"    召回率: {recall:.2%}")
            print(f"    F1分数: {f1:.2%}")
            print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")

            interference_overall_acc += accuracy
            factor_count += 1

        if factor_count > 0:
            interference_overall_acc /= factor_count
            print(f"\n  Interference Overall 准确率: {interference_overall_acc:.2%}")

        # 总准确率
        total_accuracy = (growth_level_acc + growth_pattern_acc + interference_overall_acc * 100) / 3
        print(f"\n[总准确率] {total_accuracy:.2f}%")

        # Growth Level 混淆矩阵
        print("\n[Growth Level 混淆矩阵]")
        print(f"              Predicted Negative  Predicted Positive")
        print(f"  Actual Negative:     {results['true_negative']:<6}            {results['false_positive']:<6}")
        print(f"  Actual Positive:     {results['false_negative']:<6}            {results['true_positive']:<6}")

        # 错误样本统计
        print("\n[错误样本统计]")
        error_by_type = defaultdict(int)
        for error in results['error_samples']:
            error_by_type[error['error_type']] += 1

        for error_type in sorted(error_by_type.keys(), key=lambda x: error_by_type[x], reverse=True):
            print(f"  {error_type}: {error_by_type[error_type]} 个错误")

        # 显示前10个错误样本
        print("\n[前10个错误样本]")
        for error in results['error_samples'][:10]:
            print(f"  {error['image_path']}")
            print(f"    类型: {error['error_type']}")
            print(f"    期望: {error['expected']}, 预测: {error['predicted']} (置信度: {error['confidence']:.2%})")

        if len(results['error_samples']) > 10:
            print(f"  ... 还有 {len(results['error_samples']) - 10} 个错误样本")

        print("\n" + "=" * 80)

    @staticmethod
    def export_results(results, output_path):
        """导出结果到 JSON"""
        # 转换 defaultdict 为普通 dict
        export_data = {
            'total_samples': results['total_samples'],
            'growth_level_correct': results['growth_level_correct'],
            'growth_pattern_correct': results['growth_pattern_correct'],
            'interference_correct': dict(results['interference_correct']),
            'interference_total': dict(results['interference_total']),
            'confusion_matrix': {
                'true_negative': results['true_negative'],
                'true_positive': results['true_positive'],
                'false_negative': results['false_negative'],
                'false_positive': results['false_positive']
            },
            'interference_stats': dict(results['interference_stats']),
            'error_samples': results['error_samples']
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n验证结果已导出到: {output_path}")


def main():
    print("BioAst ONNX Model - Dataset Validation Tool (Python)")
    print("=" * 50 + "\n")

    # 配置
    model_path = "deployment/onnx_models/mobilenetv4_v0.11.0/model.onnx"
    output_path = "csharp_style_validation_results.json"

    # 创建验证器
    validator = CSharpStyleValidator(model_path=model_path)

    # 运行验证
    results = validator.validate_test_set()

    # 打印结果
    validator.print_results(results)

    # 导出结果
    validator.export_results(results, output_path)

    print("\n验证完成!")


if __name__ == '__main__':
    main()
