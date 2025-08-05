"""
简化的个体模型分析器
针对单个模型进行详细性能分析和错误样本分析
"""

import os
import sys
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import time
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from PIL import Image
import importlib

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset import BioastDataset
from torchvision import transforms

class SimpleModelAnalyzer:
    """简化的模型分析器"""
    
    def __init__(self, model_name: str, model_config: Dict):
        self.model_name = model_name
        self.model_config = model_config
        self.input_shape = model_config['input_shape']
        self.output_dir = Path(f"reports/detailed_analysis/{model_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_models(self) -> Tuple[Optional[torch.nn.Module], Optional[ort.InferenceSession]]:
        """加载PyTorch和ONNX模型"""
        print(f"Loading models for {self.model_name}...")
        
        pytorch_model = None
        onnx_session = None
        
        try:
            # 加载PyTorch模型
            module = importlib.import_module(self.model_config['module'])
            factory_func = getattr(module, self.model_config['factory_function'])
            pytorch_model = factory_func(num_classes=2)
            pytorch_model.eval()
            
            # 加载检查点
            checkpoint_path = Path(self.model_config['checkpoint_path'])
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                pytorch_model.load_state_dict(state_dict)
                print(f"PyTorch model loaded successfully")
            else:
                print(f"Checkpoint not found: {checkpoint_path}")
                return None, None
                
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
            return None, None
        
        try:
            # 加载ONNX模型
            onnx_path = Path(f"onnx_models/{self.model_name}.onnx")
            if onnx_path.exists():
                onnx_session = ort.InferenceSession(str(onnx_path))
                print(f"ONNX model loaded successfully")
            else:
                print(f"ONNX model not found: {onnx_path}")
                return pytorch_model, None
                
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            return pytorch_model, None
        
        return pytorch_model, onnx_session
    
    def prepare_test_data(self, num_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """准备测试数据"""
        print(f"Preparing test data for {self.model_name}...")
        
        try:
            # 定义变换
            if self.input_shape[1:] == (224, 224):  # EfficientNet
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            else:  # Others use 70x70
                transform = transforms.Compose([
                    transforms.Resize((70, 70)),
                    transforms.ToTensor(),
                ])
            
            dataset = BioastDataset(data_dir='bioast_dataset', split='test', transform=transform)
            print(f"Found {len(dataset)} samples in test dataset")
            
            # 平衡采样
            negative_samples = []
            positive_samples = []
            
            for i, (data, target) in enumerate(dataset):
                if target == 0:
                    negative_samples.append((data.numpy(), target))
                else:
                    positive_samples.append((data.numpy(), target))
            
            samples_per_class = min(num_samples // 2, len(negative_samples), len(positive_samples))
            
            selected_samples = []
            selected_labels = []
            
            # 添加负样本
            for i in range(samples_per_class):
                data, label = negative_samples[i]
                selected_samples.append(data)
                selected_labels.append(label)
            
            # 添加正样本
            for i in range(samples_per_class):
                data, label = positive_samples[i]
                selected_samples.append(data)
                selected_labels.append(label)
            
            if selected_samples:
                test_data = np.stack(selected_samples, axis=0)
                test_labels = np.array(selected_labels)
                
                print(f"Loaded balanced test data - Negative: {np.sum(test_labels == 0)}, Positive: {np.sum(test_labels == 1)}")
                return test_data, test_labels
            
        except Exception as e:
            print(f"Failed to load real test data: {e}")
        
        # 回退到随机数据
        test_data = np.random.randn(num_samples, *self.input_shape).astype(np.float32)
        test_labels = np.random.randint(0, 2, num_samples)
        print(f"Using random test data, shape: {test_data.shape}")
        
        return test_data, test_labels
    
    def run_inference_analysis(self, pytorch_model: torch.nn.Module, onnx_session: ort.InferenceSession,
                             test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, Any]:
        """运行推理分析"""
        print(f"Running inference analysis...")
        
        results = {
            'pytorch_predictions': [],
            'pytorch_probabilities': [],
            'pytorch_times': [],
            'onnx_predictions': [],
            'onnx_probabilities': [],
            'onnx_times': [],
            'ground_truth': test_labels
        }
        
        # PyTorch推理
        if pytorch_model is not None:
            pytorch_model.eval()
            with torch.no_grad():
                for i, sample in enumerate(test_data):
                    input_tensor = torch.from_numpy(sample).unsqueeze(0)
                    
                    start_time = time.perf_counter()
                    try:
                        output = pytorch_model(input_tensor)
                        end_time = time.perf_counter()
                        
                        if isinstance(output, dict):
                            for key in ['classification', 'logits', 'output']:\n                                if key in output:\n                                    output = output[key]\n                                    break\n                            else:\n                                output = list(output.values())[0]\n                        elif isinstance(output, (tuple, list)):\n                            output = output[0]\n                        \n                        probabilities = torch.softmax(output, dim=1).numpy()[0]\n                        prediction = np.argmax(probabilities)\n                        \n                        results['pytorch_predictions'].append(prediction)\n                        results['pytorch_probabilities'].append(probabilities)\n                        results['pytorch_times'].append(end_time - start_time)\n                        \n                    except Exception as e:\n                        print(f\"PyTorch inference failed for sample {i}: {e}\")\n                        results['pytorch_predictions'].append(-1)\n                        results['pytorch_probabilities'].append([0.5, 0.5])\n                        results['pytorch_times'].append(0)\n        \n        # ONNX推理\n        if onnx_session is not None:\n            input_name = onnx_session.get_inputs()[0].name\n            \n            for i, sample in enumerate(test_data):\n                input_batch = np.expand_dims(sample, axis=0)\n                \n                start_time = time.perf_counter()\n                try:\n                    output = onnx_session.run(None, {input_name: input_batch})[0]\n                    end_time = time.perf_counter()\n                    \n                    if len(output.shape) > 2:\n                        output = output.reshape(output.shape[0], -1)\n                        if output.shape[1] != 2:\n                            output = output[:, :2]\n                    \n                    probabilities = self._softmax(output[0])\n                    prediction = np.argmax(probabilities)\n                    \n                    results['onnx_predictions'].append(prediction)\n                    results['onnx_probabilities'].append(probabilities)\n                    results['onnx_times'].append(end_time - start_time)\n                    \n                except Exception as e:\n                    print(f\"ONNX inference failed for sample {i}: {e}\")\n                    results['onnx_predictions'].append(-1)\n                    results['onnx_probabilities'].append([0.5, 0.5])\n                    results['onnx_times'].append(0)\n        \n        return results\n    \n    def _softmax(self, x):\n        \"\"\"安全的softmax实现\"\"\"\n        exp_x = np.exp(x - np.max(x))\n        return exp_x / np.sum(exp_x)\n    \n    def analyze_errors(self, inference_results: Dict) -> Dict[str, Any]:\n        \"\"\"分析错误和分歧\"\"\"\n        print(f\"Analyzing errors and disagreements...\")\n        \n        pytorch_preds = np.array(inference_results['pytorch_predictions'])\n        onnx_preds = np.array(inference_results['onnx_predictions'])\n        ground_truth = np.array(inference_results['ground_truth'])\n        \n        # 计算准确率\n        pytorch_accuracy = np.mean(pytorch_preds == ground_truth) * 100\n        onnx_accuracy = np.mean(onnx_preds == ground_truth) * 100\n        \n        # 计算分歧\n        disagreements = np.sum(pytorch_preds != onnx_preds)\n        \n        # 计算错误模式\n        pytorch_errors = np.sum(pytorch_preds != ground_truth)\n        onnx_errors = np.sum(onnx_preds != ground_truth)\n        \n        # 假阳性和假阴性\n        pytorch_fp = np.sum((pytorch_preds == 1) & (ground_truth == 0))\n        pytorch_fn = np.sum((pytorch_preds == 0) & (ground_truth == 1))\n        onnx_fp = np.sum((onnx_preds == 1) & (ground_truth == 0))\n        onnx_fn = np.sum((onnx_preds == 0) & (ground_truth == 1))\n        \n        analysis = {\n            'pytorch_accuracy': pytorch_accuracy,\n            'onnx_accuracy': onnx_accuracy,\n            'pytorch_errors': pytorch_errors,\n            'onnx_errors': onnx_errors,\n            'disagreements': disagreements,\n            'error_patterns': {\n                'pytorch': {'false_positives': int(pytorch_fp), 'false_negatives': int(pytorch_fn)},\n                'onnx': {'false_positives': int(onnx_fp), 'false_negatives': int(onnx_fn)}\n            }\n        }\n        \n        return analysis\n    \n    def create_error_visualization(self, test_data: np.ndarray, inference_results: Dict, error_analysis: Dict) -> str:\n        \"\"\"创建错误样本可视化\"\"\"\n        print(\"Creating error sample visualizations...\")\n        \n        pytorch_preds = np.array(inference_results['pytorch_predictions'])\n        onnx_preds = np.array(inference_results['onnx_predictions'])\n        ground_truth = np.array(inference_results['ground_truth'])\n        \n        # 找到错误样本\n        pytorch_error_indices = np.where(pytorch_preds != ground_truth)[0][:12]\n        disagreement_indices = np.where(pytorch_preds != onnx_preds)[0][:12]\n        \n        fig, axes = plt.subplots(2, 6, figsize=(18, 8))\n        fig.suptitle(f'{self.model_name} Error Analysis', fontsize=16, fontweight='bold')\n        \n        # 第一行：PyTorch错误样本\n        for i in range(6):\n            ax = axes[0, i]\n            if i < len(pytorch_error_indices):\n                idx = pytorch_error_indices[i]\n                img_data = test_data[idx]\n                \n                if len(img_data.shape) == 3:\n                    img_data = np.transpose(img_data, (1, 2, 0))\n                \n                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())\n                \n                if img_data.shape[2] == 3:\n                    img_data = np.mean(img_data, axis=2)\n                \n                ax.imshow(img_data, cmap='gray')\n                ax.set_title(f'PyTorch Error\\nPred: {pytorch_preds[idx]} | GT: {ground_truth[idx]}', fontsize=10)\n            else:\n                ax.text(0.5, 0.5, 'No more\\nerrors', ha='center', va='center')\n            ax.axis('off')\n        \n        # 第二行：模型分歧样本\n        for i in range(6):\n            ax = axes[1, i]\n            if i < len(disagreement_indices):\n                idx = disagreement_indices[i]\n                img_data = test_data[idx]\n                \n                if len(img_data.shape) == 3:\n                    img_data = np.transpose(img_data, (1, 2, 0))\n                \n                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())\n                \n                if img_data.shape[2] == 3:\n                    img_data = np.mean(img_data, axis=2)\n                \n                ax.imshow(img_data, cmap='gray')\n                ax.set_title(f'Disagreement\\nPyT: {pytorch_preds[idx]} | ONNX: {onnx_preds[idx]} | GT: {ground_truth[idx]}', fontsize=9)\n            else:\n                ax.text(0.5, 0.5, 'No more\\ndisagreements', ha='center', va='center')\n            ax.axis('off')\n        \n        plt.tight_layout()\n        \n        # 保存可视化\n        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n        fig_path = self.output_dir / f\"error_analysis_{timestamp}.png\"\n        plt.savefig(fig_path, dpi=300, bbox_inches='tight')\n        plt.close()\n        \n        return str(fig_path)\n    \n    def generate_recommendations(self, error_analysis: Dict, inference_results: Dict) -> Dict[str, Any]:\n        \"\"\"生成改进建议\"\"\"\n        print(f\"Generating improvement recommendations...\")\n        \n        pytorch_times = np.array(inference_results['pytorch_times'])\n        onnx_times = np.array(inference_results['onnx_times'])\n        speedup = np.mean(pytorch_times) / np.mean(onnx_times) if np.mean(onnx_times) > 0 else 0\n        \n        recommendations = {\n            'model_name': self.model_name,\n            'architecture_type': self.model_config.get('architecture_type', 'Unknown'),\n            'priority_issues': [],\n            'performance_improvements': [],\n            'accuracy_improvements': [],\n            'onnx_optimization': [],\n            'overall_assessment': ''\n        }\n        \n        total_samples = len(inference_results['ground_truth'])\n        pytorch_accuracy = error_analysis['pytorch_accuracy']\n        disagreements = error_analysis['disagreements']\n        \n        # 优先级问题识别\n        if disagreements > total_samples * 0.05:\n            recommendations['priority_issues'].append({\n                'issue': 'High PyTorch-ONNX disagreement rate',\n                'severity': 'HIGH',\n                'description': f'{disagreements}/{total_samples} samples ({disagreements/total_samples*100:.1f}%) show different predictions'\n            })\n        \n        if speedup < 1.5:\n            recommendations['priority_issues'].append({\n                'issue': 'Poor ONNX performance optimization',\n                'severity': 'MEDIUM',\n                'description': f'ONNX speedup only {speedup:.2f}x, expected >2x'\n            })\n        \n        if pytorch_accuracy < 95:\n            recommendations['priority_issues'].append({\n                'issue': 'Low model accuracy on test data',\n                'severity': 'HIGH',\n                'description': f'PyTorch accuracy {pytorch_accuracy:.1f}% below expected performance'\n            })\n        \n        # 性能改进建议\n        if speedup < 2:\n            recommendations['performance_improvements'].append(\n                \"Consider quantization techniques (INT8) to improve ONNX inference speed\"\n            )\n            recommendations['performance_improvements'].append(\n                \"Explore ONNX Runtime optimization options (graph optimization, execution providers)\"\n            )\n        \n        # 准确性改进建议\n        false_positives = error_analysis['error_patterns']['pytorch']['false_positives']\n        false_negatives = error_analysis['error_patterns']['pytorch']['false_negatives']\n        \n        if false_positives > false_negatives * 1.5:\n            recommendations['accuracy_improvements'].append(\n                \"High false positive rate - consider adjusting decision threshold or improving negative class representation\"\n            )\n        elif false_negatives > false_positives * 1.5:\n            recommendations['accuracy_improvements'].append(\n                \"High false negative rate - improve positive class detection through data augmentation\"\n            )\n        \n        # ONNX优化建议\n        if disagreements > 0:\n            recommendations['onnx_optimization'].append(\n                \"Investigate numerical precision differences between PyTorch and ONNX implementations\"\n            )\n        \n        # 总体评估\n        if len(recommendations['priority_issues']) == 0 and speedup > 3 and pytorch_accuracy > 97:\n            recommendations['overall_assessment'] = \"EXCELLENT - Model performs well with good ONNX optimization\"\n        elif len(recommendations['priority_issues']) <= 1 and speedup > 2 and pytorch_accuracy > 95:\n            recommendations['overall_assessment'] = \"GOOD - Minor improvements needed\"\n        elif len(recommendations['priority_issues']) <= 2:\n            recommendations['overall_assessment'] = \"FAIR - Several improvements recommended\"\n        else:\n            recommendations['overall_assessment'] = \"NEEDS IMPROVEMENT - Critical issues need attention\"\n        \n        return recommendations\n    \n    def generate_html_report(self, inference_results: Dict, error_analysis: Dict,\n                           recommendations: Dict, visualization_path: str) -> str:\n        \"\"\"生成HTML报告\"\"\"\n        print(f\"Generating HTML report...\")\n        \n        # 编码图片\n        chart_base64 = \"\"\n        try:\n            with open(visualization_path, 'rb') as f:\n                chart_base64 = base64.b64encode(f.read()).decode('utf-8')\n        except:\n            pass\n        \n        # 计算关键指标\n        pytorch_times = np.array(inference_results['pytorch_times']) * 1000\n        onnx_times = np.array(inference_results['onnx_times']) * 1000\n        speedup = np.mean(pytorch_times) / np.mean(onnx_times) if np.mean(onnx_times) > 0 else 0\n        \n        html_content = f\"\"\"\n<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n    <title>{self.model_name} - 详细分析报告</title>\n    <style>\n        body {{\n            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n            line-height: 1.6;\n            color: #333;\n            max-width: 1400px;\n            margin: 0 auto;\n            padding: 20px;\n            background-color: #f8f9fa;\n        }}\n        \n        .header {{\n            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n            color: white;\n            padding: 40px;\n            border-radius: 15px;\n            text-align: center;\n            margin-bottom: 30px;\n            box-shadow: 0 8px 16px rgba(0,0,0,0.1);\n        }}\n        \n        .header h1 {{\n            margin: 0;\n            font-size: 3em;\n            font-weight: 300;\n        }}\n        \n        .metrics-grid {{\n            display: grid;\n            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));\n            gap: 20px;\n            margin-bottom: 30px;\n        }}\n        \n        .metric-card {{\n            background: white;\n            padding: 25px;\n            border-radius: 12px;\n            box-shadow: 0 4px 12px rgba(0,0,0,0.1);\n            text-align: center;\n        }}\n        \n        .metric-card h3 {{\n            margin: 0 0 15px 0;\n            color: #495057;\n            font-size: 1.1em;\n        }}\n        \n        .metric-card .value {{\n            font-size: 2.5em;\n            font-weight: 700;\n            margin-bottom: 10px;\n        }}\n        \n        .value-excellent {{ color: #28a745; }}\n        .value-good {{ color: #17a2b8; }}\n        .value-warning {{ color: #ffc107; }}\n        .value-danger {{ color: #dc3545; }}\n        \n        .card {{\n            background: white;\n            border-radius: 12px;\n            box-shadow: 0 4px 12px rgba(0,0,0,0.1);\n            margin-bottom: 25px;\n            overflow: hidden;\n        }}\n        \n        .card-header {{\n            background: #f8f9fa;\n            padding: 20px 25px;\n            border-bottom: 1px solid #dee2e6;\n            font-weight: 600;\n            font-size: 1.3em;\n            color: #495057;\n        }}\n        \n        .card-body {{\n            padding: 25px;\n        }}\n        \n        .image-container {{\n            text-align: center;\n            margin: 20px 0;\n        }}\n        \n        .image-container img {{\n            max-width: 100%;\n            height: auto;\n            border-radius: 8px;\n            box-shadow: 0 2px 10px rgba(0,0,0,0.1);\n        }}\n        \n        .priority-high {{\n            border-left: 4px solid #dc3545;\n            background: #f8d7da;\n            padding: 15px;\n            margin: 10px 0;\n            border-radius: 4px;\n        }}\n        \n        .priority-medium {{\n            border-left: 4px solid #ffc107;\n            background: #fff3cd;\n            padding: 15px;\n            margin: 10px 0;\n            border-radius: 4px;\n        }}\n        \n        .recommendation-list {{\n            list-style: none;\n            padding: 0;\n        }}\n        \n        .recommendation-list li {{\n            padding: 10px;\n            margin: 5px 0;\n            background: #f8f9fa;\n            border-radius: 6px;\n            border-left: 3px solid #007bff;\n        }}\n        \n        .assessment-excellent {{\n            background: #d4edda;\n            color: #155724;\n            padding: 20px;\n            border-radius: 8px;\n            text-align: center;\n            font-size: 1.2em;\n            font-weight: 600;\n        }}\n        \n        .assessment-good {{\n            background: #d1ecf1;\n            color: #0c5460;\n            padding: 20px;\n            border-radius: 8px;\n            text-align: center;\n            font-size: 1.2em;\n            font-weight: 600;\n        }}\n        \n        .assessment-fair {{\n            background: #fff3cd;\n            color: #856404;\n            padding: 20px;\n            border-radius: 8px;\n            text-align: center;\n            font-size: 1.2em;\n            font-weight: 600;\n        }}\n        \n        .assessment-poor {{\n            background: #f8d7da;\n            color: #721c24;\n            padding: 20px;\n            border-radius: 8px;\n            text-align: center;\n            font-size: 1.2em;\n            font-weight: 600;\n        }}\n        \n        table {{\n            width: 100%;\n            border-collapse: collapse;\n            margin-top: 15px;\n        }}\n        \n        th, td {{\n            padding: 12px;\n            text-align: left;\n            border-bottom: 1px solid #dee2e6;\n        }}\n        \n        th {{\n            background-color: #f8f9fa;\n            font-weight: 600;\n        }}\n    </style>\n</head>\n<body>\n    <div class=\"header\">\n        <h1>{self.model_name.replace('_', ' ').title()}</h1>\n        <div class=\"subtitle\">详细性能分析报告</div>\n        <div class=\"subtitle\">{self.model_config.get('architecture_type', 'Unknown Architecture')}</div>\n        <div class=\"subtitle\">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>\n    </div>\n    \n    <div class=\"metrics-grid\">\n        <div class=\"metric-card\">\n            <h3>PyTorch准确率</h3>\n            <div class=\"value {'value-excellent' if error_analysis['pytorch_accuracy'] > 97 else 'value-good' if error_analysis['pytorch_accuracy'] > 95 else 'value-warning' if error_analysis['pytorch_accuracy'] > 90 else 'value-danger'}\">\n                {error_analysis['pytorch_accuracy']:.1f}%\n            </div>\n        </div>\n        <div class=\"metric-card\">\n            <h3>ONNX准确率</h3>\n            <div class=\"value {'value-excellent' if error_analysis['onnx_accuracy'] > 97 else 'value-good' if error_analysis['onnx_accuracy'] > 95 else 'value-warning' if error_analysis['onnx_accuracy'] > 90 else 'value-danger'}\">\n                {error_analysis['onnx_accuracy']:.1f}%\n            </div>\n        </div>\n        <div class=\"metric-card\">\n            <h3>性能加速比</h3>\n            <div class=\"value {'value-excellent' if speedup > 4 else 'value-good' if speedup > 2 else 'value-warning' if speedup > 1 else 'value-danger'}\">\n                {speedup:.2f}x\n            </div>\n        </div>\n        <div class=\"metric-card\">\n            <h3>模型分歧数</h3>\n            <div class=\"value {'value-excellent' if error_analysis['disagreements'] == 0 else 'value-good' if error_analysis['disagreements'] < 10 else 'value-warning' if error_analysis['disagreements'] < 25 else 'value-danger'}\">\n                {error_analysis['disagreements']}\n            </div>\n        </div>\n    </div>\n    \n    <div class=\"card\">\n        <div class=\"card-header\">总体评估</div>\n        <div class=\"card-body\">\n            <div class=\"assessment-{'excellent' if 'EXCELLENT' in recommendations['overall_assessment'] else 'good' if 'GOOD' in recommendations['overall_assessment'] else 'fair' if 'FAIR' in recommendations['overall_assessment'] else 'poor'}\">\n                {recommendations['overall_assessment']}\n            </div>\n        </div>\n    </div>\n\"\"\"\n        \n        # 添加优先级问题\n        if recommendations['priority_issues']:\n            html_content += \"\"\"\n    <div class=\"card\">\n        <div class=\"card-header\">优先级问题</div>\n        <div class=\"card-body\">\n\"\"\"\n            for issue in recommendations['priority_issues']:\n                priority_class = f\"priority-{issue['severity'].lower()}\"\n                html_content += f\"\"\"\n            <div class=\"{priority_class}\">\n                <h4>{issue['issue']} ({issue['severity']})</h4>\n                <p><strong>描述:</strong> {issue['description']}</p>\n            </div>\n\"\"\"\n            html_content += \"\"\"\n        </div>\n    </div>\n\"\"\"\n        \n        # 添加错误样本分析\n        if chart_base64:\n            html_content += f\"\"\"\n    <div class=\"card\">\n        <div class=\"card-header\">错误样本分析</div>\n        <div class=\"card-body\">\n            <div class=\"image-container\">\n                <img src=\"data:image/png;base64,{chart_base64}\" alt=\"错误样本分析\" />\n            </div>\n        </div>\n    </div>\n\"\"\"\n        \n        # 添加改进建议\n        html_content += \"\"\"\n    <div class=\"card\">\n        <div class=\"card-header\">改进建议</div>\n        <div class=\"card-body\">\n\"\"\"\n        \n        if recommendations['performance_improvements']:\n            html_content += \"\"\"\n            <h4>性能改进</h4>\n            <ul class=\"recommendation-list\">\n\"\"\"\n            for rec in recommendations['performance_improvements']:\n                html_content += f\"<li>{rec}</li>\"\n            html_content += \"</ul>\"\n        \n        if recommendations['accuracy_improvements']:\n            html_content += \"\"\"\n            <h4>准确性改进</h4>\n            <ul class=\"recommendation-list\">\n\"\"\"\n            for rec in recommendations['accuracy_improvements']:\n                html_content += f\"<li>{rec}</li>\"\n            html_content += \"</ul>\"\n        \n        if recommendations['onnx_optimization']:\n            html_content += \"\"\"\n            <h4>ONNX优化</h4>\n            <ul class=\"recommendation-list\">\n\"\"\"\n            for rec in recommendations['onnx_optimization']:\n                html_content += f\"<li>{rec}</li>\"\n            html_content += \"</ul>\"\n        \n        html_content += f\"\"\"\n        </div>\n    </div>\n    \n    <div class=\"card\">\n        <div class=\"card-header\">详细统计</div>\n        <div class=\"card-body\">\n            <table>\n                <tr><th>指标</th><th>PyTorch</th><th>ONNX</th></tr>\n                <tr><td>准确率</td><td>{error_analysis['pytorch_accuracy']:.2f}%</td><td>{error_analysis['onnx_accuracy']:.2f}%</td></tr>\n                <tr><td>错误数</td><td>{error_analysis['pytorch_errors']}</td><td>{error_analysis['onnx_errors']}</td></tr>\n                <tr><td>平均推理时间</td><td>{np.mean(pytorch_times):.2f} ms</td><td>{np.mean(onnx_times):.2f} ms</td></tr>\n                <tr><td>加速比</td><td colspan=\"2\">{speedup:.2f}x</td></tr>\n            </table>\n            \n            <h4>错误模式分析</h4>\n            <table>\n                <tr><th>错误类型</th><th>PyTorch</th><th>ONNX</th></tr>\n                <tr><td>假阳性</td><td>{error_analysis['error_patterns']['pytorch']['false_positives']}</td><td>{error_analysis['error_patterns']['onnx']['false_positives']}</td></tr>\n                <tr><td>假阴性</td><td>{error_analysis['error_patterns']['pytorch']['false_negatives']}</td><td>{error_analysis['error_patterns']['onnx']['false_negatives']}</td></tr>\n                <tr><td>模型分歧</td><td colspan=\"2\">{error_analysis['disagreements']} 样本</td></tr>\n            </table>\n        </div>\n    </div>\n</body>\n</html>\n        \"\"\"\n        \n        # 保存HTML报告\n        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n        html_path = self.output_dir / f\"detailed_analysis_{timestamp}.html\"\n        \n        with open(html_path, 'w', encoding='utf-8') as f:\n            f.write(html_content)\n        \n        print(f\"HTML报告已保存: {html_path}\")\n        return str(html_path)\n    \n    def run_complete_analysis(self) -> Dict[str, Any]:\n        \"\"\"运行完整分析\"\"\"\n        print(f\"\\n开始分析 {self.model_name}...\")\n        print(f\"{'='*60}\")\n        \n        try:\n            # 1. 加载模型\n            pytorch_model, onnx_session = self.load_models()\n            if pytorch_model is None:\n                return {'success': False, 'error': 'Failed to load PyTorch model'}\n            \n            # 2. 准备测试数据\n            test_data, test_labels = self.prepare_test_data(200)\n            \n            # 3. 运行推理分析\n            inference_results = self.run_inference_analysis(\n                pytorch_model, onnx_session, test_data, test_labels\n            )\n            \n            # 4. 分析错误\n            error_analysis = self.analyze_errors(inference_results)\n            \n            # 5. 创建可视化\n            visualization_path = self.create_error_visualization(test_data, inference_results, error_analysis)\n            \n            # 6. 生成建议\n            recommendations = self.generate_recommendations(error_analysis, inference_results)\n            \n            # 7. 生成HTML报告\n            html_report_path = self.generate_html_report(\n                inference_results, error_analysis, recommendations, visualization_path\n            )\n            \n            print(f\"\\n{self.model_name} 分析完成!\")\n            print(f\"HTML报告: {html_report_path}\")\n            \n            return {\n                'success': True,\n                'model_name': self.model_name,\n                'html_report': html_report_path,\n                'visualization': visualization_path,\n                'recommendations': recommendations,\n                'summary': {\n                    'pytorch_accuracy': error_analysis['pytorch_accuracy'],\n                    'onnx_accuracy': error_analysis['onnx_accuracy'],\n                    'speedup': np.mean(inference_results['pytorch_times']) / np.mean(inference_results['onnx_times']) if np.mean(inference_results['onnx_times']) > 0 else 0,\n                    'disagreements': error_analysis['disagreements'],\n                    'overall_assessment': recommendations['overall_assessment']\n                }\n            }\n            \n        except Exception as e:\n            print(f\"分析过程中出现错误: {e}\")\n            import traceback\n            traceback.print_exc()\n            return {'success': False, 'error': str(e)}\n\ndef main():\n    \"\"\"分析ResNet18-Improved模型\"\"\"\n    \n    model_config = {\n        'module': 'models.resnet_improved',\n        'factory_function': 'create_resnet18_improved',\n        'input_shape': (3, 70, 70),\n        'architecture_type': 'CNN',\n        'checkpoint_path': 'experiments/experiment_20250802_164948/resnet18_improved/best_model.pth'\n    }\n    \n    analyzer = SimpleModelAnalyzer('resnet18_improved', model_config)\n    result = analyzer.run_complete_analysis()\n    \n    if result['success']:\n        summary = result['summary']\n        print(f\"\\n{'='*60}\")\n        print(f\"ResNet18-Improved 分析结果:\")\n        print(f\"{'='*60}\")\n        print(f\"PyTorch准确率: {summary['pytorch_accuracy']:.1f}%\")\n        print(f\"ONNX准确率: {summary['onnx_accuracy']:.1f}%\")\n        print(f\"性能加速比: {summary['speedup']:.2f}x\")\n        print(f\"模型分歧数: {summary['disagreements']}\")\n        print(f\"总体评估: {summary['overall_assessment']}\")\n        print(f\"\\n详细报告: {result['html_report']}\")\n        print(f\"错误样本可视化: {result['visualization']}\")\n    else:\n        print(f\"分析失败: {result['error']}\")\n\nif __name__ == \"__main__\":\n    main()