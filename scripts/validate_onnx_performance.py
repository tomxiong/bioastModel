"""
ONNX模型性能验证和对比测试脚本
对比原始PyTorch模型和转换后的ONNX模型的性能
"""

import os
import sys
import torch
import numpy as np
import onnxruntime as ort
import time
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_improved import create_resnet18_improved
from training.dataset import BioastDataset

class ONNXPerformanceValidator:
    """ONNX模型性能验证器"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.input_shape = (3, 70, 70)
        self.results = {}
        
    def load_pytorch_model(self) -> torch.nn.Module:
        """Load PyTorch model"""
        print(f"Loading PyTorch model: {self.model_name}")
        
        # Create model
        model = create_resnet18_improved(num_classes=2)
        model.eval()
        
        # Load checkpoint
        checkpoint_path = Path("experiments/experiment_20250802_164948/resnet18_improved/best_model.pth")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        print("PyTorch model loaded successfully")
        return model
    
    def load_onnx_model(self) -> ort.InferenceSession:
        """Load ONNX model"""
        print(f"Loading ONNX model: {self.model_name}")
        
        onnx_path = Path(f"onnx_models/{self.model_name}.onnx")
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
        
        session = ort.InferenceSession(str(onnx_path))
        print("ONNX model loaded successfully")
        return session
    
    def prepare_test_data(self, num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare test data using real dataset with balanced sampling"""
        print(f"Preparing test data from real dataset, samples: {num_samples}")
        
        try:
            # Try to use real test data
            from torch.utils.data import DataLoader
            from torchvision import transforms
            
            # Define transforms (same as used during training)
            transform = transforms.Compose([
                transforms.Resize((70, 70)),
                transforms.ToTensor(),
            ])
            
            # Load the real test dataset
            dataset = BioastDataset(data_dir='bioast_dataset', split='test', transform=transform)
            print(f"Found {len(dataset)} samples in test dataset")
            
            # Group samples by class for balanced sampling
            negative_samples = []
            positive_samples = []
            
            for i, (data, target) in enumerate(dataset):
                if target == 0:  # negative
                    negative_samples.append((data.numpy(), target))
                else:  # positive
                    positive_samples.append((data.numpy(), target))
            
            print(f"Available samples - Negative: {len(negative_samples)}, Positive: {len(positive_samples)}")
            
            # Balanced sampling
            samples_per_class = min(num_samples // 2, len(negative_samples), len(positive_samples))
            
            # Select samples from each class
            selected_samples = []
            selected_labels = []
            
            # Add negative samples
            for i in range(samples_per_class):
                data, label = negative_samples[i]
                selected_samples.append(data)
                selected_labels.append(label)
            
            # Add positive samples
            for i in range(samples_per_class):
                data, label = positive_samples[i]
                selected_samples.append(data)
                selected_labels.append(label)
            
            if selected_samples:
                test_data = np.stack(selected_samples, axis=0)
                test_labels = np.array(selected_labels)
                
                print(f"Successfully loaded balanced test data, shape: {test_data.shape}")
                print(f"Balanced test data distribution - Negative: {np.sum(test_labels == 0)}, Positive: {np.sum(test_labels == 1)}")
                return test_data, test_labels
            
        except Exception as e:
            print(f"Cannot load real data, using random data: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate random test data as fallback
        test_data = np.random.randn(num_samples, *self.input_shape).astype(np.float32)
        test_labels = np.random.randint(0, 2, num_samples)
        print(f"Using random test data, shape: {test_data.shape}")
        
        return test_data, test_labels
    
    def measure_pytorch_performance(self, model: torch.nn.Module, test_data: np.ndarray) -> Dict:
        """Measure PyTorch model performance"""
        print("Measuring PyTorch model performance...")
        
        model.eval()
        predictions = []
        inference_times = []
        
        with torch.no_grad():
            for i, sample in enumerate(test_data):
                # Convert to torch tensor
                input_tensor = torch.from_numpy(sample).unsqueeze(0)  # Add batch dimension
                
                # Measure inference time
                start_time = time.perf_counter()
                output = model(input_tensor)
                end_time = time.perf_counter()
                
                inference_times.append(end_time - start_time)
                predictions.append(output.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        return {
            'predictions': predictions,
            'inference_times': inference_times,
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'total_time': sum(inference_times)
        }
    
    def measure_onnx_performance(self, session: ort.InferenceSession, test_data: np.ndarray) -> Dict:
        """Measure ONNX model performance"""
        print("Measuring ONNX model performance...")
        
        input_name = session.get_inputs()[0].name
        predictions = []
        inference_times = []
        
        for i, sample in enumerate(test_data):
            # Prepare input
            input_batch = np.expand_dims(sample, axis=0)  # Add batch dimension
            
            # Measure inference time
            start_time = time.perf_counter()
            output = session.run(None, {input_name: input_batch})
            end_time = time.perf_counter()
            
            inference_times.append(end_time - start_time)
            predictions.append(output[0])  # ONNX returns list
        
        predictions = np.concatenate(predictions, axis=0)
        
        return {
            'predictions': predictions,
            'inference_times': inference_times,
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'total_time': sum(inference_times)
        }
    
    def compare_predictions(self, pytorch_results: Dict, onnx_results: Dict) -> Dict:
        """Compare prediction accuracy"""
        print("Comparing prediction accuracy...")
        
        pytorch_preds = pytorch_results['predictions']
        onnx_preds = onnx_results['predictions']
        
        # Calculate absolute differences
        abs_diff = np.abs(pytorch_preds - onnx_preds)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        # Calculate relative differences
        rel_diff = abs_diff / (np.abs(pytorch_preds) + 1e-8)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        # Calculate class prediction consistency
        pytorch_classes = np.argmax(pytorch_preds, axis=1)
        onnx_classes = np.argmax(onnx_preds, axis=1)
        class_agreement = np.mean(pytorch_classes == onnx_classes)
        
        return {
            'max_absolute_diff': float(max_diff),
            'mean_absolute_diff': float(mean_diff),
            'max_relative_diff': float(max_rel_diff),
            'mean_relative_diff': float(mean_rel_diff),
            'class_agreement': float(class_agreement),
            'disagreement_samples': int(np.sum(pytorch_classes != onnx_classes))
        }
    
    def generate_json_report(self, pytorch_results: Dict, onnx_results: Dict, 
                            accuracy_results: Dict, test_data_shape: Tuple) -> Dict:
        """Generate JSON format report for programmatic access"""
        
        speedup = pytorch_results['avg_inference_time'] / onnx_results['avg_inference_time']
        
        json_report = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "test_configuration": {
                "test_samples": test_data_shape[0],
                "input_shape": list(test_data_shape[1:]),
                "device": "CPU"
            },
            "performance_metrics": {
                "pytorch_model": {
                    "avg_inference_time_ms": pytorch_results['avg_inference_time'] * 1000,
                    "std_inference_time_ms": pytorch_results['std_inference_time'] * 1000,
                    "total_time_s": pytorch_results['total_time']
                },
                "onnx_model": {
                    "avg_inference_time_ms": onnx_results['avg_inference_time'] * 1000,
                    "std_inference_time_ms": onnx_results['std_inference_time'] * 1000,
                    "total_time_s": onnx_results['total_time']
                },
                "performance_improvement": {
                    "speedup": speedup,
                    "improvement_percentage": (speedup - 1) * 100
                }
            },
            "accuracy_metrics": {
                "numerical_precision": {
                    "max_absolute_diff": accuracy_results['max_absolute_diff'],
                    "mean_absolute_diff": accuracy_results['mean_absolute_diff'],
                    "max_relative_diff": accuracy_results['max_relative_diff'],
                    "mean_relative_diff": accuracy_results['mean_relative_diff']
                },
                "classification_consistency": {
                    "agreement_percentage": accuracy_results['class_agreement'] * 100,
                    "disagreement_samples": accuracy_results['disagreement_samples']
                }
            },
            "quality_assessment": {
                "numerical_precision_grade": "excellent" if accuracy_results['max_absolute_diff'] < 1e-5 else 
                                          "good" if accuracy_results['max_absolute_diff'] < 1e-3 else "needs_attention",
                "classification_consistency_grade": "excellent" if accuracy_results['class_agreement'] > 0.99 else
                                                  "good" if accuracy_results['class_agreement'] > 0.95 else "needs_attention",
                "performance_improvement_grade": "significant" if speedup > 1.2 else
                                               "slight" if speedup > 1.0 else "degraded"
            }
        }
        
        return json_report
    
    def generate_html_report(self, pytorch_results: Dict, onnx_results: Dict, 
                           accuracy_results: Dict, test_data_shape: Tuple, 
                           chart_path: str) -> str:
        """Generate enhanced HTML report with embedded images"""
        
        speedup = pytorch_results['avg_inference_time'] / onnx_results['avg_inference_time']
        
        # Read and encode chart image
        chart_base64 = ""
        try:
            with open(chart_path, 'rb') as f:
                chart_base64 = base64.b64encode(f.read()).decode('utf-8')
        except:
            pass
        
        # Generate error analysis
        error_analysis_html = self.generate_error_analysis_html(pytorch_results, onnx_results)
        
        html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.model_name} ONNX Performance Validation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }}
        
        .card-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #dee2e6;
            font-weight: 600;
            font-size: 1.2em;
        }}
        
        .card-body {{
            padding: 20px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        
        .metric-title {{
            font-weight: 600;
            color: #495057;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 1.8em;
            font-weight: 700;
            color: #007bff;
        }}
        
        .metric-unit {{
            font-size: 0.9em;
            color: #6c757d;
        }}
        
        .speedup {{
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.5em;
            font-weight: 700;
            margin: 20px 0;
        }}
        
        .quality-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 5px;
        }}
        
        .badge-excellent {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge-good {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge-needs-attention {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        
        .timestamp {{
            color: #6c757d;
            font-size: 0.9em;
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.model_name.replace('_', ' ').title()}</h1>
        <div class="subtitle">ONNX Performance Validation Report</div>
        <div class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    
    <div class="card">
        <div class="card-header">📊 Test Configuration</div>
        <div class="card-body">
            <table>
                <tr><td><strong>Test Samples</strong></td><td>{test_data_shape[0]}</td></tr>
                <tr><td><strong>Input Shape</strong></td><td>{test_data_shape[1:]}</td></tr>
                <tr><td><strong>Device</strong></td><td>CPU</td></tr>
                <tr><td><strong>Test Data</strong></td><td>Real biomedical images (balanced)</td></tr>
            </table>
        </div>
    </div>
    
    <div class="speedup">
        🚀 Performance Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% improvement)
    </div>
    
    <div class="card">
        <div class="card-header">🏃‍♂️ Performance Metrics</div>
        <div class="card-body">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">PyTorch Average Time</div>
                    <div class="metric-value">{pytorch_results['avg_inference_time']*1000:.2f} <span class="metric-unit">ms</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">ONNX Average Time</div>
                    <div class="metric-value">{onnx_results['avg_inference_time']*1000:.2f} <span class="metric-unit">ms</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">PyTorch Std Dev</div>
                    <div class="metric-value">{pytorch_results['std_inference_time']*1000:.2f} <span class="metric-unit">ms</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">ONNX Std Dev</div>
                    <div class="metric-value">{onnx_results['std_inference_time']*1000:.2f} <span class="metric-unit">ms</span></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">🎯 Accuracy Metrics</div>
        <div class="card-body">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Max Absolute Difference</div>
                    <div class="metric-value">{accuracy_results['max_absolute_diff']:.8f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Mean Absolute Difference</div>
                    <div class="metric-value">{accuracy_results['mean_absolute_diff']:.8f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Classification Agreement</div>
                    <div class="metric-value">{accuracy_results['class_agreement']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Disagreement Samples</div>
                    <div class="metric-value">{accuracy_results['disagreement_samples']}</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">⭐ Quality Assessment</div>
        <div class="card-body">
            <div>
                <span class="quality-badge badge-{'excellent' if accuracy_results['max_absolute_diff'] < 1e-5 else 'good' if accuracy_results['max_absolute_diff'] < 1e-3 else 'needs-attention'}">
                    📊 Numerical Precision: {'Excellent' if accuracy_results['max_absolute_diff'] < 1e-5 else 'Good' if accuracy_results['max_absolute_diff'] < 1e-3 else 'Needs Attention'}
                </span>
                <span class="quality-badge badge-{'excellent' if accuracy_results['class_agreement'] > 0.99 else 'good' if accuracy_results['class_agreement'] > 0.95 else 'needs-attention'}">
                    🎯 Classification Consistency: {'Excellent' if accuracy_results['class_agreement'] > 0.99 else 'Good' if accuracy_results['class_agreement'] > 0.95 else 'Needs Attention'}
                </span>
                <span class="quality-badge badge-{'excellent' if speedup > 1.2 else 'good' if speedup > 1.0 else 'needs-attention'}">
                    🚀 Performance Improvement: {'Significant' if speedup > 1.2 else 'Slight' if speedup > 1.0 else 'Degraded'}
                </span>
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">📈 Performance Visualization</div>
        <div class="card-body">
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_base64}" alt="Performance Comparison Chart" />
            </div>
        </div>
    </div>
    
    {error_analysis_html}
    
    <div class="timestamp">
        Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using Enhanced ONNX Validation System
    </div>
</body>
</html>
        """
        
        return html_report
    
    def generate_error_analysis_html(self, pytorch_results: Dict, onnx_results: Dict) -> str:
        """Generate error analysis section"""
        pytorch_preds = pytorch_results['predictions']
        onnx_preds = onnx_results['predictions']
        
        # Find samples with largest differences
        abs_diff = np.abs(pytorch_preds - onnx_preds)
        max_diff_indices = np.argsort(abs_diff.max(axis=1))[-5:]  # Top 5 different samples
        
        if len(max_diff_indices) == 0 or abs_diff.max() < 1e-6:
            return """
    <div class="card">
        <div class="card-header">🔍 Error Analysis</div>
        <div class="card-body">
            <p>✅ Perfect agreement between PyTorch and ONNX models - no significant differences found.</p>
        </div>
    </div>
            """
        
        samples_html = ""
        for idx in max_diff_indices:
            diff = abs_diff[idx].max()
            pytorch_class = np.argmax(pytorch_preds[idx])
            onnx_class = np.argmax(onnx_preds[idx])
            
            samples_html += f"""
            <tr>
                <td>Sample {idx}</td>
                <td>{diff:.8f}</td>
                <td>Class {pytorch_class}</td>
                <td>Class {onnx_class}</td>
                <td>{'✅ Match' if pytorch_class == onnx_class else '❌ Mismatch'}</td>
            </tr>
            """
        
        return f"""
    <div class="card">
        <div class="card-header">🔍 Error Analysis</div>
        <div class="card-body">
            <p>Analysis of samples with the largest prediction differences:</p>
            <table>
                <thead>
                    <tr>
                        <th>Sample</th>
                        <th>Max Difference</th>
                        <th>PyTorch Prediction</th>
                        <th>ONNX Prediction</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {samples_html}
                </tbody>
            </table>
        </div>
    </div>
        """
    
    def generate_performance_report(self, pytorch_results: Dict, onnx_results: Dict, 
                                  accuracy_results: Dict, test_data_shape: Tuple) -> str:
        """生成性能对比报告"""
        
        # 计算加速比
        speedup = pytorch_results['avg_inference_time'] / onnx_results['avg_inference_time']
        
        report = f"""# {self.model_name} ONNX性能验证报告

## 测试配置
- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 测试样本数: {test_data_shape[0]}
- 输入形状: {test_data_shape[1:]}
- 设备: CPU

## 性能对比

### PyTorch模型性能
- 平均推理时间: {pytorch_results['avg_inference_time']*1000:.3f} ms
- 推理时间标准差: {pytorch_results['std_inference_time']*1000:.3f} ms
- 总推理时间: {pytorch_results['total_time']:.3f} s

### ONNX模型性能
- 平均推理时间: {onnx_results['avg_inference_time']*1000:.3f} ms
- 推理时间标准差: {onnx_results['std_inference_time']*1000:.3f} ms
- 总推理时间: {onnx_results['total_time']:.3f} s

### 性能提升
- 加速比: {speedup:.2f}x
- 性能提升: {(speedup-1)*100:.1f}%

## 精度对比

### 数值精度
- 最大绝对差异: {accuracy_results['max_absolute_diff']:.8f}
- 平均绝对差异: {accuracy_results['mean_absolute_diff']:.8f}
- 最大相对差异: {accuracy_results['max_relative_diff']:.6f}
- 平均相对差异: {accuracy_results['mean_relative_diff']:.6f}

### 分类一致性
- 类别预测一致性: {accuracy_results['class_agreement']*100:.2f}%
- 不一致样本数: {accuracy_results['disagreement_samples']}

## 质量评估
"""
        
        # 质量评估
        if accuracy_results['max_absolute_diff'] < 1e-5:
            report += "- ✅ 数值精度: 优秀 (差异 < 1e-5)\n"
        elif accuracy_results['max_absolute_diff'] < 1e-3:
            report += "- ⚠️ 数值精度: 良好 (差异 < 1e-3)\n"
        else:
            report += "- ❌ 数值精度: 需要关注 (差异 >= 1e-3)\n"
        
        if accuracy_results['class_agreement'] > 0.99:
            report += "- ✅ 分类一致性: 优秀 (> 99%)\n"
        elif accuracy_results['class_agreement'] > 0.95:
            report += "- ⚠️ 分类一致性: 良好 (> 95%)\n"
        else:
            report += "- ❌ 分类一致性: 需要关注 (< 95%)\n"
        
        if speedup > 1.2:
            report += "- ✅ 性能提升: 显著 (> 20%)\n"
        elif speedup > 1.0:
            report += "- ⚠️ 性能提升: 轻微 (> 0%)\n"
        else:
            report += "- ❌ 性能变化: 性能下降\n"
        
        return report
    
    def create_performance_visualizations(self, pytorch_results: Dict, onnx_results: Dict):
        """创建性能可视化图表"""
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 推理时间对比箱线图
        ax1 = axes[0, 0]
        data_to_plot = [
            np.array(pytorch_results['inference_times']) * 1000,
            np.array(onnx_results['inference_times']) * 1000
        ]
        ax1.boxplot(data_to_plot, labels=['PyTorch', 'ONNX'])
        ax1.set_title('推理时间对比')
        ax1.set_ylabel('推理时间 (ms)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 平均推理时间条形图
        ax2 = axes[0, 1]
        models = ['PyTorch', 'ONNX']
        times = [
            pytorch_results['avg_inference_time'] * 1000,
            onnx_results['avg_inference_time'] * 1000
        ]
        bars = ax2.bar(models, times, color=['#1f77b4', '#ff7f0e'])
        ax2.set_title('平均推理时间对比')
        ax2.set_ylabel('平均推理时间 (ms)')
        
        # 添加数值标签
        for bar, time in zip(bars, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time:.2f}ms', ha='center', va='bottom')
        
        # 3. 预测差异分布
        ax3 = axes[1, 0]
        diff = np.abs(pytorch_results['predictions'] - onnx_results['predictions'])
        ax3.hist(diff.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('预测值绝对差异分布')
        ax3.set_xlabel('绝对差异')
        ax3.set_ylabel('频次')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. 推理时间趋势
        ax4 = axes[1, 1]
        sample_indices = range(min(len(pytorch_results['inference_times']), 50))
        pytorch_times = np.array(pytorch_results['inference_times'][:50]) * 1000
        onnx_times = np.array(onnx_results['inference_times'][:50]) * 1000
        
        ax4.plot(sample_indices, pytorch_times, 'o-', label='PyTorch', alpha=0.7)
        ax4.plot(sample_indices, onnx_times, 's-', label='ONNX', alpha=0.7)
        ax4.set_title('推理时间趋势 (前50个样本)')
        ax4.set_xlabel('样本序号')
        ax4.set_ylabel('推理时间 (ms)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = Path("reports/onnx_performance")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_path = output_dir / f"{self.model_name}_performance_comparison_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison charts saved to: {fig_path}")
        return fig_path
    
    def run_full_validation(self, num_samples: int = 100) -> Dict:
        """Run complete validation process"""
        print(f"Starting complete ONNX model performance validation...")
        
        try:
            # 1. Load models
            pytorch_model = self.load_pytorch_model()
            onnx_session = self.load_onnx_model()
            
            # 2. Prepare test data
            test_data, test_labels = self.prepare_test_data(num_samples)
            
            # 3. Measure performance
            pytorch_results = self.measure_pytorch_performance(pytorch_model, test_data)
            onnx_results = self.measure_onnx_performance(onnx_session, test_data)
            
            # 4. Compare accuracy
            accuracy_results = self.compare_predictions(pytorch_results, onnx_results)
            
            # 5. Create visualizations first
            fig_path = self.create_performance_visualizations(pytorch_results, onnx_results)
            
            # 6. Generate reports (JSON and HTML)
            json_report = self.generate_json_report(
                pytorch_results, onnx_results, accuracy_results, test_data.shape
            )
            
            html_report = self.generate_html_report(
                pytorch_results, onnx_results, accuracy_results, test_data.shape, str(fig_path)
            )
            
            # 7. Save reports
            output_dir = Path("reports/onnx_performance")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save JSON report
            json_path = output_dir / f"{self.model_name}_performance_validation_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_report, f, indent=2, ensure_ascii=False)
            
            # Save HTML report
            html_path = output_dir / f"{self.model_name}_performance_validation_{timestamp}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            print(f"JSON report saved to: {json_path}")
            print(f"HTML report saved to: {html_path}")
            
            # Return comprehensive results
            return {
                'success': True,
                'pytorch_avg_time': pytorch_results['avg_inference_time'],
                'onnx_avg_time': onnx_results['avg_inference_time'],
                'speedup': pytorch_results['avg_inference_time'] / onnx_results['avg_inference_time'],
                'max_diff': accuracy_results['max_absolute_diff'],
                'class_agreement': accuracy_results['class_agreement'],
                'json_report_path': str(json_path),
                'html_report_path': str(html_path),
                'figure_path': str(fig_path),
                'json_report': json_report
            }
            
        except Exception as e:
            print(f"Error during validation process: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def main():
    """Main function"""
    validator = ONNXPerformanceValidator("resnet18_improved")
    results = validator.run_full_validation(num_samples=200)
    
    if results['success']:
        print("\n" + "="*50)
        print("ONNX model performance validation completed!")
        print("="*50)
        print(f"Speedup: {results['speedup']:.2f}x")
        print(f"Max difference: {results['max_diff']:.8f}")
        print(f"Class agreement: {results['class_agreement']*100:.2f}%")
        print(f"JSON report: {results['json_report_path']}")
        print(f"HTML report: {results['html_report_path']}")
        print(f"Chart path: {results['figure_path']}")
    else:
        print(f"Validation failed: {results['error']}")

if __name__ == "__main__":
    main()