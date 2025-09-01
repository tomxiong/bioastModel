#!/usr/bin/env python3
"""
ONNX Model Performance Testing
对ONNX模型进行性能测试，确认转换后性能未下降
"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time
import json
import os
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trainers.train_simplified_airbubble_detector_fixed import FixedSimplifiedAirBubbleDetector

class ONNXPerformanceTester:
    def __init__(self, pytorch_checkpoint, onnx_model_path):
        self.pytorch_checkpoint = pytorch_checkpoint
        self.onnx_model_path = onnx_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Initializing ONNX Performance Tester")
        print(f"📱 Device: {self.device}")
        print(f"📁 PyTorch Checkpoint: {pytorch_checkpoint}")
        print(f"📁 ONNX Model: {onnx_model_path}")
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载模型
        self.pytorch_model = self.load_pytorch_model()
        self.onnx_session = self.load_onnx_model()
        
    def load_pytorch_model(self):
        """加载PyTorch模型"""
        print("📥 Loading PyTorch model...")
        model = FixedSimplifiedAirBubbleDetector(num_classes=2).to(self.device)
        checkpoint = torch.load(self.pytorch_checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ PyTorch model loaded successfully")
        return model
    
    def load_onnx_model(self):
        """加载ONNX模型"""
        print("📥 Loading ONNX model...")
        session = ort.InferenceSession(self.onnx_model_path)
        print("✅ ONNX model loaded successfully")
        return session
    
    def prepare_test_data(self):
        """准备测试数据"""
        print("📊 Preparing test data...")
        
        test_samples = []
        # 收集测试样本
        for label, folder in [(0, 'bioast_dataset/positive/test'), (1, 'bioast_dataset/negative/test')]:
            if os.path.exists(folder):
                for img_file in os.listdir(folder):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_samples.append((os.path.join(folder, img_file), label))
        
        print(f"📊 Found {len(test_samples)} test samples")
        return test_samples
    
    def inference_speed_test(self, num_samples=1000):
        """推理速度测试"""
        print(f"⚡ Running inference speed test with {num_samples} samples...")
        
        # 创建随机测试数据
        test_input_torch = torch.randn(1, 3, 70, 70).to(self.device)
        test_input_numpy = test_input_torch.cpu().numpy().astype(np.float32)
        
        # PyTorch推理速度测试
        pytorch_times = []
        with torch.no_grad():
            # 预热
            for _ in range(10):
                _ = self.pytorch_model(test_input_torch)
            
            # 正式测试
            for _ in range(num_samples):
                start_time = time.time()
                _ = self.pytorch_model(test_input_torch)
                end_time = time.time()
                pytorch_times.append((end_time - start_time) * 1000)  # ms
        
        # ONNX推理速度测试
        onnx_times = []
        input_name = self.onnx_session.get_inputs()[0].name
        
        # 预热
        for _ in range(10):
            _ = self.onnx_session.run(None, {input_name: test_input_numpy})
        
        # 正式测试
        for _ in range(num_samples):
            start_time = time.time()
            _ = self.onnx_session.run(None, {input_name: test_input_numpy})
            end_time = time.time()
            onnx_times.append((end_time - start_time) * 1000)  # ms
        
        pytorch_avg = np.mean(pytorch_times)
        pytorch_std = np.std(pytorch_times)
        onnx_avg = np.mean(onnx_times)
        onnx_std = np.std(onnx_times)
        
        speedup = pytorch_avg / onnx_avg
        
        print(f"📊 PyTorch Average: {pytorch_avg:.3f} ± {pytorch_std:.3f} ms")
        print(f"📊 ONNX Average: {onnx_avg:.3f} ± {onnx_std:.3f} ms")
        print(f"🚀 Speedup: {speedup:.2f}x")
        
        return {
            'pytorch_avg_ms': pytorch_avg,
            'pytorch_std_ms': pytorch_std,
            'onnx_avg_ms': onnx_avg,
            'onnx_std_ms': onnx_std,
            'speedup': speedup
        }
    
    def accuracy_comparison_test(self):
        """准确率对比测试"""
        print("🎯 Running accuracy comparison test...")
        
        test_samples = self.prepare_test_data()
        
        pytorch_predictions = []
        onnx_predictions = []
        true_labels = []
        
        input_name = self.onnx_session.get_inputs()[0].name
        
        print(f"🔍 Testing on {len(test_samples)} samples...")
        
        for i, (img_path, true_label) in enumerate(test_samples):
            if i % 500 == 0:
                print(f"Progress: {i}/{len(test_samples)}")
            
            try:
                # 加载和预处理图像
                image = Image.open(img_path).convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0)
                input_numpy = input_tensor.numpy().astype(np.float32)
                
                # PyTorch推理
                with torch.no_grad():
                    pytorch_output = self.pytorch_model(input_tensor.to(self.device))
                    pytorch_pred = torch.argmax(pytorch_output, dim=1).cpu().item()
                
                # ONNX推理
                onnx_output = self.onnx_session.run(None, {input_name: input_numpy})[0]
                onnx_pred = np.argmax(onnx_output, axis=1)[0]
                
                pytorch_predictions.append(pytorch_pred)
                onnx_predictions.append(onnx_pred)
                true_labels.append(true_label)
                
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
                continue
        
        # 计算准确率
        pytorch_accuracy = accuracy_score(true_labels, pytorch_predictions)
        onnx_accuracy = accuracy_score(true_labels, onnx_predictions)
        
        # 计算预测一致性
        consistency = accuracy_score(pytorch_predictions, onnx_predictions)
        
        print(f"📊 PyTorch Accuracy: {pytorch_accuracy:.4f} ({pytorch_accuracy*100:.2f}%)")
        print(f"📊 ONNX Accuracy: {onnx_accuracy:.4f} ({onnx_accuracy*100:.2f}%)")
        print(f"🔄 Prediction Consistency: {consistency:.4f} ({consistency*100:.2f}%)")
        
        # 详细分析
        pytorch_report = classification_report(true_labels, pytorch_predictions, output_dict=True)
        onnx_report = classification_report(true_labels, onnx_predictions, output_dict=True)
        
        return {
            'pytorch_accuracy': pytorch_accuracy,
            'onnx_accuracy': onnx_accuracy,
            'prediction_consistency': consistency,
            'pytorch_report': pytorch_report,
            'onnx_report': onnx_report,
            'total_samples': len(true_labels)
        }
    
    def numerical_precision_test(self, num_tests=100):
        """数值精度测试"""
        print(f"🔢 Running numerical precision test with {num_tests} random inputs...")
        
        max_diffs = []
        mean_diffs = []
        
        input_name = self.onnx_session.get_inputs()[0].name
        
        for i in range(num_tests):
            # 生成随机输入
            test_input_torch = torch.randn(1, 3, 70, 70).to(self.device)
            test_input_numpy = test_input_torch.cpu().numpy().astype(np.float32)
            
            # PyTorch推理
            with torch.no_grad():
                pytorch_output = self.pytorch_model(test_input_torch).cpu().numpy()
            
            # ONNX推理
            onnx_output = self.onnx_session.run(None, {input_name: test_input_numpy})[0]
            
            # 计算差异
            diff = np.abs(pytorch_output - onnx_output)
            max_diffs.append(np.max(diff))
            mean_diffs.append(np.mean(diff))
        
        overall_max_diff = np.max(max_diffs)
        overall_mean_diff = np.mean(mean_diffs)
        
        print(f"📊 Maximum difference: {overall_max_diff:.8f}")
        print(f"📊 Average mean difference: {overall_mean_diff:.8f}")
        
        # 判断精度是否可接受 (深度学习模型转换中0.001以下的差异是可接受的)
        precision_acceptable = overall_max_diff < 1e-3
        print(f"✅ Precision acceptable: {precision_acceptable}")
        
        return {
            'max_difference': overall_max_diff,
            'mean_difference': overall_mean_diff,
            'precision_acceptable': precision_acceptable,
            'max_diffs': max_diffs,
            'mean_diffs': mean_diffs
        }
    
    def memory_usage_test(self):
        """内存使用测试"""
        print("💾 Running memory usage test...")
        
        import psutil
        import gc
        
        # 获取当前进程
        process = psutil.Process()
        
        # 测试PyTorch内存使用
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # PyTorch推理
        test_input = torch.randn(10, 3, 70, 70).to(self.device)
        with torch.no_grad():
            for _ in range(100):
                _ = self.pytorch_model(test_input)
        
        memory_pytorch = process.memory_info().rss / 1024 / 1024  # MB
        
        # 清理
        del test_input
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # ONNX推理
        test_input_numpy = np.random.randn(10, 3, 70, 70).astype(np.float32)
        input_name = self.onnx_session.get_inputs()[0].name
        
        for _ in range(100):
            _ = self.onnx_session.run(None, {input_name: test_input_numpy})
        
        memory_onnx = process.memory_info().rss / 1024 / 1024  # MB
        
        pytorch_usage = memory_pytorch - memory_before
        onnx_usage = memory_onnx - memory_pytorch
        
        print(f"📊 PyTorch Memory Usage: {pytorch_usage:.2f} MB")
        print(f"📊 ONNX Memory Usage: {onnx_usage:.2f} MB")
        
        return {
            'pytorch_memory_mb': pytorch_usage,
            'onnx_memory_mb': onnx_usage,
            'memory_ratio': pytorch_usage / onnx_usage if onnx_usage > 0 else float('inf')
        }
    
    def generate_performance_report(self, speed_results, accuracy_results, precision_results, memory_results):
        """生成性能测试报告"""
        print("📊 Generating performance test report...")
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            else:
                return obj
        
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'model_info': {
                'pytorch_checkpoint': self.pytorch_checkpoint,
                'onnx_model': self.onnx_model_path,
                'onnx_model_size_mb': float(os.path.getsize(self.onnx_model_path) / (1024 * 1024))
            },
            'speed_test': convert_numpy_types(speed_results),
            'accuracy_test': convert_numpy_types(accuracy_results),
            'precision_test': convert_numpy_types(precision_results),
            'memory_test': convert_numpy_types(memory_results),
            'summary': {
                'performance_degradation': bool(accuracy_results['onnx_accuracy'] < accuracy_results['pytorch_accuracy'] - 0.001),
                'speed_improvement': bool(speed_results['speedup'] > 1.0),
                'precision_maintained': bool(precision_results['precision_acceptable']),
                'overall_status': 'PASS' if (
                    accuracy_results['onnx_accuracy'] >= accuracy_results['pytorch_accuracy'] - 0.001 and
                    precision_results['precision_acceptable']
                ) else 'FAIL'
            }
        }
        
        # 生成带模型名和时间戳的报告文件名
        model_name = os.path.basename(self.pytorch_checkpoint).split('_')[0]  # 提取模型名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON报告
        report_path = f'reports/{model_name}_onnx_performance_{timestamp}.json'
        os.makedirs('reports', exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        html_path = f'reports/{model_name}_onnx_performance_{timestamp}.html'
        self.generate_html_report(report, html_path)
        
        print(f"📊 Performance test report saved to: {report_path}")
        print(f"📊 HTML report saved to: {html_path}")
        return report
    
    def generate_html_report(self, report, html_path):
        """生成HTML性能测试报告"""
        summary = report['summary']
        speed = report['speed_test']
        accuracy = report['accuracy_test']
        precision = report['precision_test']
        memory = report['memory_test']
        
        status_color = "#28a745" if summary['overall_status'] == 'PASS' else "#dc3545"
        status_icon = "✅" if summary['overall_status'] == 'PASS' else "❌"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ONNX Performance Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .status {{ background-color: {status_color}; color: white; padding: 15px; border-radius: 5px; margin: 20px 0; text-align: center; font-size: 18px; font-weight: bold; }}
        .metric-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        .comparison-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .comparison-table th, .comparison-table td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        .comparison-table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .better {{ color: #28a745; font-weight: bold; }}
        .worse {{ color: #dc3545; font-weight: bold; }}
        .equal {{ color: #6c757d; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 ONNX Performance Test Report</h1>
            <p><strong>Test Date:</strong> {report['test_timestamp']}</p>
            <p><strong>Model:</strong> SimplifiedAirBubbleDetector</p>
        </div>
        
        <div class="status">
            {status_icon} Overall Status: {summary['overall_status']}
        </div>
        
        <h2>📊 Performance Summary</h2>
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th>PyTorch</th>
                <th>ONNX</th>
                <th>Change</th>
                <th>Status</th>
            </tr>
            <tr>
                <td><strong>Accuracy</strong></td>
                <td>{accuracy['pytorch_accuracy']:.4f}</td>
                <td>{accuracy['onnx_accuracy']:.4f}</td>
                <td class="{'better' if accuracy['onnx_accuracy'] >= accuracy['pytorch_accuracy'] else 'worse'}">
                    {(accuracy['onnx_accuracy'] - accuracy['pytorch_accuracy'])*100:+.2f}%
                </td>
                <td>{'✅ Maintained' if accuracy['onnx_accuracy'] >= accuracy['pytorch_accuracy'] - 0.001 else '❌ Degraded'}</td>
            </tr>
            <tr>
                <td><strong>Inference Speed</strong></td>
                <td>{speed['pytorch_avg_ms']:.3f} ms</td>
                <td>{speed['onnx_avg_ms']:.3f} ms</td>
                <td class="{'better' if speed['speedup'] > 1 else 'worse'}">
                    {speed['speedup']:.2f}x faster
                </td>
                <td>{'✅ Improved' if speed['speedup'] > 1 else '❌ Slower'}</td>
            </tr>
            <tr>
                <td><strong>Memory Usage</strong></td>
                <td>{memory['pytorch_memory_mb']:.2f} MB</td>
                <td>{memory['onnx_memory_mb']:.2f} MB</td>
                <td class="{'better' if memory['onnx_memory_mb'] < memory['pytorch_memory_mb'] else 'worse'}">
                    {((memory['onnx_memory_mb'] - memory['pytorch_memory_mb']) / memory['pytorch_memory_mb'] * 100):+.1f}%
                </td>
                <td>{'✅ Reduced' if memory['onnx_memory_mb'] < memory['pytorch_memory_mb'] else '❌ Increased'}</td>
            </tr>
            <tr>
                <td><strong>Numerical Precision</strong></td>
                <td>Reference</td>
                <td>Max diff: {precision['max_difference']:.8f}</td>
                <td>-</td>
                <td>{'✅ Acceptable' if precision['precision_acceptable'] else '❌ Poor'}</td>
            </tr>
        </table>
        
        <h2>🎯 Detailed Results</h2>
        
        <div class="metric-card">
            <h3>⚡ Speed Performance</h3>
            <p><strong>PyTorch:</strong> {speed['pytorch_avg_ms']:.3f} ± {speed['pytorch_std_ms']:.3f} ms</p>
            <p><strong>ONNX:</strong> {speed['onnx_avg_ms']:.3f} ± {speed['onnx_std_ms']:.3f} ms</p>
            <p><strong>Speedup:</strong> {speed['speedup']:.2f}x</p>
        </div>
        
        <div class="metric-card">
            <h3>🎯 Accuracy Performance</h3>
            <p><strong>Test Samples:</strong> {accuracy['total_samples']}</p>
            <p><strong>PyTorch Accuracy:</strong> {accuracy['pytorch_accuracy']:.4f} ({accuracy['pytorch_accuracy']*100:.2f}%)</p>
            <p><strong>ONNX Accuracy:</strong> {accuracy['onnx_accuracy']:.4f} ({accuracy['onnx_accuracy']*100:.2f}%)</p>
            <p><strong>Prediction Consistency:</strong> {accuracy['prediction_consistency']:.4f} ({accuracy['prediction_consistency']*100:.2f}%)</p>
        </div>
        
        <div class="metric-card">
            <h3>🔢 Numerical Precision</h3>
            <p><strong>Maximum Difference:</strong> {precision['max_difference']:.8f}</p>
            <p><strong>Average Mean Difference:</strong> {precision['mean_difference']:.8f}</p>
            <p><strong>Precision Status:</strong> {'✅ Acceptable' if precision['precision_acceptable'] else '❌ Poor'}</p>
        </div>
        
        <div class="metric-card">
            <h3>💾 Memory Usage</h3>
            <p><strong>PyTorch Memory:</strong> {memory['pytorch_memory_mb']:.2f} MB</p>
            <p><strong>ONNX Memory:</strong> {memory['onnx_memory_mb']:.2f} MB</p>
            <p><strong>Memory Ratio:</strong> {memory['memory_ratio']:.2f}x</p>
        </div>
        
        <h2>📁 Model Information</h2>
        <div class="metric-card">
            <p><strong>ONNX Model Size:</strong> {report['model_info']['onnx_model_size_mb']:.2f} MB</p>
            <p><strong>PyTorch Checkpoint:</strong> {report['model_info']['pytorch_checkpoint']}</p>
            <p><strong>ONNX Model:</strong> {report['model_info']['onnx_model']}</p>
        </div>
        
        <h2>✅ Conclusion</h2>
        <div class="metric-card">
            <p><strong>Performance Status:</strong> {'✅ ONNX conversion successful with no performance degradation' if summary['overall_status'] == 'PASS' else '❌ ONNX conversion has performance issues'}</p>
            <p><strong>Recommendation:</strong> {'ONNX model is ready for production deployment' if summary['overall_status'] == 'PASS' else 'Review ONNX conversion settings and re-test'}</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def run_complete_test(self):
        """运行完整的性能测试"""
        print("🚀 Starting comprehensive ONNX performance test...")
        
        try:
            # 1. 推理速度测试
            print("\n" + "="*50)
            speed_results = self.inference_speed_test(num_samples=1000)
            
            # 2. 准确率对比测试
            print("\n" + "="*50)
            accuracy_results = self.accuracy_comparison_test()
            
            # 3. 数值精度测试
            print("\n" + "="*50)
            precision_results = self.numerical_precision_test(num_tests=100)
            
            # 4. 内存使用测试
            print("\n" + "="*50)
            memory_results = self.memory_usage_test()
            
            # 5. 生成报告
            print("\n" + "="*50)
            report = self.generate_performance_report(
                speed_results, accuracy_results, precision_results, memory_results
            )
            
            # 输出总结
            print("\n🎉 Performance Test Completed!")
            print(f"📊 Overall Status: {report['summary']['overall_status']}")
            print(f"🎯 Accuracy: PyTorch {accuracy_results['pytorch_accuracy']:.4f} vs ONNX {accuracy_results['onnx_accuracy']:.4f}")
            print(f"⚡ Speed: {speed_results['speedup']:.2f}x faster")
            print(f"🔢 Precision: {'Acceptable' if precision_results['precision_acceptable'] else 'Poor'}")
            print(f"📊 Reports: reports/onnx_performance_test.json & .html")
            
            return report
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    # 配置路径
    pytorch_checkpoint = "checkpoints/simplified_airbubble_detector_20250807_213233_best.pth"
    onnx_model = "onnx_models/simplified_airbubble_detector_20250807_220033.onnx"
    
    # 检查文件是否存在
    if not os.path.exists(pytorch_checkpoint):
        print(f"❌ PyTorch checkpoint not found: {pytorch_checkpoint}")
        return
    
    if not os.path.exists(onnx_model):
        print(f"❌ ONNX model not found: {onnx_model}")
        return
    
    # 运行性能测试
    tester = ONNXPerformanceTester(pytorch_checkpoint, onnx_model)
    report = tester.run_complete_test()
    
    if report and report['summary']['overall_status'] == 'PASS':
        print("\n✅ ONNX模型性能测试通过！转换后性能未下降。")
    else:
        print("\n❌ ONNX模型性能测试失败！需要检查转换设置。")

if __name__ == "__main__":
    main()