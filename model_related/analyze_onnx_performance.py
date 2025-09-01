#!/usr/bin/env python3
"""
分析和对比已转换ONNX模型的性能
"""

import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_data_loader import create_real_data_loaders

# ONNX模型目录
ONNX_MODELS_DIR = "/home/aaa/ws/bioastModel/onnx_models"

def get_onnx_models():
    """获取所有ONNX模型文件"""
    if not os.path.exists(ONNX_MODELS_DIR):
        return []
    
    onnx_files = []
    for file in os.listdir(ONNX_MODELS_DIR):
        if file.endswith('.onnx'):
            onnx_files.append(os.path.join(ONNX_MODELS_DIR, file))
    return sorted(onnx_files)

def extract_model_info(onnx_path):
    """从ONNX文件路径中提取模型信息"""
    filename = os.path.basename(onnx_path)
    model_name = filename.replace('.onnx', '')
    
    # 提取模型类型
    if 'airbubble_hybrid_net' in filename:
        model_type = 'airbubble_hybrid_net'
    elif 'coatnet' in filename:
        model_type = 'coatnet'
    elif 'convnext_micro' in filename:
        model_type = 'convnext_micro'
    elif 'densenet_compact' in filename:
        model_type = 'densenet_compact'
    elif 'efficient_cnn' in filename:
        model_type = 'efficient_cnn'
    elif 'efficientnet' in filename:
        model_type = 'efficientnet'
    elif 'inception_micro' in filename:
        model_type = 'inception_micro'
    elif 'mic_mobilenetv3' in filename:
        model_type = 'mic_mobilenetv3'
    elif 'resnet_micro' in filename:
        model_type = 'resnet_micro'
    elif 'simplified_airbubble_detector' in filename:
        model_type = 'simplified_airbubble_detector'
    else:
        model_type = 'unknown'
    
    return {
        'filename': filename,
        'model_name': model_name,
        'model_type': model_type,
        'file_path': onnx_path
    }

def load_onnx_model(onnx_path):
    """加载ONNX模型"""
    try:
        # 创建推理会话
        session = ort.InferenceSession(onnx_path)
        
        # 获取输入输出信息
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        return {
            'session': session,
            'input_name': input_info.name,
            'input_shape': input_info.shape,
            'output_name': output_info.name,
            'output_shape': output_info.shape,
            'success': True
        }
    except Exception as e:
        return {
            'session': None,
            'error': str(e),
            'success': False
        }

def test_model_inference_speed(session, input_name, num_tests=100):
    """测试模型推理速度"""
    # 创建测试输入
    test_input = np.random.randn(1, 3, 70, 70).astype(np.float32)
    
    # 预热
    for _ in range(10):
        session.run(None, {input_name: test_input})
    
    # 测试推理时间
    times = []
    for _ in range(num_tests):
        start_time = time.time()
        session.run(None, {input_name: test_input})
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    return {
        'avg_inference_time_ms': np.mean(times),
        'std_inference_time_ms': np.std(times),
        'min_inference_time_ms': np.min(times),
        'max_inference_time_ms': np.max(times),
        'fps': 1000 / np.mean(times)
    }

def test_model_accuracy(session, input_name, data_loader, max_batches=50):
    """测试模型在真实数据上的准确率"""
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    try:
        for i, (images, labels) in enumerate(data_loader):
            if i >= max_batches:
                break
            
            # 转换为numpy数组
            images_np = images.numpy()
            labels_np = labels.numpy()
            
            # 批量推理
            for j in range(images_np.shape[0]):
                input_data = np.expand_dims(images_np[j], axis=0)
                output = session.run(None, {input_name: input_data})[0]
                
                # 获取预测结果
                predicted = np.argmax(output, axis=1)[0]
                true_label = labels_np[j]
                
                predictions.append(predicted)
                true_labels.append(true_label)
                
                if predicted == true_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions,
            'true_labels': true_labels,
            'success': True
        }
    except Exception as e:
        return {
            'accuracy': 0,
            'error': str(e),
            'success': False
        }

def get_model_file_size(onnx_path):
    """获取模型文件大小"""
    try:
        size_bytes = os.path.getsize(onnx_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except:
        return 0

def analyze_onnx_models():
    """分析所有ONNX模型的性能"""
    print("🚀 开始分析ONNX模型性能...")
    
    # 获取所有ONNX模型
    onnx_models = get_onnx_models()
    print(f"📊 找到 {len(onnx_models)} 个ONNX模型")
    
    if not onnx_models:
        print("❌ 未找到任何ONNX模型")
        return
    
    # 加载真实数据
    try:
        _, val_loader, test_loader = create_real_data_loaders(batch_size=32)
        print("✅ 成功加载真实数据")
    except Exception as e:
        print(f"❌ 加载真实数据失败: {str(e)}")
        val_loader = None
        test_loader = None
    
    # 分析每个模型
    results = []
    
    for i, onnx_path in enumerate(onnx_models, 1):
        print(f"\n{'='*80}")
        print(f"🔍 分析模型 {i}/{len(onnx_models)}: {os.path.basename(onnx_path)}")
        print(f"{'='*80}")
        
        # 提取模型信息
        model_info = extract_model_info(onnx_path)
        
        # 获取文件大小
        file_size_mb = get_model_file_size(onnx_path)
        
        # 加载模型
        model_data = load_onnx_model(onnx_path)
        
        if not model_data['success']:
            print(f"❌ 模型加载失败: {model_data['error']}")
            results.append({
                **model_info,
                'file_size_mb': file_size_mb,
                'load_success': False,
                'load_error': model_data['error']
            })
            continue
        
        print(f"✅ 模型加载成功")
        print(f"   输入形状: {model_data['input_shape']}")
        print(f"   输出形状: {model_data['output_shape']}")
        print(f"   文件大小: {file_size_mb:.2f} MB")
        
        # 测试推理速度
        print("🚀 测试推理速度...")
        speed_results = test_model_inference_speed(
            model_data['session'], 
            model_data['input_name']
        )
        print(f"   平均推理时间: {speed_results['avg_inference_time_ms']:.2f} ms")
        print(f"   推理速度: {speed_results['fps']:.1f} FPS")
        
        # 测试准确率
        accuracy_results = {'success': False}
        if val_loader is not None:
            print("📊 测试验证集准确率...")
            accuracy_results = test_model_accuracy(
                model_data['session'],
                model_data['input_name'],
                val_loader
            )
            if accuracy_results['success']:
                print(f"   验证集准确率: {accuracy_results['accuracy']:.4f}")
            else:
                print(f"   准确率测试失败: {accuracy_results.get('error', 'Unknown error')}")
        
        # 记录结果
        result = {
            **model_info,
            'file_size_mb': file_size_mb,
            'load_success': True,
            'input_shape': model_data['input_shape'],
            'output_shape': model_data['output_shape'],
            'speed_results': speed_results,
            'accuracy_results': accuracy_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
        results.append(result)
    
    return results

def generate_performance_comparison(results):
    """生成性能对比报告"""
    print("\n" + "="*80)
    print("📊 ONNX模型性能对比报告")
    print("="*80)
    
    # 过滤成功加载的模型
    successful_results = [r for r in results if r.get('load_success', False)]
    
    if not successful_results:
        print("❌ 没有成功加载的模型")
        return
    
    print(f"成功分析的模型数量: {len(successful_results)}")
    
    # 创建对比表格
    comparison_data = []
    for result in successful_results:
        speed = result.get('speed_results', {})
        accuracy = result.get('accuracy_results', {})
        
        comparison_data.append({
            '模型类型': result['model_type'],
            '文件名': result['filename'],
            '文件大小(MB)': f"{result['file_size_mb']:.2f}",
            '平均推理时间(ms)': f"{speed.get('avg_inference_time_ms', 0):.2f}",
            '推理速度(FPS)': f"{speed.get('fps', 0):.1f}",
            '验证准确率': f"{accuracy.get('accuracy', 0):.4f}" if accuracy.get('success') else 'N/A'
        })
    
    # 打印对比表格
    df = pd.DataFrame(comparison_data)
    print("\n📋 性能对比表:")
    print(df.to_string(index=False))
    
    # 找出最佳模型
    print("\n🏆 性能排名:")
    
    # 按推理速度排序
    speed_sorted = sorted(successful_results, 
                         key=lambda x: x.get('speed_results', {}).get('fps', 0), 
                         reverse=True)
    print("\n⚡ 推理速度排名 (FPS):")
    for i, result in enumerate(speed_sorted[:5], 1):
        fps = result.get('speed_results', {}).get('fps', 0)
        print(f"{i}. {result['model_type']}: {fps:.1f} FPS")
    
    # 按准确率排序
    accuracy_sorted = [r for r in successful_results 
                      if r.get('accuracy_results', {}).get('success', False)]
    accuracy_sorted.sort(key=lambda x: x.get('accuracy_results', {}).get('accuracy', 0), 
                        reverse=True)
    
    if accuracy_sorted:
        print("\n🎯 准确率排名:")
        for i, result in enumerate(accuracy_sorted[:5], 1):
            acc = result.get('accuracy_results', {}).get('accuracy', 0)
            print(f"{i}. {result['model_type']}: {acc:.4f}")
    
    # 按文件大小排序
    size_sorted = sorted(successful_results, 
                        key=lambda x: x.get('file_size_mb', 0))
    print("\n💾 文件大小排名 (最小):")
    for i, result in enumerate(size_sorted[:5], 1):
        size = result.get('file_size_mb', 0)
        print(f"{i}. {result['model_type']}: {size:.2f} MB")
    
    # 综合评分
    print("\n🏅 综合评分 (速度 + 准确率 - 文件大小):")
    scored_results = []
    for result in successful_results:
        speed_score = result.get('speed_results', {}).get('fps', 0) / 100  # 归一化
        accuracy_score = result.get('accuracy_results', {}).get('accuracy', 0) if result.get('accuracy_results', {}).get('success') else 0
        size_penalty = result.get('file_size_mb', 0) / 100  # 文件大小惩罚
        
        composite_score = speed_score + accuracy_score - size_penalty
        scored_results.append((result, composite_score))
    
    scored_results.sort(key=lambda x: x[1], reverse=True)
    for i, (result, score) in enumerate(scored_results[:5], 1):
        print(f"{i}. {result['model_type']}: {score:.4f}")
    
    return successful_results

def save_performance_report(results):
    """保存性能报告"""
    # 转换numpy类型为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 转换结果中的numpy类型
    converted_results = convert_numpy_types(results)
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_models': len(results),
        'successful_models': len([r for r in results if r.get('load_success', False)]),
        'failed_models': len([r for r in results if not r.get('load_success', False)]),
        'detailed_results': converted_results
    }
    
    # 保存JSON报告
    with open('onnx_performance_analysis.json', 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 详细性能报告已保存至: onnx_performance_analysis.json")
    
    # 生成HTML报告
    generate_html_report(results)

def generate_html_report(results):
    """生成HTML格式的性能报告"""
    successful_results = [r for r in results if r.get('load_success', False)]
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ONNX模型性能分析报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .summary-card h3 {{ margin-top: 0; color: #2c3e50; }}
        .summary-card .value {{ font-size: 1.5em; font-weight: bold; color: #3498db; }}
        .performance-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .performance-table th, .performance-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .performance-table th {{ background-color: #3498db; color: white; }}
        .performance-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .ranking {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .ranking h3 {{ color: #2c3e50; margin-top: 0; }}
        .ranking ol {{ padding-left: 20px; }}
        .ranking li {{ margin: 10px 0; }}
        .best-model {{ background: #d5f4e6; border-left: 4px solid #27ae60; }}
        .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 ONNX模型性能分析报告</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>📁 总模型数</h3>
                <div class="value">{len(results)}</div>
            </div>
            <div class="summary-card">
                <h3>✅ 成功分析</h3>
                <div class="value">{len(successful_results)}</div>
            </div>
            <div class="summary-card">
                <h3>❌ 分析失败</h3>
                <div class="value">{len(results) - len(successful_results)}</div>
            </div>
        </div>
        
        <h2>📋 性能对比表</h2>
        <table class="performance-table">
            <thead>
                <tr>
                    <th>模型类型</th>
                    <th>文件名</th>
                    <th>文件大小 (MB)</th>
                    <th>平均推理时间 (ms)</th>
                    <th>推理速度 (FPS)</th>
                    <th>验证准确率</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for result in successful_results:
        speed = result.get('speed_results', {})
        accuracy = result.get('accuracy_results', {})
        
        html_content += f"""
                <tr>
                    <td><strong>{result['model_type']}</strong></td>
                    <td>{result['filename']}</td>
                    <td>{result['file_size_mb']:.2f}</td>
                    <td>{speed.get('avg_inference_time_ms', 0):.2f}</td>
                    <td>{speed.get('fps', 0):.1f}</td>
                    <td>{accuracy.get('accuracy', 0):.4f if accuracy.get('success') else 'N/A'}</td>
                </tr>
        """
    
    # 添加排名部分
    speed_sorted = sorted(successful_results, 
                         key=lambda x: x.get('speed_results', {}).get('fps', 0), 
                         reverse=True)
    
    accuracy_sorted = [r for r in successful_results 
                      if r.get('accuracy_results', {}).get('success', False)]
    accuracy_sorted.sort(key=lambda x: x.get('accuracy_results', {}).get('accuracy', 0), 
                        reverse=True)
    
    html_content += f"""
            </tbody>
        </table>
        
        <div class="ranking">
            <h3>⚡ 推理速度排名</h3>
            <ol>
    """
    
    for result in speed_sorted[:5]:
        fps = result.get('speed_results', {}).get('fps', 0)
        html_content += f"<li>{result['model_type']}: {fps:.1f} FPS</li>"
    
    html_content += """
            </ol>
        </div>
    """
    
    if accuracy_sorted:
        html_content += """
        <div class="ranking">
            <h3>🎯 准确率排名</h3>
            <ol>
        """
        
        for result in accuracy_sorted[:5]:
            acc = result.get('accuracy_results', {}).get('accuracy', 0)
            html_content += f"<li>{result['model_type']}: {acc:.4f}</li>"
        
        html_content += """
            </ol>
        </div>
        """
    
    html_content += f"""
        <div class="timestamp">
            <p>报告生成时间: {datetime.now().isoformat()}</p>
        </div>
    </div>
</body>
</html>
    """
    
    with open("onnx_performance_analysis.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 HTML性能报告已保存至: onnx_performance_analysis.html")

def main():
    """主函数"""
    try:
        # 分析ONNX模型性能
        results = analyze_onnx_models()
        
        if results:
            # 生成性能对比报告
            generate_performance_comparison(results)
            
            # 保存报告
            save_performance_report(results)
            
            print("\n🎉 ONNX模型性能分析完成!")
        else:
            print("❌ 没有找到可分析的ONNX模型")
            
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()