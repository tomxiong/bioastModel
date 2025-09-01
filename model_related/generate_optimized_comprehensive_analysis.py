#!/usr/bin/env python3
"""
优化的综合性能分析报告生成器
- 按模型名称合并重复记录
- 标记每次训练时间
- 识别每个模型的最佳性能
- 自动转换最佳模型为ONNX格式
- 进行转换前后性能对比测试
"""

import os
import sys
import json
import glob
import torch
import torch.onnx
import numpy as np
from datetime import datetime
import importlib
import traceback
from collections import defaultdict
import re

def parse_checkpoint_info():
    """解析所有checkpoint文件，按模型分组"""
    checkpoint_files = glob.glob('checkpoints/*.pth')
    models_data = defaultdict(list)
    
    for checkpoint in checkpoint_files:
        try:
            # 加载checkpoint获取性能数据
            ckpt = torch.load(checkpoint, map_location='cpu')
            
            filename = os.path.basename(checkpoint)
            # 解析文件名: model_name_YYYYMMDD_HHMMSS_best.pth
            parts = filename.replace('_best.pth', '').split('_')
            
            if len(parts) >= 3:
                # 提取模型名称和时间戳
                timestamp_parts = []
                model_name_parts = []
                
                for i, part in enumerate(parts):
                    if re.match(r'^\d{8}$', part):  # 日期格式 YYYYMMDD
                        model_name_parts = parts[:i]
                        timestamp_parts = parts[i:]
                        break
                
                if model_name_parts and timestamp_parts:
                    model_name = '_'.join(model_name_parts)
                    timestamp = '_'.join(timestamp_parts)
                    
                    # 转换时间戳为可读格式
                    try:
                        if len(timestamp_parts) >= 2:
                            date_str = timestamp_parts[0]
                            time_str = timestamp_parts[1]
                            formatted_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                        else:
                            formatted_time = timestamp
                    except:
                        formatted_time = timestamp
                    
                    model_info = {
                        'checkpoint_path': checkpoint,
                        'filename': filename,
                        'timestamp': timestamp,
                        'formatted_time': formatted_time,
                        'val_acc': float(ckpt.get('val_acc', 0)),
                        'train_acc': float(ckpt.get('train_acc', 0)),
                        'epoch': int(ckpt.get('epoch', 0)),
                        'file_size_mb': os.path.getsize(checkpoint) / (1024 * 1024)
                    }
                    
                    models_data[model_name].append(model_info)
                    
        except Exception as e:
            print(f"Error parsing {checkpoint}: {e}")
    
    return models_data

def get_existing_onnx_models():
    """获取已存在的ONNX模型"""
    onnx_files = glob.glob('onnx_models/*.onnx')
    onnx_models = {}
    
    for onnx_file in onnx_files:
        filename = os.path.basename(onnx_file)
        # 提取模型名称
        parts = filename.replace('.onnx', '').split('_')
        if len(parts) >= 2:
            # 找到时间戳位置
            for i, part in enumerate(parts):
                if re.match(r'^\d{8}$', part):
                    model_name = '_'.join(parts[:i])
                    break
            else:
                model_name = '_'.join(parts[:-1]) if len(parts) > 1 else parts[0]
            
            onnx_models[model_name] = {
                'onnx_path': onnx_file,
                'file_size_mb': os.path.getsize(onnx_file) / (1024 * 1024)
            }
    
    return onnx_models

def load_model_class(model_name):
    """加载模型类"""
    try:
        module = importlib.import_module(f'models.{model_name}')
        
        # 常见的模型类名模式
        possible_names = [
            model_name.title().replace('_', ''),
            model_name.upper(),
            model_name,
            'Model',
            'Net',
            'Network'
        ]
        
        for name in possible_names:
            if hasattr(module, name):
                return getattr(module, name)
        
        # 查找第一个nn.Module子类
        import torch.nn as nn
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                return attr
        
        raise ValueError(f"No suitable model class found in models.{model_name}")
        
    except Exception as e:
        print(f"Error loading model class for {model_name}: {e}")
        return None

def convert_best_model_to_onnx(model_name, best_checkpoint_info):
    """将最佳模型转换为ONNX格式并进行性能测试"""
    print(f"Converting best {model_name} model to ONNX...")
    
    try:
        # 加载模型类
        ModelClass = load_model_class(model_name)
        if ModelClass is None:
            return None
        
        # 创建模型实例
        try:
            model = ModelClass(num_classes=4)  # bioast_dataset有4个类别
        except:
            try:
                model = ModelClass()
            except Exception as e:
                print(f"Failed to create model instance: {e}")
                return None
        
        # 加载最佳checkpoint
        checkpoint = torch.load(best_checkpoint_info['checkpoint_path'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 创建测试输入
        dummy_input = torch.randn(1, 3, 70, 70)
        
        # 测试PyTorch模型前向传播
        with torch.no_grad():
            pytorch_output = model(dummy_input)
            if isinstance(pytorch_output, dict):
                if 'classification' in pytorch_output:
                    pytorch_output = pytorch_output['classification']
                else:
                    pytorch_output = list(pytorch_output.values())[0]
        
        # 生成ONNX文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        onnx_path = f"onnx_models/{model_name}_best_{timestamp}.onnx"
        
        # 转换为ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 性能测试
        performance_results = test_onnx_performance(
            model, onnx_path, dummy_input, pytorch_output
        )
        
        return {
            'onnx_path': onnx_path,
            'file_size_mb': os.path.getsize(onnx_path) / (1024 * 1024),
            'conversion_timestamp': timestamp,
            'performance_test': performance_results,
            'status': 'success'
        }
        
    except Exception as e:
        error_msg = f"ONNX conversion failed: {str(e)}"
        print(f"Error converting {model_name}: {error_msg}")
        traceback.print_exc()
        return {'status': 'failed', 'error': error_msg}

def test_onnx_performance(pytorch_model, onnx_path, test_input, pytorch_output):
    """测试ONNX模型性能并与PyTorch模型对比"""
    try:
        import onnxruntime as ort
        import time
        
        # 创建ONNX推理会话
        ort_session = ort.InferenceSession(onnx_path)
        
        # 准备输入数据
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        
        # 性能测试 - PyTorch
        pytorch_times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = pytorch_model(test_input)
            pytorch_times.append(time.time() - start_time)
        
        # 性能测试 - ONNX
        onnx_times = []
        for _ in range(100):
            start_time = time.time()
            _ = ort_session.run(None, ort_inputs)
            onnx_times.append(time.time() - start_time)
        
        # 获取ONNX输出进行精度对比
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # 计算精度差异
        pytorch_np = pytorch_output.detach().numpy()
        max_diff = np.max(np.abs(pytorch_np - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_np - onnx_output))
        
        # 计算性能统计
        pytorch_avg_time = np.mean(pytorch_times) * 1000  # ms
        onnx_avg_time = np.mean(onnx_times) * 1000  # ms
        speedup = pytorch_avg_time / onnx_avg_time
        
        return {
            'pytorch_avg_time_ms': float(pytorch_avg_time),
            'onnx_avg_time_ms': float(onnx_avg_time),
            'speedup_ratio': float(speedup),
            'max_accuracy_diff': float(max_diff),
            'mean_accuracy_diff': float(mean_diff),
            'accuracy_preserved': float(max_diff) < 1e-5
        }
        
    except ImportError:
        return {
            'error': 'onnxruntime not available for performance testing',
            'accuracy_preserved': 'unknown'
        }
    except Exception as e:
        return {
            'error': f'Performance test failed: {str(e)}',
            'accuracy_preserved': 'unknown'
        }

def generate_optimized_html_report(consolidated_data):
    """生成优化的HTML报告"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>生物医学图像分析 - 优化综合模型性能报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .summary-card p {{
            margin: 0;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .model-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .accuracy-excellent {{
            background-color: #d5f4e6;
            color: #27ae60;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .accuracy-good {{
            background-color: #fff3cd;
            color: #856404;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .accuracy-poor {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .status-success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-failed {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .training-history {{
            font-size: 12px;
            color: #7f8c8d;
            max-width: 200px;
        }}
        .performance-metrics {{
            font-size: 12px;
        }}
        .onnx-status {{
            text-align: center;
        }}
        .expandable {{
            cursor: pointer;
            user-select: none;
        }}
        .expandable:hover {{
            background-color: #e8f4f8;
        }}
        .training-details {{
            display: none;
            background-color: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 12px;
        }}
        .training-details.show {{
            display: block;
        }}
    </style>
    <script>
        function toggleTrainingDetails(modelName) {{
            const details = document.getElementById('details-' + modelName);
            if (details.classList.contains('show')) {{
                details.classList.remove('show');
            }} else {{
                details.classList.add('show');
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>🔬 生物医学图像分析 - 优化综合模型性能报告</h1>
        <p style="text-align: center; color: #7f8c8d; font-size: 1.1em;">
            生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{consolidated_data['total_unique_models']}</h3>
                <p>独特模型数量</p>
            </div>
            <div class="summary-card">
                <h3>{consolidated_data['successful_models']}</h3>
                <p>成功训练模型</p>
            </div>
            <div class="summary-card">
                <h3>{consolidated_data['onnx_converted_models']}</h3>
                <p>ONNX转换完成</p>
            </div>
            <div class="summary-card">
                <h3>{consolidated_data['avg_best_accuracy']:.1f}%</h3>
                <p>平均最佳准确率</p>
            </div>
        </div>
        
        <h2>📊 模型性能汇总表</h2>
        <p>点击模型名称可展开查看详细训练历史</p>
        <table>
            <thead>
                <tr>
                    <th>模型名称</th>
                    <th>最佳验证准确率</th>
                    <th>训练次数</th>
                    <th>最佳训练时间</th>
                    <th>ONNX状态</th>
                    <th>性能对比</th>
                    <th>文件大小</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 添加模型数据行
    for model_name, model_data in consolidated_data['models'].items():
        best_acc = model_data['best_performance']['val_acc']
        training_count = len(model_data['training_history'])
        best_time = model_data['best_performance']['formatted_time']
        
        # 准确率分类
        if best_acc >= 95:
            acc_class = 'accuracy-excellent'
        elif best_acc >= 80:
            acc_class = 'accuracy-good'
        else:
            acc_class = 'accuracy-poor'
        
        # ONNX状态
        onnx_info = model_data.get('onnx_conversion', {})
        if onnx_info.get('status') == 'success':
            onnx_status = '<span class="status-success">✅ 已转换</span>'
            perf_test = onnx_info.get('performance_test', {})
            if 'speedup_ratio' in perf_test:
                perf_info = f"加速比: {perf_test['speedup_ratio']:.2f}x<br>精度差异: {perf_test.get('max_accuracy_diff', 0):.2e}"
            else:
                perf_info = "性能测试未完成"
            file_size = f"PyTorch: {model_data['best_performance']['file_size_mb']:.1f}MB<br>ONNX: {onnx_info.get('file_size_mb', 0):.1f}MB"
        else:
            onnx_status = '<span class="status-failed">❌ 未转换</span>'
            perf_info = "N/A"
            file_size = f"PyTorch: {model_data['best_performance']['file_size_mb']:.1f}MB"
        
        html_content += f"""
                <tr class="expandable" onclick="toggleTrainingDetails('{model_name}')">
                    <td class="model-name">{model_name}</td>
                    <td><span class="{acc_class}">{best_acc:.2f}%</span></td>
                    <td>{training_count}</td>
                    <td>{best_time}</td>
                    <td class="onnx-status">{onnx_status}</td>
                    <td class="performance-metrics">{perf_info}</td>
                    <td class="performance-metrics">{file_size}</td>
                </tr>
                <tr>
                    <td colspan="7">
                        <div id="details-{model_name}" class="training-details">
                            <strong>训练历史详情:</strong><br>
"""
        
        # 添加训练历史
        for i, training in enumerate(model_data['training_history'], 1):
            html_content += f"""
                            {i}. {training['formatted_time']} - 验证准确率: {training['val_acc']:.2f}% (训练准确率: {training['train_acc']:.2f}%, 轮次: {training['epoch']})<br>
"""
        
        html_content += """
                        </div>
                    </td>
                </tr>
"""
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>🏆 性能排行榜</h2>
        <table>
            <thead>
                <tr>
                    <th>排名</th>
                    <th>模型名称</th>
                    <th>最佳验证准确率</th>
                    <th>训练次数</th>
                    <th>ONNX转换状态</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 按最佳性能排序
    sorted_models = sorted(
        consolidated_data['models'].items(),
        key=lambda x: x[1]['best_performance']['val_acc'],
        reverse=True
    )
    
    for rank, (model_name, model_data) in enumerate(sorted_models[:15], 1):
        best_acc = model_data['best_performance']['val_acc']
        training_count = len(model_data['training_history'])
        
        onnx_status = "✅ 已转换" if model_data.get('onnx_conversion', {}).get('status') == 'success' else "❌ 未转换"
        
        html_content += f"""
                <tr>
                    <td>{rank}</td>
                    <td class="model-name">{model_name}</td>
                    <td><span class="accuracy-excellent">{best_acc:.2f}%</span></td>
                    <td>{training_count}</td>
                    <td>{onnx_status}</td>
                </tr>
"""
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>📈 统计摘要</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{consolidated_data['total_training_runs']}</h3>
                <p>总训练次数</p>
            </div>
            <div class="summary-card">
                <h3>{consolidated_data['best_overall_accuracy']:.2f}%</h3>
                <p>最高准确率</p>
            </div>
            <div class="summary-card">
                <h3>{consolidated_data['avg_training_per_model']:.1f}</h3>
                <p>平均每模型训练次数</p>
            </div>
            <div class="summary-card">
                <h3>{consolidated_data['onnx_conversion_rate']:.1f}%</h3>
                <p>ONNX转换率</p>
            </div>
        </div>
        
        <h2>🔧 技术规格</h2>
        <ul style="font-size: 1.1em; line-height: 1.8;">
            <li><strong>数据集:</strong> 真实生物医学图像 (13,024 样本)</li>
            <li><strong>输入尺寸:</strong> 70x70 像素</li>
            <li><strong>训练框架:</strong> PyTorch</li>
            <li><strong>优化器:</strong> AdamW + CosineAnnealingLR</li>
            <li><strong>早停策略:</strong> 耐心值=8轮</li>
            <li><strong>ONNX版本:</strong> Opset 11</li>
        </ul>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>生物医学图像分析模型管理系统 - 优化版本</p>
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    return html_content

def main():
    """主函数 - 生成优化的综合分析报告"""
    print("🔍 生成优化的综合性能分析报告...")
    
    # 创建必要的目录
    os.makedirs('onnx_models', exist_ok=True)
    
    # 解析checkpoint数据
    print("📊 解析训练数据...")
    models_data = parse_checkpoint_info()
    
    # 获取现有ONNX模型
    existing_onnx = get_existing_onnx_models()
    
    # 整合数据
    consolidated_data = {
        'total_unique_models': len(models_data),
        'successful_models': 0,
        'onnx_converted_models': 0,
        'total_training_runs': 0,
        'models': {}
    }
    
    print(f"发现 {len(models_data)} 个独特模型")
    
    # 处理每个模型
    for model_name, training_history in models_data.items():
        print(f"\n处理模型: {model_name}")
        
        # 按验证准确率排序找到最佳性能
        valid_trainings = [t for t in training_history if t['val_acc'] > 0]
        if not valid_trainings:
            continue
        
        consolidated_data['successful_models'] += 1
        consolidated_data['total_training_runs'] += len(training_history)
        
        best_performance = max(valid_trainings, key=lambda x: x['val_acc'])
        
        model_info = {
            'training_history': sorted(training_history, key=lambda x: x['timestamp']),
            'best_performance': best_performance,
            'training_count': len(training_history)
        }
        
        # 检查是否已有ONNX转换
        if model_name in existing_onnx:
            model_info['onnx_conversion'] = {
                'status': 'success',
                'onnx_path': existing_onnx[model_name]['onnx_path'],
                'file_size_mb': existing_onnx[model_name]['file_size_mb']
            }
            consolidated_data['onnx_converted_models'] += 1
            print(f"  ✅ 已有ONNX转换: {existing_onnx[model_name]['onnx_path']}")
        else:
            # 转换最佳模型为ONNX
            print(f"  🔄 转换最佳模型为ONNX...")
            onnx_result = convert_best_model_to_onnx(model_name, best_performance)
            if onnx_result and onnx_result.get('status') == 'success':
                model_info['onnx_conversion'] = onnx_result
                consolidated_data['onnx_converted_models'] += 1
                print(f"  ✅ ONNX转换成功: {onnx_result['onnx_path']}")
            else:
                model_info['onnx_conversion'] = onnx_result or {'status': 'failed'}
                print(f"  ❌ ONNX转换失败")
        
        consolidated_data['models'][model_name] = model_info
    
    # 计算统计数据
    if consolidated_data['successful_models'] > 0:
        all_best_accs = [data['best_performance']['val_acc'] for data in consolidated_data['models'].values()]
        consolidated_data['avg_best_accuracy'] = sum(all_best_accs) / len(all_best_accs)
        consolidated_data['best_overall_accuracy'] = max(all_best_accs)
        consolidated_data['avg_training_per_model'] = consolidated_data['total_training_runs'] / consolidated_data['successful_models']
        consolidated_data['onnx_conversion_rate'] = (consolidated_data['onnx_converted_models'] / consolidated_data['successful_models']) * 100
    else:
        consolidated_data.update({
            'avg_best_accuracy': 0,
            'best_overall_accuracy': 0,
            'avg_training_per_model': 0,
            'onnx_conversion_rate': 0
        })
    
    # 生成HTML报告
    print("📄 生成HTML报告...")
    html_report = generate_optimized_html_report(consolidated_data)
    
    # 保存报告
    html_filename = 'optimized_comprehensive_performance_report.html'
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    # 保存JSON数据
    json_filename = 'optimized_comprehensive_performance_report.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("📊 优化综合分析完成")
    print(f"{'='*60}")
    print(f"独特模型数量: {consolidated_data['total_unique_models']}")
    print(f"成功训练模型: {consolidated_data['successful_models']}")
    print(f"总训练次数: {consolidated_data['total_training_runs']}")
    print(f"ONNX转换完成: {consolidated_data['onnx_converted_models']}")
    print(f"平均最佳准确率: {consolidated_data['avg_best_accuracy']:.2f}%")
    print(f"最高准确率: {consolidated_data['best_overall_accuracy']:.2f}%")
    print(f"ONNX转换率: {consolidated_data['onnx_conversion_rate']:.1f}%")
    print(f"\n📄 报告已生成:")
    print(f"  - HTML报告: {html_filename}")
    print(f"  - JSON数据: {json_filename}")

if __name__ == "__main__":
    main()
