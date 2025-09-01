#!/usr/bin/env python3
"""
验证 airbubble_hybrid_net ONNX 模型性能
对比 PyTorch 原始模型和 ONNX 转换后模型在训练数据集上的性能差异
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
import time
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_data_loader import RealBiomedicalDataLoader
from models.airbubble_hybrid_net import AirBubbleHybridNet

def load_pytorch_model():
    """加载 PyTorch 原始模型"""
    print("🔄 加载 PyTorch 原始模型...")
    
    # 找到最佳检查点
    checkpoint_path = "checkpoints/airbubble_hybrid_net_20250808_013453_best.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
    
    # 首先检查检查点中的模型结构来确定类别数
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 从检查点中推断类别数
    if 'classification_head.4.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['classification_head.4.weight'].shape[0]
    else:
        # 备用方法：检查其他可能的分类层
        for key in checkpoint['model_state_dict'].keys():
            if 'classification_head' in key and 'weight' in key:
                num_classes = checkpoint['model_state_dict'][key].shape[0]
                break
        else:
            num_classes = 2  # 默认为2类
    
    print(f"📊 检测到模型类别数: {num_classes}")
    
    # 加载模型
    model = AirBubbleHybridNet(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ PyTorch 模型加载成功: {checkpoint_path}")
    return model

def load_onnx_model():
    """加载 ONNX 转换后模型"""
    print("🔄 加载 ONNX 转换后模型...")
    
    onnx_path = "onnx_models/airbubble_hybrid_net_simplified_20250808_130232.onnx"
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"未找到 ONNX 文件: {onnx_path}")
    
    # 创建 ONNX Runtime 会话
    session = ort.InferenceSession(onnx_path)
    
    print(f"✅ ONNX 模型加载成功: {onnx_path}")
    print(f"   输入名称: {[input.name for input in session.get_inputs()]}")
    print(f"   输出名称: {[output.name for output in session.get_outputs()]}")
    
    return session

def load_validation_data():
    """加载验证数据集"""
    print("🔄 加载验证数据集...")
    
    from core.real_data_loader import create_real_data_loaders
    
    train_loader, val_loader, test_loader = create_real_data_loaders(batch_size=32, num_workers=2)
    
    print(f"✅ 数据集加载成功:")
    print(f"   验证集大小: {len(val_loader.dataset)}")
    print(f"   测试集大小: {len(test_loader.dataset)}")
    
    return val_loader, test_loader

def evaluate_pytorch_model(model, data_loader, device='cpu'):
    """评估 PyTorch 模型性能"""
    print("🔄 评估 PyTorch 模型性能...")
    
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            # 测量推理时间
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            
            inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            # AirBubbleHybridNet 返回字典，获取分类输出
            if isinstance(output, dict):
                classification_output = output.get('classification', output.get('class_logits', None))
                if classification_output is None:
                    # 如果没有找到分类输出，尝试获取第一个输出
                    classification_output = list(output.values())[0]
            else:
                classification_output = output
            
            pred = classification_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"   处理批次: {batch_idx}/{len(data_loader)}")
    
    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times)
    
    print(f"✅ PyTorch 模型评估完成:")
    print(f"   准确率: {accuracy:.4f}%")
    print(f"   平均推理时间: {avg_inference_time:.4f} ms/batch")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_inference_time': avg_inference_time,
        'predictions': all_predictions,
        'labels': all_labels
    }

def evaluate_onnx_model(session, data_loader):
    """评估 ONNX 模型性能"""
    print("🔄 评估 ONNX 模型性能...")
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    inference_times = []
    
    for batch_idx, (data, target) in enumerate(data_loader):
        # 转换为 numpy 数组
        data_np = data.numpy().astype(np.float32)
        
        # 测量推理时间
        start_time = time.time()
        output = session.run([output_name], {input_name: data_np})[0]
        end_time = time.time()
        
        inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        pred = np.argmax(output, axis=1)
        correct += np.sum(pred == target.numpy())
        total += target.size(0)
        
        all_predictions.extend(pred)
        all_labels.extend(target.numpy())
        
        if batch_idx % 10 == 0:
            print(f"   处理批次: {batch_idx}/{len(data_loader)}")
    
    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times)
    
    print(f"✅ ONNX 模型评估完成:")
    print(f"   准确率: {accuracy:.4f}%")
    print(f"   平均推理时间: {avg_inference_time:.4f} ms/batch")
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_inference_time': avg_inference_time,
        'predictions': all_predictions,
        'labels': all_labels
    }

def compare_predictions(pytorch_results, onnx_results):
    """比较两个模型的预测结果"""
    print("🔄 比较模型预测结果...")
    
    pytorch_preds = np.array(pytorch_results['predictions'])
    onnx_preds = np.array(onnx_results['predictions'])
    
    # 计算预测一致性
    agreement = np.sum(pytorch_preds == onnx_preds) / len(pytorch_preds)
    disagreement_indices = np.where(pytorch_preds != onnx_preds)[0]
    
    print(f"✅ 预测结果比较:")
    print(f"   预测一致性: {agreement:.4f} ({agreement*100:.2f}%)")
    print(f"   不一致样本数: {len(disagreement_indices)}")
    
    # 分析不一致的样本
    if len(disagreement_indices) > 0:
        print(f"   前10个不一致样本:")
        for i, idx in enumerate(disagreement_indices[:10]):
            print(f"     样本 {idx}: PyTorch={pytorch_preds[idx]}, ONNX={onnx_preds[idx]}, 真实={pytorch_results['labels'][idx]}")
    
    return {
        'agreement_rate': agreement,
        'disagreement_count': len(disagreement_indices),
        'disagreement_indices': disagreement_indices.tolist()
    }

def generate_validation_report(pytorch_results, onnx_results, comparison_results):
    """生成验证报告"""
    print("🔄 生成验证报告...")
    
    # 计算性能差异
    accuracy_diff = abs(pytorch_results['accuracy'] - onnx_results['accuracy'])
    speed_ratio = pytorch_results['avg_inference_time'] / onnx_results['avg_inference_time']
    
    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'model_info': {
            'pytorch_checkpoint': 'checkpoints/airbubble_hybrid_net_20250808_013453_best.pth',
            'onnx_model': 'onnx_models/airbubble_hybrid_net_simplified_20250808_130232.onnx',
            'conversion_method': 'simplified_architecture'
        },
        'dataset_info': {
            'total_samples': pytorch_results['total'],
            'dataset_source': 'bioast_dataset validation set'
        },
        'performance_comparison': {
            'pytorch_accuracy': pytorch_results['accuracy'],
            'onnx_accuracy': onnx_results['accuracy'],
            'accuracy_difference': accuracy_diff,
            'accuracy_loss_percentage': (accuracy_diff / pytorch_results['accuracy']) * 100,
            'pytorch_inference_time_ms': pytorch_results['avg_inference_time'],
            'onnx_inference_time_ms': onnx_results['avg_inference_time'],
            'speed_improvement_ratio': speed_ratio,
            'speed_improvement_percentage': (speed_ratio - 1) * 100
        },
        'prediction_consistency': {
            'agreement_rate': comparison_results['agreement_rate'],
            'disagreement_count': comparison_results['disagreement_count'],
            'consistency_percentage': comparison_results['agreement_rate'] * 100
        },
        'validation_status': 'PASSED' if accuracy_diff < 1.0 and comparison_results['agreement_rate'] > 0.95 else 'NEEDS_REVIEW',
        'recommendations': []
    }
    
    # 添加建议
    if accuracy_diff < 0.5:
        report['recommendations'].append("✅ 准确率差异极小，转换质量优秀")
    elif accuracy_diff < 1.0:
        report['recommendations'].append("⚠️ 准确率差异较小，转换质量良好")
    else:
        report['recommendations'].append("❌ 准确率差异较大，需要检查转换过程")
    
    if comparison_results['agreement_rate'] > 0.98:
        report['recommendations'].append("✅ 预测一致性极高，模型行为保持一致")
    elif comparison_results['agreement_rate'] > 0.95:
        report['recommendations'].append("⚠️ 预测一致性良好，少量差异可接受")
    else:
        report['recommendations'].append("❌ 预测一致性较低，需要进一步优化")
    
    if speed_ratio > 1.5:
        report['recommendations'].append("🚀 ONNX 模型推理速度显著提升")
    elif speed_ratio > 1.1:
        report['recommendations'].append("📈 ONNX 模型推理速度有所提升")
    else:
        report['recommendations'].append("📊 ONNX 模型推理速度与原模型相当")
    
    return report

def save_validation_report(report):
    """保存验证报告"""
    # 保存 JSON 报告
    json_path = "airbubble_hybrid_net_onnx_validation_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 验证报告保存到: {json_path}")
    
    # 生成 HTML 报告
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AirBubble Hybrid Net ONNX 验证报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.2em;
            font-weight: 300;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
            font-size: 0.9em;
        }}
        .status-passed {{
            background: #d4edda;
            color: #155724;
        }}
        .status-review {{
            background: #fff3cd;
            color: #856404;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .comparison-table th,
        .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .comparison-table th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .recommendations {{
            background: #e8f5e8;
            border: 1px solid #c3e6c3;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 8px 0;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 2px 6px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 AirBubble Hybrid Net ONNX 验证报告</h1>
            <p>PyTorch vs ONNX 模型性能对比验证</p>
            <div class="status-badge {'status-passed' if report['validation_status'] == 'PASSED' else 'status-review'}">
                {report['validation_status']}
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 核心性能指标</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{report['performance_comparison']['pytorch_accuracy']:.2f}%</div>
                    <div class="metric-label">PyTorch 准确率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['performance_comparison']['onnx_accuracy']:.2f}%</div>
                    <div class="metric-label">ONNX 准确率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['performance_comparison']['accuracy_difference']:.4f}%</div>
                    <div class="metric-label">准确率差异</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['prediction_consistency']['consistency_percentage']:.1f}%</div>
                    <div class="metric-label">预测一致性</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">⚡ 性能对比详情</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>PyTorch 模型</th>
                        <th>ONNX 模型</th>
                        <th>差异/提升</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>准确率</td>
                        <td>{report['performance_comparison']['pytorch_accuracy']:.4f}%</td>
                        <td>{report['performance_comparison']['onnx_accuracy']:.4f}%</td>
                        <td class="highlight">{report['performance_comparison']['accuracy_difference']:.4f}% 差异</td>
                    </tr>
                    <tr>
                        <td>推理时间 (ms/batch)</td>
                        <td>{report['performance_comparison']['pytorch_inference_time_ms']:.4f}</td>
                        <td>{report['performance_comparison']['onnx_inference_time_ms']:.4f}</td>
                        <td class="highlight">{report['performance_comparison']['speed_improvement_percentage']:.1f}% 提升</td>
                    </tr>
                    <tr>
                        <td>预测一致性</td>
                        <td>-</td>
                        <td>{report['prediction_consistency']['agreement_rate']:.4f}</td>
                        <td class="highlight">{report['prediction_consistency']['disagreement_count']} 个不一致样本</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2 class="section-title">📋 验证结论与建议</h2>
            <div class="recommendations">
                <ul>
    """
    
    for recommendation in report['recommendations']:
        html_content += f"<li>{recommendation}</li>"
    
    html_content += f"""
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">ℹ️ 验证信息</h2>
            <p><strong>验证时间:</strong> {report['validation_timestamp']}</p>
            <p><strong>数据集:</strong> {report['dataset_info']['dataset_source']} ({report['dataset_info']['total_samples']} 样本)</p>
            <p><strong>PyTorch 模型:</strong> {report['model_info']['pytorch_checkpoint']}</p>
            <p><strong>ONNX 模型:</strong> {report['model_info']['onnx_model']}</p>
            <p><strong>转换方法:</strong> {report['model_info']['conversion_method']}</p>
        </div>
    </div>
</body>
</html>
    """
    
    html_path = "airbubble_hybrid_net_onnx_validation_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML 验证报告保存到: {html_path}")

def main():
    """主函数"""
    print("🚀 开始 AirBubble Hybrid Net ONNX 模型验证...")
    
    try:
        # 1. 加载模型
        pytorch_model = load_pytorch_model()
        onnx_session = load_onnx_model()
        
        # 2. 加载数据
        val_loader, test_loader = load_validation_data()
        
        # 使用验证集进行测试
        data_loader = val_loader
        
        # 3. 评估 PyTorch 模型
        pytorch_results = evaluate_pytorch_model(pytorch_model, data_loader)
        
        # 4. 评估 ONNX 模型
        onnx_results = evaluate_onnx_model(onnx_session, data_loader)
        
        # 5. 比较预测结果
        comparison_results = compare_predictions(pytorch_results, onnx_results)
        
        # 6. 生成验证报告
        report = generate_validation_report(pytorch_results, onnx_results, comparison_results)
        
        # 7. 保存报告
        save_validation_report(report)
        
        # 8. 输出总结
        print("\n" + "="*60)
        print("🎉 ONNX 模型验证完成!")
        print("="*60)
        print(f"📊 验证状态: {report['validation_status']}")
        print(f"🎯 准确率差异: {report['performance_comparison']['accuracy_difference']:.4f}%")
        print(f"🔄 预测一致性: {report['prediction_consistency']['consistency_percentage']:.1f}%")
        print(f"⚡ 速度提升: {report['performance_comparison']['speed_improvement_percentage']:.1f}%")
        print("="*60)
        
        if report['validation_status'] == 'PASSED':
            print("✅ ONNX 转换质量验证通过，可以安全使用!")
        else:
            print("⚠️ ONNX 转换需要进一步检查和优化")
        
    except Exception as e:
        print(f"❌ 验证过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()