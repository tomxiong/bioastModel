#!/usr/bin/env python3
"""
分析 ConvNeXt 模型的训练结果和性能
包括 ConvNeXt Micro 和 ConvNeXt Tiny 的对比分析
"""

import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.convnext_micro import ConvnextMicro
from models.convnext_tiny import ConvNextTiny
from core.real_data_loader import create_real_data_loaders

def analyze_convnext_checkpoints():
    """分析所有 ConvNeXt 检查点"""
    print("🔍 分析 ConvNeXt 模型检查点...")
    
    checkpoint_dir = "/home/aaa/ws/bioastModel/checkpoints"
    convnext_files = []
    
    # 查找所有 ConvNeXt 相关文件
    for file in os.listdir(checkpoint_dir):
        if 'convnext' in file.lower() and file.endswith('.pth'):
            file_path = os.path.join(checkpoint_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            convnext_files.append({
                'filename': file,
                'path': file_path,
                'size_mb': round(file_size, 2)
            })
    
    print(f"📊 发现 {len(convnext_files)} 个 ConvNeXt 检查点:")
    for i, file_info in enumerate(convnext_files, 1):
        print(f"  {i}. {file_info['filename']} ({file_info['size_mb']} MB)")
    
    return convnext_files

def analyze_checkpoint_details(checkpoint_path, model_type):
    """分析单个检查点的详细信息"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        analysis = {
            'file_path': checkpoint_path,
            'file_size_mb': round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2),
            'model_type': model_type,
            'checkpoint_keys': list(checkpoint.keys()),
            'loadable': True,
            'error': None
        }
        
        # 提取训练信息
        if 'epoch' in checkpoint:
            analysis['epoch'] = checkpoint['epoch']
        if 'train_acc' in checkpoint:
            analysis['train_accuracy'] = checkpoint['train_acc']
        if 'val_acc' in checkpoint:
            analysis['val_accuracy'] = checkpoint['val_acc']
        if 'train_loss' in checkpoint:
            analysis['train_loss'] = checkpoint['train_loss']
        if 'val_loss' in checkpoint:
            analysis['val_loss'] = checkpoint['val_loss']
        
        # 分析模型状态
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            analysis['total_parameters'] = total_params
            analysis['parameter_count_mb'] = round(total_params * 4 / (1024 * 1024), 2)  # 假设 float32
        
        return analysis
        
    except Exception as e:
        return {
            'file_path': checkpoint_path,
            'file_size_mb': round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2),
            'model_type': model_type,
            'loadable': False,
            'error': str(e)
        }

def test_model_performance(checkpoint_path, model_type):
    """测试模型在真实数据上的性能"""
    print(f"🧪 测试 {model_type} 模型性能...")
    
    try:
        # 创建模型
        if 'micro' in model_type.lower():
            model = ConvnextMicro(num_classes=2)
        else:
            model = ConvNextTiny(num_classes=2)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # 测试推理
        test_input = torch.randn(1, 3, 70, 70)
        with torch.no_grad():
            output = model(test_input)
        
        # 加载真实数据进行评估
        try:
            train_loader, val_loader, test_loader = create_real_data_loaders(
                batch_size=32,
                num_workers=2
            )
            
            # 在验证集上评估
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = correct / total
            
            # 在测试集上评估
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            test_accuracy = correct / total
            
            return {
                'inference_working': True,
                'output_shape': list(output.shape),
                'output_range': [float(output.min()), float(output.max())],
                'real_data_val_accuracy': val_accuracy,
                'real_data_test_accuracy': test_accuracy,
                'error': None
            }
            
        except Exception as data_error:
            return {
                'inference_working': True,
                'output_shape': list(output.shape),
                'output_range': [float(output.min()), float(output.max())],
                'real_data_val_accuracy': None,
                'real_data_test_accuracy': None,
                'error': f"Data loading error: {str(data_error)}"
            }
            
    except Exception as e:
        return {
            'inference_working': False,
            'error': str(e)
        }

def analyze_onnx_conversion_status():
    """分析 ConvNeXt ONNX 转换状态"""
    print("🔍 分析 ConvNeXt ONNX 转换状态...")
    
    onnx_dir = "/home/aaa/ws/bioastModel/onnx_models"
    onnx_files = []
    
    for file in os.listdir(onnx_dir):
        if 'convnext' in file.lower() and file.endswith('.onnx'):
            file_path = os.path.join(onnx_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            onnx_files.append({
                'filename': file,
                'path': file_path,
                'size_mb': round(file_size, 2)
            })
    
    print(f"📊 发现 {len(onnx_files)} 个 ConvNeXt ONNX 文件:")
    for i, file_info in enumerate(onnx_files, 1):
        print(f"  {i}. {file_info['filename']} ({file_info['size_mb']} MB)")
    
    return onnx_files

def compare_convnext_variants(analyses):
    """对比不同 ConvNeXt 变体"""
    print("\n" + "="*80)
    print("📊 ConvNeXt 模型变体对比分析")
    print("="*80)
    
    # 分类模型
    micro_models = [a for a in analyses if 'micro' in a['model_type'].lower()]
    tiny_models = [a for a in analyses if 'tiny' in a['model_type'].lower()]
    
    comparison = {
        'micro_models': {
            'count': len(micro_models),
            'avg_size_mb': round(sum(m['file_size_mb'] for m in micro_models) / len(micro_models), 2) if micro_models else 0,
            'avg_params': round(sum(m.get('total_parameters', 0) for m in micro_models) / len(micro_models)) if micro_models else 0,
            'best_accuracy': max((m.get('val_accuracy', 0) for m in micro_models), default=0)
        },
        'tiny_models': {
            'count': len(tiny_models),
            'avg_size_mb': round(sum(m['file_size_mb'] for m in tiny_models) / len(tiny_models), 2) if tiny_models else 0,
            'avg_params': round(sum(m.get('total_parameters', 0) for m in tiny_models) / len(tiny_models)) if tiny_models else 0,
            'best_accuracy': max((m.get('val_accuracy', 0) for m in tiny_models), default=0)
        }
    }
    
    print(f"📊 ConvNeXt Micro 模型:")
    print(f"   数量: {comparison['micro_models']['count']}")
    print(f"   平均大小: {comparison['micro_models']['avg_size_mb']} MB")
    print(f"   平均参数: {comparison['micro_models']['avg_params']:,}")
    print(f"   最佳准确率: {comparison['micro_models']['best_accuracy']:.4f}")
    
    print(f"\n📊 ConvNeXt Tiny 模型:")
    print(f"   数量: {comparison['tiny_models']['count']}")
    print(f"   平均大小: {comparison['tiny_models']['avg_size_mb']} MB")
    print(f"   平均参数: {comparison['tiny_models']['avg_params']:,}")
    print(f"   最佳准确率: {comparison['tiny_models']['best_accuracy']:.4f}")
    
    return comparison

def generate_convnext_analysis_report(analyses, onnx_files, comparison, performance_tests):
    """生成 ConvNeXt 分析报告"""
    report = {
        'report_title': 'ConvNeXt 模型训练和转换分析报告',
        'generation_timestamp': datetime.now().isoformat(),
        'checkpoint_analyses': analyses,
        'onnx_conversion_status': onnx_files,
        'variant_comparison': comparison,
        'performance_tests': performance_tests,
        'summary': {},
        'recommendations': []
    }
    
    # 生成总结
    total_checkpoints = len(analyses)
    successful_checkpoints = len([a for a in analyses if a['loadable']])
    onnx_converted = len(onnx_files)
    
    # 找到最佳模型
    best_model = None
    best_accuracy = 0
    for analysis in analyses:
        if analysis.get('val_accuracy', 0) > best_accuracy:
            best_accuracy = analysis.get('val_accuracy', 0)
            best_model = analysis
    
    report['summary'] = {
        'total_checkpoints': total_checkpoints,
        'successful_checkpoints': successful_checkpoints,
        'success_rate': round(successful_checkpoints / total_checkpoints * 100, 1) if total_checkpoints > 0 else 0,
        'onnx_converted_count': onnx_converted,
        'onnx_conversion_rate': round(onnx_converted / total_checkpoints * 100, 1) if total_checkpoints > 0 else 0,
        'best_model': best_model,
        'best_accuracy': best_accuracy
    }
    
    # 生成建议
    recommendations = []
    
    if best_model:
        recommendations.append({
            'type': '最佳模型推荐',
            'recommendation': f"使用 {best_model['model_type']} 模型",
            'reason': f"验证准确率达到 {best_accuracy:.4f}，性能最佳",
            'file': best_model['file_path']
        })
    
    if onnx_converted > 0:
        recommendations.append({
            'type': 'ONNX 部署',
            'recommendation': '可以使用 ONNX 版本进行生产部署',
            'reason': f'已成功转换 {onnx_converted} 个模型为 ONNX 格式',
            'files': [f['filename'] for f in onnx_files]
        })
    
    if comparison['micro_models']['count'] > 0 and comparison['tiny_models']['count'] > 0:
        if comparison['micro_models']['avg_params'] < comparison['tiny_models']['avg_params']:
            recommendations.append({
                'type': '资源优化',
                'recommendation': '资源受限环境推荐使用 ConvNeXt Micro',
                'reason': f"参数更少 ({comparison['micro_models']['avg_params']:,} vs {comparison['tiny_models']['avg_params']:,})",
                'scenario': '边缘设备部署'
            })
        else:
            recommendations.append({
                'type': '性能优化',
                'recommendation': '高精度需求推荐使用 ConvNeXt Tiny',
                'reason': f"更多参数，更强表达能力",
                'scenario': '服务器端部署'
            })
    
    report['recommendations'] = recommendations
    
    # 保存报告
    report_path = "convnext_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成 HTML 报告
    generate_html_report(report)
    
    print(f"\n✅ ConvNeXt 分析报告已生成:")
    print(f"   📁 JSON: {report_path}")
    print(f"   📁 HTML: convnext_analysis_report.html")
    
    return report

def generate_html_report(report):
    """生成 HTML 格式的分析报告"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report['report_title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .summary-card h3 {{ margin-top: 0; color: #2c3e50; }}
        .model-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .model-table th, .model-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .model-table th {{ background-color: #3498db; color: white; }}
        .status-success {{ color: #27ae60; font-weight: bold; }}
        .status-error {{ color: #e74c3c; font-weight: bold; }}
        .recommendations {{ background: #d5f4e6; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 30px; }}
        .highlight {{ background: #ffffcc; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ {report['report_title']}</h1>
        
        <h2>📊 总体概况</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>📁 检查点统计</h3>
                <p><strong>总数:</strong> {report['summary']['total_checkpoints']}</p>
                <p><strong>成功:</strong> {report['summary']['successful_checkpoints']}</p>
                <p><strong>成功率:</strong> {report['summary']['success_rate']}%</p>
            </div>
            
            <div class="summary-card">
                <h3>🔄 ONNX 转换</h3>
                <p><strong>已转换:</strong> {report['summary']['onnx_converted_count']}</p>
                <p><strong>转换率:</strong> {report['summary']['onnx_conversion_rate']}%</p>
            </div>
            
            <div class="summary-card">
                <h3>🏆 最佳性能</h3>
                <p><strong>最佳准确率:</strong> {report['summary']['best_accuracy']:.4f}</p>
                <p><strong>最佳模型:</strong> {report['summary']['best_model']['model_type'] if report['summary']['best_model'] else 'N/A'}</p>
            </div>
        </div>
        
        <h2>📋 检查点详情</h2>
        <table class="model-table">
            <thead>
                <tr>
                    <th>模型类型</th>
                    <th>文件大小 (MB)</th>
                    <th>参数数量</th>
                    <th>验证准确率</th>
                    <th>状态</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for analysis in report['checkpoint_analyses']:
        status_class = 'status-success' if analysis['loadable'] else 'status-error'
        status_text = '✅ 正常' if analysis['loadable'] else '❌ 异常'
        
        html_content += f"""
                <tr>
                    <td>{analysis['model_type']}</td>
                    <td>{analysis['file_size_mb']}</td>
                    <td>{analysis.get('total_parameters', 'N/A'):,}</td>
                    <td>{analysis.get('val_accuracy', 'N/A')}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>💡 使用建议</h2>
        <div class="recommendations">
    """
    
    for rec in report['recommendations']:
        html_content += f"""
            <div style="margin-bottom: 15px;">
                <h4>{rec['type']}</h4>
                <p><strong>建议:</strong> {rec['recommendation']}</p>
                <p><strong>理由:</strong> {rec['reason']}</p>
            </div>
        """
    
    html_content += f"""
        </div>
        
        <div class="timestamp">
            <p>报告生成时间: {report['generation_timestamp']}</p>
        </div>
    </div>
</body>
</html>
    """
    
    with open("convnext_analysis_report.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """主函数"""
    print("🚀 开始 ConvNeXt 模型分析...")
    
    try:
        # 1. 分析检查点文件
        convnext_files = analyze_convnext_checkpoints()
        
        # 2. 详细分析每个检查点
        print(f"\n📊 详细分析 {len(convnext_files)} 个检查点...")
        analyses = []
        performance_tests = []
        
        for file_info in convnext_files:
            print(f"\n🔍 分析: {file_info['filename']}")
            
            # 确定模型类型
            if 'micro' in file_info['filename'].lower():
                model_type = 'ConvNeXt Micro'
            elif 'tiny' in file_info['filename'].lower():
                model_type = 'ConvNeXt Tiny'
            else:
                model_type = 'ConvNeXt'
            
            # 分析检查点
            analysis = analyze_checkpoint_details(file_info['path'], model_type)
            analyses.append(analysis)
            
            print(f"   ✅ 分析完成")
            if analysis['loadable']:
                print(f"   📊 参数数量: {analysis.get('total_parameters', 'N/A'):,}")
                print(f"   📊 验证准确率: {analysis.get('val_accuracy', 'N/A')}")
                
                # 测试性能（仅对最新的模型）
                if 'best' in file_info['filename']:
                    performance = test_model_performance(file_info['path'], model_type)
                    performance['checkpoint'] = file_info['filename']
                    performance_tests.append(performance)
                    
                    if performance['inference_working']:
                        print(f"   🧪 推理测试: ✅ 成功")
                        if performance.get('real_data_val_accuracy'):
                            print(f"   📊 真实数据验证准确率: {performance['real_data_val_accuracy']:.4f}")
                        if performance.get('real_data_test_accuracy'):
                            print(f"   📊 真实数据测试准确率: {performance['real_data_test_accuracy']:.4f}")
                    else:
                        print(f"   🧪 推理测试: ❌ 失败")
            else:
                print(f"   ❌ 检查点加载失败: {analysis['error']}")
        
        # 3. 分析 ONNX 转换状态
        print(f"\n🔄 分析 ONNX 转换状态...")
        onnx_files = analyze_onnx_conversion_status()
        
        # 4. 对比不同变体
        comparison = compare_convnext_variants(analyses)
        
        # 5. 生成分析报告
        print(f"\n📝 生成分析报告...")
        report = generate_convnext_analysis_report(analyses, onnx_files, comparison, performance_tests)
        
        print(f"\n🎉 ConvNeXt 模型分析完成!")
        return True
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)