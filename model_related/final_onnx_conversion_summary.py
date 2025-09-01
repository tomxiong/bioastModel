#!/usr/bin/env python3
"""
最终 ONNX 转换总结和性能分析报告生成
"""

import os
import json
from datetime import datetime
import glob

def generate_final_summary():
    """生成最终的 ONNX 转换总结"""
    print("🎯 生成最终 ONNX 转换总结...")
    
    # 检查所有 ONNX 文件
    onnx_files = glob.glob("onnx_models/*.onnx")
    
    # 加载综合报告
    report_path = "optimized_comprehensive_performance_report.json"
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
    else:
        print("❌ 未找到综合报告")
        return False
    
    # 分析 ONNX 文件
    onnx_analysis = {}
    for onnx_file in onnx_files:
        filename = os.path.basename(onnx_file)
        size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
        
        # 提取模型名称
        model_name = filename.split('_')[0]
        if 'simplified' in filename:
            model_name = filename.replace('_simplified', '').split('_')[0]
        elif 'fixed' in filename:
            model_name = filename.replace('_fixed', '').split('_')[0]
        
        if model_name not in onnx_analysis:
            onnx_analysis[model_name] = []
        
        onnx_analysis[model_name].append({
            'filename': filename,
            'size_mb': round(size_mb, 2),
            'path': onnx_file
        })
    
    # 生成最终总结
    summary = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_models_trained': len(report_data.get('models', {})),
        'onnx_conversion_summary': {
            'successful_conversions': 0,
            'failed_conversions': 0,
            'conversion_rate': 0,
            'total_onnx_files': len(onnx_files),
            'total_size_mb': sum(os.path.getsize(f) / (1024 * 1024) for f in onnx_files)
        },
        'model_details': {},
        'onnx_files_analysis': onnx_analysis,
        'key_findings': [],
        'recommendations': []
    }
    
    # 分析每个模型的状态
    for model_name, model_data in report_data.get('models', {}).items():
        onnx_status = model_data.get('onnx_conversion', {}).get('status', 'not_attempted')
        
        model_detail = {
            'training_status': model_data.get('training_status', 'unknown'),
            'onnx_status': onnx_status,
            'accuracy': model_data.get('best_accuracy', 'N/A'),
            'parameters': model_data.get('total_parameters', 'N/A')
        }
        
        if onnx_status == 'success':
            summary['onnx_conversion_summary']['successful_conversions'] += 1
            onnx_info = model_data.get('onnx_conversion', {})
            model_detail['onnx_size_mb'] = onnx_info.get('file_size_mb', 'N/A')
            model_detail['onnx_path'] = onnx_info.get('onnx_path', 'N/A')
        else:
            summary['onnx_conversion_summary']['failed_conversions'] += 1
            model_detail['failure_reason'] = model_data.get('onnx_conversion', {}).get('error', 'Unknown')
        
        summary['model_details'][model_name] = model_detail
    
    # 计算转换率
    total_models = summary['total_models_trained']
    successful = summary['onnx_conversion_summary']['successful_conversions']
    summary['onnx_conversion_summary']['conversion_rate'] = round((successful / total_models) * 100, 1) if total_models > 0 else 0
    
    # 生成关键发现
    summary['key_findings'] = [
        f"成功训练了 {total_models} 个生物医学图像分析模型",
        f"ONNX 转换成功率: {summary['onnx_conversion_summary']['conversion_rate']}% ({successful}/{total_models})",
        f"生成了 {len(onnx_files)} 个 ONNX 模型文件，总大小 {summary['onnx_conversion_summary']['total_size_mb']:.1f} MB",
        "复杂架构模型 (如 AirBubbleHybridNet, MicroViT) 转换困难",
        "标准 CNN 架构 (如 EfficientNet, SimplifiedAirBubbleDetector) 转换效果良好"
    ]
    
    # 生成建议
    summary['recommendations'] = [
        "对于生产部署，建议使用成功转换的 ONNX 模型以获得更好的推理性能",
        "复杂模型建议继续使用 PyTorch 版本，或考虑架构简化",
        "未来模型设计时考虑 ONNX 兼容性，避免过于复杂的动态操作",
        "定期验证 ONNX 模型性能，确保转换质量",
        "建立模型版本管理系统，跟踪 PyTorch 和 ONNX 版本对应关系"
    ]
    
    # 保存总结报告
    summary_path = "final_onnx_conversion_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 生成 HTML 报告
    generate_html_summary(summary)
    
    print("✅ 最终 ONNX 转换总结已生成")
    print(f"📊 转换成功率: {summary['onnx_conversion_summary']['conversion_rate']}%")
    print(f"📁 JSON 报告: {summary_path}")
    print(f"📁 HTML 报告: final_onnx_conversion_summary.html")
    
    return summary

def generate_html_summary(summary):
    """生成 HTML 格式的总结报告"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>生物医学图像分析模型 ONNX 转换最终总结</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .summary-card h3 {{ margin-top: 0; color: #2c3e50; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .model-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .model-table th, .model-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .model-table th {{ background-color: #3498db; color: white; }}
        .status-success {{ color: #27ae60; font-weight: bold; }}
        .status-failed {{ color: #e74c3c; font-weight: bold; }}
        .findings, .recommendations {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .findings ul, .recommendations ul {{ padding-left: 20px; }}
        .findings li, .recommendations li {{ margin: 10px 0; }}
        .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 生物医学图像分析模型 ONNX 转换最终总结</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>📊 训练模型总数</h3>
                <div class="value">{summary['total_models_trained']}</div>
            </div>
            <div class="summary-card">
                <h3>✅ ONNX 转换成功</h3>
                <div class="value">{summary['onnx_conversion_summary']['successful_conversions']}</div>
            </div>
            <div class="summary-card">
                <h3>📈 转换成功率</h3>
                <div class="value">{summary['onnx_conversion_summary']['conversion_rate']}%</div>
            </div>
            <div class="summary-card">
                <h3>📁 ONNX 文件总数</h3>
                <div class="value">{summary['onnx_conversion_summary']['total_onnx_files']}</div>
            </div>
        </div>
        
        <h2>📋 模型详细状态</h2>
        <table class="model-table">
            <thead>
                <tr>
                    <th>模型名称</th>
                    <th>训练状态</th>
                    <th>ONNX 状态</th>
                    <th>准确率</th>
                    <th>参数数量</th>
                    <th>ONNX 大小 (MB)</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for model_name, details in summary['model_details'].items():
        onnx_status_class = "status-success" if details['onnx_status'] == 'success' else "status-failed"
        onnx_size = details.get('onnx_size_mb', 'N/A')
        
        html_content += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{details['training_status']}</td>
                    <td class="{onnx_status_class}">{details['onnx_status'].upper()}</td>
                    <td>{details['accuracy']}</td>
                    <td>{details['parameters']}</td>
                    <td>{onnx_size}</td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>🔍 关键发现</h2>
        <div class="findings">
            <ul>
    """
    
    for finding in summary['key_findings']:
        html_content += f"<li>{finding}</li>"
    
    html_content += f"""
            </ul>
        </div>
        
        <h2>💡 建议和推荐</h2>
        <div class="recommendations">
            <ul>
    """
    
    for recommendation in summary['recommendations']:
        html_content += f"<li>{recommendation}</li>"
    
    html_content += f"""
            </ul>
        </div>
        
        <div class="timestamp">
            <p>报告生成时间: {summary['generation_timestamp']}</p>
        </div>
    </div>
</body>
</html>
    """
    
    with open("final_onnx_conversion_summary.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """主函数"""
    print("🚀 开始生成最终 ONNX 转换总结...")
    
    try:
        summary = generate_final_summary()
        
        if summary:
            print("\n" + "="*60)
            print("🎉 最终 ONNX 转换总结完成!")
            print("="*60)
            print(f"📊 总体成果:")
            print(f"   - 训练模型: {summary['total_models_trained']} 个")
            print(f"   - ONNX 转换成功: {summary['onnx_conversion_summary']['successful_conversions']} 个")
            print(f"   - 转换成功率: {summary['onnx_conversion_summary']['conversion_rate']}%")
            print(f"   - ONNX 文件总数: {summary['onnx_conversion_summary']['total_onnx_files']} 个")
            print(f"   - 总文件大小: {summary['onnx_conversion_summary']['total_size_mb']:.1f} MB")
            print("="*60)
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ 生成总结时出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 最终总结生成成功!")
    else:
        print("\n❌ 最终总结生成失败")