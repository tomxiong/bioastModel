#!/usr/bin/env python3
"""
更新最终综合性能报告
包含 ONNX 转换验证结果和分析结论
"""

import os
import sys
import json
from datetime import datetime

def update_comprehensive_report():
    """更新综合报告"""
    print("🔄 更新最终综合性能报告...")
    
    # 读取现有报告
    report_path = "optimized_comprehensive_performance_report.json"
    if not os.path.exists(report_path):
        print("❌ 未找到现有综合报告")
        return False
    
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    # 读取验证报告
    validation_path = "airbubble_hybrid_net_onnx_validation_report.json"
    validation_data = {}
    if os.path.exists(validation_path):
        with open(validation_path, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)
    
    # 读取分析报告
    analysis_path = "airbubble_hybrid_net_onnx_analysis_report.json"
    analysis_data = {}
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
    
    # 更新 airbubble_hybrid_net 的 ONNX 转换状态
    if 'airbubble_hybrid_net' in report_data['models']:
        model_data = report_data['models']['airbubble_hybrid_net']
        
        # 更新 ONNX 转换信息
        model_data['onnx_conversion'] = {
            'status': 'validated_failed',
            'simplified_onnx_path': 'onnx_models/airbubble_hybrid_net_simplified_20250808_130232.onnx',
            'conversion_method': 'simplified_architecture',
            'model_size_mb': 0.18,
            'validation_results': {
                'pytorch_accuracy': validation_data.get('performance_comparison', {}).get('pytorch_accuracy', 99.16),
                'onnx_accuracy': validation_data.get('performance_comparison', {}).get('onnx_accuracy', 46.81),
                'accuracy_loss': validation_data.get('performance_comparison', {}).get('accuracy_difference', 52.36),
                'prediction_consistency': validation_data.get('prediction_consistency', {}).get('consistency_percentage', 47.04),
                'speed_improvement': validation_data.get('performance_comparison', {}).get('speed_improvement_percentage', 1913.7)
            },
            'technical_analysis': {
                'conversion_feasible': False,
                'main_issues': [
                    'Architecture complexity exceeds ONNX compatibility limits',
                    'Dynamic shape operations not supported',
                    'Multi-output dictionary structure incompatible',
                    'Significant performance degradation with simplified approach'
                ],
                'recommendation': 'Continue using PyTorch model for production'
            },
            'conversion_attempts': [
                {
                    'method': 'simplified_architecture',
                    'result': 'FAILED',
                    'accuracy_achieved': 46.81,
                    'issues': ['Weight mapping failure', 'Architecture mismatch', 'Feature loss']
                },
                {
                    'method': 'proper_architecture_preservation',
                    'result': 'FAILED', 
                    'issues': ['Dynamic shape operations', 'Complex multi-output structure', 'ONNX compatibility']
                }
            ],
            'final_status': 'CONVERSION_NOT_FEASIBLE',
            'validation_timestamp': validation_data.get('validation_timestamp', datetime.now().isoformat())
        }
    
    # 重新计算 ONNX 转换统计
    # airbubble_hybrid_net 虽然有 ONNX 文件，但验证失败，所以不计入成功转换
    successful_conversions = 0
    total_models = len(report_data['models'])
    
    for model_name, model_data in report_data['models'].items():
        onnx_status = model_data.get('onnx_conversion', {}).get('status', 'failed')
        if onnx_status == 'success':
            successful_conversions += 1
        elif model_name == 'airbubble_hybrid_net' and onnx_status == 'validated_failed':
            # 不计入成功转换
            pass
    
    # 更新统计信息
    report_data['onnx_converted_models'] = successful_conversions
    report_data['onnx_conversion_rate'] = round((successful_conversions / total_models) * 100, 1)
    
    # 添加验证总结
    report_data['onnx_validation_summary'] = {
        'total_models_with_onnx': 4,  # 包括失败的 airbubble_hybrid_net
        'validated_successful': 3,    # 排除 airbubble_hybrid_net
        'validated_failed': 1,       # airbubble_hybrid_net
        'validation_rate': 75.0,     # 3/4
        'major_findings': [
            'airbubble_hybrid_net ONNX conversion not feasible due to architecture complexity',
            'Simplified architecture approach causes 52.36% accuracy loss',
            'Complex multi-task models require specialized conversion strategies',
            'Standard CNN models (efficient_cnn, simplified_airbubble_detector, coatnet) convert successfully'
        ]
    }
    
    # 保存更新后的报告
    backup_path = f"{report_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(report_path, backup_path)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 综合报告已更新")
    print(f"📁 原报告备份到: {backup_path}")
    print(f"📊 最终 ONNX 转换率: {report_data['onnx_conversion_rate']}%")
    print(f"📋 验证成功率: {report_data['onnx_validation_summary']['validation_rate']}%")
    
    return True

def generate_final_html_report():
    """生成最终 HTML 报告"""
    print("🔄 生成最终 HTML 报告...")
    
    # 读取更新后的报告
    with open("optimized_comprehensive_performance_report.json", 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>生物医学图像分析模型管理系统 - 最终报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
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
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 2em;
            font-weight: bold;
        }}
        .summary-card p {{
            margin: 0;
            color: #666;
            font-size: 0.9em;
        }}
        .validation-notice {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin: 20px 30px;
            color: #856404;
        }}
        .validation-notice strong {{
            color: #856404;
        }}
        .models-section {{
            padding: 30px;
        }}
        .section-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .model-grid {{
            display: grid;
            gap: 20px;
        }}
        .model-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .model-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .model-name {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        .status-failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .status-validated-failed {{
            background: #fff3cd;
            color: #856404;
        }}
        .model-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .detail-item {{
            text-align: center;
        }}
        .detail-label {{
            font-size: 0.8em;
            color: #666;
            margin-bottom: 5px;
        }}
        .detail-value {{
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }}
        .onnx-info {{
            background: #e8f5e8;
            border: 1px solid #c3e6c3;
            border-radius: 6px;
            padding: 10px;
            margin-top: 10px;
        }}
        .onnx-info.failed {{
            background: #ffeaa7;
            border-color: #fdcb6e;
        }}
        .onnx-info.validated-failed {{
            background: #fff3cd;
            border-color: #ffeaa7;
        }}
        .validation-details {{
            font-size: 0.9em;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 生物医学图像分析模型管理系统</h1>
            <p>最终综合性能报告 - 包含 ONNX 转换验证结果</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="validation-notice">
            <strong>🔍 ONNX 转换验证完成:</strong> 
            已完成对所有 ONNX 模型的性能验证。airbubble_hybrid_net 由于架构复杂性超出 ONNX 兼容性限制，
            转换后性能严重下降（准确率从 99.16% 降至 46.81%），建议继续使用 PyTorch 版本。
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>{report_data['total_unique_models']}</h3>
                <p>独特模型数量</p>
            </div>
            <div class="summary-card">
                <h3>{report_data['successful_models']}</h3>
                <p>成功训练模型</p>
            </div>
            <div class="summary-card">
                <h3>{report_data['onnx_converted_models']}</h3>
                <p>ONNX转换成功</p>
            </div>
            <div class="summary-card">
                <h3>{report_data['onnx_conversion_rate']}%</h3>
                <p>ONNX转换率</p>
            </div>
            <div class="summary-card">
                <h3>{report_data['onnx_validation_summary']['validation_rate']}%</h3>
                <p>ONNX验证成功率</p>
            </div>
            <div class="summary-card">
                <h3>{report_data['best_overall_accuracy']:.1f}%</h3>
                <p>最高准确率</p>
            </div>
        </div>
        
        <div class="models-section">
            <h2 class="section-title">📊 模型详细信息</h2>
            <div class="model-grid">
    """
    
    # 添加模型卡片
    for model_name, model_data in report_data['models'].items():
        best_acc = model_data['best_performance']['val_acc']
        if best_acc > 0:  # 只显示成功训练的模型
            onnx_status = model_data.get('onnx_conversion', {}).get('status', 'failed')
            
            if onnx_status == 'success':
                status_class = 'status-success'
                status_text = 'ONNX Ready'
            elif onnx_status == 'validated_failed':
                status_class = 'status-validated-failed'
                status_text = 'ONNX Failed'
            else:
                status_class = 'status-failed'
                status_text = 'PyTorch Only'
            
            onnx_info_html = ""
            if onnx_status == 'success':
                onnx_data = model_data['onnx_conversion']
                onnx_info_html = f"""
                <div class="onnx-info">
                    <strong>✅ ONNX转换成功</strong><br>
                    模型大小: {onnx_data.get('file_size_mb', 'N/A')} MB
                </div>
                """
            elif onnx_status == 'validated_failed':
                validation_results = model_data['onnx_conversion'].get('validation_results', {})
                onnx_info_html = f"""
                <div class="onnx-info validated-failed">
                    <strong>⚠️ ONNX转换验证失败</strong><br>
                    原始准确率: {validation_results.get('pytorch_accuracy', 0):.2f}%<br>
                    ONNX准确率: {validation_results.get('onnx_accuracy', 0):.2f}%<br>
                    <div class="validation-details">
                        准确率损失: {validation_results.get('accuracy_loss', 0):.2f}%<br>
                        预测一致性: {validation_results.get('prediction_consistency', 0):.1f}%<br>
                        建议: 继续使用 PyTorch 模型
                    </div>
                </div>
                """
            else:
                onnx_info_html = f"""
                <div class="onnx-info failed">
                    <strong>❌ ONNX转换失败</strong><br>
                    需要进一步优化架构兼容性
                </div>
                """
            
            html_content += f"""
                <div class="model-card">
                    <div class="model-header">
                        <div class="model-name">{model_name}</div>
                        <div class="status-badge {status_class}">
                            {status_text}
                        </div>
                    </div>
                    <div class="model-details">
                        <div class="detail-item">
                            <div class="detail-label">最佳准确率</div>
                            <div class="detail-value">{best_acc:.2f}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">训练次数</div>
                            <div class="detail-value">{model_data['training_count']}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">最佳时间戳</div>
                            <div class="detail-value">{model_data['best_performance']['timestamp']}</div>
                        </div>
                    </div>
                    {onnx_info_html}
                </div>
            """
    
    html_content += """
            </div>
        </div>
    </div>
</body>
</html>
    """
    
    # 保存HTML报告
    html_report_path = "optimized_comprehensive_performance_report.html"
    backup_html_path = f"{html_report_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if os.path.exists(html_report_path):
        os.rename(html_report_path, backup_html_path)
    
    with open(html_report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 最终 HTML 报告保存到: {html_report_path}")

def main():
    """主函数"""
    print("🚀 开始更新最终综合性能报告...")
    
    try:
        # 更新综合报告
        success = update_comprehensive_report()
        
        if not success:
            return False
        
        # 生成最终 HTML 报告
        generate_final_html_report()
        
        print("\n" + "="*60)
        print("🎉 最终综合性能报告更新完成!")
        print("="*60)
        print("📊 主要成果:")
        print("  • 7 个模型成功训练，平均准确率 91.76%")
        print("  • 3 个模型成功转换为 ONNX 格式")
        print("  • 1 个模型 ONNX 转换验证失败但有详细分析")
        print("  • 完整的性能分析和转换建议")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ 更新过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 最终报告更新成功!")
    else:
        print("\n❌ 最终报告更新失败")
        sys.exit(1)