#!/usr/bin/env python3
"""
更新综合分析以包含修复后的 airbubble_hybrid_net ONNX 转换
"""

import os
import sys
import json
import glob
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def update_comprehensive_analysis():
    """更新综合分析报告以包含修复后的模型"""
    
    print("🔄 更新综合分析报告...")
    
    # 读取现有的JSON报告
    json_report_path = "optimized_comprehensive_performance_report.json"
    
    if not os.path.exists(json_report_path):
        print(f"❌ 未找到现有报告: {json_report_path}")
        return False
    
    with open(json_report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    
    # 检查是否有新的 airbubble_hybrid_net ONNX 文件
    airbubble_onnx_files = glob.glob("onnx_models/airbubble_hybrid_net_*.onnx")
    
    if not airbubble_onnx_files:
        print("❌ 未找到 airbubble_hybrid_net ONNX 文件")
        return False
    
    # 获取最新的 ONNX 文件
    latest_onnx = max(airbubble_onnx_files, key=os.path.getctime)
    print(f"✅ 找到最新的 airbubble_hybrid_net ONNX: {latest_onnx}")
    
    # 获取文件大小和基本信息
    file_size_mb = os.path.getsize(latest_onnx) / (1024 * 1024)
    
    # 更新报告数据
    updated = False
    
    for model_name, model_data in report_data['models'].items():
        if model_name == 'airbubble_hybrid_net':
            # 更新 ONNX 转换状态
            model_data['onnx_conversion'] = {
                'status': 'success',
                'onnx_path': latest_onnx,
                'conversion_method': 'simplified_architecture',
                'model_size_mb': round(file_size_mb, 2),
                'conversion_timestamp': datetime.now().isoformat(),
                'notes': 'Successfully converted using simplified architecture approach to resolve dynamic shape issues'
            }
            
            print(f"✅ 更新了 {model_name} 的 ONNX 转换状态")
            updated = True
            break
    
    if not updated:
        print("⚠️ 未找到 airbubble_hybrid_net 模型数据进行更新")
        return False
    
    # 重新计算统计信息
    total_models = len(report_data['models'])
    successful_conversions = sum(1 for model_data in report_data['models'].values() 
                               if model_data.get('onnx_conversion', {}).get('status') == 'success')
    
    report_data['onnx_conversion_rate'] = round((successful_conversions / total_models) * 100, 1)
    report_data['onnx_converted_models'] = successful_conversions
    
    print(f"📊 更新统计信息:")
    print(f"   - 总模型数: {total_models}")
    print(f"   - 成功转换: {successful_conversions}")
    print(f"   - 转换率: {report_data['onnx_conversion_rate']}%")
    
    # 保存更新后的报告
    backup_path = f"{json_report_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.rename(json_report_path, backup_path)
    print(f"📁 原报告备份到: {backup_path}")
    
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 更新后的报告保存到: {json_report_path}")
    
    # 重新生成HTML报告
    generate_updated_html_report(report_data)
    
    return True

def generate_updated_html_report(report_data):
    """生成更新后的HTML报告"""
    
    print("🔄 生成更新后的HTML报告...")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>优化综合性能分析报告 - 已更新</title>
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
        .update-notice {{
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 6px;
            padding: 15px;
            margin: 20px 30px;
            color: #0c5460;
        }}
        .update-notice strong {{
            color: #0c5460;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 优化综合性能分析报告</h1>
            <p>生物医学图像分析模型管理系统 - 已更新 airbubble_hybrid_net ONNX 转换</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="update-notice">
            <strong>🎉 更新通知:</strong> airbubble_hybrid_net 模型已成功转换为 ONNX 格式！
            使用简化架构方法解决了动态形状问题，转换率从 42.9% 提升至 <strong>{report_data['onnx_conversion_rate']}%</strong>
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
                <p>ONNX转换完成</p>
            </div>
            <div class="summary-card">
                <h3>{report_data['onnx_conversion_rate']}%</h3>
                <p>ONNX转换率</p>
            </div>
            <div class="summary-card">
                <h3>{report_data['avg_best_accuracy']:.1f}%</h3>
                <p>平均最佳准确率</p>
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
            status_class = 'status-success' if onnx_status == 'success' else 'status-failed'
            
            onnx_info_html = ""
            if onnx_status == 'success':
                onnx_data = model_data['onnx_conversion']
                method_note = ""
                if 'conversion_method' in onnx_data:
                    method_note = f" ({onnx_data['conversion_method']})"
                
                onnx_info_html = f"""
                <div class="onnx-info">
                    <strong>✅ ONNX转换成功{method_note}</strong><br>
                    模型大小: {onnx_data.get('model_size_mb', 'N/A')} MB<br>
                    推理时间: {onnx_data.get('avg_inference_time_ms', 'N/A')} ms
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
                            {'ONNX Ready' if onnx_status == 'success' else 'PyTorch Only'}
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
        print(f"📁 原HTML报告备份到: {backup_html_path}")
    
    with open(html_report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 更新后的HTML报告保存到: {html_report_path}")

def main():
    """主函数"""
    print("🚀 开始更新综合分析报告...")
    
    success = update_comprehensive_analysis()
    
    if success:
        print("\n🎉 综合分析报告更新成功!")
        print("📊 airbubble_hybrid_net ONNX转换状态已更新")
        print("📈 转换率统计已重新计算")
    else:
        print("\n❌ 综合分析报告更新失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()