#!/usr/bin/env python3
"""
根据最新的test_results.json更新每个模型的可视化HTML报告
"""

import os
import json
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pandas as pd
from io import BytesIO

def load_model_results(experiment_path):
    """加载单个模型的测试结果"""
    result_file = os.path.join(experiment_path, 'test_results.json')
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def create_confusion_matrix_plot(cm, model_name):
    """创建混淆矩阵图表"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def create_metrics_radar_chart(results, model_name):
    """创建性能指标雷达图"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Sensitivity', 'Specificity']
    values = [
        results['accuracy'] * 100,
        results['precision'] * 100,
        results['recall'] * 100,
        results['f1_score'] * 100,
        results['auc'] * 100,
        results['sensitivity'] * 100,
        results['specificity'] * 100
    ]
    
    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]  # 闭合图形
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title(f'{model_name} - Performance Metrics', size=16, fontweight='bold', pad=20)
    ax.grid(True)
    
    # 添加数值标签
    for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
        ax.text(angle, value + 2, f'{value:.1f}%', ha='center', va='center', fontweight='bold')
    
    # 转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def create_performance_bar_chart(results, model_name):
    """创建性能指标柱状图"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Sensitivity', 'Specificity']
    values = [
        results['accuracy'] * 100,
        results['precision'] * 100,
        results['recall'] * 100,
        results['f1_score'] * 100,
        results['auc'] * 100,
        results['sensitivity'] * 100,
        results['specificity'] * 100
    ]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'])
    
    plt.title(f'{model_name} - Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def load_error_analysis_data(experiment_path):
    """加载错误分析数据"""
    analysis_dir = os.path.join(experiment_path, 'sample_analysis')
    error_data = {}
    
    # 检查错误样本CSV文件
    error_csv_path = os.path.join(analysis_dir, 'error_samples.csv')
    if os.path.exists(error_csv_path):
        try:
            error_df = pd.read_csv(error_csv_path)
            error_data['error_samples'] = error_df.to_dict('records')
            error_data['total_errors'] = len(error_df)
            error_data['false_positives'] = len(error_df[error_df['error_type'] == 'False Positive'])
            error_data['false_negatives'] = len(error_df[error_df['error_type'] == 'False Negative'])
        except Exception as e:
            print(f"⚠️ 无法加载错误样本数据: {e}")
    
    # 检查分析图表文件
    chart_files = [
        'confidence_distribution.png',
        'error_type_analysis.png',
        'error_samples_grid.png',
        'high_confidence_correct_grid.png',
        'low_confidence_correct_grid.png'
    ]
    
    error_data['charts'] = {}
    for chart_file in chart_files:
        chart_path = os.path.join(analysis_dir, chart_file)
        if os.path.exists(chart_path):
            try:
                with open(chart_path, 'rb') as f:
                    chart_base64 = base64.b64encode(f.read()).decode()
                    error_data['charts'][chart_file] = chart_base64
            except Exception as e:
                print(f"⚠️ 无法加载图表 {chart_file}: {e}")
    
    return error_data

def generate_error_samples_html(error_data):
    """生成错误样本分析HTML部分"""
    if not error_data or 'error_samples' not in error_data:
        return """
        <div class="chart-section">
            <h2>🔍 Error Sample Analysis</h2>
            <div class="highlight">
                <p>⚠️ Error analysis data not available. Run error analysis script to generate detailed error sample analysis.</p>
            </div>
        </div>
        """
    
    error_samples_table = ""
    if error_data['error_samples']:
        error_samples_table = """
        <div class="error-samples-table">
            <h4>📋 Error Samples Details</h4>
            <div style="max-height: 400px; overflow-y: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead style="background: #f8f9fa; position: sticky; top: 0;">
                        <tr>
                            <th style="border: 1px solid #ddd; padding: 8px;">Image</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">True Label</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Predicted</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Confidence</th>
                            <th style="border: 1px solid #ddd; padding: 8px;">Error Type</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for sample in error_data['error_samples'][:20]:  # 显示前20个错误样本
            true_label = 'Positive' if sample['true_label'] == 1 else 'Negative'
            pred_label = 'Positive' if sample['predicted_label'] == 1 else 'Negative'
            error_samples_table += f"""
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 8px;">{sample['image_name']}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{true_label}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{pred_label}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{sample['confidence']:.3f}</td>
                            <td style="border: 1px solid #ddd; padding: 8px;">{sample['error_type']}</td>
                        </tr>
            """
        
        error_samples_table += """
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    # 生成图表HTML
    charts_html = ""
    chart_titles = {
        'confidence_distribution.png': 'Confidence Distribution',
        'error_type_analysis.png': 'Error Type Analysis',
        'error_samples_grid.png': 'Error Samples Grid',
        'high_confidence_correct_grid.png': 'High Confidence Correct Samples',
        'low_confidence_correct_grid.png': 'Low Confidence Correct Samples'
    }
    
    for chart_file, title in chart_titles.items():
        if chart_file in error_data['charts']:
            charts_html += f"""
            <div class="chart-container">
                <h4>{title}</h4>
                <img src="data:image/png;base64,{error_data['charts'][chart_file]}" alt="{title}">
            </div>
            """
    
    return f"""
    <div class="chart-section">
        <h2>🔍 Error Sample Analysis</h2>
        
        <div class="highlight">
            <h3>📊 Error Summary</h3>
            <p><strong>Total Errors:</strong> {error_data.get('total_errors', 0)} | 
               <strong>False Positives:</strong> {error_data.get('false_positives', 0)} | 
               <strong>False Negatives:</strong> {error_data.get('false_negatives', 0)}</p>
        </div>
        
        {charts_html}
        
        {error_samples_table}
    </div>
    """

def generate_model_html_report(experiment_path, model_name):
    """为单个模型生成HTML报告"""
    results = load_model_results(experiment_path)
    if not results:
        print(f"❌ 无法加载 {model_name} 的测试结果")
        return False
    
    # 创建图表
    cm_plot = create_confusion_matrix_plot(results['confusion_matrix'], model_name)
    radar_plot = create_metrics_radar_chart(results, model_name)
    bar_plot = create_performance_bar_chart(results, model_name)
    
    # 加载错误分析数据
    error_data = load_error_analysis_data(experiment_path)
    error_analysis_html = generate_error_samples_html(error_data)
    
    # 计算额外指标
    cm = np.array(results['confusion_matrix'])
    tn, fp, fn, tp = cm.ravel()
    
    # 生成HTML内容
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} - Performance Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #007bff;
        }}
        .header h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 1.2em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 2.2em;
            font-weight: bold;
            margin: 0;
        }}
        .chart-section {{
            margin-bottom: 40px;
        }}
        .chart-section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #007bff;
            padding-left: 15px;
        }}
        .chart-container {{
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .confusion-matrix-details {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}
        .cm-detail {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .cm-detail h4 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .cm-detail .number {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .classification-report {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        .classification-report h3 {{
            margin-top: 0;
            color: #333;
        }}
        .classification-report pre {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
        .highlight {{
            background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .highlight h3 {{
            margin-top: 0;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{model_name}</h1>
            <div class="subtitle">Biomedical Colony Detection Performance Analysis</div>
        </div>
        
        <div class="highlight">
            <h3>🎯 Key Performance Summary</h3>
            <p><strong>Overall Accuracy:</strong> {results['accuracy']*100:.2f}% | 
               <strong>F1-Score:</strong> {results['f1_score']*100:.2f}% | 
               <strong>AUC:</strong> {results['auc']*100:.2f}%</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="value">{results['accuracy']*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>Precision</h3>
                <div class="value">{results['precision']*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>Recall</h3>
                <div class="value">{results['recall']*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>F1-Score</h3>
                <div class="value">{results['f1_score']*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>AUC</h3>
                <div class="value">{results['auc']*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>Sensitivity</h3>
                <div class="value">{results['sensitivity']*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>Specificity</h3>
                <div class="value">{results['specificity']*100:.2f}%</div>
            </div>
        </div>
        
        <div class="chart-section">
            <h2>📊 Performance Metrics Overview</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{bar_plot}" alt="Performance Metrics Bar Chart">
            </div>
        </div>
        
        <div class="chart-section">
            <h2>🎯 Radar Chart Analysis</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{radar_plot}" alt="Performance Radar Chart">
            </div>
        </div>
        
        <div class="chart-section">
            <h2>🔍 Confusion Matrix Analysis</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{cm_plot}" alt="Confusion Matrix">
            </div>
            
            <div class="confusion-matrix-details">
                <div class="cm-detail">
                    <h4>True Negatives</h4>
                    <div class="number">{tn}</div>
                </div>
                <div class="cm-detail">
                    <h4>False Positives</h4>
                    <div class="number">{fp}</div>
                </div>
                <div class="cm-detail">
                    <h4>False Negatives</h4>
                    <div class="number">{fn}</div>
                </div>
                <div class="cm-detail">
                    <h4>True Positives</h4>
                    <div class="number">{tp}</div>
                </div>
            </div>
        </div>
        
        <div class="classification-report">
            <h3>📋 Detailed Classification Report</h3>
            <pre>{results['classification_report']}</pre>
        </div>
        
        {error_analysis_html}
        
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
    
    # 保存HTML报告
    output_dir = os.path.join('reports', 'updated_individual_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{model_name.lower().replace("-", "_")}_analysis.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ {model_name} HTML报告已更新: {output_file}")
    return True

def update_all_model_reports():
    """更新所有模型的HTML报告"""
    experiments = [
        ('experiments/experiment_20250802_140818/efficientnet_b0', 'EfficientNet-B0'),
        ('experiments/experiment_20250802_164948/resnet18_improved', 'ResNet18-Improved'),
        ('experiments/experiment_20250802_231639/convnext_tiny', 'ConvNext-Tiny'),
        ('experiments/experiment_20250803_020217/vit_tiny', 'ViT-Tiny'),
        ('experiments/experiment_20250803_032628/coatnet', 'CoAtNet'),
        ('experiments/experiment_20250803_101438/mic_mobilenetv3', 'MIC_MobileNetV3'),
        ('experiments/experiment_20250803_102845/micro_vit', 'Micro-ViT'),
        ('experiments/experiment_20250803_115344/airbubble_hybrid_net', 'AirBubble_HybridNet')
    ]
    
    success_count = 0
    total_count = len(experiments)
    
    print("🔄 开始更新所有模型的HTML报告...")
    print(f"总共需要更新 {total_count} 个模型")
    print("="*60)
    
    for experiment_path, model_name in experiments:
        print(f"正在更新: {model_name}")
        if generate_model_html_report(experiment_path, model_name):
            success_count += 1
        print()
    
    print("="*60)
    print(f"✅ HTML报告更新完成!")
    print(f"成功更新: {success_count}/{total_count} 个模型")
    print(f"报告保存位置: reports/updated_individual_analysis/")
    
    return success_count == total_count

def create_index_html():
    """创建索引页面"""
    experiments = [
        ('EfficientNet-B0', 'efficientnet_b0_analysis.html'),
        ('ResNet18-Improved', 'resnet18_improved_analysis.html'),
        ('ConvNext-Tiny', 'convnext_tiny_analysis.html'),
        ('ViT-Tiny', 'vit_tiny_analysis.html'),
        ('CoAtNet', 'coatnet_analysis.html'),
        ('MIC_MobileNetV3', 'mic_mobilenetv3_analysis.html'),
        ('Micro-ViT', 'micro_vit_analysis.html'),
        ('AirBubble_HybridNet', 'airbubble_hybridnet_analysis.html')
    ]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Analysis - Index</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 0 30px rgba(0,0,0,0.2);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .header h1 {{
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
            font-size: 1.2em;
        }}
        .model-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .model-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-decoration: none;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .model-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            text-decoration: none;
            color: white;
        }}
        .model-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.3em;
        }}
        .model-card p {{
            margin: 0;
            opacity: 0.9;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 Biomedical Colony Detection</h1>
            <p>Model Performance Analysis Reports</p>
        </div>
        
        <div class="model-grid">
"""
    
    for model_name, filename in experiments:
        html_content += f"""            <a href="{filename}" class="model-card">
                <h3>{model_name}</h3>
                <p>View detailed performance analysis</p>
            </a>
"""
    
    html_content += f"""        </div>
        
        <div class="timestamp">
            Index updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
    
    output_file = os.path.join('reports', 'updated_individual_analysis', 'index.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 索引页面已创建: {output_file}")

def main():
    """主函数"""
    print("🔄 根据最新test_results.json更新HTML报告...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 设置matplotlib后端
    plt.style.use('default')
    
    # 更新所有模型报告
    success = update_all_model_reports()
    
    if success:
        # 创建索引页面
        create_index_html()
        print("\n🎉 所有HTML报告更新完成!")
        print("📂 可以通过以下方式查看:")
        print("   - 打开 reports/updated_individual_analysis/index.html")
        print("   - 或直接访问各个模型的HTML文件")
    else:
        print("\n⚠️ 部分报告更新失败，请检查错误信息")

if __name__ == "__main__":
    main()