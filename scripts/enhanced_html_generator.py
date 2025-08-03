#!/usr/bin/env python3
"""
增强版HTML报告生成器 - 包含完整的错误样本分析和图片显示
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pandas as pd
from PIL import Image
import shutil

def load_model_results(experiment_path):
    """加载单个模型的测试结果"""
    result_file = os.path.join(experiment_path, 'test_results.json')
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def create_charts(results, model_name, output_dir):
    """创建所有图表并保存为独立文件"""
    charts = {}
    
    # 1. 混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    chart_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    charts['confusion_matrix'] = 'confusion_matrix.png'
    
    # 2. 性能指标柱状图
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
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'performance_bar_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    charts['performance_bar'] = 'performance_bar_chart.png'
    
    # 3. 雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values_radar = values + values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values_radar, 'o-', linewidth=2, label=model_name, color='#1f77b4')
    ax.fill(angles, values_radar, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_title(f'{model_name} - Performance Metrics', size=16, fontweight='bold', pad=20)
    ax.grid(True)
    
    for angle, value, metric in zip(angles[:-1], values, metrics):
        ax.text(angle, value + 2, f'{value:.1f}%', ha='center', va='center', fontweight='bold')
    
    chart_path = os.path.join(output_dir, 'radar_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    charts['radar'] = 'radar_chart.png'
    
    return charts

def load_error_analysis_data(experiment_path):
    """加载错误分析数据"""
    analysis_dir = os.path.join(experiment_path, 'sample_analysis')
    error_data = {'has_data': False}
    
    print(f"检查错误分析目录: {analysis_dir}")
    
    if not os.path.exists(analysis_dir):
        print(f"⚠️ 错误分析目录不存在: {analysis_dir}")
        return error_data
    
    # 检查CSV文件
    error_csv_path = os.path.join(analysis_dir, 'error_samples.csv')
    detailed_csv_path = os.path.join(analysis_dir, 'detailed_predictions.csv')
    
    csv_path = error_csv_path if os.path.exists(error_csv_path) else detailed_csv_path
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"成功加载CSV文件: {csv_path}, 行数: {len(df)}")
            
            if 'error_type' in df.columns:
                error_data['error_samples'] = df.to_dict('records')
                error_data['total_errors'] = len(df)
                error_data['false_positives'] = len(df[df['error_type'] == 'False Positive'])
                error_data['false_negatives'] = len(df[df['error_type'] == 'False Negative'])
            else:
                error_df = df[df['is_correct'] == False].copy()
                if len(error_df) > 0:
                    error_df['error_type'] = error_df.apply(
                        lambda row: 'False Positive' if row['true_label'] == 0 and row['predicted_label'] == 1 
                        else 'False Negative', axis=1
                    )
                    error_data['error_samples'] = error_df.to_dict('records')
                    error_data['total_errors'] = len(error_df)
                    error_data['false_positives'] = len(error_df[error_df['error_type'] == 'False Positive'])
                    error_data['false_negatives'] = len(error_df[error_df['error_type'] == 'False Negative'])
                else:
                    error_data['error_samples'] = []
                    error_data['total_errors'] = 0
                    error_data['false_positives'] = 0
                    error_data['false_negatives'] = 0
            
            error_data['has_data'] = True
                    
        except Exception as e:
            print(f"⚠️ 无法加载错误样本数据: {e}")
    
    # 检查图表文件
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
            error_data['charts'][chart_file] = chart_file
            print(f"找到图表文件: {chart_file}")
    
    return error_data

def copy_analysis_charts(experiment_path, output_dir):
    """复制样本分析图表到输出目录"""
    analysis_dir = os.path.join(experiment_path, 'sample_analysis')
    if not os.path.exists(analysis_dir):
        return
    
    chart_files = [
        'confidence_distribution.png',
        'error_type_analysis.png',
        'error_samples_grid.png', 
        'high_confidence_correct_grid.png',
        'low_confidence_correct_grid.png'
    ]
    
    for chart_file in chart_files:
        src_path = os.path.join(analysis_dir, chart_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, chart_file)
            try:
                shutil.copy2(src_path, dst_path)
                print(f"复制图表: {chart_file}")
            except Exception as e:
                print(f"复制图表失败 {chart_file}: {e}")

def create_error_thumbnails(error_data, output_dir, data_dir="bioast_dataset"):
    """直接链接到原始图片，不创建任何副本"""
    if not error_data.get('has_data') or not error_data.get('error_samples'):
        return []
    
    thumbnail_info = []
    
    for i, sample in enumerate(error_data['error_samples'][:30]):  # 限制前30个
        try:
            image_name = sample['image_name']
            
            # 尝试不同路径
            possible_paths = [
                sample.get('image_path', ''),
                os.path.join(data_dir, 'test', 'negative', image_name),
                os.path.join(data_dir, 'test', 'positive', image_name),
                os.path.join(data_dir, 'negative', image_name),
                os.path.join(data_dir, 'positive', image_name),
            ]
            
            image_path = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    image_path = path
                    break
            
            if image_path:
                # 直接使用原始图片的绝对路径，不创建任何副本
                original_path = os.path.abspath(image_path).replace('\\', '/')
                
                thumbnail_info.append({
                    'sample': sample,
                    'thumbnail': f"file:///{original_path}",  # 缩略图也直接使用原始图片
                    'large': f"file:///{original_path}",      # 大图也直接使用原始图片
                    'index': i
                })
                    
        except Exception as e:
            print(f"⚠️ 处理错误样本图像失败 {sample.get('image_name', 'unknown')}: {e}")
    
    print(f"✅ 找到了 {len(thumbnail_info)} 个错误样本，直接链接到原始图片")
    return thumbnail_info

def generate_html_report(experiment_path, model_name):
    """生成单个模型的增强HTML报告"""
    results = load_model_results(experiment_path)
    if not results:
        print(f"❌ 无法加载 {model_name} 的测试结果")
        return False
    
    # 创建输出目录
    output_dir = os.path.join('reports', 'enhanced_individual_analysis', model_name.lower().replace("-", "_"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    charts = create_charts(results, model_name, output_dir)
    
    # 复制样本分析图表
    copy_analysis_charts(experiment_path, output_dir)
    
    # 加载错误分析数据
    error_data = load_error_analysis_data(experiment_path)
    
    # 创建错误样本缩略图
    thumbnail_info = create_error_thumbnails(error_data, output_dir)
    
    # 计算混淆矩阵指标
    cm = np.array(results['confusion_matrix'])
    tn, fp, fn, tp = cm.ravel()
    
    # 生成错误分析HTML部分
    error_analysis_html = generate_error_analysis_html(error_data, thumbnail_info)
    
    # 生成完整HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{model_name} - Enhanced Performance Analysis</title>
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
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .gallery-item {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .gallery-item:hover {{
            transform: translateY(-5px);
        }}
        .image-container {{
            text-align: center;
            margin-bottom: 10px;
        }}
        .image-container img {{
            border-radius: 5px;
            border: 2px solid #ddd;
            transition: border-color 0.3s ease;
            cursor: pointer;
        }}
        .image-container img:hover {{
            border-color: #007bff;
        }}
        .image-info {{
            text-align: center;
            font-size: 0.9em;
        }}
        .image-name {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #333;
        }}
        .prediction-info {{
            color: #666;
        }}
        .true-label {{ color: #28a745; }}
        .pred-label {{ color: #dc3545; }}
        .confidence {{ color: #007bff; }}
        .error-type {{ color: #fd7e14; font-weight: bold; }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            position: relative;
            margin: auto;
            padding: 20px;
            width: 90%;
            max-width: 800px;
            top: 50%;
            transform: translateY(-50%);
            text-align: center;
        }}
        .modal-content img {{
            max-width: 100%;
            max-height: 80vh;
            border-radius: 10px;
        }}
        .close {{
            position: absolute;
            top: 10px;
            right: 25px;
            color: white;
            font-size: 35px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{ color: #ccc; }}
        #modalCaption {{
            color: white;
            margin-top: 15px;
            font-size: 1.2em;
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
    <script>
        function openLargeImage(src, caption) {{
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = src;
            document.getElementById('modalCaption').innerHTML = caption;
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
        }}
        
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{model_name}</h1>
            <div class="subtitle">Enhanced Biomedical Colony Detection Performance Analysis</div>
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
                <img src="{charts['performance_bar']}" alt="Performance Metrics Bar Chart">
            </div>
        </div>
        
        <div class="chart-section">
            <h2>🎯 Radar Chart Analysis</h2>
            <div class="chart-container">
                <img src="{charts['radar']}" alt="Performance Radar Chart">
            </div>
        </div>
        
        <div class="chart-section">
            <h2>🔍 Confusion Matrix Analysis</h2>
            <div class="chart-container">
                <img src="{charts['confusion_matrix']}" alt="Confusion Matrix">
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
    
    <!-- 大图显示模态框 -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <img id="modalImage" src="" alt="">
            <div id="modalCaption"></div>
        </div>
    </div>
</body>
</html>"""
    
    # 保存HTML报告
    output_file = os.path.join(output_dir, 'analysis.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ {model_name} 增强HTML报告已生成: {output_file}")
    return True

def generate_error_analysis_html(error_data, thumbnail_info):
    """生成错误分析HTML部分"""
    if not error_data.get('has_data'):
        return """
        <div class="chart-section">
            <h2>🔍 Error Sample Analysis</h2>
            <div class="highlight">
                <p>⚠️ Error analysis data not available. Run error analysis script to generate detailed error sample analysis.</p>
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
        if chart_file in error_data.get('charts', {}):
            charts_html += f"""
            <div class="chart-container">
                <h4>{title}</h4>
                <img src="{chart_file}" alt="{title}">
            </div>
            """
    
    # 生成错误样本图片画廊
    gallery_html = ""
    if thumbnail_info:
        gallery_html = """
        <div class="error-samples-gallery">
            <h4>📋 Error Samples Gallery</h4>
            <div class="gallery-grid">
        """
        
        for thumb_info in thumbnail_info:
            sample = thumb_info['sample']
            true_label = 'Positive' if sample['true_label'] == 1 else 'Negative'
            pred_label = 'Positive' if sample['predicted_label'] == 1 else 'Negative'
            
            gallery_html += f"""
                <div class="gallery-item">
                    <div class="image-container">
                        <img src="{thumb_info['thumbnail']}" alt="{sample['image_name']}" 
                             onclick="openLargeImage('{thumb_info['large']}', '{sample['image_name']}')"
                             style="cursor: pointer;">
                    </div>
                    <div class="image-info">
                        <div class="image-name">{sample['image_name']}</div>
                        <div class="prediction-info">
                            <span class="true-label">True: {true_label}</span><br>
                            <span class="pred-label">Pred: {pred_label}</span><br>
                            <span class="confidence">Conf: {sample['confidence']:.3f}</span><br>
                            <span class="error-type">{sample['error_type']}</span>
                        </div>
                    </div>
                </div>
            """
        
        gallery_html += """
            </div>
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
        
        {gallery_html}
    </div>
    """

def main():
    """主函数"""
    print("🚀 生成增强版HTML报告（包含错误样本可视化）...")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
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
    
    print("🔄 开始生成所有模型的增强HTML报告...")
    print(f"总共需要处理 {total_count} 个模型")
    print("="*60)
    
    for experiment_path, model_name in experiments:
        print(f"正在处理: {model_name}")
        if generate_html_report(experiment_path, model_name):
            success_count += 1
        print()
    
    print("="*60)
    print(f"✅ 增强HTML报告生成完成!")
    print(f"成功生成: {success_count}/{total_count} 个模型")
    print(f"报告保存位置: reports/enhanced_individual_analysis/")
    
    # 创建索引页面
    create_index_page()
    
    return success_count == total_count

def create_index_page():
    """创建增强版索引页面"""
    experiments = [
        ('EfficientNet-B0', 'efficientnet_b0'),
        ('ResNet18-Improved', 'resnet18_improved'),
        ('ConvNext-Tiny', 'convnext_tiny'),
        ('ViT-Tiny', 'vit_tiny'),
        ('CoAtNet', 'coatnet'),
        ('MIC_MobileNetV3', 'mic_mobilenetv3'),
        ('Micro-ViT', 'micro_vit'),
        ('AirBubble_HybridNet', 'airbubble_hybridnet')
    ]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Model Performance Analysis - Index</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
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
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }}
        .model-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-decoration: none;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        .model-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.25);
            text-decoration: none;
            color: white;
        }}
        .model-card h3 {{
            margin: 0 0 15px 0;
            font-size: 1.4em;
            font-weight: bold;
        }}
        .model-card p {{
            margin: 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .features {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        .features h2 {{
            color: #333;
            margin-bottom: 20px;
        }}
        .features ul {{
            list-style: none;
            padding: 0;
        }}
        .features li {{
            padding: 8px 0;
            color: #666;
        }}
        .features li:before {{
            content: "✅ ";
            color: #28a745;
            font-weight: bold;
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
            <h1>🧬 Enhanced Biomedical Colony Detection</h1>
            <p>Comprehensive Model Performance Analysis with Error Sample Visualization</p>
        </div>
        
        <div class="features">
            <h2>🚀 Enhanced Features</h2>
            <ul>
                <li>Interactive error sample galleries with thumbnail previews</li>
                <li>Click-to-zoom functionality for detailed sample inspection</li>
                <li>Comprehensive error analysis with visual charts</li>
                <li>Independent chart files for better performance</li>
                <li>Enhanced UI with modern design and animations</li>
            </ul>
        </div>
        
        <div class="model-grid">
"""
    
    for model_name, folder_name in experiments:
        html_content += f"""            <a href="{folder_name}/analysis.html" class="model-card">
                <h3>{model_name}</h3>
                <p>View enhanced performance analysis with interactive error sample gallery</p>
            </a>
"""
    
    html_content += f"""        </div>
        
        <div class="timestamp">
            Enhanced index updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
    
    output_file = os.path.join('reports', 'enhanced_individual_analysis', 'index.html')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 增强版索引页面已创建: {output_file}")

if __name__ == "__main__":
    main()
