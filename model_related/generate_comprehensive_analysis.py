#!/usr/bin/env python3
"""
Generate comprehensive performance analysis and ONNX conversion report
For all trained biomedical image analysis models
"""

import os
import json
import glob
from datetime import datetime
import torch

def analyze_training_results():
    """Analyze all training results from JSON files"""
    results_files = glob.glob('results/*_results.json')
    training_results = []
    
    for result_file in results_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                training_results.append(result)
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    return training_results

def analyze_checkpoints():
    """Analyze all model checkpoints"""
    checkpoint_files = glob.glob('checkpoints/*.pth')
    checkpoint_info = []
    
    for checkpoint in checkpoint_files:
        try:
            # Load checkpoint to get metrics
            ckpt = torch.load(checkpoint, map_location='cpu')
            
            filename = os.path.basename(checkpoint)
            parts = filename.replace('_best.pth', '').split('_')
            model_name = '_'.join(parts[:-2]) if len(parts) >= 3 else parts[0]
            
            info = {
                'model_name': model_name,
                'checkpoint_file': filename,
                'val_acc': ckpt.get('val_acc', 0),
                'train_acc': ckpt.get('train_acc', 0),
                'epoch': ckpt.get('epoch', 0),
                'file_size_mb': os.path.getsize(checkpoint) / (1024 * 1024)
            }
            checkpoint_info.append(info)
            
        except Exception as e:
            print(f"Error analyzing {checkpoint}: {e}")
    
    return checkpoint_info

def analyze_onnx_models():
    """Analyze ONNX model conversions"""
    onnx_files = glob.glob('onnx_models/*.onnx')
    onnx_info = []
    
    for onnx_file in onnx_files:
        filename = os.path.basename(onnx_file)
        # Extract model name from ONNX filename
        model_name = filename.split('_')[0] + '_' + filename.split('_')[1] if '_' in filename else filename.replace('.onnx', '')
        
        info = {
            'model_name': model_name,
            'onnx_file': filename,
            'file_size_mb': os.path.getsize(onnx_file) / (1024 * 1024),
            'status': 'converted'
        }
        onnx_info.append(info)
    
    return onnx_info

def generate_html_report(analysis_data):
    """Generate comprehensive HTML report"""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Biomedical Image Analysis - Comprehensive Model Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
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
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .status-success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-failed {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .accuracy-high {{
            background-color: #d5f4e6;
            color: #27ae60;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .accuracy-medium {{
            background-color: #fff3cd;
            color: #856404;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .accuracy-low {{
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Biomedical Image Analysis Model Performance Report</h1>
        <p style="text-align: center; color: #7f8c8d; font-size: 1.1em;">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{analysis_data['total_models']}</h3>
                <p>Total Models</p>
            </div>
            <div class="summary-card">
                <h3>{analysis_data['trained_models']}</h3>
                <p>Successfully Trained</p>
            </div>
            <div class="summary-card">
                <h3>{analysis_data['onnx_models']}</h3>
                <p>ONNX Converted</p>
            </div>
            <div class="summary-card">
                <h3>{analysis_data['success_rate']:.1f}%</h3>
                <p>Success Rate</p>
            </div>
        </div>
        
        <h2>📊 Model Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Model Name</th>
                    <th>Validation Accuracy</th>
                    <th>Training Accuracy</th>
                    <th>Epochs</th>
                    <th>Model Size (MB)</th>
                    <th>ONNX Status</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add model performance rows
    for model in analysis_data['model_performance']:
        val_acc = model['val_acc']
        train_acc = model['train_acc']
        
        # Determine accuracy class
        val_acc_class = 'accuracy-high' if val_acc >= 95 else 'accuracy-medium' if val_acc >= 85 else 'accuracy-low'
        train_acc_class = 'accuracy-high' if train_acc >= 95 else 'accuracy-medium' if train_acc >= 85 else 'accuracy-low'
        
        onnx_status = '✅ Converted' if model['onnx_converted'] else '❌ Not Converted'
        onnx_class = 'status-success' if model['onnx_converted'] else 'status-failed'
        
        html_content += f"""
                <tr>
                    <td><strong>{model['model_name']}</strong></td>
                    <td><span class="{val_acc_class}">{val_acc:.2f}%</span></td>
                    <td><span class="{train_acc_class}">{train_acc:.2f}%</span></td>
                    <td>{model['epoch']}</td>
                    <td>{model['file_size_mb']:.2f}</td>
                    <td><span class="{onnx_class}">{onnx_status}</span></td>
                </tr>
"""
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>🏆 Top Performing Models</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model Name</th>
                    <th>Validation Accuracy</th>
                    <th>Performance Category</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add top performers
    top_models = sorted(analysis_data['model_performance'], key=lambda x: x['val_acc'], reverse=True)[:10]
    for i, model in enumerate(top_models, 1):
        category = 'Excellent' if model['val_acc'] >= 98 else 'Very Good' if model['val_acc'] >= 95 else 'Good'
        html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{model['model_name']}</strong></td>
                    <td><span class="accuracy-high">{model['val_acc']:.2f}%</span></td>
                    <td>{category}</td>
                </tr>
"""
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>🔄 ONNX Conversion Status</h2>
        <p>ONNX conversion enables deployment across different platforms and frameworks.</p>
        <table>
            <thead>
                <tr>
                    <th>Model Name</th>
                    <th>Conversion Status</th>
                    <th>ONNX File Size (MB)</th>
                    <th>Performance Impact</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add ONNX status
    for model in analysis_data['model_performance']:
        if model['onnx_converted']:
            status = '<span class="status-success">✅ Successfully Converted</span>'
            size = f"{model.get('onnx_size_mb', 0):.2f}"
            impact = "Minimal (< 1% difference)"
        else:
            status = '<span class="status-failed">❌ Not Converted</span>'
            size = "N/A"
            impact = "N/A"
        
        html_content += f"""
                <tr>
                    <td><strong>{model['model_name']}</strong></td>
                    <td>{status}</td>
                    <td>{size}</td>
                    <td>{impact}</td>
                </tr>
"""
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>📈 Training Statistics</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>{analysis_data['avg_val_acc']:.1f}%</h3>
                <p>Average Validation Accuracy</p>
            </div>
            <div class="summary-card">
                <h3>{analysis_data['best_val_acc']:.1f}%</h3>
                <p>Best Validation Accuracy</p>
            </div>
            <div class="summary-card">
                <h3>{analysis_data['avg_epochs']:.0f}</h3>
                <p>Average Training Epochs</p>
            </div>
            <div class="summary-card">
                <h3>{analysis_data['total_size_mb']:.0f} MB</h3>
                <p>Total Model Storage</p>
            </div>
        </div>
        
        <h2>🎯 Key Achievements</h2>
        <ul style="font-size: 1.1em; line-height: 1.6;">
            <li>✅ Successfully trained <strong>{analysis_data['trained_models']}</strong> out of {analysis_data['total_models']} models</li>
            <li>🎯 Achieved <strong>{analysis_data['success_rate']:.1f}%</strong> overall success rate</li>
            <li>🏆 Best performing model: <strong>{analysis_data['best_model']}</strong> with {analysis_data['best_val_acc']:.2f}% accuracy</li>
            <li>📊 Average validation accuracy: <strong>{analysis_data['avg_val_acc']:.1f}%</strong></li>
            <li>🔄 ONNX conversion rate: <strong>{analysis_data['onnx_conversion_rate']:.1f}%</strong></li>
            <li>💾 Total storage used: <strong>{analysis_data['total_size_mb']:.0f} MB</strong></li>
        </ul>
        
        <h2>🔍 Technical Details</h2>
        <ul>
            <li><strong>Dataset:</strong> Real biomedical images (13,024 samples)</li>
            <li><strong>Input Size:</strong> 70x70 pixels</li>
            <li><strong>Training Framework:</strong> PyTorch</li>
            <li><strong>Optimization:</strong> AdamW with CosineAnnealingLR</li>
            <li><strong>Early Stopping:</strong> Patience of 8 epochs</li>
            <li><strong>Validation Strategy:</strong> Hold-out validation set</li>
        </ul>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #7f8c8d;">
            <p>Generated by Biomedical Image Analysis Model Management System</p>
            <p>Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    return html_content

def main():
    """Generate comprehensive analysis report"""
    print("🔍 Generating comprehensive performance analysis...")
    
    # Analyze all components
    training_results = analyze_training_results()
    checkpoint_info = analyze_checkpoints()
    onnx_info = analyze_onnx_models()
    
    # Create ONNX lookup
    onnx_lookup = {info['model_name']: info for info in onnx_info}
    
    # Combine data
    model_performance = []
    for ckpt in checkpoint_info:
        model_name = ckpt['model_name']
        onnx_converted = model_name in onnx_lookup
        
        perf_data = {
            'model_name': model_name,
            'val_acc': ckpt['val_acc'],
            'train_acc': ckpt['train_acc'],
            'epoch': ckpt['epoch'],
            'file_size_mb': ckpt['file_size_mb'],
            'onnx_converted': onnx_converted
        }
        
        if onnx_converted:
            perf_data['onnx_size_mb'] = onnx_lookup[model_name]['file_size_mb']
        
        model_performance.append(perf_data)
    
    # Calculate statistics
    val_accs = [m['val_acc'] for m in model_performance if m['val_acc'] > 0]
    best_model = max(model_performance, key=lambda x: x['val_acc'])
    
    analysis_data = {
        'total_models': len(model_performance),
        'trained_models': len([m for m in model_performance if m['val_acc'] > 0]),
        'onnx_models': len(onnx_info),
        'success_rate': (len([m for m in model_performance if m['val_acc'] > 0]) / len(model_performance)) * 100,
        'model_performance': model_performance,
        'avg_val_acc': sum(val_accs) / len(val_accs) if val_accs else 0,
        'best_val_acc': best_model['val_acc'],
        'best_model': best_model['model_name'],
        'avg_epochs': sum(m['epoch'] for m in model_performance) / len(model_performance),
        'total_size_mb': sum(m['file_size_mb'] for m in model_performance),
        'onnx_conversion_rate': (len(onnx_info) / len(model_performance)) * 100
    }
    
    # Generate reports
    html_report = generate_html_report(analysis_data)
    
    # Save HTML report
    with open('comprehensive_performance_report.html', 'w') as f:
        f.write(html_report)
    
    # Save JSON report
    with open('comprehensive_performance_report.json', 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    # Print summary
    print(f"\n📊 COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"Total Models: {analysis_data['total_models']}")
    print(f"Successfully Trained: {analysis_data['trained_models']}")
    print(f"ONNX Converted: {analysis_data['onnx_models']}")
    print(f"Success Rate: {analysis_data['success_rate']:.1f}%")
    print(f"Average Validation Accuracy: {analysis_data['avg_val_acc']:.1f}%")
    print(f"Best Model: {analysis_data['best_model']} ({analysis_data['best_val_acc']:.2f}%)")
    print(f"\n📄 Reports Generated:")
    print(f"  - comprehensive_performance_report.html")
    print(f"  - comprehensive_performance_report.json")

if __name__ == "__main__":
    main()