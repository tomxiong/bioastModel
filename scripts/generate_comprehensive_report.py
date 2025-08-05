"""
Comprehensive ONNX Conversion and Performance Report Generator
Analyzes all converted models and generates summary reports
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ComprehensiveReportGenerator:
    """Generate comprehensive analysis reports for all ONNX conversions"""
    
    def __init__(self):
        self.output_dir = Path("reports/comprehensive_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_batch_results(self) -> Dict[str, Any]:
        """Load the latest batch validation results"""
        batch_dir = Path("reports/batch_validation")
        
        # Find the most recent batch result file
        batch_files = list(batch_dir.glob("batch_validation_*.json"))
        if not batch_files:
            raise FileNotFoundError("No batch validation results found")
        
        latest_file = max(batch_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading batch results from: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_performance_data(self, batch_results: Dict) -> pd.DataFrame:
        """Extract performance data into a pandas DataFrame"""
        data = []
        
        for model_name, result in batch_results['results'].items():
            if not result['success'] or not result.get('validation_result', {}).get('success', False):
                # Add failed models with placeholder data
                data.append({
                    'model_name': model_name,
                    'architecture_type': result.get('config', {}).get('architecture_type', 'Unknown'),
                    'expected_accuracy': result.get('config', {}).get('expected_accuracy'),
                    'input_shape': str(result.get('config', {}).get('input_shape', 'Unknown')),
                    'conversion_success': result.get('conversion_success', False),
                    'validation_success': False,
                    'pytorch_avg_time_ms': None,
                    'onnx_avg_time_ms': None,
                    'speedup': None,
                    'max_diff': None,
                    'class_agreement': None,
                    'error': result.get('error', 'Unknown error')
                })
                continue
            
            validation = result['validation_result']
            config = result.get('config', {})
            
            data.append({
                'model_name': model_name,
                'architecture_type': config.get('architecture_type', 'Unknown'),
                'expected_accuracy': config.get('expected_accuracy'),
                'input_shape': str(config.get('input_shape', 'Unknown')),
                'conversion_success': result.get('conversion_success', False),
                'validation_success': validation.get('success', False),
                'pytorch_avg_time_ms': validation.get('pytorch_avg_time', 0) * 1000,
                'onnx_avg_time_ms': validation.get('onnx_avg_time', 0) * 1000,
                'speedup': validation.get('speedup', 0),
                'max_diff': validation.get('max_diff', 0),
                'class_agreement': validation.get('class_agreement', 0) * 100,
                'error': None
            })
        
        return pd.DataFrame(data)
    
    def generate_performance_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance summary statistics"""
        successful_df = df[df['validation_success'] == True]
        
        if len(successful_df) == 0:
            return {
                'total_models': len(df),
                'successful_conversions': 0,
                'failed_conversions': len(df),
                'success_rate': 0.0
            }
        
        summary = {
            'total_models': len(df),
            'successful_conversions': len(successful_df),
            'failed_conversions': len(df) - len(successful_df),
            'success_rate': len(successful_df) / len(df) * 100,
            'performance_metrics': {
                'average_speedup': successful_df['speedup'].mean(),
                'best_speedup': successful_df['speedup'].max(),
                'worst_speedup': successful_df['speedup'].min(),
                'speedup_std': successful_df['speedup'].std(),
                'average_accuracy_preservation': successful_df['class_agreement'].mean(),
                'best_accuracy': successful_df['class_agreement'].max(),
                'worst_accuracy': successful_df['class_agreement'].min(),
                'perfect_accuracy_models': len(successful_df[successful_df['class_agreement'] == 100.0])
            },
            'architecture_analysis': {}
        }
        
        # Architecture-specific analysis
        for arch_type in successful_df['architecture_type'].unique():
            arch_data = successful_df[successful_df['architecture_type'] == arch_type]
            summary['architecture_analysis'][arch_type] = {
                'count': len(arch_data),
                'avg_speedup': arch_data['speedup'].mean(),
                'avg_accuracy': arch_data['class_agreement'].mean(),
                'models': arch_data['model_name'].tolist()
            }
        
        return summary
    
    def create_visualization_dashboard(self, df: pd.DataFrame) -> str:
        """Create comprehensive visualization dashboard"""
        successful_df = df[df['validation_success'] == True]
        
        if len(successful_df) == 0:
            # Create a simple error visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No successful conversions found', 
                   ha='center', va='center', fontsize=16, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = self.output_dir / f"comprehensive_dashboard_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(fig_path)
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive ONNX Conversion Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Speedup comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(successful_df['model_name'], successful_df['speedup'], 
                      color='steelblue', alpha=0.7)
        ax1.set_title('Performance Speedup by Model')
        ax1.set_ylabel('Speedup (x)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, speedup in zip(bars, successful_df['speedup']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{speedup:.2f}x', ha='center', va='bottom')
        
        # 2. Architecture type analysis
        ax2 = axes[0, 1]
        arch_counts = successful_df['architecture_type'].value_counts()
        ax2.pie(arch_counts.values, labels=arch_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Successful Conversions by Architecture Type')
        
        # 3. Accuracy preservation
        ax3 = axes[0, 2]
        ax3.bar(successful_df['model_name'], successful_df['class_agreement'], 
               color='green', alpha=0.7)
        ax3.set_title('Classification Accuracy Preservation')
        ax3.set_ylabel('Accuracy (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(90, 101)
        ax3.grid(True, alpha=0.3)
        
        # 4. Inference time comparison
        ax4 = axes[1, 0]
        x_pos = np.arange(len(successful_df))
        width = 0.35
        
        pytorch_times = ax4.bar(x_pos - width/2, successful_df['pytorch_avg_time_ms'], 
                               width, label='PyTorch', color='orange', alpha=0.7)
        onnx_times = ax4.bar(x_pos + width/2, successful_df['onnx_avg_time_ms'], 
                            width, label='ONNX', color='blue', alpha=0.7)
        
        ax4.set_title('Average Inference Time Comparison')
        ax4.set_ylabel('Time (ms)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(successful_df['model_name'], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Speedup vs Architecture scatter
        ax5 = axes[1, 1]
        arch_colors = {'CNN': 'blue', 'Vision Transformer': 'red', 'Mobile CNN': 'green', 
                      'Modern CNN': 'purple', 'Hybrid CNN-Transformer': 'orange'}
        
        for arch_type in successful_df['architecture_type'].unique():
            arch_data = successful_df[successful_df['architecture_type'] == arch_type]
            ax5.scatter(arch_data['pytorch_avg_time_ms'], arch_data['speedup'], 
                       label=arch_type, color=arch_colors.get(arch_type, 'gray'), 
                       alpha=0.7, s=100)
        
        ax5.set_title('Speedup vs PyTorch Inference Time')
        ax5.set_xlabel('PyTorch Time (ms)')
        ax5.set_ylabel('Speedup (x)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate summary stats
        avg_speedup = successful_df['speedup'].mean()
        avg_accuracy = successful_df['class_agreement'].mean()
        total_success = len(successful_df)
        total_models = len(df)
        
        summary_text = f"""
Performance Summary

Total Models: {total_models}
Successful: {total_success}
Success Rate: {total_success/total_models*100:.1f}%

Average Speedup: {avg_speedup:.2f}x
Best Speedup: {successful_df['speedup'].max():.2f}x
Worst Speedup: {successful_df['speedup'].min():.2f}x

Avg Accuracy: {avg_accuracy:.1f}%
Perfect Accuracy: {len(successful_df[successful_df['class_agreement'] == 100.0])} models

Architecture Distribution:
"""
        
        for arch, count in successful_df['architecture_type'].value_counts().items():
            summary_text += f"• {arch}: {count}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_path = self.output_dir / f"comprehensive_dashboard_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive dashboard saved: {fig_path}")
        return str(fig_path)
    
    def generate_detailed_analysis_report(self, df: pd.DataFrame, summary: Dict, 
                                        dashboard_path: str) -> str:
        """Generate detailed analysis report in HTML format"""
        successful_df = df[df['validation_success'] == True]
        failed_df = df[df['validation_success'] == False]
        
        # Encode dashboard image
        dashboard_base64 = ""
        try:
            with open(dashboard_path, 'rb') as f:
                import base64
                dashboard_base64 = base64.b64encode(f.read()).decode('utf-8')
        except:
            pass
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive ONNX Conversion Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 3em;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            margin-top: 15px;
            opacity: 0.9;
            font-size: 1.3em;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .summary-card:hover {{
            transform: translateY(-2px);
        }}
        
        .summary-card h3 {{
            margin: 0 0 15px 0;
            color: #495057;
            font-size: 1.1em;
        }}
        
        .summary-card .value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #007bff;
            margin-bottom: 10px;
        }}
        
        .summary-card .unit {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 25px;
            overflow: hidden;
        }}
        
        .card-header {{
            background: #f8f9fa;
            padding: 20px 25px;
            border-bottom: 1px solid #dee2e6;
            font-weight: 600;
            font-size: 1.3em;
            color: #495057;
        }}
        
        .card-body {{
            padding: 25px;
        }}
        
        .dashboard-container {{
            text-align: center;
            margin: 30px 0;
        }}
        
        .dashboard-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .status-success {{
            color: #28a745;
            font-weight: 600;
        }}
        
        .status-failed {{
            color: #dc3545;
            font-weight: 600;
        }}
        
        .metric-excellent {{
            color: #28a745;
            font-weight: 600;
        }}
        
        .metric-good {{
            color: #ffc107;
            font-weight: 600;
        }}
        
        .metric-poor {{
            color: #dc3545;
            font-weight: 600;
        }}
        
        .architecture-tag {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
            margin: 2px;
        }}
        
        .arch-cnn {{ background: #e3f2fd; color: #1976d2; }}
        .arch-transformer {{ background: #fce4ec; color: #c2185b; }}
        .arch-mobile {{ background: #e8f5e8; color: #388e3c; }}
        .arch-modern {{ background: #f3e5f5; color: #7b1fa2; }}
        .arch-hybrid {{ background: #fff3e0; color: #f57c00; }}
        
        .timestamp {{
            color: #6c757d;
            font-size: 0.9em;
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}
        
        .recommendations {{
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            padding: 25px;
            border-radius: 12px;
            margin: 25px 0;
        }}
        
        .recommendations h4 {{
            margin-top: 0;
            color: #495057;
        }}
        
        .recommendation-item {{
            margin: 10px 0;
            padding-left: 20px;
            position: relative;
        }}
        
        .recommendation-item:before {{
            content: "•";
            color: #007bff;
            font-weight: bold;
            position: absolute;
            left: 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ONNX Conversion Analysis</h1>
        <div class="subtitle">Comprehensive Performance Report</div>
        <div class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    
    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total Models</h3>
            <div class="value">{summary['total_models']}</div>
        </div>
        <div class="summary-card">
            <h3>Success Rate</h3>
            <div class="value">{summary['success_rate']:.1f}<span class="unit">%</span></div>
        </div>
        <div class="summary-card">
            <h3>Average Speedup</h3>
            <div class="value">{summary.get('performance_metrics', {}).get('average_speedup', 0):.2f}<span class="unit">x</span></div>
        </div>
        <div class="summary-card">
            <h3>Perfect Accuracy</h3>
            <div class="value">{summary.get('performance_metrics', {}).get('perfect_accuracy_models', 0)}</div>
            <div class="unit">models</div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">Performance Dashboard</div>
        <div class="card-body">
            <div class="dashboard-container">
                <img src="data:image/png;base64,{dashboard_base64}" alt="Performance Dashboard" />
            </div>
        </div>
    </div>
    
    <div class="card">
        <div class="card-header">Detailed Model Results</div>
        <div class="card-body">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Architecture</th>
                        <th>Status</th>
                        <th>Speedup</th>
                        <th>Accuracy</th>
                        <th>Input Shape</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add successful models
        for _, row in successful_df.iterrows():
            arch_class = row['architecture_type'].lower().replace(' ', '-').replace('cnn', 'cnn')
            arch_class_map = {
                'cnn': 'arch-cnn',
                'vision-transformer': 'arch-transformer', 
                'mobile-cnn': 'arch-mobile',
                'modern-cnn': 'arch-modern',
                'hybrid-cnn-transformer': 'arch-hybrid'
            }
            arch_css = arch_class_map.get(arch_class, 'arch-cnn')
            
            speedup_css = 'metric-excellent' if row['speedup'] > 3 else 'metric-good' if row['speedup'] > 1.5 else 'metric-poor'
            accuracy_css = 'metric-excellent' if row['class_agreement'] == 100 else 'metric-good' if row['class_agreement'] > 95 else 'metric-poor'
            
            html_content += f"""
                    <tr>
                        <td><strong>{row['model_name']}</strong></td>
                        <td><span class="architecture-tag {arch_css}">{row['architecture_type']}</span></td>
                        <td><span class="status-success">Success</span></td>
                        <td><span class="{speedup_css}">{row['speedup']:.2f}x</span></td>
                        <td><span class="{accuracy_css}">{row['class_agreement']:.1f}%</span></td>
                        <td>{row['input_shape']}</td>
                        <td>Fully functional</td>
                    </tr>
"""
        
        # Add failed models
        for _, row in failed_df.iterrows():
            arch_class_map = {
                'cnn': 'arch-cnn',
                'vision-transformer': 'arch-transformer', 
                'mobile-cnn': 'arch-mobile',
                'modern-cnn': 'arch-modern',
                'hybrid-cnn-transformer': 'arch-hybrid'
            }
            arch_class = row['architecture_type'].lower().replace(' ', '-').replace('cnn', 'cnn')
            arch_css = arch_class_map.get(arch_class, 'arch-cnn')
            
            html_content += f"""
                    <tr>
                        <td><strong>{row['model_name']}</strong></td>
                        <td><span class="architecture-tag {arch_css}">{row['architecture_type']}</span></td>
                        <td><span class="status-failed">Failed</span></td>
                        <td>N/A</td>
                        <td>N/A</td>
                        <td>{row['input_shape']}</td>
                        <td>{row.get('error', 'Unknown error')[:50]}...</td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
"""
        
        # Add architecture analysis if available
        if summary.get('architecture_analysis'):
            html_content += """
    <div class="card">
        <div class="card-header">Architecture Performance Analysis</div>
        <div class="card-body">
            <table>
                <thead>
                    <tr>
                        <th>Architecture Type</th>
                        <th>Models Count</th>
                        <th>Average Speedup</th>
                        <th>Average Accuracy</th>
                        <th>Models</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for arch_type, arch_data in summary['architecture_analysis'].items():
                models_list = ', '.join(arch_data['models'])
                html_content += f"""
                    <tr>
                        <td>{arch_type}</td>
                        <td>{arch_data['count']}</td>
                        <td>{arch_data['avg_speedup']:.2f}x</td>
                        <td>{arch_data['avg_accuracy']:.1f}%</td>
                        <td>{models_list}</td>
                    </tr>
"""
            
            html_content += """
                </tbody>
            </table>
        </div>
    </div>
"""
        
        # Add recommendations
        html_content += """
    <div class="recommendations">
        <h4>Key Findings and Recommendations</h4>
"""
        
        if len(successful_df) > 0:
            best_model = successful_df.loc[successful_df['speedup'].idxmax()]
            html_content += f"""
        <div class="recommendation-item">
            <strong>Best Performing Model:</strong> {best_model['model_name']} achieved {best_model['speedup']:.2f}x speedup with {best_model['class_agreement']:.1f}% accuracy preservation
        </div>
"""
        
        if len(failed_df) > 0:
            html_content += f"""
        <div class="recommendation-item">
            <strong>Failed Conversions:</strong> {len(failed_df)} models failed conversion. Review input shapes and model complexity
        </div>
"""
        
        if len(successful_df) > 0:
            avg_speedup = successful_df['speedup'].mean()
            if avg_speedup > 3:
                html_content += """
        <div class="recommendation-item">
            <strong>Excellent Performance:</strong> Average speedup > 3x indicates highly successful ONNX optimization
        </div>
"""
            elif avg_speedup > 1.5:
                html_content += """
        <div class="recommendation-item">
            <strong>Good Performance:</strong> Moderate speedup achieved. Consider further optimization techniques
        </div>
"""
        
        html_content += """
    </div>
    
    <div class="timestamp">
        Report generated using Enhanced ONNX Validation System on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
    </div>
</body>
</html>
        """
        
        # Save HTML report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = self.output_dir / f"comprehensive_analysis_{timestamp}.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive HTML report saved: {html_path}")
        return str(html_path)
    
    def generate_json_summary(self, df: pd.DataFrame, summary: Dict) -> str:
        """Generate JSON summary report"""
        # Convert DataFrame to records for JSON serialization
        model_details = []
        for _, row in df.iterrows():
            model_details.append({
                'model_name': row['model_name'],
                'architecture_type': row['architecture_type'],
                'expected_accuracy': row['expected_accuracy'],
                'input_shape': row['input_shape'],
                'conversion_success': bool(row['conversion_success']),
                'validation_success': bool(row['validation_success']),
                'performance_metrics': {
                    'pytorch_avg_time_ms': float(row['pytorch_avg_time_ms']) if pd.notna(row['pytorch_avg_time_ms']) else None,
                    'onnx_avg_time_ms': float(row['onnx_avg_time_ms']) if pd.notna(row['onnx_avg_time_ms']) else None,
                    'speedup': float(row['speedup']) if pd.notna(row['speedup']) else None,
                    'max_diff': float(row['max_diff']) if pd.notna(row['max_diff']) else None,
                    'class_agreement_percent': float(row['class_agreement']) if pd.notna(row['class_agreement']) else None
                },
                'error': row['error'] if pd.notna(row['error']) else None
            })
        
        json_summary = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_onnx_analysis',
                'total_models_analyzed': len(df)
            },
            'summary_statistics': summary,
            'model_details': model_details,
            'recommendations': self._generate_recommendations(df, summary)
        }
        
        # Save JSON summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.output_dir / f"comprehensive_summary_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        
        print(f"JSON summary saved: {json_path}")
        return str(json_path)
    
    def _generate_recommendations(self, df: pd.DataFrame, summary: Dict) -> List[str]:
        """Generate automated recommendations based on analysis"""
        recommendations = []
        successful_df = df[df['validation_success'] == True]
        
        if len(successful_df) == 0:
            return ["All conversions failed. Review model architectures and conversion strategies."]
        
        # Performance recommendations
        avg_speedup = successful_df['speedup'].mean()
        if avg_speedup > 4:
            recommendations.append("Excellent ONNX performance achieved with >4x average speedup")
        elif avg_speedup > 2:
            recommendations.append("Good ONNX performance with >2x average speedup")
        else:
            recommendations.append("Moderate speedup achieved. Consider model optimization techniques")
        
        # Accuracy recommendations
        perfect_accuracy = len(successful_df[successful_df['class_agreement'] == 100.0])
        if perfect_accuracy == len(successful_df):
            recommendations.append("Perfect accuracy preservation across all models")
        elif perfect_accuracy > len(successful_df) * 0.8:
            recommendations.append("Excellent accuracy preservation in most models")
        else:
            recommendations.append("Some accuracy degradation observed. Review conversion parameters")
        
        # Architecture-specific recommendations
        if 'Vision Transformer' in successful_df['architecture_type'].values:
            vit_data = successful_df[successful_df['architecture_type'] == 'Vision Transformer']
            if len(vit_data) > 0 and vit_data['speedup'].mean() > 2:
                recommendations.append("Vision Transformer models show good ONNX compatibility")
        
        return recommendations
    
    def run_comprehensive_analysis(self):
        """Run complete comprehensive analysis"""
        print("Starting comprehensive ONNX conversion analysis...")
        
        try:
            # Load batch results
            batch_results = self.load_batch_results()
            
            # Extract performance data
            df = self.extract_performance_data(batch_results)
            print(f"Analyzed {len(df)} models")
            
            # Generate summary statistics
            summary = self.generate_performance_summary(df)
            print(f"Success rate: {summary['success_rate']:.1f}%")
            
            # Create visualization dashboard
            dashboard_path = self.create_visualization_dashboard(df)
            
            # Generate detailed HTML report
            html_report_path = self.generate_detailed_analysis_report(df, summary, dashboard_path)
            
            # Generate JSON summary
            json_report_path = self.generate_json_summary(df, summary)
            
            print(f"\nComprehensive analysis completed!")
            print(f"HTML Report: {html_report_path}")
            print(f"JSON Summary: {json_report_path}")
            print(f"Dashboard: {dashboard_path}")
            
            return {
                'html_report': html_report_path,
                'json_summary': json_report_path,
                'dashboard': dashboard_path,
                'summary': summary
            }
            
        except Exception as e:
            print(f"Error during comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def main():
    """Main function"""
    generator = ComprehensiveReportGenerator()
    results = generator.run_comprehensive_analysis()
    
    if results.get('success', True):
        print("\nAnalysis Summary:")
        summary = results.get('summary', {})
        print(f"Total Models: {summary.get('total_models', 0)}")
        print(f"Successful: {summary.get('successful_conversions', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        perf_metrics = summary.get('performance_metrics', {})
        if perf_metrics:
            print(f"Average Speedup: {perf_metrics.get('average_speedup', 0):.2f}x")
            print(f"Perfect Accuracy Models: {perf_metrics.get('perfect_accuracy_models', 0)}")

if __name__ == "__main__":
    main()