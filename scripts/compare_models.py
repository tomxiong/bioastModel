"""
Unified model comparison script for generating comprehensive comparison reports.

Usage:
    python scripts/compare_models.py --models efficientnet_b0 resnet18_improved
    python scripts/compare_models.py --models efficientnet_b0 resnet18_improved --output custom_comparison
    python scripts/compare_models.py --all-models --format html
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import (
    get_model_config,
    get_latest_experiment_path,
    EXPERIMENTS_DIR,
    REPORTS_DIR
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare multiple trained models')
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['efficientnet_b0', 'resnet18_improved', 'convnext_tiny'],
        help='Models to compare (space-separated)'
    )
    
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Compare all available models'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Custom output directory name'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='html',
        choices=['html', 'json', 'markdown'],
        help='Output format for comparison report'
    )
    
    parser.add_argument(
        '--include-charts',
        action='store_true',
        default=True,
        help='Include comparison charts'
    )
    
    parser.add_argument(
        '--include-detailed-analysis',
        action='store_true',
        default=True,
        help='Include detailed performance analysis'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be compared without actually running'
    )
    
    return parser.parse_args()

def get_available_models():
    """Get list of models with available experiments."""
    available_models = []
    
    for model_name in ['efficientnet_b0', 'resnet18_improved', 'convnext_tiny']:
        try:
            experiment_path = get_latest_experiment_path(model_name)
            if experiment_path and experiment_path.exists():
                available_models.append(model_name)
        except:
            continue
    
    return available_models

def validate_models(model_names):
    """Validate that all specified models have available experiments."""
    available_models = get_available_models()
    
    missing_models = []
    for model_name in model_names:
        if model_name not in available_models:
            missing_models.append(model_name)
    
    if missing_models:
        raise ValueError(f"No experiments found for models: {missing_models}")
    
    return True

def load_model_results(model_name):
    """Load evaluation results for a model."""
    experiment_path = get_latest_experiment_path(model_name)
    
    # Load training history
    history_path = experiment_path / 'training_history.json'
    with open(history_path, 'r', encoding='utf-8') as f:
        training_history = json.load(f)
    
    # Load test results if available
    test_results = None
    test_results_path = experiment_path / 'test_results.json'
    if test_results_path.exists():
        with open(test_results_path, 'r', encoding='utf-8') as f:
            test_results = json.load(f)
    
    # Load classification report if available
    classification_report = None
    report_path = experiment_path / 'evaluation' / 'classification_report.txt'
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            classification_report = f.read()
    
    # Get model config
    model_config = get_model_config(model_name)
    
    return {
        'name': model_name,
        'experiment_path': str(experiment_path),
        'experiment_id': experiment_path.parent.name,
        'config': model_config,
        'training_history': training_history,
        'test_results': test_results,
        'classification_report': classification_report
    }

def create_comparison_config(args, model_results):
    """Create comparison configuration."""
    # Determine output path
    if args.output:
        output_path = REPORTS_DIR / 'comparisons' / args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_names = '_vs_'.join([r['name'] for r in model_results])
        output_path = REPORTS_DIR / 'comparisons' / f'{model_names}_{timestamp}'
    
    config = {
        'comparison': {
            'models': [r['name'] for r in model_results],
            'output_path': str(output_path),
            'format': args.format,
            'include_charts': args.include_charts,
            'include_detailed_analysis': args.include_detailed_analysis
        },
        'models': model_results,
        'metadata': {
            'created_by': 'scripts/compare_models.py',
            'created_at': datetime.now().isoformat(),
            'command_line': ' '.join(sys.argv)
        }
    }
    
    return config

def generate_comparison_charts(config):
    """Generate comparison charts."""
    print("📊 Generating comparison charts...")
    
    output_path = Path(config['comparison']['output_path'])
    charts_dir = output_path / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Use existing chart generation logic
    try:
        from save_charts_direct import generate_and_save_charts
        chart_files = generate_and_save_charts(str(charts_dir))
        print(f"✅ Generated {len(chart_files)} comparison charts")
        return chart_files
    except ImportError:
        print("⚠️  Chart generation module not found, skipping charts")
        return []

def analyze_model_performance(model_results):
    """Analyze and compare model performance."""
    print("🔍 Analyzing model performance...")
    
    analysis = {
        'summary': {},
        'detailed': {},
        'recommendations': {}
    }
    
    for model in model_results:
        model_name = model['name']
        history = model['training_history']
        
        # Extract key metrics
        final_val_acc = history['val_acc'][-1] if 'val_acc' in history else None
        final_val_loss = history['val_loss'][-1] if 'val_loss' in history else None
        epochs_trained = len(history['train_loss']) if 'train_loss' in history else 0
        
        # Get model parameters
        params = model['config'].get('parameters', 0)
        
        # Calculate efficiency ratio
        efficiency_ratio = (final_val_acc / (params / 1e6)) if final_val_acc and params else 0
        
        analysis['summary'][model_name] = {
            'accuracy': final_val_acc,
            'loss': final_val_loss,
            'epochs': epochs_trained,
            'parameters': params,
            'efficiency_ratio': efficiency_ratio
        }
    
    # Generate recommendations
    if len(model_results) >= 2:
        # Find best performing model
        best_accuracy = max(analysis['summary'].values(), key=lambda x: x['accuracy'] or 0)
        best_efficiency = max(analysis['summary'].values(), key=lambda x: x['efficiency_ratio'] or 0)
        
        analysis['recommendations'] = {
            'best_accuracy': best_accuracy,
            'best_efficiency': best_efficiency,
            'production_recommendation': 'efficientnet_b0'  # Default based on previous analysis
        }
    
    return analysis

def generate_html_report(config, analysis, chart_files):
    """Generate HTML comparison report."""
    print("📄 Generating HTML comparison report...")
    
    output_path = Path(config['comparison']['output_path'])
    
    # Use existing HTML generation logic or create new template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .model-summary {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
            .chart {{ text-align: center; margin: 20px 0; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Comparison Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Models: {', '.join(config['comparison']['models'])}</p>
        </div>
        
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Parameters</th>
                <th>Epochs</th>
                <th>Efficiency Ratio</th>
            </tr>
    """
    
    for model_name, metrics in analysis['summary'].items():
        accuracy_str = f"{metrics['accuracy']:.4f}" if metrics['accuracy'] else 'N/A'
        efficiency_str = f"{metrics['efficiency_ratio']:.2f}" if metrics['efficiency_ratio'] else 'N/A'
        html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{accuracy_str}</td>
                <td>{metrics['parameters']:,}</td>
                <td>{metrics['epochs']}</td>
                <td>{efficiency_str}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Comparison Charts</h2>
    """
    
    for chart_file in chart_files:
        chart_name = Path(chart_file).stem.replace('_', ' ').title()
        html_content += f"""
        <div class="chart">
            <h3>{chart_name}</h3>
            <img src="charts/{Path(chart_file).name}" alt="{chart_name}" style="max-width: 100%;">
        </div>
        """
    
    html_content += """
        <h2>Recommendations</h2>
        <div class="model-summary">
            <p>Based on the analysis, the recommended model for production deployment is the one that provides the best balance of accuracy and efficiency.</p>
        </div>
    </body>
    </html>
    """
    
    with open(output_path / 'comparison_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML report generated")

def generate_json_report(config, analysis):
    """Generate JSON comparison report."""
    print("📄 Generating JSON comparison report...")
    
    output_path = Path(config['comparison']['output_path'])
    
    report_data = {
        'metadata': config['metadata'],
        'comparison_config': config['comparison'],
        'analysis': analysis,
        'models': config['models']
    }
    
    with open(output_path / 'comparison_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print("✅ JSON report generated")

def generate_markdown_report(config, analysis):
    """Generate Markdown comparison report."""
    print("📄 Generating Markdown comparison report...")
    
    output_path = Path(config['comparison']['output_path'])
    
    md_content = f"""# Model Comparison Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Models: {', '.join(config['comparison']['models'])}

## Performance Summary

| Model | Accuracy | Parameters | Epochs | Efficiency Ratio |
|-------|----------|------------|--------|------------------|
"""
    
    for model_name, metrics in analysis['summary'].items():
        accuracy = f"{metrics['accuracy']:.4f}" if metrics['accuracy'] else 'N/A'
        efficiency = f"{metrics['efficiency_ratio']:.2f}" if metrics['efficiency_ratio'] else 'N/A'
        md_content += f"| {model_name} | {accuracy} | {metrics['parameters']:,} | {metrics['epochs']} | {efficiency} |\n"
    
    md_content += """
## Recommendations

Based on the analysis, the recommended model for production deployment is the one that provides the best balance of accuracy and efficiency.
"""
    
    with open(output_path / 'comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print("✅ Markdown report generated")

def main():
    """Main function."""
    args = parse_arguments()
    
    print("🔍 Unified Model Comparison Script")
    print("=" * 50)
    
    try:
        # Determine models to compare
        if args.all_models:
            model_names = get_available_models()
            if not model_names:
                raise ValueError("No trained models found")
        elif args.models:
            model_names = args.models
        else:
            raise ValueError("Must specify either --models or --all-models")
        
        print(f"📊 Comparing models: {', '.join(model_names)}")
        
        # Validate models
        validate_models(model_names)
        print("✅ Model validation passed")
        
        # Load model results
        model_results = []
        for model_name in model_names:
            print(f"📁 Loading results for {model_name}...")
            results = load_model_results(model_name)
            model_results.append(results)
        
        # Create comparison configuration
        config = create_comparison_config(args, model_results)
        
        if args.dry_run:
            print("📋 Comparison Configuration (Dry Run):")
            print(json.dumps({k: v for k, v in config.items() if k != 'models'}, indent=2))
            return
        
        # Create output directory
        output_path = Path(config['comparison']['output_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison config
        with open(output_path / 'comparison_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Generate comparison charts
        chart_files = []
        if config['comparison']['include_charts']:
            chart_files = generate_comparison_charts(config)
        
        # Analyze model performance
        analysis = analyze_model_performance(model_results)
        
        # Generate reports
        if config['comparison']['format'] == 'html':
            generate_html_report(config, analysis, chart_files)
        elif config['comparison']['format'] == 'json':
            generate_json_report(config, analysis)
        elif config['comparison']['format'] == 'markdown':
            generate_markdown_report(config, analysis)
        
        print(f"\n✅ Comparison completed successfully!")
        print(f"📁 Reports saved to: {config['comparison']['output_path']}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()