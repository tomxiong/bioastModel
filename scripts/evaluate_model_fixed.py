#!/usr/bin/env python3
"""
统一模型评估脚本 (修复版)
"""

import argparse
import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.evaluator import ModelEvaluator
from training.dataset import create_data_loaders
from core.config import get_model_config, DATA_DIR

def load_model_and_weights(model_name, experiment_path, device):
    """加载模型和权重"""
    print(f"📁 Loading model: {model_name}")
    
    # 获取模型配置
    model_config = get_model_config(model_name)
    
    # 动态导入并创建模型
    if model_name == 'efficientnet_b0':
        from models.efficientnet import create_efficientnet_b0
        model = create_efficientnet_b0(num_classes=2)
    elif model_name == 'resnet18_improved':
        from models.resnet_improved import create_resnet18_improved
        model = create_resnet18_improved(num_classes=2)
    elif model_name == 'convnext_tiny':
        from models.convnext_tiny import create_convnext_tiny
        model = create_convnext_tiny(num_classes=2)
    elif model_name == 'coatnet':
        from models.coatnet import create_coatnet
        model = create_coatnet(variant='tiny', num_classes=2)
    elif model_name == 'vit_tiny':
        from models.vit_tiny import create_vit_tiny
        model = create_vit_tiny(num_classes=2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # 加载权重
    weights_path = os.path.join(experiment_path, 'best_model.pth')
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        
        # 处理不同的权重保存格式
        if 'model_state_dict' in checkpoint:
            # 完整的训练状态保存格式
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            # 包含训练信息的格式，但模型权重直接保存
            state_dict = {k: v for k, v in checkpoint.items() 
                         if k not in ['epoch', 'optimizer_state_dict', 'val_acc', 'val_loss']}
            model.load_state_dict(state_dict)
        else:
            # 直接的模型权重格式
            model.load_state_dict(checkpoint)
        
        print(f"✅ Loaded weights from: {weights_path}")
    else:
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    return model

def find_experiment_path(model_name, experiment_name=None):
    """查找实验路径"""
    experiments_dir = Path("experiments")
    
    if experiment_name:
        # 指定实验名称
        experiment_path = experiments_dir / experiment_name / model_name
        if experiment_path.exists():
            return str(experiment_path)
        else:
            raise FileNotFoundError(f"Experiment not found: {experiment_path}")
    else:
        # 查找最新的实验
        model_experiments = []
        for exp_dir in experiments_dir.glob("experiment_*"):
            model_path = exp_dir / model_name
            if model_path.exists() and (model_path / "best_model.pth").exists():
                model_experiments.append(model_path)
        
        if not model_experiments:
            raise FileNotFoundError(f"No completed experiments found for {model_name}")
        
        # 返回最新的实验
        latest_experiment = max(model_experiments, key=lambda x: x.stat().st_mtime)
        return str(latest_experiment)

def main():
    parser = argparse.ArgumentParser(description='Unified Model Evaluation Script (Fixed)')
    parser.add_argument('--model', 
                       choices=['efficientnet_b0', 'resnet18_improved', 'convnext_tiny', 'coatnet', 'vit_tiny'],
                       required=True,
                       help='Model name to evaluate')
    parser.add_argument('--experiment', 
                       help='Specific experiment name (optional)')
    parser.add_argument('--output', 
                       default='reports/individual',
                       help='Output directory for reports')
    parser.add_argument('--format', 
                       choices=['html', 'json', 'txt', 'all'],
                       default='all',
                       help='Report format')
    parser.add_argument('--include-samples', 
                       action='store_true',
                       help='Include sample predictions in report')
    parser.add_argument('--include-visualizations', 
                       action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    print("🔍 Unified Model Evaluation Script (Fixed)")
    print("=" * 50)
    
    try:
        # 查找实验路径
        experiment_path = find_experiment_path(args.model, args.experiment)
        print(f"📁 Found experiment: {experiment_path}")
        
        # 验证实验完整性
        required_files = ['best_model.pth', 'config.json']
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(experiment_path, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Evaluation failed: Missing required files in {experiment_path}: {missing_files}")
            return
        
        print("✅ Experiment validation passed")
        
        if args.dry_run:
            print("🔍 Dry run mode - showing what would be done:")
            print(f"  - Load model: {args.model}")
            print(f"  - Load weights from: {experiment_path}")
            print(f"  - Evaluate on test dataset")
            print(f"  - Generate reports in: {args.output}/{args.model}")
            return
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Using device: {device}")
        
        # 加载模型
        model = load_model_and_weights(args.model, experiment_path, device)
        
        # 创建数据加载器
        print("📂 Loading datasets...")
        data_loaders = create_data_loaders(
            DATA_DIR,
            batch_size=32,
            num_workers=2
        )
        
        # 设置输出路径
        output_dir = Path(args.output) / args.model
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📊 Output path: {output_dir}")
        
        # 创建评估器
        evaluator = ModelEvaluator(model, device)
        
        # 运行评估
        print("🔍 Starting evaluation...")
        results = evaluator.evaluate(
            data_loaders['test'], 
            save_dir=str(output_dir) if args.include_visualizations else None
        )
        
        # 保存结果
        results_file = output_dir / 'evaluation_results.json'
        
        # 准备可序列化的结果
        serializable_results = {}
        for key, value in results.items():
            if key == 'confusion_matrix':
                serializable_results[key] = value.tolist()
            elif key in ['precision_per_class', 'recall_per_class', 'f1_per_class']:
                serializable_results[key] = value.tolist()
            elif isinstance(value, (int, float, str)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        if 'html' in args.format or args.format == 'all':
            generate_html_report(results, args.model, output_dir)
        
        if 'txt' in args.format or args.format == 'all':
            generate_text_report(results, args.model, output_dir)
        
        print("✅ Evaluation completed successfully!")
        print(f"📁 Reports saved to: {output_dir}")
        
        # 打印主要结果
        print("\n📊 Key Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  Sensitivity: {results['sensitivity']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def generate_html_report(results, model_name, output_dir):
    """生成HTML报告"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{model_name.upper()} Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metric {{ margin: 10px 0; }}
            .metric-value {{ font-weight: bold; color: #2196F3; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{model_name.upper()} Model Evaluation Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Performance Metrics</h2>
        <div class="metric">Accuracy: <span class="metric-value">{results['accuracy']:.4f}</span></div>
        <div class="metric">Precision: <span class="metric-value">{results['precision']:.4f}</span></div>
        <div class="metric">Recall: <span class="metric-value">{results['recall']:.4f}</span></div>
        <div class="metric">F1-Score: <span class="metric-value">{results['f1_score']:.4f}</span></div>
        <div class="metric">AUC: <span class="metric-value">{results['auc']:.4f}</span></div>
        <div class="metric">Sensitivity: <span class="metric-value">{results['sensitivity']:.4f}</span></div>
        <div class="metric">Specificity: <span class="metric-value">{results['specificity']:.4f}</span></div>
        
        <h2>Confusion Matrix</h2>
        <table>
            <tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>
            <tr><th>Actual Negative</th><td>{results['confusion_matrix'][0][0]}</td><td>{results['confusion_matrix'][0][1]}</td></tr>
            <tr><th>Actual Positive</th><td>{results['confusion_matrix'][1][0]}</td><td>{results['confusion_matrix'][1][1]}</td></tr>
        </table>
        
        <h2>Classification Report</h2>
        <pre>{results['classification_report']}</pre>
    </body>
    </html>
    """
    
    with open(output_dir / 'evaluation_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def generate_text_report(results, model_name, output_dir):
    """生成文本报告"""
    report_content = f"""
{model_name.upper()} Model Evaluation Report
{'=' * 50}

Performance Metrics:
  Accuracy:    {results['accuracy']:.4f}
  Precision:   {results['precision']:.4f}
  Recall:      {results['recall']:.4f}
  F1-Score:    {results['f1_score']:.4f}
  AUC:         {results['auc']:.4f}
  Sensitivity: {results['sensitivity']:.4f}
  Specificity: {results['specificity']:.4f}

Confusion Matrix:
{results['confusion_matrix']}

Classification Report:
{results['classification_report']}
    """
    
    with open(output_dir / 'evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)

if __name__ == "__main__":
    main()