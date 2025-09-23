#!/usr/bin/env python3
"""
批量转换多任务模型为ONNX格式
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import subprocess
import time
from pathlib import Path
import glob

def find_trained_models():
    """查找已训练的模型"""
    models = []
    
    # 查找已完成的实验
    experiment_dirs = [
        "experiments/fixed_efficientnet_b0_multitask_*",
        "experiments/resnet34_gpu_optimized_*", 
        "experiments/fixed_mobilenetv3_multitask_*"
    ]
    
    for pattern in experiment_dirs:
        for exp_dir in glob.glob(pattern):
            best_checkpoint = os.path.join(exp_dir, 'best.pth')
            if os.path.exists(best_checkpoint):
                # 从目录名推断模型类型
                if 'efficientnet_b0' in exp_dir:
                    model_type = 'fixed_efficientnet_b0'
                elif 'resnet34' in exp_dir:
                    model_type = 'resnet34'
                elif 'mobilenetv3' in exp_dir:
                    model_type = 'fixed_mobilenetv3'
                else:
                    continue
                
                models.append({
                    'model_type': model_type,
                    'checkpoint_path': best_checkpoint,
                    'experiment_dir': exp_dir
                })
    
    return models

def convert_model(model_info, output_dir):
    """转换单个模型"""
    print(f"\n{'='*60}")
    print(f"转换模型: {model_info['model_type']}")
    print(f"检查点: {model_info['checkpoint_path']}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 
        'scripts/convert_to_onnx_multitask.py',
        '--model', model_info['model_type'],
        '--checkpoint', model_info['checkpoint_path'],
        '--output_dir', output_dir,
        '--validate',
        '--benchmark'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 转换成功!")
            return True, result.stdout
        else:
            print("❌ 转换失败!")
            print(f"错误: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("❌ 转换超时!")
        return False, "转换超时"
    except Exception as e:
        print(f"❌ 转换异常: {e}")
        return False, str(e)

def load_training_history(experiment_dir):
    """加载训练历史"""
    history_file = os.path.join(experiment_dir, 'train_history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return None

def generate_summary_report(conversion_results, output_dir):
    """生成汇总报告"""
    report = {
        'conversion_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_models': len(conversion_results),
        'successful_conversions': sum(1 for r in conversion_results if r['success']),
        'failed_conversions': sum(1 for r in conversion_results if not r['success']),
        'models': []
    }
    
    for result in conversion_results:
        model_info = {
            'model_type': result['model_type'],
            'experiment_dir': result['experiment_dir'],
            'checkpoint_path': result['checkpoint_path'],
            'conversion_success': result['success'],
            'conversion_message': result['message'][:200] if result['message'] else None,
        }
        
        # 加载训练历史
        history = load_training_history(result['experiment_dir'])
        if history:
            best_accuracy = max(history.get('val_accuracy', [0]))
            model_info['best_validation_accuracy'] = best_accuracy
            model_info['training_epochs'] = len(history.get('val_accuracy', []))
        
        # 如果转换成功，加载转换报告
        if result['success']:
            report_path = os.path.join(output_dir, f"{result['model_type']}_conversion_report.json")
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    conversion_report = json.load(f)
                model_info['onnx_path'] = conversion_report.get('onnx_path')
                model_info['onnx_file_size_mb'] = conversion_report.get('model_info', {}).get('file_size_mb')
                model_info['validation_passed'] = conversion_report.get('validation_passed')
                
                benchmark = conversion_report.get('benchmark')
                if benchmark:
                    model_info['inference_pytorch_ms'] = benchmark.get('pytorch_time_ms')
                    model_info['inference_onnx_ms'] = benchmark.get('onnx_time_ms')
                    model_info['speedup'] = benchmark.get('speedup')
        
        report['models'].append(model_info)
    
    # 保存汇总报告
    summary_path = os.path.join(output_dir, 'conversion_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("转换汇总")
    print(f"{'='*60}")
    print(f"总模型数: {report['total_models']}")
    print(f"成功转换: {report['successful_conversions']}")
    print(f"转换失败: {report['failed_conversions']}")
    
    print(f"\n模型详情:")
    for model in report['models']:
        status = "✅" if model['conversion_success'] else "❌"
        acc = model.get('best_validation_accuracy', 0)
        size = model.get('onnx_file_size_mb', 0)
        speedup = model.get('speedup', 0)
        
        print(f"  {status} {model['model_type']}")
        print(f"     准确率: {acc:.2f}%")
        if model['conversion_success']:
            print(f"     ONNX大小: {size:.1f} MB")
            print(f"     加速比: {speedup:.2f}x")
    
    print(f"\n汇总报告已保存: {summary_path}")
    return summary_path

def main():
    print("🚀 批量转换多任务模型为ONNX格式")
    print("="*60)
    
    # 查找已训练的模型
    models = find_trained_models()
    
    if not models:
        print("❌ 未找到已训练的模型!")
        return
    
    print(f"发现 {len(models)} 个已训练的模型:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['model_type']} - {model['checkpoint_path']}")
    
    # 创建输出目录
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换所有模型
    conversion_results = []
    
    for model in models:
        success, message = convert_model(model, output_dir)
        
        conversion_results.append({
            'model_type': model['model_type'],
            'experiment_dir': model['experiment_dir'],
            'checkpoint_path': model['checkpoint_path'],
            'success': success,
            'message': message
        })
        
        # 短暂等待
        time.sleep(1)
    
    # 生成汇总报告
    summary_path = generate_summary_report(conversion_results, output_dir)
    
    print(f"\n🎉 批量转换完成!")
    print(f"ONNX模型保存位置: {output_dir}")
    print(f"汇总报告: {summary_path}")

if __name__ == "__main__":
    main()