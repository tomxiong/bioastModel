#!/usr/bin/env python3
"""
生成完整性能对比报告
包括: 整体准确率、各任务准确率、总准确率
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def calculate_overall_accuracy(eval_data):
    """计算整体准确率"""

    # 各任务准确率
    growth_level_acc = eval_data['tasks']['growth_level']['accuracy']
    growth_pattern_acc = eval_data['tasks']['growth_pattern']['accuracy']

    # Interference factors 整体准确率 (所有因子的平均)
    interference_factors = eval_data['tasks']['interference_factors']
    interference_accs = []
    for factor in ['pores', 'artifacts', 'debris', 'contamination']:
        if factor in interference_factors:
            interference_accs.append(interference_factors[factor]['accuracy'])

    interference_overall_acc = sum(interference_accs) / len(interference_accs) if interference_accs else 0

    # 总准确率 (三个任务的平均)
    total_accuracy = (growth_level_acc + growth_pattern_acc + interference_overall_acc) / 3

    return {
        'growth_level_accuracy': growth_level_acc,
        'growth_pattern_accuracy': growth_pattern_acc,
        'interference_overall_accuracy': interference_overall_acc,
        'total_accuracy': total_accuracy
    }


def generate_report():
    print('='*80)
    print('完整性能对比报告生成')
    print('='*80)

    # 读取评估数据
    v10_path = 'experiments/multilevel_mobilenetv3_v0.10.0/comprehensive_evaluation.json'
    v11_path = 'experiments/multilevel_mobilenetv4_v0.11.0/comprehensive_evaluation.json'

    with open(v10_path, 'r') as f:
        v10_data = json.load(f)

    with open(v11_path, 'r') as f:
        v11_data = json.load(f)

    # 计算整体准确率
    v10_overall = calculate_overall_accuracy(v10_data)
    v11_overall = calculate_overall_accuracy(v11_data)

    # 生成对比报告
    print('\n' + '='*80)
    print('MobileNetV3 v0.10.0 vs MobileNetV4 v0.11.0 整体性能对比')
    print('='*80)

    print('\n【整体性能指标】')
    print(f"\n1. Growth Level Accuracy:")
    print(f"   MobileNetV3 v0.10.0: {v10_overall['growth_level_accuracy']:.4f}")
    print(f"   MobileNetV4 v0.11.0: {v11_overall['growth_level_accuracy']:.4f}")
    diff_level = v11_overall['growth_level_accuracy'] - v10_overall['growth_level_accuracy']
    print(f"   差距: {diff_level:+.4f} ({diff_level*100:+.2f}%)")

    print(f"\n2. Growth Pattern Accuracy:")
    print(f"   MobileNetV3 v0.10.0: {v10_overall['growth_pattern_accuracy']:.4f}")
    print(f"   MobileNetV4 v0.11.0: {v11_overall['growth_pattern_accuracy']:.4f}")
    diff_pattern = v11_overall['growth_pattern_accuracy'] - v10_overall['growth_pattern_accuracy']
    print(f"   差距: {diff_pattern:+.4f} ({diff_pattern*100:+.2f}%)")

    print(f"\n3. Interference Overall Accuracy:")
    print(f"   MobileNetV3 v0.10.0: {v10_overall['interference_overall_accuracy']:.4f}")
    print(f"   MobileNetV4 v0.11.0: {v11_overall['interference_overall_accuracy']:.4f}")
    diff_interference = v11_overall['interference_overall_accuracy'] - v10_overall['interference_overall_accuracy']
    print(f"   差距: {diff_interference:+.4f} ({diff_interference*100:+.2f}%)")

    print(f"\n4. Total Accuracy (三任务平均):")
    print(f"   MobileNetV3 v0.10.0: {v10_overall['total_accuracy']:.4f}")
    print(f"   MobileNetV4 v0.11.0: {v11_overall['total_accuracy']:.4f}")
    diff_total = v11_overall['total_accuracy'] - v10_overall['total_accuracy']
    print(f"   差距: {diff_total:+.4f} ({diff_total*100:+.2f}%)")

    # Interference 各因子详细
    print('\n【Interference Factors 各因子准确率】')
    v10_inter = v10_data['tasks']['interference_factors']
    v11_inter = v11_data['tasks']['interference_factors']

    for factor in ['pores', 'artifacts', 'debris', 'contamination']:
        if factor in v10_inter and factor in v11_inter:
            v10_acc = v10_inter[factor]['accuracy']
            v11_acc = v11_inter[factor]['accuracy']
            diff = v11_acc - v10_acc
            print(f"\n{factor.capitalize()}:")
            print(f"   MobileNetV3 v0.10.0: {v10_acc:.4f}")
            print(f"   MobileNetV4 v0.11.0: {v11_acc:.4f}")
            print(f"   差距: {diff:+.4f} ({diff*100:+.2f}%)")

    # Pores 详细性能
    print('\n【Pores 核心指标详细对比】')
    v10_pores = v10_inter['pores']
    v11_pores = v11_inter['pores']

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score'
    }

    for metric in metrics:
        v10_val = v10_pores[metric]
        v11_val = v11_pores[metric]
        diff = v11_val - v10_val
        print(f"\n{metric_names[metric]}:")
        print(f"   MobileNetV3 v0.10.0: {v10_val:.4f}")
        print(f"   MobileNetV4 v0.11.0: {v11_val:.4f}")
        print(f"   差距: {diff:+.4f} ({diff*100:+.2f}%)")

    # 混淆矩阵
    print('\n【Pores 混淆矩阵对比】')
    v10_cm = v10_pores['confusion_matrix']
    v11_cm = v11_pores['confusion_matrix']

    print(f"\nMobileNetV3 v0.10.0:")
    print(f"   TN: {v10_cm[0][0]:4d}  |  FP: {v10_cm[0][1]:4d}")
    print(f"   FN: {v10_cm[1][0]:4d}  |  TP: {v10_cm[1][1]:4d}")

    print(f"\nMobileNetV4 v0.11.0:")
    print(f"   TN: {v11_cm[0][0]:4d}  |  FP: {v11_cm[0][1]:4d}")
    print(f"   FN: {v11_cm[1][0]:4d}  |  TP: {v11_cm[1][1]:4d}")

    print(f"\n错误样本变化:")
    diff_fn = v11_cm[1][0] - v10_cm[1][0]
    diff_fp = v11_cm[0][1] - v10_cm[0][1]
    diff_total = abs(diff_fn) + abs(diff_fp)

    print(f"   FN (漏检): {v10_cm[1][0]} → {v11_cm[1][0]} ({diff_fn:+d}, {diff_fn/v10_cm[1][0]*100:+.1f}%)")
    print(f"   FP (误检): {v10_cm[0][1]} → {v11_cm[0][1]} ({diff_fp:+d}, {diff_fp/v10_cm[0][1]*100:+.1f}%)")
    print(f"   净误差变化: {diff_total:+d} 个")

    # 保存报告数据
    report_data = {
        'v0.10.0': v10_overall,
        'v0.11.0': v11_overall,
        'differences': {
            'growth_level_accuracy': diff_level,
            'growth_pattern_accuracy': diff_pattern,
            'interference_overall_accuracy': diff_interference,
            'total_accuracy': diff_total
        }
    }

    output_path = 'experiments/overall_performance_comparison.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print('\n' + '='*80)
    print(f'✅ 完整性能对比报告已保存到: {output_path}')
    print('='*80)

    return report_data


if __name__ == '__main__':
    generate_report()
