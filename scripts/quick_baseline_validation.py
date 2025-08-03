"""
快速基线性能验证脚本
获取当前最佳模型的基线性能数据
"""

import json
import os
from datetime import datetime

def analyze_existing_results():
    """分析现有实验结果，获取基线性能"""
    print("🔍 分析现有实验结果...")
    
    experiments_dir = "experiments"
    results = []
    
    # 遍历所有实验目录
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
            
        # 查找模型子目录
        for model_dir in os.listdir(exp_path):
            model_path = os.path.join(exp_path, model_dir)
            if not os.path.isdir(model_path):
                continue
                
            # 读取测试结果
            test_results_path = os.path.join(model_path, "test_results.json")
            if os.path.exists(test_results_path):
                try:
                    with open(test_results_path, 'r', encoding='utf-8') as f:
                        test_data = json.load(f)
                    
                    result = {
                        'experiment': exp_dir,
                        'model': model_dir,
                        'accuracy': test_data.get('accuracy', 0) * 100,
                        'precision': test_data.get('precision', 0) * 100,
                        'recall': test_data.get('recall', 0) * 100,
                        'f1_score': test_data.get('f1_score', 0) * 100,
                        'path': model_path
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"⚠️ 读取 {test_results_path} 失败: {e}")
    
    return results

def generate_baseline_report(results):
    """生成基线性能报告"""
    if not results:
        print("❌ 未找到任何实验结果")
        return
    
    # 按准确率排序
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n📊 当前模型性能基线分析")
    print("=" * 60)
    
    # 显示最佳模型
    best_model = results[0]
    print(f"🏆 最佳模型: {best_model['model']}")
    print(f"   实验: {best_model['experiment']}")
    print(f"   准确率: {best_model['accuracy']:.2f}%")
    print(f"   精确率: {best_model['precision']:.2f}%")
    print(f"   召回率: {best_model['recall']:.2f}%")
    print(f"   F1分数: {best_model['f1_score']:.2f}%")
    
    # 计算假阴性率（估算）
    false_negative_rate = 100 - best_model['recall']
    print(f"   假阴性率: {false_negative_rate:.2f}%")
    
    print(f"\n📈 所有模型性能对比:")
    print("-" * 60)
    for i, result in enumerate(results[:8], 1):  # 显示前8个
        print(f"{i:2d}. {result['model']:<20} | 准确率: {result['accuracy']:6.2f}% | F1: {result['f1_score']:6.2f}%")
    
    # 气孔检测器有效性评估
    print(f"\n🎯 气孔检测器有效性评估:")
    print("-" * 60)
    
    # 设定目标
    target_accuracy = 92.0
    target_precision = 90.0
    target_recall = 88.0
    target_f1 = 89.0
    max_false_negative_rate = 12.0
    
    # 评估最佳模型
    accuracy_gap = target_accuracy - best_model['accuracy']
    precision_gap = target_precision - best_model['precision']
    recall_gap = target_recall - best_model['recall']
    f1_gap = target_f1 - best_model['f1_score']
    fn_rate_gap = false_negative_rate - max_false_negative_rate
    
    print(f"当前最佳模型 vs 目标:")
    print(f"  准确率: {best_model['accuracy']:6.2f}% vs {target_accuracy:6.2f}% (差距: {accuracy_gap:+6.2f}%)")
    print(f"  精确率: {best_model['precision']:6.2f}% vs {target_precision:6.2f}% (差距: {precision_gap:+6.2f}%)")
    print(f"  召回率: {best_model['recall']:6.2f}% vs {target_recall:6.2f}% (差距: {recall_gap:+6.2f}%)")
    print(f"  F1分数: {best_model['f1_score']:6.2f}% vs {target_f1:6.2f}% (差距: {f1_gap:+6.2f}%)")
    print(f"  假阴性率: {false_negative_rate:6.2f}% vs {max_false_negative_rate:6.2f}% (差距: {fn_rate_gap:+6.2f}%)")
    
    # 有效性判断
    criteria_met = 0
    total_criteria = 5
    
    if best_model['accuracy'] >= target_accuracy:
        criteria_met += 1
        print(f"  ✅ 准确率达标")
    else:
        print(f"  ❌ 准确率未达标 (需提升 {abs(accuracy_gap):.2f}%)")
    
    if best_model['precision'] >= target_precision:
        criteria_met += 1
        print(f"  ✅ 精确率达标")
    else:
        print(f"  ❌ 精确率未达标 (需提升 {abs(precision_gap):.2f}%)")
    
    if best_model['recall'] >= target_recall:
        criteria_met += 1
        print(f"  ✅ 召回率达标")
    else:
        print(f"  ❌ 召回率未达标 (需提升 {abs(recall_gap):.2f}%)")
    
    if best_model['f1_score'] >= target_f1:
        criteria_met += 1
        print(f"  ✅ F1分数达标")
    else:
        print(f"  ❌ F1分数未达标 (需提升 {abs(f1_gap):.2f}%)")
    
    if false_negative_rate <= max_false_negative_rate:
        criteria_met += 1
        print(f"  ✅ 假阴性率达标")
    else:
        print(f"  ❌ 假阴性率过高 (需降低 {abs(fn_rate_gap):.2f}%)")
    
    # 综合评估
    effectiveness_score = (criteria_met / total_criteria) * 100
    
    print(f"\n🎯 综合有效性评估:")
    print(f"  达标指标: {criteria_met}/{total_criteria}")
    print(f"  有效性分数: {effectiveness_score:.1f}/100")
    
    if effectiveness_score >= 80:
        status = "🟢 基本有效"
        recommendation = "模型表现良好，可进行微调优化"
    elif effectiveness_score >= 60:
        status = "🟡 部分有效"
        recommendation = "需要针对性改进，重点提升未达标指标"
    else:
        status = "🔴 需要改进"
        recommendation = "建议重新训练或调整架构"
    
    print(f"  有效性状态: {status}")
    print(f"  建议: {recommendation}")
    
    # 下一步行动计划
    print(f"\n🚀 下一步行动计划:")
    print("-" * 60)
    
    if accuracy_gap > 0:
        print(f"1. 🎯 提升准确率: 当前 {best_model['accuracy']:.2f}% → 目标 {target_accuracy:.2f}%")
        print(f"   - 使用增强型气孔检测器架构")
        print(f"   - 改进数据增强策略")
    
    if fn_rate_gap > 0:
        print(f"2. ⚠️ 降低假阴性率: 当前 {false_negative_rate:.2f}% → 目标 ≤{max_false_negative_rate:.2f}%")
        print(f"   - 优化假阴性控制机制")
        print(f"   - 调整分类阈值")
    
    if recall_gap > 0:
        print(f"3. 📈 提升召回率: 当前 {best_model['recall']:.2f}% → 目标 {target_recall:.2f}%")
        print(f"   - 增强浊度分类能力")
        print(f"   - 改进小气孔检测敏感性")
    
    print(f"\n✅ 基线验证完成！最佳模型路径: {best_model['path']}")
    
    return best_model

def main():
    """主函数"""
    print("🚀 快速基线性能验证")
    print("=" * 50)
    
    # 分析现有结果
    results = analyze_existing_results()
    
    if results:
        # 生成基线报告
        best_model = generate_baseline_report(results)
        
        # 保存基线数据
        baseline_data = {
            'timestamp': datetime.now().isoformat(),
            'best_model': best_model,
            'all_results': results[:10]  # 保存前10个结果
        }
        
        os.makedirs('experiments/baseline_validation', exist_ok=True)
        with open('experiments/baseline_validation/baseline_performance.json', 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 基线数据已保存: experiments/baseline_validation/baseline_performance.json")
    else:
        print("❌ 未找到任何实验结果，请先运行模型训练")

if __name__ == "__main__":
    main()