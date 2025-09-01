#!/usr/bin/env python3
"""
ONNX模型性能分析总结报告
基于完整的性能分析结果
"""

import json
from datetime import datetime

def load_analysis_results():
    """加载分析结果"""
    try:
        with open('onnx_performance_analysis.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"无法加载分析结果: {e}")
        return None

def generate_summary_report():
    """生成总结报告"""
    data = load_analysis_results()
    if not data:
        return
    
    results = data['detailed_results']
    successful_results = [r for r in results if r.get('load_success', False)]
    
    print("="*80)
    print("📊 ONNX模型性能分析总结报告")
    print("="*80)
    print(f"分析时间: {data['analysis_timestamp']}")
    print(f"总模型数: {data['total_models']}")
    print(f"成功分析: {data['successful_models']}")
    print(f"分析失败: {data['failed_models']}")
    
    # 创建性能对比表
    print("\n📋 详细性能对比:")
    print("-" * 120)
    print(f"{'模型类型':<25} {'文件大小(MB)':<12} {'推理时间(ms)':<12} {'速度(FPS)':<12} {'准确率':<10}")
    print("-" * 120)
    
    for result in successful_results:
        model_type = result['model_type']
        file_size = result['file_size_mb']
        speed = result['speed_results']
        accuracy = result['accuracy_results']
        
        inference_time = speed['avg_inference_time_ms']
        fps = speed['fps']
        acc = accuracy['accuracy'] if accuracy.get('success', True) else 0
        
        print(f"{model_type:<25} {file_size:<12.2f} {inference_time:<12.2f} {fps:<12.1f} {acc:<10.4f}")
    
    # 性能排名
    print("\n🏆 性能排名分析:")
    
    # 1. 推理速度排名
    print("\n⚡ 推理速度排名 (FPS):")
    speed_sorted = sorted(successful_results, 
                         key=lambda x: x['speed_results']['fps'], 
                         reverse=True)
    for i, result in enumerate(speed_sorted[:10], 1):
        fps = result['speed_results']['fps']
        print(f"{i:2d}. {result['model_type']:<25} {fps:>8.1f} FPS")
    
    # 2. 准确率排名
    print("\n🎯 准确率排名:")
    accuracy_sorted = sorted(successful_results, 
                           key=lambda x: x['accuracy_results']['accuracy'], 
                           reverse=True)
    for i, result in enumerate(accuracy_sorted[:10], 1):
        acc = result['accuracy_results']['accuracy']
        print(f"{i:2d}. {result['model_type']:<25} {acc:>8.4f}")
    
    # 3. 文件大小排名 (最小优先)
    print("\n💾 文件大小排名 (最小优先):")
    size_sorted = sorted(successful_results, 
                        key=lambda x: x['file_size_mb'])
    for i, result in enumerate(size_sorted[:10], 1):
        size = result['file_size_mb']
        print(f"{i:2d}. {result['model_type']:<25} {size:>8.2f} MB")
    
    # 4. 综合评分
    print("\n🏅 综合评分 (速度权重0.3 + 准确率权重0.6 + 大小惩罚0.1):")
    scored_results = []
    
    # 归一化因子
    max_fps = max(r['speed_results']['fps'] for r in successful_results)
    max_acc = max(r['accuracy_results']['accuracy'] for r in successful_results)
    max_size = max(r['file_size_mb'] for r in successful_results)
    
    for result in successful_results:
        speed_score = (result['speed_results']['fps'] / max_fps) * 0.3
        accuracy_score = (result['accuracy_results']['accuracy'] / max_acc) * 0.6
        size_penalty = (result['file_size_mb'] / max_size) * 0.1
        
        composite_score = speed_score + accuracy_score - size_penalty
        scored_results.append((result, composite_score))
    
    scored_results.sort(key=lambda x: x[1], reverse=True)
    for i, (result, score) in enumerate(scored_results[:10], 1):
        print(f"{i:2d}. {result['model_type']:<25} {score:>8.4f}")
    
    # 最佳模型推荐
    print("\n🌟 最佳模型推荐:")
    
    best_overall = scored_results[0][0]
    best_speed = speed_sorted[0]
    best_accuracy = accuracy_sorted[0]
    best_size = size_sorted[0]
    
    print(f"\n🏆 综合最佳: {best_overall['model_type']}")
    print(f"   - 文件大小: {best_overall['file_size_mb']:.2f} MB")
    print(f"   - 推理速度: {best_overall['speed_results']['fps']:.1f} FPS")
    print(f"   - 准确率: {best_overall['accuracy_results']['accuracy']:.4f}")
    
    print(f"\n⚡ 速度最快: {best_speed['model_type']}")
    print(f"   - 推理速度: {best_speed['speed_results']['fps']:.1f} FPS")
    print(f"   - 推理时间: {best_speed['speed_results']['avg_inference_time_ms']:.2f} ms")
    
    print(f"\n🎯 准确率最高: {best_accuracy['model_type']}")
    print(f"   - 准确率: {best_accuracy['accuracy_results']['accuracy']:.4f}")
    print(f"   - 推理速度: {best_accuracy['speed_results']['fps']:.1f} FPS")
    
    print(f"\n💾 体积最小: {best_size['model_type']}")
    print(f"   - 文件大小: {best_size['file_size_mb']:.2f} MB")
    print(f"   - 推理速度: {best_size['speed_results']['fps']:.1f} FPS")
    
    # 应用场景推荐
    print("\n🎯 应用场景推荐:")
    
    print("\n📱 移动端部署 (优先考虑体积和速度):")
    mobile_candidates = [r for r in successful_results if r['file_size_mb'] < 5.0]
    if mobile_candidates:
        mobile_best = max(mobile_candidates, key=lambda x: x['speed_results']['fps'])
        print(f"   推荐: {mobile_best['model_type']}")
        print(f"   - 文件大小: {mobile_best['file_size_mb']:.2f} MB")
        print(f"   - 推理速度: {mobile_best['speed_results']['fps']:.1f} FPS")
        print(f"   - 准确率: {mobile_best['accuracy_results']['accuracy']:.4f}")
    
    print("\n🏥 医疗诊断 (优先考虑准确率):")
    medical_candidates = [r for r in successful_results if r['accuracy_results']['accuracy'] > 0.95]
    if medical_candidates:
        medical_best = max(medical_candidates, key=lambda x: x['accuracy_results']['accuracy'])
        print(f"   推荐: {medical_best['model_type']}")
        print(f"   - 准确率: {medical_best['accuracy_results']['accuracy']:.4f}")
        print(f"   - 推理速度: {medical_best['speed_results']['fps']:.1f} FPS")
        print(f"   - 文件大小: {medical_best['file_size_mb']:.2f} MB")
    
    print("\n⚡ 实时处理 (优先考虑速度):")
    realtime_candidates = [r for r in successful_results if r['speed_results']['fps'] > 1000]
    if realtime_candidates:
        realtime_best = max(realtime_candidates, key=lambda x: x['accuracy_results']['accuracy'])
        print(f"   推荐: {realtime_best['model_type']}")
        print(f"   - 推理速度: {realtime_best['speed_results']['fps']:.1f} FPS")
        print(f"   - 准确率: {realtime_best['accuracy_results']['accuracy']:.4f}")
        print(f"   - 文件大小: {realtime_best['file_size_mb']:.2f} MB")
    
    # 性能分析洞察
    print("\n💡 性能分析洞察:")
    
    # 速度vs准确率分析
    high_speed_models = [r for r in successful_results if r['speed_results']['fps'] > 1000]
    high_accuracy_models = [r for r in successful_results if r['accuracy_results']['accuracy'] > 0.95]
    
    print(f"\n📈 高速模型 (>1000 FPS): {len(high_speed_models)} 个")
    print(f"📈 高精度模型 (>95%): {len(high_accuracy_models)} 个")
    
    # 找出速度和准确率都高的模型
    balanced_models = [r for r in successful_results 
                      if r['speed_results']['fps'] > 1000 and r['accuracy_results']['accuracy'] > 0.95]
    
    if balanced_models:
        print(f"🎯 速度和准确率都优秀的模型: {len(balanced_models)} 个")
        for model in balanced_models:
            print(f"   - {model['model_type']}: {model['speed_results']['fps']:.1f} FPS, {model['accuracy_results']['accuracy']:.4f} 准确率")
    
    # 文件大小分析
    small_models = [r for r in successful_results if r['file_size_mb'] < 1.0]
    medium_models = [r for r in successful_results if 1.0 <= r['file_size_mb'] < 20.0]
    large_models = [r for r in successful_results if r['file_size_mb'] >= 20.0]
    
    print(f"\n📊 模型大小分布:")
    print(f"   小型模型 (<1MB): {len(small_models)} 个")
    print(f"   中型模型 (1-20MB): {len(medium_models)} 个")
    print(f"   大型模型 (≥20MB): {len(large_models)} 个")
    
    print("\n" + "="*80)
    print("📄 报告生成完成!")
    print("="*80)

if __name__ == "__main__":
    generate_summary_report()