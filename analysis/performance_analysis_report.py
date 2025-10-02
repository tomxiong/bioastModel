#!/usr/bin/env python3
"""
性能分析报告生成脚本
分析三个模型的性能、复杂度和优化建议
"""

import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import sys
import os

# 添加项目根目录到路径
sys.path.append('/home/aaa/ws/bioastModel')

from models.multilevel_mobilenetv3 import create_multilevel_mobilenetv3
from models.optimized_multilevel_mobilenetv3 import create_optimized_multilevel_mobilenetv3
from models.simple_enhanced_multilevel_mobilenetv3 import create_simple_enhanced_multilevel_mobilenetv3

class ModelPerformanceAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_info = {}
        self.training_histories = {}
        
    def load_training_histories(self):
        """加载训练历史数据"""
        histories = {
            'original': '/home/aaa/ws/bioastModel/experiments/multilevel_mobilenetv3_ds/training_history.json',
            'optimized': '/home/aaa/ws/bioastModel/experiments/optimized_multilevel_mobilenetv3/optimized_training_history.json',
            'simple_enhanced': '/home/aaa/ws/bioastModel/experiments/simple_enhanced_multilevel_mobilenetv3/simple_enhanced_training_history.json'
        }
        
        for model_name, path in histories.items():
            try:
                with open(path, 'r') as f:
                    self.training_histories[model_name] = json.load(f)
                print(f"✅ 加载 {model_name} 训练历史成功")
            except FileNotFoundError:
                print(f"❌ 未找到 {model_name} 训练历史文件: {path}")
                self.training_histories[model_name] = None
    
    def analyze_model_complexity(self, model, model_name: str):
        """分析模型复杂度"""
        model.eval()
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算模型大小（MB）
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # 测试推理时间
        dummy_input = torch.randn(1, 1, 70, 70).to(self.device)
        model = model.to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # 测试推理时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        import time
        
        inference_times = []
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        
        complexity_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'params_millions': total_params / 1e6
        }
        
        self.models_info[model_name] = complexity_info
        return complexity_info
    
    def extract_performance_metrics(self):
        """提取性能指标"""
        performance_metrics = {}
        
        # 原始模型
        if self.training_histories['original']:
            hist = self.training_histories['original']
            final_accuracies = {
                'growth_level': hist['val_accuracy']['growth_level'][-1],
                'growth_pattern': hist['val_accuracy']['growth_pattern'][-1],
                'interference_factors': hist['val_accuracy']['interference_factors'][-1]
            }
            overall_accuracy = np.mean(list(final_accuracies.values()))
            
            performance_metrics['original'] = {
                'final_accuracies': final_accuracies,
                'overall_accuracy': overall_accuracy,
                'final_val_loss': hist['val_loss'][-1],
                'epochs_trained': len(hist['val_loss']),
                'best_val_loss': min(hist['val_loss']),
                'convergence_epoch': hist['val_loss'].index(min(hist['val_loss'])) + 1
            }
        
        # 优化版模型
        if self.training_histories['optimized']:
            hist = self.training_histories['optimized']
            final_accuracies = hist['val_accuracies'][-1]
            overall_accuracy = np.mean(list(final_accuracies.values()))
            
            performance_metrics['optimized'] = {
                'final_accuracies': final_accuracies,
                'overall_accuracy': overall_accuracy,
                'final_val_loss': hist['val_losses'][-1]['total'],
                'epochs_trained': len(hist['val_losses']),
                'best_val_loss': min([loss['total'] for loss in hist['val_losses']]),
                'convergence_epoch': [loss['total'] for loss in hist['val_losses']].index(min([loss['total'] for loss in hist['val_losses']])) + 1
            }
        
        # 简单增强版模型
        if self.training_histories['simple_enhanced']:
            hist = self.training_histories['simple_enhanced']
            # 计算最终准确率
            overall_accuracy = hist['val_accuracy'][-1]
            
            performance_metrics['simple_enhanced'] = {
                'overall_accuracy': overall_accuracy,
                'final_val_loss': hist['val_loss'][-1],
                'epochs_trained': len(hist['val_loss']),
                'best_val_loss': min(hist['val_loss']),
                'convergence_epoch': hist['val_loss'].index(min(hist['val_loss'])) + 1
            }
        
        return performance_metrics
    
    def generate_comparison_report(self):
        """生成对比报告"""
        print("\n" + "="*80)
        print("📊 模型性能分析报告")
        print("="*80)
        
        # 创建模型实例并分析复杂度
        models = {
            'original': create_multilevel_mobilenetv3(input_channels=1),
            'optimized': create_optimized_multilevel_mobilenetv3(input_channels=1),
            'simple_enhanced': create_simple_enhanced_multilevel_mobilenetv3(input_channels=1)
        }
        
        print("\n1. 模型复杂度分析")
        print("-" * 50)
        
        for model_name, model in models.items():
            print(f"\n🔍 分析 {model_name.upper()} 模型...")
            complexity = self.analyze_model_complexity(model, model_name)
            
            print(f"   参数总数: {complexity['total_params']:,}")
            print(f"   可训练参数: {complexity['trainable_params']:,}")
            print(f"   模型大小: {complexity['model_size_mb']:.2f} MB")
            print(f"   平均推理时间: {complexity['avg_inference_time_ms']:.2f} ± {complexity['std_inference_time_ms']:.2f} ms")
        
        # 性能指标分析
        performance_metrics = self.extract_performance_metrics()
        
        print("\n2. 性能指标对比")
        print("-" * 50)
        
        for model_name, metrics in performance_metrics.items():
            print(f"\n📈 {model_name.upper()} 模型:")
            print(f"   整体准确率: {metrics['overall_accuracy']:.4f}")
            print(f"   最终验证损失: {metrics['final_val_loss']:.4f}")
            print(f"   训练轮数: {metrics['epochs_trained']}")
            print(f"   最佳验证损失: {metrics['best_val_loss']:.4f}")
            print(f"   收敛轮数: {metrics['convergence_epoch']}")
            
            if 'final_accuracies' in metrics:
                print("   各任务准确率:")
                for task, acc in metrics['final_accuracies'].items():
                    print(f"     - {task}: {acc:.4f}")
        
        # 效率分析
        print("\n3. 效率分析")
        print("-" * 50)
        
        efficiency_scores = {}
        for model_name in self.models_info.keys():
            if model_name in performance_metrics:
                complexity = self.models_info[model_name]
                performance = performance_metrics[model_name]
                
                # 计算效率分数 (准确率 / 参数量(百万))
                efficiency_score = performance['overall_accuracy'] / complexity['params_millions']
                efficiency_scores[model_name] = efficiency_score
                
                print(f"\n⚡ {model_name.upper()}:")
                print(f"   效率分数: {efficiency_score:.2f}")
                print(f"   准确率/参数比: {performance['overall_accuracy']:.4f} / {complexity['params_millions']:.2f}M")
        
        # 推荐最佳模型
        print("\n4. 模型推荐")
        print("-" * 50)
        
        if efficiency_scores:
            best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
            best_accuracy = max(performance_metrics.items(), key=lambda x: x[1]['overall_accuracy'])
            
            print(f"\n🏆 最高效率模型: {best_efficiency[0].upper()}")
            print(f"   效率分数: {best_efficiency[1]:.2f}")
            
            print(f"\n🎯 最高准确率模型: {best_accuracy[0].upper()}")
            print(f"   准确率: {best_accuracy[1]['overall_accuracy']:.4f}")
        
        return {
            'complexity': self.models_info,
            'performance': performance_metrics,
            'efficiency': efficiency_scores
        }
    
    def generate_optimization_recommendations(self, analysis_results):
        """生成优化建议"""
        print("\n5. 优化建议")
        print("-" * 50)
        
        recommendations = []
        
        # 基于分析结果的建议
        complexity = analysis_results['complexity']
        performance = analysis_results['performance']
        
        # 1. 模型选择建议
        print("\n📋 模型选择建议:")
        
        if 'simple_enhanced' in performance and performance['simple_enhanced']['overall_accuracy'] > 0.9:
            print("✅ 推荐使用 Simple Enhanced 模型:")
            print("   - 准确率最高且稳定")
            print("   - 训练过程收敛良好")
            print("   - 适合生产环境部署")
            recommendations.append("使用Simple Enhanced模型作为主要方案")
        
        # 2. 训练策略建议
        print("\n🎯 训练策略建议:")
        
        if 'optimized' in performance:
            opt_perf = performance['optimized']
            if opt_perf['overall_accuracy'] < 0.7:
                print("⚠️  复杂优化策略问题:")
                print("   - 避免过度复杂的损失函数组合")
                print("   - 简化自适应权重机制")
                print("   - 使用渐进式优化方法")
                recommendations.append("避免过度复杂的优化策略")
        
        # 3. 架构优化建议
        print("\n🏗️  架构优化建议:")
        
        # 分析参数效率
        param_efficiency = {}
        for model_name in complexity.keys():
            if model_name in performance:
                acc = performance[model_name]['overall_accuracy']
                params = complexity[model_name]['params_millions']
                param_efficiency[model_name] = acc / params
        
        if param_efficiency:
            most_efficient = max(param_efficiency.items(), key=lambda x: x[1])
            print(f"✨ 最优参数效率: {most_efficient[0].upper()}")
            print("   建议采用类似的轻量化设计")
            recommendations.append(f"参考{most_efficient[0]}的轻量化设计")
        
        # 4. 部署建议
        print("\n🚀 部署建议:")
        
        # 推理时间分析
        inference_times = {name: info['avg_inference_time_ms'] 
                          for name, info in complexity.items()}
        
        if inference_times:
            fastest_model = min(inference_times.items(), key=lambda x: x[1])
            print(f"⚡ 最快推理: {fastest_model[0].upper()} ({fastest_model[1]:.2f}ms)")
            
            for model_name, time_ms in inference_times.items():
                if time_ms < 5.0:
                    print(f"   ✅ {model_name}: 适合实时应用")
                elif time_ms < 10.0:
                    print(f"   ⚠️  {model_name}: 适合准实时应用")
                else:
                    print(f"   ❌ {model_name}: 仅适合离线处理")
        
        # 5. 进一步优化方向
        print("\n🔮 进一步优化方向:")
        print("   1. 知识蒸馏: 用大模型指导小模型训练")
        print("   2. 模型剪枝: 移除不重要的连接和神经元")
        print("   3. 量化优化: INT8量化减少模型大小")
        print("   4. 架构搜索: 自动寻找最优网络结构")
        print("   5. 多尺度训练: 提高模型泛化能力")
        
        return recommendations
    
    def save_analysis_report(self, analysis_results, recommendations):
        """保存分析报告"""
        report_dir = Path('/home/aaa/ws/bioastModel/analysis/performance_reports')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细数据
        with open(report_dir / 'detailed_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        # 保存推荐建议
        with open(report_dir / 'recommendations.txt', 'w', encoding='utf-8') as f:
            f.write("模型优化建议\n")
            f.write("="*50 + "\n\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"\n📄 分析报告已保存到: {report_dir}")
        
    def run_complete_analysis(self):
        """运行完整分析"""
        print("🚀 开始性能分析...")
        
        # 加载训练历史
        self.load_training_histories()
        
        # 生成对比报告
        analysis_results = self.generate_comparison_report()
        
        # 生成优化建议
        recommendations = self.generate_optimization_recommendations(analysis_results)
        
        # 保存报告
        self.save_analysis_report(analysis_results, recommendations)
        
        print("\n✅ 性能分析完成!")
        return analysis_results, recommendations

if __name__ == "__main__":
    analyzer = ModelPerformanceAnalyzer()
    results, recommendations = analyzer.run_complete_analysis()