#!/usr/bin/env python3
"""
对比分析 EfficientNet 和 EfficientNet V2 的区别
分析参数数量差异和架构改进
"""

import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.efficientnet import EfficientNetCustom
from models.efficientnet_v2 import EfficientNetV2, create_efficientnetv2_s

def analyze_efficientnet_architecture():
    """分析 EfficientNet 架构"""
    print("🔍 分析 EfficientNet 架构...")
    
    model = EfficientNetCustom(num_classes=2)
    
    # 计算参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 分析层结构
    layer_analysis = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                layer_analysis[name] = {
                    'type': type(module).__name__,
                    'parameters': param_count
                }
    
    # 测试输入输出
    test_input = torch.randn(1, 3, 70, 70)
    with torch.no_grad():
        output = model(test_input)
    
    analysis = {
        'model_name': 'EfficientNet',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_shape': list(test_input.shape),
        'output_shape': list(output.shape),
        'layer_count': len(layer_analysis),
        'key_components': {
            'backbone': 'MobileNet-inspired blocks with squeeze-and-excitation',
            'scaling_method': 'Compound scaling (depth, width, resolution)',
            'activation': 'Swish activation function',
            'normalization': 'Batch Normalization',
            'attention': 'Squeeze-and-Excitation blocks'
        },
        'architecture_features': [
            'Mobile Inverted Bottleneck Convolution (MBConv)',
            'Squeeze-and-Excitation attention',
            'Compound scaling strategy',
            'Neural Architecture Search (NAS) optimized',
            'Efficient parameter usage'
        ]
    }
    
    print(f"✅ EfficientNet 分析完成:")
    print(f"   总参数: {total_params:,} ({total_params/1000000:.1f}M)")
    print(f"   层数: {len(layer_analysis)}")
    print(f"   输入: {test_input.shape} -> 输出: {output.shape}")
    
    return analysis

def analyze_efficientnet_v2_architecture():
    """分析 EfficientNet V2 架构"""
    print("🔍 分析 EfficientNet V2 架构...")
    
    model = create_efficientnetv2_s(num_classes=2)
    
    # 计算参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 分析层结构
    layer_analysis = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                layer_analysis[name] = {
                    'type': type(module).__name__,
                    'parameters': param_count
                }
    
    # 测试输入输出
    test_input = torch.randn(1, 3, 70, 70)
    with torch.no_grad():
        output = model(test_input)
    
    analysis = {
        'model_name': 'EfficientNet V2',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_shape': list(test_input.shape),
        'output_shape': list(output.shape),
        'layer_count': len(layer_analysis),
        'key_components': {
            'backbone': 'Fused-MBConv + MBConv blocks',
            'scaling_method': 'Progressive learning with adaptive regularization',
            'activation': 'SiLU (Swish) activation function',
            'normalization': 'Batch Normalization',
            'attention': 'Squeeze-and-Excitation blocks'
        },
        'architecture_features': [
            'Fused-MBConv blocks (early stages)',
            'Traditional MBConv blocks (later stages)',
            'Progressive learning strategy',
            'Adaptive regularization (dropout, data augmentation)',
            'Improved training efficiency',
            'Better parameter efficiency vs accuracy trade-off'
        ]
    }
    
    print(f"✅ EfficientNet V2 分析完成:")
    print(f"   总参数: {total_params:,} ({total_params/1000000:.1f}M)")
    print(f"   层数: {len(layer_analysis)}")
    print(f"   输入: {test_input.shape} -> 输出: {output.shape}")
    
    return analysis

def compare_architectures(efficientnet_analysis, efficientnet_v2_analysis):
    """对比两个架构的差异"""
    print("\n" + "="*80)
    print("📊 EfficientNet vs EfficientNet V2 详细对比")
    print("="*80)
    
    comparison = {
        'comparison_timestamp': datetime.now().isoformat(),
        'parameter_comparison': {
            'efficientnet_params': efficientnet_analysis['total_parameters'],
            'efficientnet_v2_params': efficientnet_v2_analysis['total_parameters'],
            'parameter_ratio': efficientnet_v2_analysis['total_parameters'] / efficientnet_analysis['total_parameters'],
            'parameter_increase': efficientnet_v2_analysis['total_parameters'] - efficientnet_analysis['total_parameters']
        },
        'architectural_differences': {},
        'key_improvements': [],
        'trade_offs': []
    }
    
    # 参数对比
    param_ratio = comparison['parameter_comparison']['parameter_ratio']
    param_increase = comparison['parameter_comparison']['parameter_increase']
    
    print(f"📊 参数数量对比:")
    print(f"   EfficientNet:    {efficientnet_analysis['total_parameters']:,} ({efficientnet_analysis['total_parameters']/1000000:.1f}M)")
    print(f"   EfficientNet V2: {efficientnet_v2_analysis['total_parameters']:,} ({efficientnet_v2_analysis['total_parameters']/1000000:.1f}M)")
    print(f"   增加倍数: {param_ratio:.1f}x")
    print(f"   增加数量: {param_increase:,} ({param_increase/1000000:.1f}M)")
    
    # 架构差异分析
    print(f"\n🏗️ 架构差异分析:")
    
    differences = {
        'block_types': {
            'efficientnet': 'MBConv (Mobile Inverted Bottleneck)',
            'efficientnet_v2': 'Fused-MBConv + MBConv 混合'
        },
        'training_strategy': {
            'efficientnet': '固定训练策略',
            'efficientnet_v2': '渐进式学习 + 自适应正则化'
        },
        'optimization_focus': {
            'efficientnet': '模型大小和准确率平衡',
            'efficientnet_v2': '训练效率和推理速度优化'
        },
        'scaling_approach': {
            'efficientnet': '复合缩放 (深度+宽度+分辨率)',
            'efficientnet_v2': '改进的缩放策略 + 渐进式训练'
        }
    }
    
    for category, diff in differences.items():
        print(f"   {category}:")
        print(f"     EfficientNet:    {diff['efficientnet']}")
        print(f"     EfficientNet V2: {diff['efficientnet_v2']}")
    
    comparison['architectural_differences'] = differences
    
    # V2 的关键改进
    improvements = [
        {
            'improvement': 'Fused-MBConv 块',
            'description': '在早期阶段使用融合的 MBConv 块，减少内存访问，提高训练速度',
            'benefit': '训练速度提升 2-3x，内存使用更高效'
        },
        {
            'improvement': '渐进式学习',
            'description': '从小图像开始训练，逐步增加图像尺寸和正则化强度',
            'benefit': '训练时间减少 5-11x，同时保持或提升准确率'
        },
        {
            'improvement': '自适应正则化',
            'description': '根据图像尺寸动态调整 dropout 和数据增强强度',
            'benefit': '更好的泛化能力，减少过拟合'
        },
        {
            'improvement': '改进的 NAS 搜索空间',
            'description': '扩展搜索空间，包含更多的操作类型和连接模式',
            'benefit': '找到更优的架构配置'
        },
        {
            'improvement': '更好的参数效率',
            'description': '在相同参数预算下实现更高准确率，或用更少参数达到相同准确率',
            'benefit': '更好的准确率/参数数量权衡'
        }
    ]
    
    print(f"\n🚀 EfficientNet V2 的关键改进:")
    for i, imp in enumerate(improvements, 1):
        print(f"   {i}. {imp['improvement']}")
        print(f"      描述: {imp['description']}")
        print(f"      优势: {imp['benefit']}")
    
    comparison['key_improvements'] = improvements
    
    # 权衡分析
    trade_offs = [
        {
            'aspect': '参数数量',
            'efficientnet': '更少参数 (1.6M)',
            'efficientnet_v2': '更多参数 (20.3M)',
            'impact': 'V2 模型更大，需要更多存储和计算资源'
        },
        {
            'aspect': '训练复杂度',
            'efficientnet': '标准训练流程',
            'efficientnet_v2': '需要渐进式训练策略',
            'impact': 'V2 训练实现更复杂，但效率更高'
        },
        {
            'aspect': '推理速度',
            'efficientnet': '轻量级，推理快',
            'efficientnet_v2': '参数多，推理相对慢',
            'impact': '边缘设备部署时需要考虑计算限制'
        },
        {
            'aspect': '准确率潜力',
            'efficientnet': '在小数据集上表现良好',
            'efficientnet_v2': '在大数据集上优势明显',
            'impact': '数据集规模影响模型选择'
        }
    ]
    
    print(f"\n⚖️ 权衡分析:")
    for trade_off in trade_offs:
        print(f"   {trade_off['aspect']}:")
        print(f"     EfficientNet:    {trade_off['efficientnet']}")
        print(f"     EfficientNet V2: {trade_off['efficientnet_v2']}")
        print(f"     影响: {trade_off['impact']}")
    
    comparison['trade_offs'] = trade_offs
    
    return comparison

def analyze_checkpoint_performance():
    """分析检查点中的性能表现"""
    print(f"\n📈 分析实际训练性能...")
    
    # 分析 EfficientNet 检查点
    efficientnet_checkpoint = "/home/aaa/ws/bioastModel/checkpoints/efficientnet_20250808_014214_best.pth"
    efficientnet_v2_checkpoint = "/home/aaa/ws/bioastModel/checkpoints/efficientnet_v2_20250808_071027_best.pth"
    
    performance_comparison = {
        'efficientnet': {'available': False},
        'efficientnet_v2': {'available': False}
    }
    
    # 分析 EfficientNet 性能
    if os.path.exists(efficientnet_checkpoint):
        try:
            checkpoint = torch.load(efficientnet_checkpoint, map_location='cpu')
            performance_comparison['efficientnet'] = {
                'available': True,
                'file_size_mb': round(os.path.getsize(efficientnet_checkpoint) / (1024 * 1024), 2),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'train_accuracy': checkpoint.get('train_acc', 'N/A'),
                'val_accuracy': checkpoint.get('val_acc', 'N/A'),
                'parameters': 1578391
            }
            print(f"   ✅ EfficientNet 性能:")
            print(f"      验证准确率: {performance_comparison['efficientnet']['val_accuracy']:.4f}")
            print(f"      训练轮次: {performance_comparison['efficientnet']['epoch']}")
            print(f"      文件大小: {performance_comparison['efficientnet']['file_size_mb']} MB")
        except Exception as e:
            print(f"   ❌ EfficientNet 检查点分析失败: {str(e)}")
    
    # 分析 EfficientNet V2 性能
    if os.path.exists(efficientnet_v2_checkpoint):
        try:
            checkpoint = torch.load(efficientnet_v2_checkpoint, map_location='cpu')
            performance_comparison['efficientnet_v2'] = {
                'available': True,
                'file_size_mb': round(os.path.getsize(efficientnet_v2_checkpoint) / (1024 * 1024), 2),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'train_accuracy': checkpoint.get('train_acc', 'N/A'),
                'val_accuracy': checkpoint.get('val_acc', 'N/A'),
                'parameters': 20334032
            }
            print(f"   ✅ EfficientNet V2 性能:")
            print(f"      验证准确率: {performance_comparison['efficientnet_v2'].get('val_accuracy', 'N/A')}")
            print(f"      训练轮次: {performance_comparison['efficientnet_v2']['epoch']}")
            print(f"      文件大小: {performance_comparison['efficientnet_v2']['file_size_mb']} MB")
        except Exception as e:
            print(f"   ❌ EfficientNet V2 检查点分析失败: {str(e)}")
    else:
        print(f"   ⚠️ EfficientNet V2 检查点不存在")
    
    return performance_comparison

def generate_comparison_report(efficientnet_analysis, efficientnet_v2_analysis, comparison, performance):
    """生成对比分析报告"""
    report = {
        'report_title': 'EfficientNet vs EfficientNet V2 架构对比分析',
        'generation_timestamp': datetime.now().isoformat(),
        'efficientnet_analysis': efficientnet_analysis,
        'efficientnet_v2_analysis': efficientnet_v2_analysis,
        'detailed_comparison': comparison,
        'performance_comparison': performance,
        'recommendations': []
    }
    
    # 生成推荐
    recommendations = []
    
    if performance['efficientnet']['available']:
        efficientnet_acc = performance['efficientnet'].get('val_accuracy', 0)
        if isinstance(efficientnet_acc, (int, float)) and efficientnet_acc > 0.99:
            recommendations.append({
                'scenario': '小规模数据集 + 资源受限环境',
                'recommendation': 'EfficientNet',
                'reason': f'参数少 (1.6M)，性能优秀 ({efficientnet_acc:.2%})，适合边缘设备部署'
            })
    
    recommendations.extend([
        {
            'scenario': '大规模数据集 + 充足计算资源',
            'recommendation': 'EfficientNet V2',
            'reason': '更强的表达能力，更好的训练效率，适合大规模训练'
        },
        {
            'scenario': '快速原型开发',
            'recommendation': 'EfficientNet',
            'reason': '训练简单，收敛快，参数少，调试方便'
        },
        {
            'scenario': '生产环境高精度需求',
            'recommendation': 'EfficientNet V2',
            'reason': '更先进的架构设计，更好的准确率潜力'
        },
        {
            'scenario': '移动端/边缘设备部署',
            'recommendation': 'EfficientNet',
            'reason': '模型小，推理快，内存占用少'
        }
    ])
    
    report['recommendations'] = recommendations
    
    # 保存报告
    report_path = "efficientnet_vs_v2_comparison.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 生成 HTML 报告
    generate_html_comparison_report(report)
    
    print(f"\n✅ 对比分析报告已生成:")
    print(f"   📁 JSON: {report_path}")
    print(f"   📁 HTML: efficientnet_vs_v2_comparison.html")
    
    return report

def generate_html_comparison_report(report):
    """生成 HTML 格式的对比报告"""
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report['report_title']}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 20px 0; }}
        .model-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
        .model-card.v2 {{ border-left-color: #e74c3c; }}
        .model-card h3 {{ margin-top: 0; color: #2c3e50; }}
        .param-comparison {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .improvement-list {{ background: #d5f4e6; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .trade-off-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .trade-off-table th, .trade-off-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .trade-off-table th {{ background-color: #3498db; color: white; }}
        .recommendations {{ background: #fff3cd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 30px; }}
        .highlight {{ background: #ffffcc; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 {report['report_title']}</h1>
        
        <div class="comparison-grid">
            <div class="model-card">
                <h3>📱 EfficientNet</h3>
                <p><strong>参数数量:</strong> {report['efficientnet_analysis']['total_parameters']:,} ({report['efficientnet_analysis']['total_parameters']/1000000:.1f}M)</p>
                <p><strong>核心特点:</strong></p>
                <ul>
                    <li>MBConv 移动倒置瓶颈块</li>
                    <li>Squeeze-and-Excitation 注意力</li>
                    <li>复合缩放策略</li>
                    <li>NAS 优化架构</li>
                    <li>高效参数利用</li>
                </ul>
            </div>
            
            <div class="model-card v2">
                <h3>🚀 EfficientNet V2</h3>
                <p><strong>参数数量:</strong> {report['efficientnet_v2_analysis']['total_parameters']:,} ({report['efficientnet_v2_analysis']['total_parameters']/1000000:.1f}M)</p>
                <p><strong>核心特点:</strong></p>
                <ul>
                    <li>Fused-MBConv + MBConv 混合</li>
                    <li>渐进式学习策略</li>
                    <li>自适应正则化</li>
                    <li>改进的训练效率</li>
                    <li>更好的准确率/参数权衡</li>
                </ul>
            </div>
        </div>
        
        <h2>📊 参数数量对比</h2>
        <div class="param-comparison">
            <p><strong>参数增长:</strong> EfficientNet V2 比 EfficientNet 多 <span class="highlight">{report['detailed_comparison']['parameter_comparison']['parameter_ratio']:.1f}倍</span> 参数</p>
            <p><strong>具体数字:</strong> 从 {report['efficientnet_analysis']['total_parameters']/1000000:.1f}M 增加到 {report['efficientnet_v2_analysis']['total_parameters']/1000000:.1f}M，增加了 {report['detailed_comparison']['parameter_comparison']['parameter_increase']/1000000:.1f}M 参数</p>
            <p><strong>原因分析:</strong></p>
            <ul>
                <li>更深的网络结构 (更多层)</li>
                <li>更宽的特征通道</li>
                <li>Fused-MBConv 块参数开销</li>
                <li>改进的搜索空间带来的复杂度</li>
            </ul>
        </div>
        
        <h2>🚀 EfficientNet V2 的关键改进</h2>
        <div class="improvement-list">
    """
    
    for improvement in report['detailed_comparison']['key_improvements']:
        html_content += f"""
            <div style="margin-bottom: 15px;">
                <h4>{improvement['improvement']}</h4>
                <p><strong>描述:</strong> {improvement['description']}</p>
                <p><strong>优势:</strong> {improvement['benefit']}</p>
            </div>
        """
    
    html_content += f"""
        </div>
        
        <h2>⚖️ 权衡分析</h2>
        <table class="trade-off-table">
            <thead>
                <tr>
                    <th>对比方面</th>
                    <th>EfficientNet</th>
                    <th>EfficientNet V2</th>
                    <th>影响</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for trade_off in report['detailed_comparison']['trade_offs']:
        html_content += f"""
                <tr>
                    <td><strong>{trade_off['aspect']}</strong></td>
                    <td>{trade_off['efficientnet']}</td>
                    <td>{trade_off['efficientnet_v2']}</td>
                    <td>{trade_off['impact']}</td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>
        
        <h2>💡 使用建议</h2>
        <div class="recommendations">
    """
    
    for rec in report['recommendations']:
        html_content += f"""
            <div style="margin-bottom: 15px;">
                <h4>{rec['scenario']}</h4>
                <p><strong>推荐:</strong> {rec['recommendation']}</p>
                <p><strong>理由:</strong> {rec['reason']}</p>
            </div>
        """
    
    html_content += f"""
        </div>
        
        <div class="timestamp">
            <p>报告生成时间: {report['generation_timestamp']}</p>
        </div>
    </div>
</body>
</html>
    """
    
    with open("efficientnet_vs_v2_comparison.html", 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """主函数"""
    print("🚀 开始 EfficientNet vs EfficientNet V2 对比分析...")
    
    try:
        # 1. 分析 EfficientNet 架构
        print("\n" + "="*60)
        efficientnet_analysis = analyze_efficientnet_architecture()
        
        # 2. 分析 EfficientNet V2 架构
        print("\n" + "="*60)
        efficientnet_v2_analysis = analyze_efficientnet_v2_architecture()
        
        # 3. 对比架构差异
        comparison = compare_architectures(efficientnet_analysis, efficientnet_v2_analysis)
        
        # 4. 分析实际性能
        performance = analyze_checkpoint_performance()
        
        # 5. 生成对比报告
        print(f"\n📝 生成对比分析报告...")
        report = generate_comparison_report(efficientnet_analysis, efficientnet_v2_analysis, comparison, performance)
        
        print(f"\n🎉 EfficientNet vs EfficientNet V2 对比分析完成!")
        return True
        
    except Exception as e:
        print(f"❌ 对比分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)