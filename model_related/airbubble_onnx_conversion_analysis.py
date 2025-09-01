#!/usr/bin/env python3
"""
AirBubble Hybrid Net ONNX 转换分析和建议
分析当前转换问题并提供解决方案
"""

import os
import sys
import json
from datetime import datetime

def analyze_conversion_issues():
    """分析转换问题"""
    print("🔍 分析 AirBubble Hybrid Net ONNX 转换问题...")
    
    issues = {
        "architecture_complexity": {
            "description": "AirBubbleHybridNet 包含复杂的多任务架构",
            "details": [
                "多个检测头：ring_detector, center_detector, edge_detector",
                "Transformer 块与 CNN 的混合架构",
                "动态形状的卷积操作",
                "多输出字典结构"
            ],
            "impact": "ONNX 导出时遇到不支持的操作"
        },
        "dynamic_shapes": {
            "description": "模型中存在动态形状操作",
            "details": [
                "InvertedResidual 块中的动态卷积",
                "自适应池化操作",
                "条件分支结构"
            ],
            "impact": "ONNX Runtime 无法处理未知形状的卷积"
        },
        "multi_output_structure": {
            "description": "模型返回多个输出的字典结构",
            "details": [
                "classification 输出",
                "turbidity 输出", 
                "bubble_params 输出",
                "quality 输出"
            ],
            "impact": "ONNX 导出需要明确的输出结构"
        },
        "weight_mismatch": {
            "description": "简化架构导致权重不匹配",
            "details": [
                "原始模型权重无法正确映射到简化架构",
                "特征提取层的维度不一致",
                "分类头的参数数量不匹配"
            ],
            "impact": "模型性能严重下降（99.16% -> 46.81%）"
        }
    }
    
    return issues

def generate_recommendations():
    """生成解决建议"""
    print("💡 生成解决建议...")
    
    recommendations = {
        "immediate_solutions": [
            {
                "title": "保持 PyTorch 模型用于生产",
                "description": "鉴于 ONNX 转换的复杂性，建议继续使用原始 PyTorch 模型",
                "pros": ["保持 99.16% 的高准确率", "完整的功能支持", "稳定可靠"],
                "cons": ["推理速度较慢（29.79 ms/batch）", "部署复杂度较高"]
            },
            {
                "title": "使用 TorchScript 替代 ONNX",
                "description": "TorchScript 对复杂 PyTorch 模型有更好的支持",
                "pros": ["更好的 PyTorch 兼容性", "保持模型完整性", "相对简单的转换"],
                "cons": ["仍然依赖 PyTorch 运行时", "跨平台支持有限"]
            }
        ],
        "long_term_solutions": [
            {
                "title": "重新设计 ONNX 友好的架构",
                "description": "从头设计一个 ONNX 兼容的模型架构",
                "steps": [
                    "简化多任务结构为单一分类任务",
                    "使用标准卷积操作替代动态操作",
                    "避免复杂的条件分支",
                    "使用固定形状的操作"
                ],
                "effort": "高",
                "timeline": "2-3 周"
            },
            {
                "title": "分阶段转换策略",
                "description": "将复杂模型分解为多个简单的 ONNX 模型",
                "steps": [
                    "特征提取器转换为 ONNX",
                    "分类头单独转换",
                    "后处理逻辑在应用层实现"
                ],
                "effort": "中等",
                "timeline": "1-2 周"
            }
        ],
        "alternative_approaches": [
            {
                "title": "使用 OpenVINO",
                "description": "Intel OpenVINO 对复杂模型有更好的支持",
                "benefits": ["更好的模型优化", "硬件加速支持", "复杂架构兼容性"]
            },
            {
                "title": "使用 TensorRT",
                "description": "NVIDIA TensorRT 用于 GPU 推理优化",
                "benefits": ["极高的推理速度", "GPU 优化", "动态形状支持"]
            }
        ]
    }
    
    return recommendations

def create_conversion_report():
    """创建转换分析报告"""
    print("📊 创建转换分析报告...")
    
    issues = analyze_conversion_issues()
    recommendations = generate_recommendations()
    
    # 读取验证报告数据
    validation_report_path = "airbubble_hybrid_net_onnx_validation_report.json"
    validation_data = {}
    
    if os.path.exists(validation_report_path):
        with open(validation_report_path, 'r', encoding='utf-8') as f:
            validation_data = json.load(f)
    
    report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "model_info": {
            "name": "AirBubbleHybridNet",
            "original_accuracy": 99.16,
            "onnx_accuracy": 46.81,
            "accuracy_loss": 52.36,
            "architecture_type": "Multi-task CNN-Transformer Hybrid"
        },
        "conversion_attempts": [
            {
                "method": "simplified_architecture",
                "result": "FAILED",
                "accuracy_achieved": 46.81,
                "issues": ["Weight mapping failure", "Architecture mismatch", "Feature loss"]
            },
            {
                "method": "proper_architecture_preservation", 
                "result": "FAILED",
                "issues": ["Dynamic shape operations", "Complex multi-output structure", "ONNX compatibility"]
            }
        ],
        "technical_issues": issues,
        "recommendations": recommendations,
        "validation_results": validation_data.get("performance_comparison", {}),
        "conclusion": {
            "status": "CONVERSION_NOT_FEASIBLE",
            "reason": "Architecture complexity exceeds ONNX compatibility limits",
            "recommended_action": "Continue using PyTorch model for production"
        }
    }
    
    return report

def save_analysis_report(report):
    """保存分析报告"""
    # 保存 JSON 报告
    json_path = "airbubble_hybrid_net_onnx_analysis_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 分析报告保存到: {json_path}")
    
    # 生成 HTML 报告
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AirBubble Hybrid Net ONNX 转换分析报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.2em;
            font-weight: 300;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
            font-size: 0.9em;
            background: #f8d7da;
            color: #721c24;
        }}
        .section {{
            padding: 30px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 10px;
        }}
        .issue-card {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        .issue-title {{
            font-weight: bold;
            color: #856404;
            margin-bottom: 10px;
        }}
        .issue-details {{
            margin: 10px 0;
        }}
        .issue-details ul {{
            margin: 5px 0;
            padding-left: 20px;
        }}
        .recommendation-card {{
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        .recommendation-title {{
            font-weight: bold;
            color: #0c5460;
            margin-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #e74c3c;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #e74c3c;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .conclusion {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }}
        .conclusion h3 {{
            color: #721c24;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 AirBubble Hybrid Net ONNX 转换分析</h1>
            <p>技术问题分析与解决方案建议</p>
            <div class="status-badge">
                {report['conclusion']['status']}
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">📊 转换尝试结果</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{report['model_info']['original_accuracy']:.2f}%</div>
                    <div class="metric-label">原始模型准确率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['model_info']['onnx_accuracy']:.2f}%</div>
                    <div class="metric-label">ONNX 模型准确率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report['model_info']['accuracy_loss']:.2f}%</div>
                    <div class="metric-label">准确率损失</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(report['conversion_attempts'])}</div>
                    <div class="metric-label">转换尝试次数</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">⚠️ 技术问题分析</h2>
    """
    
    for issue_key, issue_data in report['technical_issues'].items():
        html_content += f"""
            <div class="issue-card">
                <div class="issue-title">{issue_data['description']}</div>
                <div class="issue-details">
                    <strong>具体问题:</strong>
                    <ul>
        """
        for detail in issue_data['details']:
            html_content += f"<li>{detail}</li>"
        
        html_content += f"""
                    </ul>
                    <strong>影响:</strong> {issue_data['impact']}
                </div>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2 class="section-title">💡 解决方案建议</h2>
    """
    
    for rec in report['recommendations']['immediate_solutions']:
        html_content += f"""
            <div class="recommendation-card">
                <div class="recommendation-title">🚀 {rec['title']}</div>
                <p>{rec['description']}</p>
                <strong>优点:</strong> {', '.join(rec['pros'])}<br>
                <strong>缺点:</strong> {', '.join(rec['cons'])}
            </div>
        """
    
    html_content += f"""
        </div>
        
        <div class="section">
            <h2 class="section-title">🎯 结论与建议</h2>
            <div class="conclusion">
                <h3>{report['conclusion']['status']}</h3>
                <p><strong>原因:</strong> {report['conclusion']['reason']}</p>
                <p><strong>建议行动:</strong> {report['conclusion']['recommended_action']}</p>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">ℹ️ 分析信息</h2>
            <p><strong>分析时间:</strong> {report['analysis_timestamp']}</p>
            <p><strong>模型类型:</strong> {report['model_info']['architecture_type']}</p>
            <p><strong>转换方法:</strong> 简化架构、完整架构保持</p>
        </div>
    </div>
</body>
</html>
    """
    
    html_path = "airbubble_hybrid_net_onnx_analysis_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML 分析报告保存到: {html_path}")

def main():
    """主函数"""
    print("🚀 开始 AirBubble Hybrid Net ONNX 转换分析...")
    
    try:
        # 创建分析报告
        report = create_conversion_report()
        
        # 保存报告
        save_analysis_report(report)
        
        # 输出总结
        print("\n" + "="*60)
        print("📋 ONNX 转换分析完成")
        print("="*60)
        print(f"🎯 结论: {report['conclusion']['status']}")
        print(f"📉 准确率损失: {report['model_info']['accuracy_loss']:.2f}%")
        print(f"💡 建议: {report['conclusion']['recommended_action']}")
        print("="*60)
        
        print("\n🔍 主要发现:")
        print("  • AirBubbleHybridNet 架构过于复杂，超出 ONNX 兼容性限制")
        print("  • 简化架构方法导致严重的性能损失（52.36%）")
        print("  • 动态形状操作和多输出结构是主要障碍")
        
        print("\n💡 推荐方案:")
        print("  • 继续使用 PyTorch 模型进行生产部署")
        print("  • 考虑使用 TorchScript 进行模型优化")
        print("  • 长期考虑重新设计 ONNX 友好的架构")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ ONNX 转换分析报告已生成!")
    else:
        print("\n❌ 分析报告生成失败")
        sys.exit(1)