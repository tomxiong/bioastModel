"""
生成集成了可视化图表的综合模型对比报告
"""

import os
import base64
from fixed_comparison_visualizations import generate_all_comparison_charts

def create_integrated_html_report():
    """创建包含所有图表的集成HTML报告"""
    print("🔧 生成集成的综合对比报告...")
    
    # 生成所有图表
    print("📊 生成对比图表...")
    charts, data = generate_all_comparison_charts()
    
    # 读取HTML模板
    with open('model_comparison_report.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 替换图表占位符
    replacements = {
        '<div id="radar-chart">': f'''<div id="radar-chart">
                    <img src="data:image/png;base64,{charts['radar']}" alt="Performance Radar Chart Comparison" style="max-width: 100%; height: auto;">''',
        
        '<div id="training-chart">': f'''<div id="training-chart">
                    <img src="data:image/png;base64,{charts['training_history']}" alt="Training History Comparison" style="max-width: 100%; height: auto;">''',
        
        '<div id="efficiency-chart">': f'''<div id="efficiency-chart">
                    <img src="data:image/png;base64,{charts['efficiency']}" alt="Efficiency Analysis Chart" style="max-width: 100%; height: auto;">''',
        
        '<div id="confusion-matrix-chart">': f'''<div id="confusion-matrix-chart">
                    <img src="data:image/png;base64,{charts['confusion_matrix']}" alt="Confusion Matrix Comparison" style="max-width: 100%; height: auto;">'''
    }
    
    # 应用替换
    for old_text, new_text in replacements.items():
        html_content = html_content.replace(old_text, new_text)
    
    # 添加收敛分析和ResNet训练曲线图表到训练分析部分
    convergence_section = f'''
            <div class="chart-container">
                <h3>收敛分析对比</h3>
                <img src="data:image/png;base64,{charts['convergence']}" alt="Convergence Analysis Chart" style="max-width: 100%; height: auto;">
                <p><em>左图：验证损失收敛对比，右图：学习率调度策略对比</em></p>
            </div>
            
            <div class="chart-container">
                <h3>ResNet-18 Improved 详细训练过程</h3>
                <img src="data:image/png;base64,{charts['resnet_training']}" alt="ResNet Training Curves" style="max-width: 100%; height: auto;">
                <p><em>ResNet-18 Improved的完整训练过程：损失曲线、准确率曲线、学习率变化和训练稳定性分析</em></p>
            </div>
        </div>'''
    
    # 在训练分析部分末尾添加收敛分析
    html_content = html_content.replace(
        '</div>\n        \n        <div id="efficiency" class="tab-content">',
        convergence_section + '\n        \n        <div id="efficiency" class="tab-content">'
    )
    
    # 更新生成时间和统计信息
    import datetime
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 添加详细的统计信息
    stats_section = f'''
        <div class="model-card">
            <h3>📈 详细统计信息</h3>
            <div class="pros-cons">
                <div class="pros">
                    <h4>EfficientNet-B0 统计</h4>
                    <ul>
                        <li><strong>参数量：</strong>{data['efficientnet']['params']:.2f}M</li>
                        <li><strong>训练轮数：</strong>{data['efficientnet']['epochs']}轮</li>
                        <li><strong>最终验证准确率：</strong>{max(data['efficientnet']['history']['val_acc']):.4f}</li>
                        <li><strong>最低验证损失：</strong>{min(data['efficientnet']['history']['val_loss']):.4f}</li>
                        <li><strong>效率比：</strong>{data['efficientnet']['test']['accuracy']/data['efficientnet']['params']:.2f}%/M</li>
                    </ul>
                </div>
                <div class="pros">
                    <h4>ResNet-18 Improved 统计</h4>
                    <ul>
                        <li><strong>参数量：</strong>{data['resnet']['params']:.2f}M</li>
                        <li><strong>训练轮数：</strong>{data['resnet']['epochs']}轮</li>
                        <li><strong>最终验证准确率：</strong>{max(data['resnet']['history']['val_acc']):.4f}</li>
                        <li><strong>最低验证损失：</strong>{min(data['resnet']['history']['val_loss']):.4f}</li>
                        <li><strong>效率比：</strong>{data['resnet']['test']['accuracy']/data['resnet']['params']:.2f}%/M</li>
                    </ul>
                </div>
            </div>
        </div>'''
    
    # 在性能对比部分添加统计信息
    html_content = html_content.replace(
        '</div>\n        \n        <div id="training" class="tab-content">',
        stats_section + '\n        </div>\n        \n        <div id="training" class="tab-content">'
    )
    
    # 保存集成报告
    output_path = 'integrated_model_comparison_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 集成报告已生成: {output_path}")
    print(f"📊 包含 {len(charts)} 个可视化图表")
    print(f"🎯 EfficientNet-B0 效率比: {data['efficientnet']['test']['accuracy']/data['efficientnet']['params']:.2f}%/M")
    print(f"🎯 ResNet-18 效率比: {data['resnet']['test']['accuracy']/data['resnet']['params']:.2f}%/M")
    
    return output_path

if __name__ == "__main__":
    create_integrated_html_report()