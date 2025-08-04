"""
比较simplified_airbubble_detector模型与其他模型的性能
使用相同的数据集和评估标准
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def load_model_results(model_name):
    """加载模型测试结果"""
    experiments_dir = Path("experiments")
    model_dirs = list(experiments_dir.glob(f"**/{model_name}"))
    
    if not model_dirs:
        logging.warning(f"未找到{model_name}的实验目录")
        return None
    
    # 按修改时间排序，获取最新的实验目录
    latest_dir = max(model_dirs, key=os.path.getmtime)
    
    # 查找测试结果文件
    test_results_file = latest_dir / f"{model_name}_test_results.json"
    if not test_results_file.exists():
        logging.warning(f"未找到{model_name}的测试结果文件")
        return None
    
    try:
        with open(test_results_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        logging.error(f"加载{model_name}测试结果失败: {e}")
        return None

def load_all_model_results():
    """加载所有模型的测试结果"""
    model_names = [
        'simplified_airbubble_detector',
        'efficientnet_b0',
        'resnet18_improved',
        'convnext_tiny',
        'coatnet',
        'vit_tiny',
        'mic_mobilenetv3',
        'micro_vit',
        'airbubble_hybrid_net'
    ]
    
    results = {}
    for model_name in model_names:
        model_results = load_model_results(model_name)
        if model_results:
            results[model_name] = model_results
    
    return results

def create_comparison_table(results):
    """创建比较表格"""
    if not results:
        logging.error("没有可用的模型结果")
        return None
    
    # 准备表格数据
    data = []
    for model_name, model_results in results.items():
        row = {
            '模型': model_name,
            '准确率 (%)': model_results.get('test_acc', 0) * 100,
            'F1分数 (%)': model_results.get('test_f1', 0) * 100,
        }
        
        # 添加类别特定的指标（如果有）
        if 'test_class_acc' in model_results:
            row['阴性准确率 (%)'] = model_results['test_class_acc'].get('0', 0) * 100
            row['阳性准确率 (%)'] = model_results['test_class_acc'].get('1', 0) * 100
        
        if 'test_precision' in model_results:
            row['阴性精确率 (%)'] = model_results['test_precision'].get('0', 0) * 100
            row['阳性精确率 (%)'] = model_results['test_precision'].get('1', 0) * 100
        
        if 'test_recall' in model_results:
            row['阴性召回率 (%)'] = model_results['test_recall'].get('0', 0) * 100
            row['阳性召回率 (%)'] = model_results['test_recall'].get('1', 0) * 100
        
        # 添加参数数量（如果有）
        if 'params' in model_results:
            row['参数数量'] = model_results['params']
        
        data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 按准确率排序
    df = df.sort_values('准确率 (%)', ascending=False)
    
    return df

def plot_comparison_charts(results):
    """绘制比较图表"""
    if not results:
        logging.error("没有可用的模型结果")
        return
    
    # 准备数据
    model_names = list(results.keys())
    accuracies = [results[model]['test_acc'] * 100 for model in model_names]
    f1_scores = [results[model]['test_f1'] * 100 for model in model_names]
    
    # 按准确率排序
    sorted_indices = np.argsort(accuracies)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制准确率和F1分数
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='准确率')
    plt.bar(x + width/2, f1_scores, width, label='F1分数')
    
    plt.xlabel('模型')
    plt.ylabel('百分比 (%)')
    plt.title('模型性能比较')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    os.makedirs('reports/model_comparison', exist_ok=True)
    plt.savefig('reports/model_comparison/performance_comparison.png', dpi=300)
    plt.close()
    
    # 如果有类别特定的指标，绘制类别性能图表
    has_class_metrics = any('test_class_acc' in results[model] for model in results)
    if has_class_metrics:
        plt.figure(figsize=(14, 10))
        
        # 准备数据
        neg_acc = []
        pos_acc = []
        neg_prec = []
        pos_prec = []
        neg_rec = []
        pos_rec = []
        
        for model in model_names:
            if 'test_class_acc' in results[model]:
                neg_acc.append(results[model]['test_class_acc'].get('0', 0) * 100)
                pos_acc.append(results[model]['test_class_acc'].get('1', 0) * 100)
            else:
                neg_acc.append(0)
                pos_acc.append(0)
            
            if 'test_precision' in results[model]:
                neg_prec.append(results[model]['test_precision'].get('0', 0) * 100)
                pos_prec.append(results[model]['test_precision'].get('1', 0) * 100)
            else:
                neg_prec.append(0)
                pos_prec.append(0)
            
            if 'test_recall' in results[model]:
                neg_rec.append(results[model]['test_recall'].get('0', 0) * 100)
                pos_rec.append(results[model]['test_recall'].get('1', 0) * 100)
            else:
                neg_rec.append(0)
                pos_rec.append(0)
        
        # 绘制类别准确率
        plt.subplot(3, 1, 1)
        plt.bar(x - width/2, neg_acc, width, label='阴性(不生长)准确率')
        plt.bar(x + width/2, pos_acc, width, label='阳性(生长)准确率')
        plt.ylabel('准确率 (%)')
        plt.title('类别准确率比较')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 绘制类别精确率
        plt.subplot(3, 1, 2)
        plt.bar(x - width/2, neg_prec, width, label='阴性(不生长)精确率')
        plt.bar(x + width/2, pos_prec, width, label='阳性(生长)精确率')
        plt.ylabel('精确率 (%)')
        plt.title('类别精确率比较')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 绘制类别召回率
        plt.subplot(3, 1, 3)
        plt.bar(x - width/2, neg_rec, width, label='阴性(不生长)召回率')
        plt.bar(x + width/2, pos_rec, width, label='阳性(生长)召回率')
        plt.ylabel('召回率 (%)')
        plt.title('类别召回率比较')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/model_comparison/class_performance_comparison.png', dpi=300)
        plt.close()

def generate_comparison_report(df):
    """生成比较报告"""
    if df is None:
        logging.error("没有可用的比较数据")
        return
    
    # 创建报告目录
    os.makedirs('reports/model_comparison', exist_ok=True)
    
    # 生成报告
    report_path = 'reports/model_comparison/model_comparison_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 模型性能比较报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 性能概览\n\n")
        f.write("下表按准确率降序排列所有模型的性能指标：\n\n")
        
        # 将DataFrame转换为Markdown表格
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 图表分析\n\n")
        f.write("### 总体性能比较\n\n")
        f.write("![性能比较](performance_comparison.png)\n\n")
        
        # 如果有类别特定的指标，添加类别性能图表
        if '阴性准确率 (%)' in df.columns:
            f.write("### 类别性能比较\n\n")
            f.write("![类别性能比较](class_performance_comparison.png)\n\n")
        
        f.write("## 分析结论\n\n")
        
        # 获取最佳模型
        best_model = df.iloc[0]['模型']
        best_acc = df.iloc[0]['准确率 (%)']
        
        f.write(f"1. **最佳模型**: {best_model}，准确率达到 {best_acc:.2f}%\n")
        
        # 如果simplified_airbubble_detector在前三名中
        simplified_rank = df[df['模型'] == 'simplified_airbubble_detector'].index[0] + 1 if 'simplified_airbubble_detector' in df['模型'].values else None
        if simplified_rank is not None and simplified_rank <= 3:
            f.write(f"2. **简化气泡检测器表现优异**: 在所有模型中排名第{simplified_rank}，证明其设计有效\n")
        elif simplified_rank is not None:
            f.write(f"2. **简化气泡检测器表现一般**: 在所有模型中排名第{simplified_rank}\n")
        
        # 类别性能分析
        if '阴性准确率 (%)' in df.columns and '阳性准确率 (%)' in df.columns:
            avg_neg_acc = df['阴性准确率 (%)'].mean()
            avg_pos_acc = df['阳性准确率 (%)'].mean()
            
            if avg_neg_acc > avg_pos_acc:
                f.write(f"3. **类别不平衡**: 大多数模型在阴性(不生长)样本上表现更好，平均准确率为{avg_neg_acc:.2f}%，而阳性(生长)样本的平均准确率为{avg_pos_acc:.2f}%\n")
            else:
                f.write(f"3. **类别平衡**: 大多数模型在阳性(生长)样本上表现更好，平均准确率为{avg_pos_acc:.2f}%，而阴性(不生长)样本的平均准确率为{avg_neg_acc:.2f}%\n")
        
        # 参数效率分析
        if '参数数量' in df.columns:
            df['参数效率'] = df['准确率 (%)'] / df['参数数量']
            most_efficient_model = df.iloc[df['参数效率'].argmax()]['模型']
            f.write(f"4. **参数效率**: {most_efficient_model}模型在参数效率方面表现最佳，即在相同参数量下提供最高的准确率\n")
    
    logging.info(f"比较报告已生成: {report_path}")
    return report_path

def main():
    """主函数"""
    logging.info("开始比较模型性能...")
    
    # 加载所有模型的测试结果
    results = load_all_model_results()
    
    if not results:
        logging.error("没有找到任何模型的测试结果")
        return
    
    logging.info(f"成功加载{len(results)}个模型的测试结果")
    
    # 创建比较表格
    df = create_comparison_table(results)
    
    # 绘制比较图表
    plot_comparison_charts(results)
    
    # 生成比较报告
    report_path = generate_comparison_report(df)
    
    logging.info("模型比较完成")
    logging.info(f"报告已保存至: {report_path}")

if __name__ == "__main__":
    main()