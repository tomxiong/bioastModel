"""
生成ONNX转换报告
收集所有模型的转换结果，生成详细的转换报告
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def check_onnx_model_exists(model_name):
    """检查ONNX模型是否存在
    
    Args:
        model_name: 模型名称
        
    Returns:
        是否存在
    """
    onnx_path = Path("onnx_models") / f"{model_name}.onnx"
    return onnx_path.exists()

def get_onnx_model_size(model_name):
    """获取ONNX模型大小
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型大小（MB）
    """
    onnx_path = Path("onnx_models") / f"{model_name}.onnx"
    if onnx_path.exists():
        size_bytes = onnx_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    return None

def get_model_performance(model_name):
    """获取模型性能指标
    
    Args:
        model_name: 模型名称
        
    Returns:
        性能指标字典
    """
    # 查找最新的实验目录
    experiments_dir = Path("experiments")
    model_dirs = list(experiments_dir.glob(f"**/{model_name}"))
    
    if not model_dirs:
        logging.warning(f"未找到{model_name}的实验目录")
        return None
    
    # 按修改时间排序，获取最新的实验目录
    latest_dir = max(model_dirs, key=os.path.getmtime)
    
    # 查找性能指标文件
    metrics_file = latest_dir / "metrics.json"
    
    if not metrics_file.exists():
        logging.warning(f"未找到{model_name}的性能指标文件")
        return None
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logging.error(f"加载{model_name}性能指标失败: {e}")
        return None

def generate_report():
    """生成转换报告"""
    # 所有模型
    all_models = [
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
    
    # 收集结果
    results = []
    
    for model_name in all_models:
        exists = check_onnx_model_exists(model_name)
        size = get_onnx_model_size(model_name) if exists else None
        performance = get_model_performance(model_name)
        
        accuracy = None
        f1_score = None
        
        if performance and 'validation' in performance:
            if 'accuracy' in performance['validation']:
                accuracy = performance['validation']['accuracy']
            if 'f1' in performance['validation']:
                f1_score = performance['validation']['f1']
        
        results.append({
            "模型名称": model_name,
            "ONNX转换状态": "成功" if exists else "失败",
            "模型大小(MB)": size if exists else None,
            "验证准确率": f"{accuracy:.2%}" if accuracy is not None else None,
            "验证F1分数": f"{f1_score:.2%}" if f1_score is not None else None
        })
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 创建报告目录
    report_dir = Path("reports/onnx_conversion")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV报告
    csv_path = report_dir / "conversion_results.csv"
    df.to_csv(csv_path, index=False)
    
    # 生成Markdown报告
    md_path = report_dir / "conversion_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ONNX模型转换报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 转换结果\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # 统计成功和失败的数量
        success_count = df[df["ONNX转换状态"] == "成功"].shape[0]
        failed_count = df[df["ONNX转换状态"] == "失败"].shape[0]
        
        f.write(f"## 统计\n\n")
        f.write(f"- 成功转换: {success_count}/{len(all_models)}\n")
        f.write(f"- 失败: {failed_count}/{len(all_models)}\n")
        
        # 添加模型大小比较
        if success_count > 0:
            f.write("\n## 模型大小比较\n\n")
            size_df = df[df["模型大小(MB)"].notna()].sort_values(by="模型大小(MB)")
            f.write(size_df[["模型名称", "模型大小(MB)"]].to_markdown(index=False))
            
            # 添加模型大小可视化描述
            f.write("\n\n### 模型大小分析\n\n")
            smallest_model = size_df.iloc[0]
            largest_model = size_df.iloc[-1]
            f.write(f"- 最小模型: {smallest_model['模型名称']} ({smallest_model['模型大小(MB)']} MB)\n")
            f.write(f"- 最大模型: {largest_model['模型名称']} ({largest_model['模型大小(MB)']} MB)\n")
            f.write(f"- 大小比例: {largest_model['模型大小(MB)'] / smallest_model['模型大小(MB)']:.2f}倍\n")
        
        # 添加性能比较
        performance_df = df[df["验证准确率"].notna()]
        if len(performance_df) > 0:
            f.write("\n## 性能比较\n\n")
            performance_df = performance_df.sort_values(by="验证准确率", ascending=False)
            f.write(performance_df[["模型名称", "验证准确率", "验证F1分数"]].to_markdown(index=False))
    
    logging.info(f"报告已保存至: {md_path}")
    
    # 生成HTML报告
    try:
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        # 创建模型大小比较图
        if success_count > 0:
            plt.figure(figsize=(10, 6))
            size_df = df[df["模型大小(MB)"].notna()].sort_values(by="模型大小(MB)")
            plt.bar(size_df["模型名称"], size_df["模型大小(MB)"])
            plt.title("模型大小比较")
            plt.xlabel("模型")
            plt.ylabel("大小 (MB)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # 保存图像到内存
            buf = BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("ascii")
            plt.close()
            
            # 创建HTML报告
            html_path = report_dir / "conversion_report.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("<!DOCTYPE html>\n")
                f.write("<html>\n")
                f.write("<head>\n")
                f.write("    <title>ONNX模型转换报告</title>\n")
                f.write("    <style>\n")
                f.write("        body { font-family: Arial, sans-serif; margin: 20px; }\n")
                f.write("        table { border-collapse: collapse; width: 100%; }\n")
                f.write("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
                f.write("        th { background-color: #f2f2f2; }\n")
                f.write("        tr:nth-child(even) { background-color: #f9f9f9; }\n")
                f.write("        .success { color: green; }\n")
                f.write("        .failure { color: red; }\n")
                f.write("    </style>\n")
                f.write("</head>\n")
                f.write("<body>\n")
                f.write(f"    <h1>ONNX模型转换报告</h1>\n")
                f.write(f"    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
                
                f.write("    <h2>转换结果</h2>\n")
                f.write("    <table>\n")
                f.write("        <tr>\n")
                for col in df.columns:
                    f.write(f"            <th>{col}</th>\n")
                f.write("        </tr>\n")
                
                for _, row in df.iterrows():
                    f.write("        <tr>\n")
                    for col in df.columns:
                        value = row[col]
                        if col == "ONNX转换状态":
                            if value == "成功":
                                f.write(f"            <td class='success'>{value}</td>\n")
                            else:
                                f.write(f"            <td class='failure'>{value}</td>\n")
                        else:
                            f.write(f"            <td>{value}</td>\n")
                    f.write("        </tr>\n")
                f.write("    </table>\n")
                
                f.write("    <h2>统计</h2>\n")
                f.write("    <ul>\n")
                f.write(f"        <li>成功转换: {success_count}/{len(all_models)}</li>\n")
                f.write(f"        <li>失败: {failed_count}/{len(all_models)}</li>\n")
                f.write("    </ul>\n")
                
                if success_count > 0:
                    f.write("    <h2>模型大小比较</h2>\n")
                    f.write(f"    <img src='data:image/png;base64,{img_str}' alt='模型大小比较'>\n")
                    
                    f.write("    <h3>模型大小分析</h3>\n")
                    f.write("    <ul>\n")
                    f.write(f"        <li>最小模型: {smallest_model['模型名称']} ({smallest_model['模型大小(MB)']} MB)</li>\n")
                    f.write(f"        <li>最大模型: {largest_model['模型名称']} ({largest_model['模型大小(MB)']} MB)</li>\n")
                    f.write(f"        <li>大小比例: {largest_model['模型大小(MB)'] / smallest_model['模型大小(MB)']:.2f}倍</li>\n")
                    f.write("    </ul>\n")
                
                if len(performance_df) > 0:
                    f.write("    <h2>性能比较</h2>\n")
                    f.write("    <table>\n")
                    f.write("        <tr>\n")
                    for col in ["模型名称", "验证准确率", "验证F1分数"]:
                        f.write(f"            <th>{col}</th>\n")
                    f.write("        </tr>\n")
                    
                    for _, row in performance_df.iterrows():
                        f.write("        <tr>\n")
                        for col in ["模型名称", "验证准确率", "验证F1分数"]:
                            f.write(f"            <td>{row[col]}</td>\n")
                        f.write("        </tr>\n")
                    f.write("    </table>\n")
                
                f.write("</body>\n")
                f.write("</html>\n")
            
            logging.info(f"HTML报告已保存至: {html_path}")
    except Exception as e:
        logging.error(f"生成HTML报告失败: {e}")

def main():
    """主函数"""
    generate_report()

if __name__ == "__main__":
    main()