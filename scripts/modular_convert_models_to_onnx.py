"""
模块化ONNX转换控制脚本
提供命令行接口，可选择性地转换一个或多个模型
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 导入模型转换器
import importlib

def get_converter_class(model_name):
    """获取模型转换器类
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型转换器类
    """
    try:
        # 动态导入转换器模块
        module_name = f"converters.{model_name.lower()}_converter"
        module = importlib.import_module(module_name)
        
        # 获取转换器类名
        class_name = ''.join(word.capitalize() for word in model_name.split('_')) + 'Converter'
        
        # 获取转换器类
        converter_class = getattr(module, class_name)
        return converter_class
    except (ImportError, AttributeError) as e:
        logging.error(f"获取{model_name}转换器类失败: {e}")
        return None

def convert_model(model_name):
    """转换指定模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        转换结果字典
    """
    logging.info(f"\n{'='*50}")
    logging.info(f"处理模型: {model_name}")
    
    # 检查ONNX文件是否已存在
    onnx_path = Path("onnx_models") / f"{model_name}.onnx"
    if onnx_path.exists() and not args.force:
        logging.info(f"{model_name}的ONNX模型已存在，跳过转换")
        return {
            "模型名称": model_name,
            "转换状态": "已存在",
            "验证状态": "未验证"
        }
    
    # 获取转换器类
    converter_class = get_converter_class(model_name)
    
    if converter_class is None:
        logging.error(f"未找到{model_name}的转换器")
        return {
            "模型名称": model_name,
            "转换状态": "失败",
            "验证状态": "未验证",
            "错误信息": "未找到转换器"
        }
    
    # 创建转换器实例
    converter = converter_class()
    
    # 转换模型
    try:
        success = converter.convert()
        
        if not success:
            logging.error(f"转换{model_name}为ONNX格式失败")
            return {
                "模型名称": model_name,
                "转换状态": "失败",
                "验证状态": "未验证",
                "错误信息": "转换失败"
            }
        
        logging.info(f"{model_name}已成功转换为ONNX格式")
        return {
            "模型名称": model_name,
            "转换状态": "成功",
            "验证状态": "成功"
        }
    except Exception as e:
        logging.error(f"转换{model_name}时发生错误: {e}")
        return {
            "模型名称": model_name,
            "转换状态": "失败",
            "验证状态": "未验证",
            "错误信息": str(e)
        }

def generate_report(results):
    """生成转换报告
    
    Args:
        results: 转换结果列表
    """
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
        success_count = df[(df["转换状态"] == "成功") & (df["验证状态"] == "成功")].shape[0]
        already_exist_count = df[df["转换状态"] == "已存在"].shape[0]
        failed_count = len(results) - success_count - already_exist_count
        
        f.write(f"## 统计\n\n")
        f.write(f"- 成功转换并验证: {success_count}/{len(results)}\n")
        f.write(f"- 已存在: {already_exist_count}/{len(results)}\n")
        f.write(f"- 失败: {failed_count}/{len(results)}\n")
        
        # 添加失败详情
        if failed_count > 0:
            f.write("\n## 失败详情\n\n")
            failed_df = df[df["转换状态"] == "失败"]
            f.write(failed_df.to_markdown(index=False))
    
    logging.info(f"报告已保存至: {md_path}")

def main():
    """主函数"""
    # 所有可用的模型
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
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='模块化ONNX转换控制脚本')
    parser.add_argument('--models', nargs='+', choices=all_models + ['all'], default=['all'],
                        help='要转换的模型，可以指定多个，或使用"all"转换所有模型')
    parser.add_argument('--force', action='store_true', help='强制重新转换已存在的模型')
    
    global args
    args = parser.parse_args()
    
    # 确定要转换的模型
    models_to_convert = all_models if 'all' in args.models else args.models
    
    logging.info(f"将转换以下模型: {', '.join(models_to_convert)}")
    
    # 转换模型
    results = []
    for model_name in models_to_convert:
        result = convert_model(model_name)
        results.append(result)
    
    # 生成报告
    generate_report(results)
    
    # 打印结果摘要
    success_count = sum(1 for r in results if r["转换状态"] == "成功" and r["验证状态"] == "成功")
    already_exist_count = sum(1 for r in results if r["转换状态"] == "已存在")
    failed_count = len(results) - success_count - already_exist_count
    
    logging.info("\n转换结果摘要:")
    logging.info(f"- 成功转换并验证: {success_count}/{len(results)}")
    logging.info(f"- 已存在: {already_exist_count}/{len(results)}")
    logging.info(f"- 失败: {failed_count}/{len(results)}")
    
    if failed_count > 0:
        logging.warning("部分模型转换失败，请查看报告了解详情")

if __name__ == "__main__":
    main()