"""
增强型ONNX转换控制脚本
支持单个或批量模型转换，使用增强型转换器和多种转换策略
"""

import os
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import importlib

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_enhanced_converter_class(model_name):
    """获取增强型模型转换器类
    
    Args:
        model_name: 模型名称
        
    Returns:
        增强型模型转换器类
    """
    try:
        # 首先尝试增强型转换器
        module_name = f"converters.enhanced_{model_name.lower()}_converter"
        module = importlib.import_module(module_name)
        
        # 获取增强型转换器类名
        class_name = 'Enhanced' + ''.join(word.capitalize() for word in model_name.split('_')) + 'Converter'
        
        # 获取转换器类
        converter_class = getattr(module, class_name)
        logging.info(f"找到增强型转换器: {class_name}")
        return converter_class
    except (ImportError, AttributeError) as e:
        logging.warning(f"未找到增强型转换器，尝试标准转换器: {e}")
        
        # 回退到标准转换器
        try:
            module_name = f"converters.{model_name.lower()}_converter"
            module = importlib.import_module(module_name)
            
            # 获取标准转换器类名
            class_name = ''.join(word.capitalize() for word in model_name.split('_')) + 'Converter'
            
            # 获取转换器类
            converter_class = getattr(module, class_name)
            logging.info(f"找到标准转换器: {class_name}")
            return converter_class
        except (ImportError, AttributeError) as e2:
            logging.error(f"获取{model_name}转换器类失败: {e2}")
            return None

def convert_model(model_name, force=False):
    """转换指定模型
    
    Args:
        model_name: 模型名称
        force: 是否强制重新转换
        
    Returns:
        转换结果字典
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"开始处理模型: {model_name}")
    
    # 检查ONNX文件是否已存在
    onnx_path = Path("onnx_models") / f"{model_name}.onnx"
    if onnx_path.exists() and not force:
        logging.info(f"{model_name}的ONNX模型已存在，跳过转换")
        return {
            "模型名称": model_name,
            "转换状态": "已存在",
            "验证状态": "未验证",
            "转换时间": "N/A",
            "模型大小(MB)": round(onnx_path.stat().st_size / (1024 * 1024), 2) if onnx_path.exists() else None
        }
    
    # 获取转换器类
    converter_class = get_enhanced_converter_class(model_name)
    
    if converter_class is None:
        logging.error(f"未找到{model_name}的转换器")
        return {
            "模型名称": model_name,
            "转换状态": "失败",
            "验证状态": "未验证",
            "错误信息": "未找到转换器",
            "转换时间": "N/A",
            "模型大小(MB)": None
        }
    
    # 创建转换器实例
    try:
        converter = converter_class()
        logging.info(f"创建转换器实例成功: {converter_class.__name__}")
    except Exception as e:
        logging.error(f"创建转换器实例失败: {e}")
        return {
            "模型名称": model_name,
            "转换状态": "失败",
            "验证状态": "未验证",
            "错误信息": f"创建转换器失败: {e}",
            "转换时间": "N/A",
            "模型大小(MB)": None
        }
    
    # 转换模型
    start_time = datetime.now()
    try:
        success = converter.convert()
        end_time = datetime.now()
        conversion_time = (end_time - start_time).total_seconds()
        
        if not success:
            logging.error(f"转换{model_name}为ONNX格式失败")
            return {
                "模型名称": model_name,
                "转换状态": "失败",
                "验证状态": "未验证",
                "错误信息": "转换失败",
                "转换时间": f"{conversion_time:.2f}秒",
                "模型大小(MB)": None
            }
        
        # 获取模型大小
        model_size = round(onnx_path.stat().st_size / (1024 * 1024), 2) if onnx_path.exists() else None
        
        logging.info(f"{model_name}已成功转换为ONNX格式")
        return {
            "模型名称": model_name,
            "转换状态": "成功",
            "验证状态": "成功",
            "转换时间": f"{conversion_time:.2f}秒",
            "模型大小(MB)": model_size
        }
    except Exception as e:
        end_time = datetime.now()
        conversion_time = (end_time - start_time).total_seconds()
        logging.error(f"转换{model_name}时发生错误: {e}")
        return {
            "模型名称": model_name,
            "转换状态": "失败",
            "验证状态": "未验证",
            "错误信息": str(e),
            "转换时间": f"{conversion_time:.2f}秒",
            "模型大小(MB)": None
        }

def generate_comprehensive_report(results):
    """生成综合转换报告
    
    Args:
        results: 转换结果列表
    """
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 创建报告目录
    report_dir = Path("reports/enhanced_onnx_conversion")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV报告
    csv_path = report_dir / f"enhanced_conversion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # 生成Markdown报告
    md_path = report_dir / f"enhanced_conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 增强型ONNX模型转换报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 转换结果表格
        f.write("## 转换结果\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # 统计信息
        success_count = df[(df["转换状态"] == "成功") & (df["验证状态"] == "成功")].shape[0]
        already_exist_count = df[df["转换状态"] == "已存在"].shape[0]
        failed_count = len(results) - success_count - already_exist_count
        
        f.write(f"## 统计信息\n\n")
        f.write(f"- 总模型数: {len(results)}\n")
        f.write(f"- 成功转换并验证: {success_count}\n")
        f.write(f"- 已存在: {already_exist_count}\n")
        f.write(f"- 转换失败: {failed_count}\n")
        f.write(f"- 成功率: {((success_count + already_exist_count) / len(results) * 100):.1f}%\n\n")
        
        # 模型大小统计
        size_data = df[df["模型大小(MB)"].notna()]["模型大小(MB)"]
        if not size_data.empty:
            f.write(f"## 模型大小统计\n\n")
            f.write(f"- 平均大小: {size_data.mean():.2f} MB\n")
            f.write(f"- 最大模型: {size_data.max():.2f} MB\n")
            f.write(f"- 最小模型: {size_data.min():.2f} MB\n")
            f.write(f"- 总大小: {size_data.sum():.2f} MB\n\n")
        
        # 失败详情
        if failed_count > 0:
            f.write("## 失败详情\n\n")
            failed_df = df[df["转换状态"] == "失败"]
            for _, row in failed_df.iterrows():
                f.write(f"### {row['模型名称']}\n")
                f.write(f"- 错误信息: {row.get('错误信息', 'N/A')}\n")
                f.write(f"- 转换时间: {row.get('转换时间', 'N/A')}\n\n")
        
        # 转换时间分析
        time_data = df[df["转换时间"] != "N/A"]["转换时间"]
        if not time_data.empty:
            # 提取数字部分
            time_values = []
            for time_str in time_data:
                try:
                    if isinstance(time_str, str) and "秒" in time_str:
                        time_values.append(float(time_str.replace("秒", "")))
                except:
                    pass
            
            if time_values:
                f.write(f"## 转换时间分析\n\n")
                f.write(f"- 平均转换时间: {sum(time_values)/len(time_values):.2f} 秒\n")
                f.write(f"- 最长转换时间: {max(time_values):.2f} 秒\n")
                f.write(f"- 最短转换时间: {min(time_values):.2f} 秒\n\n")
        
        # 建议和后续步骤
        f.write("## 建议和后续步骤\n\n")
        if failed_count > 0:
            f.write("1. 对于转换失败的模型，建议检查模型架构和权重文件\n")
            f.write("2. 可以尝试使用不同的转换策略或降低ONNX操作集版本\n")
            f.write("3. 检查模型的特殊层是否与ONNX兼容\n\n")
        
        if success_count > 0:
            f.write("4. 对于成功转换的模型，建议进行性能验证\n")
            f.write("5. 使用真实数据测试ONNX模型的推理性能\n")
            f.write("6. 比较原始PyTorch模型和ONNX模型的输出一致性\n\n")
    
    logging.info(f"综合报告已保存至: {md_path}")
    return md_path

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
    parser = argparse.ArgumentParser(description='增强型ONNX转换控制脚本')
    parser.add_argument('--models', nargs='+', choices=all_models + ['all'], default=['airbubble_hybrid_net'],
                        help='要转换的模型，可以指定多个，或使用"all"转换所有模型')
    parser.add_argument('--force', action='store_true', help='强制重新转换已存在的模型')
    parser.add_argument('--priority', choices=all_models, help='优先处理的模型')
    
    args = parser.parse_args()
    
    # 确定要转换的模型
    models_to_convert = all_models if 'all' in args.models else args.models
    
    # 如果指定了优先模型，将其移到列表前面
    if args.priority and args.priority in models_to_convert:
        models_to_convert.remove(args.priority)
        models_to_convert.insert(0, args.priority)
    
    logging.info(f"将转换以下模型: {', '.join(models_to_convert)}")
    if args.priority:
        logging.info(f"优先处理模型: {args.priority}")
    
    # 转换模型
    results = []
    for model_name in models_to_convert:
        result = convert_model(model_name, args.force)
        results.append(result)
        
        # 如果是优先模型且转换失败，询问是否继续
        if model_name == args.priority and result["转换状态"] == "失败":
            logging.warning(f"优先模型 {model_name} 转换失败")
    
    # 生成综合报告
    report_path = generate_comprehensive_report(results)
    
    # 打印结果摘要
    success_count = sum(1 for r in results if r["转换状态"] == "成功" and r["验证状态"] == "成功")
    already_exist_count = sum(1 for r in results if r["转换状态"] == "已存在")
    failed_count = len(results) - success_count - already_exist_count
    
    logging.info(f"\n{'='*60}")
    logging.info("转换结果摘要:")
    logging.info(f"- 总模型数: {len(results)}")
    logging.info(f"- 成功转换并验证: {success_count}")
    logging.info(f"- 已存在: {already_exist_count}")
    logging.info(f"- 转换失败: {failed_count}")
    logging.info(f"- 成功率: {((success_count + already_exist_count) / len(results) * 100):.1f}%")
    logging.info(f"- 详细报告: {report_path}")
    
    if failed_count > 0:
        logging.warning("部分模型转换失败，请查看详细报告了解具体原因")
        return 1
    else:
        logging.info("所有模型转换成功！")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)