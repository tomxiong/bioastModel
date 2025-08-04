"""
训练和转换新增模型的统一脚本
"""

import os
import sys
import logging
import subprocess
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

def run_command(command, description):
    """运行命令并记录结果"""
    logging.info(f"开始 {description}...")
    logging.info(f"执行命令: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode == 0:
            logging.info(f"✅ {description} 成功完成")
            if result.stdout:
                logging.info(f"输出: {result.stdout[-500:]}")  # 只显示最后500字符
            return True
        else:
            logging.error(f"❌ {description} 失败")
            logging.error(f"错误代码: {result.returncode}")
            if result.stderr:
                logging.error(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"❌ {description} 超时")
        return False
    except Exception as e:
        logging.error(f"❌ {description} 异常: {e}")
        return False

def check_data_availability():
    """检查数据是否可用"""
    data_dir = Path("data")
    if not data_dir.exists():
        logging.warning("数据目录不存在，将创建示例数据结构")
        # 创建基本数据结构
        (data_dir / "train").mkdir(parents=True, exist_ok=True)
        (data_dir / "val").mkdir(parents=True, exist_ok=True)
        (data_dir / "test").mkdir(parents=True, exist_ok=True)
        
        # 创建类别目录
        for split in ["train", "val", "test"]:
            (data_dir / split / "normal").mkdir(exist_ok=True)
            (data_dir / split / "abnormal").mkdir(exist_ok=True)
        
        logging.info("已创建基本数据结构，请添加实际数据文件")
        return False
    
    return True

def train_model(model_name):
    """训练指定模型"""
    script_path = f"scripts/train_{model_name}.py"
    
    if not Path(script_path).exists():
        logging.error(f"训练脚本不存在: {script_path}")
        return False
    
    command = f"python {script_path}"
    return run_command(command, f"训练 {model_name}")

def convert_model(model_name):
    """转换指定模型为ONNX"""
    script_path = f"converters/{model_name}_converter.py"
    
    if not Path(script_path).exists():
        logging.error(f"转换脚本不存在: {script_path}")
        return False
    
    command = f"python {script_path}"
    return run_command(command, f"转换 {model_name} 为ONNX")

def validate_onnx_models():
    """验证所有ONNX模型"""
    command = "python scripts/fixed_quick_validate_all_onnx.py"
    return run_command(command, "验证所有ONNX模型")

def create_summary_report(results):
    """创建总结报告"""
    report_dir = Path("reports/new_models_training")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"training_summary_{timestamp}.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 新增模型训练和转换总结报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 处理的模型\n\n")
        f.write("| 模型名称 | 训练状态 | ONNX转换状态 | 备注 |\n")
        f.write("|---------|---------|-------------|------|\n")
        
        for model_name, result in results.items():
            training_status = "✅ 成功" if result['training'] else "❌ 失败"
            conversion_status = "✅ 成功" if result['conversion'] else "❌ 失败"
            notes = result.get('notes', '')
            f.write(f"| {model_name} | {training_status} | {conversion_status} | {notes} |\n")
        
        f.write("\n## 模型特性\n\n")
        f.write("### mic_mobilenetv3\n")
        f.write("- MIC专用的MobileNetV3架构\n")
        f.write("- 集成气泡检测和浊度分析功能\n")
        f.write("- 多任务学习能力\n")
        f.write("- 参数量: ~2.5M\n\n")
        
        f.write("### micro_vit\n")
        f.write("- 微型Vision Transformer\n")
        f.write("- 专为70x70小图像优化\n")
        f.write("- 5x5超小patch size\n")
        f.write("- 浊度感知位置编码\n")
        f.write("- 参数量: ~1.8M\n\n")
        
        f.write("### airbubble_hybrid_net\n")
        f.write("- CNN-Transformer混合架构\n")
        f.write("- 专门的气泡检测和光学畸变校正\n")
        f.write("- 支持4类分类（包括气泡干扰类）\n")
        f.write("- 参数量: ~3.2M\n\n")
        
        # 统计信息
        total_models = len(results)
        successful_training = sum(1 for r in results.values() if r['training'])
        successful_conversion = sum(1 for r in results.values() if r['conversion'])
        
        f.write("## 统计信息\n\n")
        f.write(f"- 总模型数: {total_models}\n")
        f.write(f"- 训练成功: {successful_training}/{total_models}\n")
        f.write(f"- ONNX转换成功: {successful_conversion}/{total_models}\n")
        
        if successful_training == total_models and successful_conversion == total_models:
            f.write("\n✅ **所有新增模型都已成功训练和转换！**\n")
        else:
            f.write("\n⚠️ **部分模型处理失败，请检查日志信息**\n")
    
    logging.info(f"总结报告已保存至: {report_path}")
    return report_path

def main():
    """主函数"""
    logging.info("="*60)
    logging.info("开始训练和转换新增模型")
    logging.info("="*60)
    
    # 新增模型列表
    new_models = [
        'mic_mobilenetv3',
        'micro_vit', 
        'airbubble_hybrid_net'
    ]
    
    # 检查数据可用性
    if not check_data_availability():
        logging.warning("数据不可用，将跳过训练步骤")
        skip_training = True
    else:
        skip_training = False
    
    results = {}
    
    for model_name in new_models:
        logging.info(f"\n{'='*40}")
        logging.info(f"处理模型: {model_name}")
        logging.info(f"{'='*40}")
        
        results[model_name] = {
            'training': False,
            'conversion': False,
            'notes': ''
        }
        
        # 训练模型
        if not skip_training:
            if train_model(model_name):
                results[model_name]['training'] = True
                logging.info(f"✅ {model_name} 训练成功")
            else:
                results[model_name]['notes'] += '训练失败; '
                logging.error(f"❌ {model_name} 训练失败")
                # 训练失败时跳过转换
                continue
        else:
            results[model_name]['notes'] += '跳过训练(无数据); '
            logging.info(f"⏭️ 跳过 {model_name} 训练（无数据）")
        
        # 转换为ONNX
        if convert_model(model_name):
            results[model_name]['conversion'] = True
            logging.info(f"✅ {model_name} ONNX转换成功")
        else:
            results[model_name]['notes'] += 'ONNX转换失败; '
            logging.error(f"❌ {model_name} ONNX转换失败")
    
    # 验证所有ONNX模型
    logging.info(f"\n{'='*40}")
    logging.info("验证所有ONNX模型")
    logging.info(f"{'='*40}")
    
    if validate_onnx_models():
        logging.info("✅ ONNX模型验证成功")
    else:
        logging.error("❌ ONNX模型验证失败")
    
    # 创建总结报告
    report_path = create_summary_report(results)
    
    # 最终总结
    logging.info(f"\n{'='*60}")
    logging.info("处理完成总结")
    logging.info(f"{'='*60}")
    
    for model_name, result in results.items():
        status = "✅" if result['training'] and result['conversion'] else "❌"
        logging.info(f"{status} {model_name}: 训练={result['training']}, ONNX={result['conversion']}")
    
    logging.info(f"详细报告: {report_path}")
    logging.info("="*60)

if __name__ == "__main__":
    main()