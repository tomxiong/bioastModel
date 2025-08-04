"""
单独转换指定模型为ONNX格式
"""

import os
import sys
import argparse
import logging
import importlib
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

def convert_model(model_name, force=False):
    """转换指定模型
    
    Args:
        model_name: 模型名称
        force: 是否强制重新转换
        
    Returns:
        是否成功
    """
    logging.info(f"\n{'='*50}")
    logging.info(f"处理模型: {model_name}")
    
    # 检查ONNX文件是否已存在
    onnx_path = Path("onnx_models") / f"{model_name}.onnx"
    if onnx_path.exists() and not force:
        logging.info(f"{model_name}的ONNX模型已存在，跳过转换")
        return True
    
    # 获取转换器类
    converter_class = get_converter_class(model_name)
    
    if converter_class is None:
        logging.error(f"未找到{model_name}的转换器")
        return False
    
    # 创建转换器实例
    converter = converter_class()
    
    # 转换模型
    try:
        success = converter.convert()
        
        if not success:
            logging.error(f"转换{model_name}为ONNX格式失败")
            return False
        
        logging.info(f"{model_name}已成功转换为ONNX格式")
        return True
    except Exception as e:
        logging.error(f"转换{model_name}时发生错误: {e}")
        return False

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
    parser = argparse.ArgumentParser(description='单独转换指定模型为ONNX格式')
    parser.add_argument('model', choices=all_models, help='要转换的模型名称')
    parser.add_argument('--force', action='store_true', help='强制重新转换已存在的模型')
    
    args = parser.parse_args()
    
    # 转换模型
    success = convert_model(args.model, args.force)
    
    if success:
        logging.info(f"{args.model}模型已成功转换为ONNX格式")
    else:
        logging.error(f"{args.model}模型转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main()