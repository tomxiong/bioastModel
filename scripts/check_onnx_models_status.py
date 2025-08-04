"""
检查所有ONNX模型的状态
"""

import os
import sys
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

def validate_onnx_model(model_name):
    """验证ONNX模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        是否有效
    """
    try:
        import onnx
        
        # 加载ONNX模型
        onnx_path = Path("onnx_models") / f"{model_name}.onnx"
        if not onnx_path.exists():
            return False
        
        # 检查模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        return True
    except Exception as e:
        logging.error(f"验证{model_name}的ONNX模型失败: {e}")
        return False

def check_models_status():
    """检查所有模型的状态"""
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
        valid = validate_onnx_model(model_name) if exists else False
        
        results.append({
            "模型名称": model_name,
            "ONNX模型存在": "是" if exists else "否",
            "模型大小(MB)": size if exists else None,
            "模型有效": "是" if valid else "否" if exists else "N/A"
        })
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 打印结果
    logging.info("\nONNX模型状态:")
    logging.info(f"\n{df.to_string(index=False)}")
    
    # 统计
    exists_count = df[df["ONNX模型存在"] == "是"].shape[0]
    valid_count = df[df["模型有效"] == "是"].shape[0]
    
    logging.info(f"\n统计:")
    logging.info(f"- 已存在的模型: {exists_count}/{len(all_models)}")
    logging.info(f"- 有效的模型: {valid_count}/{len(all_models)}")
    
    return df

def main():
    """主函数"""
    check_models_status()

if __name__ == "__main__":
    main()