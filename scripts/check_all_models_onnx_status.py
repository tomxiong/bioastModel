"""
检查所有模型的ONNX转换状态
"""

import os
import logging
from pathlib import Path
import json
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def check_onnx_models():
    """检查ONNX模型的状态"""
    logging.info("检查ONNX模型状态...")
    
    # 模型列表
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
    
    # 检查ONNX目录
    onnx_dir = Path("onnx_models")
    if not onnx_dir.exists():
        logging.warning("ONNX目录不存在")
        onnx_dir.mkdir(exist_ok=True)
    
    # 检查每个模型的ONNX文件
    results = []
    for model_name in model_names:
        onnx_path = onnx_dir / f"{model_name}.onnx"
        status = "已转换" if onnx_path.exists() else "未转换"
        
        # 检查模型大小
        size_mb = round(onnx_path.stat().st_size / (1024 * 1024), 2) if onnx_path.exists() else 0
        
        results.append({
            "模型名称": model_name,
            "ONNX状态": status,
            "大小 (MB)": size_mb
        })
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 打印结果
    logging.info("\nONNX模型状态:")
    logging.info(f"\n{df.to_string(index=False)}")
    
    # 统计已转换和未转换的模型数量
    converted_count = df[df["ONNX状态"] == "已转换"].shape[0]
    not_converted_count = df[df["ONNX状态"] == "未转换"].shape[0]
    
    logging.info(f"\n已转换模型: {converted_count}/{len(model_names)}")
    logging.info(f"未转换模型: {not_converted_count}/{len(model_names)}")
    
    # 保存结果
    os.makedirs("reports/onnx_status", exist_ok=True)
    df.to_csv("reports/onnx_status/onnx_models_status.csv", index=False)
    
    # 生成Markdown报告
    with open("reports/onnx_status/onnx_models_status.md", "w", encoding="utf-8") as f:
        f.write("# ONNX模型转换状态报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 模型状态\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        f.write(f"## 统计\n\n")
        f.write(f"- 已转换模型: {converted_count}/{len(model_names)}\n")
        f.write(f"- 未转换模型: {not_converted_count}/{len(model_names)}\n")
    
    logging.info(f"报告已保存至: reports/onnx_status/onnx_models_status.md")
    
    return df

def main():
    """主函数"""
    check_onnx_models()

if __name__ == "__main__":
    main()