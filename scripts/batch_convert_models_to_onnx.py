"""
批量将所有模型转换为ONNX格式
"""

import os
import torch
import torch.onnx
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import importlib
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def find_latest_checkpoint(model_name):
    """查找最新的模型检查点"""
    experiments_dir = Path("experiments")
    model_dirs = list(experiments_dir.glob(f"**/{model_name}"))
    
    if not model_dirs:
        logging.warning(f"未找到{model_name}的实验目录")
        return None
    
    # 按修改时间排序，获取最新的实验目录
    latest_dir = max(model_dirs, key=os.path.getmtime)
    
    # 查找最新的检查点文件
    checkpoint_files = list(latest_dir.glob("*.pth"))
    
    if not checkpoint_files:
        logging.warning(f"未找到{model_name}的检查点文件")
        return None
    
    # 按修改时间排序，获取最新的检查点文件
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

def load_model_config(model_name):
    """加载模型配置"""
    experiments_dir = Path("experiments")
    model_dirs = list(experiments_dir.glob(f"**/{model_name}"))
    
    if not model_dirs:
        logging.warning(f"未找到{model_name}的实验目录")
        return None
    
    # 按修改时间排序，获取最新的实验目录
    latest_dir = max(model_dirs, key=os.path.getmtime)
    
    # 查找配置文件
    config_file = latest_dir / "config.json"
    
    if not config_file.exists():
        logging.warning(f"未找到{model_name}的配置文件")
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"加载{model_name}配置失败: {e}")
        return None

def get_model_class(model_name):
    """获取模型类"""
    model_mapping = {
        'simplified_airbubble_detector': ('models.simplified_airbubble_detector', 'SimplifiedAirbubbleDetector'),
        'efficientnet_b0': ('models.efficientnet_b0', 'EfficientNetB0'),
        'resnet18_improved': ('models.resnet18_improved', 'ResNet18Improved'),
        'convnext_tiny': ('models.convnext_tiny', 'ConvNextTiny'),
        'coatnet': ('models.coatnet', 'CoAtNet'),
        'vit_tiny': ('models.vit_tiny', 'ViTTiny'),
        'mic_mobilenetv3': ('models.mic_mobilenetv3', 'MICMobileNetV3'),
        'micro_vit': ('models.micro_vit', 'MicroViT'),
        'airbubble_hybrid_net': ('models.airbubble_hybrid_net', 'AirbubbleHybridNet')
    }
    
    if model_name not in model_mapping:
        logging.error(f"未知的模型名称: {model_name}")
        return None, None
    
    module_name, class_name = model_mapping[model_name]
    
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        return model_class
    except Exception as e:
        logging.error(f"导入模型类失败: {e}")
        return None

def get_model_input_shape(model_name):
    """获取模型输入形状"""
    input_shapes = {
        'simplified_airbubble_detector': (3, 70, 70),
        'efficientnet_b0': (3, 224, 224),
        'resnet18_improved': (3, 224, 224),
        'convnext_tiny': (3, 224, 224),
        'coatnet': (3, 224, 224),
        'vit_tiny': (3, 224, 224),
        'mic_mobilenetv3': (3, 224, 224),
        'micro_vit': (3, 224, 224),
        'airbubble_hybrid_net': (3, 224, 224)
    }
    
    return input_shapes.get(model_name, (3, 224, 224))

def convert_to_onnx(model_name, opset_version=11):
    """将模型转换为ONNX格式"""
    logging.info(f"开始将{model_name}转换为ONNX格式...")
    
    # 查找最新的检查点文件
    checkpoint_path = find_latest_checkpoint(model_name)
    
    if checkpoint_path is None:
        return False
    
    logging.info(f"找到最新的检查点文件: {checkpoint_path}")
    
    # 加载模型配置
    config = load_model_config(model_name)
    
    if config is None:
        return False
    
    # 获取模型类
    model_class = get_model_class(model_name)
    
    if model_class is None:
        return False
    
    # 获取模型输入形状
    input_shape = get_model_input_shape(model_name)
    
    # 创建模型实例
    try:
        model = model_class()
        model.eval()
    except Exception as e:
        logging.error(f"创建模型实例失败: {e}")
        return False
    
    # 加载模型权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # 检查权重键名是否匹配
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 尝试直接加载
            model.load_state_dict(checkpoint)
        
        logging.info("模型权重加载成功")
    except Exception as e:
        logging.error(f"加载模型权重失败: {e}")
        return False
    
    # 创建输入张量
    dummy_input = torch.randn(1, *input_shape)
    
    # 创建ONNX目录
    onnx_dir = Path("onnx_models")
    onnx_dir.mkdir(exist_ok=True)
    
    # 设置ONNX文件路径
    onnx_path = onnx_dir / f"{model_name}.onnx"
    
    # 导出ONNX模型
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logging.info(f"ONNX模型已保存至: {onnx_path}")
        return True
    except Exception as e:
        logging.error(f"导出ONNX模型失败: {e}")
        return False

def validate_onnx_model(model_name):
    """验证ONNX模型"""
    logging.info(f"验证{model_name}的ONNX模型...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        # 加载ONNX模型
        onnx_path = Path("onnx_models") / f"{model_name}.onnx"
        onnx_model = onnx.load(onnx_path)
        
        # 检查模型
        onnx.checker.check_model(onnx_model)
        logging.info("ONNX模型检查通过")
        
        # 创建推理会话
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # 获取模型输入形状
        input_shape = get_model_input_shape(model_name)
        
        # 创建输入张量
        dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
        
        # 运行推理
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        logging.info("ONNX模型推理测试通过")
        logging.info(f"输出形状: {ort_outputs[0].shape}")
        
        return True
    except Exception as e:
        logging.error(f"验证ONNX模型失败: {e}")
        return False

def batch_convert_models():
    """批量转换模型"""
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
    
    results = []
    
    for model_name in model_names:
        logging.info(f"\n{'='*50}")
        logging.info(f"处理模型: {model_name}")
        
        # 检查ONNX文件是否已存在
        onnx_path = Path("onnx_models") / f"{model_name}.onnx"
        if onnx_path.exists():
            logging.info(f"{model_name}的ONNX模型已存在，跳过转换")
            results.append({
                "模型名称": model_name,
                "转换状态": "已存在",
                "验证状态": "未验证"
            })
            continue
        
        # 转换模型
        convert_success = convert_to_onnx(model_name)
        
        if not convert_success:
            logging.error(f"转换{model_name}为ONNX格式失败")
            results.append({
                "模型名称": model_name,
                "转换状态": "失败",
                "验证状态": "未验证"
            })
            continue
        
        # 验证ONNX模型
        validate_success = validate_onnx_model(model_name)
        
        if not validate_success:
            logging.error(f"验证{model_name}的ONNX模型失败")
            results.append({
                "模型名称": model_name,
                "转换状态": "成功",
                "验证状态": "失败"
            })
            continue
        
        logging.info(f"{model_name}已成功转换为ONNX格式并通过验证")
        results.append({
            "模型名称": model_name,
            "转换状态": "成功",
            "验证状态": "成功"
        })
    
    # 创建结果DataFrame
    import pandas as pd
    df = pd.DataFrame(results)
    
    # 打印结果
    logging.info("\n转换结果:")
    logging.info(f"\n{df.to_string(index=False)}")
    
    # 保存结果
    os.makedirs("reports/onnx_conversion", exist_ok=True)
    df.to_csv("reports/onnx_conversion/conversion_results.csv", index=False)
    
    # 生成Markdown报告
    with open("reports/onnx_conversion/conversion_report.md", "w", encoding="utf-8") as f:
        f.write("# ONNX模型转换报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 转换结果\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # 统计成功和失败的数量
        success_count = df[(df["转换状态"] == "成功") & (df["验证状态"] == "成功")].shape[0]
        already_exist_count = df[df["转换状态"] == "已存在"].shape[0]
        failed_count = len(model_names) - success_count - already_exist_count
        
        f.write(f"## 统计\n\n")
        f.write(f"- 成功转换并验证: {success_count}/{len(model_names)}\n")
        f.write(f"- 已存在: {already_exist_count}/{len(model_names)}\n")
        f.write(f"- 失败: {failed_count}/{len(model_names)}\n")
    
    logging.info(f"报告已保存至: reports/onnx_conversion/conversion_report.md")
    
    return df

def main():
    """主函数"""
    batch_convert_models()

if __name__ == "__main__":
    main()