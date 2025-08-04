"""
将simplified_airbubble_detector模型转换为ONNX格式
"""

import os
import torch
import torch.onnx
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

# 导入项目模块
from models.simplified_airbubble_detector import SimplifiedAirbubbleDetector

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

def convert_to_onnx(model_name, input_shape=(3, 70, 70), opset_version=11):
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
    
    # 创建模型实例
    model = SimplifiedAirbubbleDetector()
    
    # 加载模型权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
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
        
        # 创建输入张量
        input_shape = (3, 70, 70)
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

def main():
    """主函数"""
    model_name = "simplified_airbubble_detector"
    
    # 转换模型
    success = convert_to_onnx(model_name)
    
    if not success:
        logging.error(f"转换{model_name}为ONNX格式失败")
        return
    
    # 验证ONNX模型
    success = validate_onnx_model(model_name)
    
    if not success:
        logging.error(f"验证{model_name}的ONNX模型失败")
        return
    
    logging.info(f"{model_name}已成功转换为ONNX格式并通过验证")

if __name__ == "__main__":
    main()