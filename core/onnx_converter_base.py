"""
ONNX转换基础模块
提供共享功能，如模型检查点查找、配置加载、ONNX验证等
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

class ONNXConverterBase:
    """ONNX转换基础类"""
    
    def __init__(self, model_name):
        """初始化转换器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.onnx_dir = Path("onnx_models")
        self.onnx_dir.mkdir(exist_ok=True)
        self.onnx_path = self.onnx_dir / f"{model_name}.onnx"
    
    def find_latest_checkpoint(self):
        """查找最新的模型检查点"""
        experiments_dir = Path("experiments")
        model_dirs = list(experiments_dir.glob(f"**/{self.model_name}"))
        
        if not model_dirs:
            logging.warning(f"未找到{self.model_name}的实验目录")
            return None
        
        # 按修改时间排序，获取最新的实验目录
        latest_dir = max(model_dirs, key=os.path.getmtime)
        
        # 查找最新的检查点文件
        checkpoint_files = list(latest_dir.glob("*.pth"))
        
        if not checkpoint_files:
            logging.warning(f"未找到{self.model_name}的检查点文件")
            return None
        
        # 按修改时间排序，获取最新的检查点文件
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint
    
    def load_model_config(self):
        """加载模型配置"""
        experiments_dir = Path("experiments")
        model_dirs = list(experiments_dir.glob(f"**/{self.model_name}"))
        
        if not model_dirs:
            logging.warning(f"未找到{self.model_name}的实验目录")
            return None
        
        # 按修改时间排序，获取最新的实验目录
        latest_dir = max(model_dirs, key=os.path.getmtime)
        
        # 查找配置文件
        config_file = latest_dir / "config.json"
        
        if not config_file.exists():
            logging.warning(f"未找到{self.model_name}的配置文件")
            return None
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logging.error(f"加载{self.model_name}配置失败: {e}")
            return None
    
    def load_model(self, model_class, checkpoint_path):
        """加载模型
        
        Args:
            model_class: 模型类
            checkpoint_path: 检查点路径
            
        Returns:
            加载的模型
        """
        try:
            # 创建模型实例
            model = model_class()
            model.eval()
            
            # 加载模型权重
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
            return model
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            return None
    
    def convert_to_onnx(self, model, input_shape, opset_version=11, dynamic_axes=True):
        """将模型转换为ONNX格式
        
        Args:
            model: 模型实例
            input_shape: 输入形状，如(3, 224, 224)
            opset_version: ONNX操作集版本
            dynamic_axes: 是否使用动态轴
            
        Returns:
            是否成功
        """
        try:
            # 创建输入张量
            dummy_input = torch.randn(1, *input_shape)
            
            # 设置动态轴
            if dynamic_axes:
                dynamic_axes_dict = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            else:
                dynamic_axes_dict = None
            
            # 导出ONNX模型
            torch.onnx.export(
                model,
                dummy_input,
                self.onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes_dict
            )
            
            logging.info(f"ONNX模型已保存至: {self.onnx_path}")
            return True
        except Exception as e:
            logging.error(f"导出ONNX模型失败: {e}")
            return False
    
    def validate_onnx_model(self, input_shape):
        """验证ONNX模型
        
        Args:
            input_shape: 输入形状，如(3, 224, 224)
            
        Returns:
            是否成功
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # 加载ONNX模型
            onnx_model = onnx.load(self.onnx_path)
            
            # 检查模型
            onnx.checker.check_model(onnx_model)
            logging.info("ONNX模型检查通过")
            
            # 创建推理会话
            ort_session = ort.InferenceSession(str(self.onnx_path))
            
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
    
    def convert(self):
        """转换模型为ONNX格式
        
        此方法应由子类实现
        
        Returns:
            是否成功
        """
        raise NotImplementedError("子类必须实现此方法")