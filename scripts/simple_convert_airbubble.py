"""
简化版AirBubbleHybridNet模型转换器
"""

import os
import sys
import logging
import torch
import onnx
import numpy as np
from pathlib import Path

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.airbubble_hybrid_net import create_airbubble_hybrid_net

def main():
    """主函数"""
    model_name = "airbubble_hybrid_net"
    input_shape = (3, 70, 70)
    
    # 确保ONNX模型目录存在
    onnx_dir = Path("onnx_models")
    onnx_dir.mkdir(exist_ok=True)
    
    onnx_path = onnx_dir / f"{model_name}.onnx"
    
    # 查找最新的检查点文件
    checkpoint_path = Path("experiments/experiment_20250803_115344/airbubble_hybrid_net/best_model.pth")
    
    if not checkpoint_path.exists():
        logging.error(f"检查点文件不存在: {checkpoint_path}")
        return
    
    logging.info(f"使用检查点文件: {checkpoint_path}")
    
    # 创建模型实例
    try:
        model = create_airbubble_hybrid_net(num_classes=2, model_size='base')
        model.eval()
        
        # 加载模型权重
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        
        # 打印检查点内容的键
        logging.info(f"检查点内容的键: {list(checkpoint.keys())}")
        
        # 检查权重键名是否匹配
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 尝试直接加载
            state_dict = checkpoint
        
        # 打印状态字典的前几个键
        keys = list(state_dict.keys())
        logging.info(f"状态字典的前5个键: {keys[:5]}")
        
        # 处理base_model前缀问题
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('base_model.'):
                new_key = key[len('base_model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 打印模型的键
        model_keys = list(model.state_dict().keys())
        logging.info(f"模型的前5个键: {model_keys[:5]}")
        
        # 尝试加载处理后的权重
        model.load_state_dict(new_state_dict)
        
        logging.info("模型权重加载成功")
    except Exception as e:
        logging.error(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建示例输入
    dummy_input = torch.randn(1, *input_shape)
    
    # 导出为ONNX格式
    try:
        # 简化模型输出，只保留分类结果
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                outputs = self.model(x)
                return outputs['classification']
        
        wrapped_model = ModelWrapper(model)
        
        # 测试包装模型
        with torch.no_grad():
            test_output = wrapped_model(dummy_input)
            logging.info(f"测试输出形状: {test_output.shape}")
        
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )
        
        logging.info(f"ONNX模型已保存至: {onnx_path}")
        
        # 验证ONNX模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("ONNX模型检查通过")
        
        logging.info(f"{model_name}已成功转换为ONNX格式")
    except Exception as e:
        logging.error(f"导出ONNX模型失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()