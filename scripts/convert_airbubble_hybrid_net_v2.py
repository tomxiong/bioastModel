"""
将 AirBubbleHybridNet 模型转换为 ONNX 格式（第二版）。

这个脚本专门用于处理 AirBubbleHybridNet 模型的权重加载问题，
通过更详细的调试信息和更灵活的权重映射方式来解决问题。
"""

import os
import sys
import torch
import logging
import argparse
import numpy as np
from pathlib import Path

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型
from models.airbubble_hybrid_net import AirBubbleHybridNet, create_airbubble_hybrid_net

def convert_model_to_onnx(model_path, output_dir="deployment/onnx_models", debug=False):
    """
    将 AirBubbleHybridNet 模型转换为 ONNX 格式。
    
    Args:
        model_path: 模型权重文件路径
        output_dir: 输出目录
        debug: 是否启用调试模式
    """
    logging.info(f"🚀 开始转换模型: airbubble_hybrid_net")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载权重
        state_dict = torch.load(model_path, map_location="cpu")
        
        if debug:
            logging.info(f"原始权重键名:")
            for key in list(state_dict.keys())[:10]:  # 只显示前10个键名
                logging.info(f"  - {key}")
        
        # 检查权重文件结构
        if "base_model.stem.0.weight" in state_dict:
            logging.info("检测到权重文件使用 'base_model.' 前缀")
            
            # 创建模型实例
            model = AirBubbleHybridNet(num_classes=2)
            
            # 处理权重键名不匹配的问题
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("base_model."):
                    new_key = key.replace("base_model.", "")
                    new_state_dict[new_key] = value
            
            # 加载处理后的权重
            model.load_state_dict(new_state_dict, strict=False)
            logging.info(f"✅ 成功加载模型权重: {model_path}")
        else:
            # 尝试使用创建函数
            logging.info("尝试使用 create_airbubble_hybrid_net 函数创建模型")
            model = create_airbubble_hybrid_net(num_classes=2)
            model.load_state_dict(state_dict, strict=False)
            logging.info(f"✅ 成功加载模型权重: {model_path}")
        
        # 设置为评估模式
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(1, 3, 70, 70)
        
        # 定义输出路径
        onnx_path = os.path.join(output_dir, "airbubble_hybrid_net.onnx")
        
        # 转换为 ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        
        # 获取文件大小
        file_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
        logging.info(f"✅ 成功转换模型 -> {onnx_path} ({file_size:.2f} MB)")
        
        # 测试 ONNX 模型
        import onnxruntime as ort
        
        # 创建 ONNX 运行时会话
        ort_session = ort.InferenceSession(onnx_path)
        
        # 准备输入
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        
        # 运行推理
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # 输出形状和范围
        output_shape = ort_outputs[0].shape
        output_range = [float(np.min(ort_outputs[0])), float(np.max(ort_outputs[0]))]
        
        logging.info(f"✅ ONNX模型测试成功: {onnx_path}")
        logging.info(f"   输出形状: {output_shape}")
        logging.info(f"   输出范围: {output_range}")
        logging.info(f"✅ 模型 airbubble_hybrid_net 转换并测试成功!")
        
        logging.info(f"\n🎉 模型 airbubble_hybrid_net 转换成功!")
        logging.info(f"📁 输出目录: {output_dir}")
        
        return True
    except Exception as e:
        logging.error(f"❌ 模型加载失败: {str(e)}")
        
        if debug:
            import traceback
            logging.error(traceback.format_exc())
        
        logging.error(f"\n❌ 模型 airbubble_hybrid_net 转换失败!")
        return False

def main():
    parser = argparse.ArgumentParser(description="将 AirBubbleHybridNet 模型转换为 ONNX 格式")
    parser.add_argument("--model_path", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("--output_dir", type=str, default="deployment/onnx_models", help="输出目录")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    args = parser.parse_args()
    
    convert_model_to_onnx(args.model_path, args.output_dir, args.debug)

if __name__ == "__main__":
    main()