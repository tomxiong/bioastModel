"""
将剩余的模型转换为ONNX格式
支持的模型：
- coatnet
- convnext_tiny
- vit_tiny
- airbubble_hybrid_net
- mic_mobilenetv3
- micro_vit
- enhanced_airbubble_detector
"""

import os
import sys
import torch
import logging
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import importlib

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入配置
from core.config.model_configs import MODEL_CONFIGS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ModelONNXConverter:
    def __init__(self, output_dir="deployment/onnx_models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.model_configs = MODEL_CONFIGS
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _get_checkpoint_path(self, model_name):
        """获取模型检查点路径"""
        # 尝试多种可能的路径
        possible_paths = [
            # 模型名称目录下的best_model.pth
            f"experiments/{model_name}/best_model.pth",
            # 模型名称目录下的模型名_best.pth
            f"experiments/{model_name}/{model_name}_best.pth",
            # 实验目录下的模型目录下的best_model.pth
            f"experiments/{self.model_configs[model_name].get('experiment_pattern', '')}/{model_name}/best_model.pth",
            # 实验目录下的模型名_best.pth
            f"experiments/{self.model_configs[model_name].get('experiment_pattern', '')}/{model_name}_best.pth",
            # 模型名称目录下的模型基础名_best.pth
            f"experiments/{model_name}/{model_name.split('_')[0]}_best.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        # 如果找不到，返回默认路径
        return f"experiments/{model_name}/{model_name}_best.pth"
    
    def load_model(self, model_name):
        """加载指定的模型"""
        try:
            # 获取模型配置
            model_config = self.model_configs.get(model_name, {})
            
            if not model_config:
                raise ValueError(f"未找到模型配置: {model_name}")
            
            # 获取模型类和模块路径
            class_name = model_config.get('class_name')
            module_path = model_config.get('module_path')
            
            if not class_name or not module_path:
                raise ValueError(f"模型配置缺少class_name或module_path: {model_name}")
            
            # 动态导入模块
            module = importlib.import_module(module_path)
            
            # 获取模型创建函数或类
            model_class = None
            
            # 尝试获取create_模型名函数
            create_func_name = f"create_{model_name}"
            if hasattr(module, create_func_name):
                model_class = getattr(module, create_func_name)
                model = model_class(num_classes=2)
            # 尝试获取类名
            elif hasattr(module, class_name):
                model_class = getattr(module, class_name)
                model = model_class(num_classes=2)
            else:
                raise ValueError(f"未找到模型类或创建函数: {class_name}")
            
            # 加载模型权重
            checkpoint_path = self._get_checkpoint_path(model_name)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 尝试加载模型权重
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                # 如果直接加载失败，尝试处理权重键
                state_dict = checkpoint['model_state_dict']
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            
            logger.info(f"✅ 成功加载模型权重: {checkpoint_path}")
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败 {model_name}: {str(e)}")
            return None
    
    def convert_to_onnx(self, model_name, model, opset_version=11):
        """将模型转换为ONNX格式"""
        try:
            onnx_path = os.path.join(self.output_dir, f"{model_name}.onnx")
            
            # 准备输入张量
            dummy_input = torch.randn(1, 3, 70, 70, device=self.device)
            
            # 导出为ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 检查文件大小
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
            
            logger.info(f"✅ 成功转换 {model_name} -> {onnx_path} ({file_size:.2f} MB)")
            
            return {
                "model_name": model_name,
                "onnx_path": onnx_path,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"❌ 转换失败 {model_name}: {str(e)}")
            return None
    
    def test_onnx_model(self, onnx_path):
        """测试ONNX模型"""
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # 创建推理会话
            session = ort.InferenceSession(onnx_path)
            
            # 准备输入数据
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            dummy_input = np.random.randn(1, 3, 70, 70).astype(np.float32)
            
            # 运行推理
            outputs = session.run(None, {input_name: dummy_input})
            output = outputs[0]
            
            logger.info(f"✅ ONNX模型测试成功: {onnx_path}")
            logger.info(f"   输出形状: {output.shape}")
            logger.info(f"   输出范围: [{output.min()} {output.max()}]")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX模型测试失败: {str(e)}")
            return False
    
    def convert_single_model(self, model_name):
        """转换单个模型"""
        logger.info(f"\n📦 正在处理模型: {model_name}")
        logger.info(f"   描述: {self.model_configs.get(model_name, {}).get('description', '未知')}")
        
        # 加载模型
        model = self.load_model(model_name)
        if model is None:
            return False
        
        # 转换为ONNX
        result = self.convert_to_onnx(model_name, model)
        if result is None:
            return False
        
        # 测试ONNX模型
        if self.test_onnx_model(result["onnx_path"]):
            logger.info(f"✅ 模型 {model_name} 转换并测试成功!")
            return True
        else:
            return False
    
    def convert_all_models(self, model_names):
        """转换多个模型"""
        results = {}
        
        for model_name in model_names:
            logger.info(f"\n🔄 开始转换模型: {model_name}")
            success = self.convert_single_model(model_name)
            results[model_name] = success
        
        # 打印转换结果摘要
        logger.info("\n📊 转换结果摘要:")
        for model_name, success in results.items():
            status = "✅ 成功" if success else "❌ 失败"
            logger.info(f"   {model_name}: {status}")
        
        # 计算成功率
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        logger.info(f"\n📈 总体成功率: {success_rate:.1f}% ({success_count}/{total_count})")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="将多个模型转换为ONNX格式")
    parser.add_argument("--output_dir", type=str, default="deployment/onnx_models", 
                        help="ONNX模型输出目录")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["coatnet", "convnext_tiny", "vit_tiny", 
                                "airbubble_hybrid_net", "mic_mobilenetv3", 
                                "micro_vit", "enhanced_airbubble_detector"],
                        help="要转换的模型名称列表")
    args = parser.parse_args()
    
    logger.info("🚀 开始批量转换模型")
    logger.info(f"📋 待转换模型: {', '.join(args.models)}")
    
    converter = ModelONNXConverter(output_dir=args.output_dir)
    results = converter.convert_all_models(args.models)
    
    # 检查是否所有模型都转换成功
    all_success = all(results.values())
    
    if all_success:
        logger.info("\n🎉 所有模型转换成功!")
    else:
        logger.warning("\n⚠️ 部分模型转换失败，请查看日志了解详情")
    
    logger.info(f"📁 输出目录: {converter.output_dir}")

if __name__ == "__main__":
    main()