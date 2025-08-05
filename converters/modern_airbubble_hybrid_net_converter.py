"""
现代化AirBubbleHybridNet模型转换器
使用PyTorch 2.5+ TorchDynamo导出器和最新最佳实践
"""

import os
import sys
import logging
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.modern_onnx_converter_base import ModernONNXConverterBase, ModernConversionStrategy
from models.airbubble_hybrid_net import AirBubbleHybridNet, create_airbubble_hybrid_net

class ModernAirbubbleHybridNetConverter(ModernONNXConverterBase):
    """现代化AirBubbleHybridNet模型转换器"""
    
    def __init__(self):
        """初始化转换器"""
        super().__init__("airbubble_hybrid_net")
        self.input_shape = (3, 70, 70)  # 模型特定的输入形状
        
        # 为混合网络定制的现代化转换策略
        self.hybrid_strategies = [
            # 优先使用最新的TorchDynamo导出器
            ModernConversionStrategy("混合网络现代化动态", 18, True, True, True, False),
            ModernConversionStrategy("混合网络现代化静态", 18, False, True, True, False),
            
            # 兼容版本（针对复杂的Transformer结构）
            ModernConversionStrategy("混合网络兼容动态", 17, True, True, True, False),
            ModernConversionStrategy("混合网络兼容静态", 17, False, True, True, False),
            
            # 回退到传统方式但使用较高opset
            ModernConversionStrategy("混合网络传统高版本动态", 16, True, False, True, False),
            ModernConversionStrategy("混合网络传统高版本静态", 16, False, False, True, False),
            
            # 中等版本（针对Transformer兼容性）
            ModernConversionStrategy("混合网络传统中版本动态", 13, True, False, True, False),
            ModernConversionStrategy("混合网络传统中版本静态", 13, False, False, True, False),
            
            # 保守版本
            ModernConversionStrategy("混合网络保守版本", 11, False, False, True, False),
            
            # 调试模式
            ModernConversionStrategy("混合网络调试模式", 18, False, True, True, True),
        ]
    
    def find_model_checkpoint(self) -> Optional[Path]:
        """查找AirBubbleHybridNet的检查点文件"""
        # 首先尝试基类的方法
        checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is not None:
            return checkpoint_path
        
        # 如果没找到，尝试一些已知的路径
        known_paths = [
            "experiments/experiment_20250803_115344/airbubble_hybrid_net/best_model.pth",
            "experiments/experiment_20250803_115344/airbubble_hybrid_net/final_model.pth",
            "experiments/airbubble_hybrid_net/best_model.pth",
        ]
        
        for path_str in known_paths:
            path = Path(path_str)
            if path.exists():
                self.log_message(f"使用已知路径找到检查点: {path}")
                return path
        
        self.log_message("未找到任何检查点文件", "ERROR")
        return None
    
    def create_model_instance(self) -> Optional[torch.nn.Module]:
        """创建AirBubbleHybridNet模型实例"""
        try:
            self.log_message("创建现代化AirBubbleHybridNet模型实例...")
            
            # 首先尝试使用create函数
            model = create_airbubble_hybrid_net(num_classes=2, model_size='base')
            model.eval()
            
            self.log_message("模型实例创建成功")
            return model
        except Exception as e:
            self.log_message(f"使用create_airbubble_hybrid_net创建模型失败: {e}", "ERROR")
            
            # 尝试直接使用类
            try:
                self.log_message("尝试直接使用AirBubbleHybridNet类...")
                model = AirBubbleHybridNet(num_classes=2)
                model.eval()
                
                self.log_message("使用类创建模型成功")
                return model
            except Exception as e2:
                self.log_message(f"使用AirBubbleHybridNet类创建模型也失败: {e2}", "ERROR")
                return None
    
    def validate_model_before_conversion(self, model: torch.nn.Module) -> bool:
        """在转换前验证模型"""
        try:
            self.log_message("验证混合网络模型是否可以正常前向传播...")
            
            # 创建测试输入
            test_input = torch.randn(1, *self.input_shape)
            
            # 测试前向传播
            with torch.no_grad():
                output = model(test_input)
            
            # 处理混合网络的复杂输出
            if isinstance(output, dict):
                self.log_message(f"模型输出是字典，键: {list(output.keys())}")
                if 'classification' in output:
                    output = output['classification']
                    self.log_message("使用 'classification' 键的输出")
                else:
                    # 使用第一个输出
                    output = list(output.values())[0]
                    self.log_message("使用第一个输出值")
            
            self.log_message(f"模型输出形状: {output.shape}")
            self.log_message(f"模型输出范围: [{output.min():.6f}, {output.max():.6f}]")
            
            # 检查输出合理性
            if torch.any(torch.isnan(output)):
                self.log_message("警告: 模型输出包含NaN值", "WARNING")
                return False
            
            if torch.any(torch.isinf(output)):
                self.log_message("警告: 模型输出包含无穷大值", "WARNING")
                return False
            
            self.log_message("混合网络模型验证通过")
            return True
            
        except Exception as e:
            self.log_message(f"模型验证失败: {e}", "ERROR")
            import traceback
            self.log_message(f"详细错误: {traceback.format_exc()}", "ERROR")
            return False
    
    def create_onnx_compatible_wrapper(self, model: torch.nn.Module) -> torch.nn.Module:
        """创建ONNX兼容的模型包装器"""
        
        class AirBubbleHybridNetONNXWrapper(torch.nn.Module):
            """ONNX兼容的AirBubbleHybridNet包装器"""
            
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                """前向传播，确保输出格式ONNX兼容"""
                outputs = self.model(x)
                
                # 如果是字典输出，只返回分类结果
                if isinstance(outputs, dict):
                    if 'classification' in outputs:
                        return outputs['classification']
                    else:
                        # 返回第一个张量输出
                        for key, value in outputs.items():
                            if isinstance(value, torch.Tensor):
                                return value
                        # 如果没有找到张量，返回第一个值
                        return list(outputs.values())[0]
                
                # 如果是元组或列表，返回第一个元素
                if isinstance(outputs, (tuple, list)):
                    return outputs[0]
                
                # 直接返回张量
                return outputs
        
        wrapped_model = AirBubbleHybridNetONNXWrapper(model)
        wrapped_model.eval()
        
        self.log_message("创建ONNX兼容包装器成功")
        return wrapped_model
    
    def convert(self) -> bool:
        """使用现代化方法转换模型为ONNX格式
        
        Returns:
            是否成功
        """
        self.log_message(f"开始使用现代化方法将{self.model_name}转换为ONNX格式...")
        
        # 1. 查找检查点文件
        checkpoint_path = self.find_model_checkpoint()
        if checkpoint_path is None:
            self.save_modern_conversion_report(False, {'error': '未找到检查点文件'})
            return False
        
        # 2. 创建模型实例
        model = self.create_model_instance()
        if model is None:
            self.save_modern_conversion_report(False, {'error': '无法创建模型实例'})
            return False
        
        # 3. 安全加载检查点
        loaded_model = self.load_model_safely(lambda: model, checkpoint_path)
        if loaded_model is None:
            self.save_modern_conversion_report(False, {'error': '无法加载模型检查点'})
            return False
        
        # 4. 验证模型
        if not self.validate_model_before_conversion(loaded_model):
            self.save_modern_conversion_report(False, {'error': '模型验证失败'})
            return False
        
        # 5. 创建ONNX兼容包装器
        wrapped_model = self.create_onnx_compatible_wrapper(loaded_model)
        
        # 6. 验证包装器
        if not self.validate_model_before_conversion(wrapped_model):
            self.save_modern_conversion_report(False, {'error': '包装器验证失败'})
            return False
        
        # 7. 使用现代化回退机制转换
        success, conversion_info = self.convert_with_modern_fallback(wrapped_model, self.input_shape, self.hybrid_strategies)
        
        # 8. 保存现代化转换报告
        self.save_modern_conversion_report(success, conversion_info)
        
        if success:
            self.log_message(f"{self.model_name}已成功使用现代化方法转换为ONNX格式并通过验证")
        else:
            self.log_message(f"{self.model_name}现代化转换失败", "ERROR")
        
        return success

def main():
    """主函数"""
    converter = ModernAirbubbleHybridNetConverter()
    success = converter.convert()
    
    if success:
        logging.info(f"AirBubbleHybridNet模型已成功使用现代化方法转换为ONNX格式")
    else:
        logging.error(f"AirBubbleHybridNet模型现代化转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main()