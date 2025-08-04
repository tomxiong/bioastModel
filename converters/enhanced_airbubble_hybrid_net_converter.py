"""
增强型AirBubbleHybridNet模型转换器
使用多种转换策略和自动回退机制
"""

import os
import sys
import logging
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_onnx_converter_base import EnhancedONNXConverterBase, ConversionStrategy
from models.airbubble_hybrid_net import AirBubbleHybridNet, create_airbubble_hybrid_net

class EnhancedAirbubbleHybridNetConverter(EnhancedONNXConverterBase):
    """增强型AirBubbleHybridNet模型转换器"""
    
    def __init__(self):
        """初始化转换器"""
        super().__init__("airbubble_hybrid_net")
        self.input_shape = (3, 70, 70)  # 模型特定的输入形状
        
        # 为AirBubbleHybridNet定制的转换策略
        self.custom_strategies = [
            # 首先尝试较高版本的动态轴
            ConversionStrategy("混合网络高版本动态", 13, True, True, False),
            ConversionStrategy("混合网络中版本动态", 12, True, True, False),
            ConversionStrategy("混合网络标准版本动态", 11, True, True, False),
            
            # 然后尝试静态轴（混合网络可能对动态轴支持不好）
            ConversionStrategy("混合网络高版本静态", 13, False, True, False),
            ConversionStrategy("混合网络中版本静态", 12, False, True, False),
            ConversionStrategy("混合网络标准版本静态", 11, False, True, False),
            
            # 最后尝试较低版本和禁用常量折叠
            ConversionStrategy("混合网络低版本静态", 10, False, True, False),
            ConversionStrategy("混合网络最低版本", 9, False, False, False),
            
            # 如果还是失败，尝试详细模式进行调试
            ConversionStrategy("混合网络调试模式", 11, False, True, True),
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
            self.log_message("创建AirBubbleHybridNet模型实例...")
            
            # 尝试使用create函数
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
    
    def preprocess_state_dict_for_hybrid_net(self, state_dict: Dict) -> Dict:
        """为混合网络预处理状态字典"""
        self.log_message("预处理混合网络状态字典...")
        
        # 首先使用基类的处理方法
        processed_dict = self.process_state_dict(state_dict)
        
        # 混合网络特定的处理
        # 检查是否有特殊的键名模式
        keys = list(processed_dict.keys())
        
        # 记录一些关键信息
        self.log_message(f"状态字典包含 {len(keys)} 个键")
        if keys:
            self.log_message(f"前几个键: {keys[:5]}")
            self.log_message(f"后几个键: {keys[-5:]}")
        
        # 检查是否有不匹配的键
        # 这里可以添加更多特定于混合网络的键名处理逻辑
        
        return processed_dict
    
    def validate_model_before_conversion(self, model: torch.nn.Module) -> bool:
        """在转换前验证模型"""
        try:
            self.log_message("验证模型是否可以正常前向传播...")
            
            # 创建测试输入
            test_input = torch.randn(1, *self.input_shape)
            
            # 测试前向传播
            with torch.no_grad():
                output = model(test_input)
            
            # 检查输出
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
            
            # 检查输出是否合理
            if output.shape[0] != 1:
                self.log_message(f"警告: 批次维度不是1: {output.shape[0]}", "WARNING")
            
            if len(output.shape) < 2:
                self.log_message(f"警告: 输出维度可能不正确: {output.shape}", "WARNING")
            
            self.log_message("模型验证通过")
            return True
            
        except Exception as e:
            self.log_message(f"模型验证失败: {e}", "ERROR")
            import traceback
            self.log_message(f"详细错误: {traceback.format_exc()}", "ERROR")
            return False
    
    def convert(self) -> bool:
        """转换模型为ONNX格式
        
        Returns:
            是否成功
        """
        self.log_message(f"开始将{self.model_name}转换为ONNX格式...")
        
        # 1. 查找检查点文件
        checkpoint_path = self.find_model_checkpoint()
        if checkpoint_path is None:
            self.save_conversion_report(False, {'error': '未找到检查点文件'})
            return False
        
        # 2. 创建模型实例
        model = self.create_model_instance()
        if model is None:
            self.save_conversion_report(False, {'error': '无法创建模型实例'})
            return False
        
        # 3. 加载检查点
        try:
            self.log_message("加载检查点...")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            
            # 获取状态字典
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.log_message("使用 'model_state_dict' 键")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                self.log_message("使用 'state_dict' 键")
            else:
                state_dict = checkpoint
                self.log_message("直接使用检查点作为状态字典")
            
            # 预处理状态字典
            processed_state_dict = self.preprocess_state_dict_for_hybrid_net(state_dict)
            
            # 加载状态字典
            model.load_state_dict(processed_state_dict)
            self.log_message("检查点加载成功")
            
        except Exception as e:
            self.log_message(f"加载检查点失败: {e}", "ERROR")
            self.save_conversion_report(False, {'error': f'加载检查点失败: {e}'})
            return False
        
        # 4. 验证模型
        if not self.validate_model_before_conversion(model):
            self.save_conversion_report(False, {'error': '模型验证失败'})
            return False
        
        # 5. 使用回退机制转换
        success, conversion_info = self.convert_with_fallback(model, self.input_shape, self.custom_strategies)
        
        # 6. 保存转换报告
        self.save_conversion_report(success, conversion_info)
        
        if success:
            self.log_message(f"{self.model_name}已成功转换为ONNX格式并通过验证")
        else:
            self.log_message(f"{self.model_name}转换失败", "ERROR")
        
        return success

def main():
    """主函数"""
    converter = EnhancedAirbubbleHybridNetConverter()
    success = converter.convert()
    
    if success:
        logging.info(f"AirBubbleHybridNet模型已成功转换为ONNX格式")
    else:
        logging.error(f"AirBubbleHybridNet模型转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main()