"""
现代化ResNet18Improved模型转换器（测试版本）
用于测试现代化ONNX转换流程
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
from models.resnet_improved import create_resnet18_improved

class ModernResNet18ImprovedConverter(ModernONNXConverterBase):
    """现代化ResNet18Improved模型转换器"""
    
    def __init__(self):
        """初始化转换器"""
        super().__init__("resnet18_improved")
        self.input_shape = (3, 70, 70)  # 模型特定的输入形状
        
        # ResNet18的现代化转换策略（相对简单的CNN，应该转换成功率高）
        self.resnet_strategies = [
            # 优先使用最新的TorchDynamo导出器
            ModernConversionStrategy("ResNet18现代化动态", 18, True, True, True, False),
            ModernConversionStrategy("ResNet18现代化静态", 18, False, True, True, False),
            
            # 兼容版本
            ModernConversionStrategy("ResNet18兼容动态", 17, True, True, True, False),
            ModernConversionStrategy("ResNet18兼容静态", 17, False, True, True, False),
            
            # 传统高版本
            ModernConversionStrategy("ResNet18传统高版本动态", 16, True, False, True, False),
            ModernConversionStrategy("ResNet18传统高版本静态", 16, False, False, True, False),
            
            # 传统中版本
            ModernConversionStrategy("ResNet18传统中版本动态", 13, True, False, True, False),
            ModernConversionStrategy("ResNet18传统中版本静态", 13, False, False, True, False),
            
            # 保守版本
            ModernConversionStrategy("ResNet18保守版本", 11, False, False, True, False),
            
            # 调试模式
            ModernConversionStrategy("ResNet18调试模式", 18, False, True, True, True),
        ]
    
    def find_model_checkpoint(self) -> Optional[Path]:
        """查找ResNet18Improved的检查点文件"""
        # 首先尝试基类的方法
        checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is not None:
            return checkpoint_path
        
        # 尝试已知路径
        known_paths = [
            "experiments/experiment_20250802_164948/resnet18_improved/best_model.pth",
        ]
        
        for path_str in known_paths:
            path = Path(path_str)
            if path.exists():
                self.log_message(f"使用已知路径找到检查点: {path}")
                return path
        
        self.log_message("未找到任何检查点文件", "ERROR")
        return None
    
    def create_model_instance(self) -> Optional[torch.nn.Module]:
        """创建ResNet18Improved模型实例"""
        try:
            self.log_message("创建ResNet18Improved模型实例...")
            
            # 使用create函数，通常ResNet模型比较简单
            model = create_resnet18_improved(num_classes=2)
            model.eval()
            
            self.log_message("ResNet18Improved模型实例创建成功")
            return model
        except Exception as e:
            self.log_message(f"创建ResNet18Improved模型失败: {e}", "ERROR")
            return None
    
    def validate_model_before_conversion(self, model: torch.nn.Module) -> bool:
        """在转换前验证模型"""
        try:
            self.log_message("验证ResNet18Improved模型是否可以正常前向传播...")
            
            # 创建测试输入
            test_input = torch.randn(1, *self.input_shape)
            
            # 测试前向传播
            with torch.no_grad():
                output = model(test_input)
            
            self.log_message(f"模型输出形状: {output.shape}")
            self.log_message(f"模型输出范围: [{output.min():.6f}, {output.max():.6f}]")
            
            # 检查输出合理性
            if torch.any(torch.isnan(output)):
                self.log_message("警告: 模型输出包含NaN值", "WARNING")
                return False
            
            if torch.any(torch.isinf(output)):
                self.log_message("警告: 模型输出包含无穷大值", "WARNING")
                return False
            
            # ResNet通常输出logits，检查形状是否正确
            if output.shape != (1, 2):  # batch_size=1, num_classes=2
                self.log_message(f"警告: 输出形状可能不正确，期望(1,2)，实际{output.shape}", "WARNING")
            
            self.log_message("ResNet18Improved模型验证通过")
            return True
            
        except Exception as e:
            self.log_message(f"模型验证失败: {e}", "ERROR")
            import traceback
            self.log_message(f"详细错误: {traceback.format_exc()}", "ERROR")
            return False
    
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
        
        # 5. 使用现代化回退机制转换（ResNet不需要包装器）
        success, conversion_info = self.convert_with_modern_fallback(loaded_model, self.input_shape, self.resnet_strategies)
        
        # 6. 保存现代化转换报告
        self.save_modern_conversion_report(success, conversion_info)
        
        if success:
            self.log_message(f"{self.model_name}已成功使用现代化方法转换为ONNX格式并通过验证")
        else:
            self.log_message(f"{self.model_name}现代化转换失败", "ERROR")
        
        return success

def main():
    """主函数"""
    converter = ModernResNet18ImprovedConverter()
    success = converter.convert()
    
    if success:
        logging.info(f"ResNet18Improved模型已成功使用现代化方法转换为ONNX格式")
    else:
        logging.error(f"ResNet18Improved模型现代化转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main()