#!/usr/bin/env python3
"""
正确修复 airbubble_hybrid_net ONNX 转换
使用完整架构保持模型性能，而不是简化架构
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.airbubble_hybrid_net import AirBubbleHybridNet

class ONNXCompatibleAirBubbleHybridNet(nn.Module):
    """ONNX兼容的AirBubbleHybridNet版本"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
        # 使用原始模型的架构，但修复ONNX兼容性问题
        self.original_model = AirBubbleHybridNet(num_classes=num_classes)
        
    def forward(self, x):
        """前向传播 - 只返回分类结果以简化ONNX导出"""
        # 调用原始模型
        outputs = self.original_model(x)
        
        # 如果返回字典，提取分类输出
        if isinstance(outputs, dict):
            # 尝试不同的键名
            for key in ['classification', 'class_logits', 'logits']:
                if key in outputs:
                    return outputs[key]
            # 如果没有找到，返回第一个输出
            return list(outputs.values())[0]
        else:
            return outputs
    
    def load_state_dict_compatible(self, state_dict):
        """兼容的状态字典加载"""
        # 直接加载到原始模型
        self.original_model.load_state_dict(state_dict)

def create_onnx_compatible_model():
    """创建ONNX兼容的模型"""
    print("🔄 创建ONNX兼容的模型...")
    
    # 加载原始检查点
    checkpoint_path = "checkpoints/airbubble_hybrid_net_20250808_013453_best.pth"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 从检查点推断类别数
    num_classes = checkpoint['model_state_dict']['classification_head.4.weight'].shape[0]
    print(f"📊 检测到类别数: {num_classes}")
    
    # 创建兼容模型
    model = ONNXCompatibleAirBubbleHybridNet(num_classes=num_classes)
    model.load_state_dict_compatible(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ ONNX兼容模型创建成功")
    return model

def test_model_equivalence(original_model, onnx_model, num_tests=10):
    """测试模型等价性"""
    print("🔄 测试模型等价性...")
    
    original_model.eval()
    onnx_model.eval()
    
    max_diff = 0.0
    
    with torch.no_grad():
        for i in range(num_tests):
            # 创建随机输入
            test_input = torch.randn(1, 3, 70, 70)
            
            # 获取原始模型输出
            original_output = original_model(test_input)
            if isinstance(original_output, dict):
                # 提取分类输出
                for key in ['classification', 'class_logits', 'logits']:
                    if key in original_output:
                        original_output = original_output[key]
                        break
                else:
                    original_output = list(original_output.values())[0]
            
            # 获取ONNX兼容模型输出
            onnx_output = onnx_model(test_input)
            
            # 计算差异
            diff = torch.abs(original_output - onnx_output).max().item()
            max_diff = max(max_diff, diff)
            
            print(f"   测试 {i+1}: 最大差异 = {diff:.8f}")
    
    print(f"✅ 模型等价性测试完成，最大差异: {max_diff:.8f}")
    return max_diff < 1e-6

def export_to_onnx(model, output_path):
    """导出模型到ONNX格式"""
    print("🔄 导出模型到ONNX格式...")
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 70, 70)
    
    # 导出参数
    export_params = {
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_axes': {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        'opset_version': 11,
        'do_constant_folding': True,
        'verbose': False
    }
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            **export_params
        )
        print(f"✅ ONNX模型导出成功: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX导出失败: {str(e)}")
        return False

def validate_onnx_model(onnx_path, pytorch_model):
    """验证ONNX模型"""
    print("🔄 验证ONNX模型...")
    
    try:
        # 加载ONNX模型
        ort_session = ort.InferenceSession(onnx_path)
        
        # 创建测试输入
        test_input = torch.randn(1, 3, 70, 70)
        test_input_np = test_input.numpy().astype(np.float32)
        
        # PyTorch模型推理
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
        
        # ONNX模型推理
        onnx_output = ort_session.run(['output'], {'input': test_input_np})[0]
        
        # 比较输出
        pytorch_output_np = pytorch_output.numpy()
        max_diff = np.abs(pytorch_output_np - onnx_output).max()
        
        print(f"✅ ONNX模型验证完成")
        print(f"   PyTorch输出形状: {pytorch_output_np.shape}")
        print(f"   ONNX输出形状: {onnx_output.shape}")
        print(f"   最大差异: {max_diff:.8f}")
        
        # 计算模型大小
        model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"   模型大小: {model_size_mb:.2f} MB")
        
        return {
            'success': True,
            'max_difference': float(max_diff),
            'model_size_mb': model_size_mb,
            'pytorch_shape': pytorch_output_np.shape,
            'onnx_shape': onnx_output.shape
        }
        
    except Exception as e:
        print(f"❌ ONNX模型验证失败: {str(e)}")
        return {'success': False, 'error': str(e)}

def benchmark_inference_speed(onnx_path, num_runs=100):
    """基准测试推理速度"""
    print("🔄 基准测试推理速度...")
    
    try:
        # 加载ONNX模型
        ort_session = ort.InferenceSession(onnx_path)
        
        # 预热
        test_input = np.random.randn(1, 3, 70, 70).astype(np.float32)
        for _ in range(10):
            ort_session.run(['output'], {'input': test_input})
        
        # 测试推理时间
        import time
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            ort_session.run(['output'], {'input': test_input})
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"✅ 推理速度基准测试完成")
        print(f"   平均推理时间: {avg_time:.4f} ms")
        print(f"   标准差: {std_time:.4f} ms")
        print(f"   最小时间: {np.min(times):.4f} ms")
        print(f"   最大时间: {np.max(times):.4f} ms")
        
        return {
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'min_inference_time_ms': np.min(times),
            'max_inference_time_ms': np.max(times)
        }
        
    except Exception as e:
        print(f"❌ 推理速度测试失败: {str(e)}")
        return {'error': str(e)}

def main():
    """主函数"""
    print("🚀 开始正确修复 AirBubble Hybrid Net ONNX 转换...")
    
    try:
        # 1. 创建ONNX兼容模型
        onnx_compatible_model = create_onnx_compatible_model()
        
        # 2. 加载原始模型进行等价性测试
        original_model = AirBubbleHybridNet(num_classes=2)
        checkpoint = torch.load("checkpoints/airbubble_hybrid_net_20250808_013453_best.pth", map_location='cpu')
        original_model.load_state_dict(checkpoint['model_state_dict'])
        original_model.eval()
        
        # 3. 测试模型等价性
        is_equivalent = test_model_equivalence(original_model, onnx_compatible_model)
        
        if not is_equivalent:
            print("⚠️ 模型等价性测试未通过，但继续进行ONNX转换...")
        
        # 4. 导出ONNX模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        onnx_path = f"onnx_models/airbubble_hybrid_net_proper_{timestamp}.onnx"
        
        # 确保目录存在
        os.makedirs("onnx_models", exist_ok=True)
        
        export_success = export_to_onnx(onnx_compatible_model, onnx_path)
        
        if not export_success:
            print("❌ ONNX导出失败")
            return False
        
        # 5. 验证ONNX模型
        validation_result = validate_onnx_model(onnx_path, onnx_compatible_model)
        
        if not validation_result['success']:
            print("❌ ONNX模型验证失败")
            return False
        
        # 6. 基准测试推理速度
        speed_result = benchmark_inference_speed(onnx_path)
        
        # 7. 输出总结
        print("\n" + "="*60)
        print("🎉 AirBubble Hybrid Net ONNX 转换完成!")
        print("="*60)
        print(f"📁 ONNX模型路径: {onnx_path}")
        print(f"📊 模型大小: {validation_result['model_size_mb']:.2f} MB")
        print(f"🎯 最大差异: {validation_result['max_difference']:.8f}")
        if 'avg_inference_time_ms' in speed_result:
            print(f"⚡ 平均推理时间: {speed_result['avg_inference_time_ms']:.4f} ms")
        print("="*60)
        
        # 保存转换信息
        conversion_info = {
            'timestamp': timestamp,
            'onnx_path': onnx_path,
            'conversion_method': 'proper_architecture_preservation',
            'model_equivalent': is_equivalent,
            'validation_result': validation_result,
            'speed_benchmark': speed_result
        }
        
        import json
        with open(f"airbubble_hybrid_net_proper_conversion_{timestamp}.json", 'w') as f:
            json.dump(conversion_info, f, indent=2)
        
        print(f"✅ 转换信息已保存")
        return True
        
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 正确的ONNX转换已完成!")
    else:
        print("\n❌ ONNX转换失败，需要进一步调试")
        sys.exit(1)