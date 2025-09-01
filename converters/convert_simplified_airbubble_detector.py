#!/usr/bin/env python3
"""
ONNX Converter for Simplified Air Bubble Detector
将训练好的模型转换为ONNX格式
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型定义
from trainers.train_simplified_airbubble_detector_fixed import FixedSimplifiedAirBubbleDetector

class SimplifiedAirBubbleDetectorConverter:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model_id = "simplified_airbubble_detector"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 确保输出目录存在
        os.makedirs("onnx_models", exist_ok=True)
        
        # 生成输出路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.onnx_path = f"onnx_models/{self.model_id}_{timestamp}.onnx"
        
        print(f"🔄 Initializing ONNX converter for {self.model_id}")
        print(f"📱 Device: {self.device}")
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"📁 Output ONNX: {self.onnx_path}")
    
    def load_model(self):
        """加载训练好的模型"""
        print("📥 Loading trained model...")
        
        # 创建模型实例
        model = FixedSimplifiedAirBubbleDetector(num_classes=2).to(self.device)
        
        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Validation accuracy from checkpoint: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        
        return model
    
    def convert_to_onnx(self, model):
        """转换模型为ONNX格式"""
        print("🔄 Converting model to ONNX...")
        
        # 创建示例输入 (batch_size=1, channels=3, height=70, width=70)
        dummy_input = torch.randn(1, 3, 70, 70, device=self.device)
        
        # 验证模型可以处理输入
        with torch.no_grad():
            test_output = model(dummy_input)
            print(f"✅ Model test passed. Output shape: {test_output.shape}")
        
        # 转换为ONNX
        torch.onnx.export(
            model,                          # 模型
            dummy_input,                    # 示例输入
            self.onnx_path,                # 输出路径
            export_params=True,             # 导出参数
            opset_version=11,              # ONNX opset版本
            do_constant_folding=True,       # 常量折叠优化
            input_names=['input'],          # 输入名称
            output_names=['output'],        # 输出名称
            dynamic_axes={                  # 动态轴
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model saved to: {self.onnx_path}")
        return self.onnx_path
    
    def verify_onnx_model(self):
        """验证ONNX模型"""
        print("🔍 Verifying ONNX model...")
        
        try:
            # 加载ONNX模型
            onnx_model = onnx.load(self.onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model structure is valid")
            
            # 创建推理会话
            ort_session = ort.InferenceSession(self.onnx_path)
            
            # 获取输入输出信息
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            
            print(f"📊 Input info: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
            print(f"📊 Output info: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
            
            # 测试推理
            test_input = np.random.randn(1, 3, 70, 70).astype(np.float32)
            ort_outputs = ort_session.run(None, {input_info.name: test_input})
            
            print(f"✅ ONNX inference test passed. Output shape: {ort_outputs[0].shape}")
            
            # 性能测试
            import time
            num_runs = 100
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = ort_session.run(None, {input_info.name: test_input})
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs * 1000  # ms
            
            print(f"⚡ Average inference time: {avg_time:.2f} ms")
            
            return {
                'model_valid': True,
                'input_shape': list(input_info.shape),
                'output_shape': list(output_info.shape),
                'avg_inference_time_ms': float(avg_time),
                'onnx_size_mb': float(os.path.getsize(self.onnx_path) / (1024 * 1024))
            }
            
        except Exception as e:
            print(f"❌ ONNX verification failed: {e}")
            return {'model_valid': False, 'error': str(e)}
    
    def compare_pytorch_onnx(self, pytorch_model):
        """比较PyTorch和ONNX模型输出"""
        print("🔍 Comparing PyTorch vs ONNX outputs...")
        
        try:
            # 创建测试输入
            test_input_torch = torch.randn(1, 3, 70, 70, device=self.device)
            test_input_numpy = test_input_torch.cpu().numpy().astype(np.float32)
            
            # PyTorch推理
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input_torch).cpu().numpy()
            
            # ONNX推理
            ort_session = ort.InferenceSession(self.onnx_path)
            onnx_output = ort_session.run(None, {'input': test_input_numpy})[0]
            
            # 计算差异
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            
            print(f"📊 Max difference: {max_diff:.8f}")
            print(f"📊 Mean difference: {mean_diff:.8f}")
            
            # 判断是否一致
            is_consistent = max_diff < 1e-5
            print(f"✅ Models are {'consistent' if is_consistent else 'inconsistent'}")
            
            return {
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'is_consistent': bool(is_consistent)
            }
            
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
            return {'comparison_failed': True, 'error': str(e)}
    
    def generate_conversion_report(self, verification_results, comparison_results):
        """生成转换报告"""
        print("📊 Generating conversion report...")
        
        report = {
            'model_id': self.model_id,
            'conversion_timestamp': datetime.now().isoformat(),
            'checkpoint_path': self.checkpoint_path,
            'onnx_path': self.onnx_path,
            'verification_results': verification_results,
            'comparison_results': comparison_results,
            'status': 'success' if verification_results.get('model_valid', False) else 'failed'
        }
        
        # 保存JSON报告
        report_path = f"reports/{self.model_id}_onnx_conversion.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Conversion report saved to: {report_path}")
        return report
    
    def run_conversion(self):
        """运行完整的转换流程"""
        try:
            print(f"🚀 Starting ONNX conversion for {self.model_id}")
            
            # 加载模型
            model = self.load_model()
            
            # 转换为ONNX
            onnx_path = self.convert_to_onnx(model)
            
            # 验证ONNX模型
            verification_results = self.verify_onnx_model()
            
            # 比较输出一致性
            comparison_results = self.compare_pytorch_onnx(model)
            
            # 生成报告
            report = self.generate_conversion_report(verification_results, comparison_results)
            
            if verification_results.get('model_valid', False):
                print(f"✅ ONNX conversion completed successfully!")
                print(f"📁 ONNX model: {onnx_path}")
                print(f"📊 Model size: {verification_results.get('onnx_size_mb', 0):.2f} MB")
                print(f"⚡ Inference time: {verification_results.get('avg_inference_time_ms', 0):.2f} ms")
                return True, onnx_path
            else:
                print(f"❌ ONNX conversion failed!")
                return False, None
                
        except Exception as e:
            print(f"❌ Conversion pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Simplified Air Bubble Detector to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to the PyTorch checkpoint file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    converter = SimplifiedAirBubbleDetectorConverter(args.checkpoint)
    success, onnx_path = converter.run_conversion()
    
    if success:
        print(f"🎉 Conversion completed successfully!")
        print(f"📁 ONNX model saved to: {onnx_path}")
    else:
        print(f"❌ Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()