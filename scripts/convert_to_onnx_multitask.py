#!/usr/bin/env python3
"""
核心边界优化多任务MIC MobileNetV3模型ONNX转换脚本
Convert Core Boundary Optimization Multitask MIC MobileNetV3 model to ONNX format
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.onnx
import onnxruntime as ort
import json
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from models.multitask_mic_mobilenetv3 import create_multitask_mic_mobilenetv3
from training.enhanced_multitask_dataset import create_multitask_dataloaders

class MultitaskMicOnnxConverter:
    """多任务MIC MobileNetV3模型ONNX转换器"""
    
    def __init__(self, experiment_dir="experiments/core_boundary_optimization"):
        self.experiment_dir = Path(experiment_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.dataloaders = None
        
        print(f"🔧 初始化ONNX转换器")
        print(f"   实验目录: {self.experiment_dir}")
        print(f"   设备: {self.device}")
    
    def load_model(self):
        """加载训练好的模型"""
        print("\n📊 加载模型...")
        
        # 查找最佳模型文件
        model_path = self.experiment_dir / "best_model.pth"
        if not model_path.exists():
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 创建数据加载器获取标签映射
        self.dataloaders = create_multitask_dataloaders(
            data_root="/home/aaa/ws/bioastModel/ds/images",
            annotations_file="m9e1n170.json",
            batch_size=1,
            num_workers=1,
            seed=42
        )
        
        # 获取标签映射信息
        dataset = self.dataloaders['train'].dataset
        num_growth_patterns = len(dataset.label_mappings['growth_pattern'])
        num_interference_factors = len(dataset.label_mappings['interference_factors'])
        
        print(f"   生长模式类别数: {num_growth_patterns}")
        print(f"   干扰因素类别数: {num_interference_factors}")
        
        # 创建模型
        self.model = create_multitask_mic_mobilenetv3(
            num_classes=2,
            num_growth_patterns=num_growth_patterns,
            num_interference_factors=num_interference_factors,
            width_mult=1.0,
            input_channels=1  # 灰度图像
        )
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ 模型加载完成 (Epoch {checkpoint['epoch']})")
        print(f"   验证准确率: {checkpoint.get('best_val_acc', 'N/A')}")
        
        # 保存标签映射信息供C#使用
        self.save_label_mappings(dataset.label_mappings)
        
        return True
    
    def save_label_mappings(self, label_mappings):
        """保存标签映射信息供C#使用"""
        mappings_path = self.experiment_dir / "label_mappings.json"
        
        # 转换为可序列化格式
        serializable_mappings = {}
        for key, mapping in label_mappings.items():
            if isinstance(mapping, dict):
                serializable_mappings[key] = mapping
            else:
                serializable_mappings[key] = dict(mapping)
        
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_mappings, f, indent=2, ensure_ascii=False)
        
        print(f"💾 标签映射已保存: {mappings_path}")

    
    def convert_to_onnx(self, output_path=None):
        """转换模型为ONNX格式"""
        if self.model is None:
            print("❌ 请先加载模型")
            return False
        
        print("\n🔄 开始ONNX转换...")
        
        # 设置输出路径
        if output_path is None:
            output_path = self.experiment_dir / "multitask_mic_mobilenetv3.onnx"
        else:
            output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建示例输入 (批次大小1, 1通道, 70x70像素)
        dummy_input = torch.randn(1, 1, 70, 70)
        
        # 输入输出名称
        input_names = ['image']
        output_names = ['classification', 'growth_pattern', 'interference_factors']
        
        try:
            # 转换为ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    'image': {0: 'batch_size'},
                    'classification': {0: 'batch_size'},
                    'growth_pattern': {0: 'batch_size'},
                    'interference_factors': {0: 'batch_size'}
                }
            )
            
            print(f"✅ ONNX模型已保存: {output_path}")
            print(f"   文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # 验证ONNX模型
            self.verify_onnx_model(output_path, dummy_input)
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ ONNX转换失败: {str(e)}")
            return False
    
    def verify_onnx_model(self, onnx_path, test_input):
        """验证ONNX模型正确性"""
        print("\n🔍 验证ONNX模型...")
        
        try:
            import onnx
            
            # 加载并验证ONNX模型
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX模型结构验证通过")
            
            # 创建推理会话
            ort_session = ort.InferenceSession(str(onnx_path))
            
            # 获取输入输出信息
            input_name = ort_session.get_inputs()[0].name
            output_names = [output.name for output in ort_session.get_outputs()]
            
            print(f"   输入名称: {input_name}")
            print(f"   输出名称: {output_names}")
            
            # PyTorch推理
            with torch.no_grad():
                pytorch_outputs = self.model(test_input)
            
            # ONNX推理
            onnx_inputs = {input_name: test_input.numpy()}
            onnx_outputs = ort_session.run(output_names, onnx_inputs)
            
            # 比较输出
            print("\n📊 输出对比:")
            for i, (name, pytorch_out, onnx_out) in enumerate(zip(
                output_names, 
                [pytorch_outputs['classification'], pytorch_outputs['growth_pattern'], pytorch_outputs['interference_factors']],
                onnx_outputs
            )):
                diff = np.abs(pytorch_out.numpy() - onnx_out).max()
                print(f"   {name}: 最大差异 = {diff:.6f}")
                
                if diff < 1e-5:
                    print(f"   ✅ {name} 输出一致")
                else:
                    print(f"   ⚠️ {name} 输出差异较大")
            
            print("✅ ONNX模型验证完成")
            return True
            
        except ImportError:
            print("⚠️ 缺少onnx库，跳过验证")
            print("   安装命令: uv pip install onnx onnxruntime")
            return True
        except Exception as e:
            print(f"❌ ONNX模型验证失败: {str(e)}")
            return False
    
    def generate_model_info(self):
        """生成模型信息文件供C#使用"""
        if self.model is None:
            return
        
        print("\n📋 生成模型信息...")
        
        # 模型信息
        model_info = {
            "model_name": "MultitaskMIC_MobileNetV3",
            "version": "1.0",
            "description": "Core Boundary Optimization Multitask MIC MobileNetV3 for colony classification",
            "input_shape": [1, 1, 70, 70],  # [batch_size, channels, height, width]
            "input_type": "float32",
            "input_description": "Grayscale image normalized to [0,1] range",
            "outputs": {
                "classification": {
                    "shape": [1, 2],
                    "type": "float32", 
                    "description": "Binary classification probabilities [negative, positive]",
                    "classes": ["negative", "positive"]
                },
                "growth_pattern": {
                    "shape": [1, "num_growth_patterns"],
                    "type": "float32",
                    "description": "Growth pattern classification probabilities"
                },
                "interference_factors": {
                    "shape": [1, "num_interference_factors"], 
                    "type": "float32",
                    "description": "Interference factor detection probabilities (multi-label)"
                }
            },
            "preprocessing": {
                "resize": [70, 70],
                "convert_to_grayscale": True,
                "normalize": {
                    "mean": [0.485],
                    "std": [0.229]
                },
                "data_format": "CHW"  # Channel, Height, Width
            },
            "performance": {
                "accuracy": "96.27%",
                "optimized_for": "boundary cases and pore interference detection"
            }
        }
        
        # 保存模型信息
        info_path = self.experiment_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 模型信息已保存: {info_path}")
def main():
    parser = argparse.ArgumentParser(description='Convert Core Boundary Optimization Multitask MIC MobileNetV3 to ONNX')
    parser.add_argument('--experiment_dir', type=str, 
                        default='experiments/core_boundary_optimization',
                        help='实验目录路径')
    parser.add_argument('--output', type=str, default=None,
                        help='ONNX输出文件路径')
    parser.add_argument('--deployment_dir', type=str, default='deployment/onnx_models',
                        help='部署目录路径')
    
    args = parser.parse_args()
    
    print("🚀 核心边界优化多任务MIC MobileNetV3 ONNX转换")
    print("=" * 60)
    
    # 创建转换器
    converter = MultitaskMicOnnxConverter(args.experiment_dir)
    
    # 加载模型
    if not converter.load_model():
        print("❌ 模型加载失败")
        return
    
    # 转换为ONNX
    onnx_path = converter.convert_to_onnx(args.output)
    if not onnx_path:
        print("❌ ONNX转换失败")
        return
    
    # 生成模型信息
    converter.generate_model_info()
    
    # 复制到部署目录
    deployment_dir = Path(args.deployment_dir)
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_deployment_path = deployment_dir / "multitask_mic_mobilenetv3.onnx"
    import shutil
    shutil.copy2(onnx_path, onnx_deployment_path)
    print(f"📦 模型已复制到部署目录: {onnx_deployment_path}")
    
    print("\n🎊 ONNX转换完成！")
    print(f"📄 ONNX模型: {onnx_path}")
    print(f"📊 模型信息: {converter.experiment_dir}/model_info.json")
    print(f"🏷️ 标签映射: {converter.experiment_dir}/label_mappings.json")
    print(f"🚀 部署模型: {onnx_deployment_path}")

if __name__ == "__main__":
    main()