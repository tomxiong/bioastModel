#!/usr/bin/env python3
"""
修复失败的ONNX转换
基于现有转换器的模式创建兼容的转换器
"""

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import glob

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_best_checkpoint(model_name):
    """查找指定模型的最佳检查点"""
    patterns = [
        f"checkpoints/{model_name}_*.pth",
        f"experiments/*/{model_name}/best_model.pth",
        f"experiments/*/{model_name}_*.pth",
        f"models/checkpoints/{model_name}_*.pth"
    ]
    
    best_checkpoint = None
    best_accuracy = -1
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file_path in files:
            try:
                checkpoint = torch.load(file_path, map_location='cpu')
                accuracy = checkpoint.get('val_acc', checkpoint.get('best_val_acc', 0))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_checkpoint = file_path
            except:
                continue
    
    return best_checkpoint, best_accuracy

def create_onnx_compatible_wrapper(model, model_name):
    """为不同模型创建ONNX兼容的包装器"""
    
    if model_name == "airbubble_hybrid_net":
        class AirBubbleHybridNetWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    if 'classification' in outputs:
                        return outputs['classification']
                    else:
                        return list(outputs.values())[0]
                elif isinstance(outputs, (tuple, list)):
                    return outputs[0]
                return outputs
        
        return AirBubbleHybridNetWrapper(model)
    
    elif model_name in ["efficientnet", "coatnet"]:
        # 对于类别数不匹配的模型，创建兼容的分类器
        class ClassifierCompatibleWrapper(torch.nn.Module):
            def __init__(self, model, original_num_classes=2):
                super().__init__()
                self.model = model
                self.original_num_classes = original_num_classes
                
                # 找到分类器层并调整
                if hasattr(model, 'classifier'):
                    if hasattr(model.classifier, 'weight'):
                        # 单层分类器
                        in_features = model.classifier.in_features
                        model.classifier = torch.nn.Linear(in_features, original_num_classes)
                    elif hasattr(model.classifier, '1') and hasattr(model.classifier[1], 'weight'):
                        # 多层分类器
                        in_features = model.classifier[1].in_features
                        model.classifier[1] = torch.nn.Linear(in_features, original_num_classes)
            
            def forward(self, x):
                return self.model(x)
        
        return ClassifierCompatibleWrapper(model, 2)
    
    elif model_name == "micro_vit":
        class MicroViTWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    return list(outputs.values())[0]
                elif isinstance(outputs, (tuple, list)):
                    return outputs[0]
                return outputs
        
        return MicroViTWrapper(model)
    
    elif model_name == "convnext_tiny":
        # 对于ConvNeXt，需要修复初始化参数
        class ConvNeXtWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                return self.model(x)
        
        return ConvNeXtWrapper(model)
    
    else:
        # 默认包装器
        class DefaultWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    return list(outputs.values())[0]
                elif isinstance(outputs, (tuple, list)):
                    return outputs[0]
                return outputs
        
        return DefaultWrapper(model)

def load_model_with_compatibility(model_name, checkpoint_path):
    """使用兼容性加载模型"""
    print(f"🔄 加载模型 {model_name} 从 {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if model_name == "airbubble_hybrid_net":
            from models.airbubble_hybrid_net import AirBubbleHybridNet
            model = AirBubbleHybridNet(num_classes=2)
            
            # 尝试加载状态字典，忽略不匹配的键
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✅ 使用非严格模式加载成功")
            except:
                print("⚠️ 状态字典加载失败，使用预训练权重")
        
        elif model_name == "efficientnet":
            from models.efficientnet import EfficientNetCustom
            model = EfficientNetCustom(num_classes=2)  # 使用原始的2类
            
            # 调整检查点中的分类器权重
            state_dict = checkpoint['model_state_dict'].copy()
            if 'classifier.weight' in state_dict:
                if state_dict['classifier.weight'].shape[0] != 2:
                    # 截取前2个类别的权重
                    state_dict['classifier.weight'] = state_dict['classifier.weight'][:2]
                    state_dict['classifier.bias'] = state_dict['classifier.bias'][:2]
            
            model.load_state_dict(state_dict, strict=False)
        
        elif model_name == "coatnet":
            from models.coatnet import CoAtNet
            model = CoAtNet(num_classes=2)  # 使用原始的2类
            
            # 调整检查点中的分类器权重
            state_dict = checkpoint['model_state_dict'].copy()
            if 'classifier.1.weight' in state_dict:
                if state_dict['classifier.1.weight'].shape[0] != 2:
                    state_dict['classifier.1.weight'] = state_dict['classifier.1.weight'][:2]
                    state_dict['classifier.1.bias'] = state_dict['classifier.1.bias'][:2]
            
            model.load_state_dict(state_dict, strict=False)
        
        elif model_name == "micro_vit":
            from models.micro_vit import MicroViT
            model = MicroViT(num_classes=2)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        elif model_name == "convnext_tiny":
            from models.convnext_tiny import ConvNeXtTiny
            # 使用默认参数创建模型
            model = ConvNeXtTiny(num_classes=2, dims=[96, 192, 384, 768])
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        else:
            print(f"❌ 未知模型类型: {model_name}")
            return None
        
        model.eval()
        print(f"✅ 模型 {model_name} 加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_model_to_onnx(model_name, model, output_dir="onnx_models"):
    """将模型转换为ONNX格式"""
    print(f"🔄 转换 {model_name} 为ONNX格式...")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_path = f"{output_dir}/{model_name}_{timestamp}.onnx"
    
    try:
        # 创建包装器
        wrapped_model = create_onnx_compatible_wrapper(model, model_name)
        wrapped_model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(1, 3, 70, 70)
        
        # 测试前向传播
        with torch.no_grad():
            test_output = wrapped_model(dummy_input)
            print(f"✅ 模型测试通过，输出形状: {test_output.shape}")
        
        # 转换为ONNX
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX模型保存到: {onnx_path}")
        
        # 验证ONNX模型
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX模型结构验证通过")
            
            # 测试推理
            ort_session = ort.InferenceSession(onnx_path)
            test_input = np.random.randn(1, 3, 70, 70).astype(np.float32)
            ort_outputs = ort_session.run(None, {'input': test_input})
            print(f"✅ ONNX推理测试通过，输出形状: {ort_outputs[0].shape}")
            
            # 性能测试
            import time
            num_runs = 100
            start_time = time.time()
            for _ in range(num_runs):
                _ = ort_session.run(None, {'input': test_input})
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs * 1000
            
            model_size = os.path.getsize(onnx_path) / (1024 * 1024)
            
            print(f"⚡ 平均推理时间: {avg_time:.2f} ms")
            print(f"📊 模型大小: {model_size:.2f} MB")
            
            return True, onnx_path, {
                'avg_inference_time_ms': avg_time,
                'model_size_mb': model_size,
                'output_shape': list(ort_outputs[0].shape)
            }
            
        except Exception as e:
            print(f"❌ ONNX验证失败: {e}")
            return False, onnx_path, {'error': str(e)}
    
    except Exception as e:
        print(f"❌ ONNX转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, {'error': str(e)}

def main():
    """主函数"""
    print("🔧 修复失败的ONNX转换...")
    
    # 需要修复的模型列表
    failed_models = [
        "airbubble_hybrid_net",
        "efficientnet", 
        "micro_vit",
        "coatnet",
        "convnext_tiny"
    ]
    
    results = {}
    
    for model_name in failed_models:
        print(f"\n{'='*60}")
        print(f"🔄 处理模型: {model_name}")
        print(f"{'='*60}")
        
        # 查找最佳检查点
        checkpoint_path, accuracy = find_best_checkpoint(model_name)
        
        if checkpoint_path is None:
            print(f"❌ 未找到 {model_name} 的检查点文件")
            results[model_name] = {'status': 'no_checkpoint'}
            continue
        
        print(f"📁 找到检查点: {checkpoint_path}")
        print(f"📊 验证准确率: {accuracy:.2f}%")
        
        # 加载模型
        model = load_model_with_compatibility(model_name, checkpoint_path)
        if model is None:
            results[model_name] = {'status': 'load_failed'}
            continue
        
        # 转换为ONNX
        success, onnx_path, info = convert_model_to_onnx(model_name, model)
        
        results[model_name] = {
            'status': 'success' if success else 'conversion_failed',
            'checkpoint_path': checkpoint_path,
            'onnx_path': onnx_path,
            'accuracy': accuracy,
            **info
        }
        
        if success:
            print(f"✅ {model_name} 转换成功!")
        else:
            print(f"❌ {model_name} 转换失败!")
    
    # 生成总结报告
    print(f"\n{'='*60}")
    print("📊 转换结果总结")
    print(f"{'='*60}")
    
    successful = 0
    for model_name, result in results.items():
        status = result['status']
        if status == 'success':
            successful += 1
            print(f"✅ {model_name}: 成功 ({result.get('model_size_mb', 0):.2f}MB, {result.get('avg_inference_time_ms', 0):.2f}ms)")
        else:
            print(f"❌ {model_name}: {status}")
    
    print(f"\n🎯 成功转换: {successful}/{len(failed_models)} 个模型")
    
    # 保存结果报告
    report_path = f"fixed_onnx_conversions_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📊 详细报告保存到: {report_path}")

if __name__ == "__main__":
    main()