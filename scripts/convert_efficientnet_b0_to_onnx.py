#!/usr/bin/env python3
"""
EfficientNet-B0多任务模型ONNX转换脚本
将训练好的EfficientNet-B0多任务模型转换为ONNX格式
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.multitask_efficientnet_b0 import create_multitask_efficientnet_b0_standard


def find_latest_experiment(pattern='ni_multitask_efficientnet_b0'):
    """查找最新的实验目录"""
    experiments_dir = Path(project_root) / 'experiments'
    if not experiments_dir.exists():
        raise FileNotFoundError("实验目录不存在")
    
    # 查找匹配的实验目录
    matching_dirs = [d for d in experiments_dir.iterdir() 
                    if d.is_dir() and pattern in d.name]
    
    if not matching_dirs:
        raise FileNotFoundError(f"未找到匹配的实验目录：{pattern}")
    
    # 按时间排序，返回最新的
    latest_dir = sorted(matching_dirs, key=lambda x: x.name)[-1]
    return latest_dir


def load_trained_model(model_path, device='cuda'):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    
    # 加载检查点
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 创建模型
    model = create_multitask_efficientnet_b0_standard(pretrained=False)
    model = model.to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ 模型加载成功")
    return model, checkpoint


def test_model_inference(model, device='cuda'):
    """测试模型推理"""
    print("测试模型推理...")
    
    # 创建测试输入
    test_input = torch.randn(1, 1, 70, 70).to(device)
    
    with torch.no_grad():
        outputs = model(test_input)
    
    print("模型输出:")
    for task_name, output_tensor in outputs.items():
        print(f"  {task_name}: {output_tensor.shape}")
    
    print("✓ 模型推理测试通过")
    return outputs


def convert_to_onnx(model, onnx_path, device='cuda'):
    """转换模型为ONNX格式"""
    print(f"开始ONNX转换: {onnx_path}")
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 1, 70, 70).to(device)
    
    # 获取输出名称
    with torch.no_grad():
        sample_outputs = model(dummy_input)
    output_names = list(sample_outputs.keys())
    
    print(f"输出任务: {output_names}")
    
    # ONNX导出
    try:
        torch.onnx.export(
            model,                          # 模型
            dummy_input,                    # 输入
            onnx_path,                     # 输出路径
            export_params=True,            # 导出参数
            opset_version=11,              # ONNX opset版本
            do_constant_folding=True,      # 常量折叠优化
            input_names=['input'],         # 输入名称
            output_names=output_names,     # 输出名称
            dynamic_axes={                 # 动态轴
                'input': {0: 'batch_size'},
                **{name: {0: 'batch_size'} for name in output_names}
            }
        )
        print("✓ ONNX导出成功")
        
    except Exception as e:
        print(f"✗ ONNX导出失败: {e}")
        raise e


def validate_onnx_model(onnx_path, pytorch_model, device='cuda'):
    """验证ONNX模型的正确性"""
    print("验证ONNX模型...")
    
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("警告: onnx或onnxruntime未安装，跳过验证")
        return
    
    # 检查ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX模型结构检查通过")
    
    # 创建推理会话
    ort_session = ort.InferenceSession(onnx_path)
    
    # 准备测试数据
    test_input = np.random.randn(1, 1, 70, 70).astype(np.float32)
    test_input_torch = torch.from_numpy(test_input).to(device)
    
    # PyTorch推理
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_outputs = pytorch_model(test_input_torch)
    
    # ONNX推理
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # 比较结果
    output_names = list(pytorch_outputs.keys())
    max_diff = 0
    
    print("输出对比:")
    for i, task_name in enumerate(output_names):
        pytorch_out = pytorch_outputs[task_name].cpu().numpy()
        onnx_out = ort_outputs[i]
        
        diff = np.abs(pytorch_out - onnx_out).max()
        max_diff = max(max_diff, diff)
        
        print(f"  {task_name}: 最大差异 = {diff:.6f}")
    
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"✓ ONNX模型验证通过 (最大差异: {max_diff:.6f})")
    else:
        print(f"⚠ ONNX模型精度可能有问题 (最大差异: {max_diff:.6f})")
    
    return max_diff


def save_model_info(model, onnx_path, checkpoint_info, output_dir):
    """保存模型信息"""
    model_info = model.get_model_info()
    
    # 获取ONNX文件大小
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    # 合并信息
    full_info = {
        'model_architecture': model_info,
        'training_info': {
            'best_val_loss': checkpoint_info.get('best_val_loss'),
            'final_val_loss': checkpoint_info.get('final_val_loss'),
            'epoch': checkpoint_info.get('epoch'),
            'val_task_accs': checkpoint_info.get('val_task_accs', {})
        },
        'onnx_info': {
            'file_path': str(onnx_path),
            'file_size_mb': round(onnx_size_mb, 2),
            'opset_version': 11
        }
    }
    
    # 保存信息文件
    info_path = output_dir / 'model_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(full_info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 模型信息保存至: {info_path}")
    return full_info


def main():
    """主函数"""
    print("=== EfficientNet-B0多任务模型ONNX转换 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 查找最新实验
        experiment_dir = find_latest_experiment('ni_multitask_efficientnet_b0')
        print(f"实验目录: {experiment_dir}")
        
        # 2. 查找最佳模型
        best_model_path = experiment_dir / 'best_model.pth'
        if not best_model_path.exists():
            print("最佳模型不存在，尝试使用最终模型")
            best_model_path = experiment_dir / 'final_model.pth'
        
        if not best_model_path.exists():
            raise FileNotFoundError("未找到训练好的模型文件")
        
        # 3. 加载模型
        model, checkpoint = load_trained_model(best_model_path, device)
        
        # 4. 测试推理
        test_model_inference(model, device)
        
        # 5. 创建输出目录
        output_dir = Path(project_root) / 'onnx_models' / 'efficientnet_b0_multitask'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 6. ONNX转换
        onnx_path = output_dir / 'efficientnet_b0_multitask.onnx'
        convert_to_onnx(model, onnx_path, device)
        
        # 7. 验证ONNX模型
        validate_onnx_model(onnx_path, model, device)
        
        # 8. 保存模型信息
        model_info = save_model_info(model, onnx_path, checkpoint, output_dir)
        
        # 9. 显示总结
        print("\n=== 转换完成 ===")
        print(f"ONNX模型路径: {onnx_path}")
        print(f"模型文件大小: {model_info['onnx_info']['file_size_mb']:.2f} MB")
        print(f"模型参数数量: {model_info['model_architecture']['total_parameters']:,}")
        
        if 'val_task_accs' in model_info['training_info']:
            print("各任务验证准确率:")
            for task, acc in model_info['training_info']['val_task_accs'].items():
                if isinstance(acc, (int, float)):
                    print(f"  {task}: {acc:.4f}")
        
        print("✓ EfficientNet-B0多任务模型ONNX转换成功完成！")
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())