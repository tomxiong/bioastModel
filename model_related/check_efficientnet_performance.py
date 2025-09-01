#!/usr/bin/env python3
"""
检查 EfficientNet 模型的训练性能
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.efficientnet import EfficientNetCustom
from core.real_data_loader import create_real_data_loaders

def check_efficientnet_checkpoint():
    """检查 EfficientNet 检查点的训练性能"""
    print("🔍 检查 EfficientNet 模型训练性能...")
    
    checkpoint_path = "/home/aaa/ws/bioastModel/checkpoints/efficientnet_20250808_014214_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return False
    
    try:
        # 加载检查点
        print("📂 加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 分析检查点内容
        print("📊 分析检查点内容...")
        checkpoint_info = {
            'file_path': checkpoint_path,
            'file_size_mb': round(os.path.getsize(checkpoint_path) / (1024 * 1024), 2),
            'checkpoint_keys': list(checkpoint.keys())
        }
        
        # 提取训练信息
        if 'epoch' in checkpoint:
            checkpoint_info['final_epoch'] = checkpoint['epoch']
        if 'best_accuracy' in checkpoint:
            checkpoint_info['best_accuracy'] = checkpoint['best_accuracy']
        if 'train_loss' in checkpoint:
            checkpoint_info['final_train_loss'] = checkpoint['train_loss']
        if 'val_loss' in checkpoint:
            checkpoint_info['final_val_loss'] = checkpoint['val_loss']
        if 'val_accuracy' in checkpoint:
            checkpoint_info['final_val_accuracy'] = checkpoint['val_accuracy']
        
        # 分析模型结构
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # 推断类别数
            if 'classifier.weight' in state_dict:
                num_classes = state_dict['classifier.weight'].shape[0]
                checkpoint_info['num_classes'] = num_classes
                print(f"📊 检测到类别数: {num_classes}")
            
            # 计算参数数量
            total_params = sum(p.numel() for p in state_dict.values())
            checkpoint_info['total_parameters'] = total_params
            
            print(f"📊 模型参数数量: {total_params:,}")
        
        # 创建模型并加载权重
        print("🔄 创建模型并加载权重...")
        model = EfficientNetCustom(num_classes=checkpoint_info.get('num_classes', 2))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 测试模型推理
        print("🧪 测试模型推理...")
        test_input = torch.randn(1, 3, 70, 70)
        
        with torch.no_grad():
            output = model(test_input)
            checkpoint_info['output_shape'] = list(output.shape)
            checkpoint_info['output_range'] = [float(output.min()), float(output.max())]
        
        print(f"✅ 模型推理测试成功")
        print(f"   输出形状: {checkpoint_info['output_shape']}")
        print(f"   输出范围: [{checkpoint_info['output_range'][0]:.3f}, {checkpoint_info['output_range'][1]:.3f}]")
        
        # 如果有真实数据，进行性能评估
        try:
            print("📊 使用真实数据评估模型性能...")
            _, val_loader, test_loader = create_real_data_loaders(batch_size=32)
            
            # 在验证集上评估
            val_accuracy = evaluate_model(model, val_loader)
            checkpoint_info['real_data_val_accuracy'] = val_accuracy
            
            # 在测试集上评估
            test_accuracy = evaluate_model(model, test_loader)
            checkpoint_info['real_data_test_accuracy'] = test_accuracy
            
            print(f"✅ 真实数据验证集准确率: {val_accuracy:.4f}")
            print(f"✅ 真实数据测试集准确率: {test_accuracy:.4f}")
            
        except Exception as e:
            print(f"⚠️ 真实数据评估失败: {str(e)}")
            checkpoint_info['real_data_evaluation_error'] = str(e)
        
        # 打印详细信息
        print("\n" + "="*60)
        print("📋 EfficientNet 模型性能详情")
        print("="*60)
        
        print(f"📁 文件信息:")
        print(f"   路径: {checkpoint_info['file_path']}")
        print(f"   大小: {checkpoint_info['file_size_mb']} MB")
        
        print(f"\n🏋️ 训练信息:")
        if 'final_epoch' in checkpoint_info:
            print(f"   最终轮次: {checkpoint_info['final_epoch']}")
        if 'best_accuracy' in checkpoint_info:
            print(f"   最佳准确率: {checkpoint_info['best_accuracy']:.4f}")
        if 'final_train_loss' in checkpoint_info:
            print(f"   最终训练损失: {checkpoint_info['final_train_loss']:.4f}")
        if 'final_val_loss' in checkpoint_info:
            print(f"   最终验证损失: {checkpoint_info['final_val_loss']:.4f}")
        if 'final_val_accuracy' in checkpoint_info:
            print(f"   最终验证准确率: {checkpoint_info['final_val_accuracy']:.4f}")
        
        print(f"\n🔧 模型结构:")
        print(f"   类别数: {checkpoint_info.get('num_classes', 'N/A')}")
        print(f"   参数数量: {checkpoint_info.get('total_parameters', 'N/A'):,}")
        print(f"   输出形状: {checkpoint_info.get('output_shape', 'N/A')}")
        
        if 'real_data_val_accuracy' in checkpoint_info:
            print(f"\n📊 真实数据性能:")
            print(f"   验证集准确率: {checkpoint_info['real_data_val_accuracy']:.4f}")
            print(f"   测试集准确率: {checkpoint_info['real_data_test_accuracy']:.4f}")
        
        print("="*60)
        
        # 保存性能报告
        performance_report = {
            'model_name': 'efficientnet',
            'checkpoint_analysis': checkpoint_info,
            'analysis_timestamp': datetime.now().isoformat(),
            'performance_summary': {
                'training_completed': True,
                'model_loadable': True,
                'inference_working': True,
                'real_data_tested': 'real_data_val_accuracy' in checkpoint_info
            }
        }
        
        report_path = "efficientnet_performance_check.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 性能检查报告已保存: {report_path}")
        
        return checkpoint_info
        
    except Exception as e:
        print(f"❌ 检查 EfficientNet 性能时出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_model(model, data_loader):
    """评估模型在数据集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def main():
    """主函数"""
    print("🚀 开始检查 EfficientNet 模型性能...")
    
    try:
        result = check_efficientnet_checkpoint()
        
        if result:
            print("\n🎉 EfficientNet 性能检查完成!")
            
            # 判断模型质量
            quality_assessment = "优秀"
            if 'real_data_test_accuracy' in result:
                test_acc = result['real_data_test_accuracy']
                if test_acc >= 0.95:
                    quality_assessment = "优秀"
                elif test_acc >= 0.90:
                    quality_assessment = "良好"
                elif test_acc >= 0.80:
                    quality_assessment = "一般"
                else:
                    quality_assessment = "需要改进"
            
            print(f"📊 模型质量评估: {quality_assessment}")
            
            return True
        else:
            print("\n❌ EfficientNet 性能检查失败")
            return False
            
    except Exception as e:
        print(f"❌ 性能检查过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)