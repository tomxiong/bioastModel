"""
验证ONNX格式的简化版气孔检测器模型
"""

import os
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
from models.simplified_airbubble_detector import SimplifiedAirBubbleDetector, generate_synthetic_data

def load_pytorch_model(checkpoint_path):
    """加载PyTorch模型"""
    print(f"🔍 加载PyTorch模型: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model = SimplifiedAirBubbleDetector()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ 成功加载PyTorch模型")
        return model
    except Exception as e:
        print(f"❌ 加载PyTorch模型失败: {e}")
        return None

def load_onnx_model(onnx_path):
    """加载ONNX模型"""
    print(f"🔍 加载ONNX模型: {onnx_path}")
    
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX模型文件不存在: {onnx_path}")
        return None
    
    try:
        session = ort.InferenceSession(onnx_path)
        print(f"✅ 成功加载ONNX模型")
        return session
    except Exception as e:
        print(f"❌ 加载ONNX模型失败: {e}")
        return None

def generate_test_data(num_samples=100):
    """生成测试数据"""
    print(f"🔍 生成{num_samples}个测试样本...")
    
    X, y = generate_synthetic_data(num_samples)
    print(f"✅ 成功生成测试数据: X形状={X.shape}, y形状={y.shape}")
    
    return X, y

def compare_model_outputs(pytorch_model, onnx_session, test_data):
    """比较PyTorch和ONNX模型的输出"""
    print("🔍 比较PyTorch和ONNX模型的输出...")
    
    X = test_data
    
    # PyTorch模型推理
    with torch.no_grad():
        pytorch_outputs = pytorch_model(torch.tensor(X, dtype=torch.float32)).numpy()
    
    # ONNX模型推理
    onnx_inputs = {onnx_session.get_inputs()[0].name: X.astype(np.float32)}
    onnx_outputs = onnx_session.run(None, onnx_inputs)[0]
    
    # 计算差异
    max_diff = np.max(np.abs(pytorch_outputs - onnx_outputs))
    mean_diff = np.mean(np.abs(pytorch_outputs - onnx_outputs))
    
    print(f"📊 最大绝对差异: {max_diff:.6f}")
    print(f"📊 平均绝对差异: {mean_diff:.6f}")
    
    # 判断是否一致
    is_consistent = max_diff < 1e-4
    status = "✅ 一致" if is_consistent else "❌ 不一致"
    print(f"📊 模型输出: {status}")
    
    # 比较预测结果
    pytorch_preds = np.argmax(pytorch_outputs, axis=1)
    onnx_preds = np.argmax(onnx_outputs, axis=1)
    
    prediction_match = np.mean(pytorch_preds == onnx_preds) * 100
    print(f"📊 预测结果匹配率: {prediction_match:.2f}%")
    
    return {
        'pytorch_outputs': pytorch_outputs,
        'onnx_outputs': onnx_outputs,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'is_consistent': is_consistent,
        'prediction_match': prediction_match
    }

def evaluate_model_performance(pytorch_model, onnx_session, X, y):
    """评估模型性能"""
    print("🔍 评估模型性能...")
    
    # PyTorch模型推理
    with torch.no_grad():
        pytorch_outputs = pytorch_model(torch.tensor(X, dtype=torch.float32)).numpy()
    
    # ONNX模型推理
    onnx_inputs = {onnx_session.get_inputs()[0].name: X.astype(np.float32)}
    onnx_outputs = onnx_session.run(None, onnx_inputs)[0]
    
    # 计算准确率
    pytorch_preds = np.argmax(pytorch_outputs, axis=1)
    onnx_preds = np.argmax(onnx_outputs, axis=1)
    
    pytorch_accuracy = np.mean(pytorch_preds == y) * 100
    onnx_accuracy = np.mean(onnx_preds == y) * 100
    
    print(f"📊 PyTorch模型准确率: {pytorch_accuracy:.2f}%")
    print(f"📊 ONNX模型准确率: {onnx_accuracy:.2f}%")
    
    return {
        'pytorch_accuracy': pytorch_accuracy,
        'onnx_accuracy': onnx_accuracy
    }

def plot_comparison_results(results, save_path):
    """绘制比较结果"""
    print("🔍 绘制比较结果...")
    
    pytorch_outputs = results['pytorch_outputs']
    onnx_outputs = results['onnx_outputs']
    
    # 选择前10个样本进行可视化
    n_samples = min(10, pytorch_outputs.shape[0])
    
    plt.figure(figsize=(15, 10))
    
    # 绘制PyTorch和ONNX的输出比较
    for i in range(n_samples):
        plt.subplot(2, 5, i+1)
        
        x = np.arange(pytorch_outputs.shape[1])
        width = 0.35
        
        plt.bar(x - width/2, pytorch_outputs[i], width, label='PyTorch')
        plt.bar(x + width/2, onnx_outputs[i], width, label='ONNX')
        
        plt.title(f'样本 {i+1}')
        plt.xlabel('类别')
        plt.ylabel('输出值')
        plt.xticks(x)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ 比较结果图表已保存到: {save_path}")

def main():
    """主函数"""
    print("🔍 验证ONNX格式的简化版气孔检测器模型")
    print("=" * 60)
    
    # 路径设置
    checkpoint_path = "experiments/simplified_airbubble_detector/simplified_airbubble_best.pth"
    onnx_path = "deployment/onnx_models/simplified_airbubble_detector.onnx"
    results_path = "experiments/simplified_airbubble_detector/onnx_validation_results.png"
    
    # 加载模型
    pytorch_model = load_pytorch_model(checkpoint_path)
    onnx_session = load_onnx_model(onnx_path)
    
    if not pytorch_model or not onnx_session:
        return
    
    # 生成测试数据
    X, y = generate_test_data(num_samples=200)
    
    # 比较模型输出
    comparison_results = compare_model_outputs(pytorch_model, onnx_session, X)
    
    # 评估模型性能
    performance_results = evaluate_model_performance(pytorch_model, onnx_session, X, y)
    
    # 绘制比较结果
    plot_comparison_results(comparison_results, results_path)
    
    print("\n📋 验证结果摘要:")
    print(f"  - 模型输出一致性: {'✅ 通过' if comparison_results['is_consistent'] else '❌ 失败'}")
    print(f"  - 预测结果匹配率: {comparison_results['prediction_match']:.2f}%")
    print(f"  - PyTorch模型准确率: {performance_results['pytorch_accuracy']:.2f}%")
    print(f"  - ONNX模型准确率: {performance_results['onnx_accuracy']:.2f}%")
    
    print("\n✅ 验证完成")

if __name__ == "__main__":
    main()