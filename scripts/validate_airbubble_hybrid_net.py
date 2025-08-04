"""
验证airbubble_hybrid_net模型的ONNX转换效果
比较原始PyTorch模型和转换后的ONNX模型在测试数据上的性能
"""

import os
import sys
import logging
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.airbubble_hybrid_net import create_airbubble_hybrid_net
from scripts.create_simple_airbubble_model import SimpleAirBubbleModel
from core.data_loader import MICDataLoader, create_data_loaders

def load_original_model():
    """加载原始AirBubbleHybridNet模型"""
    try:
        # 查找最新的检查点文件
        checkpoint_path = Path("experiments/experiment_20250803_115344/airbubble_hybrid_net/best_model.pth")
        
        if not checkpoint_path.exists():
            logging.error(f"检查点文件不存在: {checkpoint_path}")
            return None
        
        # 创建模型实例
        model = create_airbubble_hybrid_net(num_classes=2, model_size='base')
        model.eval()
        
        # 加载模型权重
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        
        # 检查权重键名是否匹配
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 尝试直接加载
            state_dict = checkpoint
        
        # 处理base_model前缀问题
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('base_model.'):
                new_key = key[len('base_model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 尝试加载处理后的权重
        model.load_state_dict(new_state_dict)
        
        logging.info("原始模型加载成功")
        return model
    except Exception as e:
        logging.error(f"加载原始模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_simple_model():
    """加载简化模型"""
    try:
        model = SimpleAirBubbleModel(num_classes=2)
        model.eval()
        logging.info("简化模型加载成功")
        return model
    except Exception as e:
        logging.error(f"加载简化模型失败: {e}")
        return None

def load_onnx_model(model_name):
    """加载ONNX模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        model: ONNX模型
    """
    try:
        onnx_path = Path(f"onnx_models/{model_name}.onnx")
        
        if not onnx_path.exists():
            logging.error(f"ONNX模型文件不存在: {onnx_path}")
            return None
        
        # 创建ONNX运行时会话
        session = ort.InferenceSession(str(onnx_path))
        
        # 获取输入和输出名称
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 获取输入形状
        input_shape = session.get_inputs()[0].shape
        logging.info(f"ONNX模型输入形状: {input_shape}")
        
        logging.info(f"ONNX模型 {model_name} 加载成功")
        
        # 返回一个函数，该函数接受输入张量并返回模型输出
        def onnx_model(input_tensor):
            # 确保输入是numpy数组
            if isinstance(input_tensor, torch.Tensor):
                input_np = input_tensor.numpy()
            else:
                input_np = input_tensor
            
            # 运行ONNX模型
            output = session.run([output_name], {input_name: input_np})
            
            # 转换为torch张量
            return torch.tensor(output[0])
        
        return onnx_model
    except Exception as e:
        logging.error(f"加载ONNX模型 {model_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_models_on_test_data(original_model, simple_model, onnx_model):
    """在测试数据上比较模型的性能"""
    # 创建数据加载器
    data_loader = MICDataLoader()
    
    # 获取测试数据加载器
    _, _, test_loader = create_data_loaders(data_loader, batch_size=32)
    
    # 收集预测结果
    original_preds = []
    simple_preds = []
    onnx_preds = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            # 原始模型预测
            if original_model is not None:
                original_outputs = original_model(images)
                if isinstance(original_outputs, dict):
                    original_outputs = original_outputs['classification']
                original_preds.extend(torch.argmax(original_outputs, dim=1).numpy())
            
            # 简化模型预测
            if simple_model is not None:
                simple_outputs = simple_model(images)
                simple_preds.extend(torch.argmax(simple_outputs, dim=1).numpy())
            
            # ONNX模型预测
            if onnx_model is not None:
                onnx_outputs = onnx_model(images)
                onnx_preds.extend(torch.argmax(onnx_outputs, dim=1).numpy())
            
            # 真实标签
            true_labels.extend(labels.numpy())
    
    # 转换为numpy数组
    true_labels = np.array(true_labels)
    
    # 计算性能指标
    results = {}
    
    if original_model is not None:
        original_preds = np.array(original_preds)
        results['original'] = {
            'accuracy': accuracy_score(true_labels, original_preds),
            'f1': f1_score(true_labels, original_preds, average='weighted'),
            'predictions': original_preds
        }
    
    if simple_model is not None:
        simple_preds = np.array(simple_preds)
        results['simple'] = {
            'accuracy': accuracy_score(true_labels, simple_preds),
            'f1': f1_score(true_labels, simple_preds, average='weighted'),
            'predictions': simple_preds
        }
    
    if onnx_model is not None:
        onnx_preds = np.array(onnx_preds)
        results['onnx'] = {
            'accuracy': accuracy_score(true_labels, onnx_preds),
            'f1': f1_score(true_labels, onnx_preds, average='weighted'),
            'predictions': onnx_preds
        }
    
    # 计算模型之间的一致性
    if 'original' in results and 'simple' in results:
        results['original_vs_simple'] = {
            'agreement': np.mean(results['original']['predictions'] == results['simple']['predictions'])
        }
    
    if 'original' in results and 'onnx' in results:
        results['original_vs_onnx'] = {
            'agreement': np.mean(results['original']['predictions'] == results['onnx']['predictions'])
        }
    
    if 'simple' in results and 'onnx' in results:
        results['simple_vs_onnx'] = {
            'agreement': np.mean(results['simple']['predictions'] == results['onnx']['predictions'])
        }
    
    # 生成报告
    reports_dir = Path("reports/airbubble_model_validation")
    reports_dir.mkdir(exist_ok=True, parents=True)
    
    report = f"""# AirBubbleHybridNet模型验证报告

## 性能指标

| 模型 | 准确率 | F1分数 |
|-----|-------|-------|
"""
    
    if 'original' in results:
        report += f"| 原始模型 | {results['original']['accuracy']:.4f} | {results['original']['f1']:.4f} |\n"
    
    if 'simple' in results:
        report += f"| 简化模型 | {results['simple']['accuracy']:.4f} | {results['simple']['f1']:.4f} |\n"
    
    if 'onnx' in results:
        report += f"| ONNX模型 | {results['onnx']['accuracy']:.4f} | {results['onnx']['f1']:.4f} |\n"
    
    report += f"""
## 模型一致性

"""
    
    if 'original_vs_simple' in results:
        agreement = results['original_vs_simple']['agreement']
        report += f"- 原始模型与简化模型预测一致性: {agreement:.4f} ({int(agreement * len(true_labels))}/{len(true_labels)})\n"
    
    if 'original_vs_onnx' in results:
        agreement = results['original_vs_onnx']['agreement']
        report += f"- 原始模型与ONNX模型预测一致性: {agreement:.4f} ({int(agreement * len(true_labels))}/{len(true_labels)})\n"
    
    if 'simple_vs_onnx' in results:
        agreement = results['simple_vs_onnx']['agreement']
        report += f"- 简化模型与ONNX模型预测一致性: {agreement:.4f} ({int(agreement * len(true_labels))}/{len(true_labels)})\n"
    
    report += f"""
## 详细分类报告

### 原始模型

```
"""
    
    if 'original' in results:
        report += classification_report(true_labels, results['original']['predictions'])
    
    report += f"""
```

### 简化模型

```
"""
    
    if 'simple' in results:
        report += classification_report(true_labels, results['simple']['predictions'])
    
    report += f"""
```

### ONNX模型

```
"""
    
    if 'onnx' in results:
        report += classification_report(true_labels, results['onnx']['predictions'])
    
    report += f"""
```

## 结论

"""
    
    # 添加结论
    if 'original' in results and 'onnx' in results:
        accuracy_diff = abs(results['original']['accuracy'] - results['onnx']['accuracy'])
        agreement = results['original_vs_onnx']['agreement']
        
        if accuracy_diff < 0.01 and agreement > 0.99:
            report += "ONNX模型与原始PyTorch模型表现几乎完全一致，转换过程没有导致性能下降。\n"
        elif accuracy_diff < 0.05 and agreement > 0.95:
            report += "ONNX模型与原始PyTorch模型表现非常接近，转换过程导致了轻微的性能变化。\n"
        else:
            report += "ONNX模型与原始PyTorch模型表现存在差异，转换过程可能导致了一定的性能下降。\n"
    
    if 'simple' in results and 'onnx' in results:
        accuracy_diff = abs(results['simple']['accuracy'] - results['onnx']['accuracy'])
        agreement = results['simple_vs_onnx']['agreement']
        
        if accuracy_diff < 0.01 and agreement > 0.99:
            report += "ONNX模型与简化PyTorch模型表现几乎完全一致，转换过程没有导致性能下降。\n"
        elif accuracy_diff < 0.05 and agreement > 0.95:
            report += "ONNX模型与简化PyTorch模型表现非常接近，转换过程导致了轻微的性能变化。\n"
        else:
            report += "ONNX模型与简化PyTorch模型表现存在差异，转换过程可能导致了一定的性能下降。\n"
    
    # 保存报告
    with open(reports_dir / "validation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    logging.info(f"验证报告已保存至: {reports_dir / 'validation_report.md'}")
    
    return results

def main():
    """主函数"""
    # 加载原始模型
    original_model = load_original_model()
    
    # 加载简化模型
    simple_model = load_simple_model()
    
    # 加载ONNX模型
    onnx_model = load_onnx_model("airbubble_hybrid_net")
    
    # 在测试数据上比较模型
    evaluate_models_on_test_data(original_model, simple_model, onnx_model)

if __name__ == "__main__":
    main()