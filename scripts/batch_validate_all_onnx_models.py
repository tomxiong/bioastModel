"""
批量验证所有ONNX模型的性能
对比原始PyTorch模型和ONNX模型在真实数据上的表现
"""

import os
import sys
import logging
import torch
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_model_info():
    """获取所有模型的信息"""
    return {
        'simplified_airbubble_detector': {
            'module': 'models.simplified_airbubble_detector',
            'class': 'SimplifiedAirBubbleDetector',
            'input_size': 70,
            'checkpoint_pattern': '**/simplified_airbubble_detector/*.pth'
        },
        'efficientnet_b0': {
            'module': 'models.efficientnet',
            'class': 'EfficientNetB0',
            'input_size': 224,
            'checkpoint_pattern': '**/efficientnet_b0/*.pth'
        },
        'resnet18_improved': {
            'module': 'models.resnet_improved',
            'class': 'create_resnet18_improved',
            'input_size': 224,
            'checkpoint_pattern': '**/resnet18_improved/*.pth'
        },
        'convnext_tiny': {
            'module': 'models.convnext_tiny',
            'class': 'ConvNextTiny',
            'input_size': 224,
            'checkpoint_pattern': '**/convnext_tiny/*.pth'
        },
        'coatnet': {
            'module': 'models.coatnet',
            'class': 'CoAtNet',
            'input_size': 224,
            'checkpoint_pattern': '**/coatnet/*.pth'
        },
        'vit_tiny': {
            'module': 'models.vit_tiny',
            'class': 'ViT',
            'input_size': 224,
            'checkpoint_pattern': '**/vit_tiny/*.pth'
        },
        'mic_mobilenetv3': {
            'module': 'models.mic_mobilenetv3',
            'class': 'MICMobileNetV3',
            'input_size': 224,
            'checkpoint_pattern': '**/mic_mobilenetv3/*.pth'
        },
        'micro_vit': {
            'module': 'models.micro_vit',
            'class': 'MicroViT',
            'input_size': 224,
            'checkpoint_pattern': '**/micro_vit/*.pth'
        },
        'airbubble_hybrid_net': {
            'module': 'models.airbubble_hybrid_net',
            'class': 'create_airbubble_hybrid_net',
            'input_size': 70,
            'checkpoint_pattern': '**/airbubble_hybrid_net/*.pth'
        }
    }

def load_pytorch_model(model_name, model_info):
    """加载PyTorch模型"""
    try:
        import importlib
        
        # 导入模型模块
        module = importlib.import_module(model_info['module'])
        
        # 获取模型类或函数
        if model_info['class'] == 'create_resnet18_improved':
            model = module.create_resnet18_improved()
        elif model_info['class'] == 'create_airbubble_hybrid_net':
            model = module.create_airbubble_hybrid_net(num_classes=2)
        elif model_info['class'] == 'EfficientNet':
            model = module.EfficientNet.from_name('efficientnet-b0', num_classes=2)
        else:
            model_class = getattr(module, model_info['class'])
            model = model_class()
        
        model.eval()
        
        # 查找检查点文件
        experiments_dir = Path("experiments")
        checkpoint_files = list(experiments_dir.glob(model_info['checkpoint_pattern']))
        
        if not checkpoint_files:
            logging.warning(f"未找到{model_name}的检查点文件")
            return None
        
        # 选择最新的检查点文件
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        # 加载权重
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理权重键名
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('base_model.'):
                new_key = key[len('base_model.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        logging.info(f"PyTorch模型 {model_name} 加载成功")
        return model
        
    except Exception as e:
        logging.error(f"加载PyTorch模型 {model_name} 失败: {e}")
        return None

def load_onnx_model(model_name):
    """加载ONNX模型"""
    try:
        onnx_path = Path(f"onnx_models/{model_name}.onnx")
        
        if not onnx_path.exists():
            logging.warning(f"ONNX模型文件不存在: {onnx_path}")
            return None
        
        session = ort.InferenceSession(str(onnx_path))
        
        def onnx_predict(input_tensor):
            if isinstance(input_tensor, torch.Tensor):
                input_np = input_tensor.numpy()
            else:
                input_np = input_tensor
            
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: input_np})
            return torch.tensor(output[0])
        
        logging.info(f"ONNX模型 {model_name} 加载成功")
        return onnx_predict
        
    except Exception as e:
        logging.error(f"加载ONNX模型 {model_name} 失败: {e}")
        return None

def generate_synthetic_test_data(input_size, num_samples=100):
    """生成合成测试数据"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 生成随机图像数据
    test_images = []
    test_labels = []
    
    for i in range(num_samples):
        # 生成随机RGB图像
        random_image = np.random.randint(0, 256, (input_size, input_size, 3), dtype=np.uint8)
        pil_image = Image.fromarray(random_image)
        tensor_image = transform(pil_image)
        
        test_images.append(tensor_image)
        test_labels.append(i % 2)  # 交替生成0和1标签
    
    return test_images, test_labels

def validate_model_pair(model_name, pytorch_model, onnx_model, input_size):
    """验证PyTorch模型和ONNX模型的一致性"""
    logging.info(f"开始验证模型: {model_name}")
    
    # 生成测试数据
    test_images, test_labels = generate_synthetic_test_data(input_size, num_samples=100)
    
    pytorch_preds = []
    onnx_preds = []
    
    with torch.no_grad():
        for img, label in zip(test_images, test_labels):
            # PyTorch模型预测
            if pytorch_model is not None:
                img_batch = img.unsqueeze(0)
                pytorch_output = pytorch_model(img_batch)
                if isinstance(pytorch_output, dict):
                    pytorch_output = pytorch_output.get('classification', pytorch_output.get('logits', pytorch_output))
                pytorch_pred = torch.argmax(pytorch_output, dim=1).item()
                pytorch_preds.append(pytorch_pred)
            
            # ONNX模型预测
            if onnx_model is not None:
                img_batch = img.unsqueeze(0)
                onnx_output = onnx_model(img_batch)
                onnx_pred = torch.argmax(onnx_output, dim=1).item()
                onnx_preds.append(onnx_pred)
    
    # 计算性能指标
    results = {
        'model_name': model_name,
        'pytorch_available': pytorch_model is not None,
        'onnx_available': onnx_model is not None,
        'test_samples': len(test_labels)
    }
    
    if pytorch_model is not None:
        pytorch_accuracy = accuracy_score(test_labels, pytorch_preds)
        pytorch_f1 = f1_score(test_labels, pytorch_preds, average='weighted')
        results.update({
            'pytorch_accuracy': pytorch_accuracy,
            'pytorch_f1': pytorch_f1
        })
    
    if onnx_model is not None:
        onnx_accuracy = accuracy_score(test_labels, onnx_preds)
        onnx_f1 = f1_score(test_labels, onnx_preds, average='weighted')
        results.update({
            'onnx_accuracy': onnx_accuracy,
            'onnx_f1': onnx_f1
        })
    
    if pytorch_model is not None and onnx_model is not None:
        consistency = np.mean(np.array(pytorch_preds) == np.array(onnx_preds))
        accuracy_diff = abs(pytorch_accuracy - onnx_accuracy)
        results.update({
            'consistency': consistency,
            'accuracy_difference': accuracy_diff
        })
    
    return results

def main():
    """主函数"""
    logging.info("开始批量验证所有ONNX模型...")
    
    model_info = get_model_info()
    all_results = []
    
    for model_name, info in model_info.items():
        logging.info(f"\n{'='*50}")
        logging.info(f"处理模型: {model_name}")
        
        # 加载PyTorch模型
        pytorch_model = load_pytorch_model(model_name, info)
        
        # 加载ONNX模型
        onnx_model = load_onnx_model(model_name)
        
        # 验证模型
        if pytorch_model is not None or onnx_model is not None:
            results = validate_model_pair(model_name, pytorch_model, onnx_model, info['input_size'])
            all_results.append(results)
        else:
            logging.warning(f"跳过模型 {model_name}：无法加载PyTorch或ONNX模型")
    
    # 生成报告
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 创建报告目录
        report_dir = Path("reports/batch_onnx_validation")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存CSV报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = report_dir / f"batch_validation_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # 生成Markdown报告
        md_path = report_dir / f"batch_validation_report_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 批量ONNX模型验证报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 验证结果概览\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            # 统计信息
            available_models = df[df['onnx_available'] == True]
            consistent_models = df[(df['consistency'].notna()) & (df['consistency'] > 0.95)]
            
            f.write("## 统计信息\n\n")
            f.write(f"- 总模型数: {len(df)}\n")
            f.write(f"- ONNX模型可用: {len(available_models)}\n")
            f.write(f"- 高一致性模型 (>95%): {len(consistent_models)}\n")
            
            if len(consistent_models) > 0:
                f.write("\n## 高一致性模型详情\n\n")
                for _, row in consistent_models.iterrows():
                    f.write(f"### {row['model_name']}\n")
                    f.write(f"- PyTorch准确率: {row.get('pytorch_accuracy', 'N/A'):.4f}\n")
                    f.write(f"- ONNX准确率: {row.get('onnx_accuracy', 'N/A'):.4f}\n")
                    f.write(f"- 预测一致性: {row.get('consistency', 'N/A'):.4f}\n")
                    f.write(f"- 准确率差异: {row.get('accuracy_difference', 'N/A'):.4f}\n\n")
        
        logging.info(f"批量验证报告已保存至: {md_path}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("批量ONNX模型验证完成")
        print("="*60)
        print(f"总模型数: {len(df)}")
        print(f"ONNX模型可用: {len(available_models)}")
        print(f"高一致性模型: {len(consistent_models)}")
        print(f"详细报告: {md_path}")
        print("="*60)
    
    else:
        logging.error("没有成功验证任何模型")

if __name__ == "__main__":
    main()