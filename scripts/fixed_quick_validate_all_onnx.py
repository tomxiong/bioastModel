"""
修复版快速验证所有ONNX模型的基本状态
"""

import os
import sys
import logging
import torch
import numpy as np
import onnxruntime as ort
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config.model_configs import MODEL_CONFIGS

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def check_onnx_model_basic(model_name):
    """检查ONNX模型的基本信息"""
    try:
        onnx_path = Path(f"onnx_models/{model_name}.onnx")
        
        if not onnx_path.exists():
            return {
                'model_name': model_name,
                'exists': False,
                'size_mb': None,
                'input_shape': None,
                'output_shape': None,
                'actual_output_shape': None,
                'loadable': False,
                'inference_success': False,
                'error': 'File not found'
            }
        
        # 获取文件大小
        size_bytes = onnx_path.stat().st_size
        size_mb = round(size_bytes / (1024 * 1024), 2)
        
        # 尝试加载ONNX模型
        session = ort.InferenceSession(str(onnx_path))
        
        # 获取输入输出信息
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        input_shape = input_info.shape
        output_shape = output_info.shape
        
        # 根据模型配置或ONNX模型的实际输入形状创建测试输入
        if len(input_shape) == 4:  # [batch, channels, height, width]
            # 从ONNX模型的输入形状获取实际尺寸
            if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                height, width = input_shape[2], input_shape[3]
            else:
                # 如果是动态形状，使用模型配置中的input_size
                if model_name in MODEL_CONFIGS:
                    size = MODEL_CONFIGS[model_name]['input_size']
                    height, width = size, size
                else:
                    height, width = 70, 70  # 默认尺寸
            
            test_input = np.random.randn(1, 3, height, width).astype(np.float32)
        else:
            # 处理其他形状的输入
            if all(isinstance(dim, int) for dim in input_shape):
                test_input = np.random.randn(*input_shape).astype(np.float32)
            else:
                # 动态形状，使用batch_size=1
                shape = [1 if isinstance(dim, str) else dim for dim in input_shape]
                test_input = np.random.randn(*shape).astype(np.float32)
        
        try:
            output = session.run(None, {input_info.name: test_input})
            inference_success = True
            actual_output_shape = output[0].shape
        except Exception as e:
            inference_success = False
            actual_output_shape = f"Error: {str(e)}"
        
        return {
            'model_name': model_name,
            'exists': True,
            'size_mb': size_mb,
            'input_shape': str(input_shape),
            'output_shape': str(output_shape),
            'actual_output_shape': str(actual_output_shape),
            'loadable': True,
            'inference_success': inference_success,
            'error': None if inference_success else str(actual_output_shape)
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'exists': onnx_path.exists() if 'onnx_path' in locals() else False,
            'size_mb': None,
            'input_shape': None,
            'output_shape': None,
            'actual_output_shape': None,
            'loadable': False,
            'inference_success': False,
            'error': str(e)
        }

def main():
    """主函数"""
    logging.info("开始快速验证所有ONNX模型...")
    
    # 所有模型列表
    all_models = [
        'simplified_airbubble_detector',
        'efficientnet_b0',
        'resnet18_improved',
        'convnext_tiny',
        'coatnet',
        'vit_tiny',
        'mic_mobilenetv3',
        'micro_vit',
        'airbubble_hybrid_net'
    ]
    
    results = []
    
    for model_name in all_models:
        logging.info(f"检查模型: {model_name}")
        result = check_onnx_model_basic(model_name)
        results.append(result)
        print(f"✓ {model_name}: {'成功' if result['inference_success'] else '失败'}")
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 创建报告目录
    report_dir = Path("reports/quick_onnx_validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存CSV报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = report_dir / f"fixed_validation_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # 生成Markdown报告
    md_path = report_dir / f"fixed_validation_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 修复后ONNX模型验证报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 验证结果概览\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # 统计信息
        total_models = len(df)
        existing_models = len(df[df['exists'] == True])
        loadable_models = len(df[df['loadable'] == True])
        inference_success_models = len(df[df['inference_success'] == True])
        
        f.write("## 统计信息\n\n")
        f.write(f"- 总模型数: {total_models}\n")
        f.write(f"- 文件存在: {existing_models}/{total_models}\n")
        f.write(f"- 可加载: {loadable_models}/{total_models}\n")
        f.write(f"- 推理成功: {inference_success_models}/{total_models}\n")
        
        # 模型大小统计
        size_df = df[df['size_mb'].notna()].sort_values('size_mb')
        if len(size_df) > 0:
            f.write(f"\n## 模型大小分析\n\n")
            f.write(f"- 最小模型: {size_df.iloc[0]['model_name']} ({size_df.iloc[0]['size_mb']} MB)\n")
            f.write(f"- 最大模型: {size_df.iloc[-1]['model_name']} ({size_df.iloc[-1]['size_mb']} MB)\n")
            f.write(f"- 总大小: {size_df['size_mb'].sum():.2f} MB\n")
        
        # 修复前后对比
        f.write(f"\n## 修复说明\n\n")
        f.write("### 主要修复内容\n")
        f.write("1. **输入尺寸自适应**: 根据ONNX模型的实际输入形状创建测试数据\n")
        f.write("2. **动态形状处理**: 正确处理包含动态维度的模型\n")
        f.write("3. **模型配置集成**: 使用model_configs.py中的配置信息\n")
        f.write("4. **错误处理增强**: 更详细的错误信息和异常处理\n\n")
        
        # 错误信息
        error_df = df[df['error'].notna()]
        if len(error_df) > 0:
            f.write(f"## 仍存在的问题\n\n")
            for _, row in error_df.iterrows():
                f.write(f"- **{row['model_name']}**: {row['error']}\n")
        else:
            f.write(f"## ✅ 所有模型验证通过\n\n")
            f.write("所有ONNX模型都能正常加载和推理，输入尺寸不匹配问题已解决。\n")
    
    logging.info(f"修复后验证报告已保存至: {md_path}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("修复后ONNX模型验证完成")
    print("="*60)
    print(f"总模型数: {total_models}")
    print(f"文件存在: {existing_models}/{total_models}")
    print(f"可加载: {loadable_models}/{total_models}")
    print(f"推理成功: {inference_success_models}/{total_models}")
    if len(size_df) > 0:
        print(f"总大小: {size_df['size_mb'].sum():.2f} MB")
    print(f"详细报告: {md_path}")
    print("="*60)
    
    return md_path

if __name__ == "__main__":
    main()