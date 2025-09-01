#!/usr/bin/env python3
"""
检查模型检查点的ONNX转换状态
"""

import os
import sys
import json
import re
from datetime import datetime
from collections import defaultdict

# 目录路径
CHECKPOINTS_DIR = "/home/aaa/ws/bioastModel/checkpoints"
ONNX_MODELS_DIR = "/home/aaa/ws/bioastModel/onnx_models"
CONVERTERS_DIR = "/home/aaa/ws/bioastModel/converters"

def get_model_checkpoints():
    """获取所有模型检查点文件（不包括备份目录中的文件）"""
    checkpoints = []
    for file in os.listdir(CHECKPOINTS_DIR):
        if file.endswith('.pth') and not file.startswith('backup'):
            checkpoints.append(file)
    return sorted(checkpoints)

def get_onnx_models():
    """获取所有ONNX模型文件"""
    if not os.path.exists(ONNX_MODELS_DIR):
        return []
    
    onnx_files = []
    for file in os.listdir(ONNX_MODELS_DIR):
        if file.endswith('.onnx'):
            onnx_files.append(file)
    return sorted(onnx_files)

def get_converter_scripts():
    """获取所有转换器脚本"""
    if not os.path.exists(CONVERTERS_DIR):
        return []
    
    converter_scripts = []
    for file in os.listdir(CONVERTERS_DIR):
        if file.startswith('convert_') and file.endswith('.py'):
            converter_scripts.append(file)
    return sorted(converter_scripts)

def extract_model_type(filename):
    """从文件名中提取模型类型"""
    # 移除时间戳和后缀
    model_type = re.sub(r'_\d{8}_\d{6}_best\.pth$', '', filename)
    return model_type

def check_onnx_conversion_status():
    """检查模型检查点的ONNX转换状态"""
    # 获取所有模型检查点
    checkpoints = get_model_checkpoints()
    print(f"📊 找到 {len(checkpoints)} 个模型检查点")
    
    # 获取所有ONNX模型
    onnx_models = get_onnx_models()
    print(f"📊 找到 {len(onnx_models)} 个ONNX模型")
    
    # 获取所有转换器脚本
    converter_scripts = get_converter_scripts()
    print(f"📊 找到 {len(converter_scripts)} 个转换器脚本")
    
    # 检查每个模型检查点的转换状态
    conversion_status = []
    
    for checkpoint in checkpoints:
        model_type = extract_model_type(checkpoint)
        
        # 检查是否有对应的ONNX模型
        onnx_exists = False
        matching_onnx = []
        for onnx_file in onnx_models:
            if model_type.lower() in onnx_file.lower():
                onnx_exists = True
                matching_onnx.append(onnx_file)
        
        # 检查是否有对应的转换器脚本
        converter_exists = False
        matching_converter = None
        for script in converter_scripts:
            if model_type.lower() in script.lower():
                converter_exists = True
                matching_converter = script
                break
        
        # 记录状态
        status = {
            'checkpoint': checkpoint,
            'model_type': model_type,
            'onnx_converted': onnx_exists,
            'matching_onnx': matching_onnx if onnx_exists else None,
            'converter_exists': converter_exists,
            'converter_script': matching_converter
        }
        conversion_status.append(status)
    
    return conversion_status

def generate_conversion_report(status_list):
    """生成转换状态报告"""
    # 统计数据
    total_models = len(status_list)
    converted_models = sum(1 for status in status_list if status['onnx_converted'])
    unconverted_models = total_models - converted_models
    models_with_converter = sum(1 for status in status_list if status['converter_exists'])
    models_without_converter = total_models - models_with_converter
    
    # 打印报告
    print("\n" + "="*80)
    print("📊 ONNX转换状态报告")
    print("="*80)
    print(f"总模型数: {total_models}")
    print(f"已转换为ONNX: {converted_models} ({converted_models/total_models*100:.1f}%)")
    print(f"未转换为ONNX: {unconverted_models} ({unconverted_models/total_models*100:.1f}%)")
    print(f"有转换器脚本: {models_with_converter} ({models_with_converter/total_models*100:.1f}%)")
    print(f"无转换器脚本: {models_without_converter} ({models_without_converter/total_models*100:.1f}%)")
    print("="*80)
    
    # 已转换模型列表
    print("\n🟢 已转换为ONNX的模型:")
    for i, status in enumerate(sorted([s for s in status_list if s['onnx_converted']], key=lambda x: x['model_type']), 1):
        print(f"{i}. {status['model_type']} -> {', '.join(status['matching_onnx'])}")
    
    # 未转换但有转换器的模型列表
    print("\n🟡 未转换但有转换器脚本的模型:")
    for i, status in enumerate(sorted([s for s in status_list if not s['onnx_converted'] and s['converter_exists']], key=lambda x: x['model_type']), 1):
        print(f"{i}. {status['model_type']} (转换器: {status['converter_script']})")
    
    # 未转换且无转换器的模型列表
    print("\n🔴 未转换且无转换器脚本的模型:")
    for i, status in enumerate(sorted([s for s in status_list if not s['onnx_converted'] and not s['converter_exists']], key=lambda x: x['model_type']), 1):
        print(f"{i}. {status['model_type']}")
    
    # 保存报告为JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_models': total_models,
        'converted_models': converted_models,
        'unconverted_models': unconverted_models,
        'models_with_converter': models_with_converter,
        'models_without_converter': models_without_converter,
        'conversion_status': status_list
    }
    
    with open('onnx_conversion_status_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 详细报告已保存至: onnx_conversion_status_report.json")
    
    # 生成转换脚本
    generate_conversion_script(status_list)

def generate_conversion_script(status_list):
    """生成用于转换未转换模型的脚本"""
    # 找出未转换但有转换器的模型
    models_to_convert = [s for s in status_list if not s['onnx_converted'] and s['converter_exists']]
    
    if not models_to_convert:
        print("\n✅ 所有有转换器的模型都已转换为ONNX")
        return
    
    # 生成转换脚本
    script_content = """#!/usr/bin/env python3
\"\"\"
批量转换模型到ONNX格式
\"\"\"

import os
import sys
import subprocess
import time
from datetime import datetime

def run_converter(converter_script, model_type):
    \"\"\"运行转换器脚本\"\"\"
    print(f"\\n{'='*80}")
    print(f"🔄 转换模型: {model_type}")
    print(f"🔄 使用转换器: {converter_script}")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(['python3', converter_script], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               check=True)
        print(f"✅ 转换成功: {model_type}")
        print(f"📝 输出:\\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {model_type}")
        print(f"📝 错误:\\n{e.stderr}")
        return False

def main():
    \"\"\"主函数\"\"\"
    print("🚀 开始批量转换模型到ONNX格式...")
    
    # 转换器脚本列表
    converters = [
"""
    
    # 添加转换器脚本
    for status in models_to_convert:
        script_content += f"        ('{os.path.join('converters', status['converter_script'])}', '{status['model_type']}'),\n"
    
    script_content += """    ]
    
    # 运行转换器
    results = []
    for converter_script, model_type in converters:
        success = run_converter(converter_script, model_type)
        results.append({
            'model_type': model_type,
            'converter_script': converter_script,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        # 等待一段时间，避免资源冲突
        time.sleep(1)
    
    # 打印结果摘要
    print("\\n" + "="*80)
    print("📊 转换结果摘要")
    print("="*80)
    print(f"总计: {len(converters)} 个模型")
    print(f"成功: {sum(1 for r in results if r['success'])} 个")
    print(f"失败: {sum(1 for r in results if not r['success'])} 个")
    
    # 打印失败列表
    failed = [r for r in results if not r['success']]
    if failed:
        print("\\n❌ 转换失败的模型:")
        for i, r in enumerate(failed, 1):
            print(f"{i}. {r['model_type']} (转换器: {r['converter_script']})")
    
    print("\\n✅ 批量转换完成")

if __name__ == "__main__":
    main()
"""
    
    # 保存脚本
    with open('convert_remaining_models_to_onnx.py', 'w') as f:
        f.write(script_content)
    
    print(f"\n📄 已生成转换脚本: convert_remaining_models_to_onnx.py")
    print(f"   该脚本将转换 {len(models_to_convert)} 个未转换但有转换器的模型")

if __name__ == "__main__":
    print("🚀 开始检查模型检查点的ONNX转换状态...")
    status_list = check_onnx_conversion_status()
    generate_conversion_report(status_list)