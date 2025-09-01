#!/usr/bin/env python3
"""
批量转换模型到ONNX格式
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_converter(converter_script, model_type):
    """运行转换器脚本"""
    print(f"\n{'='*80}")
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
        print(f"📝 输出:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {model_type}")
        print(f"📝 错误:\n{e.stderr}")
        return False

def main():
    """主函数"""
    print("🚀 开始批量转换模型到ONNX格式...")
    
    # 转换器脚本列表
    converters = [
        ('converters/convert_micro_vit.py', 'micro_vit'),
    ]
    
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
    print("\n" + "="*80)
    print("📊 转换结果摘要")
    print("="*80)
    print(f"总计: {len(converters)} 个模型")
    print(f"成功: {sum(1 for r in results if r['success'])} 个")
    print(f"失败: {sum(1 for r in results if not r['success'])} 个")
    
    # 打印失败列表
    failed = [r for r in results if not r['success']]
    if failed:
        print("\n❌ 转换失败的模型:")
        for i, r in enumerate(failed, 1):
            print(f"{i}. {r['model_type']} (转换器: {r['converter_script']})")
    
    print("\n✅ 批量转换完成")

if __name__ == "__main__":
    main()
