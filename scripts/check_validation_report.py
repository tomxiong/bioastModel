"""
检查验证报告是否已经生成
"""

import os
from pathlib import Path
import time

def check_report():
    """检查验证报告是否已经生成"""
    report_path = Path("reports/simplified_detector_validation/validation_report.md")
    
    if report_path.exists():
        print(f"✅ 验证报告已生成: {report_path}")
        
        # 获取文件大小和修改时间
        size = os.path.getsize(report_path)
        mtime = os.path.getmtime(report_path)
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        
        print(f"📊 文件大小: {size/1024:.2f} KB")
        print(f"🕒 修改时间: {mtime_str}")
        
        # 检查是否有混淆矩阵图像
        cm_path = Path("reports/simplified_detector_validation/confusion_matrix.png")
        if cm_path.exists():
            print(f"✅ 混淆矩阵图像已生成: {cm_path}")
        else:
            print(f"❌ 混淆矩阵图像未生成")
        
        # 检查样本图像目录
        correct_dir = Path("reports/simplified_detector_validation/correct_samples")
        incorrect_dir = Path("reports/simplified_detector_validation/incorrect_samples")
        
        if correct_dir.exists():
            correct_samples = list(correct_dir.glob("*.png"))
            print(f"✅ 正确样本目录已创建，包含 {len(correct_samples)} 个样本图像")
        else:
            print(f"❌ 正确样本目录未创建")
        
        if incorrect_dir.exists():
            incorrect_samples = list(incorrect_dir.glob("*.png"))
            print(f"✅ 错误样本目录已创建，包含 {len(incorrect_samples)} 个样本图像")
        else:
            print(f"❌ 错误样本目录未创建")
        
        return True
    else:
        print(f"❌ 验证报告尚未生成: {report_path}")
        
        # 检查报告目录是否存在
        report_dir = Path("reports/simplified_detector_validation")
        if report_dir.exists():
            print(f"✅ 报告目录已创建: {report_dir}")
            
            # 检查目录中的文件
            files = list(report_dir.glob("*"))
            if files:
                print(f"📁 目录中包含 {len(files)} 个文件:")
                for file in files:
                    print(f"  - {file.name}")
            else:
                print(f"📁 目录为空")
        else:
            print(f"❌ 报告目录尚未创建")
        
        return False

if __name__ == "__main__":
    check_report()