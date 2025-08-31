"""
MobileNetV5 Environment Setup Script
检查和验证环境配置
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("要求Python 3.8或更高版本")
        return False
    else:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True


def check_virtual_environment():
    """检查是否在虚拟环境中"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 已在虚拟环境中")
        return True
    else:
        print("❌ 未检测到虚拟环境")
        print("请先激活虚拟环境:")
        print("  Windows: .venv\\Scripts\\activate")
        print("  Linux/Mac: source .venv/bin/activate")
        return False


def check_pytorch():
    """检查PyTorch安装"""
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
        return True
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def check_dependencies():
    """检查必需的依赖"""
    required_packages = [
        'torchvision',
        'numpy', 
        'matplotlib',
        'seaborn',
        'pandas',
        'scikit-learn',
        'tqdm',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✅ {package}")
            else:
                __import__(package)
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ 所有依赖都已安装")
        return True


def check_data_directory():
    """检查数据目录"""
    data_dir = Path("../bioast_dataset")
    if data_dir.exists():
        print(f"✅ 数据目录存在: {data_dir}")
        
        # 检查子目录
        required_subdirs = ['positive/train', 'positive/val', 'positive/test',
                          'negative/train', 'negative/val', 'negative/test']
        
        missing_subdirs = []
        for subdir in required_subdirs:
            subdir_path = data_dir / subdir
            if not subdir_path.exists():
                missing_subdirs.append(subdir)
            else:
                # 统计文件数量
                file_count = len(list(subdir_path.glob("*.png")))
                print(f"  📁 {subdir}: {file_count} 个文件")
        
        if missing_subdirs:
            print(f"❌ 缺失的子目录: {', '.join(missing_subdirs)}")
            return False
        else:
            print("✅ 数据目录结构正确")
            return True
    else:
        print(f"❌ 数据目录不存在: {data_dir}")
        return False


def check_output_directory():
    """检查输出目录"""
    output_dir = Path("../experiments/mobilenetv5")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 输出目录可写: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ 输出目录不可写: {e}")
        return False


def run_test_import():
    """运行测试导入"""
    try:
        # 测试导入MobileNetV5模块
        sys.path.append(str(Path(__file__).parent))
        from models import create_mobilenetv5
        
        # 创建测试模型
        model = create_mobilenetv5('mobilenetv5', num_classes=2, input_size=70)
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        return False


def main():
    """主检查函数"""
    print("=" * 60)
    print("MobileNetV5 环境检查")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("虚拟环境", check_virtual_environment),
        ("PyTorch", check_pytorch),
        ("依赖包", check_dependencies),
        ("数据目录", check_data_directory),
        ("输出目录", check_output_directory),
        ("模型导入", run_test_import)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n检查 {check_name}:")
        print("-" * 30)
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有检查通过！可以开始训练")
        print("\n下一步:")
        print("1. cd mobilenetv5")
        print("2. python train.py --model mobilenetv5 --config quick_test")
    else:
        print("存在问题，请先解决上述错误")
    print("=" * 60)


if __name__ == "__main__":
    main()