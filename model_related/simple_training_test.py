#!/usr/bin/env python3
"""
简单的训练环境测试脚本
用于验证Python虚拟环境和依赖包是否正确安装
"""

import sys
import os
from pathlib import Path

def test_core_packages():
    """测试核心包是否可用"""
    print("=== 测试核心机器学习包 ===")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} - 数值计算库")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__} - 数据处理库")
    except ImportError as e:
        print(f"✗ Pandas 导入失败: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__} - 机器学习库")
    except ImportError as e:
        print(f"✗ Scikit-learn 导入失败: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__} - 可视化库")
    except ImportError as e:
        print(f"✗ Matplotlib 导入失败: {e}")
        return False
    
    return True

def test_deep_learning_packages():
    """测试深度学习包是否可用"""
    print("\n=== 测试深度学习包 ===")
    
    # 测试PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  - CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA设备数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"⚠ PyTorch 暂未安装完成: {e}")
    
    # 测试TensorFlow
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        print(f"  - GPU可用: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except ImportError as e:
        print(f"⚠ TensorFlow 暂未安装完成: {e}")

def test_environment():
    """测试Python环境"""
    print("=== Python环境信息 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"虚拟环境: {os.environ.get('VIRTUAL_ENV', '未激活')}")

def create_sample_data():
    """创建示例数据用于测试"""
    print("\n=== 创建示例数据 ===")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        
        # 创建示例分类数据
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=10,
            n_classes=2,
            random_state=42
        )
        
        # 转换为DataFrame
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # 保存数据
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        data_path = data_dir / 'sample_data.csv'
        df.to_csv(data_path, index=False)
        
        print(f"✓ 示例数据已创建: {data_path}")
        print(f"  - 样本数量: {len(df)}")
        print(f"  - 特征数量: {X.shape[1]}")
        print(f"  - 类别分布: {pd.Series(y).value_counts().to_dict()}")
        
        return str(data_path)
        
    except Exception as e:
        print(f"✗ 创建示例数据失败: {e}")
        return None

def test_simple_training():
    """测试简单的模型训练"""
    print("\n=== 测试简单模型训练 ===")
    
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        
        # 创建示例数据
        data_path = create_sample_data()
        if not data_path:
            return False
        
        # 加载数据
        df = pd.read_csv(data_path)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        print("训练随机森林模型...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ 模型训练完成")
        print(f"  - 训练样本: {len(X_train)}")
        print(f"  - 测试样本: {len(X_test)}")
        print(f"  - 测试准确率: {accuracy:.4f}")
        
        # 保存模型
        try:
            import joblib
            model_dir = Path('models')
            model_dir.mkdir(exist_ok=True)
            
            model_path = model_dir / 'sample_model.pkl'
            joblib.dump(model, model_path)
            print(f"✓ 模型已保存: {model_path}")
        except Exception as e:
            print(f"⚠ 模型保存失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型训练失败: {e}")
        return False

def main():
    """主函数"""
    print("BioAst模型训练环境测试")
    print("=" * 50)
    
    # 测试环境
    test_environment()
    
    # 测试核心包
    core_ok = test_core_packages()
    
    # 测试深度学习包
    test_deep_learning_packages()
    
    # 如果核心包可用，测试简单训练
    if core_ok:
        training_ok = test_simple_training()
        
        print("\n=== 测试总结 ===")
        if training_ok:
            print("✓ 训练环境准备完成！")
            print("✓ 可以开始进行模型训练")
        else:
            print("⚠ 训练环境基本可用，但存在一些问题")
    else:
        print("\n=== 测试总结 ===")
        print("✗ 核心包安装不完整，请等待安装完成")
    
    print("\n下一步操作建议:")
    print("1. 等待所有包安装完成")
    print("2. 运行 python train_single_model.py 进行单模型训练")
    print("3. 运行 python train_all_models.py 进行批量训练")
    print("4. 查看 README.md 了解更多使用方法")

if __name__ == '__main__':
    main()