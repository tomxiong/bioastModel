# BioAst模型管理系统 - 手动操作指南

本指南专为需要手动进行单个模型训练、结果分析和对比分析的研究人员设计。

## 📋 目录

1. [快速开始](#快速开始)
2. [单个模型训练](#单个模型训练)
3. [结果分析](#结果分析)
4. [模型对比分析](#模型对比分析)
5. [数据集更新流程](#数据集更新流程)
6. [报告规范](#报告规范)
7. [文件组织规范](#文件组织规范)

## 🚀 快速开始

### 环境准备

```bash
# 1. 激活虚拟环境
cd d:\ws1\bioastModel
venv\Scripts\activate

# 2. 安装依赖（首次运行）
pip install -r requirements.txt

# 3. 验证环境
python -c "import torch; print('PyTorch版本:', torch.__version__)"
```

### 项目结构理解

```
bioastModel/
├── models/                 # 模型定义文件
├── scripts/               # 训练和评估脚本
├── experiments/           # 实验结果存储
├── reports/              # 分析报告
├── data/                 # 数据集
└── configs/              # 配置文件
```

## 🎯 单个模型训练

### 方法1: 使用现有训练脚本

```bash
# 训练特定模型
python scripts/train_model.py --model efficientnet_b0
python scripts/train_model.py --model resnet18_improved
python scripts/train_model.py --model airbubble_hybrid_net
```

### 方法2: 使用统一训练接口

```bash
# 使用main.py进行单模型训练
python main.py --mode train --model efficientnet_b0
```

### 方法3: 使用集成管理器（推荐）

创建训练脚本 `train_single.py`：

```python
from utils.integration import ModelLifecycleManager
from utils.config import ConfigManager
import sys

def train_single_model(model_name, data_path=None):
    """训练单个模型"""
    
    # 初始化管理器
    config = ConfigManager().get_default_config()
    manager = ModelLifecycleManager(config)
    manager.start_services()
    
    # 模型配置
    model_configs = {
        'efficientnet_b0': {
            'name': 'EfficientNet-B0',
            'description': '轻量级高效模型',
            'model_type': 'classification',
            'algorithm': 'efficientnet_b0',
            'data_config': {
                'data_path': data_path or 'bioast_dataset',
                'image_size': (70, 70),
                'batch_size': 32,
                'test_size': 0.2
            },
            'training_config': {
                'epochs': 50,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        },
        'resnet18_improved': {
            'name': 'ResNet18-Improved',
            'description': '改进版ResNet18',
            'model_type': 'classification',
            'algorithm': 'resnet18_improved',
            'data_config': {
                'data_path': data_path or 'bioast_dataset',
                'image_size': (70, 70),
                'batch_size': 32,
                'test_size': 0.2
            },
            'training_config': {
                'epochs': 50,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        },
        'airbubble_hybrid_net': {
            'name': 'AirBubble-HybridNet',
            'description': '混合架构菌落检测模型',
            'model_type': 'classification',
            'algorithm': 'airbubble_hybrid_net',
            'data_config': {
                'data_path': data_path or 'bioast_dataset',
                'image_size': (70, 70),
                'batch_size': 32,
                'test_size': 0.2
            },
            'training_config': {
                'epochs': 50,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            }
        }
    }
    
    if model_name not in model_configs:
        print(f"错误: 不支持的模型 {model_name}")
        print(f"支持的模型: {list(model_configs.keys())}")
        return None
    
    model_config = model_configs[model_name]
    
    print(f"开始训练模型: {model_config['name']}")
    
    # 创建训练工作流
    workflow_id = manager.create_training_workflow(
        model_config=model_config,
        data_config=model_config['data_config'],
        training_config=model_config['training_config']
    )
    
    print(f"工作流ID: {workflow_id}")
    
    # 执行训练
    success = manager.execute_workflow(workflow_id)
    
    if success:
        print("✅ 模型训练成功！")
        
        # 获取训练结果
        workflow_status = manager.get_workflow_status(workflow_id)
        experiment_id = workflow_status.get('experiment_id')
        
        if experiment_id:
            # 生成实验报告
            report_path = manager.generate_experiment_report(
                experiment_id=experiment_id,
                output_format='html'
            )
            print(f"📊 实验报告已生成: {report_path}")
            
            # 获取模型信息
            models = manager.list_models()
            latest_model = models[-1] if models else None
            
            if latest_model:
                print(f"🎯 模型ID: {latest_model['id']}")
                print(f"📈 性能指标: {latest_model.get('performance', {})}")
                
                return {
                    'model_id': latest_model['id'],
                    'experiment_id': experiment_id,
                    'workflow_id': workflow_id,
                    'report_path': report_path,
                    'performance': latest_model.get('performance', {})
                }
    else:
        print("❌ 模型训练失败")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python train_single.py <model_name> [data_path]")
        print("支持的模型: efficientnet_b0, resnet18_improved, airbubble_hybrid_net")
        sys.exit(1)
    
    model_name = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = train_single_model(model_name, data_path)
    if result:
        print("\n=== 训练完成 ===")
        print(f"模型ID: {result['model_id']}")
        print(f"实验ID: {result['experiment_id']}")
        print(f"报告路径: {result['report_path']}")
```

使用方法：
```bash
# 训练EfficientNet-B0
python train_single.py efficientnet_b0

# 训练ResNet18-Improved
python train_single.py resnet18_improved

# 使用自定义数据路径
python train_single.py airbubble_hybrid_net /path/to/your/dataset
```

## 📊 结果分析

### 单模型分析

创建分析脚本 `analyze_single.py`：

```python
from utils.integration import ModelLifecycleManager
from utils.config import ConfigManager
import sys

def analyze_model(model_id):
    """分析单个模型"""
    
    config = ConfigManager().get_default_config()
    manager = ModelLifecycleManager(config)
    manager.start_services()
    
    # 获取模型信息
    model = manager.get_model(model_id)
    if not model:
        print(f"错误: 找不到模型 {model_id}")
        return
    
    print(f"=== 模型分析: {model['name']} ===")
    print(f"模型ID: {model['id']}")
    print(f"创建时间: {model.get('created_at', 'N/A')}")
    print(f"模型类型: {model.get('model_type', 'N/A')}")
    print(f"算法: {model.get('algorithm', 'N/A')}")
    
    # 性能指标
    performance = model.get('performance', {})
    if performance:
        print("\n📈 性能指标:")
        for metric, value in performance.items():
            print(f"  {metric}: {value}")
    
    # 获取相关实验
    experiments = manager.list_experiments()
    model_experiments = [exp for exp in experiments if exp.get('model_id') == model_id]
    
    if model_experiments:
        print(f"\n🧪 相关实验 ({len(model_experiments)}个):")
        for exp in model_experiments:
            print(f"  - {exp['id']}: {exp.get('name', 'N/A')} (状态: {exp.get('status', 'N/A')})")
    
    # 生成详细报告
    if model_experiments:
        latest_exp = model_experiments[-1]
        report_path = manager.generate_experiment_report(
            experiment_id=latest_exp['id'],
            output_format='html'
        )
        print(f"\n📊 详细报告: {report_path}")
    
    # 可视化
    if model_experiments:
        latest_exp = model_experiments[-1]
        try:
            # 生成训练曲线
            curve_path = manager.visualize_training_curves(
                experiment_id=latest_exp['id']
            )
            print(f"📈 训练曲线: {curve_path}")
        except Exception as e:
            print(f"⚠️ 无法生成训练曲线: {e}")

def list_all_models():
    """列出所有模型"""
    
    config = ConfigManager().get_default_config()
    manager = ModelLifecycleManager(config)
    manager.start_services()
    
    models = manager.list_models()
    
    if not models:
        print("没有找到任何模型")
        return
    
    print(f"=== 所有模型 ({len(models)}个) ===")
    for i, model in enumerate(models, 1):
        performance = model.get('performance', {})
        accuracy = performance.get('accuracy', 'N/A')
        print(f"{i}. {model['name']} (ID: {model['id']})")
        print(f"   准确率: {accuracy}")
        print(f"   创建时间: {model.get('created_at', 'N/A')}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python analyze_single.py list                    # 列出所有模型")
        print("  python analyze_single.py <model_id>             # 分析特定模型")
        sys.exit(1)
    
    if sys.argv[1] == 'list':
        list_all_models()
    else:
        model_id = sys.argv[1]
        analyze_model(model_id)
```

使用方法：
```bash
# 列出所有模型
python analyze_single.py list

# 分析特定模型
python analyze_single.py model_12345
```

## 🔄 模型对比分析

创建对比脚本 `compare_models.py`：

```python
from utils.integration import ModelLifecycleManager
from utils.config import ConfigManager
import sys

def compare_models(model_ids):
    """对比多个模型"""
    
    config = ConfigManager().get_default_config()
    manager = ModelLifecycleManager(config)
    manager.start_services()
    
    print(f"=== 模型对比分析 ({len(model_ids)}个模型) ===")
    
    # 获取模型信息
    models = []
    for model_id in model_ids:
        model = manager.get_model(model_id)
        if model:
            models.append(model)
        else:
            print(f"⚠️ 警告: 找不到模型 {model_id}")
    
    if len(models) < 2:
        print("错误: 至少需要2个有效模型进行对比")
        return
    
    # 显示基本信息对比
    print("\n📋 基本信息对比:")
    print(f"{'模型名称':<20} {'模型ID':<15} {'算法':<20} {'准确率':<10}")
    print("-" * 70)
    
    for model in models:
        performance = model.get('performance', {})
        accuracy = performance.get('accuracy', 'N/A')
        print(f"{model['name']:<20} {model['id']:<15} {model.get('algorithm', 'N/A'):<20} {accuracy:<10}")
    
    # 性能指标对比
    print("\n📈 性能指标详细对比:")
    all_metrics = set()
    for model in models:
        performance = model.get('performance', {})
        all_metrics.update(performance.keys())
    
    if all_metrics:
        for metric in sorted(all_metrics):
            print(f"\n{metric}:")
            for model in models:
                performance = model.get('performance', {})
                value = performance.get(metric, 'N/A')
                print(f"  {model['name']}: {value}")
    
    # 生成对比报告
    try:
        report_path = manager.generate_comparison_report(
            model_ids=[model['id'] for model in models],
            output_format='html'
        )
        print(f"\n📊 详细对比报告: {report_path}")
    except Exception as e:
        print(f"⚠️ 无法生成对比报告: {e}")
    
    # 可视化对比
    try:
        dashboard_url = manager.create_interactive_dashboard(
            model_ids=[model['id'] for model in models]
        )
        print(f"🌐 交互式仪表板: {dashboard_url}")
    except Exception as e:
        print(f"⚠️ 无法创建仪表板: {e}")
    
    # 推荐最佳模型
    best_model = None
    best_accuracy = 0
    
    for model in models:
        performance = model.get('performance', {})
        accuracy = performance.get('accuracy', 0)
        if isinstance(accuracy, (int, float)) and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    
    if best_model:
        print(f"\n🏆 推荐模型: {best_model['name']} (准确率: {best_accuracy})")

def compare_top_models(top_n=5):
    """对比性能最好的N个模型"""
    
    config = ConfigManager().get_default_config()
    manager = ModelLifecycleManager(config)
    manager.start_services()
    
    models = manager.list_models()
    
    if not models:
        print("没有找到任何模型")
        return
    
    # 按准确率排序
    def get_accuracy(model):
        performance = model.get('performance', {})
        accuracy = performance.get('accuracy', 0)
        return accuracy if isinstance(accuracy, (int, float)) else 0
    
    sorted_models = sorted(models, key=get_accuracy, reverse=True)
    top_models = sorted_models[:top_n]
    
    print(f"=== Top {len(top_models)} 模型对比 ===")
    
    model_ids = [model['id'] for model in top_models]
    compare_models(model_ids)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python compare_models.py top [N]                # 对比性能最好的N个模型(默认5个)")
        print("  python compare_models.py <model_id1> <model_id2> [model_id3] ...  # 对比指定模型")
        sys.exit(1)
    
    if sys.argv[1] == 'top':
        top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        compare_top_models(top_n)
    else:
        model_ids = sys.argv[1:]
        compare_models(model_ids)
```

使用方法：
```bash
# 对比性能最好的5个模型
python compare_models.py top

# 对比性能最好的3个模型
python compare_models.py top 3

# 对比指定模型
python compare_models.py model_123 model_456 model_789
```

## 🔄 数据集更新流程

### 1. 数据集更新检测

创建 `check_dataset_updates.py`：

```python
import os
import hashlib
import json
from datetime import datetime

def calculate_dataset_hash(dataset_path):
    """计算数据集哈希值"""
    hash_md5 = hashlib.md5()
    
    for root, dirs, files in os.walk(dataset_path):
        for file in sorted(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

def check_dataset_changes(dataset_path, hash_file='dataset_hash.json'):
    """检查数据集是否有变化"""
    
    current_hash = calculate_dataset_hash(dataset_path)
    
    # 读取之前的哈希值
    previous_hash = None
    if os.path.exists(hash_file):
        try:
            with open(hash_file, 'r') as f:
                data = json.load(f)
                previous_hash = data.get('hash')
        except:
            pass
    
    # 保存当前哈希值
    with open(hash_file, 'w') as f:
        json.dump({
            'hash': current_hash,
            'timestamp': datetime.now().isoformat(),
            'dataset_path': dataset_path
        }, f, indent=2)
    
    if previous_hash is None:
        print("首次检查数据集")
        return True
    elif previous_hash != current_hash:
        print("⚠️ 检测到数据集变化，需要重新训练模型")
        return True
    else:
        print("✅ 数据集无变化")
        return False

if __name__ == "__main__":
    dataset_path = "bioast_dataset"  # 修改为你的数据集路径
    
    if check_dataset_changes(dataset_path):
        print("\n建议执行以下操作:")
        print("1. 重新训练所有模型")
        print("2. 更新模型性能对比")
        print("3. 生成新的分析报告")
        
        print("\n快速重训练命令:")
        models = ['efficientnet_b0', 'resnet18_improved', 'airbubble_hybrid_net']
        for model in models:
            print(f"python train_single.py {model}")
```

### 2. 批量重训练脚本

创建 `retrain_all.py`：

```python
from utils.integration import ModelLifecycleManager
from utils.config import ConfigManager
import time

def retrain_all_models(data_path=None):
    """重新训练所有模型"""
    
    models_to_train = [
        'efficientnet_b0',
        'resnet18_improved', 
        'airbubble_hybrid_net',
        'micro_vit',
        'convnext_tiny'
    ]
    
    results = []
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"开始训练: {model_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # 这里调用之前定义的train_single_model函数
        from train_single import train_single_model
        result = train_single_model(model_name, data_path)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result:
            result['training_time'] = training_time
            results.append(result)
            print(f"✅ {model_name} 训练完成 (耗时: {training_time:.2f}秒)")
        else:
            print(f"❌ {model_name} 训练失败")
    
    # 生成汇总报告
    print(f"\n{'='*50}")
    print("训练汇总")
    print(f"{'='*50}")
    
    for result in results:
        performance = result.get('performance', {})
        accuracy = performance.get('accuracy', 'N/A')
        training_time = result.get('training_time', 0)
        print(f"模型: {result['model_id']}")
        print(f"  准确率: {accuracy}")
        print(f"  训练时间: {training_time:.2f}秒")
        print(f"  报告: {result['report_path']}")
        print()
    
    # 自动生成对比分析
    if len(results) >= 2:
        print("生成模型对比分析...")
        model_ids = [result['model_id'] for result in results]
        
        config = ConfigManager().get_default_config()
        manager = ModelLifecycleManager(config)
        manager.start_services()
        
        try:
            comparison_report = manager.generate_comparison_report(
                model_ids=model_ids,
                output_format='html'
            )
            print(f"📊 对比报告: {comparison_report}")
        except Exception as e:
            print(f"⚠️ 无法生成对比报告: {e}")

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    retrain_all_models(data_path)
```

## 📋 报告规范

### 报告文件命名规范

```
reports/
├── experiments/
│   ├── exp_YYYYMMDD_HHMMSS_<model_name>.html
│   └── exp_YYYYMMDD_HHMMSS_<model_name>.json
├── comparisons/
│   ├── comparison_YYYYMMDD_HHMMSS.html
│   └── comparison_YYYYMMDD_HHMMSS.json
└── summaries/
    ├── summary_YYYYMMDD.html
    └── summary_YYYYMMDD.json
```

### 报告内容规范

每个实验报告应包含：

1. **基本信息**
   - 模型名称和ID
   - 训练时间
   - 数据集信息
   - 超参数配置

2. **性能指标**
   - 准确率 (Accuracy)
   - 精确率 (Precision)
   - 召回率 (Recall)
   - F1分数 (F1-Score)
   - 混淆矩阵

3. **训练过程**
   - 训练曲线
   - 损失函数变化
   - 验证集性能变化

4. **错误分析**
   - 错误样本分析
   - 分类错误统计

## 📁 文件组织规范

### 实验文件组织

```
experiments/
├── YYYYMMDD_HHMMSS_<model_name>/
│   ├── config.json              # 训练配置
│   ├── model.pth               # 训练好的模型
│   ├── training_log.txt        # 训练日志
│   ├── metrics.json            # 性能指标
│   ├── plots/                  # 图表文件
│   │   ├── training_curve.png
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   └── artifacts/              # 其他产物
│       ├── predictions.csv
│       └── error_samples/
```

### 配置文件模板

创建 `config_template.json`：

```json
{
  "model": {
    "name": "模型名称",
    "type": "classification",
    "algorithm": "算法名称",
    "version": "1.0.0"
  },
  "data": {
    "dataset_path": "数据集路径",
    "image_size": [70, 70],
    "batch_size": 32,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
  },
  "training": {
    "epochs": 50,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "early_stopping": {
      "patience": 10,
      "min_delta": 0.001
    }
  },
  "evaluation": {
    "metrics": ["accuracy", "precision", "recall", "f1_score"],
    "save_predictions": true,
    "save_error_analysis": true
  }
}
```

## 🔧 实用工具脚本

### 快速状态检查

创建 `quick_status.py`：

```python
from utils.integration import ModelLifecycleManager
from utils.config import ConfigManager

def quick_status():
    """快速查看系统状态"""
    
    config = ConfigManager().get_default_config()
    manager = ModelLifecycleManager(config)
    manager.start_services()
    
    # 模型统计
    models = manager.list_models()
    print(f"📊 系统状态概览")
    print(f"模型总数: {len(models)}")
    
    if models:
        # 按准确率排序
        def get_accuracy(model):
            performance = model.get('performance', {})
            accuracy = performance.get('accuracy', 0)
            return accuracy if isinstance(accuracy, (int, float)) else 0
        
        sorted_models = sorted(models, key=get_accuracy, reverse=True)
        
        print(f"\n🏆 性能排行榜 (Top 5):")
        for i, model in enumerate(sorted_models[:5], 1):
            accuracy = get_accuracy(model)
            print(f"{i}. {model['name']}: {accuracy:.4f}")
    
    # 实验统计
    experiments = manager.list_experiments()
    print(f"\n🧪 实验总数: {len(experiments)}")
    
    # 最近的实验
    if experiments:
        recent_experiments = sorted(experiments, key=lambda x: x.get('created_at', ''), reverse=True)[:3]
        print(f"\n📅 最近实验:")
        for exp in recent_experiments:
            print(f"  - {exp.get('name', 'N/A')} ({exp.get('status', 'N/A')})")

if __name__ == "__main__":
    quick_status()
```

### 清理工具

创建 `cleanup.py`：

```python
import os
import shutil
from datetime import datetime, timedelta

def cleanup_old_files(days=30):
    """清理超过指定天数的文件"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # 清理目录
    cleanup_dirs = [
        'experiments',
        'reports',
        'logs'
    ]
    
    for dir_name in cleanup_dirs:
        if not os.path.exists(dir_name):
            continue
            
        print(f"清理目录: {dir_name}")
        
        for item in os.listdir(dir_name):
            item_path = os.path.join(dir_name, item)
            
            # 获取文件/目录的修改时间
            mtime = datetime.fromtimestamp(os.path.getmtime(item_path))
            
            if mtime < cutoff_date:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  删除目录: {item}")
                else:
                    os.remove(item_path)
                    print(f"  删除文件: {item}")

if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"清理 {days} 天前的文件...")
    cleanup_old_files(days)
    print("清理完成")
```

## 🚀 完整工作流示例

### 新数据集训练完整流程

```bash
# 1. 检查数据集变化
python check_dataset_updates.py

# 2. 如果有变化，重新训练所有模型
python retrain_all.py

# 3. 查看训练结果
python quick_status.py

# 4. 对比最好的3个模型
python compare_models.py top 3

# 5. 分析特定模型
python analyze_single.py <model_id>
```

### 单模型调优流程

```bash
# 1. 训练基础模型
python train_single.py efficientnet_b0

# 2. 分析结果
python analyze_single.py <model_id>

# 3. 调整超参数后重新训练
# (修改train_single.py中的配置)
python train_single.py efficientnet_b0

# 4. 对比不同版本
python compare_models.py <model_id_v1> <model_id_v2>
```

## 📝 注意事项

1. **数据备份**: 训练前确保数据集已备份
2. **资源监控**: 训练时监控GPU/CPU使用情况
3. **日志保存**: 所有训练过程都会自动记录日志
4. **版本管理**: 每次训练都会创建新的模型版本
5. **报告归档**: 定期清理旧的报告文件

## 🔗 相关文件

- `main.py`: 系统主入口
- `utils/integration.py`: 核心管理器
- `utils/config.py`: 配置管理
- `requirements.txt`: 依赖包列表
- `README.md`: 系统完整文档

---

**提示**: 这个指南专注于手动操作，如果需要更多自动化功能，可以参考完整的系统文档。