# System Patterns

菌落检测项目的代码模式和标准文档
2025-01-02 更新 - 系统模式建立

## Coding Patterns

### 环境管理模式
```bash
# 必须使用本地虚拟环境 .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 验证环境
python -c "import sys; print('Python path:', sys.executable)"
```

**重要规则**: 所有Python命令和脚本执行都必须在.venv虚拟环境中进行，确保依赖包的一致性和隔离性。

### 配置管理模式
```python
# 统一配置加载模式
from core.config import get_model_config, get_training_config

# 获取模型配置
model_config = get_model_config('resnet18_improved')
training_config = get_training_config('default')
```

### 路径管理模式
```python
# 统一路径访问
from core.config import (
    EXPERIMENTS_DIR, 
    REPORTS_DIR,
    get_latest_experiment_path,
    get_model_report_path
)

# 获取实验路径
experiment_path = get_latest_experiment_path('resnet18_improved')
report_path = get_model_report_path('resnet18_improved')
```

### 错误处理模式
```python
# 标准错误处理和用户反馈
try:
    # 操作代码
    result = perform_operation()
    print(f"✅ 操作成功: {result}")
except FileNotFoundError as e:
    print(f"❌ 文件未找到: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 操作失败: {str(e)}")
    sys.exit(1)
```

### 向后兼容模式
```python
# 配置文件兼容性检查
config_file = experiment_path / 'config.json'
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    # 创建默认配置用于向后兼容
    config = create_default_config(model_name)
    print(f"⚠️  使用默认配置 (向后兼容)")
```

## Architectural Patterns

### 分层架构模式
```
应用层 (scripts/) 
    ↓
配置层 (core/config/)
    ↓  
业务层 (training/, models/)
    ↓
数据层 (experiments/, reports/)
```

### 工厂模式 - 模型创建
```python
# models/__init__.py 中的模型工厂
def create_model(model_name, num_classes=2):
    if model_name == 'efficientnet_b0':
        return EfficientNetB0(num_classes)
    elif model_name == 'resnet18_improved':
        return ResNet18Improved(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

### 策略模式 - 训练配置
```python
# 不同模型使用不同训练策略
training_configs = {
    'efficientnet_b0': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'adam'
    },
    'resnet18_improved': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'sgd'
    }
}
```

### 观察者模式 - 训练监控
```python
# 训练过程中的状态监控
class TrainingMonitor:
    def on_epoch_end(self, epoch, logs):
        # 记录训练历史
        # 保存检查点
        # 更新可视化
        pass
```

## Testing Patterns

### 脚本测试模式
```python
# 干运行模式用于测试
if args.dry_run:
    print("📋 配置预览 (干运行模式):")
    print(json.dumps(config, indent=2))
    return
```

### 验证模式
```python
# 实验验证标准流程
def validate_experiment(experiment_path):
    required_files = ['best_model.pth', 'training_history.json']
    optional_files = ['config.json']
    
    # 检查必需文件
    missing_files = []
    for file in required_files:
        if not (experiment_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(f"缺少必需文件: {missing_files}")
    
    return True
```

### 兼容性测试模式
```python
# 向后兼容性测试
def test_backward_compatibility():
    # 测试无config.json的实验
    # 测试默认配置生成
    # 测试路径兼容性
    pass
```

### 集成测试模式
```python
# 端到端工作流测试
def test_complete_workflow():
    # 1. 配置加载测试
    # 2. 模型创建测试  
    # 3. 数据加载测试
    # 4. 训练流程测试
    # 5. 评估流程测试
    pass
```

## File Organization Patterns

### 标准目录结构
```
项目根目录/
├── core/                   # 核心功能模块
│   └── config/            # 配置管理
├── scripts/               # 可执行脚本
├── models/                # 模型定义
├── training/              # 训练相关
├── experiments/           # 实验结果
│   └── experiment_ID/     # 单个实验
│       └── model_name/    # 模型特定结果
├── reports/               # 评估报告
│   ├── individual/        # 单模型报告
│   └── comparison/        # 对比报告
└── memory-bank/           # 项目记忆
```

### 命名约定
- **脚本文件**: 动词_名词.py (如 train_model.py)
- **配置文件**: 名词_configs.py (如 model_configs.py)  
- **实验目录**: experiment_YYYYMMDD_HHMMSS
- **报告文件**: 描述性名称 + 时间戳

### 文件内容组织
```python
# 标准Python文件结构
"""
文件描述和用途说明
"""

# 导入部分
import sys
import os
from pathlib import Path

# 项目导入
sys.path.append(str(Path(__file__).parent.parent))
from core.config import ...

# 常量定义
CONSTANT_NAME = "value"

# 函数定义
def function_name():
    """函数文档字符串"""
    pass

# 主程序
if __name__ == "__main__":
    main()