# FUA迭代平台部署指南

## 概述

FUA迭代平台是一个基于Bmad（Build-Measure-Analyze-Decide）工作流的自动化模型改进系统。该平台支持数据集迭代管理、参数优化、自动化训练流水线和模型验证分析。

## 系统要求

### 硬件要求
- CPU: 多核处理器（推荐4核以上）
- 内存: 8GB以上（推荐16GB）
- 存储: 20GB可用空间
- GPU: 可选，支持CUDA的NVIDIA GPU（推荐用于训练）

### 软件要求
- 操作系统: Linux, macOS, Windows
- Python: 3.8或更高版本
- 包管理器: uv（推荐）或pip

## 安装步骤

### 1. 克隆代码库

```bash
git clone <repository-url>
cd bioastModel
```

### 2. 设置虚拟环境

```bash
# 使用uv创建虚拟环境
uv venv

# 激活虚拟环境
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 3. 安装依赖

```bash
# 使用uv安装依赖
uv pip install -r requirements.txt

# 或者使用pip
pip install -r requirements.txt
```

### 4. 验证安装

```bash
# 运行测试验证安装
python fua/tests/run_tests.py unit

# 检查主要模块
python -c "from fua.dataset_iteration_manager import DatasetVersionManager; print('Dataset module OK')"
python -c "from fua.parameter_optimizer import ParameterOptimizer; print('Optimizer module OK')"
python -c "from fua.bmad_workflow_engine import BmadWorkflowEngine; print('Bmad engine OK')"
```

## 配置

### 1. 数据集配置

确保数据集目录结构如下：
```
bioast_dataset/
├── train/
│   ├── negative/
│   └── positive/
├── val/
│   ├── negative/
│   └── positive/
└── test/
    ├── negative/
    └── positive/
```

### 2. 创建配置文件

在项目根目录创建 `fua_config.json`：

```json
{
  "dataset_path": "bioast_dataset",
  "models_path": "models",
  "experiments_path": "experiments",
  "workflow_storage_path": "fua/workflows",
  "validation_results_path": "fua/validation_results",
  "parameter_history_path": "fua/parameter_history",
  "gpu_available": true,
  "max_concurrent_jobs": 1,
  "default_parameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "adam"
  }
}
```

## 使用指南

### 1. 数据集管理

```python
from fua.dataset_iteration_manager import DatasetVersionManager, DatasetIncrementalUpdater

# 创建版本管理器
version_manager = DatasetVersionManager("bioast_dataset")

# 创建新版本
version_info = version_manager.create_version("v1.0", "初始数据集")

# 添加新数据
updater = DatasetIncrementalUpdater("bioast_dataset")
result = updater.add_new_data(
    "path/to/new/data",
    "train",
    "positive",
    {"source": "experiment_1"}
)
```

### 2. 参数优化

```python
from fua.parameter_optimizer import ParameterHistoryManager, ParameterOptimizer

# 创建优化器
history_manager = ParameterHistoryManager()
optimizer = ParameterOptimizer("resnet18", history_manager)

# 获取参数建议
suggestion = optimizer.suggest_parameters("adaptive")
print(f"建议参数: {suggestion}")
```

### 3. 训练流水线

```python
from fua.training_pipeline import PipelineManager

# 创建流水线管理器
manager = PipelineManager()

# 快速训练
job_id = manager.quick_train("resnet18", epochs=30)

# 智能训练（自动调优）
smart_job_id = manager.smart_train("efficientnet_b0")
```

### 4. Bmad工作流

```python
from fua.bmad_workflow_engine import BmadWorkflowEngine

# 创建工作流引擎
engine = BmadWorkflowEngine()

# 创建工作流
workflow_id = engine.create_workflow(
    "colony_optimization",
    "resnet18",
    {
        "target_accuracy": 0.95,
        "max_iterations": 10
    }
)

# 启动工作流
engine.start_workflow(workflow_id)

# 监控进度
status = engine.get_workflow_status(workflow_id)
print(f"工作流状态: {status}")
```

### 5. 模型验证

```python
from fua.validation_engine import ValidationEngine

# 创建验证引擎
validator = ValidationEngine()

# 验证模型
result = validator.validate_model(
    "path/to/model.pth",
    "path/to/validation_dataset",
    "resnet18",
    "test_set"
)
```

## Web界面使用

### 启动Web服务

```bash
# 启动FUA Web界面
python fua/web/app.py

# 默认地址: http://localhost:5000
```

### 主要功能

1. **仪表板**: 查看系统状态和工作流概览
2. **实验管理**: 创建、监控和管理训练实验
3. **数据集管理**: 上传、查看和管理数据集
4. **参数优化**: 可视化参数优化结果
5. **帮助文档**: 查看系统使用指南

## API参考

### 核心模块API

详见各模块源代码中的docstring文档。

### REST API端点

- `GET /api/dataset` - 获取数据集信息
- `POST /api/dataset/upload` - 上传数据集文件
- `GET /api/experiments` - 获取实验列表
- `POST /api/experiments` - 创建新实验
- `GET /api/workflows` - 获取工作流列表
- `POST /api/workflows` - 创建新工作流

## 监控和日志

### 日志位置

- 训练日志: `experiments/experiment_*/training.log`
- 工作流日志: `fua/workflows/*/bmad_workflow.log`
- 验证日志: `fua/validation_results/*/validation.log`

### 监控指标

- 训练准确率和损失
- 参数优化历史
- 工作流执行状态
- 系统资源使用率

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用混合精度训练
   - 启用梯度累积

2. **数据加载错误**
   - 检查数据集路径
   - 验证图像格式
   - 确认目录结构

3. **模型不收敛**
   - 调整学习率
   - 增加训练轮数
   - 检查数据质量

4. **工作流卡住**
   - 查看工作流日志
   - 检查资源使用情况
   - 重启工作流

### 调试模式

```bash
# 启用详细日志
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from fua.bmad_workflow_engine import BmadWorkflowEngine
engine = BmadWorkflowEngine()
"
```

## 性能优化

### 1. 数据加载优化
- 使用SSD存储
- 预加载数据集
- 启用多进程数据加载

### 2. 训练优化
- 使用混合精度训练
- 启用梯度累积
- 优化数据增强流程

### 3. 系统优化
- 增加并行工作进程
- 使用GPU加速
- 优化内存使用

## 扩展开发

### 添加新模型

1. 在 `models/` 目录创建新模型文件
2. 实现 `create_<model_name>` 工厂函数
3. 更新 `core/config/model_configs.py`
4. 添加相应的ONNX转换器

### 自定义优化策略

继承 `ParameterOptimizer` 类：

```python
class CustomOptimizer(ParameterOptimizer):
    def suggest_parameters(self, strategy):
        # 实现自定义优化逻辑
        return suggested_params
```

### 扩展验证指标

在 `ValidationEngine` 中添加新的指标计算方法。

## 贡献指南

1. Fork项目仓库
2. 创建功能分支
3. 提交代码更改
4. 运行测试套件
5. 提交Pull Request

## 许可证

[项目许可证信息]

## 联系方式

- 问题反馈: [GitHub Issues]
- 技术支持: [支持邮箱]
- 文档更新: [文档仓库]