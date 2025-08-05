# BioAst模型管理系统

一个专为生物信息学设计的完整模型生命周期管理平台，支持AI和人类协同管控。

## 🚀 系统特性

### 核心功能
- **完整的模型生命周期管理**: 从定义、训练、验证到部署的全流程管理
- **智能实验跟踪**: 自动记录实验参数、指标和结果
- **版本控制**: 完整的模型版本管理和变更追踪
- **自动化工作流**: 可配置的训练和评估管道
- **多格式报告**: 支持JSON、Markdown、HTML等多种格式
- **可视化仪表板**: 直观的Web界面和丰富的图表
- **任务调度**: 支持定时任务和批处理

### AI-人类协作设计
- **双重接口**: AI使用结构化JSON接口，人类使用可视化界面
- **智能决策支持**: AI提供建议，人类进行最终决策
- **完整审计日志**: 所有操作都有详细记录
- **异常检测**: 自动识别异常情况并提醒人工介入

## 📁 项目结构

```
bioastModel/
├── model_registry/          # 模型注册表
│   ├── __init__.py
│   ├── registry.py         # 模型注册管理
│   └── version_control.py  # 版本控制
├── experiment_manager/      # 实验管理
│   ├── __init__.py
│   ├── experiment.py       # 实验定义
│   ├── tracker.py          # 实验跟踪
│   └── database.py         # 数据存储
├── dashboard/              # 可视化仪表板
│   ├── __init__.py
│   ├── dashboard.py        # Web仪表板
│   ├── visualization.py    # 图表生成
│   └── report_generator.py # 报告生成
├── workflow/               # 工作流自动化
│   ├── __init__.py
│   ├── automation.py       # 工作流引擎
│   ├── pipeline.py         # 模型管道
│   └── scheduler.py        # 任务调度
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── integration.py      # 系统集成
│   ├── config.py           # 配置管理
│   ├── logger.py           # 日志系统
│   ├── validators.py       # 验证器
│   └── helpers.py          # 辅助函数
├── main.py                 # 主入口
├── README.md               # 说明文档
└── requirements.txt        # 依赖包
```

## 🛠️ 安装和配置

### 环境要求
- Python 3.8+
- 推荐使用虚拟环境

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd bioastModel
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **初始化配置**
```bash
python main.py --mode demo
```

## 🎯 解决方案

我们提供四种不同的解决方案，适应不同规模和需求的团队：

### 解决方案1: 基础模型管道

**适用场景**: 个人研究者或小团队

**特点**:
- 简单易用的模型训练流程
- 基础的实验跟踪
- 标准化的模型注册
- 基本的报告生成

**使用示例**:
```python
from main import BioAstModelSystem, create_sample_model_config

# 初始化系统
system = BioAstModelSystem()
system.start_services()

# 创建模型配置
model_config = create_sample_model_config()
model_config.update({
    'name': 'MyBioModel',
    'data_config': {
        'data_path': 'path/to/your/data.csv',
        'target_column': 'label'
    }
})

# 执行训练
workflow_id = system.create_new_model_workflow(model_config)
success = system.execute_model_training(workflow_id)

# 生成报告
if success:
    report_path = system.generate_system_report()
    print(f"报告已生成: {report_path}")
```

### 解决方案2: 自动化模型管道

**适用场景**: 中型团队，需要自动化处理

**特点**:
- 全自动化的模型训练流程
- 智能的超参数优化
- 自动化的模型比较和选择
- 定时任务和批处理
- 完整的版本控制

**核心组件**:
- 工作流自动化引擎
- 任务调度器
- 模型版本控制
- 性能监控

### 解决方案3: 企业级模型平台

**适用场景**: 大型企业，生产环境部署

**特点**:
- 完整的MLOps流程
- 多用户权限管理
- 分布式训练支持
- 模型部署和服务
- 高级监控和告警
- 数据血缘追踪

**技术架构**:
- 微服务架构
- 容器编排
- 消息队列
- 分布式存储
- 负载均衡

### 解决方案4: AI-人类协作平台

**适用场景**: 需要AI和人类协同管控的环境

**特点**:
- 双重接口设计 (AI结构化 + 人类可视化)
- 智能决策建议
- 人工审核节点
- 异常自动检测和人工确认
- 完整的操作审计

**AI接口特性**:
- JSON格式的结构化数据
- RESTful API
- 自动化工作流
- 机器可读的状态信息

**人类接口特性**:
- 直观的Web仪表板
- 丰富的可视化图表
- Markdown格式报告
- 交互式操作界面

## 🚀 快速开始

### 1. 查看解决方案
```bash
python main.py --mode solutions
```

### 2. 查看特定解决方案
```bash
python main.py --mode solutions --solution 1
```

### 3. 运行演示
```bash
python main.py --mode demo
```

### 4. 启动系统
```bash
python main.py --mode run
```

## 📊 使用示例

### 创建新模型

```python
from utils.integration import ModelLifecycleManager
from utils.config import ConfigManager

# 初始化管理器
config = ConfigManager().get_default_config()
manager = ModelLifecycleManager(config)
manager.start_services()

# 定义模型配置
model_config = {
    'name': 'ProteinClassifier',
    'description': '蛋白质分类模型',
    'model_type': 'classification',
    'algorithm': 'random_forest',
    'data_config': {
        'data_path': 'data/protein_data.csv',
        'target_column': 'protein_type',
        'test_size': 0.2
    },
    'training_config': {
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 10
        }
    }
}

# 创建训练工作流
workflow_id = manager.create_training_workflow(
    model_config=model_config,
    data_config=model_config['data_config'],
    training_config=model_config['training_config']
)

# 执行训练
success = manager.execute_workflow(workflow_id)

if success:
    print("模型训练成功！")
    
    # 获取训练结果
    workflow_status = manager.get_workflow_status(workflow_id)
    print(f"工作流状态: {workflow_status}")
    
    # 生成报告
    report_path = manager.generate_experiment_report(
        experiment_id=workflow_status.get('experiment_id'),
        output_format='html'
    )
    print(f"报告已生成: {report_path}")
```

### 模型比较

```python
# 获取所有模型
models = manager.list_models()
print(f"共有 {len(models)} 个模型")

# 选择要比较的模型
model_ids = [model['id'] for model in models[:3]]  # 比较前3个模型

# 生成比较报告
comparison_report = manager.generate_comparison_report(
    model_ids=model_ids,
    output_format='html'
)
print(f"比较报告: {comparison_report}")
```

### 可视化分析

```python
# 创建可视化
visualizer = manager.visualizer

# 绘制训练曲线
training_curve = visualizer.plot_training_curves(
    experiment_id='exp_123456',
    save_path='reports/training_curve.png'
)

# 创建交互式仪表板
dashboard_url = visualizer.create_interactive_dashboard(
    model_ids=['model_1', 'model_2'],
    port=8080
)
print(f"仪表板地址: {dashboard_url}")
```

## 🔧 配置说明

### 基础配置

```yaml
# config.yaml
base:
  project_name: "BioAst"
  data_dir: "./data"
  model_dir: "./models"
  log_dir: "./logs"
  log_level: "INFO"

model_registry:
  storage_backend: "local"
  storage_path: "./registry"
  enable_versioning: true
  max_versions: 10

experiment:
  tracking_backend: "sqlite"
  database_url: "sqlite:///experiments.db"
  auto_log_metrics: true
  save_artifacts: true

dashboard:
  host: "localhost"
  port: 8080
  enable_auth: false
  theme: "default"

workflow:
  max_parallel_jobs: 4
  retry_attempts: 3
  timeout_minutes: 60
```

### 高级配置

```python
from utils.config import Config, ConfigManager

# 创建自定义配置
config = Config(
    base=BaseConfig(
        project_name="MyBioProject",
        data_dir="/path/to/data",
        model_dir="/path/to/models"
    ),
    model_registry=ModelRegistryConfig(
        storage_backend="s3",
        storage_path="s3://my-bucket/models"
    ),
    experiment=ExperimentConfig(
        tracking_backend="mlflow",
        database_url="postgresql://user:pass@host:5432/db"
    )
)

# 保存配置
config_manager = ConfigManager()
config_manager.save_config(config, "custom_config.yaml")
```

## 📈 监控和报告

### 实验监控

系统提供多种监控方式：

1. **实时监控**: Web仪表板显示实时训练进度
2. **指标跟踪**: 自动记录损失、准确率等关键指标
3. **资源监控**: CPU、内存、GPU使用情况
4. **异常检测**: 自动识别训练异常

### 报告生成

支持多种格式的报告：

- **JSON报告**: 机器可读的结构化数据
- **Markdown报告**: 人类友好的文档格式
- **HTML报告**: 包含图表的交互式报告
- **PDF报告**: 适合打印和分享

## 🔒 安全和权限

### 访问控制
- 基于角色的权限管理
- API密钥认证
- 操作审计日志

### 数据安全
- 数据加密存储
- 安全的模型传输
- 敏感信息脱敏

## 🤝 AI-人类协作

### 协作模式

1. **AI主导模式**: AI自动执行大部分任务，人类监督
2. **人类主导模式**: 人类控制关键决策，AI提供支持
3. **协作模式**: AI和人类共同参与决策过程

### 接口设计

**AI接口**:
```python
# AI使用结构化API
result = manager.train_model({
    "model_type": "classification",
    "data_path": "data.csv",
    "hyperparameters": {...}
})

# 返回结构化结果
{
    "status": "success",
    "model_id": "model_123",
    "metrics": {
        "accuracy": 0.95,
        "f1_score": 0.93
    },
    "artifacts": [
        "model.pkl",
        "report.json"
    ]
}
```

**人类接口**:
- Web仪表板: http://localhost:8080
- 可视化图表和交互式界面
- 拖拽式工作流编辑器
- 一键生成报告

## 📚 API文档

### 核心API

#### 模型管理
```python
# 注册模型
model_id = manager.register_model(
    name="MyModel",
    model_path="path/to/model.pkl",
    metadata={"version": "1.0.0"}
)

# 获取模型
model = manager.get_model(model_id)

# 更新模型性能
manager.update_model_performance(
    model_id=model_id,
    metrics={"accuracy": 0.95}
)
```

#### 实验管理
```python
# 创建实验
experiment_id = manager.create_experiment(
    name="Experiment1",
    config={"learning_rate": 0.01}
)

# 记录指标
manager.log_metrics(
    experiment_id=experiment_id,
    metrics={"loss": 0.1, "accuracy": 0.9},
    step=100
)

# 完成实验
manager.complete_experiment(experiment_id)
```

#### 工作流管理
```python
# 创建工作流
workflow_id = manager.create_training_workflow(
    model_config=config,
    data_config=data_config
)

# 执行工作流
success = manager.execute_workflow(workflow_id)

# 获取状态
status = manager.get_workflow_status(workflow_id)
```

## 🔧 扩展和定制

### 自定义验证器

```python
from utils.validators import BaseValidator, ValidationResult

class CustomModelValidator(BaseValidator):
    def validate(self, model_path):
        result = self._create_result()
        # 自定义验证逻辑
        return result
```

### 自定义工作流步骤

```python
from workflow.automation import WorkflowStep

def custom_preprocessing_step(context):
    """自定义预处理步骤"""
    data = context.get('data')
    # 处理逻辑
    context['processed_data'] = processed_data
    return True

# 注册步骤
workflow.add_step(WorkflowStep(
    name="custom_preprocessing",
    function=custom_preprocessing_step,
    dependencies=["data_loading"]
))
```

### 自定义报告模板

```python
from dashboard.report_generator import ReportGenerator

# 自定义模板
custom_template = """
# {{ experiment.name }}

## 结果
- 准确率: {{ metrics.accuracy }}
- F1分数: {{ metrics.f1_score }}

## 图表
{{ charts.training_curve }}
"""

# 使用自定义模板
report_generator = ReportGenerator()
report_generator.add_template("custom", custom_template)
```

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径
   - 确认模型格式支持
   - 查看错误日志

2. **训练过程中断**
   - 检查数据格式
   - 确认内存是否充足
   - 查看训练日志

3. **仪表板无法访问**
   - 检查端口是否被占用
   - 确认防火墙设置
   - 查看服务状态

### 日志分析

```bash
# 查看系统日志
tail -f logs/bioast.log

# 查看错误日志
tail -f logs/error.log

# 查看特定模块日志
grep "model_registry" logs/bioast.log
```

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black bioastModel/
flake8 bioastModel/
```

## 📄 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 📞 支持和联系

- 文档: [项目文档](https://docs.bioast.com)
- 问题反馈: [GitHub Issues](https://github.com/bioast/issues)
- 邮件支持: support@bioast.com
- 社区讨论: [Discord](https://discord.gg/bioast)

---

**BioAst模型管理系统** - 让AI和人类协同工作，构建更好的生物信息学模型。