# FUA增量数据集使用指南

## 概述

FUA迭代平台的增量数据集功能允许您：
- 追踪数据集版本变化
- 安全地添加新数据而不影响现有训练
- 自动检测和避免重复数据
- 分析数据集质量和平衡性
- 与Bmad工作流无缝集成

## 基本使用流程

### 1. 初始化数据集管理器

```python
from fua.dataset_iteration_manager import DatasetVersionManager, DatasetIncrementalUpdater

# 创建数据集版本管理器
version_manager = DatasetVersionManager("bioast_dataset")

# 创建增量更新器
updater = DatasetIncrementalUpdater("bioast_dataset")
```

### 2. 创建初始版本

```python
# 创建数据集的初始版本
version_info = version_manager.create_version(
    version_name="v1.0",
    description="初始数据集，包含1000个样本"
)

print(f"创建版本: {version_info['version']}")
print(f"统计数据: {version_info['stats']}")
```

### 3. 添加新数据

#### 添加单个文件

```python
# 添加单个图像文件
result = updater.add_new_data(
    source_path="/path/to/new_image.jpg",
    target_split="train",  # train, val, 或 test
    label="positive",      # positive 或 negative
    metadata={
        "source": "experiment_2",
        "collector": "researcher_A",
        "date": "2024-01-15"
    }
)

print(f"添加结果: {result}")
# 输出: {'added': 1, 'duplicates': 0, 'errors': 0, 'files': [...]}
```

#### 添加整个目录

```python
# 添加目录中的所有文件
result = updater.add_new_data(
    source_path="/path/to/new_positive_samples/",
    target_split="train",
    label="positive",
    metadata={"batch": "batch_001"}
)

print(f"成功添加: {result['added']} 个文件")
print(f"重复文件: {result['duplicates']} 个")
```

### 4. 创建新版本

```python
# 添加数据后创建新版本
new_version = version_manager.create_version(
    version_name="v1.1",
    description="添加了50个新的正样本"
)

print(f"新版本统计: {new_version['stats']}")
```

## 高级功能

### 1. 数据集版本管理

```python
# 列出所有版本
versions = version_manager.list_versions()
for version in versions:
    print(f"版本 {version['version']}: {version['description']}")

# 获取特定版本信息
version_info = version_manager.get_version_info("v1.0")
print(f"v1.0的创建时间: {version_info['created_at']}")
```

### 2. 数据集分析

```python
from fua.dataset_iteration_manager import DatasetAnalyzer

# 创建分析器
analyzer = DatasetAnalyzer("bioast_dataset")

# 生成质量报告
report = analyzer.generate_quality_report()

print("=== 数据集质量报告 ===")
print(f"总图像数: {report['summary']['total_images']}")
print(f"问题数量: {len(report['quality_issues'])}")
print("建议:")
for suggestion in report['recommendations']:
    print(f"  - {suggestion}")
```

### 3. 数据集缺口分析

```python
# 分析数据集不平衡问题
gaps = updater.analyze_dataset_gaps()

print("=== 数据集缺口分析 ===")
print(f"类别不平衡情况:")
for split, imbalance in gaps['class_imbalance'].items():
    print(f"  {split}: 正样本比例 {imbalance['positive_ratio']:.2%}")

print("\n改进建议:")
for recommendation in gaps['recommendations']:
    print(f"  - {recommendation}")
```

## 与Bmad工作流集成

### 1. 在工作流中使用数据集版本

```python
from fua.bmad_workflow_engine import BmadWorkflowEngine

# 创建工作流引擎
engine = BmadWorkflowEngine()

# 创建使用特定数据集版本的工作流
workflow_id = engine.create_workflow(
    name="incremental_training",
    model_name="resnet18",
    initial_config={
        "dataset_version": "v1.1",  # 指定数据集版本
        "target_accuracy": 0.95
    }
)

# 启动工作流
engine.start_workflow(workflow_id)
```

### 2. 工作流中自动更新数据集

```python
# 在Decide阶段决定是否需要更多数据
def decide_phase(self, workflow_id, analyze_result):
    # ... 决策逻辑
    
    # 如果准确率不足，触发数据增强
    if metrics["accuracy"] < 0.9:
        decision["data_augmentation_needed"] = True
        decision["next_actions"].append("收集更多训练数据")
        
        # 自动触发数据更新流程
        self._trigger_data_collection(workflow_id)
```

## 实际使用示例

### 示例1：迭代改进数据集

```python
import os
from pathlib import Path
from fua.dataset_iteration_manager import DatasetVersionManager, DatasetIncrementalUpdater

def iterative_dataset_improvement():
    """迭代改进数据集的完整示例"""
    
    # 1. 初始化
    version_manager = DatasetVersionManager("bioast_dataset")
    updater = DatasetIncrementalUpdater("bioast_dataset")
    
    # 2. 分析当前数据集
    gaps = updater.analyze_dataset_gaps()
    print("初始数据集分析:", gaps)
    
    # 3. 根据分析结果添加数据
    if gaps['recommendations']:
        print("根据建议添加数据...")
        
        # 假设需要更多正样本
        new_data_dir = Path("/new_data/positive_samples")
        if new_data_dir.exists():
            result = updater.add_new_data(
                str(new_data_dir),
                "train",
                "positive",
                {"iteration": "1"}
            )
            print(f"添加了 {result['added']} 个新样本")
    
    # 4. 创建新版本
    new_version = version_manager.create_version(
        f"v1.{version_manager.metadata['current_version'].split('.')[-1] + 1}",
        "迭代改进后的数据集"
    )
    
    # 5. 重新训练模型
    print(f"使用新版本 {new_version['version']} 重新训练...")
    
    return new_version['version']
```

### 示例2：自动化数据收集流程

```python
import time
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewDataHandler(FileSystemEventHandler):
    """监控新数据文件的处理器"""
    
    def __init__(self, updater):
        self.updater = updater
        
    def on_created(self, event):
        if event.is_dir:
            return
            
        if event.src_path.endswith('.jpg'):
            print(f"发现新文件: {event.src_path}")
            
            # 自动添加到数据集
            result = self.updater.add_new_data(
                event.src_path,
                "train",
                "positive",  # 可根据文件名判断
                {"auto_added": True, "timestamp": time.time()}
            )
            
            if result['added'] > 0:
                print("文件已添加到数据集")

def setup_auto_collection(watch_path):
    """设置自动数据收集"""
    updater = DatasetIncrementalUpdater("bioast_dataset")
    
    # 创建事件处理器
    event_handler = NewDataHandler(updater)
    
    # 创建观察者
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    
    # 开始监控
    observer.start()
    print(f"开始监控 {watch_path} 目录...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()
```

## 最佳实践

### 1. 数据集版本命名

```python
# 使用语义化版本号
versions = ["v1.0.0", "v1.1.0", "v1.2.0", "v2.0.0"]

# 版本号规则：
# v主版本号.次版本号.修订号
# - 主版本号：重大数据变更
# - 次版本号：新增数据
# - 修订号：错误修复或小量补充
```

### 2. 元数据管理

```python
# 为添加的数据添加丰富的元数据
metadata = {
    # 数据来源
    "source": "experiment_3",
    "collection_method": "microscope",
    
    # 质量信息
    "quality_score": 0.95,
    "verified": True,
    
    # 标注信息
    "annotator": "expert_1",
    "annotation_date": "2024-01-15",
    
    # 业务信息
    "project": "colony_detection",
    "batch_id": "B00123"
}
```

### 3. 批量处理优化

```python
import concurrent.futures
from pathlib import Path

def batch_add_files(updater, file_paths, target_split, label, metadata):
    """批量添加文件以提高效率"""
    
    def add_single_file(file_path):
        return updater.add_new_data(
            str(file_path),
            target_split,
            label,
            metadata
        )
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file_path in file_paths:
            futures.append(executor.submit(add_single_file, file_path))
        
        # 收集结果
        total_added = 0
        total_duplicates = 0
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            total_added += result['added']
            total_duplicates += result['duplicates']
    
    return {
        'total_added': total_added,
        'total_duplicates': total_duplicates,
        'total_files': len(file_paths)
    }
```

## 故障排除

### 1. 常见问题

**问题：添加文件时出现权限错误**
```python
# 解决方案：检查文件权限
import os
file_path = "/path/to/file.jpg"
print(f"文件可读: {os.access(file_path, os.R_OK)}")
print(f"目录可写: {os.access(os.path.dirname(file_path), os.W_OK)}")
```

**问题：重复文件检测不准确**
```python
# 检查文件哈希
from fua.dataset_iteration_manager import DatasetIncrementalUpdater

updater = DatasetIncrementalUpdater("bioast_dataset")
file_path = Path("/path/to/file.jpg")
file_hash = updater._calculate_file_hash(file_path)
print(f"文件哈希: {file_hash}")
```

### 2. 性能优化

**大批量数据处理：**
```python
# 分批处理大量文件
def process_large_dataset(file_list, batch_size=1000):
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i + batch_size]
        result = batch_add_files(updater, batch, "train", "positive", {})
        print(f"处理批次 {i//batch_size + 1}: {result}")
```

### 3. 数据验证

```python
# 添加前验证图像
from PIL import Image
import cv2

def validate_image(file_path):
    """验证图像文件是否有效"""
    try:
        # 检查文件大小
        if file_path.stat().st_size == 0:
            return False, "空文件"
        
        # 尝试打开图像
        with Image.open(file_path) as img:
            img.verify()
        
        # 检查尺寸
        img = cv2.imread(str(file_path))
        if img is None:
            return False, "无法读取图像"
        
        if img.shape != (70, 70, 3):
            return False, f"尺寸错误: {img.shape}"
        
        return True, "验证通过"
        
    except Exception as e:
        return False, str(e)
```

## 总结

FUA增量数据集功能提供了完整的数据集版本管理和迭代改进能力。通过合理使用这些功能，您可以：

1. **系统化地改进数据集质量**
2. **追踪每次变更的影响**
3. **避免数据重复和混乱**
4. **与训练流程无缝集成**
5. **自动化数据收集和处理**

记住的关键点：
- 始终创建版本以跟踪变更
- 使用丰富的元数据记录数据来源
- 定期分析数据集质量和平衡性
- 与Bmad工作流结合实现自动化改进