# FUA多任务学习适配方案

## 概述

本文档详细说明了如何将FUA系统适配为支持多任务学习的架构，以处理多层标注的生物图像分类任务。

## 1. 标注体系定义

### 1.1 生长级别分类 (Growth Level)
- **类别**: negative (0), positive (1), weak_growth (2)
- **任务类型**: 单标签分类
- **输出层**: 3个神经元 + Softmax

### 1.2 生长模式分类 (Growth Pattern)
- **类别**: clean (0), clustered (1), scattered (2), small_dots (4), 等9类
- **任务类型**: 单标签分类
- **输出层**: 9个神经元 + Softmax

### 1.3 干扰因素分类 (Interference Mapping)
- **类别**: pores (0), debris (1), artifacts (2)
- **任务类型**: 多标签分类（一个图像可能有多个干扰因素）
- **输出层**: 3个神经元 + Sigmoid

### 1.4 精细分类 (Fine-grained)
- **类别**: 40个组合类别
- **任务类型**: 单标签分类
- **输出层**: 40个神经元 + Softmax

## 2. 数据结构适配

### 2.1 新的标注文件格式

```json
{
  "image_id": "image_001",
  "file_path": "bioast_dataset/train/image_001.png",
  "annotations": {
    "growth_level": "positive",
    "growth_pattern": "clustered",
    "interference_mapping": ["pores", "debris"],
    "fine_grained": "positive_clustered_pores"
  },
  "split": "train"
}
```

### 2.2 数据集目录结构

```
bioast_dataset/
├── annotations/
│   ├── train_annotations.json
│   ├── val_annotations.json
│   └── test_annotations.json
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## 3. 模型架构适配

### 3.1 多任务模型设计

```python
class MultitaskBioastModel(nn.Module):
    def __init__(self, 
                 backbone: nn.Module,
                 num_growth_level: int = 3,
                 num_growth_pattern: int = 9,
                 num_interference: int = 3,
                 num_fine_grained: int = 40,
                 feature_dim: int = 576):
        super().__init__()
        
        # 共享特征提取器
        self.backbone = backbone
        
        # 任务特定的头部
        self.heads = nn.ModuleDict({
            'growth_level': self._create_classification_head(feature_dim, num_growth_level),
            'growth_pattern': self._create_classification_head(feature_dim, num_growth_pattern),
            'interference_mapping': self._create_multilabel_head(feature_dim, num_interference),
            'fine_grained': self._create_classification_head(feature_dim, num_fine_grained)
        })
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 8, feature_dim, 1),
            nn.Sigmoid()
        )
    
    def _create_classification_head(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_dim // 2, out_dim)
        )
    
    def _create_multilabel_head(self, in_dim: int, out_dim: int):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_dim // 2, out_dim)
        )
    
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        
        # 应用注意力
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # 各任务预测
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(attended_features)
        
        return outputs
```

### 3.2 损失函数设计

```python
class MultitaskLoss(nn.Module):
    def __init__(self, task_weights=None):
        super().__init__()
        self.task_weights = task_weights or {
            'growth_level': 1.0,
            'growth_pattern': 1.0,
            'interference_mapping': 0.5,
            'fine_grained': 1.0
        }
        
        # 任务特定的损失函数
        self.criterion = {
            'growth_level': nn.CrossEntropyLoss(),
            'growth_pattern': nn.CrossEntropyLoss(),
            'interference_mapping': nn.BCEWithLogitsLoss(),
            'fine_grained': nn.CrossEntropyLoss()
        }
    
    def forward(self, outputs, targets):
        total_loss = 0
        
        for task_name in outputs.keys():
            task_output = outputs[task_name]
            task_target = targets[task_name]
            
            # 计算任务损失
            if task_name == 'interference_mapping':
                # 多标签损失
                task_loss = self.criterion[task_name](task_output, task_target.float())
            else:
                # 单标签损失
                task_loss = self.criterion[task_name](task_output, task_target)
            
            # 加权累加
            total_loss += self.task_weights[task_name] * task_loss
        
        return total_loss
```

## 4. 数据加载器适配

### 4.1 多标签数据集类

```python
class MultitaskBioastDataset(Dataset):
    def __init__(self, annotation_file: str, transform=None):
        """
        Args:
            annotation_file: JSON标注文件路径
            transform: 图像变换
        """
        self.transform = transform
        
        # 加载标注数据
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 类别映射
        self.mappings = {
            'growth_level': {
                'negative': 0, 'positive': 1, 'weak_growth': 2
            },
            'growth_pattern': {
                'clean': 0, 'clustered': 1, 'scattered': 2,
                'heavy_growth': 3, 'small_dots': 4,
                'irregular_areas': 5, 'light_gray': 6,
                'default_positive': 7, 'default_weak_growth': 8
            },
            'interference_mapping': {
                'pores': 0, 'debris': 1, 'artifacts': 2
            },
            'fine_grained': self._generate_fine_grained_mapping()
        }
    
    def _generate_fine_grained_mapping(self):
        """生成40个精细类别的映射"""
        mapping = {}
        idx = 0
        
        # 阴性变体
        mapping['negative_clean'] = idx
        mapping['negative_pores'] = idx + 1
        mapping['negative_debris'] = idx + 2
        idx += 3
        
        # 阳性聚集型变体
        for interference in ['pores', 'debris', 'artifacts']:
            mapping[f'positive_clustered_{interference}'] = idx
            idx += 1
        
        # 其他组合...
        return mapping
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 加载图像
        image = Image.open(ann['file_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 准备多任务标签
        labels = {}
        
        # 生长级别
        labels['growth_level'] = self.mappings['growth_level'][
            ann['annotations']['growth_level']
        ]
        
        # 生长模式
        labels['growth_pattern'] = self.mappings['growth_pattern'][
            ann['annotations']['growth_pattern']
        ]
        
        # 干扰因素（多标签）
        interference_labels = [0] * 3
        for interference in ann['annotations']['interference_mapping']:
            interference_labels[self.mappings['interference_mapping'][interference]] = 1
        labels['interference_mapping'] = torch.tensor(interference_labels)
        
        # 精细分类
        labels['fine_grained'] = self.mappings['fine_grained'][
            ann['annotations']['fine_grained']
        ]
        
        return image, labels
```

## 5. 训练流程适配

### 5.1 多任务训练器

```python
class MultitaskTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 损失函数
        self.criterion = MultitaskLoss(config.get('task_weights'))
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 100)
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_amp') else None
        
        # 指标记录
        self.metrics_history = {task: [] for task in model.heads.keys()}
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # 前向传播
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            total_loss += loss.item()
            
            # 记录各任务指标
            self._update_metrics(outputs, targets)
        
        return total_loss / len(self.train_loader)
    
    def _update_metrics(self, outputs, targets):
        """更新各任务的指标"""
        for task_name in outputs.keys():
            if task_name == 'interference_mapping':
                # 多标签准确率
                preds = torch.sigmoid(outputs[task_name]) > 0.5
                acc = (preds == targets[task_name]).float().mean()
            else:
                # 单标签准确率
                preds = outputs[task_name].argmax(dim=1)
                acc = (preds == targets[task_name]).float().mean()
            
            self.metrics_history[task_name].append(acc.item())
```

## 6. 评估指标适配

### 6.1 多任务评估器

```python
class MultitaskEvaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
    
    def evaluate(self):
        self.model.eval()
        all_predictions = {task: [] for task in self.model.heads.keys()}
        all_targets = {task: [] for task in self.model.heads.keys()}
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                for task_name in outputs.keys():
                    if task_name == 'interference_mapping':
                        preds = torch.sigmoid(outputs[task_name]) > 0.5
                    else:
                        preds = outputs[task_name].argmax(dim=1)
                    
                    all_predictions[task_name].extend(preds.cpu().numpy())
                    all_targets[task_name].extend(targets[task_name].numpy())
        
        # 计算各任务指标
        results = {}
        for task_name in all_predictions.keys():
            if task_name == 'interference_mapping':
                results[task_name] = self._evaluate_multilabel(
                    all_predictions[task_name],
                    all_targets[task_name]
                )
            else:
                results[task_name] = self._evaluate_classification(
                    all_predictions[task_name],
                    all_targets[task_name]
                )
        
        return results
    
    def _evaluate_classification(self, preds, targets):
        """评估单标签分类"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _evaluate_multilabel(self, preds, targets):
        """评估多标签分类"""
        from sklearn.metrics import hamming_loss, f1_score
        
        hamming = hamming_loss(targets, preds)
        f1_micro = f1_score(targets, preds, average='micro')
        f1_macro = f1_score(targets, preds, average='macro')
        
        return {
            'hamming_loss': hamming,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro
        }
```

## 7. FUA系统集成方案

### 7.1 数据集迭代管理器适配

```python
class MultitaskDatasetVersionManager(DatasetVersionManager):
    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.task_names = ['growth_level', 'growth_pattern', 
                         'interference_mapping', 'fine_grained']
    
    def calculate_multitask_stats(self) -> Dict:
        """计算多任务数据集统计"""
        stats = {}
        
        for task in self.task_names:
            if task == 'interference_mapping':
                # 多标签统计
                stats[task] = self._calculate_multilabel_stats(task)
            else:
                # 单标签统计
                stats[task] = self._calculate_single_label_stats(task)
        
        return stats
    
    def _calculate_multilabel_stats(self, task: str) -> Dict:
        """计算多标签统计"""
        label_counts = [0] * 3  # 3个干扰因素类别
        
        for sample in self.current_data:
            for label in sample['annotations'][task]:
                label_counts[label] += 1
        
        return {
            'total_samples': len(self.current_data),
            'label_counts': label_counts,
            'cooccurrence_matrix': self._calculate_cooccurrence(task)
        }
```

### 7.2 参数优化器适配

```python
class MultitaskParameterOptimizer(ParameterOptimizer):
    def __init__(self, model_name: str, history_manager: ParameterHistoryManager):
        super().__init__(model_name, history_manager)
        self.task_names = ['growth_level', 'growth_pattern', 
                         'interference_mapping', 'fine_grained']
    
    def suggest_parameters(self, strategy: str = "adaptive") -> Dict:
        """为多任务模型建议参数"""
        base_params = super().suggest_parameters(strategy)
        
        # 多任务特定参数
        multitask_params = {
            'task_weights': {
                'growth_level': 1.0,
                'growth_pattern': 1.0,
                'interference_mapping': 0.5,
                'fine_grained': 1.0
            },
            'attention_dim': base_params.get('hidden_dim', 128) // 8,
            'dropout_rate': 0.2
        }
        
        return {**base_params, **multitask_params}
```

## 8. 部署和推理

### 8.1 多任务推理API

```python
class MultitaskInferenceAPI:
    def __init__(self, model_path: str):
        self.model = torch.load(model_path)
        self.model.eval()
        self.transform = get_transforms('test')
    
    def predict(self, image_path: str) -> Dict:
        """单图像预测"""
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(image)
        
        # 后处理
        results = {}
        for task_name, output in outputs.items():
            if task_name == 'interference_mapping':
                # 多标签后处理
                probs = torch.sigmoid(output).squeeze().numpy()
                results[task_name] = {
                    'predictions': (probs > 0.5).astype(int).tolist(),
                    'probabilities': probs.tolist(),
                    'labels': ['pores', 'debris', 'artifacts']
                }
            else:
                # 单标签后处理
                probs = F.softmax(output, dim=1).squeeze().numpy()
                pred_class = probs.argmax()
                results[task_name] = {
                    'predicted_class': int(pred_class),
                    'confidence': float(probs[pred_class]),
                    'probabilities': probs.tolist()
                }
        
        return results
```

## 9. 迁移策略

### 9.1 数据迁移步骤

1. **创建标注转换脚本**：
   - 将现有的positive/negative目录结构转换为JSON标注格式
   - 为每个图像添加多层标注信息

2. **数据集重组**：
   - 将所有图像移至统一目录
   - 创建train/val/test的标注文件

3. **渐进式训练**：
   - 先在growth_level任务上预训练
   - 逐步添加其他任务

### 9.2 模型迁移步骤

1. **Backbone复用**：
   - 使用现有预训练模型作为共享backbone
   - 添加新的任务头部

2. **增量训练**：
   - 冻结backbone，只训练新头部
   - 解冻整个模型进行端到端微调

## 10. 实施建议

1. **优先级顺序**：
   - 第一阶段：实现growth_level和growth_pattern分类
   - 第二阶段：添加interference_mapping多标签分类
   - 第三阶段：实现fine_grained精细分类

2. **质量控制**：
   - 为每个任务设置独立的验证集
   - 监控各任务的学习曲线

3. **性能优化**：
   - 使用知识蒸馏压缩模型
   - 实现动态任务权重调整

此适配方案确保FUA系统能够平滑过渡到多任务学习架构，同时保持系统的可扩展性和维护性。