# 基于3000张渐进式标注的实施策略

## 数据分布分析

### 当前状况
```
总样本: 15,000张
├── Negative: 9,000张 (60%)
└── Positive: 6,000张 (40%)

首批标注: 3,000张
├── Negative: 1,800张 (保持6:4比例)
└── Positive: 1,200张
```

### 样本选择策略
**多样性采样 + 代表性保证**

## 阶段1: 首批3000张策略设计

### 1.1 Smart Sampling策略
```python
def intelligent_sampling(total_data, n_samples=3000):
    """
    智能采样策略：确保样本的代表性和多样性
    """
    sampling_strategy = {
        'diversity_sampling': 0.4,    # 40% - 最大化样本多样性
        'uncertainty_sampling': 0.3,  # 30% - 基于预训练模型的不确定性
        'random_sampling': 0.2,       # 20% - 随机采样保证无偏
        'edge_case_sampling': 0.1     # 10% - 边界案例和困难样本
    }
    
    selected_samples = []
    
    # 1. 多样性采样 - 基于图像特征聚类
    features = extract_features_pretrained(total_data)  # 使用ImageNet预训练特征
    clusters = KMeans(n_clusters=50).fit(features)
    diversity_samples = select_from_clusters(clusters, int(n_samples * 0.4))
    
    # 2. 不确定性采样 - 使用简单二分类模型
    binary_model = train_quick_binary_model(total_data)
    uncertainty_samples = select_uncertain_samples(binary_model, int(n_samples * 0.3))
    
    # 3. 随机采样
    remaining_samples = set(range(len(total_data))) - set(diversity_samples) - set(uncertainty_samples)
    random_samples = random.sample(list(remaining_samples), int(n_samples * 0.2))
    
    # 4. 边界案例 - 基于图像质量指标
    edge_samples = select_edge_cases(total_data, int(n_samples * 0.1))
    
    return diversity_samples + uncertainty_samples + random_samples + edge_samples
```

### 1.2 数据分割策略
```python
def create_progressive_splits(samples_3k):
    """
    3000样本的分割策略
    """
    # 确保negative/positive比例在各集合中保持6:4
    splits = {
        'train': 2100,      # 70% - 训练集
        'val': 600,         # 20% - 验证集  
        'test': 300         # 10% - 测试集(固定，后续不变)
    }
    
    # 分层抽样确保比例一致
    train_data, temp_data = train_test_split(
        samples_3k, test_size=0.3, stratify=labels, random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.33, stratify=temp_labels, random_state=42
    )
    
    return {
        'train': train_data,  # 2100张
        'val': val_data,      # 600张
        'test': test_data     # 300张 (固定基准)
    }
```

## 阶段2: 渐进式标注与模型更新

### 2.1 标注批次规划
```python
annotation_batches = {
    'batch_1': {
        'size': 3000,
        'focus': 'baseline + growth_level',
        'timeline': '1-2周',
        'expected_acc': '85-90%'
    },
    'batch_2': {
        'size': 2000,  # 累计5000
        'focus': 'growth_pattern初步标注',
        'timeline': '2-3周',
        'expected_acc': '88-92%'
    },
    'batch_3': {
        'size': 2000,  # 累计7000
        'focus': 'growth_pattern完善 + interference',
        'timeline': '2-3周', 
        'expected_acc': '90-94%'
    },
    'batch_4': {
        'size': 3000,  # 累计10000
        'focus': 'fine_grained 40类',
        'timeline': '3-4周',
        'expected_acc': '92-95%'
    },
    'batch_5': {
        'size': 5000,  # 全部15000
        'focus': '完整多任务系统',
        'timeline': '2-3周',
        'expected_acc': '94-96%'
    }
}
```

### 2.2 主动学习框架
```python
class ProgressiveActiveLearning:
    def __init__(self, initial_model):
        self.model = initial_model
        self.labeled_pool = []
        self.unlabeled_pool = []
        self.performance_history = []
        
    def select_next_batch(self, batch_size=2000, strategy='hybrid'):
        """
        选择下一批最有价值的样本进行标注
        """
        if strategy == 'hybrid':
            # 混合策略：不确定性 + 多样性 + 任务平衡
            uncertain_samples = self.uncertainty_sampling(batch_size * 0.5)
            diverse_samples = self.diversity_sampling(batch_size * 0.3)
            balanced_samples = self.task_balanced_sampling(batch_size * 0.2)
            
            return uncertain_samples + diverse_samples + balanced_samples
            
    def uncertainty_sampling(self, n_samples):
        """基于模型预测不确定性选择样本"""
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for batch in self.unlabeled_pool:
                outputs = self.model(batch)
                # 计算各任务的不确定性
                task_uncertainties = []
                
                for task, pred in outputs.items():
                    if task == 'interference':  # 多标签任务
                        prob = torch.sigmoid(pred)
                        uncertainty = torch.sum(prob * (1 - prob), dim=1)
                    else:  # 分类任务
                        prob = F.softmax(pred, dim=1)
                        uncertainty = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
                    task_uncertainties.append(uncertainty)
                
                # 综合不确定性
                combined_uncertainty = torch.stack(task_uncertainties).mean(0)
                uncertainties.extend(combined_uncertainty.cpu().numpy())
        
        # 选择最不确定的样本
        top_indices = np.argsort(uncertainties)[-int(n_samples):]
        return [self.unlabeled_pool[i] for i in top_indices]
    
    def update_model(self, new_labeled_data):
        """使用新标注数据更新模型"""
        # 合并新旧数据
        self.labeled_pool.extend(new_labeled_data)
        
        # 渐进式学习策略
        if len(self.labeled_pool) < 5000:
            # 小数据量：使用较强的正则化
            self.train_with_strong_regularization()
        elif len(self.labeled_pool) < 10000:
            # 中等数据量：平衡正则化
            self.train_with_moderate_regularization()
        else:
            # 大数据量：轻度正则化，关注性能
            self.train_with_light_regularization()
            
        # 记录性能变化
        performance = self.evaluate()
        self.performance_history.append(performance)
        
        return performance
```

### 2.3 渐进式多任务学习
```python
class ProgressiveMultiTaskLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        
        # 渐进式任务头
        self.task_heads = nn.ModuleDict()
        self.task_status = {}  # 跟踪每个任务的激活状态
        
        # 共享特征层
        self.shared_features = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def add_task(self, task_name, num_classes, task_type='classification'):
        """动态添加新任务"""
        if task_type == 'classification':
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        elif task_type == 'multilabel':
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        
        self.task_status[task_name] = True
        
    def forward(self, x, active_tasks=None):
        # 特征提取
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features).flatten(1)
        shared_feat = self.shared_features(features)
        
        outputs = {}
        
        # 只计算激活的任务
        active_tasks = active_tasks or list(self.task_heads.keys())
        
        for task in active_tasks:
            if task in self.task_heads and self.task_status.get(task, False):
                outputs[task] = self.task_heads[task](shared_feat)
                
        return outputs
    
    def freeze_task(self, task_name):
        """冻结特定任务的参数"""
        if task_name in self.task_heads:
            for param in self.task_heads[task_name].parameters():
                param.requires_grad = False
            self.task_status[task_name] = False
            
    def unfreeze_task(self, task_name):
        """解冻特定任务的参数"""
        if task_name in self.task_heads:
            for param in self.task_heads[task_name].parameters():
                param.requires_grad = True
            self.task_status[task_name] = True
```

## 阶段3: 标注质量控制系统

### 3.1 一致性检查机制
```python
class AnnotationQualityControl:
    def __init__(self):
        self.consistency_threshold = 0.8
        self.confidence_threshold = 0.9
        
    def multi_annotator_agreement(self, annotations):
        """多标注员一致性检查"""
        agreements = {}
        
        for task in ['growth_level', 'growth_pattern', 'interference', 'fine_grained']:
            if task in annotations:
                # 计算Fleiss' kappa
                agreement_score = self.calculate_fleiss_kappa(annotations[task])
                agreements[task] = agreement_score
                
        return agreements
    
    def model_assisted_validation(self, model, new_annotations):
        """模型辅助验证标注质量"""
        model.eval()
        discrepancies = []
        
        with torch.no_grad():
            for sample, annotation in new_annotations:
                prediction = model(sample.unsqueeze(0))
                
                # 检查模型预测与人工标注的差异
                for task, pred in prediction.items():
                    if task in annotation:
                        prob = F.softmax(pred, dim=1) if task != 'interference' else torch.sigmoid(pred)
                        confidence = torch.max(prob).item()
                        predicted_label = torch.argmax(pred, dim=1).item() if task != 'interference' else (prob > 0.5).int()
                        
                        if confidence > self.confidence_threshold and predicted_label != annotation[task]:
                            discrepancies.append({
                                'sample_id': sample.id,
                                'task': task,
                                'human_label': annotation[task],
                                'model_prediction': predicted_label,
                                'model_confidence': confidence
                            })
                            
        return discrepancies
    
    def suggest_reannotation(self, discrepancies):
        """建议重新标注的样本"""
        high_priority = []
        
        for disc in discrepancies:
            if disc['model_confidence'] > 0.95:  # 模型非常确信
                high_priority.append(disc)
                
        return sorted(high_priority, key=lambda x: x['model_confidence'], reverse=True)
```

### 3.2 标注辅助工具设计
```python
class SmartAnnotationTool:
    def __init__(self, model):
        self.model = model
        self.annotation_history = []
        
    def get_annotation_suggestions(self, image):
        """为图像提供标注建议"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(image.unsqueeze(0))
            suggestions = {}
            
            for task, pred in predictions.items():
                if task == 'interference':  # 多标签
                    probs = torch.sigmoid(pred)
                    confident_labels = (probs > 0.8).int().squeeze()
                    uncertain_labels = ((probs > 0.3) & (probs < 0.7)).int().squeeze()
                    
                    suggestions[task] = {
                        'confident': confident_labels.tolist(),
                        'uncertain': uncertain_labels.tolist(),
                        'probabilities': probs.squeeze().tolist()
                    }
                else:  # 分类任务
                    probs = F.softmax(pred, dim=1).squeeze()
                    max_prob, max_idx = torch.max(probs, dim=0)
                    
                    suggestions[task] = {
                        'predicted_class': max_idx.item(),
                        'confidence': max_prob.item(),
                        'all_probabilities': probs.tolist()
                    }
                    
        return suggestions
    
    def adaptive_interface(self, task_progress):
        """根据标注进度自适应界面"""
        interface_config = {}
        
        # 根据已有标注数量调整界面复杂度
        if task_progress['total_annotated'] < 3000:
            interface_config['show_predictions'] = True
            interface_config['show_confidence'] = True
            interface_config['enable_batch_mode'] = False
        elif task_progress['total_annotated'] < 7000:
            interface_config['show_predictions'] = True
            interface_config['show_confidence'] = False
            interface_config['enable_batch_mode'] = True
        else:
            interface_config['show_predictions'] = False
            interface_config['show_confidence'] = False
            interface_config['enable_batch_mode'] = True
            interface_config['enable_auto_annotation'] = True
            
        return interface_config
```

## 阶段4: 性能优化策略

### 4.1 小样本学习增强
```python
def few_shot_enhancement_pipeline():
    """小样本场景下的性能增强"""
    enhancement_strategies = [
        # 1. 数据增强
        {
            'name': 'Progressive Data Augmentation',
            'method': create_progressive_augmentation(),
            'expected_gain': '3-5% accuracy'
        },
        
        # 2. 元学习
        {
            'name': 'MAML for Multi-task',
            'method': implement_maml_multitask(),
            'expected_gain': '2-4% accuracy'
        },
        
        # 3. 自监督预训练
        {
            'name': 'Contrastive Learning',
            'method': contrastive_pretraining(),
            'expected_gain': '5-8% accuracy'
        },
        
        # 4. 知识蒸馏
        {
            'name': 'Teacher-Student Framework',
            'method': progressive_distillation(),
            'expected_gain': '2-3% accuracy'
        }
    ]
    
    return enhancement_strategies

def create_progressive_augmentation():
    """渐进式数据增强"""
    def augment_by_stage(stage, samples_count):
        if samples_count < 3000:
            # 小样本：保守增强
            return A.Compose([
                A.RandomRotate90(p=0.3),
                A.Flip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2)
            ])
        elif samples_count < 7000:
            # 中等样本：适度增强
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(var_limit=(0.0, 0.005), p=0.2)
            ])
        else:
            # 大样本：激进增强
            return A.Compose([
                A.RandomRotate90(p=0.7),
                A.Flip(p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.GaussNoise(var_limit=(0.0, 0.01), p=0.3),
                A.ElasticTransform(p=0.2)
            ])
    
    return augment_by_stage
```

### 4.2 动态损失权重调整
```python
class DynamicLossWeighting:
    def __init__(self, num_tasks=4):
        self.num_tasks = num_tasks
        self.task_losses_history = []
        self.weights = torch.ones(num_tasks)
        
    def update_weights(self, current_losses, epoch):
        """基于任务学习进度动态调整权重"""
        self.task_losses_history.append(current_losses)
        
        if len(self.task_losses_history) > 10:  # 有足够历史数据
            # 计算各任务的学习速度
            recent_losses = torch.stack(self.task_losses_history[-10:])
            loss_trends = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
            
            # 学习慢的任务给予更高权重
            self.weights = F.softmax(-loss_trends, dim=0) * self.num_tasks
            
    def get_weighted_loss(self, task_losses):
        """计算加权总损失"""
        weighted_losses = self.weights * torch.stack(list(task_losses.values()))
        return torch.sum(weighted_losses)
```

## 成本效益分析

### 标注成本估算
```python
annotation_cost_analysis = {
    'batch_1': {
        'samples': 3000,
        'time_per_sample': '45秒',  # 包含growth_level
        'total_time': '37.5小时',
        'cost_estimate': '¥3,000-5,000'
    },
    'batch_2': {
        'samples': 2000,
        'time_per_sample': '60秒',  # 增加growth_pattern
        'total_time': '33.3小时',
        'cost_estimate': '¥3,500-6,000'
    },
    'batch_3': {
        'samples': 2000,
        'time_per_sample': '75秒',  # 完善pattern + interference
        'total_time': '41.7小时',
        'cost_estimate': '¥4,500-7,500'
    },
    'batch_4': {
        'samples': 3000,
        'time_per_sample': '90秒',  # 40类fine-grained
        'total_time': '75小时',
        'cost_estimate': '¥7,500-12,000'
    },
    'batch_5': {
        'samples': 5000,
        'time_per_sample': '30秒',  # 模型辅助，主要验证
        'total_time': '41.7小时',
        'cost_estimate': '¥4,000-6,500'
    }
}

# 总估算
total_cost = {
    'total_time': '229.2小时',
    'total_cost': '¥22,500-37,000',
    'cost_per_sample': '¥1.5-2.5'  # 最终单样本成本
}
```

### 性能预期
```python
performance_milestones = {
    '3k_samples': {
        'growth_level_acc': '85-90%',
        'overall_confidence': '中等'
    },
    '5k_samples': {
        'growth_level_acc': '90-93%',
        'growth_pattern_acc': '75-82%',
        'overall_confidence': '较高'
    },
    '7k_samples': {
        'growth_level_acc': '92-95%',
        'growth_pattern_acc': '82-87%',
        'interference_map': '70-78%',
        'overall_confidence': '高'
    },
    '10k_samples': {
        'all_tasks_baseline': '达到实用水平',
        'fine_grained_acc': '65-75%',
        'joint_accuracy': '55-65%',
        'overall_confidence': '很高'
    },
    '15k_samples': {
        'fine_grained_acc': '75-82%',
        'joint_accuracy': '65-75%',
        'production_ready': True
    }
}
```

## 实施建议

### 立即行动项 (本周)
1. **选择首批3000样本**：使用智能采样策略
2. **搭建基础pipeline**：数据加载、预处理、评估
3. **训练baseline模型**：EfficientNet-B0 + 二分类

### 短期目标 (1个月)
1. **完成首批标注**：3000样本的完整标注
2. **验证技术可行性**：达到85%+的growth_level准确率
3. **优化标注工具**：基于首批经验改进标注界面

### 中期目标 (3个月)
1. **达到7000样本**：累计标注量
2. **多任务系统初步成型**：3个主要任务达到实用水平
3. **标注效率优化**：模型辅助标注节省50%时间

这个策略的最大优势是**风险可控、成本渐进、效果可验证**。每个阶段都有明确的成功指标，可以根据实际效果调整后续策略。