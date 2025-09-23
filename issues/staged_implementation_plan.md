# 基于15000样本的分阶段实施方案

## 数据现状分析

### 当前数据分布
- **总样本**: 15,000个
- **已标注**: Growth Level (negative/positive 二分类)
- **待标注**: 3个细分类 + 9个模式类 + 3个干扰因素 + 40个精细类别

### 预估标注工作量
```
阶段性标注策略:
├── 阶段1: Growth Level 三分类 (负样本细分) - 约1周
├── 阶段2: Growth Pattern 标注 - 约2-3周  
├── 阶段3: Interference Mapping - 约1-2周
└── 阶段4: Fine-grained 40类 - 约3-4周
```

## 分阶段实施策略

### 🚀 阶段1: 基础二分类优化 (当前可立即执行)

#### 1.1 利用现有数据建立强基线
```python
# 阶段1架构
class Stage1Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用EfficientNet-B0作为backbone
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        
        # 二分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # negative/positive
        )
        
        # 特征提取器 - 为后续任务保存特征
        self.feature_extractor = nn.Identity()
    
    def forward(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features).flatten(1)
        
        # 保存特征用于后续分析
        extracted_features = self.feature_extractor(features)
        logits = self.classifier(features)
        
        return logits, extracted_features
```

#### 1.2 数据增强策略
```python
def get_stage1_transforms():
    return {
        'train': A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=0.3
            ),
            A.GaussNoise(var_limit=(0.0, 0.01), p=0.2),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2()
        ])
    }
```

#### 1.3 关键收益
- 快速验证数据质量和模型可行性
- 建立evaluation pipeline
- 提取特征用于后续标注辅助

### 🎯 阶段2: 三分类扩展 (标注完成后)

#### 2.1 smart标注策略
```python
# 利用已训练模型辅助标注
def assisted_labeling_pipeline(model, unlabeled_data):
    """
    使用已训练的二分类模型辅助三分类标注
    """
    model.eval()
    predictions = []
    uncertainties = []
    
    with torch.no_grad():
        for batch in unlabeled_data:
            logits, features = model(batch)
            probs = F.softmax(logits, dim=1)
            
            # 计算不确定性
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            uncertainties.extend(entropy.cpu().numpy())
            predictions.extend(probs.cpu().numpy())
    
    # 优先标注高不确定性样本
    uncertain_indices = np.argsort(uncertainties)[::-1]
    
    return {
        'priority_samples': uncertain_indices[:1000],  # 优先标注1000个
        'confident_samples': uncertain_indices[-5000:], # 可能自动标注
        'predictions': predictions
    }
```

#### 2.2 渐进式学习架构
```python
class Stage2Progressive(nn.Module):
    def __init__(self, stage1_model):
        super().__init__()
        # 继承stage1的特征提取器
        self.backbone = stage1_model.backbone
        
        # 冻结部分layers进行渐进学习
        for param in self.backbone.features[:4].parameters():
            param.requires_grad = False
            
        # 三分类头
        self.growth_level_head = nn.Linear(1280, 3)  # negative, positive, weak_growth
        
        # 为下个阶段准备的特征
        self.shared_features = nn.Linear(1280, 512)
        
    def forward(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features).flatten(1)
        
        shared_feat = self.shared_features(features)
        growth_level = self.growth_level_head(features)
        
        return growth_level, shared_feat
```

### 🔄 阶段3: 多任务架构构建

#### 3.1 渐进式多任务学习
```python
class ProgressiveMultiTask(nn.Module):
    def __init__(self, pretrained_backbone):
        super().__init__()
        self.backbone = pretrained_backbone
        
        # 共享特征层
        self.shared_features = nn.Linear(1280, 512)
        
        # 任务特定头部
        self.heads = nn.ModuleDict({
            'growth_level': nn.Linear(512, 3),
            'growth_pattern': nn.Linear(512 + 3, 9),  # 包含growth_level信息
            'interference': nn.Linear(512, 3),        # 多标签
            'fine_grained': nn.Linear(512 + 3 + 9, 40)  # 包含所有上级信息
        })
        
        # 任务权重学习
        self.task_weights = nn.Parameter(torch.ones(4))
        
    def forward(self, x, available_tasks=None):
        # 特征提取
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features).flatten(1)
        shared_feat = self.shared_features(features)
        
        outputs = {}
        
        # 层次化预测
        growth_level = self.heads['growth_level'](shared_feat)
        outputs['growth_level'] = growth_level
        
        if 'growth_pattern' in available_tasks:
            pattern_input = torch.cat([shared_feat, growth_level], dim=1)
            outputs['growth_pattern'] = self.heads['growth_pattern'](pattern_input)
        
        if 'interference' in available_tasks:
            outputs['interference'] = self.heads['interference'](shared_feat)
            
        if 'fine_grained' in available_tasks:
            fine_input = torch.cat([
                shared_feat, 
                growth_level, 
                outputs.get('growth_pattern', torch.zeros_like(growth_level))
            ], dim=1)
            outputs['fine_grained'] = self.heads['fine_grained'](fine_input)
            
        return outputs
```

#### 3.2 智能标注辅助系统
```python
class SmartAnnotationAssistant:
    def __init__(self, model, confidence_threshold=0.9):
        self.model = model
        self.threshold = confidence_threshold
        
    def suggest_labels(self, images):
        """为新样本建议标签"""
        self.model.eval()
        suggestions = []
        
        with torch.no_grad():
            predictions = self.model(images)
            
            for task, pred in predictions.items():
                probs = F.softmax(pred, dim=1)
                max_probs, max_indices = torch.max(probs, dim=1)
                
                # 高置信度样本可以自动标注
                confident_mask = max_probs > self.threshold
                
                suggestions.append({
                    'task': task,
                    'auto_labels': max_indices[confident_mask],
                    'manual_review': max_indices[~confident_mask],
                    'confidences': max_probs
                })
                
        return suggestions
    
    def active_learning_selection(self, unlabeled_pool, n_samples=500):
        """选择最有价值的样本进行标注"""
        uncertainties = self.calculate_uncertainty(unlabeled_pool)
        diversity_scores = self.calculate_diversity(unlabeled_pool)
        
        # 结合不确定性和多样性选择样本
        combined_scores = 0.7 * uncertainties + 0.3 * diversity_scores
        selected_indices = np.argsort(combined_scores)[-n_samples:]
        
        return selected_indices
```

## 数据管理策略

### 数据分割建议
```python
def create_stratified_splits(data, test_size=0.15, val_size=0.15):
    """
    创建分层抽样的数据分割，确保各类别在各集合中比例一致
    """
    # 基于growth_level进行分层
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, test_size=test_size, 
        stratify=labels['growth_level'], random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size),
        stratify=y_temp['growth_level'], random_state=42
    )
    
    return {
        'train': (X_train, y_train),     # ~10,500 samples
        'val': (X_val, y_val),           # ~2,250 samples  
        'test': (X_test, y_test)         # ~2,250 samples
    }
```

### 数据质量控制
```python
class DataQualityController:
    def __init__(self):
        self.quality_checkers = [
            BlurrinessDetector(threshold=0.1),
            ContrastChecker(min_contrast=0.05),
            NoiseEstimator(max_noise_level=0.1),
            LabelConsistencyChecker()
        ]
    
    def quality_audit(self, dataset):
        """数据质量审计"""
        issues = []
        
        for checker in self.quality_checkers:
            flagged_samples = checker.check(dataset)
            issues.extend(flagged_samples)
            
        # 生成质量报告
        quality_report = {
            'total_samples': len(dataset),
            'flagged_samples': len(set(issues)),
            'quality_score': 1 - len(set(issues)) / len(dataset),
            'issues_by_type': Counter([issue.type for issue in issues])
        }
        
        return quality_report, issues
```

## 性能监控与评估

### 多任务评估框架
```python
class MultiTaskEvaluator:
    def __init__(self):
        self.metrics = {
            'classification': ['accuracy', 'f1_macro', 'f1_weighted'],
            'multilabel': ['average_precision', 'f1_micro', 'f1_macro'],
            'consistency': ['joint_accuracy', 'task_agreement']
        }
    
    def comprehensive_eval(self, predictions, targets):
        results = {}
        
        # 单任务指标
        for task in ['growth_level', 'growth_pattern', 'fine_grained']:
            results[task] = self.evaluate_classification(
                predictions[task], targets[task]
            )
        
        # 多标签任务
        results['interference'] = self.evaluate_multilabel(
            predictions['interference'], targets['interference']
        )
        
        # 跨任务一致性
        results['consistency'] = self.evaluate_consistency(predictions, targets)
        
        # 整体性能
        results['overall'] = self.compute_overall_score(results)
        
        return results
```

## 实施时间线

### 第1-2周: 基础建设
- [ ] 数据pipeline搭建
- [ ] 二分类baseline训练
- [ ] 评估框架建立
- [ ] 初步性能验证

### 第3-4周: 标注扩展
- [ ] Growth Level三分类标注
- [ ] 辅助标注工具开发
- [ ] 三分类模型训练
- [ ] 性能对比分析

### 第5-8周: 多任务构建
- [ ] Growth Pattern标注完成
- [ ] 多任务架构实现
- [ ] 渐进式学习实验
- [ ] 超参数优化

### 第9-12周: 完整系统
- [ ] 所有任务标注完成
- [ ] 完整多任务模型
- [ ] 高级优化技术
- [ ] 最终性能评估

## 关键成功要素

### 1. 标注质量控制
- 建立标注guideline和示例
- 多人标注一致性检查
- 难例讨论机制
- 标注工具界面优化

### 2. 模型迭代策略
- 每个阶段保持详细实验记录
- A/B测试不同架构选择
- 渐进式复杂度增加
- 及时调整based on中间结果

### 3. 数据利用效率
- 充分利用unlabeled数据进行预训练
- 主动学习减少标注工作量
- 数据增强提高样本多样性
- 跨任务知识迁移

## 预期性能目标

基于15000样本规模，预期各任务性能：

| 任务 | 预期准确率 | 挑战程度 |
|------|-----------|----------|
| Growth Level (3类) | 90-95% | 低 |
| Growth Pattern (9类) | 80-88% | 中 |
| Interference (多标签) | 75-85% | 中-高 |
| Fine-grained (40类) | 70-80% | 高 |
| Joint Accuracy | 60-75% | 很高 |

这个渐进式策略能够：
1. **最大化现有数据价值**：立即开始训练而不等待全部标注
2. **降低标注负担**：利用模型辅助标注
3. **风险控制**：每个阶段验证可行性
4. **知识积累**：前期经验指导后期优化