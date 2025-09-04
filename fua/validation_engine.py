"""
FUA 验证和分析模块
支持独立验证集测试、模型对比和深度分析
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ValidationEngine:
    """验证引擎 - 在独立数据集上测试模型"""
    
    def __init__(self, config_path: str = "fua/validation_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results_path = Path(self.config["results_path"])
        self.results_path.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """加载配置"""
        default_config = {
            "results_path": "fua/validation_results",
            "validation_datasets": {
                "external": "external_validation_set",
                "stress_test": "stress_test_set",
                "edge_cases": "edge_cases_set"
            },
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc",
                "confusion_matrix",
                "per_class_metrics"
            ],
            "visualization": {
                "save_plots": True,
                "plot_format": "png",
                "dpi": 300
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def validate_model(self, model_path: str, dataset_path: str, 
                      model_name: str = None, dataset_name: str = None) -> Dict:
        """在指定数据集上验证模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        validation_id = f"{model_name or 'unknown'}_{dataset_name or 'unknown'}_{timestamp}"
        
        result_dir = self.results_path / validation_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 加载模型
            model = self._load_model(model_path)
            model.eval()
            
            # 加载数据集
            dataset = self._load_dataset(dataset_path)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # 运行验证
            predictions, targets, probabilities = self._run_inference(model, dataloader)
            
            # 计算指标
            metrics = self._calculate_metrics(predictions, targets, probabilities)
            
            # 生成分析
            analysis = self._generate_analysis(predictions, targets, probabilities, dataset)
            
            # 生成可视化
            if self.config["visualization"]["save_plots"]:
                self._generate_visualizations(predictions, targets, probabilities, result_dir)
            
            # 保存结果
            result = {
                "validation_id": validation_id,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "model_path": model_path,
                "dataset_path": dataset_path,
                "timestamp": timestamp,
                "metrics": metrics,
                "analysis": analysis,
                "sample_count": len(dataset),
                "result_dir": str(result_dir)
            }
            
            with open(result_dir / "validation_result.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            return {
                "validation_id": validation_id,
                "error": str(e),
                "timestamp": timestamp,
                "success": False
            }
    
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        # 这里需要根据实际的模型格式进行调整
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 假设checkpoint包含模型状态
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 需要知道模型结构才能加载
            # 这里简化处理，实际需要根据模型类型创建相应实例
            model = self._create_model_from_checkpoint(checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 假设checkpoint直接是模型状态
            model = self._create_model_from_checkpoint(checkpoint)
            model.load_state_dict(checkpoint)
        
        return model
    
    def _create_model_from_checkpoint(self, checkpoint: Dict) -> nn.Module:
        """从检查点创建模型（简化实现）"""
        # 实际实现需要根据模型类型动态创建
        # 这里只是一个示例
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(64, 2)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return DummyModel()
    
    def _load_dataset(self, dataset_path: str):
        """加载数据集"""
        # 简化实现，实际需要使用项目的数据加载器
        class DummyDataset:
            def __init__(self, path):
                self.path = Path(path)
                self.samples = list(self.path.glob("**/*.jpg"))
                self.transform = self._get_transform()
            
            def _get_transform(self):
                from torchvision import transforms
                return transforms.Compose([
                    transforms.Resize((70, 70)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path = self.samples[idx]
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transform(image)
                
                # 从路径推断标签
                label = 1 if 'positive' in str(img_path) else 0
                
                return image, label
        
        return DummyDataset(dataset_path)
    
    def _run_inference(self, model: nn.Module, dataloader: DataLoader) -> Tuple:
        """运行推理"""
        device = next(model.parameters()).device
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                          probabilities: np.ndarray) -> Dict:
        """计算指标"""
        # 基础指标
        cm = confusion_matrix(targets, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,
            "confusion_matrix": cm.tolist(),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        }
        
        # AUC（如果提供了概率）
        if probabilities.shape[1] == 2:
            from sklearn.metrics import roc_auc_score
            try:
                metrics["auc"] = roc_auc_score(targets, probabilities[:, 1])
            except:
                metrics["auc"] = 0.5
        
        # 分类报告
        metrics["classification_report"] = classification_report(
            targets, predictions, output_dict=True
        )
        
        return metrics
    
    def _generate_analysis(self, predictions: np.ndarray, targets: np.ndarray,
                          probabilities: np.ndarray, dataset) -> Dict:
        """生成分析报告"""
        analysis = {
            "error_analysis": {},
            "confidence_analysis": {},
            "recommendations": []
        }
        
        # 错误分析
        errors = predictions != targets
        if np.any(errors):
            error_indices = np.where(errors)[0]
            analysis["error_analysis"] = {
                "error_count": int(np.sum(errors)),
                "error_rate": float(np.mean(errors)),
                "false_positive_rate": float(np.mean((predictions == 1) & (targets == 0))),
                "false_negative_rate": float(np.mean((predictions == 0) & (targets == 1))),
                "confidence_distribution": {
                    "correct": float(np.mean(np.max(probabilities[~errors], axis=1))),
                    "incorrect": float(np.mean(np.max(probabilities[errors], axis=1)))
                }
            }
            
            # 生成建议
            if analysis["error_analysis"]["false_positive_rate"] > 0.1:
                analysis["recommendations"].append(
                    "假阳性率较高，建议调整分类阈值或收集更多负样本"
                )
            
            if analysis["error_analysis"]["false_negative_rate"] > 0.1:
                analysis["recommendations"].append(
                    "假阴性率较高，建议调整模型结构或增强特征提取"
                )
        
        # 置信度分析
        confidences = np.max(probabilities, axis=1)
        analysis["confidence_analysis"] = {
            "mean_confidence": float(np.mean(confidences)),
            "low_confidence_count": int(np.sum(confidences < 0.7)),
            "high_confidence_correct": float(np.mean((confidences > 0.9) & (predictions == targets)))
        }
        
        return analysis
    
    def _generate_visualizations(self, predictions: np.ndarray, targets: np.ndarray,
                                probabilities: np.ndarray, result_dir: Path):
        """生成可视化图表"""
        # 混淆矩阵热图
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(result_dir / f'confusion_matrix.{self.config["visualization"]["plot_format"]}',
                   dpi=self.config["visualization"]["dpi"])
        plt.close()
        
        # 置信度分布
        plt.figure(figsize=(10, 6))
        correct_conf = np.max(probabilities[predictions == targets], axis=1)
        incorrect_conf = np.max(probabilities[predictions != targets], axis=1)
        
        plt.hist(correct_conf, bins=30, alpha=0.7, label='Correct', density=True)
        plt.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', density=True)
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir / f'confidence_distribution.{self.config["visualization"]["plot_format"]}',
                   dpi=self.config["visualization"]["dpi"])
        plt.close()
        
        # ROC曲线（如果适用）
        if probabilities.shape[1] == 2:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(result_dir / f'roc_curve.{self.config["visualization"]["plot_format"]}',
                       dpi=self.config["visualization"]["dpi"])
            plt.close()


class ModelComparator:
    """模型对比器"""
    
    def __init__(self, validation_engine: ValidationEngine):
        self.validation_engine = validation_engine
    
    def compare_models(self, model_configs: List[Dict], 
                      dataset_path: str) -> Dict:
        """对比多个模型"""
        comparison_id = f"comparison_{int(datetime.now().timestamp())}"
        
        results = {}
        for config in model_configs:
            print(f"Validating {config['name']}...")
            result = self.validation_engine.validate_model(
                config["path"],
                dataset_path,
                config["name"],
                config.get("dataset_name", "default")
            )
            results[config["name"]] = result
        
        # 生成对比报告
        comparison_report = self._generate_comparison_report(results, comparison_id)
        
        return {
            "comparison_id": comparison_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "models": list(model_configs),
            "results": results,
            "report": comparison_report
        }
    
    def _generate_comparison_report(self, results: Dict, comparison_id: str) -> Dict:
        """生成对比报告"""
        report = {
            "summary": {},
            "rankings": {},
            "insights": [],
            "recommendations": []
        }
        
        # 收集所有模型的指标
        metrics_data = []
        for model_name, result in results.items():
            if result.get("success") and "metrics" in result:
                metrics = result["metrics"]
                metrics_data.append({
                    "model": model_name,
                    "accuracy": metrics.get("accuracy", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "auc": metrics.get("auc", 0)
                })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # 排名
            report["rankings"] = {
                metric: df.sort_values(metric, ascending=False)["model"].tolist()
                for metric in ["accuracy", "f1_score", "auc"]
            }
            
            # 最佳模型
            best_model = df.loc[df["f1_score"].idxmax(), "model"]
            report["summary"]["best_model"] = best_model
            report["summary"]["best_f1_score"] = df["f1_score"].max()
            
            # 生成洞察
            if df["accuracy"].std() > 0.05:
                report["insights"].append("模型间准确率差异显著，建议进一步分析原因")
            
            # 生成建议
            report["recommendations"].append(
                f"推荐使用 {best_model} 作为生产模型，其F1分数为 {df['f1_score'].max():.4f}"
            )
        
        return report


class ImprovementAnalyzer:
    """改进分析器 - 识别改进机会"""
    
    def __init__(self):
        self.validation_engine = ValidationEngine()
    
    def analyze_improvement_opportunities(self, model_path: str, 
                                         dataset_path: str) -> Dict:
        """分析改进机会"""
        # 运行验证
        result = self.validation_engine.validate_model(model_path, dataset_path)
        
        if not result.get("success"):
            return {"error": "Validation failed", "details": result}
        
        analysis = {
            "model_info": {
                "model_path": model_path,
                "dataset_path": dataset_path,
                "validation_id": result.get("validation_id", "unknown")
            },
            "improvement_areas": [],
            "data_suggestions": [],
            "parameter_suggestions": [],
            "architecture_suggestions": []
        }
        
        metrics = result["metrics"]
        analysis_result = result.get("analysis", {})
        
        # 基于指标分析改进机会
        if metrics["accuracy"] < 0.9:
            analysis["improvement_areas"].append("整体准确率偏低，需要全面优化")
        
        if metrics["precision"] < 0.85:
            analysis["improvement_areas"].append("精确率不足，可能存在过多假阳性")
            analysis["data_suggestions"].append("增加更多负样本或改进负样本质量")
        
        if metrics["recall"] < 0.85:
            analysis["improvement_areas"].append("召回率不足，可能遗漏过多正样本")
            analysis["data_suggestions"].append("增加更多正样本或使用数据增强")
        
        # 基于错误分析
        if "error_analysis" in analysis_result:
            error_rate = analysis_result["error_analysis"].get("error_rate", 0)
            if error_rate > 0.15:
                analysis["parameter_suggestions"].append(
                    "错误率较高，建议调整学习率或增加训练轮数"
                )
        
        # 生成优先级建议
        analysis["priority_actions"] = self._prioritize_actions(analysis)
        
        return analysis
    
    def _prioritize_actions(self, analysis: Dict) -> List[Dict]:
        """优先级排序改进措施"""
        actions = []
        
        # 基于改进区域的紧急程度排序
        for area in analysis["improvement_areas"]:
            if "准确率" in area:
                actions.append({
                    "action": "数据增强和收集",
                    "priority": "high",
                    "description": "增加训练数据量，特别是错误分类的样本"
                })
            elif "精确率" in area:
                actions.append({
                    "action": "调整分类阈值",
                    "priority": "medium",
                    "description": "提高分类阈值以减少假阳性"
                })
            elif "召回率" in area:
                actions.append({
                    "action": "模型结构优化",
                    "priority": "medium",
                    "description": "增加模型容量或使用注意力机制"
                })
        
        return actions


# 使用示例
if __name__ == "__main__":
    # 创建验证引擎
    validator = ValidationEngine()
    
    # 示例：验证模型
    # result = validator.validate_model(
    #     model_path="path/to/model.pth",
    #     dataset_path="path/to/validation_set",
    #     model_name="resnet18",
    #     dataset_name="external_test"
    # )
    # print(f"Validation result: {json.dumps(result, indent=2)}")
    
    # 模型对比
    comparator = ModelComparator(validator)
    # comparison = comparator.compare_models([
    #     {"name": "model_v1", "path": "path/to/model1.pth"},
    #     {"name": "model_v2", "path": "path/to/model2.pth"}
    # ], "path/to/test_set")
    # print(f"Comparison result: {json.dumps(comparison, indent=2)}")
    
    # 改进分析
    analyzer = ImprovementAnalyzer()
    # improvement = analyzer.analyze_improvement_opportunities(
    #     "path/to/model.pth",
    #     "path/to/dataset"
    # )
    # print(f"Improvement analysis: {json.dumps(improvement, indent=2)}")