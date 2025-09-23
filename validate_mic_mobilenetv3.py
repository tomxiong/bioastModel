#!/usr/bin/env python3
"""
M16 MultiTask MobileNetV3模型验证脚本
使用NI多任务测试集对训练好的M16_MultiTask_MobileNetV3模型进行性能验证
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           roc_auc_score, confusion_matrix, classification_report,
                           multilabel_confusion_matrix, hamming_loss)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.ni_multitask_dataset import NIMultitaskDataset
from models.multitask_models import MultitaskBioastModel

class M16MultitaskValidator:
    def __init__(self, model_path, data_dir="dataset_ni_multitask", device=None):
        """
        初始化验证器
        
        Args:
            model_path: 模型文件路径
            data_dir: NI多任务数据集路径
            device: 使用的设备
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"模型文件: {model_path}")
        print(f"数据集路径: {data_dir}")
        
        # 数据预处理 - 与训练时保持一致
        self.transform = transforms.Compose([
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载测试数据集
        self.test_dataset = NIMultitaskDataset(
            data_root=self.data_dir,
            split='test',
            transform=self.transform,
            target_size=(70, 70),
            grayscale=False  # 使用RGB
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"测试集大小: {len(self.test_dataset)}")
        print(f"批次数量: {len(self.test_loader)}")
        
        # 获取任务配置
        self.task_configs = self.test_dataset.dataset_info['tasks']
        print(f"任务配置: {list(self.task_configs.keys())}")
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self):
        """加载训练好的M16多任务模型"""
        print("正在加载M16 MultiTask MobileNetV3模型...")
        
        # 创建任务配置
        task_configs = {
            'growth_level': {
                'num_classes': 3,
                'multilabel': False,
                'weight': 1.0
            },
            'growth_pattern': {
                'num_classes': 9,
                'multilabel': False,
                'weight': 1.0
            },
            'interference_factors': {
                'num_classes': 5,  # 根据dataset_info.json调整
                'multilabel': True,
                'weight': 0.5
            },
            'fine_grained': {
                'num_classes': 8,  # 根据dataset_info.json调整
                'multilabel': False,
                'weight': 1.0
            }
        }
        
        # 创建模型
        model = MultitaskBioastModel(
            backbone_name='mic_mobilenetv3',
            task_configs=task_configs,
            feature_dim=576,  # 根据实际backbone输出调整
            dropout_rate=0.15,
            use_attention=True
        )
        
        # 加载权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"从checkpoint中加载model_state_dict")
                if 'best_val_accuracy' in checkpoint:
                    print(f"训练时最佳验证准确率: {checkpoint['best_val_accuracy']:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("直接加载checkpoint")
                
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        model = model.to(self.device)
        model.eval()
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型加载完成: M16_MultiTask_MobileNetV3")
        print(f"   参数数量: {total_params:,}")
        print(f"   任务数量: {len(task_configs)}")
        
        return model
    
    def evaluate(self, save_results=True):
        """
        评估多任务模型性能
        
        Args:
            save_results: 是否保存结果
            
        Returns:
            dict: 评估结果
        """
        print("\n开始M16多任务模型性能评估...")
        print("="*50)
        
        # 存储所有任务的预测结果
        task_results = {}
        for task_name in self.task_configs:
            task_results[task_name] = {
                'predictions': [],
                'labels': [],
                'probabilities': []
            }
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.test_loader):
                images = images.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                batch_size = images.size(0)
                total_samples += batch_size
                
                # 处理每个任务的输出
                for task_name in self.task_configs:
                    if task_name in outputs and task_name in targets:
                        logits = outputs[task_name]
                        labels = targets[task_name].cpu().numpy()
                        
                        # 计算概率
                        if self.task_configs[task_name]['type'] == 'multi_label_classification':
                            # 多标签：sigmoid
                            probs = torch.sigmoid(logits).cpu().numpy()
                            preds = (probs > 0.5).astype(int)
                        else:
                            # 单标签：softmax
                            probs = torch.softmax(logits, dim=1).cpu().numpy()
                            preds = np.argmax(probs, axis=1)
                        
                        task_results[task_name]['predictions'].extend(preds)
                        task_results[task_name]['labels'].extend(labels)
                        task_results[task_name]['probabilities'].extend(probs)
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"处理进度: {batch_idx + 1}/{len(self.test_loader)} 批次")
        
        # 计算每个任务的性能指标
        task_metrics = {}
        
        for task_name, task_config in self.task_configs.items():
            if task_name not in task_results or not task_results[task_name]['predictions']:
                continue
                
            predictions = np.array(task_results[task_name]['predictions'])
            labels = np.array(task_results[task_name]['labels'])
            probabilities = np.array(task_results[task_name]['probabilities'])
            
            task_metrics[task_name] = self._compute_task_metrics(
                task_name, task_config, predictions, labels, probabilities
            )
        
        # 整合结果
        results = {
            'model_info': {
                'name': 'M16_MultiTask_MobileNetV3',
                'path': self.model_path,
                'test_samples': total_samples,
                'tasks': list(self.task_configs.keys())
            },
            'task_metrics': task_metrics,
            'overall_performance': self._compute_overall_metrics(task_metrics),
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'data_dir': self.data_dir
            }
        }
        
        # 打印结果
        self._print_results(results)
        
        # 保存结果
        if save_results:
            self._save_results(results)
        
        return results
    
    def _compute_task_metrics(self, task_name, task_config, predictions, labels, probabilities):
        """计算单个任务的性能指标"""
        metrics = {}
        
        if task_config['type'] == 'multi_label_classification':
            # 多标签分类指标
            metrics['hamming_loss'] = hamming_loss(labels, predictions)
            metrics['accuracy'] = accuracy_score(labels, predictions)
            
            # 每个类别的指标
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, predictions, average=None, zero_division=0
                )
                metrics['precision_per_class'] = precision.tolist()
                metrics['recall_per_class'] = recall.tolist()
                metrics['f1_per_class'] = f1.tolist()
                metrics['support_per_class'] = support.tolist()
                
                # 宏平均和微平均
                metrics['precision_macro'] = precision_recall_fscore_support(
                    labels, predictions, average='macro', zero_division=0
                )[0]
                metrics['recall_macro'] = precision_recall_fscore_support(
                    labels, predictions, average='macro', zero_division=0
                )[1]
                metrics['f1_macro'] = precision_recall_fscore_support(
                    labels, predictions, average='macro', zero_division=0
                )[2]
                
                metrics['precision_micro'] = precision_recall_fscore_support(
                    labels, predictions, average='micro', zero_division=0
                )[0]
                metrics['recall_micro'] = precision_recall_fscore_support(
                    labels, predictions, average='micro', zero_division=0
                )[1]
                metrics['f1_micro'] = precision_recall_fscore_support(
                    labels, predictions, average='micro', zero_division=0
                )[2]
            except Exception as e:
                print(f"计算{task_name}多标签指标时出错: {e}")
                
        else:
            # 单标签分类指标
            metrics['accuracy'] = accuracy_score(labels, predictions)
            
            try:
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, predictions, average='weighted', zero_division=0
                )
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1_score'] = f1
                
                # 每个类别的指标
                precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                    labels, predictions, average=None, zero_division=0
                )
                metrics['precision_per_class'] = precision_per_class.tolist()
                metrics['recall_per_class'] = recall_per_class.tolist()
                metrics['f1_per_class'] = f1_per_class.tolist()
                metrics['support_per_class'] = support_per_class.tolist()
                
                # 混淆矩阵
                cm = confusion_matrix(labels, predictions)
                metrics['confusion_matrix'] = cm.tolist()
                
                # AUC (如果是二分类或多分类)
                try:
                    if task_config['num_classes'] == 2:
                        metrics['auc'] = roc_auc_score(labels, probabilities[:, 1])
                    elif task_config['num_classes'] > 2:
                        metrics['auc'] = roc_auc_score(labels, probabilities, 
                                                     multi_class='ovr', average='weighted')
                except Exception as e:
                    print(f"计算{task_name} AUC时出错: {e}")
                    metrics['auc'] = 0.0
                    
            except Exception as e:
                print(f"计算{task_name}单标签指标时出错: {e}")
        
        return metrics
    
    def _compute_overall_metrics(self, task_metrics):
        """计算整体性能指标"""
        overall = {}
        
        # 计算所有任务的平均准确率
        accuracies = []
        f1_scores = []
        
        for task_name, metrics in task_metrics.items():
            if 'accuracy' in metrics:
                accuracies.append(metrics['accuracy'])
            if 'f1_score' in metrics:
                f1_scores.append(metrics['f1_score'])
            elif 'f1_macro' in metrics:
                f1_scores.append(metrics['f1_macro'])
        
        if accuracies:
            overall['mean_accuracy'] = np.mean(accuracies)
        if f1_scores:
            overall['mean_f1_score'] = np.mean(f1_scores)
            
        # 各任务性能汇总
        overall['task_summary'] = {}
        for task_name, metrics in task_metrics.items():
            summary = {}
            if 'accuracy' in metrics:
                summary['accuracy'] = metrics['accuracy']
            if 'f1_score' in metrics:
                summary['f1_score'] = metrics['f1_score']
            elif 'f1_macro' in metrics:
                summary['f1_score'] = metrics['f1_macro']
            if 'auc' in metrics:
                summary['auc'] = metrics['auc']
            overall['task_summary'][task_name] = summary
            
        return overall
    
    def _print_results(self, results):
        """打印多任务评估结果"""
        print("\n" + "="*60)
        print("📊 M16 MultiTask MobileNetV3 模型评估结果")
        print("="*60)
        
        model_info = results['model_info']
        print(f"🎯 模型信息:")
        print(f"   模型名称: {model_info['name']}")
        print(f"   测试样本: {model_info['test_samples']}")
        print(f"   任务数量: {len(model_info['tasks'])}")
        print(f"   任务列表: {', '.join(model_info['tasks'])}")
        
        # 整体性能
        overall = results['overall_performance']
        print(f"\n🏆 整体性能:")
        if 'mean_accuracy' in overall:
            print(f"   平均准确率: {overall['mean_accuracy']:.4f}")
        if 'mean_f1_score' in overall:
            print(f"   平均F1分数: {overall['mean_f1_score']:.4f}")
        
        # 各任务详细结果
        print(f"\n📋 各任务详细性能:")
        task_metrics = results['task_metrics']
        
        for task_name, metrics in task_metrics.items():
            task_config = self.task_configs[task_name]
            print(f"\n   📝 {task_name} ({task_config['num_classes']}类):")
            
            if 'accuracy' in metrics:
                print(f"      准确率: {metrics['accuracy']:.4f}")
            if 'f1_score' in metrics:
                print(f"      F1分数: {metrics['f1_score']:.4f}")
            elif 'f1_macro' in metrics:
                print(f"      F1分数(宏): {metrics['f1_macro']:.4f}")
            if 'auc' in metrics and metrics['auc'] > 0:
                print(f"      AUC: {metrics['auc']:.4f}")
            if 'hamming_loss' in metrics:
                print(f"      汉明损失: {metrics['hamming_loss']:.4f}")
                
            # 显示混淆矩阵维度信息
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                print(f"      混淆矩阵: {cm.shape[0]}×{cm.shape[1]}")
        
        print(f"\n✅ 评估完成！")
        
    def _save_results(self, results):
        """保存评估结果"""
        # 确定保存路径
        model_dir = Path(self.model_path).parent
        results_file = model_dir / "multitask_validation_results.json"
        
        # 保存JSON结果
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 验证结果已保存到: {results_file}")
        
        # 生成可视化
        self._create_visualizations(results, model_dir)
    
    def _create_visualizations(self, results, output_dir):
        """创建多任务可视化图表"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        task_metrics = results['task_metrics']
        
        # 1. 各任务准确率对比
        plt.figure(figsize=(12, 6))
        task_names = []
        accuracies = []
        
        for task_name, metrics in task_metrics.items():
            if 'accuracy' in metrics:
                task_names.append(task_name.replace('_', '\n'))
                accuracies.append(metrics['accuracy'])
        
        if task_names and accuracies:
            bars = plt.bar(task_names, accuracies, 
                          color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            plt.title('M16 MultiTask MobileNetV3 - 各任务准确率', fontsize=14, fontweight='bold')
            plt.ylabel('准确率')
            plt.ylim(0, 1.05)
            
            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'multitask_accuracies.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. F1分数对比
        plt.figure(figsize=(12, 6))
        f1_scores = []
        
        for task_name, metrics in task_metrics.items():
            if 'f1_score' in metrics:
                f1_scores.append(metrics['f1_score'])
            elif 'f1_macro' in metrics:
                f1_scores.append(metrics['f1_macro'])
            else:
                f1_scores.append(0)
        
        if task_names and f1_scores:
            bars = plt.bar(task_names, f1_scores, 
                          color=['#16A085', '#8E44AD', '#E67E22', '#E74C3C'])
            plt.title('M16 MultiTask MobileNetV3 - 各任务F1分数', fontsize=14, fontweight='bold')
            plt.ylabel('F1分数')
            plt.ylim(0, 1.05)
            
            # 添加数值标签
            for bar, f1 in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'multitask_f1_scores.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 为每个任务创建混淆矩阵（如果有的话）
        for task_name, metrics in task_metrics.items():
            if 'confusion_matrix' in metrics:
                plt.figure(figsize=(8, 6))
                cm = np.array(metrics['confusion_matrix'])
                
                # 获取类别名称
                class_names = self.test_dataset.dataset_info['label_mappings'][task_name]
                labels = [name for name, _ in sorted(class_names.items(), key=lambda x: x[1])]
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=labels[:cm.shape[1]], 
                           yticklabels=labels[:cm.shape[0]])
                plt.title(f'{task_name.replace("_", " ").title()} - 混淆矩阵')
                plt.ylabel('真实标签')
                plt.xlabel('预测标签')
                plt.tight_layout()
                plt.savefig(output_dir / f'{task_name}_confusion_matrix.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"📊 可视化图表已保存到: {output_dir}")

def main():
    """主函数"""
    print("🔍 M16 MultiTask MobileNetV3 模型验证器")
    print("="*50)
    
    # 模型路径
    model_path = "experiments/mic_mobilenetv3/best.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保模型已经训练完成并且路径正确")
        return
    
    # 检查数据集
    data_dir = "dataset_ni_multitask"
    if not os.path.exists(data_dir):
        print(f"❌ 数据集不存在: {data_dir}")
        print("请确保多任务数据集已经创建")
        return
    
    try:
        # 创建验证器
        validator = M16MultitaskValidator(
            model_path=model_path,
            data_dir=data_dir
        )
        
        # 执行验证
        results = validator.evaluate(save_results=True)
        
        print("\n🎉 验证完成！")
        overall = results['overall_performance']
        if 'mean_accuracy' in overall:
            print(f"   平均准确率: {overall['mean_accuracy']:.4f}")
        if 'mean_f1_score' in overall:
            print(f"   平均F1分数: {overall['mean_f1_score']:.4f}")
        
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()