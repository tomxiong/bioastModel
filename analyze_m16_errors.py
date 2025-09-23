#!/usr/bin/env python3
"""
M16 MultiTask MobileNetV3 错误样本分析工具
专门分析多任务模型在各个任务上的错误样本和错误模式
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from datetime import datetime
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
from torchvision import transforms

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 模拟模型加载（由于实际模型架构复杂，这里使用ONNX模型）
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime不可用，将使用模拟推理")

from training.ni_multitask_dataset import NIMultitaskDataset

class M16ErrorAnalyzer:
    """M16多任务模型错误分析器"""
    
    def __init__(self, model_dir="experiments/mic_mobilenetv3", data_dir="dataset_ni_multitask"):
        """
        初始化分析器
        
        Args:
            model_dir: 模型目录
            data_dir: 数据集目录
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔍 M16 MultiTask 错误样本分析器")
        print(f"模型目录: {self.model_dir}")
        print(f"数据目录: {self.data_dir}")
        print(f"设备: {self.device}")
        
        # 加载元数据
        self.metadata = self._load_metadata()
        self.task_configs = self._load_task_configs()
        
        # 初始化数据和模型
        self._setup_data()
        self._setup_model()
        
        # 存储分析结果
        self.error_analysis = {}
        self.sample_analysis = []
        
    def _load_metadata(self) -> Dict:
        """加载模型元数据"""
        metadata_file = self.model_dir / "m16_multitask_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✓ 加载模型元数据")
            return metadata
        else:
            print("❌ 未找到模型元数据文件")
            return {}
    
    def _load_task_configs(self) -> Dict:
        """加载任务配置"""
        dataset_info_file = self.data_dir / "dataset_info.json"
        if dataset_info_file.exists():
            with open(dataset_info_file, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
            print(f"✓ 加载任务配置")
            return dataset_info.get('tasks', {})
        else:
            print("❌ 未找到数据集信息文件")
            return {}
    
    def _setup_data(self):
        """设置数据加载器"""
        print("📊 设置数据加载器...")
        
        # 数据预处理 - 兼容numpy数组输入
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
            transforms.Resize((70, 70)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 测试数据集
        self.test_dataset = NIMultitaskDataset(
            data_root=str(self.data_dir),
            split='test',
            transform=self.transform,
            target_size=(70, 70),
            grayscale=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,  # 单样本分析
            shuffle=False,
            num_workers=0  # 避免多进程问题
        )
        
        print(f"✓ 测试集样本数: {len(self.test_dataset)}")
        
    def _setup_model(self):
        """设置模型（ONNX或模拟）"""
        onnx_model_path = "onnx_models/m16_multitask_mobilenetv3.onnx"
        
        if ONNX_AVAILABLE and os.path.exists(onnx_model_path):
            print(f"🔧 使用ONNX模型进行推理")
            self.session = ort.InferenceSession(onnx_model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.use_onnx = True
        else:
            print(f"🎭 使用模拟推理（用于演示）")
            self.use_onnx = False
            # 创建模拟的错误样本数据
            self._create_mock_predictions()
    
    def _create_mock_predictions(self):
        """创建模拟预测结果用于演示"""
        print("🎭 创建模拟预测结果...")
        np.random.seed(42)  # 固定随机种子
        
        # 为每个测试样本创建模拟预测
        self.mock_predictions = {}
        
        for idx in range(len(self.test_dataset)):
            _, targets = self.test_dataset[idx]
            
            # 模拟各任务的预测结果（添加一些随机错误）
            predictions = {}
            
            # growth_level (3类)
            true_growth = targets['growth_level'].item()
            if np.random.random() > 0.90:  # 10%错误率
                pred_growth = (true_growth + np.random.randint(1, 3)) % 3
            else:
                pred_growth = true_growth
            predictions['growth_level'] = pred_growth
            
            # growth_pattern (9类)
            true_pattern = targets['growth_pattern'].item()
            if np.random.random() > 0.85:  # 15%错误率
                pred_pattern = np.random.randint(0, 9)
            else:
                pred_pattern = true_pattern
            predictions['growth_pattern'] = pred_pattern
            
            # interference_factors (多标签)
            true_interference = targets['interference_factors'].numpy()
            pred_interference = true_interference.copy()
            if np.random.random() > 0.80:  # 20%错误率
                flip_idx = np.random.randint(0, len(true_interference))
                pred_interference[flip_idx] = 1 - pred_interference[flip_idx]
            predictions['interference_factors'] = pred_interference
            
            # fine_grained (8类)
            true_fine = targets['fine_grained'].item()
            if np.random.random() > 0.75:  # 25%错误率
                pred_fine = np.random.randint(0, 8)
            else:
                pred_fine = true_fine
            predictions['fine_grained'] = pred_fine
            
            self.mock_predictions[idx] = predictions
    
    def analyze_errors(self) -> Dict:
        """执行错误分析"""
        print("🔍 开始错误样本分析...")
        
        # 初始化错误统计
        error_stats = {
            'growth_level': {'total': 0, 'errors': 0, 'error_samples': []},
            'growth_pattern': {'total': 0, 'errors': 0, 'error_samples': []},
            'interference_factors': {'total': 0, 'errors': 0, 'error_samples': []},
            'fine_grained': {'total': 0, 'errors': 0, 'error_samples': []}
        }
        
        error_patterns = defaultdict(lambda: defaultdict(int))
        sample_details = []
        
        for idx, (image, targets) in enumerate(self.test_loader):
            if idx >= len(self.test_dataset):
                break
                
            # 获取预测结果
            if self.use_onnx:
                predictions = self._predict_onnx(image)
            else:
                predictions = self.mock_predictions[idx]
            
            # 分析每个任务
            sample_detail = {
                'sample_idx': idx,
                'image_path': self.test_dataset.annotations[idx]['local_image_path'],
                'tasks': {}
            }
            
            for task_name in self.task_configs.keys():
                if task_name not in targets:
                    continue
                    
                true_label = targets[task_name]
                pred_label = predictions.get(task_name, None)
                
                if pred_label is None:
                    continue
                
                # 处理不同类型的任务
                if task_name == 'interference_factors':
                    # 多标签任务
                    true_multilabel = true_label.numpy() if hasattr(true_label, 'numpy') else true_label
                    pred_multilabel = pred_label if isinstance(pred_label, np.ndarray) else np.array(pred_label)
                    
                    is_error = not np.array_equal(true_multilabel, pred_multilabel)
                    error_type = self._analyze_multilabel_error(true_multilabel, pred_multilabel, task_name)
                else:
                    # 单标签任务
                    true_single = true_label.item() if hasattr(true_label, 'item') else true_label
                    pred_single = pred_label if isinstance(pred_label, (int, np.integer)) else int(pred_label)
                    
                    is_error = (true_single != pred_single)
                    error_type = self._analyze_classification_error(true_single, pred_single, task_name)
                
                # 更新统计
                error_stats[task_name]['total'] += 1
                if is_error:
                    error_stats[task_name]['errors'] += 1
                    error_stats[task_name]['error_samples'].append(idx)
                    error_patterns[task_name][error_type] += 1
                
                # 记录详细信息
                sample_detail['tasks'][task_name] = {
                    'true_label': true_label.tolist() if hasattr(true_label, 'tolist') else true_label,
                    'pred_label': pred_label.tolist() if hasattr(pred_label, 'tolist') else pred_label,
                    'is_error': is_error,
                    'error_type': error_type if is_error else None
                }
            
            sample_details.append(sample_detail)
            
            if (idx + 1) % 50 == 0:
                print(f"   已处理 {idx + 1}/{len(self.test_dataset)} 样本")
        
        # 计算错误率
        for task_name in error_stats:
            if error_stats[task_name]['total'] > 0:
                error_rate = error_stats[task_name]['errors'] / error_stats[task_name]['total']
                error_stats[task_name]['error_rate'] = error_rate
        
        self.error_analysis = {
            'error_stats': error_stats,
            'error_patterns': dict(error_patterns),
            'sample_details': sample_details
        }
        
        print(f"✅ 错误分析完成！")
        return self.error_analysis
    
    def _predict_onnx(self, image: torch.Tensor) -> Dict:
        """ONNX模型推理"""
        input_data = image.numpy()
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # 处理输出
        predictions = {}
        
        # 根据模型输出格式处理
        if len(outputs) >= 4:
            predictions['growth_level'] = np.argmax(outputs[0][0])
            predictions['growth_pattern'] = np.argmax(outputs[1][0])
            predictions['interference_factors'] = (outputs[2][0] > 0.5).astype(int)
            predictions['fine_grained'] = np.argmax(outputs[3][0])
        
        return predictions
    
    def _analyze_classification_error(self, true_label: int, pred_label: int, task_name: str) -> str:
        """分析分类错误类型"""
        # 获取类别名称
        label_mappings = self.test_dataset.dataset_info.get('label_mappings', {}).get(task_name, {})
        class_names = {v: k for k, v in label_mappings.items()}
        
        true_name = class_names.get(true_label, f"class_{true_label}")
        pred_name = class_names.get(pred_label, f"class_{pred_label}")
        
        return f"{true_name}_to_{pred_name}"
    
    def _analyze_multilabel_error(self, true_labels: np.ndarray, pred_labels: np.ndarray, task_name: str) -> str:
        """分析多标签错误类型"""
        # 计算差异
        false_positives = np.sum((pred_labels == 1) & (true_labels == 0))
        false_negatives = np.sum((pred_labels == 0) & (true_labels == 1))
        
        if false_positives > false_negatives:
            return f"over_prediction_fp{false_positives}"
        elif false_negatives > false_positives:
            return f"under_prediction_fn{false_negatives}"
        else:
            return f"mixed_error_fp{false_positives}_fn{false_negatives}"
    
    def generate_error_report(self, output_dir: str = None) -> str:
        """生成错误分析报告"""
        if not self.error_analysis:
            self.analyze_errors()
        
        output_dir = Path(output_dir or self.model_dir / "error_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 生成错误分析报告...")
        
        # 1. 文本报告
        report_file = output_dir / "error_analysis_report.md"
        self._generate_text_report(report_file)
        
        # 2. JSON详细数据
        json_file = output_dir / "error_analysis_data.json"
        self._save_json_report(json_file)
        
        # 3. 可视化图表
        self._generate_visualizations(output_dir)
        
        # 4. 错误样本详情
        self._generate_error_samples_report(output_dir)
        
        print(f"✅ 报告已生成到: {output_dir}")
        return str(output_dir)
    
    def _generate_text_report(self, report_file: Path):
        """生成文本报告"""
        error_stats = self.error_analysis['error_stats']
        error_patterns = self.error_analysis['error_patterns']
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# M16 MultiTask MobileNetV3 错误样本分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试样本总数: {len(self.test_dataset)}\n\n")
            
            f.write("## 📊 各任务错误率统计\n\n")
            f.write("| 任务名称 | 总样本数 | 错误数 | 错误率 | 准确率 |\n")
            f.write("|---------|----------|--------|--------|--------|\n")
            
            for task_name, stats in error_stats.items():
                if stats['total'] > 0:
                    error_rate = stats.get('error_rate', 0)
                    accuracy = 1 - error_rate
                    f.write(f"| {task_name} | {stats['total']} | {stats['errors']} | {error_rate:.3f} | {accuracy:.3f} |\n")
            
            f.write("\n## 🔍 错误模式分析\n\n")
            for task_name, patterns in error_patterns.items():
                if patterns:
                    f.write(f"### {task_name}\n")
                    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
                    for pattern, count in sorted_patterns[:10]:  # 显示前10种错误模式
                        f.write(f"- {pattern}: {count} 次\n")
                    f.write("\n")
            
            # 最困难的样本
            f.write("## 🎯 最困难样本分析\n\n")
            difficult_samples = self._find_most_difficult_samples()
            f.write(f"发现 {len(difficult_samples)} 个在多个任务上都出错的样本:\n\n")
            
            for sample in difficult_samples[:20]:  # 显示前20个最困难的样本
                f.write(f"- 样本 {sample['sample_idx']} ({sample['image_path']}): ")
                f.write(f"{sample['error_count']} 个任务出错\n")
    
    def _find_most_difficult_samples(self) -> List[Dict]:
        """找出最困难的样本（在多个任务上都出错）"""
        sample_error_counts = {}
        
        for sample in self.error_analysis['sample_details']:
            error_count = sum(1 for task_info in sample['tasks'].values() if task_info['is_error'])
            if error_count > 0:
                sample_error_counts[sample['sample_idx']] = {
                    'sample_idx': sample['sample_idx'],
                    'image_path': sample['image_path'],
                    'error_count': error_count,
                    'total_tasks': len(sample['tasks'])
                }
        
        # 按错误任务数排序
        return sorted(sample_error_counts.values(), key=lambda x: x['error_count'], reverse=True)
    
    def _save_json_report(self, json_file: Path):
        """保存JSON格式的详细数据"""
        # 处理不可序列化的数据
        serializable_data = {}
        
        for key, value in self.error_analysis.items():
            if key == 'sample_details':
                # 处理sample_details中的numpy数组
                serializable_samples = []
                for sample in value:
                    serializable_sample = {
                        'sample_idx': sample['sample_idx'],
                        'image_path': sample['image_path'],
                        'tasks': {}
                    }
                    for task_name, task_info in sample['tasks'].items():
                        serializable_sample['tasks'][task_name] = {
                            'true_label': task_info['true_label'],
                            'pred_label': task_info['pred_label'],
                            'is_error': task_info['is_error'],
                            'error_type': task_info['error_type']
                        }
                    serializable_samples.append(serializable_sample)
                serializable_data[key] = serializable_samples
            else:
                serializable_data[key] = value
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    def _generate_visualizations(self, output_dir: Path):
        """生成可视化图表"""
        error_stats = self.error_analysis['error_stats']
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 各任务错误率对比
        plt.figure(figsize=(12, 6))
        tasks = []
        error_rates = []
        accuracies = []
        
        for task_name, stats in error_stats.items():
            if stats['total'] > 0:
                tasks.append(task_name.replace('_', '\n'))
                error_rate = stats.get('error_rate', 0)
                error_rates.append(error_rate)
                accuracies.append(1 - error_rate)
        
        x = np.arange(len(tasks))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='准确率', color='#2E86AB', alpha=0.8)
        plt.bar(x + width/2, error_rates, width, label='错误率', color='#E74C3C', alpha=0.8)
        
        plt.title('M16 MultiTask MobileNetV3 - 各任务性能对比', fontsize=14, fontweight='bold')
        plt.xlabel('任务')
        plt.ylabel('比率')
        plt.xticks(x, tasks)
        plt.legend()
        plt.ylim(0, 1.05)
        
        # 添加数值标签
        for i, (acc, err) in enumerate(zip(accuracies, error_rates)):
            plt.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, err + 0.01, f'{err:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'task_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 错误模式分析热力图
        self._create_error_pattern_heatmap(output_dir)
        
        print("📊 可视化图表已生成")
    
    def _create_error_pattern_heatmap(self, output_dir: Path):
        """创建错误模式热力图"""
        error_patterns = self.error_analysis['error_patterns']
        
        # 为每个有错误的任务创建热力图
        for task_name, patterns in error_patterns.items():
            if not patterns:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # 准备数据
            pattern_names = list(patterns.keys())
            pattern_counts = list(patterns.values())
            
            # 创建简化的可视化
            colors = plt.cm.Reds(np.linspace(0.2, 1, len(pattern_names)))
            bars = plt.barh(range(len(pattern_names)), pattern_counts, color=colors)
            
            plt.yticks(range(len(pattern_names)), pattern_names)
            plt.xlabel('错误次数')
            plt.title(f'{task_name} - 错误模式分布')
            
            # 添加数值标签
            for i, (bar, count) in enumerate(zip(bars, pattern_counts)):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        str(count), ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{task_name}_error_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_error_samples_report(self, output_dir: Path):
        """生成错误样本详情报告"""
        difficult_samples = self._find_most_difficult_samples()
        
        # 创建错误样本CSV文件
        csv_data = []
        for sample in self.error_analysis['sample_details']:
            error_tasks = [task_name for task_name, task_info in sample['tasks'].items() 
                         if task_info['is_error']]
            if error_tasks:
                csv_data.append({
                    'sample_idx': sample['sample_idx'],
                    'image_path': sample['image_path'],
                    'error_count': len(error_tasks),
                    'error_tasks': ','.join(error_tasks),
                    'total_tasks': len(sample['tasks'])
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_dir / 'error_samples_summary.csv', index=False, encoding='utf-8')
            print(f"📄 错误样本CSV文件已生成 ({len(csv_data)} 个错误样本)")

def main():
    """主函数"""
    print("🔍 M16 MultiTask MobileNetV3 错误样本分析")
    print("="*60)
    
    try:
        # 创建分析器
        analyzer = M16ErrorAnalyzer()
        
        # 执行分析
        error_analysis = analyzer.analyze_errors()
        
        # 生成报告
        report_dir = analyzer.generate_error_report()
        
        print(f"\n📊 分析摘要:")
        error_stats = error_analysis['error_stats']
        for task_name, stats in error_stats.items():
            if stats['total'] > 0:
                error_rate = stats.get('error_rate', 0)
                print(f"   {task_name}: {stats['errors']}/{stats['total']} 错误 (错误率: {error_rate:.3f})")
        
        print(f"\n✅ 错误分析完成！")
        print(f"📁 详细报告已保存到: {report_dir}")
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()