#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比分析脚本

这个脚本用于对比分析已训练的模型，生成对比报告和可视化图表。

使用方法:
    python compare_models.py                           # 对比所有已训练模型
    python compare_models.py --models model1 model2   # 对比指定模型
    python compare_models.py --top 5                  # 对比性能最好的5个模型
    python compare_models.py --generate-report        # 生成详细报告
"""

import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ModelComparator:
    """
    模型对比分析器
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.checkpoints_dir = Path("checkpoints")
        self.models_data = []
        
        print(f"🔍 模型对比分析器初始化")
        print(f"📁 检查点目录: {self.checkpoints_dir}")
        print(f"📊 输出目录: {self.output_dir}")
    
    def scan_trained_models(self) -> List[Dict[str, Any]]:
        """
        扫描已训练的模型
        
        Returns:
            List[Dict]: 模型信息列表
        """
        models = []
        
        if not self.checkpoints_dir.exists():
            print(f"❌ 检查点目录不存在: {self.checkpoints_dir}")
            return models
        
        print(f"\n🔍 扫描已训练模型...")
        
        for model_dir in self.checkpoints_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            best_checkpoint = model_dir / "best.pth"
            history_file = model_dir / "training_history.json"
            
            if not best_checkpoint.exists():
                print(f"⚠️ {model_name}: 缺少最佳检查点")
                continue
            
            model_info = {
                'name': model_name,
                'checkpoint_path': str(best_checkpoint),
                'model_dir': str(model_dir)
            }
            
            # 读取训练历史
            if history_file.exists():
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                    
                    model_info.update({
                        'best_val_acc': history.get('best_val_acc', 0),
                        'best_epoch': history.get('best_epoch', 0),
                        'train_losses': history.get('train_losses', []),
                        'train_accuracies': history.get('train_accuracies', []),
                        'val_losses': history.get('val_losses', []),
                        'val_accuracies': history.get('val_accuracies', []),
                        'total_epochs': len(history.get('train_losses', []))
                    })
                    
                    print(f"✅ {model_name}: 最佳验证准确率 {history.get('best_val_acc', 0):.2f}%")
                    
                except Exception as e:
                    print(f"⚠️ {model_name}: 读取训练历史失败 - {e}")
                    model_info.update({
                        'best_val_acc': 0,
                        'best_epoch': 0,
                        'total_epochs': 0
                    })
            else:
                print(f"⚠️ {model_name}: 缺少训练历史文件")
                model_info.update({
                    'best_val_acc': 0,
                    'best_epoch': 0,
                    'total_epochs': 0
                })
            
            models.append(model_info)
        
        print(f"\n📊 找到 {len(models)} 个已训练模型")
        return models
    
    def filter_models(self, models: List[Dict], model_names: Optional[List[str]] = None, 
                     top_k: Optional[int] = None) -> List[Dict]:
        """
        过滤模型
        
        Args:
            models: 模型列表
            model_names: 指定的模型名称列表
            top_k: 选择性能最好的k个模型
        
        Returns:
            List[Dict]: 过滤后的模型列表
        """
        if model_names:
            # 按指定名称过滤
            filtered = [m for m in models if m['name'] in model_names]
            missing = set(model_names) - {m['name'] for m in filtered}
            if missing:
                print(f"⚠️ 未找到模型: {', '.join(missing)}")
            return filtered
        
        if top_k:
            # 按性能排序并选择前k个
            sorted_models = sorted(models, key=lambda x: x.get('best_val_acc', 0), reverse=True)
            return sorted_models[:top_k]
        
        return models
    
    def create_comparison_table(self, models: List[Dict]) -> pd.DataFrame:
        """
        创建对比表格
        
        Args:
            models: 模型列表
        
        Returns:
            pd.DataFrame: 对比表格
        """
        data = []
        
        for model in models:
            row = {
                '模型名称': model['name'],
                '最佳验证准确率(%)': f"{model.get('best_val_acc', 0):.2f}",
                '最佳轮次': model.get('best_epoch', 0),
                '总训练轮次': model.get('total_epochs', 0),
                '最终训练损失': f"{model.get('train_losses', [0])[-1]:.4f}" if model.get('train_losses') else 'N/A',
                '最终验证损失': f"{model.get('val_losses', [0])[-1]:.4f}" if model.get('val_losses') else 'N/A'
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def plot_training_curves(self, models: List[Dict], save_path: Optional[str] = None):
        """
        绘制训练曲线对比图
        
        Args:
            models: 模型列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型训练曲线对比', fontsize=16, fontweight='bold')
        
        # 颜色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_name = model['name']
            color = colors[i]
            
            train_losses = model.get('train_losses', [])
            val_losses = model.get('val_losses', [])
            train_accs = model.get('train_accuracies', [])
            val_accs = model.get('val_accuracies', [])
            
            if not train_losses:
                continue
            
            epochs = range(1, len(train_losses) + 1)
            
            # 训练损失
            axes[0, 0].plot(epochs, train_losses, label=model_name, color=color, linewidth=2)
            
            # 验证损失
            if val_losses:
                axes[0, 1].plot(epochs, val_losses, label=model_name, color=color, linewidth=2)
            
            # 训练准确率
            if train_accs:
                axes[1, 0].plot(epochs, train_accs, label=model_name, color=color, linewidth=2)
            
            # 验证准确率
            if val_accs:
                axes[1, 1].plot(epochs, val_accs, label=model_name, color=color, linewidth=2)
        
        # 设置子图
        axes[0, 0].set_title('训练损失')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('验证损失')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('训练准确率')
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('准确率 (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('验证准确率')
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('准确率 (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线图已保存: {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, models: List[Dict], save_path: Optional[str] = None):
        """
        绘制性能对比图
        
        Args:
            models: 模型列表
            save_path: 保存路径
        """
        if not models:
            print("❌ 没有模型数据可供对比")
            return
        
        # 准备数据
        model_names = [model['name'] for model in models]
        accuracies = [model.get('best_val_acc', 0) for model in models]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('模型性能对比', fontsize=16, fontweight='bold')
        
        # 柱状图
        bars = ax1.bar(model_names, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        ax1.set_title('最佳验证准确率对比')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_ylim(0, 100)
        
        # 在柱子上添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 旋转x轴标签
        ax1.tick_params(axis='x', rotation=45)
        
        # 饼图 - 显示相对性能
        if len(models) > 1:
            # 计算相对性能（归一化）
            total_acc = sum(accuracies)
            if total_acc > 0:
                relative_performance = [acc/total_acc * 100 for acc in accuracies]
                
                wedges, texts, autotexts = ax2.pie(relative_performance, labels=model_names, 
                                                   autopct='%1.1f%%', startangle=90)
                ax2.set_title('相对性能分布')
                
                # 美化饼图
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, '需要至少2个模型\n才能显示相对性能', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('相对性能分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 性能对比图已保存: {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self, models: List[Dict], save_path: Optional[str] = None) -> str:
        """
        生成详细的对比报告
        
        Args:
            models: 模型列表
            save_path: 保存路径
        
        Returns:
            str: 报告内容
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
# BioAst模型对比分析报告

**生成时间**: {timestamp}
**对比模型数量**: {len(models)}

## 📊 模型性能概览

"""
        
        if not models:
            report += "❌ 没有找到已训练的模型。\n"
            return report
        
        # 排序模型（按性能）
        sorted_models = sorted(models, key=lambda x: x.get('best_val_acc', 0), reverse=True)
        
        # 性能排行榜
        report += "### 🏆 性能排行榜\n\n"
        report += "| 排名 | 模型名称 | 最佳验证准确率 | 最佳轮次 | 总训练轮次 |\n"
        report += "|------|----------|----------------|----------|------------|\n"
        
        for i, model in enumerate(sorted_models, 1):
            name = model['name']
            acc = model.get('best_val_acc', 0)
            best_epoch = model.get('best_epoch', 0)
            total_epochs = model.get('total_epochs', 0)
            
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}"
            report += f"| {medal} | {name} | {acc:.2f}% | {best_epoch} | {total_epochs} |\n"
        
        # 统计信息
        accuracies = [m.get('best_val_acc', 0) for m in models]
        if accuracies:
            report += f"\n### 📈 统计信息\n\n"
            report += f"- **最高准确率**: {max(accuracies):.2f}%\n"
            report += f"- **最低准确率**: {min(accuracies):.2f}%\n"
            report += f"- **平均准确率**: {np.mean(accuracies):.2f}%\n"
            report += f"- **准确率标准差**: {np.std(accuracies):.2f}%\n"
        
        # 详细分析
        report += "\n### 🔍 详细分析\n\n"
        
        for model in sorted_models:
            name = model['name']
            acc = model.get('best_val_acc', 0)
            best_epoch = model.get('best_epoch', 0)
            total_epochs = model.get('total_epochs', 0)
            
            train_losses = model.get('train_losses', [])
            val_losses = model.get('val_losses', [])
            
            report += f"#### {name}\n\n"
            report += f"- **最佳验证准确率**: {acc:.2f}%\n"
            report += f"- **达到最佳性能的轮次**: {best_epoch}\n"
            report += f"- **总训练轮次**: {total_epochs}\n"
            
            if train_losses:
                final_train_loss = train_losses[-1]
                report += f"- **最终训练损失**: {final_train_loss:.4f}\n"
            
            if val_losses:
                final_val_loss = val_losses[-1]
                report += f"- **最终验证损失**: {final_val_loss:.4f}\n"
            
            # 训练效率分析
            if best_epoch > 0 and total_epochs > 0:
                efficiency = (best_epoch / total_epochs) * 100
                report += f"- **训练效率**: {efficiency:.1f}% (在{efficiency:.1f}%的训练时间内达到最佳性能)\n"
            
            report += "\n"
        
        # 建议
        report += "### 💡 建议\n\n"
        
        if len(models) > 1:
            best_model = sorted_models[0]
            report += f"1. **推荐模型**: {best_model['name']} (准确率: {best_model.get('best_val_acc', 0):.2f}%)\n"
            
            # 分析训练效率
            efficient_models = [m for m in models if m.get('best_epoch', 0) > 0 and m.get('total_epochs', 0) > 0]
            if efficient_models:
                efficiency_scores = [(m['best_epoch'] / m['total_epochs']) for m in efficient_models]
                most_efficient_idx = np.argmin(efficiency_scores)
                most_efficient = efficient_models[most_efficient_idx]
                
                report += f"2. **训练效率最高**: {most_efficient['name']} (在第{most_efficient['best_epoch']}轮达到最佳性能)\n"
            
            # 性能差异分析
            acc_range = max(accuracies) - min(accuracies)
            if acc_range < 5:
                report += "3. **性能差异**: 各模型性能相近，可以考虑选择训练效率更高或参数量更少的模型\n"
            elif acc_range > 20:
                report += "3. **性能差异**: 各模型性能差异较大，建议选择性能最好的模型\n"
            else:
                report += "3. **性能差异**: 各模型性能有一定差异，建议综合考虑性能和效率\n"
        
        report += "\n### 📝 备注\n\n"
        report += "- 本报告基于训练过程中的验证集性能生成\n"
        report += "- 实际部署时建议在独立测试集上进一步验证\n"
        report += "- 可以考虑模型集成来进一步提升性能\n"
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 详细报告已保存: {save_path}")
        
        return report
    
    def compare_models(self, model_names: Optional[List[str]] = None, 
                      top_k: Optional[int] = None, 
                      generate_report: bool = False,
                      show_plots: bool = True) -> Dict[str, Any]:
        """
        执行模型对比分析
        
        Args:
            model_names: 指定要对比的模型名称
            top_k: 对比性能最好的k个模型
            generate_report: 是否生成详细报告
            show_plots: 是否显示图表
        
        Returns:
            Dict: 对比结果
        """
        print(f"\n🚀 开始模型对比分析")
        
        # 扫描模型
        all_models = self.scan_trained_models()
        
        if not all_models:
            print("❌ 没有找到已训练的模型")
            return {'models': [], 'comparison_table': None}
        
        # 过滤模型
        models_to_compare = self.filter_models(all_models, model_names, top_k)
        
        if not models_to_compare:
            print("❌ 没有符合条件的模型")
            return {'models': [], 'comparison_table': None}
        
        print(f"\n📊 对比 {len(models_to_compare)} 个模型:")
        for model in models_to_compare:
            print(f"  - {model['name']}: {model.get('best_val_acc', 0):.2f}%")
        
        # 创建对比表格
        comparison_table = self.create_comparison_table(models_to_compare)
        print(f"\n📋 模型对比表格:")
        print(comparison_table.to_string(index=False))
        
        # 保存表格
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        table_path = self.output_dir / f"model_comparison_table_{timestamp}.csv"
        comparison_table.to_csv(table_path, index=False, encoding='utf-8-sig')
        print(f"📄 对比表格已保存: {table_path}")
        
        results = {
            'models': models_to_compare,
            'comparison_table': comparison_table,
            'table_path': str(table_path)
        }
        
        if show_plots:
            # 绘制训练曲线
            curves_path = self.output_dir / f"training_curves_{timestamp}.png"
            self.plot_training_curves(models_to_compare, str(curves_path))
            results['curves_path'] = str(curves_path)
            
            # 绘制性能对比
            performance_path = self.output_dir / f"performance_comparison_{timestamp}.png"
            self.plot_performance_comparison(models_to_compare, str(performance_path))
            results['performance_path'] = str(performance_path)
        
        if generate_report:
            # 生成详细报告
            report_path = self.output_dir / f"detailed_comparison_report_{timestamp}.md"
            report_content = self.generate_detailed_report(models_to_compare, str(report_path))
            results['report_path'] = str(report_path)
            results['report_content'] = report_content
        
        print(f"\n✅ 模型对比分析完成")
        print(f"📁 输出目录: {self.output_dir}")
        
        return results

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='模型对比分析工具')
    parser.add_argument('--models', nargs='+', help='指定要对比的模型名称')
    parser.add_argument('--top', type=int, help='对比性能最好的k个模型')
    parser.add_argument('--generate-report', action='store_true', help='生成详细报告')
    parser.add_argument('--no-plots', action='store_true', help='不显示图表')
    parser.add_argument('--output-dir', default='reports', help='输出目录')
    
    args = parser.parse_args()
    
    print("🧬 BioAst模型对比分析工具")
    print("=" * 50)
    
    # 创建对比器
    comparator = ModelComparator(output_dir=args.output_dir)
    
    # 执行对比
    results = comparator.compare_models(
        model_names=args.models,
        top_k=args.top,
        generate_report=args.generate_report,
        show_plots=not args.no_plots
    )
    
    if results['models']:
        print(f"\n🎯 对比结果摘要:")
        print(f"  对比模型数: {len(results['models'])}")
        print(f"  输出文件:")
        
        for key, path in results.items():
            if key.endswith('_path') and path:
                print(f"    - {key}: {path}")
    else:
        print("\n❌ 没有可对比的模型")
        print("\n💡 提示:")
        print("  1. 确保已经训练了一些模型")
        print("  2. 检查 checkpoints/ 目录是否存在")
        print("  3. 使用 python train_single_model.py --list_models 查看可用模型")

if __name__ == "__main__":
    main()