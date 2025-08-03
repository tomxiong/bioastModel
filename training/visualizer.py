"""
训练可视化模块
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
from PIL import Image
import cv2

class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: str = './visualizations'):
        """
        Args:
            save_dir: 可视化结果保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体，与evaluator.py保持一致
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        # 设置默认字体大小
        plt.rcParams['font.size'] = 10
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            model_name: str = 'Model'):
        """绘制训练历史曲线"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} 训练历史', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失', linewidth=2)
        ax1.set_title('损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
        ax2.set_title('准确率曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 学习率曲线
        ax3 = axes[1, 0]
        ax3.plot(epochs, history['lr'], 'g-', linewidth=2)
        ax3.set_title('学习率变化')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. 训练vs验证性能差异
        ax4 = axes[1, 1]
        train_val_loss_diff = [t - v for t, v in zip(history['train_loss'], history['val_loss'])]
        train_val_acc_diff = [v - t for t, v in zip(history['train_acc'], history['val_acc'])]
        
        ax4.plot(epochs, train_val_loss_diff, 'purple', label='损失差异 (训练-验证)', linewidth=2)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(epochs, train_val_acc_diff, 'orange', label='准确率差异 (验证-训练)', linewidth=2)
        
        ax4.set_title('过拟合监控')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('损失差异', color='purple')
        ax4_twin.set_ylabel('准确率差异', color='orange')
        ax4.grid(True, alpha=0.3)
        
        # 添加图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{model_name}_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练历史图表已保存: {save_path}")
    
    def visualize_feature_maps(self, model: nn.Module, sample_images: torch.Tensor, 
                             labels: torch.Tensor, device: torch.device,
                             model_name: str = 'Model'):
        """可视化特征图"""
        
        model.eval()
        sample_images = sample_images.to(device)
        
        with torch.no_grad():
            # 获取特征图
            if hasattr(model, 'get_feature_maps'):
                features = model.get_feature_maps(sample_images)
            else:
                print("模型不支持特征图提取")
                return
        
        # 选择几个代表性的特征图层
        selected_features = features[::2]  # 每隔一层选择
        if len(selected_features) > 6:
            selected_features = selected_features[:6]
        
        num_samples = min(4, sample_images.size(0))
        num_layers = len(selected_features)
        
        fig, axes = plt.subplots(num_samples, num_layers + 1, 
                               figsize=(3 * (num_layers + 1), 3 * num_samples))
        fig.suptitle(f'{model_name} 特征图可视化', fontsize=16, fontweight='bold')
        
        for sample_idx in range(num_samples):
            # 原始图像
            ax = axes[sample_idx, 0] if num_samples > 1 else axes[0]
            original_img = sample_images[sample_idx].cpu()
            # 反归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            original_img = original_img * std + mean
            original_img = torch.clamp(original_img, 0, 1)
            
            ax.imshow(original_img.permute(1, 2, 0))
            ax.set_title(f'原图 (标签: {labels[sample_idx].item()})')
            ax.axis('off')
            
            # 特征图
            for layer_idx, (layer_name, feature_map) in enumerate(selected_features):
                ax = axes[sample_idx, layer_idx + 1] if num_samples > 1 else axes[layer_idx + 1]
                
                # 取第一个通道或平均所有通道
                feat = feature_map[sample_idx]
                if feat.dim() == 3:  # [C, H, W]
                    feat = feat.mean(dim=0)  # 平均所有通道
                
                feat = feat.cpu().numpy()
                
                im = ax.imshow(feat, cmap='viridis')
                ax.set_title(f'{layer_name}\n{feat.shape}')
                ax.axis('off')
                
                # 添加颜色条
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{model_name}_feature_maps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"特征图可视化已保存: {save_path}")
    
    def visualize_grad_cam(self, model: nn.Module, sample_images: torch.Tensor,
                          labels: torch.Tensor, device: torch.device,
                          target_layer: str = 'layer4', model_name: str = 'Model'):
        """Grad-CAM可视化"""
        
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
        except ImportError:
            print("需要安装 pytorch-grad-cam: pip install grad-cam")
            return
        
        model.eval()
        sample_images = sample_images.to(device)
        
        # 找到目标层
        target_layers = []
        for name, module in model.named_modules():
            if target_layer in name:
                target_layers.append(module)
                break
        
        if not target_layers:
            print(f"未找到目标层: {target_layer}")
            return
        
        # 创建GradCAM
        cam = GradCAM(model=model, target_layers=target_layers)
        
        num_samples = min(4, sample_images.size(0))
        fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
        fig.suptitle(f'{model_name} Grad-CAM可视化', fontsize=16, fontweight='bold')
        
        for i in range(num_samples):
            # 原始图像
            input_tensor = sample_images[i:i+1]
            
            # 反归一化用于显示
            original_img = sample_images[i].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            original_img = original_img * std + mean
            original_img = torch.clamp(original_img, 0, 1)
            original_img_np = original_img.permute(1, 2, 0).numpy()
            
            # 生成CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # 叠加CAM
            visualization = show_cam_on_image(original_img_np, grayscale_cam, use_rgb=True)
            
            # 显示原图
            ax1 = axes[0, i] if num_samples > 1 else axes[0]
            ax1.imshow(original_img_np)
            ax1.set_title(f'原图 {i+1}\n标签: {labels[i].item()}')
            ax1.axis('off')
            
            # 显示CAM
            ax2 = axes[1, i] if num_samples > 1 else axes[1]
            ax2.imshow(visualization)
            ax2.set_title(f'Grad-CAM {i+1}')
            ax2.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{model_name}_grad_cam.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Grad-CAM可视化已保存: {save_path}")
    
    def plot_model_comparison(self, models_history: Dict[str, Dict[str, List[float]]]):
        """比较多个模型的训练历史"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('多模型训练对比', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_history)))
        
        # 1. 训练损失对比
        ax1 = axes[0, 0]
        for (model_name, history), color in zip(models_history.items(), colors):
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], color=color, 
                    label=f'{model_name} 训练', linewidth=2, alpha=0.8)
            ax1.plot(epochs, history['val_loss'], color=color, 
                    label=f'{model_name} 验证', linewidth=2, linestyle='--', alpha=0.8)
        
        ax1.set_title('训练损失对比')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 验证准确率对比
        ax2 = axes[0, 1]
        for (model_name, history), color in zip(models_history.items(), colors):
            epochs = range(1, len(history['val_acc']) + 1)
            ax2.plot(epochs, history['val_acc'], color=color, 
                    label=model_name, linewidth=2, alpha=0.8)
        
        ax2.set_title('验证准确率对比')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 最终性能对比
        ax3 = axes[1, 0]
        model_names = list(models_history.keys())
        final_train_acc = [history['train_acc'][-1] for history in models_history.values()]
        final_val_acc = [history['val_acc'][-1] for history in models_history.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, final_train_acc, width, label='训练准确率', alpha=0.8)
        ax3.bar(x + width/2, final_val_acc, width, label='验证准确率', alpha=0.8)
        
        ax3.set_title('最终性能对比')
        ax3.set_xlabel('模型')
        ax3.set_ylabel('准确率')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.set_ylim([0, 1])
        
        # 4. 收敛速度对比
        ax4 = axes[1, 1]
        for (model_name, history), color in zip(models_history.items(), colors):
            # 计算达到90%最佳验证准确率所需的epoch数
            best_val_acc = max(history['val_acc'])
            target_acc = 0.9 * best_val_acc
            
            convergence_epoch = len(history['val_acc'])
            for i, acc in enumerate(history['val_acc']):
                if acc >= target_acc:
                    convergence_epoch = i + 1
                    break
            
            ax4.bar(model_name, convergence_epoch, color=color, alpha=0.8)
        
        ax4.set_title('收敛速度对比 (达到90%最佳性能)')
        ax4.set_xlabel('模型')
        ax4.set_ylabel('所需Epoch数')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'models_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"模型对比图表已保存: {save_path}")
    
    def visualize_predictions(self, model: nn.Module, data_loader, device: torch.device,
                            num_samples: int = 16, model_name: str = 'Model'):
        """可视化模型预测结果"""
        
        model.eval()
        
        # 收集样本
        images_list = []
        labels_list = []
        predictions_list = []
        probabilities_list = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                images_list.append(images.cpu())
                labels_list.append(labels.cpu())
                predictions_list.append(predictions.cpu())
                probabilities_list.append(probabilities.cpu())
                
                if len(images_list) * images.size(0) >= num_samples:
                    break
        
        # 合并数据
        all_images = torch.cat(images_list, dim=0)[:num_samples]
        all_labels = torch.cat(labels_list, dim=0)[:num_samples]
        all_predictions = torch.cat(predictions_list, dim=0)[:num_samples]
        all_probabilities = torch.cat(probabilities_list, dim=0)[:num_samples]
        
        # 创建可视化
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle(f'{model_name} 预测结果可视化', fontsize=16, fontweight='bold')
        
        class_names = ['negative', 'positive']
        
        for i in range(num_samples):
            row, col = divmod(i, cols)
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # 反归一化图像
            img = all_images[i]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            ax.imshow(img.permute(1, 2, 0))
            
            # 标题信息
            true_label = class_names[all_labels[i].item()]
            pred_label = class_names[all_predictions[i].item()]
            confidence = all_probabilities[i].max().item()
            
            # 根据预测正确性设置颜色
            color = 'green' if all_labels[i] == all_predictions[i] else 'red'
            
            ax.set_title(f'真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.3f}',
                        color=color, fontweight='bold')
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_samples, rows * cols):
            row, col = divmod(i, cols)
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{model_name}_predictions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"预测结果可视化已保存: {save_path}")
    
    def create_training_summary(self, models_results: Dict[str, Dict], 
                              models_history: Dict[str, Dict[str, List[float]]]):
        """创建训练总结报告"""
        
        # 创建HTML报告
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>模型训练总结报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; color: #333; }
                .section { margin: 30px 0; }
                .model-card { 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 20px 0; 
                    background-color: #f9f9f9;
                }
                .metric { display: inline-block; margin: 10px 20px; }
                .metric-value { font-weight: bold; color: #2196F3; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; }
                .best { background-color: #e8f5e8; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>生物抗菌素敏感性测试 - 模型训练总结报告</h1>
                <p>菌落检测二分类任务</p>
            </div>
        """
        
        # 添加模型对比表
        html_content += """
            <div class="section">
                <h2>模型性能对比</h2>
                <table>
                    <tr>
                        <th>模型</th>
                        <th>准确率</th>
                        <th>精确率</th>
                        <th>召回率</th>
                        <th>F1分数</th>
                        <th>AUC</th>
                        <th>敏感性</th>
                        <th>特异性</th>
                    </tr>
        """
        
        # 找到最佳模型
        best_model = max(models_results.items(), key=lambda x: x[1]['accuracy'])
        
        for model_name, results in models_results.items():
            is_best = model_name == best_model[0]
            row_class = 'best' if is_best else ''
            
            html_content += f"""
                    <tr class="{row_class}">
                        <td><strong>{model_name}</strong></td>
                        <td>{results['accuracy']:.4f}</td>
                        <td>{results['precision']:.4f}</td>
                        <td>{results['recall']:.4f}</td>
                        <td>{results['f1_score']:.4f}</td>
                        <td>{results['auc']:.4f}</td>
                        <td>{results['sensitivity']:.4f}</td>
                        <td>{results['specificity']:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                <p><em>绿色背景表示最佳模型</em></p>
            </div>
        """
        
        # 添加详细分析
        html_content += """
            <div class="section">
                <h2>详细分析</h2>
        """
        
        for model_name, results in models_results.items():
            is_best = model_name == best_model[0]
            
            html_content += f"""
                <div class="model-card">
                    <h3>{model_name} {'🏆 (最佳模型)' if is_best else ''}</h3>
                    <div class="metric">
                        <span>准确率:</span> 
                        <span class="metric-value">{results['accuracy']:.4f}</span>
                    </div>
                    <div class="metric">
                        <span>AUC:</span> 
                        <span class="metric-value">{results['auc']:.4f}</span>
                    </div>
                    <div class="metric">
                        <span>敏感性:</span> 
                        <span class="metric-value">{results['sensitivity']:.4f}</span>
                    </div>
                    <div class="metric">
                        <span>特异性:</span> 
                        <span class="metric-value">{results['specificity']:.4f}</span>
                    </div>
                    
                    <h4>混淆矩阵</h4>
                    <table style="width: 300px;">
                        <tr><th></th><th>预测 Negative</th><th>预测 Positive</th></tr>
                        <tr><th>真实 Negative</th><td>{results['confusion_matrix'][0][0]}</td><td>{results['confusion_matrix'][0][1]}</td></tr>
                        <tr><th>真实 Positive</th><td>{results['confusion_matrix'][1][0]}</td><td>{results['confusion_matrix'][1][1]}</td></tr>
                    </table>
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>训练建议</h2>
                <ul>
                    <li>数据集相对平衡，正负样本比例为 2910:2401</li>
                    <li>图像尺寸为70x70，适合轻量级模型</li>
                    <li>正样本(菌落)像素值较暗(均值151.8)，负样本较亮(均值245.1)</li>
                    <li>建议使用数据增强技术提高模型泛化能力</li>
                    <li>可以尝试集成学习方法进一步提升性能</li>
                </ul>
            </div>
            
        </body>
        </html>
        """
        
        # 保存HTML报告
        report_path = os.path.join(self.save_dir, 'training_summary_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"训练总结报告已保存: {report_path}")

if __name__ == "__main__":
    # 测试可视化器
    visualizer = TrainingVisualizer('./test_visualizations')
    
    # 模拟训练历史
    history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.7, 0.5, 0.4, 0.35, 0.3],
        'train_acc': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_acc': [0.65, 0.75, 0.8, 0.82, 0.85],
        'lr': [0.001, 0.001, 0.0005, 0.0002, 0.0001]
    }
    
    visualizer.plot_training_history(history, 'TestModel')
    print("可视化测试完成")
