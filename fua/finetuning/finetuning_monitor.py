"""
微调监控器

提供训练过程中的实时监控、可视化和分析功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
import logging
import time
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TrainingMetrics:
    """训练指标收集器"""
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.epoch_metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        self.current_step = 0
        self.current_epoch = 0
        
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """更新指标"""
        if step is not None:
            self.current_step = step
        
        # 更新滑动窗口
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.step_metrics[key].append((self.current_step, value))
    
    def update_epoch(self, metrics: Dict[str, float]):
        """更新epoch指标"""
        self.current_epoch += 1
        for key, value in metrics.items():
            self.epoch_metrics[key].append((self.current_epoch, value))
    
    def get_average(self, metric_name: str, window: Optional[int] = None) -> float:
        """获取指标平均值"""
        window = window or self.window_size
        values = list(self.metrics[metric_name])[-window:]
        return np.mean(values) if values else 0.0
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """获取最新指标值"""
        if self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def get_trend(self, metric_name: str, window: int = 10) -> float:
        """获取指标趋势（斜率）"""
        if len(self.metrics[metric_name]) < window:
            return 0.0
        
        values = list(self.metrics[metric_name])[-window:]
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """获取指标统计信息"""
        values = list(self.metrics[metric_name])
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'latest': values[-1],
            'trend': self.get_trend(metric_name)
        }
    
    def export_metrics(self, filepath: str):
        """导出指标到文件"""
        data = {
            'step_metrics': dict(self.step_metrics),
            'epoch_metrics': dict(self.epoch_metrics),
            'current_step': self.current_step,
            'current_epoch': self.current_epoch
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class GradientMonitor:
    """梯度监控器"""
    
    def __init__(self, model: nn.Module, log_frequency: int = 100):
        """
        Args:
            model: 要监控的模型
            log_frequency: 日志频率
        """
        self.model = model
        self.log_frequency = log_frequency
        self.gradient_stats = defaultdict(list)
        self.hooks = []
        self.step_count = 0
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """注册梯度钩子"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_backward_hook(
                    lambda grad, name=name: self._gradient_hook(grad, name)
                )
                self.hooks.append(hook)
    
    def _gradient_hook(self, grad, name):
        """梯度钩子函数"""
        if grad is not None:
            if self.step_count % self.log_frequency == 0:
                stats = {
                    'mean': grad.data.mean().item(),
                    'std': grad.data.std().item(),
                    'min': grad.data.min().item(),
                    'max': grad.data.max().item(),
                    'norm': grad.data.norm().item(),
                    'zeros': torch.sum(grad.data == 0).item(),
                    'nan': torch.isnan(grad.data).sum().item(),
                    'inf': torch.isinf(grad.data).sum().item()
                }
                self.gradient_stats[name].append((self.step_count, stats))
    
    def step(self):
        """步进计数器"""
        self.step_count += 1
    
    def get_gradient_summary(self) -> Dict[str, Any]:
        """获取梯度摘要"""
        summary = {
            'total_parameters': len(list(self.model.parameters())),
            'parameters_with_grad': 0,
            'problematic_gradients': [],
            'gradient_norms': {}
        }
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                summary['parameters_with_grad'] += 1
                
                # 计算梯度范数
                norm = param.grad.data.norm().item()
                summary['gradient_norms'][name] = norm
                
                # 检查问题梯度
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    summary['problematic_gradients'].append(name)
                elif norm < 1e-8:  # 梯度消失
                    summary['problematic_gradients'].append(f"{name} (vanishing)")
                elif norm > 10.0:  # 梯度爆炸
                    summary['problematic_gradients'].append(f"{name} (exploding)")
        
        return summary
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class ActivationMonitor:
    """激活值监控器"""
    
    def __init__(self, model: nn.Module, log_frequency: int = 100):
        """
        Args:
            model: 要监控的模型
            log_frequency: 日志频率
        """
        self.model = model
        self.log_frequency = log_frequency
        self.activation_stats = defaultdict(list)
        self.hooks = []
        self.step_count = 0
        self.dead_neurons = {}
    
    def _register_hooks(self):
        """注册前向钩子"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hook = module.register_forward_hook(
                    lambda module, input, output, name=name: self._forward_hook(module, input, output, name)
                )
                self.hooks.append(hook)
    
    def _forward_hook(self, module, input, output, name):
        """前向钩子函数"""
        if self.step_count % self.log_frequency == 0:
            if isinstance(output, torch.Tensor):
                stats = {
                    'mean': output.data.mean().item(),
                    'std': output.data.std().item(),
                    'min': output.data.min().item(),
                    'max': output.data.max().item(),
                    'zeros': torch.sum(output.data == 0).item(),
                    'dead_ratio': torch.sum(output.data == 0).float().mean().item()
                }
                self.activation_stats[name].append((self.step_count, stats))
                
                # 记录死亡神经元
                dead_ratio = stats['dead_ratio']
                if dead_ratio > 0.8:  # 80%以上神经元死亡
                    self.dead_neurons[name] = dead_ratio
    
    def step(self):
        """步进计数器"""
        self.step_count += 1
    
    def get_activation_summary(self) -> Dict[str, Any]:
        """获取激活值摘要"""
        summary = {
            'monitored_layers': len(self.hooks),
            'dead_neurons': self.dead_neurons,
            'layer_stats': {}
        }
        
        for name, stats_list in self.activation_stats.items():
            if stats_list:
                latest_stats = stats_list[-1][1]
                summary['layer_stats'][name] = latest_stats
        
        return summary
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class FineTuningMonitor:
    """微调监控器主类"""
    
    def __init__(self,
                 model: nn.Module,
                 log_dir: str = './logs',
                 enable_wandb: bool = False,
                 enable_tensorboard: bool = False,
                 wandb_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model: 要监控的模型
            log_dir: 日志目录
            enable_wandb: 是否启用 wandb
            enable_tensorboard: 是否启用 tensorboard
            wandb_config: wandb 配置
        """
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 初始化指标收集器
        self.metrics = TrainingMetrics(window_size=100)
        self.gradient_monitor = GradientMonitor(model)
        self.activation_monitor = ActivationMonitor(model)
        
        # 初始化日志系统
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        
        if self.enable_wandb:
            self._init_wandb(wandb_config or {})
        
        if self.enable_tensorboard:
            self._init_tensorboard()
        
        # 监控配置
        self.checkpoint_frequency = 1000
        self.eval_frequency = 100
        self.save_best_only = True
        self.best_metric = None
        
        # 训练状态
        self.start_time = time.time()
        self.total_steps = 0
        self.epochs_completed = 0
        
        logger.info(f"微调监控器初始化完成")
        logger.info(f"日志目录: {self.log_dir}")
        logger.info(f"WandB: {'启用' if self.enable_wandb else '禁用'}")
        logger.info(f"TensorBoard: {'启用' if self.enable_tensorboard else '禁用'}")
    
    def _init_wandb(self, config: Dict[str, Any]):
        """初始化 wandb"""
        wandb.init(
            project=config.get('project', 'fua-finetuning'),
            name=config.get('name', f'finetuning-{datetime.now().strftime("%Y%m%d-%H%M%S")}'),
            config=config.get('config', {}),
            dir=str(self.log_dir)
        )
        wandb.watch(self.model)
    
    def _init_tensorboard(self):
        """初始化 tensorboard"""
        log_dir = self.log_dir / 'tensorboard'
        log_dir.mkdir(exist_ok=True)
        self.tb_writer = SummaryWriter(str(log_dir))
    
    def update_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """更新训练指标"""
        # 更新本地指标
        self.metrics.update(metrics, step)
        
        # 记录到 wandb
        if self.enable_wandb:
            wandb.log(metrics, step=step)
        
        # 记录到 tensorboard
        if self.enable_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
    
    def update_epoch_metrics(self, metrics: Dict[str, float]):
        """更新 epoch 指标"""
        self.metrics.update_epoch(metrics)
        
        epoch = self.metrics.current_epoch
        
        if self.enable_wandb:
            wandb.log({f"epoch/{k}": v for k, v in metrics.items()}, step=epoch)
        
        if self.enable_tensorboard:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f"epoch/{key}", value, epoch)
    
    def log_gradients(self, step: int):
        """记录梯度信息"""
        self.gradient_monitor.step()
        
        if step % self.gradient_monitor.log_frequency == 0:
            summary = self.gradient_monitor.get_gradient_summary()
            
            # 记录到 wandb
            if self.enable_wandb:
                grad_metrics = {
                    f"grad/norm_{name.replace('.', '_')}": norm
                    for name, norm in summary['gradient_norms'].items()
                }
                grad_metrics['grad/problematic_count'] = len(summary['problematic_gradients'])
                wandb.log(grad_metrics, step=step)
            
            # 记录到 tensorboard
            if self.enable_tensorboard:
                for name, norm in summary['gradient_norms'].items():
                    self.tb_writer.add_scalar(f"grad/norm_{name.replace('.', '_')}", norm, step)
    
    def log_activations(self, step: int):
        """记录激活值信息"""
        self.activation_monitor.step()
        
        if step % self.activation_monitor.log_frequency == 0:
            summary = self.activation_monitor.get_activation_summary()
            
            # 记录死亡神经元
            if self.enable_wandb and summary['dead_neurons']:
                dead_metrics = {
                    f"activation/dead_ratio_{name.replace('.', '_')}": ratio
                    for name, ratio in summary['dead_neurons'].items()
                }
                wandb.log(dead_metrics, step=step)
    
    def log_model_stats(self, step: int):
        """记录模型统计信息"""
        # 计算参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_ratio': trainable_params / total_params
        }
        
        # 计算内存使用
        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
        
        # 计算训练速度
        if step > 0:
            elapsed_time = time.time() - self.start_time
            stats['steps_per_second'] = step / elapsed_time
            stats['seconds_per_step'] = elapsed_time / step
        
        if self.enable_wandb:
            wandb.log({'model_stats': stats}, step=step)
        
        if self.enable_tensorboard:
            for key, value in stats.items():
                self.tb_writer.add_scalar(f"model_stats/{key}", value, step)
    
    def evaluate_model(self, 
                      eval_loader: DataLoader,
                      criterion: nn.Module,
                      step: int) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(next(self.model.parameters()).device), target.to(next(self.model.parameters()).device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(eval_loader)
        accuracy = correct / total
        
        eval_metrics = {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy
        }
        
        # 更新指标
        self.update_metrics(eval_metrics, step)
        
        # 检查是否是最佳模型
        if self.best_metric is None or accuracy > self.best_metric:
            self.best_metric = accuracy
            if self.save_best_only:
                self.save_checkpoint(step, is_best=True)
        
        return eval_metrics
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """保存检查点"""
        checkpoint_dir = self.log_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'step': step,
            'epoch': self.metrics.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': dict(self.metrics.epoch_metrics),
            'best_metric': self.best_metric
        }
        
        # 保存最新检查点
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
        
        # 定期保存检查点
        if step % self.checkpoint_frequency == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_step_{step}.pth')
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """生成监控报告"""
        report = {
            'training_summary': {
                'total_steps': self.total_steps,
                'epochs_completed': self.epochs_completed,
                'best_metric': self.best_metric,
                'training_time': time.time() - self.start_time
            },
            'metrics_summary': {},
            'gradient_summary': self.gradient_monitor.get_gradient_summary(),
            'activation_summary': self.activation_monitor.get_activation_summary()
        }
        
        # 添加指标摘要
        for metric_name in self.metrics.metrics.keys():
            report['metrics_summary'][metric_name] = self.metrics.get_statistics(metric_name)
        
        # 保存报告
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_metrics(self, metrics: List[str], save_dir: Optional[str] = None):
        """绘制指标曲线"""
        if save_dir is None:
            save_dir = self.log_dir / 'plots'
        save_dir.mkdir(exist_ok=True)
        
        for metric_name in metrics:
            plt.figure(figsize=(10, 6))
            
            # 绘制步级指标
            if metric_name in self.metrics.step_metrics:
                steps, values = zip(*self.metrics.step_metrics[metric_name])
                plt.plot(steps, values, label=metric_name, alpha=0.7)
            
            # 绘制epoch指标
            if metric_name in self.metrics.epoch_metrics:
                epochs, values = zip(*self.metrics.epoch_metrics[metric_name])
                plt.plot(epochs, values, 'o-', label=f'{metric_name} (epoch)', linewidth=2)
            
            plt.xlabel('Step/Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{metric_name} over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(save_dir / f'{metric_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def cleanup(self):
        """清理资源"""
        # 移除钩子
        self.gradient_monitor.remove_hooks()
        self.activation_monitor.remove_hooks()
        
        # 关闭 tensorboard
        if self.enable_tensorboard:
            self.tb_writer.close()
        
        # 完成 wandb
        if self.enable_wandb:
            wandb.finish()
        
        # 导出最终指标
        self.metrics.export_metrics(self.log_dir / 'metrics.json')
        
        # 生成最终报告
        self.generate_report(self.log_dir / 'final_report.json')
        
        logger.info("微调监控器已清理")


# 工厂函数
def create_finetuning_monitor(model: nn.Module,
                            log_dir: str = './logs',
                            **kwargs) -> FineTuningMonitor:
    """创建微调监控器的工厂函数"""
    return FineTuningMonitor(model, log_dir, **kwargs)