"""
高级微调器

集成所有微调功能的高级微调器主类
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import logging
import time
from pathlib import Path
import json
from datetime import datetime

from .layered_lr_scheduler import LayeredLRScheduler, create_layered_scheduler
from .loss_function_factory import create_loss
from .architecture_modifier import ArchitectureModifier, create_architecture_modifier
from .finetuning_monitor import FineTuningMonitor, create_finetuning_monitor

logger = logging.getLogger(__name__)


class FineTuningConfig:
    """微调配置类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 学习率配置
        self.base_lr = config.get('base_lr', 1e-4)
        self.layer_groups = config.get('layer_groups', [])
        self.scheduler_type = config.get('scheduler_type', 'cosine')
        self.scheduler_params = config.get('scheduler_params', {})
        self.warmup_steps = config.get('warmup_steps', 0)
        
        # 损失函数配置
        self.loss_config = config.get('loss_config', {'type': 'cross_entropy'})
        
        # 优化器配置
        self.optimizer_type = config.get('optimizer_type', 'adam')
        self.optimizer_params = config.get('optimizer_params', {})
        
        # 训练配置
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config.get('num_epochs', 10)
        self.gradient_clip = config.get('gradient_clip', None)
        self.early_stopping_patience = config.get('early_stopping_patience', None)
        
        # 监控配置
        self.log_frequency = config.get('log_frequency', 100)
        self.eval_frequency = config.get('eval_frequency', 100)
        self.save_frequency = config.get('save_frequency', 1000)
        
        # 架构修改配置
        self.architecture_modifications = config.get('architecture_modifications', [])
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置"""
        # 检查必需的配置项
        if not self.layer_groups:
            logger.warning("未配置层组，将使用默认学习率")
        
        # 检查损失函数配置
        if 'type' not in self.loss_config:
            raise ValueError("损失函数配置必须包含 'type' 字段")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()


class FineTuningResult:
    """微调结果类"""
    
    def __init__(self,
                 config: FineTuningConfig,
                 best_model_state: Dict[str, Any],
                 metrics_history: Dict[str, List[float]],
                 training_time: float,
                 best_metric: float,
                 best_epoch: int):
        self.config = config
        self.best_model_state = best_model_state
        self.metrics_history = metrics_history
        self.training_time = training_time
        self.best_metric = best_metric
        self.best_epoch = best_epoch
        self.timestamp = datetime.now()
    
    def save(self, save_path: str):
        """保存结果"""
        result_data = {
            'config': self.config.to_dict(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'training_time': self.training_time,
            'timestamp': self.timestamp.isoformat(),
            'metrics_history': self.metrics_history
        }
        
        # 保存结果数据
        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # 保存最佳模型
        model_path = Path(save_path).with_suffix('.pth')
        torch.save(self.best_model_state, model_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        return {
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'training_time': self.training_time,
            'total_epochs': len(self.metrics_history.get('train_loss', [])),
            'final_train_loss': self.metrics_history.get('train_loss', [])[-1] if self.metrics_history.get('train_loss') else None,
            'final_val_loss': self.metrics_history.get('val_loss', [])[-1] if self.metrics_history.get('val_loss') else None,
            'final_val_accuracy': self.metrics_history.get('val_accuracy', [])[-1] if self.metrics_history.get('val_accuracy') else None
        }


class AdvancedFineTuner:
    """高级微调器主类"""
    
    def __init__(self,
                 model: nn.Module,
                 config: Union[Dict[str, Any], FineTuningConfig],
                 device: Optional[torch.device] = None):
        """
        初始化高级微调器
        
        Args:
            model: 要微调的模型
            config: 微调配置
            device: 设备
        """
        self.model = model
        self.config = config if isinstance(config, FineTuningConfig) else FineTuningConfig(config)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 将模型移到设备
        self.model = self.model.to(self.device)
        
        # 初始化组件
        self.arch_modifier = None
        self.lr_scheduler = None
        self.loss_fn = None
        self.monitor = None
        self.optimizer = None
        
        # 训练状态
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = None
        self.best_epoch = 0
        self.early_stopping_counter = 0
        self.should_stop = False
        
        # 历史记录
        self.train_metrics_history = []
        self.val_metrics_history = []
        
        logger.info(f"高级微调器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"基础学习率: {self.config.base_lr}")
        logger.info(f"批大小: {self.config.batch_size}")
        logger.info(f"训练轮数: {self.config.num_epochs}")
    
    def setup(self):
        """设置微调环境"""
        logger.info("设置微调环境...")
        
        # 1. 应用架构修改
        if self.config.architecture_modifications:
            self._apply_architecture_modifications()
        
        # 2. 创建分层学习率调度器
        self._setup_lr_scheduler()
        
        # 3. 创建损失函数
        self._setup_loss_function()
        
        # 4. 创建优化器
        self._setup_optimizer()
        
        # 5. 创建监控器
        self._setup_monitor()
        
        logger.info("微调环境设置完成")
    
    def _apply_architecture_modifications(self):
        """应用架构修改"""
        self.arch_modifier = create_architecture_modifier(self.model)
        
        for mod in self.config.architecture_modifications:
            mod_type = mod['type']
            
            if mod_type == 'add_layer':
                self.arch_modifier.add_layer(
                    parent_name=mod['parent_name'],
                    layer_type=mod['layer_type'],
                    layer_config=mod['layer_config'],
                    insert_position=mod.get('insert_position', 'after')
                )
            elif mod_type == 'remove_layer':
                self.arch_modifier.remove_layer(mod['layer_name'])
            elif mod_type == 'adjust_dimensions':
                self.arch_modifier.adjust_layer_dimensions(
                    layer_name=mod['layer_name'],
                    new_dimensions=mod['new_dimensions']
                )
            elif mod_type == 'add_skip_connection':
                self.arch_modifier.add_skip_connection(
                    from_layer=mod['from_layer'],
                    to_layer=mod['to_layer'],
                    connection_type=mod.get('connection_type', 'residual')
                )
            elif mod_type == 'freeze_layers':
                self.arch_modifier.freeze_layers(
                    layer_names=mod['layer_names'],
                    freeze_bn=mod.get('freeze_bn', True)
                )
    
    def _setup_lr_scheduler(self):
        """设置学习率调度器"""
        if self.config.layer_groups:
            self.lr_scheduler = create_layered_scheduler(
                model=self.model,
                base_lr=self.config.base_lr,
                layer_groups_config=self.config.layer_groups,
                scheduler_type=self.config.scheduler_type,
                scheduler_params=self.config.scheduler_params,
                warmup_steps=self.config.warmup_steps
            )
        else:
            # 如果没有配置层组，使用默认调度器
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                torch.optim.Adam(self.model.parameters(), lr=self.config.base_lr),
                T_max=self.config.num_epochs
            )
    
    def _setup_loss_function(self):
        """设置损失函数"""
        self.loss_fn = create_loss(self.config.loss_config)
    
    def _setup_optimizer(self):
        """设置优化器"""
        if hasattr(self.lr_scheduler, 'param_groups'):
            # 使用分层调度器的参数组
            if self.config.optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.lr_scheduler.param_groups,
                    **self.config.optimizer_params
                )
            elif self.config.optimizer_type == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.lr_scheduler.param_groups,
                    **self.config.optimizer_params
                )
            else:
                raise ValueError(f"不支持的优化器类型: {self.config.optimizer_type}")
        else:
            # 使用标准优化器
            if self.config.optimizer_type == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.config.base_lr,
                    **self.config.optimizer_params
                )
            elif self.config.optimizer_type == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.config.base_lr,
                    **self.config.optimizer_params
                )
            else:
                raise ValueError(f"不支持的优化器类型: {self.config.optimizer_type}")
    
    def _setup_monitor(self):
        """设置监控器"""
        log_dir = f'./logs/finetuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.monitor = create_finetuning_monitor(
            model=self.model,
            log_dir=log_dir,
            enable_wandb=False,  # 默认关闭，可通过配置启用
            enable_tensorboard=False
        )
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # 记录 epoch 开始时间
        epoch_start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            # 优化器步进
            self.optimizer.step()
            
            # 更新统计
            epoch_loss += loss.item()
            _, predicted = output.max(1)
            epoch_total += target.size(0)
            epoch_correct += predicted.eq(target).sum().item()
            
            # 更新步数
            self.current_step += 1
            
            # 记录指标
            if self.current_step % self.config.log_frequency == 0:
                step_metrics = {
                    'train_loss': loss.item(),
                    'train_accuracy': predicted.eq(target).sum().item() / target.size(0),
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.monitor.update_metrics(step_metrics, self.current_step)
                
                # 记录梯度
                self.monitor.log_gradients(self.current_step)
                
                # 记录激活值
                self.monitor.log_activations(self.current_step)
                
                # 记录模型统计
                self.monitor.log_model_stats(self.current_step)
                
                logger.info(
                    f"Step {self.current_step}: Loss={loss.item():.4f}, "
                    f"Acc={predicted.eq(target).sum().item() / target.size(0):.4f}, "
                    f"LR={self.optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # 定期验证
            if val_loader and self.current_step % self.config.eval_frequency == 0:
                val_metrics = self._validate(val_loader)
                
                # 早停检查
                if self.config.early_stopping_patience:
                    if self._check_early_stopping(val_metrics):
                        logger.info(f"早停触发，在第 {self.current_epoch} 轮停止训练")
                        self.should_stop = True
                        break
        
        # 计算 epoch 指标
        epoch_metrics = {
            'train_loss': epoch_loss / len(train_loader),
            'train_accuracy': epoch_correct / epoch_total,
            'epoch_time': time.time() - epoch_start_time
        }
        
        # 更新学习率调度器
        if isinstance(self.lr_scheduler, LayeredLRScheduler):
            self.lr_scheduler.step()
        elif hasattr(self.lr_scheduler, 'step'):
            self.lr_scheduler.step()
        
        return epoch_metrics
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        return self.monitor.evaluate_model(
            val_loader, 
            self.loss_fn, 
            self.current_step
        )
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """检查早停条件"""
        # 使用验证准确率作为指标
        current_metric = val_metrics.get('val_accuracy', 0)
        
        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = self.current_epoch
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def finetune(self,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                save_dir: Optional[str] = None) -> FineTuningResult:
        """
        执行微调
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            save_dir: 保存目录
            
        Returns:
            微调结果
        """
        logger.info("开始微调...")
        
        # 设置环境
        self.setup()
        
        # 创建保存目录
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            if self.should_stop:
                break
            
            self.current_epoch = epoch + 1
            logger.info(f"\nEpoch {self.current_epoch}/{self.config.num_epochs}")
            
            # 训练一个 epoch
            train_metrics = self.train_epoch(train_loader, val_loader)
            self.train_metrics_history.append(train_metrics)
            
            # 验证
            if val_loader:
                val_metrics = self._validate(val_loader)
                self.val_metrics_history.append(val_metrics)
                
                # 更新 epoch 指标
                epoch_metrics = {**train_metrics, **val_metrics}
                self.monitor.update_epoch_metrics(epoch_metrics)
                
                logger.info(
                    f"Epoch {self.current_epoch}: "
                    f"Train Loss={train_metrics['train_loss']:.4f}, "
                    f"Train Acc={train_metrics['train_accuracy']:.4f}, "
                    f"Val Loss={val_metrics['eval_loss']:.4f}, "
                    f"Val Acc={val_metrics['eval_accuracy']:.4f}"
                )
            else:
                self.monitor.update_epoch_metrics(train_metrics)
                logger.info(
                    f"Epoch {self.current_epoch}: "
                    f"Train Loss={train_metrics['train_loss']:.4f}, "
                    f"Train Acc={train_metrics['train_accuracy']:.4f}"
                )
            
            # 保存检查点
            if save_dir and (self.current_epoch % 10 == 0 or self.current_epoch == self.config.num_epochs):
                checkpoint = {
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': self.best_metric,
                    'config': self.config.to_dict()
                }
                torch.save(checkpoint, save_dir / f'checkpoint_epoch_{self.current_epoch}.pth')
        
        # 计算总训练时间
        training_time = time.time() - start_time
        
        # 加载最佳模型
        if val_loader and save_dir:
            best_checkpoint = torch.load(save_dir / 'best.pth', map_location=self.device)
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # 整理结果
        result = FineTuningResult(
            config=self.config,
            best_model_state=self.model.state_dict(),
            metrics_history={
                'train': self.train_metrics_history,
                'val': self.val_metrics_history
            },
            training_time=training_time,
            best_metric=self.best_metric or 0.0,
            best_epoch=self.best_epoch
        )
        
        # 保存结果
        if save_dir:
            result.save(save_dir / 'finetuning_result.json')
            
            # 生成可视化
            self.monitor.plot_metrics(
                ['train_loss', 'train_accuracy', 'eval_loss', 'eval_accuracy'],
                save_dir / 'plots'
            )
        
        # 清理监控器
        self.monitor.cleanup()
        
        logger.info(f"微调完成！")
        logger.info(f"总训练时间: {training_time:.2f} 秒")
        logger.info(f"最佳指标: {self.best_metric:.4f} (Epoch {self.best_epoch})")
        
        return result
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_ratio': trainable_params / total_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # 假设float32
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }
        
        # 添加架构修改摘要
        if self.arch_modifier:
            summary['architecture_modifications'] = self.arch_modifier.get_modification_summary()
        
        return summary


# 工厂函数
def create_advanced_finetuner(model: nn.Module,
                             config: Dict[str, Any],
                             device: Optional[torch.device] = None) -> AdvancedFineTuner:
    """创建高级微调器的工厂函数"""
    return AdvancedFineTuner(model, config, device)


# 预定义配置
def get_default_finetuning_config(model_type: str = 'resnet') -> Dict[str, Any]:
    """获取默认微调配置"""
    configs = {
        'resnet': {
            'base_lr': 1e-4,
            'layer_groups': [
                {
                    'name': 'early_layers',
                    'layer_names': ['conv1', 'bn1', 'layer1.*'],
                    'lr_multiplier': 0.1
                },
                {
                    'name': 'middle_layers',
                    'layer_names': ['layer2.*', 'layer3.*'],
                    'lr_multiplier': 0.5
                },
                {
                    'name': 'late_layers',
                    'layer_names': ['layer4.*', 'fc'],
                    'lr_multiplier': 1.0
                }
            ],
            'scheduler_type': 'cosine',
            'scheduler_params': {'T_max': 100},
            'loss_config': {'type': 'cross_entropy'},
            'optimizer_type': 'adam',
            'optimizer_params': {'weight_decay': 1e-4},
            'batch_size': 32,
            'num_epochs': 50,
            'gradient_clip': 1.0,
            'early_stopping_patience': 10
        },
        'vit': {
            'base_lr': 5e-5,
            'layer_groups': [
                {
                    'name': 'patch_embed',
                    'layer_names': ['patch_embed.*'],
                    'lr_multiplier': 0.1
                },
                {
                    'name': 'early_blocks',
                    'layer_names': ['blocks.[0-5].*'],
                    'lr_multiplier': 0.3
                },
                {
                    'name': 'middle_blocks',
                    'layer_names': ['blocks.[6-9].*'],
                    'lr_multiplier': 0.6
                },
                {
                    'name': 'late_blocks',
                    'layer_names': ['blocks.[1-9].*', 'norm.*', 'head.*'],
                    'lr_multiplier': 1.0
                }
            ],
            'scheduler_type': 'cosine',
            'loss_config': {'type': 'label_smoothing', 'params': {'smoothing': 0.1}},
            'optimizer_type': 'adamw',
            'optimizer_params': {'weight_decay': 0.05},
            'batch_size': 64,
            'num_epochs': 100,
            'warmup_steps': 1000,
            'gradient_clip': 1.0,
            'early_stopping_patience': 15
        }
    }
    
    return configs.get(model_type, configs['resnet'])