"""
分层学习率调度器

实现支持不同层组使用不同学习率的调度器
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Union, Optional, Callable, Any
import logging
from itertools import chain
import numpy as np

logger = logging.getLogger(__name__)


class LayerGroup:
    """层组定义"""
    
    def __init__(self, 
                 name: str,
                 layer_names: List[str],
                 lr_multiplier: float = 1.0,
                 weight_decay_multiplier: float = 1.0):
        """
        初始化层组
        
        Args:
            name: 层组名称
            layer_names: 包含的层名列表（支持通配符）
            lr_multiplier: 学习率倍数
            weight_decay_multiplier: 权重衰减倍数
        """
        self.name = name
        self.layer_names = layer_names
        self.lr_multiplier = lr_multiplier
        self.weight_decay_multiplier = weight_decay_multiplier
        self.parameters = []  # 实际匹配的参数
        
    def match_layers(self, model: nn.Module) -> List[nn.Parameter]:
        """匹配模型中的层"""
        matched_params = []
        matched_names = []
        
        for name, param in model.named_parameters():
            # 检查是否匹配任何层名模式
            for pattern in self.layer_names:
                if self._match_pattern(name, pattern):
                    matched_params.append(param)
                    matched_names.append(name)
                    break
        
        self.parameters = matched_params
        logger.info(f"层组 '{self.name}' 匹配了 {len(matched_params)} 个参数")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"匹配的参数: {matched_names[:5]}{'...' if len(matched_names) > 5 else ''}")
        
        return matched_params
    
    def _match_pattern(self, name: str, pattern: str) -> bool:
        """检查参数名是否匹配模式"""
        # 支持通配符 * 和 ?
        pattern = pattern.replace('.', '\.')
        pattern = pattern.replace('*', '.*')
        pattern = pattern.replace('?', '.')
        
        import re
        return re.fullmatch(pattern, name) is not None


class LayeredLRScheduler:
    """分层学习率调度器"""
    
    def __init__(self,
                 model: nn.Module,
                 base_lr: float,
                 layer_groups: List[Dict[str, Any]],
                 scheduler_type: str = 'cosine',
                 scheduler_params: Optional[Dict[str, Any]] = None,
                 warmup_steps: int = 0,
                 warmup_start_lr: float = 0.0):
        """
        初始化分层学习率调度器
        
        Args:
            model: PyTorch模型
            base_lr: 基础学习率
            layer_groups: 层组配置列表
            scheduler_type: 调度器类型
            scheduler_params: 调度器参数
            warmup_steps: 预热步数
            warmup_start_lr: 预热起始学习率
        """
        self.model = model
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0
        
        # 创建层组
        self.layer_groups = []
        for group_config in layer_groups:
            group = LayerGroup(
                name=group_config['name'],
                layer_names=group_config['layer_names'],
                lr_multiplier=group_config.get('lr_multiplier', 1.0),
                weight_decay_multiplier=group_config.get('weight_decay_multiplier', 1.0)
            )
            group.match_layers(model)
            self.layer_groups.append(group)
        
        # 检查是否有参数未被分配
        all_group_params = set()
        for group in self.layer_groups:
            all_group_params.update(id(p) for p in group.parameters)
        
        unmatched_params = []
        for param in model.parameters():
            if id(param) not in all_group_params and param.requires_grad:
                unmatched_params.append(param)
        
        if unmatched_params:
            logger.warning(f"发现 {len(unmatched_params)} 个未分配的参数，将使用默认学习率")
            # 创建默认组
            default_group = LayerGroup(
                name='default',
                layer_names=['unmatched'],
                lr_multiplier=1.0,
                weight_decay_multiplier=1.0
            )
            default_group.parameters = unmatched_params
            self.layer_groups.append(default_group)
        
        # 创建参数组
        self.param_groups = self._create_param_groups()
        
        # 创建基础调度器
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        self.base_scheduler = self._create_base_scheduler()
        
        logger.info(f"分层学习率调度器初始化完成")
        logger.info(f"基础学习率: {base_lr}")
        for group in self.layer_groups:
            logger.info(f"层组 '{group.name}': lr={base_lr * group.lr_multiplier}, "
                       f"参数数量={len(group.parameters)}")
    
    def _create_param_groups(self) -> List[Dict[str, Any]]:
        """创建优化器参数组"""
        param_groups = []
        
        for group in self.layer_groups:
            if not group.parameters:
                continue
                
            param_groups.append({
                'params': group.parameters,
                'lr': self.base_lr * group.lr_multiplier,
                'weight_decay': self.scheduler_params.get('weight_decay', 0.0) * group.weight_decay_multiplier,
                'group_name': group.name,
                'lr_multiplier': group.lr_multiplier
            })
        
        return param_groups
    
    def _create_base_scheduler(self) -> _LRScheduler:
        """创建基础学习率调度器"""
        # 创建一个虚拟的优化器来创建调度器
        dummy_optimizer = torch.optim.Adam(self.param_groups, lr=self.base_lr)
        
        if self.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                dummy_optimizer,
                T_max=self.scheduler_params.get('T_max', 100),
                eta_min=self.scheduler_params.get('eta_min', 0)
            )
        elif self.scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                dummy_optimizer,
                step_size=self.scheduler_params.get('step_size', 30),
                gamma=self.scheduler_params.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                dummy_optimizer,
                gamma=self.scheduler_params.get('gamma', 0.95)
            )
        elif self.scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                dummy_optimizer,
                mode=self.scheduler_params.get('mode', 'min'),
                factor=self.scheduler_params.get('factor', 0.1),
                patience=self.scheduler_params.get('patience', 10),
                verbose=True
            )
        elif self.scheduler_type == 'one_cycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                dummy_optimizer,
                max_lr=[pg['lr'] for pg in self.param_groups],
                total_steps=self.scheduler_params.get('total_steps', 100),
                pct_start=self.scheduler_params.get('pct_start', 0.3),
                anneal_strategy=self.scheduler_params.get('anneal_strategy', 'cos')
            )
        else:
            raise ValueError(f"不支持的调度器类型: {self.scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """执行一步调度"""
        self.current_step += 1
        
        # 预热
        if self.current_step <= self.warmup_steps:
            self._warmup_step()
        else:
            # 正常调度
            if self.scheduler_type == 'reduce_on_plateau':
                if metric is None:
                    logger.warning("ReduceLROnPlateau 需要提供 metric 参数")
                else:
                    self.base_scheduler.step(metric)
            else:
                self.base_scheduler.step()
    
    def _warmup_step(self):
        """预热步骤"""
        # 线性预热
        warmup_progress = self.current_step / self.warmup_steps
        for param_group in self.param_groups:
            target_lr = param_group['lr']
            start_lr = self.warmup_start_lr * param_group['lr_multiplier']
            param_group['lr'] = start_lr + (target_lr - start_lr) * warmup_progress
    
    def get_lr(self) -> List[float]:
        """获取当前学习率"""
        return [pg['lr'] for pg in self.param_groups]
    
    def get_lr_by_group(self) -> Dict[str, float]:
        """按组获取当前学习率"""
        return {pg['group_name']: pg['lr'] for pg in self.param_groups}
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'current_step': self.current_step,
            'base_scheduler_state': self.base_scheduler.state_dict(),
            'param_groups': self.param_groups
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.current_step = state_dict['current_step']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler_state'])
        # 恢复参数组学习率
        for saved_group, current_group in zip(state_dict['param_groups'], self.param_groups):
            current_group['lr'] = saved_group['lr']
    
    def print_lr(self, is_verbose: bool = True):
        """打印学习率信息"""
        if is_verbose:
            print(f"Step {self.current_step}:")
            for group in self.param_groups:
                print(f"  {group['group_name']}: {group['lr']:.6f}")


class DifferentialLearningRateFinder:
    """差分学习率查找器"""
    
    def __init__(self, 
                 model: nn.Module,
                 layer_groups: List[Dict[str, Any]],
                 criterion: nn.Module,
                 device: torch.device):
        """
        初始化学习率查找器
        
        Args:
            model: 模型
            layer_groups: 层组配置
            criterion: 损失函数
            device: 设备
        """
        self.model = model
        self.layer_groups = layer_groups
        self.criterion = criterion
        self.device = device
    
    def find_lr(self, 
                data_loader,
                init_lr: float = 1e-7,
                final_lr: float = 10,
                num_iter: int = 100,
                beta: float = 0.98) -> Dict[str, List[float]]:
        """
        查找最佳学习率
        
        Args:
            data_loader: 数据加载器
            init_lr: 初始学习率
            final_lr: 最终学习率
            num_iter: 迭代次数
            beta: 平滑系数
            
        Returns:
            各层组的学习率和损失历史
        """
        model = self.model.to(self.device)
        model.train()
        
        # 创建分层学习率调度器
        scheduler = LayeredLRScheduler(
            model=model,
            base_lr=init_lr,
            layer_groups=self.layer_groups,
            scheduler_type='linear',
            scheduler_params={'total_steps': num_iter}
        )
        
        optimizer = torch.optim.Adam(scheduler.param_groups)
        
        # 记录损失
        avg_loss = 0.0
        best_loss = float('inf')
        losses = []
        lrs = {group['name']: [] for group in self.layer_groups}
        
        batch_idx = 0
        for data, target in data_loader:
            if batch_idx >= num_iter:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            
            # 平滑损失
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta ** (batch_idx + 1))
            
            # 记录最佳损失
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # 记录学习率和损失
            for group in scheduler.layer_groups:
                lrs[group.name].append(scheduler.base_lr * group.lr_multiplier)
            losses.append(smoothed_loss)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 线性增加学习率
            lr_mult = (final_lr / init_lr) ** (1 / num_iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_mult
            
            batch_idx += 1
        
        return {
            'losses': losses,
            'lrs': lrs,
            'best_loss': best_loss
        }


# 工厂函数
def create_layered_scheduler(model: nn.Module,
                             base_lr: float,
                             layer_groups_config: List[Dict[str, Any]],
                             scheduler_type: str = 'cosine',
                             **kwargs) -> LayeredLRScheduler:
    """创建分层学习率调度器的工厂函数"""
    return LayeredLRScheduler(
        model=model,
        base_lr=base_lr,
        layer_groups=layer_groups_config,
        scheduler_type=scheduler_type,
        **kwargs
    )


def create_lr_finder(model: nn.Module,
                    layer_groups_config: List[Dict[str, Any]],
                    criterion: nn.Module,
                    device: torch.device) -> DifferentialLearningRateFinder:
    """创建学习率查找器的工厂函数"""
    return DifferentialLearningRateFinder(
        model=model,
        layer_groups=layer_groups_config,
        criterion=criterion,
        device=device
    )


# 预定义的层组配置
def get_resnet_layer_groups() -> List[Dict[str, Any]]:
    """ResNet 预定义层组"""
    return [
        {
            'name': 'early_layers',
            'layer_names': ['conv1', 'bn1', 'layer1.*'],
            'lr_multiplier': 0.1,
            'weight_decay_multiplier': 1.0
        },
        {
            'name': 'middle_layers',
            'layer_names': ['layer2.*', 'layer3.*'],
            'lr_multiplier': 0.5,
            'weight_decay_multiplier': 1.0
        },
        {
            'name': 'late_layers',
            'layer_names': ['layer4.*', 'fc'],
            'lr_multiplier': 1.0,
            'weight_decay_multiplier': 1.0
        }
    ]


def get_vit_layer_groups() -> List[Dict[str, Any]]:
    """Vision Transformer 预定义层组"""
    return [
        {
            'name': 'patch_embed',
            'layer_names': ['patch_embed.*'],
            'lr_multiplier': 0.1,
            'weight_decay_multiplier': 1.0
        },
        {
            'name': 'early_blocks',
            'layer_names': ['blocks.[0-5].*'],
            'lr_multiplier': 0.3,
            'weight_decay_multiplier': 1.0
        },
        {
            'name': 'middle_blocks',
            'layer_names': ['blocks.[6-9].*'],
            'lr_multiplier': 0.6,
            'weight_decay_multiplier': 1.0
        },
        {
            'name': 'late_blocks',
            'layer_names': ['blocks.[1-9].*', 'norm.*', 'head.*'],
            'lr_multiplier': 1.0,
            'weight_decay_multiplier': 1.0
        }
    ]


def get_efficientnet_layer_groups() -> List[Dict[str, Any]]:
    """EfficientNet 预定义层组"""
    return [
        {
            'name': 'stem',
            'layer_names': ['conv_stem.*', 'bn1.*'],
            'lr_multiplier': 0.1,
            'weight_decay_multiplier': 1.0
        },
        {
            'name': 'early_blocks',
            'layer_names': ['blocks.[0-9].*'],
            'lr_multiplier': 0.3,
            'weight_decay_multiplier': 1.0
        },
        {
            'name': 'middle_blocks',
            'layer_names': ['blocks.[1-2].*'],
            'lr_multiplier': 0.6,
            'weight_decay_multiplier': 1.0
        },
        {
            'name': 'late_blocks',
            'layer_names': ['blocks.[3-9].*', 'conv_head.*', 'bn2.*', 'classifier.*'],
            'lr_multiplier': 1.0,
            'weight_decay_multiplier': 1.0
        }
    ]