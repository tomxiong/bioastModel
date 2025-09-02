"""
架构安全修改器

提供安全的模型架构修改功能，包括层添加、删除和维度调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import copy
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


class ArchitectureModifier:
    """架构安全修改器"""
    
    def __init__(self, model: nn.Module, safety_checks: bool = True):
        """
        初始化架构修改器
        
        Args:
            model: 要修改的模型
            safety_checks: 是否启用安全检查
        """
        self.original_model = copy.deepcopy(model)
        self.model = model
        self.safety_checks = safety_checks
        self.modification_history = []
        
        # 注册钩子用于监控
        self.hooks = []
        self.activation_stats = {}
        
    def add_layer(self, 
                  parent_name: str,
                  layer_type: str,
                  layer_config: Dict[str, Any],
                  insert_position: str = 'after',
                  validate: bool = True) -> bool:
        """
        安全地添加层
        
        Args:
            parent_name: 父层名称
            layer_type: 要添加的层类型
            layer_config: 层配置
            insert_position: 插入位置 ('before' 或 'after')
            validate: 是否验证修改
            
        Returns:
            修改是否成功
        """
        try:
            if validate:
                if not self._validate_add_layer(parent_name, layer_type, layer_config):
                    return False
            
            # 创建新层
            new_layer = self._create_layer(layer_type, layer_config)
            if new_layer is None:
                logger.error(f"无法创建层类型: {layer_type}")
                return False
            
            # 查找父层位置
            parent_module = self._find_module(parent_name)
            if parent_module is None:
                logger.error(f"找不到父层: {parent_name}")
                return False
            
            # 获取父容器的路径
            container_path, parent_key = self._get_container_path(parent_name)
            if container_path is None:
                logger.error(f"无法找到父层容器: {parent_name}")
                return False
            
            container = self._find_module(container_path)
            if not isinstance(container, (nn.Sequential, nn.ModuleList)):
                logger.error(f"父容器必须是 Sequential 或 ModuleList: {container_path}")
                return False
            
            # 执行插入
            success = self._insert_layer(container, parent_key, new_layer, insert_position)
            if success:
                # 记录修改
                self.modification_history.append({
                    'type': 'add_layer',
                    'parent_name': parent_name,
                    'layer_type': layer_type,
                    'layer_config': layer_config,
                    'insert_position': insert_position
                })
                logger.info(f"成功添加层: {layer_type} {layer_config}")
                
                # 如果启用安全检查，验证模型完整性
                if self.safety_checks:
                    self._validate_model_integrity()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"添加层失败: {e}")
            return False
    
    def remove_layer(self, layer_name: str, validate: bool = True) -> bool:
        """
        安全地移除层
        
        Args:
            layer_name: 要移除的层名称
            validate: 是否验证修改
            
        Returns:
            修改是否成功
        """
        try:
            if validate:
                if not self._validate_remove_layer(layer_name):
                    return False
            
            # 获取容器路径
            container_path, layer_key = self._get_container_path(layer_name)
            if container_path is None:
                logger.error(f"无法找到层容器: {layer_name}")
                return False
            
            container = self._find_module(container_path)
            if not isinstance(container, (nn.Sequential, nn.ModuleList)):
                logger.error(f"容器必须是 Sequential 或 ModuleList: {container_path}")
                return False
            
            # 检查是否是关键层
            if self._is_critical_layer(layer_name):
                logger.warning(f"尝试移除关键层: {layer_name}")
                if self.safety_checks:
                    logger.error("安全检查: 禁止移除关键层")
                    return False
            
            # 执行移除
            if isinstance(container, nn.Sequential):
                # Sequential 需要重建 OrderedDict
                new_modules = OrderedDict()
                for name, module in container._modules.items():
                    if name != layer_key:
                        new_modules[name] = module
                container._modules = new_modules
            else:  # ModuleList
                # 找到索引并移除
                for i, module in enumerate(container):
                    if i == int(layer_key):
                        del container[i]
                        break
            
            # 记录修改
            self.modification_history.append({
                'type': 'remove_layer',
                'layer_name': layer_name
            })
            logger.info(f"成功移除层: {layer_name}")
            
            # 验证完整性
            if self.safety_checks:
                self._validate_model_integrity()
            
            return True
            
        except Exception as e:
            logger.error(f"移除层失败: {e}")
            return False
    
    def adjust_layer_dimensions(self, 
                               layer_name: str,
                               new_dimensions: Dict[str, int],
                               validate: bool = True) -> bool:
        """
        调整层维度
        
        Args:
            layer_name: 层名称
            new_dimensions: 新维度配置
            validate: 是否验证修改
            
        Returns:
            修改是否成功
        """
        try:
            if validate:
                if not self._validate_dimension_adjustment(layer_name, new_dimensions):
                    return False
            
            layer = self._find_module(layer_name)
            if layer is None:
                logger.error(f"找不到层: {layer_name}")
                return False
            
            # 保存原始配置
            original_config = self._get_layer_config(layer)
            
            # 调整维度
            success = self._adjust_dimensions(layer, new_dimensions)
            if success:
                # 记录修改
                self.modification_history.append({
                    'type': 'adjust_dimensions',
                    'layer_name': layer_name,
                    'original_config': original_config,
                    'new_dimensions': new_dimensions
                })
                logger.info(f"成功调整层维度: {layer_name} -> {new_dimensions}")
                
                # 验证完整性
                if self.safety_checks:
                    self._validate_model_integrity()
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"调整维度失败: {e}")
            return False
    
    def add_skip_connection(self, 
                           from_layer: str,
                           to_layer: str,
                           connection_type: str = 'residual',
                           validate: bool = True) -> bool:
        """
        添加跳跃连接
        
        Args:
            from_layer: 起始层
            to_layer: 目标层
            connection_type: 连接类型 ('residual', 'dense', 'attention')
            validate: 是否验证
            
        Returns:
            是否成功
        """
        try:
            if validate:
                if not self._validate_skip_connection(from_layer, to_layer, connection_type):
                    return False
            
            # 查找层
            from_module = self._find_module(from_layer)
            to_module = self._find_module(to_layer)
            
            if from_module is None or to_module is None:
                logger.error(f"找不到层: {from_layer} 或 {to_layer}")
                return False
            
            # 创建跳跃连接
            skip_connection = self._create_skip_connection(
                from_module, to_module, connection_type
            )
            
            if skip_connection is None:
                logger.error(f"无法创建 {connection_type} 跳跃连接")
                return False
            
            # 插入跳跃连接
            container_path, to_key = self._get_container_path(to_layer)
            container = self._find_module(container_path)
            
            if isinstance(container, nn.Sequential):
                # 创建包含跳跃连接的新模块
                skip_module = SkipConnectionModule(
                    to_module, skip_connection, connection_type
                )
                container._modules[to_key] = skip_module
            else:
                logger.error("跳跃连接只能添加到 Sequential 容器中")
                return False
            
            # 记录修改
            self.modification_history.append({
                'type': 'add_skip_connection',
                'from_layer': from_layer,
                'to_layer': to_layer,
                'connection_type': connection_type
            })
            logger.info(f"成功添加跳跃连接: {from_layer} -> {to_layer} ({connection_type})")
            
            return True
            
        except Exception as e:
            logger.error(f"添加跳跃连接失败: {e}")
            return False
    
    def freeze_layers(self, 
                     layer_names: List[str],
                     freeze_bn: bool = True) -> bool:
        """
        冻结指定层
        
        Args:
            layer_names: 要冻结的层名列表
            freeze_bn: 是否冻结批归一化层
            
        Returns:
            是否成功
        """
        try:
            frozen_count = 0
            
            for name, module in self.model.named_modules():
                # 检查是否匹配任何模式
                should_freeze = any(self._match_pattern(name, pattern) 
                                  for pattern in layer_names)
                
                if should_freeze:
                    # 冻结参数
                    for param in module.parameters():
                        param.requires_grad = False
                    
                    # 特殊处理批归一化层
                    if freeze_bn and isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        module.eval()
                        module.track_running_stats = False
                    
                    frozen_count += 1
            
            if frozen_count > 0:
                self.modification_history.append({
                    'type': 'freeze_layers',
                    'layer_names': layer_names,
                    'freeze_bn': freeze_bn,
                    'frozen_count': frozen_count
                })
                logger.info(f"成功冻结 {frozen_count} 个层")
                return True
            else:
                logger.warning("没有找到匹配的层进行冻结")
                return False
                
        except Exception as e:
            logger.error(f"冻结层失败: {e}")
            return False
    
    def unfreeze_layers(self, layer_names: List[str]) -> bool:
        """
        解冻指定层
        
        Args:
            layer_names: 要解冻的层名列表
            
        Returns:
            是否成功
        """
        try:
            unfrozen_count = 0
            
            for name, module in self.model.named_modules():
                should_unfreeze = any(self._match_pattern(name, pattern) 
                                    for pattern in layer_names)
                
                if should_unfreeze:
                    # 解冻参数
                    for param in module.parameters():
                        param.requires_grad = True
                    
                    # 恢复批归一化层
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        module.train()
                        module.track_running_stats = True
                    
                    unfrozen_count += 1
            
            if unfrozen_count > 0:
                self.modification_history.append({
                    'type': 'unfreeze_layers',
                    'layer_names': layer_names,
                    'unfrozen_count': unfrozen_count
                })
                logger.info(f"成功解冻 {unfrozen_count} 个层")
                return True
            else:
                logger.warning("没有找到匹配的层进行解冻")
                return False
                
        except Exception as e:
            logger.error(f"解冻层失败: {e}")
            return False
    
    def get_modification_summary(self) -> Dict[str, Any]:
        """获取修改摘要"""
        summary = {
            'total_modifications': len(self.modification_history),
            'modifications_by_type': {},
            'affected_layers': set(),
            'can_revert': len(self.modification_history) > 0
        }
        
        for mod in self.modification_history:
            mod_type = mod['type']
            summary['modifications_by_type'][mod_type] = \
                summary['modifications_by_type'].get(mod_type, 0) + 1
            
            # 收集受影响的层
            if 'layer_name' in mod:
                summary['affected_layers'].add(mod['layer_name'])
            if 'parent_name' in mod:
                summary['affected_layers'].add(mod['parent_name'])
            if 'from_layer' in mod:
                summary['affected_layers'].add(mod['from_layer'])
            if 'to_layer' in mod:
                summary['affected_layers'].add(mod['to_layer'])
        
        summary['affected_layers'] = list(summary['affected_layers'])
        return summary
    
    def revert_modifications(self) -> bool:
        """恢复所有修改"""
        try:
            # 恢复原始模型
            self.model.load_state_dict(self.original_model.state_dict())
            
            # 清空修改历史
            self.modification_history = []
            
            logger.info("已恢复所有架构修改")
            return True
            
        except Exception as e:
            logger.error(f"恢复修改失败: {e}")
            return False
    
    # 私有辅助方法
    def _validate_add_layer(self, parent_name: str, layer_type: str, config: Dict[str, Any]) -> bool:
        """验证添加层操作"""
        # 检查父层是否存在
        if self._find_module(parent_name) is None:
            logger.error(f"父层不存在: {parent_name}")
            return False
        
        # 检查层类型是否支持
        supported_types = ['conv', 'linear', 'batchnorm', 'dropout', 'activation', 'pooling']
        if layer_type not in supported_types:
            logger.error(f"不支持的层类型: {layer_type}")
            return False
        
        # 检查配置是否完整
        required_configs = {
            'conv': ['in_channels', 'out_channels', 'kernel_size'],
            'linear': ['in_features', 'out_features'],
            'batchnorm': ['num_features'],
            'dropout': ['p'],
            'activation': ['activation_type'],
            'pooling': ['pool_type', 'kernel_size']
        }
        
        required = required_configs.get(layer_type, [])
        missing = [req for req in required if req not in config]
        if missing:
            logger.error(f"缺少必需的配置参数: {missing}")
            return False
        
        return True
    
    def _validate_remove_layer(self, layer_name: str) -> bool:
        """验证移除层操作"""
        # 检查层是否存在
        if self._find_module(layer_name) is None:
            logger.error(f"层不存在: {layer_name}")
            return False
        
        # 检查是否是输入或输出层
        if layer_name in ['conv1', 'fc', 'classifier', 'head']:
            logger.warning(f"尝试移除可能的输入/输出层: {layer_name}")
            if self.safety_checks:
                logger.error("安全检查: 禁止移除输入/输出层")
                return False
        
        return True
    
    def _validate_dimension_adjustment(self, layer_name: str, new_dims: Dict[str, int]) -> bool:
        """验证维度调整"""
        layer = self._find_module(layer_name)
        if layer is None:
            logger.error(f"层不存在: {layer_name}")
            return False
        
        # 检查维度兼容性
        if isinstance(layer, nn.Conv2d):
            if 'out_channels' in new_dims:
                # 需要检查下一层的输入通道
                next_layer = self._find_next_layer(layer_name)
                if next_layer and hasattr(next_layer, 'in_channels'):
                    if next_layer.in_channels != new_dims['out_channels']:
                        logger.error("维度不匹配: 输出通道与下一层输入通道不符")
                        return False
        
        return True
    
    def _validate_skip_connection(self, from_layer: str, to_layer: str, conn_type: str) -> bool:
        """验证跳跃连接"""
        # 检查层是否存在
        if self._find_module(from_layer) is None or self._find_module(to_layer) is None:
            logger.error(f"层不存在: {from_layer} 或 {to_layer}")
            return False
        
        # 检查连接类型
        if conn_type not in ['residual', 'dense', 'attention']:
            logger.error(f"不支持的连接类型: {conn_type}")
            return False
        
        return True
    
    def _create_layer(self, layer_type: str, config: Dict[str, Any]) -> Optional[nn.Module]:
        """创建层"""
        try:
            if layer_type == 'conv':
                return nn.Conv2d(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    kernel_size=config['kernel_size'],
                    stride=config.get('stride', 1),
                    padding=config.get('padding', 0),
                    bias=config.get('bias', True)
                )
            elif layer_type == 'linear':
                return nn.Linear(
                    in_features=config['in_features'],
                    out_features=config['out_features'],
                    bias=config.get('bias', True)
                )
            elif layer_type == 'batchnorm':
                return nn.BatchNorm2d(config['num_features'])
            elif layer_type == 'dropout':
                return nn.Dropout2d(p=config['p'])
            elif layer_type == 'activation':
                act_type = config['activation_type']
                if act_type == 'relu':
                    return nn.ReLU(inplace=config.get('inplace', True))
                elif act_type == 'leaky_relu':
                    return nn.LeakyReLU(
                        negative_slope=config.get('negative_slope', 0.01),
                        inplace=config.get('inplace', True)
                    )
                elif act_type == 'gelu':
                    return nn.GELU()
            elif layer_type == 'pooling':
                pool_type = config['pool_type']
                if pool_type == 'max':
                    return nn.MaxPool2d(
                        kernel_size=config['kernel_size'],
                        stride=config.get('stride', config['kernel_size'])
                    )
                elif pool_type == 'avg':
                    return nn.AvgPool2d(
                        kernel_size=config['kernel_size'],
                        stride=config.get('stride', config['kernel_size'])
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"创建层失败: {e}")
            return None
    
    def _find_module(self, module_name: str) -> Optional[nn.Module]:
        """查找模块"""
        try:
            return dict(self.model.named_modules())[module_name]
        except KeyError:
            return None
    
    def _get_container_path(self, module_name: str) -> Tuple[Optional[str], Optional[str]]:
        """获取容器路径和键"""
        parts = module_name.split('.')
        if len(parts) < 2:
            return None, None
        
        container_path = '.'.join(parts[:-1])
        key = parts[-1]
        
        # 检查是否是数字索引
        if key.isdigit():
            key = int(key)
        
        return container_path, key
    
    def _insert_layer(self, 
                    container: Union[nn.Sequential, nn.ModuleList],
                    parent_key: Union[str, int],
                    new_layer: nn.Module,
                    position: str) -> bool:
        """插入层到容器"""
        try:
            if isinstance(container, nn.Sequential):
                new_modules = OrderedDict()
                inserted = False
                
                for name, module in container._modules.items():
                    if not inserted and position == 'before' and name == parent_key:
                        new_name = f"{name}_inserted_{len(new_modules)}"
                        new_modules[new_name] = new_layer
                        inserted = True
                    
                    new_modules[name] = module
                    
                    if not inserted and position == 'after' and name == parent_key:
                        new_name = f"{name}_inserted_{len(new_modules)}"
                        new_modules[new_name] = new_layer
                        inserted = True
                
                if not inserted:
                    new_modules[f"inserted_{len(new_modules)}"] = new_layer
                
                container._modules = new_modules
                
            elif isinstance(container, nn.ModuleList):
                parent_idx = int(parent_key) if isinstance(parent_key, str) and parent_key.isdigit() else parent_key
                
                if position == 'before':
                    container.insert(parent_idx, new_layer)
                else:
                    container.insert(parent_idx + 1, new_layer)
            
            return True
            
        except Exception as e:
            logger.error(f"插入层失败: {e}")
            return False
    
    def _adjust_dimensions(self, layer: nn.Module, new_dims: Dict[str, int]) -> bool:
        """调整层维度"""
        try:
            if isinstance(layer, nn.Conv2d):
                if 'out_channels' in new_dims:
                    # 创建新的卷积层
                    old_layer = layer
                    new_layer = nn.Conv2d(
                        in_channels=old_layer.in_channels,
                        out_channels=new_dims['out_channels'],
                        kernel_size=old_layer.kernel_size,
                        stride=old_layer.stride,
                        padding=old_layer.padding,
                        bias=old_layer.bias is not None
                    )
                    
                    # 复制权重
                    with torch.no_grad():
                        min_channels = min(old_layer.out_channels, new_dims['out_channels'])
                        new_layer.weight[:min_channels] = old_layer.weight[:min_channels]
                        if old_layer.bias is not None:
                            new_layer.bias[:min_channels] = old_layer.bias[:min_channels]
                    
                    # 替换层
                    self._replace_layer(layer, new_layer)
                    
            elif isinstance(layer, nn.Linear):
                if 'out_features' in new_dims:
                    old_layer = layer
                    new_layer = nn.Linear(
                        in_features=old_layer.in_features,
                        out_features=new_dims['out_features'],
                        bias=old_layer.bias is not None
                    )
                    
                    # 复制权重
                    with torch.no_grad():
                        min_features = min(old_layer.out_features, new_dims['out_features'])
                        new_layer.weight[:min_features] = old_layer.weight[:min_features]
                        if old_layer.bias is not None:
                            new_layer.bias[:min_features] = old_layer.bias[:min_features]
                    
                    self._replace_layer(layer, new_layer)
            
            return True
            
        except Exception as e:
            logger.error(f"调整维度失败: {e}")
            return False
    
    def _replace_layer(self, old_layer: nn.Module, new_layer: nn.Module):
        """替换层（需要通过父容器）"""
        # 这个实现需要在父容器中查找并替换
        # 简化实现，实际使用时可能需要更复杂的逻辑
        for name, module in self.model.named_modules():
            if module is old_layer:
                # 获取父容器
                container_path, key = self._get_container_path(name)
                if container_path:
                    container = self._find_module(container_path)
                    if isinstance(container, (nn.Sequential, nn.ModuleList)):
                        container._modules[key] = new_layer
                        break
    
    def _create_skip_connection(self, 
                               from_module: nn.Module,
                               to_module: nn.Module,
                               conn_type: str) -> Optional[nn.Module]:
        """创建跳跃连接"""
        if conn_type == 'residual':
            return ResidualConnection()
        elif conn_type == 'dense':
            return DenseConnection()
        elif conn_type == 'attention':
            return AttentionConnection(from_module, to_module)
        
        return None
    
    def _is_critical_layer(self, layer_name: str) -> bool:
        """检查是否是关键层"""
        critical_patterns = ['conv1', 'stem', 'head', 'classifier', 'fc']
        return any(pattern in layer_name.lower() for pattern in critical_patterns)
    
    def _match_pattern(self, name: str, pattern: str) -> bool:
        """检查名称是否匹配模式"""
        import re
        pattern = pattern.replace('.', '\.')
        pattern = pattern.replace('*', '.*')
        pattern = pattern.replace('?', '.')
        return re.fullmatch(pattern, name) is not None
    
    def _find_next_layer(self, layer_name: str) -> Optional[nn.Module]:
        """查找下一层"""
        # 简化实现，实际需要更复杂的逻辑
        return None
    
    def _get_layer_config(self, layer: nn.Module) -> Dict[str, Any]:
        """获取层配置"""
        config = {}
        
        if isinstance(layer, nn.Conv2d):
            config = {
                'type': 'conv2d',
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size,
                'stride': layer.stride,
                'padding': layer.padding
            }
        elif isinstance(layer, nn.Linear):
            config = {
                'type': 'linear',
                'in_features': layer.in_features,
                'out_features': layer.out_features
            }
        
        return config
    
    def _validate_model_integrity(self):
        """验证模型完整性"""
        try:
            # 执行前向传播测试
            dummy_input = torch.randn(1, 3, 70, 70)
            with torch.no_grad():
                output = self.model(dummy_input)
            
            # 检查输出
            if output is None:
                raise ValueError("模型输出为 None")
            
            # 检查参数梯度
            param_count = sum(p.numel() for p in self.model.parameters())
            grad_count = sum(p.grad is not None for p in self.model.parameters() if p.requires_grad)
            
            logger.debug(f"模型完整性检查通过 - 参数: {param_count}, 梯度: {grad_count}")
            
        except Exception as e:
            logger.error(f"模型完整性检查失败: {e}")
            if self.safety_checks:
                raise


# 辅助类
class SkipConnectionModule(nn.Module):
    """跳跃连接模块"""
    
    def __init__(self, main_module: nn.Module, skip_module: nn.Module, conn_type: str):
        super().__init__()
        self.main_module = main_module
        self.skip_module = skip_module
        self.conn_type = conn_type
    
    def forward(self, x):
        main_output = self.main_module(x)
        
        if self.conn_type == 'residual':
            # 残差连接需要维度匹配
            if main_output.shape == x.shape:
                return main_output + x
            else:
                return main_output
        elif self.conn_type == 'dense':
            # 密集连接
            return torch.cat([main_output, x], dim=1)
        else:
            return main_output


class ResidualConnection(nn.Module):
    """残差连接"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, residual):
        if x.shape == residual.shape:
            return x + residual
        else:
            # 维度不匹配时使用投影
            return x + self._project_residual(x, residual)
    
    def _project_residual(self, x, residual):
        # 简单的投影实现
        if len(x.shape) == 4:  # CNN
            if residual.shape[1] != x.shape[1]:
                # 调整通道数
                residual = F.adaptive_avg_pool2d(residual, x.shape[2:])
                residual = F.conv2d(residual, torch.eye(x.shape[1], residual.shape[1]).view(x.shape[1], residual.shape[1], 1, 1))
        return residual


class DenseConnection(nn.Module):
    """密集连接"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, new_features):
        return torch.cat([x, new_features], dim=1)


class AttentionConnection(nn.Module):
    """注意力连接"""
    
    def __init__(self, from_module: nn.Module, to_module: nn.Module):
        super().__init__()
        # 简化的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(from_module.out_features if hasattr(from_module, 'out_features') else 512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, context):
        # 计算注意力权重
        attn_weights = torch.sigmoid(self.attention(context.mean(dim=[2, 3]) if len(context.shape) == 4 else context))
        return x * attn_weights


# 工厂函数
def create_architecture_modifier(model: nn.Module, safety_checks: bool = True) -> ArchitectureModifier:
    """创建架构修改器的工厂函数"""
    return ArchitectureModifier(model, safety_checks)