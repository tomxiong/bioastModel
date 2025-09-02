"""
架构修改器测试
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fua.finetuning.architecture_modifier import ArchitectureModifier, create_architecture_modifier


class TestArchitectureModifier(unittest.TestCase):
    """架构修改器测试"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建简单的测试模型
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        )
        
        # 创建修改器
        self.modifier = create_architecture_modifier(self.model)
    
    def test_add_layer(self):
        """测试添加层"""
        print("\n测试添加层...")
        
        # 在第一个卷积层后添加 dropout
        success = self.modifier.add_layer(
            parent_name='0',
            layer_type='dropout',
            layer_config={'p': 0.5},
            insert_position='after'
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.model), 7)  # 原来有6层，现在有7层
        
        # 验证层类型
        self.assertIsInstance(self.model[1], nn.Dropout2d)
    
    def test_remove_layer(self):
        """测试移除层"""
        print("\n测试移除层...")
        
        # 移除一个 ReLU 层
        success = self.modifier.remove_layer('1')
        
        self.assertTrue(success)
        self.assertEqual(len(self.model), 5)  # 原来有6层，现在有5层
    
    def test_adjust_layer_dimensions(self):
        """测试调整层维度"""
        print("\n测试调整层维度...")
        
        # 调整最后一个全连接层的输出维度
        success = self.modifier.adjust_layer_dimensions(
            layer_name='5',
            new_dimensions={'out_features': 20}
        )
        
        self.assertTrue(success)
        self.assertEqual(self.model[5].out_features, 20)
    
    def test_freeze_layers(self):
        """测试冻结层"""
        print("\n测试冻结层...")
        
        # 冻结所有卷积层
        success = self.modifier.freeze_layers(['0', '2'])
        
        self.assertTrue(success)
        
        # 检查参数是否被冻结
        self.assertFalse(self.model[0].weight.requires_grad)
        self.assertFalse(self.model[2].weight.requires_grad)
        
        # ReLU 层应该仍然可训练
        self.assertTrue(self.model[1].weight.requires_grad if hasattr(self.model[1], 'weight') else True)
    
    def test_unfreeze_layers(self):
        """测试解冻层"""
        print("\n测试解冻层...")
        
        # 先冻结，再解冻
        self.modifier.freeze_layers(['0', '2'])
        success = self.modifier.unfreeze_layers(['0'])
        
        self.assertTrue(success)
        
        # 第一个卷积层应该被解冻
        self.assertTrue(self.model[0].weight.requires_grad)
        # 第二个卷积层应该仍然被冻结
        self.assertFalse(self.model[2].weight.requires_grad)
    
    def test_add_skip_connection(self):
        """测试添加跳跃连接"""
        print("\n测试添加跳跃连接...")
        
        # 创建更深的模型用于测试跳跃连接
        deep_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        
        modifier = create_architecture_modifier(deep_model)
        
        # 添加残差连接
        success = modifier.add_skip_connection(
            from_layer='0',
            to_layer='4',
            connection_type='residual'
        )
        
        self.assertTrue(success)
    
    def test_modification_summary(self):
        """测试修改摘要"""
        print("\n测试修改摘要...")
        
        # 执行一些修改
        self.modifier.add_layer(
            parent_name='0',
            layer_type='dropout',
            layer_config={'p': 0.5},
            insert_position='after'
        )
        self.modifier.freeze_layers(['2'])
        
        # 获取摘要
        summary = self.modifier.get_modification_summary()
        
        self.assertEqual(summary['total_modifications'], 2)
        self.assertIn('add_layer', summary['modifications_by_type'])
        self.assertIn('freeze_layers', summary['modifications_by_type'])
    
    def test_revert_modifications(self):
        """测试恢复修改"""
        print("\n测试恢复修改...")
        
        # 保存原始状态
        original_state = self.model.state_dict()
        
        # 执行修改
        self.modifier.add_layer(
            parent_name='0',
            layer_type='dropout',
            layer_config={'p': 0.5},
            insert_position='after'
        )
        self.modifier.adjust_layer_dimensions(
            layer_name='5',
            new_dimensions={'out_features': 20}
        )
        
        # 验证修改
        self.assertEqual(len(self.model), 7)
        self.assertEqual(self.model[6].out_features, 20)
        
        # 恢复修改
        success = self.modifier.revert_modifications()
        
        self.assertTrue(success)
        
        # 验证恢复
        self.assertEqual(len(self.model), 6)  # 回到原始层数
        self.assertEqual(self.model[5].out_features, 10)  # 回到原始维度
    
    def test_safety_checks(self):
        """测试安全检查"""
        print("\n测试安全检查...")
        
        # 创建带安全检查的修改器
        safe_modifier = create_architecture_modifier(self.model, safety_checks=True)
        
        # 尝试移除关键层（第一个卷积层）
        success = safe_modifier.remove_layer('0')
        
        # 应该失败，因为这是关键层
        self.assertFalse(success)
        
        # 模型应该保持不变
        self.assertEqual(len(self.model), 6)


if __name__ == '__main__':
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2)