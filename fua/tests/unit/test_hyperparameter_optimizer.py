"""
FUA 超参数优化测试

测试超参数优化器的各项功能
"""

import unittest
import tempfile
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shutil
import json

# Import FUA components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import fua


class TestHyperparameterOptimizer(unittest.TestCase):
    """超参数优化器测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建简单的测试数据
        self.create_test_data()
        
        # 定义搜索空间
        self.search_space = {
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'hidden_size': {'type': 'categorical', 'choices': [32, 64]},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5}
        }
        
        # 创建优化器
        self.optimizer = fua.create_hyperparameter_optimizer(
            model_name='test_model',
            search_space=self.search_space,
            n_trials=5,  # 减少试验次数以加快测试
            direction='maximize',
            metric='accuracy'
        )
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def create_test_data(self):
        """创建测试数据"""
        # 创建简单的线性可分数据
        X = torch.randn(100, 70*70*3)
        y = (X.sum(dim=1) > 0).long()  # 简单的标签
        
        # 创建数据集
        dataset = TensorDataset(X, y)
        train_size = 80
        test_size = 20
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    def create_simple_model(self, hidden_size=64, dropout=0.1):
        """创建简单模型"""
        class SimpleModel(nn.Module):
            def __init__(self, input_size=70*70*3, hidden_size=64, num_classes=2, dropout=0.1):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden_size, num_classes)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return SimpleModel(hidden_size=hidden_size, dropout=dropout)
    
    def simple_train(self, model, train_loader, val_loader, params=None, trial=None, patience=5):
        """简单训练函数"""
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(10):  # 只训练10个epoch以加快测试
            # 训练
            model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            val_acc = correct / total
            
            # 报告给Optuna
            if trial is not None:
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return model, None, epoch
    
    def simple_evaluate(self, model, val_loader):
        """简单评估函数"""
        model.eval()
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return {
            'accuracy': correct / total,
            'loss': total_loss / len(val_loader)
        }
    
    def test_optimizer_creation(self):
        """测试优化器创建"""
        print("\n测试优化器创建...")
        
        # 验证优化器属性
        self.assertEqual(self.optimizer.model_name, 'test_model')
        self.assertEqual(self.optimizer.search_space, self.search_space)
        self.assertEqual(self.optimizer.n_trials, 5)
        self.assertIsNotNone(self.optimizer.study)
    
    def test_optimization_execution(self):
        """测试优化执行"""
        print("\n测试优化执行...")
        
        # 执行优化
        result = self.optimizer.optimize(
            train_data=self.train_loader,
            val_data=self.test_loader,
            model_factory=self.create_simple_model,
            train_fn=self.simple_train,
            eval_fn=self.simple_evaluate,
            save_study=False
        )
        
        # 验证结果
        self.assertIsInstance(result, fua.optimization.hyperparameter_optimizer.OptimizationResult)
        self.assertIsInstance(result.best_params, dict)
        self.assertIsInstance(result.best_score, float)
        self.assertEqual(result.total_trials, 5)
        self.assertGreater(result.optimization_time, 0)
    
    def test_parameter_suggestion(self):
        """测试参数建议"""
        print("\n测试参数建议...")
        
        # 创建一个试验
        trial = self.optimizer.study.ask()
        
        # 验证参数被正确采样
        params = trial.params
        for param_name in self.search_space.keys():
            self.assertIn(param_name, params)
        
        # 验证参数范围
        self.assertGreaterEqual(params['learning_rate'], self.search_space['learning_rate']['low'])
        self.assertLessEqual(params['learning_rate'], self.search_space['learning_rate']['high'])
        self.assertIn(params['hidden_size'], self.search_space['hidden_size']['choices'])
        self.assertGreaterEqual(params['dropout'], self.search_space['dropout']['low'])
        self.assertLessEqual(params['dropout'], self.search_space['dropout']['high'])
    
    def test_study_persistence(self):
        """测试研究持久化"""
        print("\n测试研究持久化...")
        
        # 设置存储
        storage_path = f"sqlite:///{self.temp_dir}/test_study.db"
        
        # 创建带存储的优化器
        optimizer = fua.create_hyperparameter_optimizer(
            model_name='persistent_model',
            search_space=self.search_space,
            n_trials=3,
            storage=storage_path
        )
        
        # 执行优化
        result = optimizer.optimize(
            train_data=self.train_loader,
            val_data=self.test_loader,
            model_factory=self.create_simple_model,
            train_fn=self.simple_train,
            eval_fn=self.simple_evaluate,
            save_study=False
        )
        
        # 验证数据库文件存在
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'test_study.db')))
    
    def test_default_search_spaces(self):
        """测试默认搜索空间"""
        print("\n测试默认搜索空间...")
        
        # 测试不同模型的默认搜索空间
        model_types = ['resnet', 'efficientnet', 'mobilenet', 'vit']
        
        for model_type in model_types:
            search_space = fua.get_default_search_space(model_type)
            self.assertIsInstance(search_space, dict)
            self.assertGreater(len(search_space), 0)
            
            # 验证学习率参数存在
            self.assertIn('learning_rate', search_space)
            self.assertEqual(search_space['learning_rate']['type'], 'float')
    
    def test_trial_results(self):
        """测试试验结果"""
        print("\n测试试验结果...")
        
        # 执行优化
        result = self.optimizer.optimize(
            train_data=self.train_loader,
            val_data=self.test_loader,
            model_factory=self.create_simple_model,
            train_fn=self.simple_train,
            eval_fn=self.simple_evaluate,
            save_study=False
        )
        
        # 获取试验结果DataFrame
        df = self.optimizer.get_trial_results_df()
        
        # 验证DataFrame
        self.assertIsInstance(df, type(pd.DataFrame()))
        self.assertEqual(len(df), 5)  # 5个试验
        self.assertIn('trial_number', df.columns)
        self.assertIn('score', df.columns)
        self.assertIn('params', df.columns)
    
    def test_result_analysis(self):
        """测试结果分析"""
        print("\n测试结果分析...")
        
        # 执行优化
        result = self.optimizer.optimize(
            train_data=self.train_loader,
            val_data=self.test_loader,
            model_factory=self.create_simple_model,
            train_fn=self.simple_train,
            eval_fn=self.simple_evaluate,
            save_study=False
        )
        
        # 分析结果
        analysis = self.optimizer.analyze_results()
        
        # 验证分析结果
        self.assertIsInstance(analysis, dict)
        self.assertIn('basic_stats', analysis)
        self.assertIn('performance_stats', analysis)
        self.assertIn('parameter_importance', analysis)
        
        # 验证基本统计
        basic_stats = analysis['basic_stats']
        self.assertEqual(basic_stats['total_trials'], 5)
        self.assertGreater(basic_stats['success_rate'], 0)
    
    def test_cv_optimizer(self):
        """测试交叉验证优化器"""
        print("\n测试交叉验证优化器...")
        
        # 创建交叉验证优化器
        cv_optimizer = fua.create_cv_optimizer(
            model_name='cv_test_model',
            search_space=self.search_space,
            n_trials=3,
            cv_folds=3
        )
        
        # 准备数据
        X = torch.randn(60, 70*70*3)
        y = (X.sum(dim=1) > 0).long()
        
        # 执行优化
        result = cv_optimizer.optimize(
            data=X,
            labels=y,
            model_factory=self.create_simple_model,
            train_fn=self.simple_train,
            eval_fn=self.simple_evaluate,
            save_study=False
        )
        
        # 验证结果
        self.assertIsInstance(result, fua.optimization.hyperparameter_optimizer.OptimizationResult)
        self.assertGreaterEqual(result.best_score, 0)
        self.assertLessEqual(result.best_score, 1)


class TestOptimizationIntegration(unittest.TestCase):
    """优化集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_optimization_with_real_model(self):
        """测试使用真实模型的优化"""
        print("\n测试真实模型优化...")
        
        # 导入optuna
        global optuna
        import optuna
        
        # 创建更复杂的数据
        X = torch.randn(200, 70*70*3)
        # 创建更复杂的标签（使用非线性边界）
        y = ((X[:, 0] * X[:, 1] + torch.randn(200) * 0.1) > 0).long()
        
        dataset = TensorDataset(X, y)
        train_size = 160
        val_size = 40
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 定义更复杂的搜索空间
        search_space = {
            'learning_rate': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'hidden_size': {'type': 'categorical', 'choices': [64, 128, 256]},
            'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd']}
        }
        
        # 创建更复杂的模型工厂
        def create_model(**params):
            class ComplexModel(nn.Module):
                def __init__(self, hidden_size=128, dropout=0.1):
                    super().__init__()
                    self.fc1 = nn.Linear(70*70*3, hidden_size)
                    self.bn1 = nn.BatchNorm1d(hidden_size)
                    self.dropout1 = nn.Dropout(dropout)
                    self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                    self.bn2 = nn.BatchNorm1d(hidden_size // 2)
                    self.dropout2 = nn.Dropout(dropout)
                    self.fc3 = nn.Linear(hidden_size // 2, 2)
                
                def forward(self, x):
                    x = x.view(x.size(0), -1)
                    x = torch.relu(self.bn1(self.fc1(x)))
                    x = self.dropout1(x)
                    x = torch.relu(self.bn2(self.fc2(x)))
                    x = self.dropout2(x)
                    x = self.fc3(x)
                    return x
            
            return ComplexModel(hidden_size=params['hidden_size'], dropout=params['dropout'])
        
        # 创建训练函数
        def train_fn(model, train_loader, val_loader, params=None, trial=None, patience=10):
            if params['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
            
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(20):
                # 训练
                model.train()
                train_loss = 0
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        output = model(data)
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                
                val_acc = correct / total
                
                # 报告
                if trial is not None:
                    trial.report(val_acc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                # 早停
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            return model, None, epoch
        
        # 创建评估函数
        def eval_fn(model, val_loader):
            model.eval()
            correct = 0
            total = 0
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    total_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            return {
                'accuracy': correct / total,
                'loss': total_loss / len(val_loader)
            }
        
        # 创建优化器并执行
        optimizer = fua.create_hyperparameter_optimizer(
            model_name='integration_test',
            search_space=search_space,
            n_trials=10,
            direction='maximize'
        )
        
        result = optimizer.optimize(
            train_data=train_loader,
            val_data=val_loader,
            model_factory=create_model,
            train_fn=train_fn,
            eval_fn=eval_fn,
            save_study=True,
            save_dir=self.temp_dir
        )
        
        # 验证结果
        self.assertGreater(result.best_score, 0.5)  # 应该能学到一些模式
        self.assertGreaterEqual(len(result.trial_history), 1)
        
        # 验证保存的文件
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'optimization_result.json')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'trial_history.csv')))
        
        # 验证最佳模型
        best_model = optimizer.get_best_model(create_model)
        self.assertIsInstance(best_model, nn.Module)


if __name__ == '__main__':
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # 导入pandas
    import pandas as pd
    global pd
    
    # 运行测试
    unittest.main(verbosity=2)