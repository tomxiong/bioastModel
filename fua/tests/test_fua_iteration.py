"""
FUA迭代平台单元测试套件
测试数据集管理、参数优化、训练流水线、验证引擎和Bmad工作流
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import torch

# 导入被测试的模块
from fua.dataset_iteration_manager import (
    DatasetVersionManager, 
    DatasetIncrementalUpdater, 
    DatasetAnalyzer
)
from fua.parameter_optimizer import (
    ParameterHistoryManager, 
    ParameterOptimizer,
    ParameterVisualizer
)
from fua.training_pipeline import TrainingPipeline, PipelineManager
from fua.validation_engine import ValidationEngine, ModelComparator, ImprovementAnalyzer
from fua.bmad_workflow_engine import BmadWorkflowEngine, BmadDashboard


class TestDatasetVersionManager(unittest.TestCase):
    """测试数据集版本管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = DatasetVersionManager(str(self.temp_dir))
        
        # 创建测试数据集结构
        for split in ["train", "val", "test"]:
            for label in ["positive", "negative"]:
                (self.temp_dir / split / label).mkdir(parents=True, exist_ok=True)
                # 创建一些测试文件
                for i in range(5):
                    (self.temp_dir / split / label / f"test_{i}.jpg").touch()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_version(self):
        """测试创建版本"""
        version_info = self.manager.create_version("v1.0", "测试版本")
        
        self.assertEqual(version_info["version"], "v1.0")
        self.assertEqual(version_info["description"], "测试版本")
        self.assertIn("created_at", version_info)
        self.assertIn("stats", version_info)
        
        # 检查版本文件是否创建
        version_file = self.temp_dir / "versions" / "v1.0" / "metadata.json"
        self.assertTrue(version_file.exists())
    
    def test_get_version_info(self):
        """测试获取版本信息"""
        self.manager.create_version("v1.0", "测试版本")
        info = self.manager.get_version_info("v1.0")
        
        self.assertEqual(info["version"], "v1.0")
        self.assertEqual(info["description"], "测试版本")
    
    def test_list_versions(self):
        """测试列出所有版本"""
        self.manager.create_version("v1.0", "版本1")
        self.manager.create_version("v1.1", "版本2")
        
        versions = self.manager.list_versions()
        self.assertEqual(len(versions), 2)
    
    def test_calculate_dataset_stats(self):
        """测试计算数据集统计"""
        stats = self.manager._calculate_dataset_stats()
        
        self.assertIn("train", stats)
        self.assertIn("val", stats)
        self.assertIn("test", stats)
        self.assertEqual(stats["train"]["positive"], 5)


class TestDatasetIncrementalUpdater(unittest.TestCase):
    """测试数据集增量更新器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.updater = DatasetIncrementalUpdater(str(self.temp_dir))
        
        # 创建目标目录
        (self.temp_dir / "train" / "positive").mkdir(parents=True, exist_ok=True)
        
        # 创建测试源目录
        self.source_dir = self.temp_dir / "source"
        self.source_dir.mkdir()
        # Note: don't create test_file here to avoid conflicts
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_new_data_file(self):
        """测试添加单个文件"""
        # 创建测试文件
        test_file = self.source_dir / "single_test.jpg"
        test_file.write_text("test content")
        
        result = self.updater.add_new_data(
            str(test_file), 
            "train", 
            "positive",
            {"source": "test"}
        )
        
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["duplicates"], 0)
        
        # 检查文件是否复制
        target_files = list((self.temp_dir / "train" / "positive").glob("*.jpg"))
        self.assertEqual(len(target_files), 1)
    
    def test_add_new_data_directory(self):
        """测试添加目录"""
        # 创建多个测试文件（每个文件有不同的内容）
        for i in range(3):
            test_file = self.source_dir / f"test_{i}.jpg"
            test_file.write_text(f"unique content {i}")
        
        result = self.updater.add_new_data(
            str(self.source_dir), 
            "train", 
            "positive"
        )
        
        self.assertEqual(result["added"], 3)
    
    def test_duplicate_detection(self):
        """测试重复文件检测"""
        # 创建并添加一个文件
        test_file = self.source_dir / "duplicate_test.jpg"
        test_file.write_text("duplicate content")
        
        # 先添加文件
        self.updater.add_new_data(str(test_file), "train", "positive")
        
        # 尝试再次添加同一个文件
        result = self.updater.add_new_data(str(test_file), "train", "positive")
        
        self.assertEqual(result["added"], 0)
        self.assertEqual(result["duplicates"], 1)
    
    def test_analyze_dataset_gaps(self):
        """测试分析数据集缺口"""
        analysis = self.updater.analyze_dataset_gaps()
        
        self.assertIn("class_imbalance", analysis)
        self.assertIn("total_samples", analysis)
        self.assertIn("recommendations", analysis)


class TestDatasetAnalyzer(unittest.TestCase):
    """测试数据集分析器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.analyzer = DatasetAnalyzer(str(self.temp_dir))
        
        # 创建测试数据集
        for split in ["train", "val", "test"]:
            split_path = self.temp_dir / split
            for label in ["positive", "negative"]:
                label_path = split_path / label
                label_path.mkdir(parents=True, exist_ok=True)
                # 创建测试图像文件
                for i in range(10):
                    (label_path / f"image_{i}.jpg").touch()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_quality_report(self):
        """测试生成质量报告"""
        report = self.analyzer.generate_quality_report()
        
        self.assertIn("summary", report)
        self.assertIn("quality_issues", report)
        self.assertIn("statistics", report)
        self.assertIn("recommendations", report)
    
    def test_calculate_basic_stats(self):
        """测试计算基础统计"""
        stats = self.analyzer._calculate_basic_stats()
        
        self.assertIn("total_images", stats)
        self.assertIn("splits", stats)
        self.assertEqual(stats["total_images"], 60)  # 3 splits × 2 labels × 10 images


class TestParameterHistoryManager(unittest.TestCase):
    """测试参数历史管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ParameterHistoryManager(self.temp_dir)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_record_experiment(self):
        """测试记录实验"""
        params = {"learning_rate": 0.001, "batch_size": 32}
        metrics = {"accuracy": 0.95, "loss": 0.15}
        
        self.manager.record_experiment("resnet18", params, metrics, "v1.0")
        
        # 检查历史记录
        history = self.manager.get_parameter_history("resnet18")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["model_name"], "resnet18")
    
    def test_get_best_config(self):
        """测试获取最佳配置"""
        # 记录多个实验
        experiments = [
            {"params": {"lr": 0.001}, "metrics": {"accuracy": 0.90}},
            {"params": {"lr": 0.01}, "metrics": {"accuracy": 0.95}},
            {"params": {"lr": 0.0001}, "metrics": {"accuracy": 0.85}},
        ]
        
        for exp in experiments:
            self.manager.record_experiment(
                "test_model", 
                exp["params"], 
                exp["metrics"]
            )
        
        best_config = self.manager.get_best_config("test_model")
        self.assertEqual(best_config["parameters"]["lr"], 0.01)
    
    def test_analyze_parameter_importance(self):
        """测试分析参数重要性"""
        # 记录一些实验数据
        for lr in [0.001, 0.01, 0.0001]:
            for bs in [16, 32, 64]:
                params = {"learning_rate": lr, "batch_size": bs}
                # 模拟准确率与学习率的相关性
                accuracy = 0.8 + (lr - 0.001) * 10
                metrics = {"accuracy": accuracy}
                self.manager.record_experiment("test_model", params, metrics)
        
        importance = self.manager.analyze_parameter_importance("test_model")
        
        self.assertIn("learning_rate", importance)
        self.assertIn("batch_size", importance)


class TestParameterOptimizer(unittest.TestCase):
    """测试参数优化器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.history_manager = ParameterHistoryManager(self.temp_dir)
        self.optimizer = ParameterOptimizer("test_model", self.history_manager)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_random_suggestion(self):
        """测试随机参数建议"""
        suggestion = self.optimizer._random_suggestion()
        
        self.assertIn("learning_rate", suggestion)
        self.assertIn("batch_size", suggestion)
        self.assertIn("epochs", suggestion)
        self.assertIn("optimizer", suggestion)
        
        # 检查参数范围
        self.assertGreaterEqual(suggestion["learning_rate"], 0.0001)
        self.assertLessEqual(suggestion["learning_rate"], 0.1)
    
    def test_adaptive_suggestion(self):
        """测试自适应参数建议"""
        # 添加一些历史数据 - 确保有足够的变化
        param_data = [
            {"lr": 0.001, "batch": 32, "acc": 0.80},
            {"lr": 0.005, "batch": 32, "acc": 0.85},
            {"lr": 0.01, "batch": 64, "acc": 0.88},
            {"lr": 0.0005, "batch": 16, "acc": 0.82},
            {"lr": 0.002, "batch": 32, "acc": 0.87},
        ]
        
        for data in param_data:
            params = {
                "learning_rate": data["lr"],
                "batch_size": data["batch"],
                "epochs": 50,
                "optimizer": "adam"
            }
            metrics = {"accuracy": data["acc"]}
            self.history_manager.record_experiment("test_model", params, metrics)
        
        suggestion = self.optimizer._adaptive_suggestion()
        
        # 应该倾向于较高准确率的参数
        self.assertIsInstance(suggestion, dict)
        self.assertIn("learning_rate", suggestion)
    
    def test_bayesian_suggestion(self):
        """测试贝叶斯优化建议"""
        # 添加足够的历史数据
        for i in range(10):
            params = self.optimizer._random_suggestion()
            metrics = {"accuracy": np.random.random()}
            self.history_manager.record_experiment("test_model", params, metrics)
        
        suggestion = self.optimizer._bayesian_suggestion()
        
        self.assertIsInstance(suggestion, dict)
        self.assertIn("learning_rate", suggestion)
    
    def test_param_to_vector_conversion(self):
        """测试参数到向量的转换"""
        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "weight_decay": 0.0001,
            "optimizer": "adam"
        }
        
        vector = self.optimizer._param_to_vector(params)
        
        self.assertEqual(len(vector), 5)  # 5个参数
        self.assertEqual(vector[1], 32.0)  # batch_size
        self.assertEqual(vector[2], 50.0)  # epochs
        self.assertEqual(vector[4], 0)  # adam -> 0
    
    def test_vector_to_param_conversion(self):
        """测试向量到参数的转换"""
        vector = [-3, 32, 50, -4, 0]  # log10(0.001), 32, 50, log10(0.0001), adam
        
        params = self.optimizer._vector_to_param(vector)
        
        self.assertAlmostEqual(params["learning_rate"], 0.001, places=4)
        self.assertEqual(params["batch_size"], 32)
        self.assertEqual(params["epochs"], 50)
        self.assertEqual(params["optimizer"], "adam")


class TestTrainingPipeline(unittest.TestCase):
    """测试训练流水线"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = TrainingPipeline(str(Path(self.temp_dir) / "pipeline_config.json"))
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_submit_job(self, mock_run):
        """测试提交训练任务"""
        # 模拟子进程成功返回
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        job_id = self.pipeline.submit_job(
            "resnet18",
            {"learning_rate": 0.001, "epochs": 10},
            "v1.0"
        )
        
        self.assertIsInstance(job_id, str)
        self.assertTrue(job_id.startswith("job_"))
    
    def test_get_job_status(self):
        """测试获取任务状态"""
        job_id = self.pipeline.submit_job("test_model", {})
        
        status = self.pipeline.get_job_status(job_id)
        
        self.assertEqual(status["job_id"], job_id)
        self.assertIn("status", status)
    
    def test_prepare_experiment(self):
        """测试准备实验环境"""
        job = {
            "job_id": "test_job",
            "model_name": "test_model",
            "parameters": {"lr": 0.001}
        }
        
        experiment_dir = self.pipeline._prepare_experiment(job)
        
        self.assertTrue(experiment_dir.exists())
        self.assertTrue((experiment_dir / "job_config.json").exists())


class TestValidationEngine(unittest.TestCase):
    """测试验证引擎"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = ValidationEngine(str(self.temp_dir / "validation_config.json"))
        
        # 创建测试模型和数据集
        self.model_path = self.temp_dir / "test_model.pth"
        self.model_path.touch()
        
        self.dataset_path = self.temp_dir / "test_dataset"
        (self.dataset_path / "positive").mkdir(parents=True)
        (self.dataset_path / "negative").mkdir(parents=True)
        # 创建一些测试图像
        (self.dataset_path / "positive" / "img1.jpg").touch()
        (self.dataset_path / "negative" / "img2.jpg").touch()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('torch.load')
    def test_validate_model(self, mock_load):
        """测试模型验证"""
        # 模拟模型加载 - 返回一个包含所需键的字典
        mock_load.return_value = {
            "conv1.weight": torch.randn(64, 3, 3, 3),
            "conv1.bias": torch.randn(64),
            "fc.weight": torch.randn(2, 64),
            "fc.bias": torch.randn(2)
        }
        
        # 模拟推理结果
        with patch.object(self.validator, '_run_inference') as mock_infer:
            mock_infer.return_value = (
                np.array([1, 0]),  # predictions
                np.array([1, 0]),  # targets
                np.array([[0.2, 0.8], [0.7, 0.3]])  # probabilities
            )
            
            result = self.validator.validate_model(
                str(self.model_path),
                str(self.dataset_path),
                "test_model",
                "test_dataset"
            )
            
            self.assertIn("validation_id", result)
            self.assertIn("metrics", result)
            self.assertIn("analysis", result)
    
    def test_calculate_metrics(self):
        """测试指标计算"""
        predictions = np.array([1, 0, 1, 1])
        targets = np.array([1, 0, 0, 1])
        probabilities = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
        
        metrics = self.validator._calculate_metrics(predictions, targets, probabilities)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("confusion_matrix", metrics)
        
        # 验证计算结果
        self.assertEqual(metrics["true_positives"], 2)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)


class TestModelComparator(unittest.TestCase):
    """测试模型对比器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ValidationEngine(str(Path(self.temp_dir) / "validation_config.json"))
        self.comparator = ModelComparator(self.validator)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.object(ValidationEngine, 'validate_model')
    def test_compare_models(self, mock_validate):
        """测试模型对比"""
        # 模拟验证结果
        mock_validate.side_effect = [
            {
                "success": True,
                "metrics": {"accuracy": 0.95, "f1_score": 0.94},
                "validation_id": "val1"
            },
            {
                "success": True,
                "metrics": {"accuracy": 0.92, "f1_score": 0.91},
                "validation_id": "val2"
            }
        ]
        
        model_configs = [
            {"name": "model1", "path": "/path/to/model1.pth"},
            {"name": "model2", "path": "/path/to/model2.pth"}
        ]
        
        comparison = self.comparator.compare_models(model_configs, "/test/dataset")
        
        self.assertIn("comparison_id", comparison)
        self.assertIn("results", comparison)
        self.assertIn("report", comparison)
        
        # 验证报告内容
        report = comparison["report"]
        self.assertIn("summary", report)
        self.assertIn("rankings", report)
        self.assertEqual(report["summary"]["best_model"], "model1")


class TestImprovementAnalyzer(unittest.TestCase):
    """测试改进分析器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = ImprovementAnalyzer()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.object(ValidationEngine, 'validate_model')
    def test_analyze_improvement_opportunities(self, mock_validate):
        """测试改进机会分析"""
        # 模拟验证结果
        mock_validate.return_value = {
            "success": True,
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85
            },
            "analysis": {
                "error_analysis": {
                    "error_rate": 0.15,
                    "false_positive_rate": 0.12,
                    "false_negative_rate": 0.18
                }
            }
        }
        
        analysis = self.analyzer.analyze_improvement_opportunities(
            "/test/model.pth",
            "/test/dataset"
        )
        
        self.assertIn("improvement_areas", analysis)
        self.assertIn("data_suggestions", analysis)
        self.assertIn("parameter_suggestions", analysis)
        self.assertIn("priority_actions", analysis)
        
        # 验证改进建议
        self.assertTrue(len(analysis["improvement_areas"]) > 0)
        self.assertTrue(len(analysis["priority_actions"]) > 0)


class TestBmadWorkflowEngine(unittest.TestCase):
    """测试Bmad工作流引擎"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = BmadWorkflowEngine(str(self.temp_dir / "bmad_config.json"))
        
        # 修改配置使用临时目录
        self.engine.config["workflow_storage_path"] = str(self.temp_dir)
        self.engine.workflow_dir = self.temp_dir
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_workflow(self):
        """测试创建工作流"""
        workflow_id = self.engine.create_workflow(
            "test_workflow",
            "resnet18",
            {"target_accuracy": 0.95}
        )
        
        self.assertIsInstance(workflow_id, str)
        self.assertIn("test_workflow", workflow_id)
        
        # 验证工作流已保存
        self.assertIn(workflow_id, self.engine.workflows)
    
    @patch.object(TrainingPipeline, 'submit_job')
    @patch.object(TrainingPipeline, 'get_job_status')
    def test_build_phase(self, mock_status, mock_submit):
        """测试Build阶段"""
        # 模拟训练任务
        mock_job_id = "test_job_123"
        mock_submit.return_value = mock_job_id
        mock_status.return_value = {
            "status": "completed",
            "success": True,
            "metrics": {"accuracy": 0.92},
            "model_path": "/test/model.pth"
        }
        
        workflow_id = self.engine.create_workflow("test", "resnet18")
        result = self.engine._build_phase(workflow_id, 1)
        
        self.assertEqual(result["job_id"], mock_job_id)
        self.assertEqual(result["status"], "completed")
    
    def test_measure_phase(self):
        """测试Measure阶段"""
        build_result = {
            "status": "completed",
            "success": True,
            "result": {
                "metrics": {"accuracy": 0.92},
                "model_path": "/test/model.pth"
            }
        }
        
        workflow_id = self.engine.create_workflow("test", "resnet18")
        
        with patch.object(self.engine.validation_engine, 'validate_model') as mock_validate:
            mock_validate.return_value = {
                "metrics": {"validation_accuracy": 0.90}
            }
            
            result = self.engine._measure_phase(workflow_id, build_result)
            
            self.assertIn("metrics", result)
            self.assertIn("timestamp", result)
    
    def test_analyze_phase(self):
        """测试Analyze阶段"""
        measure_result = {
            "metrics": {"accuracy": 0.92, "f1_score": 0.91},
            "timestamp": "2024-01-01T00:00:00"
        }
        
        workflow_id = self.engine.create_workflow("test", "resnet18")
        # 添加前一次迭代的历史数据
        self.engine.workflows[workflow_id]["iterations"].append({
            "metrics": {"accuracy": 0.90, "f1_score": 0.89}
        })
        
        analysis = self.engine._analyze_phase(workflow_id, measure_result)
        
        self.assertIn("performance_analysis", analysis)
        self.assertIn("improvement_opportunities", analysis)
        self.assertIn("insights", analysis)
    
    def test_decide_phase(self):
        """测试Decide阶段"""
        analyze_result = {
            "improvement_opportunities": [
                {"area": "accuracy", "current_value": 0.85}
            ]
        }
        
        workflow_id = self.engine.create_workflow("test", "resnet18")
        self.engine.workflows[workflow_id]["metrics"]["improvement_rate"] = 0.01
        
        decision = self.engine._decide_phase(workflow_id, analyze_result)
        
        self.assertIn("continue", decision)
        self.assertIn("next_actions", decision)
        self.assertIn("parameter_adjustments", decision)
    
    def test_should_continue(self):
        """测试是否继续迭代"""
        workflow = {
            "current_iteration": 5,
            "metrics": {"current_accuracy": 0.99}
        }
        
        # 测试达到最大迭代次数
        self.engine.config["max_iterations"] = 5
        should_continue = self.engine._should_continue(workflow, {})
        self.assertFalse(should_continue)
        
        # 测试未达到最大迭代次数
        workflow["current_iteration"] = 3
        should_continue = self.engine._should_continue(workflow, {})
        self.assertTrue(should_continue)


class TestBmadDashboard(unittest.TestCase):
    """测试Bmad仪表板"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.engine = BmadWorkflowEngine(str(self.temp_dir / "bmad_config.json"))
        self.engine.config["workflow_storage_path"] = str(self.temp_dir)
        self.engine.workflow_dir = self.temp_dir
        self.dashboard = BmadDashboard(self.engine)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_dashboard_data(self):
        """测试获取仪表板数据"""
        # 创建一些测试工作流
        self.engine.create_workflow("workflow1", "resnet18")
        self.engine.create_workflow("workflow2", "efficientnet")
        
        data = self.dashboard.get_dashboard_data()
        
        self.assertIn("summary", data)
        self.assertIn("active_workflows", data)
        self.assertIn("recent_workflows", data)
        self.assertIn("performance_leaderboard", data)
        
        # 验证摘要信息
        summary = data["summary"]
        self.assertEqual(summary["total_workflows"], 2)


def create_test_suite():
    """创建测试套件"""
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestDatasetVersionManager,
        TestDatasetIncrementalUpdater,
        TestDatasetAnalyzer,
        TestParameterHistoryManager,
        TestParameterOptimizer,
        TestTrainingPipeline,
        TestValidationEngine,
        TestModelComparator,
        TestImprovementAnalyzer,
        TestBmadWorkflowEngine,
        TestBmadDashboard
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == "__main__":
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # 输出测试结果摘要
    print(f"\n测试结果摘要:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # 保存测试报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun,
        "failed_tests": [
            {"test": str(test), "error": str(error)} 
            for test, error in result.failures + result.errors
        ]
    }
    
    with open("test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细测试报告已保存到: test_report.json")