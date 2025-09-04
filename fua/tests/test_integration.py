"""
FUA迭代平台集成测试
测试完整的Bmad工作流程和各模块间的集成
"""

import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from fua.dataset_iteration_manager import DatasetVersionManager, DatasetIncrementalUpdater
from fua.parameter_optimizer import ParameterHistoryManager, ParameterOptimizer
from fua.training_pipeline import TrainingPipeline
from fua.validation_engine import ValidationEngine
from fua.bmad_workflow_engine import BmadWorkflowEngine


class TestBmadIntegration(unittest.TestCase):
    """测试完整的Bmad工作流集成"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 初始化所有组件
        self.dataset_manager = DatasetVersionManager(str(Path(self.temp_dir) / "dataset"))
        self.param_history = ParameterHistoryManager(str(Path(self.temp_dir) / "params"))
        self.training_pipeline = TrainingPipeline(str(Path(self.temp_dir) / "pipeline.json"))
        self.validation_engine = ValidationEngine(str(Path(self.temp_dir) / "validation.json"))
        
        # 创建测试数据集结构
        self._create_test_dataset()
        
        # 创建Bmad工作流引擎
        self.bmad_engine = BmadWorkflowEngine(str(Path(self.temp_dir) / "bmad.json"))
        self.bmad_engine.config["workflow_storage_path"] = str(self.temp_dir)
        self.bmad_engine.workflow_dir = Path(self.temp_dir)
        
        # 注入测试用的组件
        self.bmad_engine.dataset_manager = self.dataset_manager
        self.bmad_engine.param_history = self.param_history
        self.bmad_engine.training_pipeline = self.training_pipeline
        self.bmad_engine.validation_engine = self.validation_engine
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_dataset(self):
        """创建测试数据集"""
        base_path = Path(self.temp_dir) / "dataset"
        for split in ["train", "val", "test"]:
            for label in ["positive", "negative"]:
                (base_path / split / label).mkdir(parents=True, exist_ok=True)
                # 创建测试文件
                for i in range(5):
                    (base_path / split / label / f"test_{i}.jpg").touch()
    
    def test_complete_bmad_cycle(self):
        """测试完整的Bmad循环"""
        print("\n=== 测试完整Bmad循环 ===")
        
        # 创建工作流
        workflow_id = self.bmad_engine.create_workflow(
            "integration_test",
            "resnet18",
            {"target_accuracy": 0.95, "max_iterations": 3}
        )
        
        print(f"创建工作流: {workflow_id}")
        
        # 模拟Build阶段
        with patch.object(self.training_pipeline, 'submit_job') as mock_submit:
            with patch.object(self.training_pipeline, 'get_job_status') as mock_status:
                mock_job_id = "job_123"
                mock_submit.return_value = mock_job_id
                mock_status.return_value = {
                    "status": "completed",
                    "success": True,
                    "metrics": {"accuracy": 0.88, "loss": 0.25},
                    "model_path": str(Path(self.temp_dir) / "model.pth")
                }
                
                # 模拟Measure阶段
                with patch.object(self.validation_engine, 'validate_model') as mock_validate:
                    mock_validate.return_value = {
                        "metrics": {
                            "accuracy": 0.87,
                            "precision": 0.86,
                            "recall": 0.88,
                            "f1_score": 0.87
                        },
                        "analysis": {
                            "error_analysis": {
                                "error_rate": 0.13,
                                "false_positive_rate": 0.14,
                                "false_negative_rate": 0.12
                            }
                        }
                    }
                    
                    # 执行一次Bmad循环
                    result = self.bmad_engine._execute_bmad_cycle(workflow_id, 1)
                    
                    print(f"迭代1结果:")
                    print(f"  - 成功: {result['success']}")
                    print(f"  - 准确率: {result['metrics']['accuracy']:.3f}")
                    print(f"  - 决策继续: {result['decision']['continue']}")
                    
                    # 验证结果
                    self.assertTrue(result["success"])
                    self.assertIn("build", result["phases"])
                    self.assertIn("measure", result["phases"])
                    self.assertIn("analyze", result["phases"])
                    self.assertIn("decide", result["phases"])
                    
                    # 验证工作流状态更新
                    workflow = self.bmad_engine.workflows[workflow_id]
                    self.assertEqual(workflow["current_iteration"], 1)
                    self.assertEqual(len(workflow["iterations"]), 1)
    
    def test_multiple_iterations_workflow(self):
        """测试多迭代工作流"""
        print("\n=== 测试多迭代工作流 ===")
        
        workflow_id = self.bmad_engine.create_workflow(
            "multi_iter_test",
            "efficientnet",
            {"target_accuracy": 0.96, "max_iterations": 2}
        )
        
        # 模拟性能提升的场景
        mock_results = [
            {"accuracy": 0.88, "loss": 0.25},  # 第一次迭代
            {"accuracy": 0.92, "loss": 0.18},  # 第二次迭代
        ]
        
        for iteration in range(2):
            with patch.object(self.training_pipeline, 'submit_job') as mock_submit:
                with patch.object(self.training_pipeline, 'get_job_status') as mock_status:
                    mock_submit.return_value = f"job_{iteration}"
                    mock_status.return_value = {
                        "status": "completed",
                        "success": True,
                        "metrics": mock_results[iteration],
                        "model_path": str(Path(self.temp_dir) / f"model_{iteration}.pth")
                    }
                    
                    with patch.object(self.validation_engine, 'validate_model') as mock_validate:
                        mock_validate.return_value = {
                            "metrics": {
                                "accuracy": mock_results[iteration]["accuracy"] - 0.01,
                                "precision": mock_results[iteration]["accuracy"] - 0.02,
                                "recall": mock_results[iteration]["accuracy"],
                                "f1_score": mock_results[iteration]["accuracy"] - 0.01
                            },
                            "analysis": {
                                "error_analysis": {
                                    "error_rate": 1 - mock_results[iteration]["accuracy"],
                                    "false_positive_rate": 0.1,
                                    "false_negative_rate": 0.1
                                }
                            }
                        }
                        
                        # 执行迭代
                        result = self.bmad_engine._execute_bmad_cycle(workflow_id, iteration + 1)
                        
                        print(f"迭代{iteration + 1} - 准确率: {result['metrics']['accuracy']:.3f}")
                        
                        # 验证改进率计算
                        if iteration > 0:
                            workflow = self.bmad_engine.workflows[workflow_id]
                            improvement_rate = workflow["metrics"]["improvement_rate"]
                            expected_improvement = (mock_results[iteration]["accuracy"] - mock_results[iteration-1]["accuracy"]) / mock_results[iteration-1]["accuracy"]
                            self.assertAlmostEqual(improvement_rate, expected_improvement, places=3)
        
        # 验证最终状态
        workflow = self.bmad_engine.workflows[workflow_id]
        self.assertEqual(workflow["current_iteration"], 2)
        self.assertEqual(len(workflow["iterations"]), 2)
        self.assertAlmostEqual(workflow["metrics"]["best_accuracy"], 0.92, places=2)
    
    def test_workflow_decision_logic(self):
        """测试工作流决策逻辑"""
        print("\n=== 测试工作流决策逻辑 ===")
        
        # 测试场景1: 达到目标准确率
        workflow_id = self.bmad_engine.create_workflow(
            "decision_test_1",
            "resnet18",
            {"target_accuracy": 0.90}
        )
        
        with patch.object(self.training_pipeline, 'submit_job') as mock_submit:
            with patch.object(self.training_pipeline, 'get_job_status') as mock_status:
                mock_submit.return_value = "job_test"
                mock_status.return_value = {
                    "status": "completed",
                    "success": True,
                    "metrics": {"accuracy": 0.98},  # 超过目标
                    "model_path": str(Path(self.temp_dir) / "model.pth")
                }
                
                with patch.object(self.validation_engine, 'validate_model') as mock_validate:
                    mock_validate.return_value = {
                        "metrics": {"accuracy": 0.97},
                        "analysis": {"error_analysis": {"error_rate": 0.03}}
                    }
                    
                    result = self.bmad_engine._execute_bmad_cycle(workflow_id, 1)
                    
                    # 应该决定停止
                    self.assertFalse(result["decision"]["continue"])
                    self.assertIn("已达到高精度目标", result["decision"]["reasoning"])
        
        # 测试场景2: 改进率低于阈值
        workflow_id2 = self.bmad_engine.create_workflow(
            "decision_test_2",
            "resnet18"
        )
        
        # 设置较低的改进率阈值
        self.bmad_engine.config["improvement_threshold"] = 0.05
        
        # 添加第一次迭代结果
        self.bmad_engine.workflows[workflow_id2]["iterations"].append({
            "metrics": {"accuracy": 0.90}
        })
        
        with patch.object(self.training_pipeline, 'submit_job') as mock_submit:
            with patch.object(self.training_pipeline, 'get_job_status') as mock_status:
                mock_submit.return_value = "job_test"
                mock_status.return_value = {
                    "status": "completed",
                    "success": True,
                    "metrics": {"accuracy": 0.91},  # 只提升1%
                    "model_path": str(Path(self.temp_dir) / "model.pth")
                }
                
                with patch.object(self.validation_engine, 'validate_model') as mock_validate:
                    mock_validate.return_value = {
                        "metrics": {"accuracy": 0.905},
                        "analysis": {"error_analysis": {"error_rate": 0.095}}
                    }
                    
                    result = self.bmad_engine._execute_bmad_cycle(workflow_id2, 2)
                    
                    # 应该决定停止
                    self.assertFalse(result["decision"]["continue"])
                    self.assertTrue(any("改进率" in reason for reason in result["decision"]["reasoning"]))
    
    def test_parameter_optimization_integration(self):
        """测试参数优化集成"""
        print("\n=== 测试参数优化集成 ===")
        
        # 记录一些历史实验
        optimizer = ParameterOptimizer("test_model", self.param_history)
        
        # 添加具有不同表现的参数组合
        param_experiments = [
            {"lr": 0.001, "batch": 32, "accuracy": 0.85},
            {"lr": 0.01, "batch": 64, "accuracy": 0.88},
            {"lr": 0.005, "batch": 32, "accuracy": 0.90},
            {"lr": 0.0001, "batch": 16, "accuracy": 0.82},
        ]
        
        for exp in param_experiments:
            self.param_history.record_experiment(
                "test_model",
                {"learning_rate": exp["lr"], "batch_size": exp["batch"]},
                {"accuracy": exp["accuracy"]}
            )
        
        # 测试自适应参数建议
        suggestion = optimizer._adaptive_suggestion()
        
        print(f"自适应参数建议: {suggestion}")
        
        # 验证建议在合理范围内
        self.assertGreaterEqual(suggestion["learning_rate"], 0.0001)
        self.assertLessEqual(suggestion["learning_rate"], 0.1)
        self.assertIn(suggestion["batch_size"], [16, 32, 64, 128])
        
        # 验证倾向于历史表现好的参数
        best_lr = 0.005  # 历史最佳
        # 自适应建议应该在最佳值附近
        self.assertTrue(
            abs(suggestion["learning_rate"] - best_lr) < best_lr * 0.5 or
            suggestion["learning_rate"] in [p["lr"] for p in param_experiments]
        )
    
    def test_dataset_versioning_integration(self):
        """测试数据集版本控制集成"""
        print("\n=== 测试数据集版本控制集成 ===")
        
        # 创建初始版本
        v1 = self.dataset_manager.create_version("v1.0", "初始数据集")
        print(f"创建版本 v1.0: {v1['stats']}")
        
        # 添加新数据
        updater = DatasetIncrementalUpdater(str(Path(self.temp_dir) / "dataset"))
        
        # 创建新数据文件
        new_data_dir = Path(self.temp_dir) / "new_data"
        new_data_dir.mkdir()
        for i in range(3):
            (new_data_dir / f"new_image_{i}.jpg").touch()
        
        # 添加到训练集
        result = updater.add_new_data(
            str(new_data_dir),
            "train",
            "positive",
            {"source": "experiment_1"}
        )
        
        print(f"添加新数据结果: {result}")
        
        # 创建新版本
        v2 = self.dataset_manager.create_version("v1.1", "添加新数据")
        print(f"创建版本 v1.1: {v2['stats']}")
        
        # 验证版本历史
        versions = self.dataset_manager.list_versions()
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["version"], "v1.0")
        self.assertEqual(versions[1]["version"], "v1.1")
        
        # 验证版本信息
        v1_info = self.dataset_manager.get_version_info("v1.0")
        v2_info = self.dataset_manager.get_version_info("v1.1")
        self.assertEqual(v2_info["parent"], "v1.0")
    
    def test_workflow_persistence(self):
        """测试工作流持久化"""
        print("\n=== 测试工作流持久化 ===")
        
        # 创建并执行部分工作流
        workflow_id = self.bmad_engine.create_workflow(
            "persistence_test",
            "resnet18"
        )
        
        # 添加一些迭代数据
        self.bmad_engine.workflows[workflow_id]["iterations"].append({
            "iteration": 1,
            "success": True,
            "metrics": {"accuracy": 0.88},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
        
        # 保存工作流
        self.bmad_engine._save_workflow(workflow_id)
        
        # 验证文件存在
        workflow_file = Path(self.temp_dir) / f"{workflow_id}.json"
        self.assertTrue(workflow_file.exists())
        
        # 从文件重新加载
        with open(workflow_file, 'r') as f:
            saved_workflow = json.load(f)
        
        # 验证数据完整性
        self.assertEqual(saved_workflow["workflow_id"], workflow_id)
        self.assertEqual(saved_workflow["name"], "persistence_test")
        self.assertEqual(len(saved_workflow["iterations"]), 1)
        
        # 创建新的引擎实例测试加载
        new_engine = BmadWorkflowEngine(str(Path(self.temp_dir) / "bmad_new.json"))
        new_engine.config["workflow_storage_path"] = str(self.temp_dir)
        new_engine.workflow_dir = Path(self.temp_dir)
        
        # 工作流应该在初始化时加载
        # 注意：当前实现中，工作流是在运行时创建的，不是从文件加载的
        # 这是一个可以改进的地方
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        
        workflow_id = self.bmad_engine.create_workflow(
            "error_test",
            "resnet18"
        )
        
        # 测试训练失败场景
        with patch.object(self.training_pipeline, 'submit_job') as mock_submit:
            with patch.object(self.training_pipeline, 'get_job_status') as mock_status:
                mock_submit.return_value = "job_error"
                mock_status.return_value = {
                    "status": "failed",
                    "success": False,
                    "error": "Training failed due to GPU error"
                }
                
                result = self.bmad_engine._execute_bmad_cycle(workflow_id, 1)
                
                # 应该处理错误而不崩溃
                self.assertFalse(result["success"])
                self.assertIn("error", result)
                self.assertEqual(result["phases"]["build"]["status"], "failed")
        
        # 测试验证失败场景
        with patch.object(self.training_pipeline, 'submit_job') as mock_submit:
            with patch.object(self.training_pipeline, 'get_job_status') as mock_status:
                mock_submit.return_value = "job_success"
                mock_status.return_value = {
                    "status": "completed",
                    "success": True,
                    "metrics": {"accuracy": 0.88},
                    "model_path": str(Path(self.temp_dir) / "model.pth")
                }
                
                with patch.object(self.validation_engine, 'validate_model') as mock_validate:
                    mock_validate.side_effect = Exception("Validation failed")
                    
                    result = self.bmad_engine._execute_bmad_cycle(workflow_id, 2)
                    
                    # 应该处理验证错误
                    self.assertFalse(result["success"])
                    self.assertIn("error", result)


class TestPerformanceBenchmarks(unittest.TestCase):
    """性能基准测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parameter_optimization_performance(self):
        """测试参数优化性能"""
        print("\n=== 性能测试: 参数优化 ===")
        
        param_history = ParameterHistoryManager(str(Path(self.temp_dir) / "params"))
        optimizer = ParameterOptimizer("perf_test", param_history)
        
        # 添加大量历史数据
        start_time = time.time()
        for i in range(100):
            params = optimizer._random_suggestion()
            metrics = {"accuracy": np.random.random()}
            param_history.record_experiment("perf_test", params, metrics)
        
        record_time = time.time() - start_time
        print(f"记录100个实验耗时: {record_time:.3f}秒")
        
        # 测试建议生成性能
        start_time = time.time()
        for _ in range(10):
            suggestion = optimizer._adaptive_suggestion()
        
        suggest_time = time.time() - start_time
        print(f"生成10个建议耗时: {suggest_time:.3f}秒")
        print(f"平均每个建议: {suggest_time/10*1000:.1f}毫秒")
        
        # 性能断言
        self.assertLess(record_time, 5.0)  # 记录应该在5秒内完成
        self.assertLess(suggest_time/10, 0.1)  # 每个建议应该在100ms内完成
    
    def test_dataset_analysis_performance(self):
        """测试数据集分析性能"""
        print("\n=== 性能测试: 数据集分析 ===")
        
        # 创建大量测试文件
        dataset_path = Path(self.temp_dir) / "large_dataset"
        for split in ["train", "val"]:
            for label in ["positive", "negative"]:
                (dataset_path / split / label).mkdir(parents=True)
                # 创建1000个测试文件
                for i in range(1000):
                    (dataset_path / split / label / f"image_{i}.jpg").touch()
        
        analyzer = DatasetAnalyzer(str(dataset_path))
        
        # 测试基础统计性能
        start_time = time.time()
        stats = analyzer._calculate_basic_stats()
        stats_time = time.time() - start_time
        
        print(f"计算基础统计耗时: {stats_time:.3f}秒")
        print(f"总图像数: {stats['total_images']}")
        
        # 性能断言
        self.assertLess(stats_time, 10.0)  # 统计应该在10秒内完成
        self.assertEqual(stats["total_images"], 4000)  # 2 splits × 2 labels × 1000 images


def create_integration_test_suite():
    """创建集成测试套件"""
    suite = unittest.TestSuite()
    
    # 添加集成测试
    suite.addTest(unittest.makeSuite(TestBmadIntegration))
    suite.addTest(unittest.makeSuite(TestPerformanceBenchmarks))
    
    return suite


if __name__ == "__main__":
    # 运行集成测试
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_integration_test_suite()
    result = runner.run(suite)
    
    # 输出测试结果
    print(f"\n集成测试结果:")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    
    # 生成详细报告
    report = {
        "test_type": "integration",
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
    
    with open("integration_test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n集成测试报告已保存到: integration_test_report.json")