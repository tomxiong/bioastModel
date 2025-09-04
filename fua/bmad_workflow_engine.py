"""
FUA Bmad工作流引擎
实现Build-Measure-Analyze-Decide循环的自动化工作流
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging
import numpy as np

from fua.dataset_iteration_manager import DatasetVersionManager, DatasetIncrementalUpdater, DatasetAnalyzer
from fua.parameter_optimizer import ParameterHistoryManager, ParameterOptimizer
from fua.training_pipeline import TrainingPipeline, PipelineManager
from fua.validation_engine import ValidationEngine, ModelComparator, ImprovementAnalyzer


class BmadWorkflowEngine:
    """Bmad工作流引擎"""
    
    def __init__(self, config_path: str = "fua/bmad_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 初始化组件
        self.dataset_manager = DatasetVersionManager()
        self.dataset_updater = DatasetIncrementalUpdater()
        self.dataset_analyzer = DatasetAnalyzer()
        self.param_history = ParameterHistoryManager()
        self.training_pipeline = TrainingPipeline()
        self.validation_engine = ValidationEngine()
        self.improvement_analyzer = ImprovementAnalyzer()
        
        # 创建工作流存储目录
        self.workflow_dir = Path(self.config["workflow_storage_path"])
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        
        # 工作流状态
        self.workflows = {}
        self.current_workflow = None
        
        # 设置日志
        self._setup_logging()
    
    def _load_config(self) -> Dict:
        """加载配置"""
        default_config = {
            "workflow_storage_path": "fua/workflows",
            "max_iterations": 10,
            "improvement_threshold": 0.02,
            "validation_datasets": {
                "primary": "bioast_dataset/test",
                "external": "external_validation_set"
            },
            "auto_decision": True,
            "notification_settings": {
                "email": None,
                "webhook": None
            },
            "build_config": {
                "default_epochs": 50,
                "early_stopping": True,
                "save_checkpoints": True
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def _setup_logging(self):
        """设置日志"""
        log_path = self.workflow_dir / "bmad_workflow.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("BmadWorkflow")
    
    def create_workflow(self, name: str, model_name: str, 
                       initial_config: Dict = None) -> str:
        """创建新的Bmad工作流"""
        workflow_id = f"{name}_{int(time.time())}"
        
        workflow = {
            "workflow_id": workflow_id,
            "name": name,
            "model_name": model_name,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "current_phase": None,
            "iterations": [],
            "current_iteration": 0,
            "config": initial_config or {},
            "metrics": {
                "best_accuracy": 0,
                "current_accuracy": 0,
                "improvement_rate": 0
            }
        }
        
        self.workflows[workflow_id] = workflow
        self._save_workflow(workflow_id)
        
        self.logger.info(f"Created workflow: {workflow_id}")
        return workflow_id
    
    def start_workflow(self, workflow_id: str) -> bool:
        """启动工作流"""
        if workflow_id not in self.workflows:
            self.logger.error(f"Workflow not found: {workflow_id}")
            return False
        
        workflow = self.workflows[workflow_id]
        workflow["status"] = "running"
        self.current_workflow = workflow_id
        
        # 启动工作流执行
        self._execute_workflow(workflow_id)
        
        return True
    
    def _execute_workflow(self, workflow_id: str):
        """执行工作流主循环"""
        workflow = self.workflows[workflow_id]
        max_iterations = self.config["max_iterations"]
        
        while workflow["current_iteration"] < max_iterations:
            iteration_num = workflow["current_iteration"] + 1
            self.logger.info(f"Starting iteration {iteration_num} for workflow {workflow_id}")
            
            # 执行Bmad循环
            result = self._execute_bmad_cycle(workflow_id, iteration_num)
            
            # 记录迭代结果
            workflow["iterations"].append(result)
            workflow["current_iteration"] = iteration_num
            
            # 更新指标
            if result["success"]:
                workflow["metrics"]["current_accuracy"] = result["metrics"]["accuracy"]
                if workflow["metrics"]["current_accuracy"] > workflow["metrics"]["best_accuracy"]:
                    workflow["metrics"]["best_accuracy"] = workflow["metrics"]["current_accuracy"]
                
                # 计算改进率
                if iteration_num > 1:
                    prev_accuracy = workflow["iterations"][-2]["metrics"]["accuracy"]
                    improvement = (workflow["metrics"]["current_accuracy"] - prev_accuracy) / prev_accuracy
                    workflow["metrics"]["improvement_rate"] = improvement
            
            # 保存工作流状态
            self._save_workflow(workflow_id)
            
            # 检查是否继续
            if not self._should_continue(workflow, result):
                break
        
        # 工作流完成
        workflow["status"] = "completed"
        workflow["completed_at"] = datetime.now().isoformat()
        self._save_workflow(workflow_id)
        
        self.logger.info(f"Workflow {workflow_id} completed after {workflow['current_iteration']} iterations")
    
    def _execute_bmad_cycle(self, workflow_id: str, iteration_num: int) -> Dict:
        """执行单个Bmad循环"""
        workflow = self.workflows[workflow_id]
        cycle_result = {
            "iteration": iteration_num,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "phases": {},
            "metrics": {},
            "decision": None
        }
        
        try:
            # 1. Build阶段
            self.logger.info(f"Build phase for iteration {iteration_num}")
            build_result = self._build_phase(workflow_id, iteration_num)
            cycle_result["phases"]["build"] = build_result
            
            # 2. Measure阶段
            self.logger.info(f"Measure phase for iteration {iteration_num}")
            measure_result = self._measure_phase(workflow_id, build_result)
            cycle_result["phases"]["measure"] = measure_result
            cycle_result["metrics"] = measure_result["metrics"]
            
            # 3. Analyze阶段
            self.logger.info(f"Analyze phase for iteration {iteration_num}")
            analyze_result = self._analyze_phase(workflow_id, measure_result)
            cycle_result["phases"]["analyze"] = analyze_result
            
            # 4. Decide阶段
            self.logger.info(f"Decide phase for iteration {iteration_num}")
            decide_result = self._decide_phase(workflow_id, analyze_result)
            cycle_result["phases"]["decide"] = decide_result
            cycle_result["decision"] = decide_result
            
        except Exception as e:
            cycle_result["success"] = False
            cycle_result["error"] = str(e)
            self.logger.error(f"Error in iteration {iteration_num}: {e}")
        
        return cycle_result
    
    def _build_phase(self, workflow_id: str, iteration_num: int) -> Dict:
        """Build阶段 - 构建和训练模型"""
        workflow = self.workflows[workflow_id]
        
        # 准备训练参数
        if iteration_num == 1:
            # 首次迭代使用初始配置或历史最佳
            params = self._get_initial_parameters(workflow)
        else:
            # 后续迭代基于分析结果调整参数
            params = self._get_adapted_parameters(workflow, iteration_num)
        
        # 提交训练任务
        job_id = self.training_pipeline.submit_job(
            model_name=workflow["model_name"],
            parameters=params,
            dataset_version=workflow.get("current_dataset_version")
        )
        
        # 等待训练完成（简化实现）
        while True:
            status = self.training_pipeline.get_job_status(job_id)
            if status["status"] in ["completed", "failed"]:
                break
            time.sleep(5)
        
        return {
            "job_id": job_id,
            "parameters": params,
            "status": status["status"],
            "result": status
        }
    
    def _measure_phase(self, workflow_id: str, build_result: Dict) -> Dict:
        """Measure阶段 - 测量性能"""
        workflow = self.workflows[workflow_id]
        
        metrics = {}
        
        if build_result["status"] == "completed" and build_result["result"].get("success"):
            # 获取训练指标
            metrics = build_result["result"].get("metrics", {})
            
            # 在验证集上测试
            if "model_path" in build_result["result"]:
                validation_datasets = self.config["validation_datasets"]
                
                for dataset_name, dataset_path in validation_datasets.items():
                    if Path(dataset_path).exists():
                        val_result = self.validation_engine.validate_model(
                            build_result["result"]["model_path"],
                            dataset_path,
                            workflow["model_name"],
                            dataset_name
                        )
                        metrics[f"validation_{dataset_name}"] = val_result["metrics"]
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def _analyze_phase(self, workflow_id: str, measure_result: Dict) -> Dict:
        """Analyze阶段 - 分析结果"""
        workflow = self.workflows[workflow_id]
        
        analysis = {
            "performance_analysis": {},
            "improvement_opportunities": [],
            "comparisons": {},
            "insights": []
        }
        
        metrics = measure_result["metrics"]
        
        # 性能趋势分析
        if workflow["current_iteration"] > 1:
            prev_metrics = workflow["iterations"][-1]["metrics"]
            for key in ["accuracy", "f1_score", "precision", "recall"]:
                if key in metrics and key in prev_metrics:
                    change = metrics[key] - prev_metrics[key]
                    analysis["performance_analysis"][f"{key}_change"] = change
                    
                    if change > 0:
                        analysis["insights"].append(f"{key}提升了{change:.4f}")
                    elif change < 0:
                        analysis["insights"].append(f"{key}下降了{abs(change):.4f}")
        
        # 改进机会分析
        if metrics.get("accuracy", 0) < 0.95:
            analysis["improvement_opportunities"].append({
                "area": "accuracy",
                "current_value": metrics.get("accuracy", 0),
                "target": 0.95,
                "suggestion": "考虑增加训练数据或调整模型结构"
            })
        
        # 与历史最佳对比
        best_accuracy = workflow["metrics"]["best_accuracy"]
        if metrics.get("accuracy", 0) > best_accuracy:
            analysis["comparisons"]["vs_best"] = "improved"
            analysis["insights"].append("创造了新的最佳性能！")
        else:
            analysis["comparisons"]["vs_best"] = "not_improved"
        
        return analysis
    
    def _decide_phase(self, workflow_id: str, analyze_result: Dict) -> Dict:
        """Decide阶段 - 决策下一步"""
        workflow = self.workflows[workflow_id]
        
        decision = {
            "continue": True,
            "next_actions": [],
            "parameter_adjustments": {},
            "data_augmentation_needed": False,
            "reasoning": []
        }
        
        metrics = workflow["metrics"]
        
        # 检查改进率
        if workflow["current_iteration"] > 1:
            improvement_rate = metrics["improvement_rate"]
            
            if improvement_rate < self.config["improvement_threshold"]:
                decision["continue"] = False
                decision["reasoning"].append(f"改进率{improvement_rate:.4f}低于阈值{self.config['improvement_threshold']}")
        
        # 检查是否达到目标
        if metrics["current_accuracy"] >= 0.98:
            decision["continue"] = False
            decision["reasoning"].append("已达到高精度目标")
        
        # 决定下一步行动
        if decision["continue"]:
            # 基于分析结果决定行动
            if any("accuracy" in opp.get("area", "") for opp in analyze_result["improvement_opportunities"]):
                decision["next_actions"].append("数据增强")
                decision["data_augmentation_needed"] = True
            
            # 参数调整建议
            if metrics["current_accuracy"] < 0.9:
                decision["parameter_adjustments"]["learning_rate"] = "reduce_by_factor_2"
                decision["parameter_adjustments"]["batch_size"] = "increase"
                decision["next_actions"].append("调整超参数")
        
        # 自动决策或人工决策
        if self.config["auto_decision"]:
            # 执行自动决策
            if decision["data_augmentation_needed"]:
                self._trigger_data_augmentation(workflow_id)
            
            if decision["parameter_adjustments"]:
                self._apply_parameter_adjustments(workflow_id, decision["parameter_adjustments"])
        else:
            decision["requires_human_input"] = True
        
        return decision
    
    def _get_initial_parameters(self, workflow: Dict) -> Dict:
        """获取初始参数"""
        # 尝试从历史获取最佳参数
        best_config = self.param_history.get_best_config(workflow["model_name"])
        if best_config:
            return best_config.get("parameters", {})
        
        # 使用默认参数
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": self.config["build_config"]["default_epochs"],
            "optimizer": "adam",
            "weight_decay": 0.0001
        }
    
    def _get_adapted_parameters(self, workflow: Dict, iteration_num: int) -> Dict:
        """获取调整后的参数"""
        # 使用参数优化器生成新参数
        optimizer = ParameterOptimizer(workflow["model_name"], self.param_history)
        return optimizer.suggest_parameters("adaptive")
    
    def _should_continue(self, workflow: Dict, cycle_result: Dict) -> bool:
        """判断是否继续迭代"""
        # 检查最大迭代次数
        if workflow["current_iteration"] >= self.config["max_iterations"]:
            return False
        
        # 检查最后一次决策
        if "decision" in cycle_result:
            return cycle_result["decision"].get("continue", True)
        
        return True
    
    def _trigger_data_augmentation(self, workflow_id: str):
        """触发数据增强"""
        # 这里可以实现数据增强逻辑
        self.logger.info(f"Triggering data augmentation for workflow {workflow_id}")
    
    def _apply_parameter_adjustments(self, workflow_id: str, adjustments: Dict):
        """应用参数调整"""
        self.logger.info(f"Applying parameter adjustments for workflow {workflow_id}: {adjustments}")
    
    def _save_workflow(self, workflow_id: str):
        """保存工作流状态"""
        workflow = self.workflows[workflow_id]
        workflow_file = self.workflow_dir / f"{workflow_id}.json"
        
        with open(workflow_file, 'w') as f:
            json.dump(workflow, f, indent=2)
    
    def get_workflow_status(self, workflow_id: str) -> Dict:
        """获取工作流状态"""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id].copy()
        
        # 移除一些内部细节
        if "iterations" in workflow:
            workflow["iteration_count"] = len(workflow["iterations"])
            workflow["last_iteration"] = workflow["iterations"][-1] if workflow["iterations"] else None
        
        return workflow
    
    def list_workflows(self) -> List[Dict]:
        """列出所有工作流"""
        return [
            {
                "workflow_id": wf_id,
                "name": wf["name"],
                "model_name": wf["model_name"],
                "status": wf["status"],
                "created_at": wf["created_at"],
                "current_iteration": wf["current_iteration"],
                "best_accuracy": wf["metrics"]["best_accuracy"]
            }
            for wf_id, wf in self.workflows.items()
        ]
    
    def generate_workflow_report(self, workflow_id: str) -> Dict:
        """生成工作流报告"""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        
        report = {
            "workflow_info": {
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "model_name": workflow["model_name"],
                "created_at": workflow["created_at"],
                "completed_at": workflow.get("completed_at"),
                "status": workflow["status"],
                "total_iterations": workflow["current_iteration"]
            },
            "performance_summary": workflow["metrics"],
            "iteration_details": [],
            "key_insights": [],
            "recommendations": []
        }
        
        # 分析每次迭代
        for iteration in workflow["iterations"]:
            if iteration["success"]:
                iter_summary = {
                    "iteration": iteration["iteration"],
                    "accuracy": iteration["metrics"].get("accuracy", 0),
                    "build_status": iteration["phases"]["build"]["status"],
                    "decision": iteration["decision"].get("continue", True)
                }
                report["iteration_details"].append(iter_summary)
        
        # 生成洞察
        if workflow["metrics"]["best_accuracy"] > 0.95:
            report["key_insights"].append("模型达到了优秀性能水平")
        
        if workflow["current_iteration"] >= self.config["max_iterations"]:
            report["recommendations"].append("考虑增加最大迭代次数以进一步优化")
        
        # 保存报告
        report_file = self.workflow_dir / f"{workflow_id}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


class BmadDashboard:
    """Bmad工作流仪表板"""
    
    def __init__(self, workflow_engine: BmadWorkflowEngine):
        self.engine = workflow_engine
    
    def get_dashboard_data(self) -> Dict:
        """获取仪表板数据"""
        workflows = self.engine.list_workflows()
        
        dashboard_data = {
            "summary": {
                "total_workflows": len(workflows),
                "running_workflows": len([w for w in workflows if w["status"] == "running"]),
                "completed_workflows": len([w for w in workflows if w["status"] == "completed"]),
                "average_accuracy": np.mean([w["best_accuracy"] for w in workflows]) if workflows else 0
            },
            "active_workflows": [w for w in workflows if w["status"] == "running"],
            "recent_workflows": sorted(workflows, key=lambda x: x["created_at"], reverse=True)[:5],
            "performance_leaderboard": sorted(
                [w for w in workflows if w["best_accuracy"] > 0],
                key=lambda x: x["best_accuracy"],
                reverse=True
            )[:10]
        }
        
        return dashboard_data


# 使用示例
if __name__ == "__main__":
    # 创建Bmad工作流引擎
    engine = BmadWorkflowEngine()
    
    # 创建工作流
    workflow_id = engine.create_workflow(
        name="colony_detection_optimization",
        model_name="resnet18",
        initial_config={
            "target_accuracy": 0.95,
            "max_iterations": 5
        }
    )
    
    print(f"Created workflow: {workflow_id}")
    
    # 启动工作流
    # engine.start_workflow(workflow_id)
    
    # 获取仪表板数据
    dashboard = BmadDashboard(engine)
    data = dashboard.get_dashboard_data()
    print(f"Dashboard data: {json.dumps(data, indent=2)}")