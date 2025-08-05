"""模型生命周期管理器

整合所有组件，提供统一的API接口。
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import threading

# 导入各个组件
from ..model_registry import ModelRegistry, ModelMetadata, VersionControl
from ..experiment_manager import ExperimentTracker, Experiment, ExperimentConfig
from ..dashboard import Dashboard, ReportGenerator, Visualizer
from ..workflow import WorkflowAutomation, TaskScheduler, ModelPipeline
from .config import Config
from .logger import get_logger


class ModelLifecycleManager:
    """模型生命周期管理器
    
    整合模型注册、实验管理、工作流自动化、任务调度和可视化仪表板，
    提供统一的模型生命周期管理接口。
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 base_dir: str = ".",
                 auto_start: bool = True):
        """
        初始化模型生命周期管理器
        
        Args:
            config: 配置对象
            base_dir: 基础目录
            auto_start: 是否自动启动服务
        """
        self.config = config or Config()
        self.base_dir = Path(base_dir)
        self.logger = get_logger(__name__)
        
        # 初始化各个组件
        self._init_components()
        
        # 服务状态
        self.is_running = False
        self._lock = threading.Lock()
        
        if auto_start:
            self.start_services()
    
    def _init_components(self):
        """初始化各个组件"""
        # 模型注册器
        self.model_registry = ModelRegistry(
            registry_file=str(self.base_dir / "registry" / "models.json")
        )
        
        # 实验跟踪器
        self.experiment_tracker = ExperimentTracker(
            storage_dir=str(self.base_dir / "experiments")
        )
        
        # 工作流自动化
        self.workflow_automation = WorkflowAutomation(
            storage_dir=str(self.base_dir / "workflows")
        )
        
        # 任务调度器
        self.task_scheduler = TaskScheduler(
            max_workers=self.config.get('scheduler.max_workers', 4),
            storage_dir=str(self.base_dir / "scheduler")
        )
        
        # 可视化仪表板
        self.dashboard = Dashboard(
            model_registry=self.model_registry,
            experiment_tracker=self.experiment_tracker,
            port=self.config.get('dashboard.port', 5000)
        )
        
        # 报告生成器
        self.report_generator = ReportGenerator(
            model_registry=self.model_registry,
            experiment_tracker=self.experiment_tracker,
            output_dir=str(self.base_dir / "reports")
        )
        
        # 可视化器
        self.visualizer = Visualizer(
            output_dir=str(self.base_dir / "visualizations")
        )
        
        self.logger.info("所有组件初始化完成")
    
    def start_services(self):
        """启动所有服务"""
        with self._lock:
            if self.is_running:
                self.logger.warning("服务已在运行")
                return
            
            try:
                # 启动任务调度器
                self.task_scheduler.start_scheduler()
                
                # 启动仪表板
                self.dashboard.start_server()
                
                self.is_running = True
                self.logger.info("所有服务已启动")
                
            except Exception as e:
                self.logger.error(f"启动服务失败: {e}")
                raise
    
    def stop_services(self):
        """停止所有服务"""
        with self._lock:
            if not self.is_running:
                self.logger.warning("服务未运行")
                return
            
            try:
                # 停止任务调度器
                self.task_scheduler.stop_scheduler()
                
                # 停止仪表板
                self.dashboard.stop_server()
                
                self.is_running = False
                self.logger.info("所有服务已停止")
                
            except Exception as e:
                self.logger.error(f"停止服务失败: {e}")
    
    # ==================== 模型管理 ====================
    
    def register_model(self, 
                      model_path: str,
                      name: Optional[str] = None,
                      description: str = "",
                      tags: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """注册新模型"""
        try:
            model_id = self.model_registry.register_model(
                model_path=model_path,
                name=name,
                description=description,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            self.logger.info(f"模型已注册: {name or model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"模型注册失败: {e}")
            raise
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """获取模型信息"""
        return self.model_registry.get_model(model_id)
    
    def list_models(self, 
                   tags: Optional[List[str]] = None,
                   status: Optional[str] = None) -> List[ModelMetadata]:
        """列出模型"""
        return self.model_registry.list_models(tags=tags, status=status)
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """搜索模型"""
        return self.model_registry.search_models(query)
    
    def update_model_performance(self, 
                               model_id: str,
                               metrics: Dict[str, float]):
        """更新模型性能"""
        self.model_registry.update_performance(model_id, metrics)
        self.logger.info(f"模型性能已更新: {model_id}")
    
    def create_model_version(self, 
                           model_id: str,
                           version_type: str = "minor",
                           changelog: str = "") -> str:
        """创建模型版本"""
        version_control = VersionControl()
        version = version_control.add_version(
            model_id=model_id,
            version_type=version_type,
            changelog=changelog
        )
        
        self.logger.info(f"模型版本已创建: {model_id} v{version}")
        return version
    
    # ==================== 实验管理 ====================
    
    def create_experiment(self, 
                         name: str,
                         model_id: str,
                         config: Dict[str, Any],
                         description: str = "") -> str:
        """创建实验"""
        try:
            exp_config = ExperimentConfig(
                model_id=model_id,
                dataset_path=config.get('dataset_path', ''),
                hyperparameters=config.get('hyperparameters', {}),
                training_config=config.get('training_config', {})
            )
            
            experiment_id = self.experiment_tracker.create_experiment(
                name=name,
                config=exp_config,
                description=description
            )
            
            self.logger.info(f"实验已创建: {name} ({experiment_id})")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"创建实验失败: {e}")
            raise
    
    def start_experiment(self, experiment_id: str):
        """开始实验"""
        self.experiment_tracker.start_experiment(experiment_id)
        self.logger.info(f"实验已开始: {experiment_id}")
    
    def log_epoch_result(self, 
                        experiment_id: str,
                        epoch: int,
                        metrics: Dict[str, float]):
        """记录训练轮次结果"""
        self.experiment_tracker.log_epoch_result(experiment_id, epoch, metrics)
    
    def complete_experiment(self, 
                          experiment_id: str,
                          final_metrics: Dict[str, float]):
        """完成实验"""
        self.experiment_tracker.complete_experiment(experiment_id, final_metrics)
        self.logger.info(f"实验已完成: {experiment_id}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """获取实验信息"""
        return self.experiment_tracker.get_experiment(experiment_id)
    
    def list_experiments(self, 
                        model_id: Optional[str] = None,
                        status: Optional[str] = None) -> List[Experiment]:
        """列出实验"""
        return self.experiment_tracker.list_experiments(model_id=model_id, status=status)
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """比较实验"""
        return self.experiment_tracker.compare_experiments(experiment_ids)
    
    # ==================== 工作流管理 ====================
    
    def create_training_workflow(self, 
                               model_name: str,
                               dataset_path: str,
                               config: Dict[str, Any]) -> str:
        """创建训练工作流"""
        workflow_config = self.workflow_automation.create_model_training_workflow(
            model_name, dataset_path, config
        )
        
        self.logger.info(f"训练工作流已创建: {model_name}")
        return workflow_config.workflow_id
    
    def create_comparison_workflow(self, 
                                 model_ids: List[str],
                                 metrics: List[str]) -> str:
        """创建模型比较工作流"""
        workflow_config = self.workflow_automation.create_model_comparison_workflow(
            model_ids, metrics
        )
        
        self.logger.info(f"比较工作流已创建: {len(model_ids)} 个模型")
        return workflow_config.workflow_id
    
    def execute_workflow(self, workflow_id: str) -> str:
        """执行工作流"""
        execution_id = self.workflow_automation.execute_workflow_async(workflow_id)
        self.logger.info(f"工作流已开始执行: {workflow_id}")
        return execution_id
    
    def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        return self.workflow_automation.get_execution_status(execution_id)
    
    # ==================== 任务调度 ====================
    
    def schedule_training_task(self, 
                             model_name: str,
                             dataset_path: str,
                             config: Dict[str, Any],
                             schedule_config: Optional[Dict[str, Any]] = None) -> str:
        """调度训练任务"""
        task_id = self.task_scheduler.create_model_training_task(
            model_name, dataset_path, config, schedule_config
        )
        
        self.logger.info(f"训练任务已调度: {model_name}")
        return task_id
    
    def schedule_evaluation_task(self, 
                               model_ids: List[str],
                               schedule_config: Optional[Dict[str, Any]] = None) -> str:
        """调度评估任务"""
        task_id = self.task_scheduler.create_model_evaluation_task(
            model_ids, schedule_config
        )
        
        self.logger.info(f"评估任务已调度: {len(model_ids)} 个模型")
        return task_id
    
    def schedule_cleanup_task(self, 
                            days_to_keep: int = 30,
                            schedule_config: Optional[Dict[str, Any]] = None) -> str:
        """调度清理任务"""
        task_id = self.task_scheduler.create_cleanup_task(
            days_to_keep, schedule_config
        )
        
        self.logger.info(f"清理任务已调度: 保留 {days_to_keep} 天")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        return self.task_scheduler.get_task_status(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        result = self.task_scheduler.cancel_task(task_id)
        if result:
            self.logger.info(f"任务已取消: {task_id}")
        return result
    
    # ==================== 报告生成 ====================
    
    def generate_experiment_report(self, 
                                 experiment_id: str,
                                 format: str = "html") -> str:
        """生成实验报告"""
        if format == "json":
            report_path = self.report_generator.generate_experiment_report_json(experiment_id)
        elif format == "markdown":
            report_path = self.report_generator.generate_experiment_report_markdown(experiment_id)
        else:  # html
            report_path = self.report_generator.generate_experiment_report_html(experiment_id)
        
        self.logger.info(f"实验报告已生成: {report_path}")
        return report_path
    
    def generate_comparison_report(self, 
                                 experiment_ids: List[str],
                                 format: str = "html") -> str:
        """生成比较报告"""
        if format == "json":
            report_path = self.report_generator.generate_comparison_report_json(experiment_ids)
        elif format == "markdown":
            report_path = self.report_generator.generate_comparison_report_markdown(experiment_ids)
        else:  # html
            report_path = self.report_generator.generate_comparison_report_html(experiment_ids)
        
        self.logger.info(f"比较报告已生成: {report_path}")
        return report_path
    
    def generate_model_registry_report(self, format: str = "html") -> str:
        """生成模型注册表报告"""
        if format == "json":
            report_path = self.report_generator.generate_registry_report_json()
        elif format == "markdown":
            report_path = self.report_generator.generate_registry_report_markdown()
        else:  # html
            report_path = self.report_generator.generate_registry_report_html()
        
        self.logger.info(f"模型注册表报告已生成: {report_path}")
        return report_path
    
    # ==================== 可视化 ====================
    
    def plot_training_curves(self, 
                           experiment_id: str,
                           save_path: Optional[str] = None) -> str:
        """绘制训练曲线"""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"实验不存在: {experiment_id}")
        
        plot_path = self.visualizer.plot_training_curves(
            experiment.metrics.training_history,
            experiment.metrics.validation_history,
            title=f"Training Curves - {experiment.name}",
            save_path=save_path
        )
        
        self.logger.info(f"训练曲线已生成: {plot_path}")
        return plot_path
    
    def plot_model_comparison(self, 
                            experiment_ids: List[str],
                            metric: str = "accuracy",
                            save_path: Optional[str] = None) -> str:
        """绘制模型比较图"""
        experiments = [self.get_experiment(eid) for eid in experiment_ids]
        experiments = [exp for exp in experiments if exp is not None]
        
        if not experiments:
            raise ValueError("没有有效的实验")
        
        plot_path = self.visualizer.plot_model_comparison(
            experiments, metric, save_path
        )
        
        self.logger.info(f"模型比较图已生成: {plot_path}")
        return plot_path
    
    def create_interactive_dashboard(self, 
                                   experiment_ids: List[str],
                                   save_path: Optional[str] = None) -> str:
        """创建交互式仪表板"""
        experiments = [self.get_experiment(eid) for eid in experiment_ids]
        experiments = [exp for exp in experiments if exp is not None]
        
        dashboard_path = self.visualizer.create_interactive_dashboard(
            experiments, save_path
        )
        
        self.logger.info(f"交互式仪表板已创建: {dashboard_path}")
        return dashboard_path
    
    # ==================== 管道执行 ====================
    
    def create_training_pipeline(self, 
                               model_name: str,
                               config: Dict[str, Any]) -> ModelPipeline:
        """创建训练管道"""
        pipeline = ModelPipeline(
            name=f"training_{model_name}",
            description=f"训练管道: {model_name}"
        )
        
        # 添加标准训练阶段
        pipeline.add_stage("data_validation", "验证数据", config.get('data_config', {}))
        pipeline.add_stage("data_preprocessing", "数据预处理", config.get('preprocessing_config', {}))
        pipeline.add_stage("model_training", "模型训练", config.get('training_config', {}))
        pipeline.add_stage("model_evaluation", "模型评估", config.get('evaluation_config', {}))
        pipeline.add_stage("model_registration", "模型注册", {'model_name': model_name})
        
        self.logger.info(f"训练管道已创建: {model_name}")
        return pipeline
    
    def execute_pipeline(self, pipeline: ModelPipeline) -> str:
        """执行管道"""
        execution_id = pipeline.execute()
        self.logger.info(f"管道已开始执行: {pipeline.name}")
        return execution_id
    
    # ==================== 统计信息 ====================
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'models': self.model_registry.get_statistics(),
            'experiments': self.experiment_tracker.get_statistics(),
            'workflows': self.workflow_automation.get_statistics(),
            'tasks': self.task_scheduler.get_statistics(),
            'system_status': {
                'is_running': self.is_running,
                'dashboard_url': f"http://localhost:{self.config.get('dashboard.port', 5000)}",
                'base_dir': str(self.base_dir)
            }
        }
    
    def get_dashboard_url(self) -> str:
        """获取仪表板URL"""
        port = self.config.get('dashboard.port', 5000)
        return f"http://localhost:{port}"
    
    # ==================== 上下文管理器 ====================
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_services()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_services()
    
    # ==================== 便捷方法 ====================
    
    def quick_train_model(self, 
                         model_name: str,
                         dataset_path: str,
                         config: Dict[str, Any],
                         auto_register: bool = True) -> Dict[str, str]:
        """快速训练模型"""
        # 创建实验
        experiment_id = self.create_experiment(
            name=f"Training {model_name}",
            model_id=model_name,
            config=config,
            description=f"快速训练 {model_name}"
        )
        
        # 创建并执行训练管道
        pipeline = self.create_training_pipeline(model_name, config)
        execution_id = self.execute_pipeline(pipeline)
        
        # 生成报告
        report_path = self.generate_experiment_report(experiment_id, "html")
        
        result = {
            'experiment_id': experiment_id,
            'execution_id': execution_id,
            'report_path': report_path
        }
        
        if auto_register:
            # 自动注册模型（假设训练完成后模型保存在指定路径）
            model_path = config.get('model_output_path', f"models/{model_name}")
            if os.path.exists(model_path):
                model_id = self.register_model(
                    model_path=model_path,
                    name=model_name,
                    description=f"通过快速训练创建的模型",
                    tags=['quick_train']
                )
                result['model_id'] = model_id
        
        self.logger.info(f"快速训练已启动: {model_name}")
        return result
    
    def quick_compare_models(self, 
                           model_ids: List[str],
                           metrics: List[str] = None) -> Dict[str, str]:
        """快速比较模型"""
        if not metrics:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # 创建比较工作流
        workflow_id = self.create_comparison_workflow(model_ids, metrics)
        execution_id = self.execute_workflow(workflow_id)
        
        # 获取相关实验
        experiments = []
        for model_id in model_ids:
            model_experiments = self.list_experiments(model_id=model_id)
            if model_experiments:
                experiments.extend([exp.experiment_id for exp in model_experiments[:1]])  # 取最新的实验
        
        # 生成比较报告
        if experiments:
            report_path = self.generate_comparison_report(experiments, "html")
            plot_path = self.plot_model_comparison(experiments)
        else:
            report_path = ""
            plot_path = ""
        
        result = {
            'workflow_id': workflow_id,
            'execution_id': execution_id,
            'report_path': report_path,
            'plot_path': plot_path
        }
        
        self.logger.info(f"快速比较已启动: {len(model_ids)} 个模型")
        return result
    
    def setup_monitoring(self, 
                        cleanup_days: int = 30,
                        backup_enabled: bool = True,
                        backup_path: str = "backups") -> Dict[str, str]:
        """设置监控和维护任务"""
        task_ids = {}
        
        # 设置清理任务
        cleanup_task_id = self.schedule_cleanup_task(
            days_to_keep=cleanup_days,
            schedule_config={'weekday': 6, 'at': '02:00'}  # 周日凌晨2点
        )
        task_ids['cleanup'] = cleanup_task_id
        
        # 设置备份任务
        if backup_enabled:
            backup_task_id = self.task_scheduler.create_backup_task(
                backup_path=backup_path,
                schedule_config={'at': '01:00'}  # 每天凌晨1点
            )
            task_ids['backup'] = backup_task_id
        
        self.logger.info("监控和维护任务已设置")
        return task_ids