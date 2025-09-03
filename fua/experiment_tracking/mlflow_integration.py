"""
FUA MLflow Integration Module

提供MLflow实验跟踪和模型注册功能的集成
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class FUAExperimentTracker:
    """FUA实验跟踪器"""
    
    def __init__(self, 
                 tracking_uri: Optional[str] = None,
                 experiment_name: str = "fua_experiments",
                 registry_uri: Optional[str] = None):
        """
        初始化实验跟踪器
        
        Args:
            tracking_uri: MLflow跟踪服务器URI
            experiment_name: 实验名称
            registry_uri: 模型注册表URI
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        self.experiment_name = experiment_name
        self.registry_uri = registry_uri or self.tracking_uri
        
        # 设置MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        
        # 创建MLflow客户端
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        # 获取或创建实验
        self.experiment_id = self._get_or_create_experiment(experiment_name)
        
        # 当前运行ID
        self.current_run_id = None
        
        logger.info(f"MLflow Experiment Tracker initialized")
        logger.info(f"Tracking URI: {self.tracking_uri}")
        logger.info(f"Experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def _get_or_create_experiment(self, experiment_name: str) -> str:
        """获取或创建实验"""
        try:
            # 尝试获取实验
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                # 创建新实验
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
            return experiment_id
        except Exception as e:
            logger.error(f"Failed to get or create experiment: {e}")
            # 使用默认实验
            return "0"
    
    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None,
                  description: Optional[str] = None) -> str:
        """
        开始新的运行
        
        Args:
            run_name: 运行名称
            tags: 运行标签
            description: 运行描述
            
        Returns:
            run_id: 运行ID
        """
        try:
            if tags is None:
                tags = {}
            
            # 添加默认标签
            tags.update({
                "project": "FUA",
                "framework": "pytorch",
                "timestamp": datetime.now().isoformat()
            })
            
            # 开始运行
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags,
                description=description
            )
            
            self.current_run_id = run.info.run_id
            logger.info(f"Started MLflow run: {self.current_run_id}")
            
            return self.current_run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None
    
    def end_run(self, status: str = "FINISHED"):
        """结束当前运行"""
        if self.current_run_id:
            try:
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run: {self.current_run_id}")
                self.current_run_id = None
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """记录参数"""
        try:
            for key, value in params.items():
                self.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_param(self, key: str, value: Any):
        """记录单个参数"""
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.error(f"Failed to log parameter {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录指标"""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_model(self, 
                  model: nn.Module,
                  model_name: str,
                  input_example: Optional[torch.Tensor] = None,
                  signature: Optional[Any] = None):
        """记录PyTorch模型"""
        try:
            # 如果有input_example，转换为numpy数组
            if input_example is not None:
                input_example = input_example.detach().cpu().numpy()
            
            # 记录模型
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                input_example=input_example,
                signature=signature
            )
            
            logger.info(f"Logged model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_artifacts(self, 
                      artifact_dir: str,
                      artifact_path: Optional[str] = None):
        """记录工件"""
        try:
            mlflow.log_artifacts(artifact_dir, artifact_path)
            logger.info(f"Logged artifacts from: {artifact_dir}")
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
    
    def log_artifact(self, 
                     local_path: str,
                     artifact_path: Optional[str] = None):
        """记录单个工件"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_text(self, 
                 text: str,
                 artifact_file: str):
        """记录文本工件"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(text)
                temp_path = f.name
            
            self.log_artifact(temp_path, artifact_file)
            os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to log text: {e}")
    
    def log_dict(self, 
                 dictionary: Dict[str, Any],
                 artifact_file: str):
        """记录字典为JSON工件"""
        try:
            text = json.dumps(dictionary, indent=2)
            self.log_text(text, artifact_file)
        except Exception as e:
            logger.error(f"Failed to log dictionary: {e}")
    
    def set_tag(self, key: str, value: str):
        """设置标签"""
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.error(f"Failed to set tag: {e}")
    
    def delete_tag(self, key: str):
        """删除标签"""
        try:
            mlflow.delete_tag(key)
        except Exception as e:
            logger.error(f"Failed to delete tag: {e}")
    
    def get_run(self, run_id: str) -> Optional[Any]:
        """获取运行信息"""
        try:
            return self.client.get_run(run_id)
        except Exception as e:
            logger.error(f"Failed to get run: {e}")
            return None
    
    def search_runs(self, 
                    filter_string: str = "",
                    run_view_type: ViewType = ViewType.ACTIVE_ONLY,
                    max_results: int = 1000) -> List[Any]:
        """搜索运行"""
        try:
            return self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                run_view_type=run_view_type,
                max_results=max_results
            )
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def get_best_run(self, metric_name: str, ascending: bool = False) -> Optional[Any]:
        """获取最佳运行"""
        try:
            runs = self.search_runs()
            if not runs:
                return None
            
            # 根据指标排序
            sorted_runs = sorted(
                runs,
                key=lambda run: run.data.metrics.get(metric_name, float('-inf')),
                reverse=not ascending
            )
            
            return sorted_runs[0] if sorted_runs else None
            
        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None


class FUAModelRegistry:
    """FUA模型注册器"""
    
    def __init__(self, 
                 registry_uri: Optional[str] = None):
        """
        初始化模型注册器
        
        Args:
            registry_uri: 模型注册表URI
        """
        self.registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI", "mlruns")
        self.client = MlflowClient(registry_uri=self.registry_uri)
        
        logger.info(f"FUA Model Registry initialized")
        logger.info(f"Registry URI: {self.registry_uri}")
    
    def register_model(self,
                      model_name: str,
                      run_id: str,
                      model_path: str = "model",
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> str:
        """
        注册模型
        
        Args:
            model_name: 模型名称
            run_id: 运行ID
            model_path: 模型路径
            description: 模型描述
            tags: 模型标签
            
        Returns:
            model_version: 模型版本
        """
        try:
            # 检查并创建注册模型
            try:
                self.client.get_registered_model(model_name)
            except Exception:
                # 模型不存在，创建它
                self.client.create_registered_model(
                    name=model_name,
                    description=description or f"FUA model: {model_name}"
                )
            
            # 创建模型版本
            model_version = self.client.create_model_version(
                name=model_name,
                source=f"runs:/{run_id}/{model_path}",
                run_id=run_id
            )
            
            # 更新描述和标签
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=model_version.version,
                        key=key,
                        value=value
                    )
            
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def transition_model_version_stage(self,
                                      model_name: str,
                                      version: str,
                                      stage: str,
                                      archive_existing_versions: bool = False):
        """转换模型版本阶段"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model version stage: {e}")
    
    def delete_model_version(self, model_name: str, version: str):
        """删除模型版本"""
        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Deleted model version: {model_name} v{version}")
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
    
    def delete_model(self, model_name: str):
        """删除模型"""
        try:
            self.client.delete_registered_model(name=model_name)
            logger.info(f"Deleted model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
    
    def get_model_version(self, model_name: str, version: str) -> Optional[Any]:
        """获取模型版本信息"""
        try:
            return self.client.get_model_version(name=model_name, version=version)
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
    
    def get_latest_model_version(self, model_name: str, stage: str = None) -> Optional[Any]:
        """获取最新模型版本"""
        try:
            filter_string = f"name='{model_name}'"
            if stage:
                filter_string += f" and stage='{stage}'"
            
            model_versions = self.client.search_model_versions(filter_string)
            if model_versions:
                # 按版本号排序
                sorted_versions = sorted(
                    model_versions,
                    key=lambda mv: int(mv.version),
                    reverse=True
                )
                return sorted_versions[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None
    
    def list_models(self) -> List[Any]:
        """列出所有注册的模型"""
        try:
            return self.client.search_registered_models()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_model_uri(self, model_name: str, version: str = None) -> Optional[str]:
        """获取模型URI"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                # 获取最新版本
                latest_version = self.get_latest_model_version(model_name)
                if latest_version:
                    model_uri = f"models:/{model_name}/{latest_version.version}"
                else:
                    return None
            
            return model_uri
        except Exception as e:
            logger.error(f"Failed to get model URI: {e}")
            return None
    
    def load_model(self, model_name: str, version: str = None) -> Any:
        """加载模型"""
        try:
            model_uri = self.get_model_uri(model_name, version)
            if model_uri:
                return mlflow.pytorch.load_model(model_uri)
            return None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None


class FUAMLflowIntegration:
    """FUA MLflow集成主类"""
    
    def __init__(self,
                 tracking_uri: Optional[str] = None,
                 registry_uri: Optional[str] = None,
                 experiment_name: str = "fua_experiments"):
        """
        初始化FUA MLflow集成
        
        Args:
            tracking_uri: MLflow跟踪服务器URI
            registry_uri: 模型注册表URI
            experiment_name: 实验名称
        """
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.experiment_name = experiment_name
        
        # 初始化组件
        self.tracker = FUAExperimentTracker(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            registry_uri=registry_uri
        )
        
        self.registry = FUAModelRegistry(registry_uri=registry_uri)
        
        logger.info("FUA MLflow Integration initialized")
    
    def create_training_run(self,
                           model_name: str,
                           model_config: Dict[str, Any],
                           training_config: Dict[str, Any],
                           run_name: Optional[str] = None,
                           tags: Optional[Dict[str, str]] = None) -> str:
        """
        创建训练运行
        
        Args:
            model_name: 模型名称
            model_config: 模型配置
            training_config: 训练配置
            run_name: 运行名称
            tags: 运行标签
            
        Returns:
            run_id: 运行ID
        """
        try:
            # 创建运行名称
            if not run_name:
                run_name = f"{model_name}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 添加默认标签
            if tags is None:
                tags = {}
            tags.update({
                "model_name": model_name,
                "type": "training"
            })
            
            # 开始运行
            run_id = self.tracker.start_run(run_name=run_name, tags=tags)
            
            # 记录配置
            self.tracker.log_params({
                "model_config": json.dumps(model_config),
                "training_config": json.dumps(training_config)
            })
            
            # 记录模型特定参数
            for key, value in model_config.items():
                if isinstance(value, (str, int, float, bool)):
                    self.tracker.log_param(f"model_{key}", value)
            
            # 记录训练特定参数
            for key, value in training_config.items():
                if isinstance(value, (str, int, float, bool)):
                    self.tracker.log_param(f"training_{key}", value)
            
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to create training run: {e}")
            return None
    
    def log_training_metrics(self,
                            metrics: Dict[str, float],
                            step: Optional[int] = None):
        """记录训练指标"""
        try:
            self.tracker.log_metrics(metrics, step)
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")
    
    def log_model_and_register(self,
                               model: nn.Module,
                               model_name: str,
                               model_config: Dict[str, Any],
                               input_example: Optional[torch.Tensor] = None,
                               stage: str = "Staging"):
        """记录模型并注册"""
        try:
            # 记录模型
            self.tracker.log_model(
                model=model,
                model_name="model",
                input_example=input_example
            )
            
            # 注册模型
            run_id = self.tracker.current_run_id
            if run_id:
                version = self.registry.register_model(
                    model_name=model_name,
                    run_id=run_id,
                    model_path="model",
                    description=f"FUA model: {model_name}",
                    tags={
                        "framework": "pytorch",
                        "model_type": model_config.get("model_type", "unknown"),
                        "input_size": str(model_config.get("input_size", ())),
                        "num_classes": str(model_config.get("num_classes", 2))
                    }
                )
                
                # 转换到指定阶段
                if version and stage != "None":
                    self.registry.transition_model_version_stage(
                        model_name=model_name,
                        version=version,
                        stage=stage
                    )
                
                return version
            
        except Exception as e:
            logger.error(f"Failed to log and register model: {e}")
            return None
    
    def complete_training_run(self,
                             final_metrics: Dict[str, float],
                             artifacts_dir: Optional[str] = None):
        """完成训练运行"""
        try:
            # 记录最终指标
            self.tracker.log_metrics(final_metrics)
            
            # 记录工件
            if artifacts_dir and os.path.exists(artifacts_dir):
                self.tracker.log_artifacts(artifacts_dir)
            
            # 结束运行
            self.tracker.end_run(status="FINISHED")
            
        except Exception as e:
            logger.error(f"Failed to complete training run: {e}")
            self.tracker.end_run(status="FAILED")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """记录单个工件"""
        try:
            self.tracker.log_artifact(local_path, artifact_path)
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """获取实验摘要"""
        try:
            runs = self.tracker.search_runs()
            
            summary = {
                "experiment_name": self.experiment_name,
                "total_runs": len(runs),
                "active_runs": len([r for r in runs if r.info.status == "RUNNING"]),
                "finished_runs": len([r for r in runs if r.info.status == "FINISHED"]),
                "failed_runs": len([r for r in runs if r.info.status == "FAILED"]),
                "best_runs": {}
            }
            
            # 获取各指标的最佳运行
            metrics_names = set()
            for run in runs:
                metrics_names.update(run.data.metrics.keys())
            
            for metric in metrics_names:
                best_run = self.tracker.get_best_run(metric)
                if best_run:
                    summary["best_runs"][metric] = {
                        "run_id": best_run.info.run_id,
                        "run_name": best_run.info.run_name,
                        "value": best_run.data.metrics[metric]
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {}


# 便捷函数
def create_mlflow_integration(tracking_uri: str = None,
                            registry_uri: str = None,
                            experiment_name: str = "fua_experiments") -> FUAMLflowIntegration:
    """创建FUA MLflow集成实例"""
    return FUAMLflowIntegration(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
        experiment_name=experiment_name
    )


def start_mlflow_ui(port: int = 5000, host: str = "127.0.0.1"):
    """启动MLflow UI"""
    try:
        import subprocess
        subprocess.run([
            "mlflow", "ui",
            "--port", str(port),
            "--host", host
        ])
    except Exception as e:
        logger.error(f"Failed to start MLflow UI: {e}")


# 测试函数
def test_mlflow_integration():
    """测试MLflow集成"""
    print("Testing FUA MLflow Integration...")
    
    # 创建集成实例
    integration = create_mlflow_integration(experiment_name="test_fua_experiment")
    
    # 创建模拟运行
    run_id = integration.create_training_run(
        model_name="test_model",
        model_config={"model_type": "CNN", "num_layers": 5},
        training_config={"epochs": 10, "batch_size": 32},
        run_name="test_run"
    )
    
    if run_id:
        print(f"Created training run: {run_id}")
        
        # 记录一些指标
        integration.log_training_metrics({
            "train_loss": 0.5,
            "val_loss": 0.6,
            "train_acc": 0.85,
            "val_acc": 0.82
        }, step=1)
        
        # 完成运行
        integration.complete_training_run({
            "final_train_loss": 0.3,
            "final_val_loss": 0.4,
            "final_train_acc": 0.92,
            "final_val_acc": 0.88
        })
        
        print("Training run completed successfully")
    
    # 获取实验摘要
    summary = integration.get_experiment_summary()
    print(f"Experiment summary: {summary}")
    
    print("MLflow Integration test completed!")


if __name__ == "__main__":
    test_mlflow_integration()