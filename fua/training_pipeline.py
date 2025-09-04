"""
FUA 自动化训练流水线
支持一键训练、断点续训和实时监控
"""

import os
import sys
import json
import time
import subprocess
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fua.dataset_iteration_manager import DatasetVersionManager, DatasetIncrementalUpdater
from fua.parameter_optimizer import ParameterHistoryManager, ParameterOptimizer


class TrainingPipeline:
    """自动化训练流水线"""
    
    def __init__(self, config_path: str = "fua/pipeline_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 初始化组件
        self.dataset_manager = DatasetVersionManager()
        self.param_history = ParameterHistoryManager()
        
        # 流水线状态
        self.status = "idle"
        self.current_job = None
        self.job_queue = queue.Queue()
        self.results = {}
        
        # 设置日志
        self._setup_logging()
        
        # 启动工作线程
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def _load_config(self) -> Dict:
        """加载配置"""
        default_config = {
            "base_path": str(Path.cwd()),
            "experiments_path": "experiments",
            "models_path": "models",
            "venv_path": ".venv",
            "gpu_available": False,
            "max_concurrent_jobs": 1,
            "notification_email": None,
            "slack_webhook": None
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        # 确保目录存在
        for path_key in ["experiments_path", "models_path"]:
            Path(default_config[path_key]).mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _setup_logging(self):
        """设置日志"""
        log_path = Path(self.config["experiments_path"]) / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("TrainingPipeline")
    
    def _worker_loop(self):
        """工作线程主循环"""
        while True:
            try:
                # 从队列获取任务
                job = self.job_queue.get()
                if job is None:  # 停止信号
                    break
                
                self.current_job = job
                self.status = "running"
                
                # 执行任务
                self.logger.info(f"Starting job: {job['job_id']}")
                result = self._execute_job(job)
                
                # 保存结果
                self.results[job['job_id']] = result
                
                # 发送通知
                if self.config.get("notification_email"):
                    self._send_notification(job, result)
                
                self.status = "idle"
                self.current_job = None
                self.job_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Job execution failed: {e}")
                self.status = "error"
    
    def _execute_job(self, job: Dict) -> Dict:
        """执行训练任务"""
        start_time = datetime.now()
        
        try:
            # 准备实验环境
            experiment_dir = self._prepare_experiment(job)
            
            # 生成训练脚本
            script_path = self._generate_training_script(job, experiment_dir)
            
            # 执行训练
            result = self._run_training(script_path, job)
            
            # 保存结果
            result["job_id"] = job["job_id"]
            result["start_time"] = start_time.isoformat()
            result["end_time"] = datetime.now().isoformat()
            result["duration"] = (datetime.now() - start_time).total_seconds()
            
            # 记录到参数历史
            if result.get("success"):
                self.param_history.record_experiment(
                    job["model_name"],
                    job["parameters"],
                    result["metrics"],
                    job.get("dataset_version")
                )
            
            return result
            
        except Exception as e:
            return {
                "job_id": job["job_id"],
                "success": False,
                "error": str(e),
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
    
    def _prepare_experiment(self, job: Dict) -> Path:
        """准备实验环境"""
        # 创建实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{job['model_name']}_{timestamp}"
        experiment_dir = Path(self.config["experiments_path"]) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存作业配置
        with open(experiment_dir / "job_config.json", 'w') as f:
            json.dump(job, f, indent=2)
        
        return experiment_dir
    
    def _generate_training_script(self, job: Dict, experiment_dir: Path) -> Path:
        """生成训练脚本"""
        script_content = self._get_script_template(job, experiment_dir)
        script_path = experiment_dir / "train.py"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 设置执行权限
        script_path.chmod(0o755)
        
        return script_path
    
    def _get_script_template(self, job: Dict, experiment_dir: Path) -> str:
        """获取训练脚本模板"""
        template = f'''#!/usr/bin/env python3
"""
Auto-generated training script for {job['model_name']}
Generated at: {datetime.now().isoformat()}
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入必要的模块
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from datetime import datetime

# 导入模型和数据加载器
try:
    from core.config.model_configs import get_model_config, MODEL_CONFIGS
    from core.config.training_configs import get_model_specific_config
    from core.training_utils import Trainer
    from training.dataset import BioastDataset
    from training.transforms import get_train_transform, get_val_transform
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

def main():
    """主训练函数"""
    # 配置
    model_name = "{job['model_name']}"
    config = {json.dumps(job['parameters'], indent=8)}
    dataset_version = "{job.get('dataset_version', 'latest')}"
    
    print(f"Starting training for {{model_name}}")
    print(f"Parameters: {{config}}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and {self.config.get("gpu_available", False)} else "cpu")
    print(f"Using device: {{device}}")
    
    try:
        # 加载数据集
        train_dataset = BioastDataset(
            root="bioast_dataset",
            split="train",
            transform=get_train_transform()
        )
        
        val_dataset = BioastDataset(
            root="bioast_dataset",
            split="val",
            transform=get_val_transform()
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=4
        )
        
        print(f"Dataset loaded - Train: {{len(train_dataset)}}, Val: {{len(val_dataset)}}")
        
        # 创建模型
        model_config = get_model_config(model_name)
        if model_config is None:
            print(f"Model config not found for {{model_name}}")
            sys.exit(1)
        
        # 动态导入模型
        model_module = __import__(f"models.{{model_config['module']}}", fromlist=['create_{{model_name}}'])
        model_factory = getattr(model_module, f"create_{{model_name}}")
        model = model_factory(num_classes=2)
        model = model.to(device)
        
        print(f"Model created: {{model_name}}")
        
        # 设置优化器
        optimizer_name = config.get("optimizer", "adam")
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get("learning_rate", 0.001),
                weight_decay=config.get("weight_decay", 0.0001)
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.get("learning_rate", 0.001),
                momentum=0.9
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            experiment_dir="{experiment_dir}"
        )
        
        # 开始训练
        print("Starting training...")
        history = trainer.train(
            num_epochs=config.get("epochs", 50),
            save_best=True,
            early_stopping_patience=10
        )
        
        # 保存结果
        results = {{
            "model_name": model_name,
            "dataset_version": dataset_version,
            "parameters": config,
            "final_metrics": {{
                "train_accuracy": history.get("train_accuracy", [])[-1] if history.get("train_accuracy") else 0,
                "val_accuracy": history.get("val_accuracy", [])[-1] if history.get("val_accuracy") else 0,
                "train_loss": history.get("train_loss", [])[-1] if history.get("train_loss") else 0,
                "val_loss": history.get("val_loss", [])[-1] if history.get("val_loss") else 0
            }},
            "best_metrics": {{
                "val_accuracy": max(history.get("val_accuracy", [0])),
                "val_loss": min(history.get("val_loss", [float('inf')]))
            }},
            "total_epochs": len(history.get("train_loss", [])),
            "timestamp": datetime.now().isoformat()
        }}
        
        # 保存结果
        with open("{experiment_dir / 'results.json'}", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Training completed successfully!")
        print(f"Best validation accuracy: {{results['best_metrics']['val_accuracy']:.4f}}")
        
        return results
        
    except Exception as e:
        print(f"Training failed: {{e}}")
        import traceback
        traceback.print_exc()
        return {{"error": str(e)}}

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results.get("success", True) else 1)
'''
        return template
    
    def _run_training(self, script_path: Path, job: Dict) -> Dict:
        """运行训练脚本"""
        # 准备环境变量
        env = os.environ.copy()
        venv_python = Path(self.config["venv_path"]) / "bin" / "python"
        if not venv_python.exists():
            venv_python = Path(self.config["venv_path"]) / "Scripts" / "python.exe"
        
        # 运行脚本
        try:
            result = subprocess.run(
                [str(venv_python), str(script_path)],
                cwd=script_path.parent,
                capture_output=True,
                text=True,
                timeout=job.get("timeout", 3600 * 24)  # 默认24小时超时
            )
            
            # 读取结果
            results_file = script_path.parent / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                results["success"] = True
            else:
                results = {
                    "success": False,
                    "error": "Results file not found",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            return results
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Training timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def submit_job(self, model_name: str, parameters: Dict, 
                   dataset_version: str = None, priority: str = "normal") -> str:
        """提交训练任务"""
        job_id = f"job_{int(time.time())}"
        
        job = {
            "job_id": job_id,
            "model_name": model_name,
            "parameters": parameters,
            "dataset_version": dataset_version,
            "priority": priority,
            "submitted_at": datetime.now().isoformat(),
            "status": "queued"
        }
        
        self.job_queue.put(job)
        self.logger.info(f"Job submitted: {job_id}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict:
        """获取任务状态"""
        if job_id in self.results:
            return self.results[job_id]
        elif self.current_job and self.current_job["job_id"] == job_id:
            return {
                "job_id": job_id,
                "status": "running",
                "start_time": self.current_job.get("start_time")
            }
        else:
            return {"job_id": job_id, "status": "queued"}
    
    def list_jobs(self) -> List[Dict]:
        """列出所有任务"""
        jobs = []
        
        # 当前运行的任务
        if self.current_job:
            jobs.append({
                "job_id": self.current_job["job_id"],
                "model_name": self.current_job["model_name"],
                "status": "running",
                "submitted_at": self.current_job["submitted_at"]
            })
        
        # 已完成的任务
        for job_id, result in self.results.items():
            jobs.append({
                "job_id": job_id,
                "model_name": result.get("model_name", "unknown"),
                "status": "completed" if result.get("success") else "failed",
                "submitted_at": result.get("start_time"),
                "completed_at": result.get("end_time"),
                "accuracy": result.get("metrics", {}).get("accuracy", 0)
            })
        
        return sorted(jobs, key=lambda x: x.get("submitted_at", ""), reverse=True)
    
    def stop_job(self, job_id: str) -> bool:
        """停止任务"""
        # 简化实现：标记任务为停止状态
        if self.current_job and self.current_job["job_id"] == job_id:
            self.status = "stopping"
            return True
        return False
    
    def _send_notification(self, job: Dict, result: Dict):
        """发送通知（简化实现）"""
        message = f"""
Training Job {job['job_id']} Completed!

Model: {job['model_name']}
Status: {'Success' if result.get('success') else 'Failed'}
Duration: {result.get('duration', 0):.2f} seconds

"""
        if result.get("success"):
            metrics = result.get("metrics", {})
            message += f"""
Accuracy: {metrics.get('accuracy', 0):.4f}
Loss: {metrics.get('loss', 0):.4f}
"""
        else:
            message += f"Error: {result.get('error', 'Unknown error')}"
        
        print(message)  # 简化：只打印到控制台
        # 实际实现可以发送邮件或Slack通知
    
    def create_iterative_workflow(self, model_name: str, iterations: int = 5,
                                 strategy: str = "adaptive") -> str:
        """创建迭代式工作流"""
        workflow_id = f"workflow_{int(time.time())}"
        
        # 提交第一个任务
        initial_params = self._get_initial_parameters(model_name)
        self.submit_job(model_name, initial_params)
        
        # 设置后续任务的自动提交（简化实现）
        self.logger.info(f"Created iterative workflow {workflow_id} for {model_name}")
        
        return workflow_id
    
    def _get_initial_parameters(self, model_name: str) -> Dict:
        """获取初始参数"""
        # 尝试从历史获取最佳参数
        best_config = self.param_history.get_best_config(model_name)
        if best_config:
            return best_config.get("parameters", {})
        
        # 使用默认参数
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "optimizer": "adam",
            "weight_decay": 0.0001
        }


class PipelineManager:
    """流水线管理器 - 提供高级API"""
    
    def __init__(self):
        self.pipeline = TrainingPipeline()
        self.optimizer = ParameterOptimizer("default", self.pipeline.param_history)
    
    def quick_train(self, model_name: str, epochs: int = 50) -> str:
        """快速训练"""
        params = {
            "epochs": epochs,
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "adam"
        }
        return self.pipeline.submit_job(model_name, params)
    
    def smart_train(self, model_name: str, max_iterations: int = 10) -> str:
        """智能训练（自动调优）"""
        # 获取参数建议
        suggestion = self.optimizer.suggest_parameters("adaptive")
        
        # 提交任务
        return self.pipeline.submit_job(model_name, suggestion)
    
    def batch_train(self, model_names: List[str]) -> List[str]:
        """批量训练"""
        job_ids = []
        for model_name in model_names:
            job_id = self.quick_train(model_name)
            job_ids.append(job_id)
        return job_ids
    
    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        jobs = self.pipeline.list_jobs()
        
        summary = {
            "total_jobs": len(jobs),
            "completed": len([j for j in jobs if j["status"] == "completed"]),
            "failed": len([j for j in jobs if j["status"] == "failed"]),
            "running": len([j for j in jobs if j["status"] == "running"]),
            "success_rate": 0
        }
        
        if summary["completed"] > 0:
            summary["success_rate"] = summary["completed"] / (summary["completed"] + summary["failed"])
        
        return summary


# 使用示例
if __name__ == "__main__":
    # 创建流水线管理器
    manager = PipelineManager()
    
    # 快速训练
    print("Submitting quick training job...")
    job_id = manager.quick_train("resnet18", epochs=30)
    print(f"Job submitted: {job_id}")
    
    # 智能训练
    print("\nSubmitting smart training job...")
    smart_job_id = manager.smart_train("efficientnet_b0")
    print(f"Smart job submitted: {smart_job_id}")
    
    # 批量训练
    print("\nSubmitting batch training jobs...")
    batch_job_ids = manager.batch_train(["mobilenetv3", "vgg16"])
    print(f"Batch jobs submitted: {batch_job_ids}")
    
    # 获取摘要
    print("\nTraining Summary:")
    summary = manager.get_training_summary()
    print(json.dumps(summary, indent=2))