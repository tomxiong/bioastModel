#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练索引管理器
用于管理同一模型的多次训练记录，避免覆盖问题
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class TrainingIndexManager:
    """训练索引管理器"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.index_file = self.base_dir / "training_index.json"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.experiments_dir = self.base_dir / "experiments"
        
        # 确保目录存在
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)
        
        # 加载或创建索引
        self.index = self._load_index()
    
    def _load_index(self) -> Dict:
        """加载训练索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"警告: 加载训练索引失败: {e}")
                return self._create_empty_index()
        else:
            return self._create_empty_index()
    
    def _create_empty_index(self) -> Dict:
        """创建空索引结构"""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "models": {},
            "next_training_id": 1
        }
    
    def _save_index(self):
        """保存训练索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"错误: 保存训练索引失败: {e}")
    
    def register_training(self, model_name: str, training_config: Dict = None, 
                         description: str = "") -> str:
        """注册新的训练记录
        
        Args:
            model_name: 模型名称
            training_config: 训练配置
            description: 训练描述
            
        Returns:
            training_id: 训练ID
        """
        training_id = f"train_{self.index['next_training_id']:04d}"
        timestamp = datetime.now().isoformat()
        
        # 初始化模型记录
        if model_name not in self.index["models"]:
            self.index["models"][model_name] = {
                "model_name": model_name,
                "trainings": {},
                "latest_training_id": None,
                "best_training_id": None,
                "total_trainings": 0
            }
        
        # 添加训练记录
        training_record = {
            "training_id": training_id,
            "model_name": model_name,
            "start_time": timestamp,
            "end_time": None,
            "status": "started",
            "description": description,
            "config": training_config or {},
            "metrics": {},
            "paths": {
                "checkpoint_dir": f"checkpoints/{model_name}/{training_id}",
                "experiment_dir": f"experiments/{model_name}/{training_id}",
                "test_analysis_dir": f"checkpoints/{model_name}/{training_id}/test_analysis"
            },
            "files": {
                "model_file": None,
                "config_file": None,
                "log_file": None,
                "test_report": None
            }
        }
        
        self.index["models"][model_name]["trainings"][training_id] = training_record
        self.index["models"][model_name]["latest_training_id"] = training_id
        self.index["models"][model_name]["total_trainings"] += 1
        self.index["next_training_id"] += 1
        
        # 创建目录结构
        self._create_training_directories(training_record)
        
        self._save_index()
        
        print(f"✅ 已注册训练记录: {model_name} - {training_id}")
        return training_id
    
    def _create_training_directories(self, training_record: Dict):
        """创建训练目录结构"""
        paths = training_record["paths"]
        
        # 创建checkpoint目录
        checkpoint_dir = self.base_dir / paths["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建experiment目录
        experiment_dir = self.base_dir / paths["experiment_dir"]
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建测试分析目录
        test_analysis_dir = self.base_dir / paths["test_analysis_dir"]
        test_analysis_dir.mkdir(parents=True, exist_ok=True)
    
    def update_training_status(self, training_id: str, status: str, 
                              metrics: Dict = None, files: Dict = None):
        """更新训练状态"""
        # 查找训练记录
        training_record = None
        model_name = None
        
        for model, model_data in self.index["models"].items():
            if training_id in model_data["trainings"]:
                training_record = model_data["trainings"][training_id]
                model_name = model
                break
        
        if not training_record:
            print(f"错误: 未找到训练记录 {training_id}")
            return
        
        # 更新状态
        training_record["status"] = status
        if status in ["completed", "failed", "stopped"]:
            training_record["end_time"] = datetime.now().isoformat()
        
        # 更新指标
        if metrics:
            training_record["metrics"].update(metrics)
        
        # 更新文件路径
        if files:
            training_record["files"].update(files)
        
        # 如果训练完成且性能最佳，更新最佳训练ID
        if status == "completed" and metrics and "accuracy" in metrics:
            current_best = self.index["models"][model_name].get("best_training_id")
            if not current_best:
                self.index["models"][model_name]["best_training_id"] = training_id
            else:
                best_record = self.index["models"][model_name]["trainings"][current_best]
                if metrics["accuracy"] > best_record["metrics"].get("accuracy", 0):
                    self.index["models"][model_name]["best_training_id"] = training_id
        
        self._save_index()
        print(f"✅ 已更新训练状态: {training_id} -> {status}")
    
    def get_training_info(self, training_id: str) -> Optional[Dict]:
        """获取训练信息"""
        for model_data in self.index["models"].values():
            if training_id in model_data["trainings"]:
                return model_data["trainings"][training_id]
        return None
    
    def get_model_trainings(self, model_name: str) -> List[Dict]:
        """获取模型的所有训练记录"""
        if model_name not in self.index["models"]:
            return []
        
        trainings = self.index["models"][model_name]["trainings"]
        return list(trainings.values())
    
    def get_latest_training(self, model_name: str) -> Optional[Dict]:
        """获取模型的最新训练记录"""
        if model_name not in self.index["models"]:
            return None
        
        latest_id = self.index["models"][model_name].get("latest_training_id")
        if latest_id:
            return self.index["models"][model_name]["trainings"][latest_id]
        return None
    
    def get_best_training(self, model_name: str) -> Optional[Dict]:
        """获取模型的最佳训练记录"""
        if model_name not in self.index["models"]:
            return None
        
        best_id = self.index["models"][model_name].get("best_training_id")
        if best_id:
            return self.index["models"][model_name]["trainings"][best_id]
        return None
    
    def migrate_existing_checkpoints(self):
        """迁移现有的checkpoint到新的索引系统"""
        print("开始迁移现有checkpoint...")
        
        if not self.checkpoints_dir.exists():
            print("未找到checkpoints目录")
            return
        
        for model_dir in self.checkpoints_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            print(f"迁移模型: {model_name}")
            
            # 检查是否已经是新格式（包含训练ID子目录）
            subdirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('train_')]
            if subdirs:
                print(f"  {model_name} 已经是新格式，跳过")
                continue
            
            # 创建备份
            backup_dir = model_dir.parent / f"{model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(model_dir, backup_dir)
            print(f"  已创建备份: {backup_dir}")
            
            # 注册为第一次训练
            training_id = self.register_training(
                model_name=model_name,
                description="从现有checkpoint迁移"
            )
            
            # 移动文件到新目录
            new_training_dir = self.base_dir / f"checkpoints/{model_name}/{training_id}"
            
            # 移动所有文件
            for item in model_dir.iterdir():
                if item.is_file():
                    shutil.move(str(item), str(new_training_dir / item.name))
                elif item.is_dir() and item.name != training_id:
                    shutil.move(str(item), str(new_training_dir / item.name))
            
            # 更新训练记录
            files = {}
            if (new_training_dir / "best.pth").exists():
                files["model_file"] = f"checkpoints/{model_name}/{training_id}/best.pth"
            
            self.update_training_status(
                training_id=training_id,
                status="completed",
                files=files
            )
            
            print(f"  ✅ 迁移完成: {model_name} -> {training_id}")
        
        print("迁移完成!")
    
    def list_all_trainings(self) -> Dict:
        """列出所有训练记录"""
        summary = {
            "total_models": len(self.index["models"]),
            "total_trainings": sum(model["total_trainings"] for model in self.index["models"].values()),
            "models": {}
        }
        
        for model_name, model_data in self.index["models"].items():
            summary["models"][model_name] = {
                "total_trainings": model_data["total_trainings"],
                "latest_training_id": model_data.get("latest_training_id"),
                "best_training_id": model_data.get("best_training_id"),
                "trainings": []
            }
            
            for training_id, training in model_data["trainings"].items():
                summary["models"][model_name]["trainings"].append({
                    "training_id": training_id,
                    "status": training["status"],
                    "start_time": training["start_time"],
                    "end_time": training.get("end_time"),
                    "accuracy": training["metrics"].get("accuracy"),
                    "description": training.get("description", "")
                })
        
        return summary
    
    def print_training_summary(self):
        """打印训练摘要"""
        summary = self.list_all_trainings()
        
        print("\n" + "="*60)
        print("训练记录摘要")
        print("="*60)
        print(f"总模型数: {summary['total_models']}")
        print(f"总训练次数: {summary['total_trainings']}")
        print()
        
        for model_name, model_info in summary["models"].items():
            print(f"📊 模型: {model_name}")
            print(f"   训练次数: {model_info['total_trainings']}")
            print(f"   最新训练: {model_info['latest_training_id'] or 'N/A'}")
            print(f"   最佳训练: {model_info['best_training_id'] or 'N/A'}")
            
            if model_info['trainings']:
                print("   训练历史:")
                for training in sorted(model_info['trainings'], key=lambda x: x['training_id']):
                    status_icon = "✅" if training['status'] == 'completed' else "❌" if training['status'] == 'failed' else "🔄"
                    accuracy_str = f" (准确率: {training['accuracy']:.3f})" if training['accuracy'] else ""
                    print(f"     {status_icon} {training['training_id']}: {training['status']}{accuracy_str}")
            print()

def main():
    """主函数 - 命令行工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练索引管理器')
    parser.add_argument('--migrate', action='store_true', help='迁移现有checkpoint')
    parser.add_argument('--list', action='store_true', help='列出所有训练记录')
    parser.add_argument('--register', type=str, help='注册新训练记录（指定模型名）')
    parser.add_argument('--description', type=str, default='', help='训练描述')
    
    args = parser.parse_args()
    
    manager = TrainingIndexManager()
    
    if args.migrate:
        manager.migrate_existing_checkpoints()
    elif args.list:
        manager.print_training_summary()
    elif args.register:
        training_id = manager.register_training(
            model_name=args.register,
            description=args.description
        )
        print(f"已注册训练: {training_id}")
    else:
        manager.print_training_summary()

if __name__ == "__main__":
    main()