#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集管理和批量重训练脚本

这个脚本用于管理数据集的更新和模型的批量重训练。

使用方法:
    python dataset_manager.py --check                     # 检查数据集状态
    python dataset_manager.py --update-dataset            # 更新数据集
    python dataset_manager.py --retrain-all               # 重训练所有模型
    python dataset_manager.py --retrain-best              # 重训练性能最好的模型
    python dataset_manager.py --retrain-models model1 model2  # 重训练指定模型
    python dataset_manager.py --schedule-retrain          # 计划重训练任务
"""

import os
import sys
import json
import argparse
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class DatasetManager:
    """
    数据集管理器
    """
    
    def __init__(self, dataset_dir: str = "data", config_file: str = "dataset_config.json"):
        self.dataset_dir = Path(dataset_dir)
        self.config_file = Path(config_file)
        self.checkpoints_dir = Path("checkpoints")
        self.backup_dir = Path("backups")
        
        # 创建必要的目录
        self.dataset_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        print(f"📁 数据集目录: {self.dataset_dir}")
        print(f"⚙️ 配置文件: {self.config_file}")
        print(f"💾 备份目录: {self.backup_dir}")
    
    def calculate_dataset_hash(self, dataset_path: Path) -> str:
        """
        计算数据集的哈希值
        
        Args:
            dataset_path: 数据集路径
        
        Returns:
            str: 数据集哈希值
        """
        hash_md5 = hashlib.md5()
        
        if not dataset_path.exists():
            return ""
        
        # 遍历所有文件计算哈希
        for file_path in sorted(dataset_path.rglob("*")):
            if file_path.is_file():
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                    # 同时包含文件路径信息
                    hash_md5.update(str(file_path.relative_to(dataset_path)).encode())
                except Exception as e:
                    print(f"⚠️ 计算文件哈希失败 {file_path}: {e}")
        
        return hash_md5.hexdigest()
    
    def load_dataset_config(self) -> Dict[str, Any]:
        """
        加载数据集配置
        
        Returns:
            Dict: 数据集配置
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 加载配置文件失败: {e}")
        
        # 默认配置
        return {
            'last_update': None,
            'dataset_hash': None,
            'dataset_version': '1.0.0',
            'trained_models': {},
            'retrain_schedule': []
        }
    
    def save_dataset_config(self, config: Dict[str, Any]):
        """
        保存数据集配置
        
        Args:
            config: 数据集配置
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ 配置已保存: {self.config_file}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
    
    def check_dataset_status(self) -> Dict[str, Any]:
        """
        检查数据集状态
        
        Returns:
            Dict: 数据集状态信息
        """
        print(f"\n🔍 检查数据集状态...")
        
        config = self.load_dataset_config()
        current_hash = self.calculate_dataset_hash(self.dataset_dir)
        
        status = {
            'dataset_exists': self.dataset_dir.exists(),
            'dataset_path': str(self.dataset_dir),
            'current_hash': current_hash,
            'stored_hash': config.get('dataset_hash'),
            'last_update': config.get('last_update'),
            'dataset_version': config.get('dataset_version', '1.0.0'),
            'has_changes': current_hash != config.get('dataset_hash'),
            'trained_models': config.get('trained_models', {})
        }
        
        # 统计数据集信息
        if self.dataset_dir.exists():
            train_dir = self.dataset_dir / "train"
            val_dir = self.dataset_dir / "val"
            test_dir = self.dataset_dir / "test"
            
            status['structure'] = {
                'train_exists': train_dir.exists(),
                'val_exists': val_dir.exists(),
                'test_exists': test_dir.exists()
            }
            
            # 统计类别和样本数
            if train_dir.exists():
                classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
                status['classes'] = classes
                status['num_classes'] = len(classes)
                
                # 统计每个类别的样本数
                class_counts = {}
                total_samples = 0
                for class_dir in train_dir.iterdir():
                    if class_dir.is_dir():
                        count = len([f for f in class_dir.iterdir() if f.is_file()])
                        class_counts[class_dir.name] = count
                        total_samples += count
                
                status['class_counts'] = class_counts
                status['total_samples'] = total_samples
        
        # 打印状态信息
        print(f"📊 数据集状态报告:")
        print(f"  数据集路径: {status['dataset_path']}")
        print(f"  数据集存在: {'✅' if status['dataset_exists'] else '❌'}")
        print(f"  当前版本: {status['dataset_version']}")
        print(f"  最后更新: {status['last_update'] or '未知'}")
        print(f"  数据变化: {'🔄' if status['has_changes'] else '✅ 无变化'}")
        
        if 'classes' in status:
            print(f"  类别数量: {status['num_classes']}")
            print(f"  总样本数: {status['total_samples']}")
            print(f"  类别分布:")
            for class_name, count in status.get('class_counts', {}).items():
                print(f"    - {class_name}: {count} 样本")
        
        trained_models = status['trained_models']
        if trained_models:
            print(f"  已训练模型: {len(trained_models)} 个")
            for model_name, info in trained_models.items():
                dataset_version = info.get('dataset_version', '未知')
                needs_retrain = dataset_version != status['dataset_version']
                print(f"    - {model_name}: 版本 {dataset_version} {'🔄 需要重训练' if needs_retrain else '✅'}")
        
        return status
    
    def update_dataset(self, new_dataset_path: Optional[str] = None) -> bool:
        """
        更新数据集
        
        Args:
            new_dataset_path: 新数据集路径（如果提供）
        
        Returns:
            bool: 更新是否成功
        """
        print(f"\n🔄 更新数据集...")
        
        config = self.load_dataset_config()
        
        # 如果提供了新数据集路径，复制到目标位置
        if new_dataset_path:
            new_path = Path(new_dataset_path)
            if not new_path.exists():
                print(f"❌ 新数据集路径不存在: {new_dataset_path}")
                return False
            
            # 备份当前数据集
            if self.dataset_dir.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = self.backup_dir / f"dataset_backup_{timestamp}"
                print(f"💾 备份当前数据集到: {backup_path}")
                shutil.copytree(self.dataset_dir, backup_path)
            
            # 复制新数据集
            print(f"📁 复制新数据集从: {new_dataset_path}")
            if self.dataset_dir.exists():
                shutil.rmtree(self.dataset_dir)
            shutil.copytree(new_path, self.dataset_dir)
        
        # 计算新的哈希值
        new_hash = self.calculate_dataset_hash(self.dataset_dir)
        old_hash = config.get('dataset_hash')
        
        if new_hash == old_hash:
            print(f"✅ 数据集无变化")
            return True
        
        # 更新配置
        config['dataset_hash'] = new_hash
        config['last_update'] = datetime.now().isoformat()
        
        # 增加版本号
        current_version = config.get('dataset_version', '1.0.0')
        version_parts = current_version.split('.')
        if len(version_parts) >= 2:
            version_parts[1] = str(int(version_parts[1]) + 1)
            config['dataset_version'] = '.'.join(version_parts)
        else:
            config['dataset_version'] = '1.1.0'
        
        self.save_dataset_config(config)
        
        print(f"✅ 数据集更新完成")
        print(f"  新版本: {config['dataset_version']}")
        print(f"  新哈希: {new_hash[:16]}...")
        
        return True
    
    def get_trained_models(self) -> List[Dict[str, Any]]:
        """
        获取已训练的模型列表
        
        Returns:
            List[Dict]: 模型列表
        """
        models = []
        
        if not self.checkpoints_dir.exists():
            return models
        
        for model_dir in self.checkpoints_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            best_checkpoint = model_dir / "best.pth"
            history_file = model_dir / "training_history.json"
            
            if not best_checkpoint.exists():
                continue
            
            model_info = {
                'name': model_name,
                'checkpoint_path': str(best_checkpoint),
                'model_dir': str(model_dir),
                'has_history': history_file.exists()
            }
            
            # 读取训练历史
            if history_file.exists():
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                    model_info['best_val_acc'] = history.get('best_val_acc', 0)
                    model_info['training_time'] = history.get('training_time', 0)
                except Exception:
                    model_info['best_val_acc'] = 0
                    model_info['training_time'] = 0
            else:
                model_info['best_val_acc'] = 0
                model_info['training_time'] = 0
            
            models.append(model_info)
        
        return models
    
    def retrain_model(self, model_name: str, config_path: Optional[str] = None) -> bool:
        """
        重训练单个模型
        
        Args:
            model_name: 模型名称
            config_path: 配置文件路径
        
        Returns:
            bool: 重训练是否成功
        """
        print(f"\n🔄 重训练模型: {model_name}")
        
        # 构建训练命令
        cmd = ["python", "train_single_model.py", "--model", model_name]
        
        if config_path:
            cmd.extend(["--config", config_path])
        
        # 添加重训练标志
        cmd.append("--retrain")
        
        try:
            print(f"🚀 执行命令: {' '.join(cmd)}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行训练
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            # 记录结束时间
            end_time = time.time()
            training_time = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ 模型 {model_name} 重训练成功 (耗时: {training_time:.1f}秒)")
                
                # 更新配置中的模型信息
                config = self.load_dataset_config()
                if 'trained_models' not in config:
                    config['trained_models'] = {}
                
                config['trained_models'][model_name] = {
                    'dataset_version': config.get('dataset_version', '1.0.0'),
                    'retrain_time': datetime.now().isoformat(),
                    'training_duration': training_time
                }
                
                self.save_dataset_config(config)
                return True
            else:
                print(f"❌ 模型 {model_name} 重训练失败")
                print(f"错误输出: {result.stderr}")
                return False
        
        except Exception as e:
            print(f"❌ 重训练过程中发生错误: {e}")
            return False
    
    def retrain_models(self, model_names: Optional[List[str]] = None, 
                      retrain_all: bool = False, 
                      retrain_best: bool = False) -> Dict[str, bool]:
        """
        批量重训练模型
        
        Args:
            model_names: 指定要重训练的模型名称列表
            retrain_all: 是否重训练所有模型
            retrain_best: 是否只重训练性能最好的模型
        
        Returns:
            Dict[str, bool]: 每个模型的重训练结果
        """
        print(f"\n🔄 开始批量重训练...")
        
        # 获取要重训练的模型列表
        all_models = self.get_trained_models()
        
        if not all_models:
            print(f"❌ 没有找到已训练的模型")
            return {}
        
        models_to_retrain = []
        
        if retrain_all:
            models_to_retrain = all_models
            print(f"📋 将重训练所有 {len(models_to_retrain)} 个模型")
        elif retrain_best:
            # 选择性能最好的模型
            best_model = max(all_models, key=lambda x: x.get('best_val_acc', 0))
            models_to_retrain = [best_model]
            print(f"🏆 将重训练性能最好的模型: {best_model['name']} (准确率: {best_model.get('best_val_acc', 0):.2f}%)")
        elif model_names:
            # 按名称过滤
            model_name_set = set(model_names)
            models_to_retrain = [m for m in all_models if m['name'] in model_name_set]
            
            found_names = {m['name'] for m in models_to_retrain}
            missing_names = model_name_set - found_names
            
            if missing_names:
                print(f"⚠️ 未找到模型: {', '.join(missing_names)}")
            
            print(f"📋 将重训练 {len(models_to_retrain)} 个指定模型")
        else:
            print(f"❌ 请指定要重训练的模型")
            return {}
        
        # 执行重训练
        results = {}
        total_models = len(models_to_retrain)
        
        for i, model in enumerate(models_to_retrain, 1):
            model_name = model['name']
            print(f"\n📊 进度: {i}/{total_models} - 重训练 {model_name}")
            
            success = self.retrain_model(model_name)
            results[model_name] = success
            
            if success:
                print(f"✅ {model_name} 重训练成功")
            else:
                print(f"❌ {model_name} 重训练失败")
        
        # 汇总结果
        successful = sum(1 for success in results.values() if success)
        failed = total_models - successful
        
        print(f"\n📊 批量重训练完成:")
        print(f"  成功: {successful} 个")
        print(f"  失败: {failed} 个")
        
        if failed > 0:
            failed_models = [name for name, success in results.items() if not success]
            print(f"  失败模型: {', '.join(failed_models)}")
        
        return results
    
    def schedule_retrain(self, schedule_time: Optional[str] = None) -> bool:
        """
        计划重训练任务
        
        Args:
            schedule_time: 计划执行时间 (格式: YYYY-MM-DD HH:MM)
        
        Returns:
            bool: 计划是否成功
        """
        print(f"\n⏰ 计划重训练任务...")
        
        if not schedule_time:
            # 默认计划在明天同一时间
            schedule_time = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d %H:%M')
        
        try:
            scheduled_datetime = datetime.strptime(schedule_time, '%Y-%m-%d %H:%M')
        except ValueError:
            print(f"❌ 时间格式错误，请使用 YYYY-MM-DD HH:MM 格式")
            return False
        
        if scheduled_datetime <= datetime.now():
            print(f"❌ 计划时间必须在未来")
            return False
        
        config = self.load_dataset_config()
        if 'retrain_schedule' not in config:
            config['retrain_schedule'] = []
        
        # 添加计划任务
        task = {
            'scheduled_time': scheduled_datetime.isoformat(),
            'created_time': datetime.now().isoformat(),
            'status': 'pending',
            'task_type': 'retrain_all'
        }
        
        config['retrain_schedule'].append(task)
        self.save_dataset_config(config)
        
        print(f"✅ 重训练任务已计划")
        print(f"  执行时间: {schedule_time}")
        print(f"  任务类型: 重训练所有模型")
        
        return True

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='数据集管理和批量重训练工具')
    parser.add_argument('--check', action='store_true', help='检查数据集状态')
    parser.add_argument('--update-dataset', help='更新数据集（可选：指定新数据集路径）')
    parser.add_argument('--retrain-all', action='store_true', help='重训练所有模型')
    parser.add_argument('--retrain-best', action='store_true', help='重训练性能最好的模型')
    parser.add_argument('--retrain-models', nargs='+', help='重训练指定模型')
    parser.add_argument('--schedule-retrain', help='计划重训练任务 (格式: YYYY-MM-DD HH:MM)')
    parser.add_argument('--dataset-dir', default='data', help='数据集目录')
    parser.add_argument('--config-file', default='dataset_config.json', help='配置文件路径')
    
    args = parser.parse_args()
    
    print("🧬 BioAst数据集管理工具")
    print("=" * 50)
    
    # 创建数据集管理器
    manager = DatasetManager(
        dataset_dir=args.dataset_dir,
        config_file=args.config_file
    )
    
    # 执行相应操作
    if args.check:
        status = manager.check_dataset_status()
        
    elif args.update_dataset is not None:
        # 如果提供了路径参数，使用该路径；否则只更新配置
        dataset_path = args.update_dataset if args.update_dataset else None
        success = manager.update_dataset(dataset_path)
        if success:
            print(f"\n💡 提示: 数据集已更新，建议重训练模型以获得最佳性能")
    
    elif args.retrain_all:
        results = manager.retrain_models(retrain_all=True)
        
    elif args.retrain_best:
        results = manager.retrain_models(retrain_best=True)
        
    elif args.retrain_models:
        results = manager.retrain_models(model_names=args.retrain_models)
        
    elif args.schedule_retrain:
        success = manager.schedule_retrain(args.schedule_retrain)
        
    else:
        # 默认检查状态
        print("\n💡 没有指定操作，执行状态检查...")
        status = manager.check_dataset_status()
        
        print(f"\n🔧 可用操作:")
        print(f"  --check                    检查数据集状态")
        print(f"  --update-dataset [path]    更新数据集")
        print(f"  --retrain-all              重训练所有模型")
        print(f"  --retrain-best             重训练最佳模型")
        print(f"  --retrain-models m1 m2     重训练指定模型")
        print(f"  --schedule-retrain time    计划重训练任务")

if __name__ == "__main__":
    main()