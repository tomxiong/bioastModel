#!/usr/bin/env python3
"""
Model Registry Manager
管理模型注册表，按优先级执行训练、测试和ONNX转换
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

class ModelManager:
    def __init__(self, registry_file="model_registry.json"):
        self.registry_file = registry_file
        self.registry = self.load_registry()
        self.ensure_directories()
    
    def load_registry(self):
        """加载模型注册表"""
        if not os.path.exists(self.registry_file):
            raise FileNotFoundError(f"Registry file {self.registry_file} not found")
        
        with open(self.registry_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_registry(self):
        """保存模型注册表"""
        self.registry['registry_info']['last_updated'] = datetime.now().isoformat()
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        dirs = self.registry['directory_structure']
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def get_models_by_priority(self):
        """按优先级获取模型列表"""
        models = []
        for model_id, model_info in self.registry['models'].items():
            models.append((model_info['priority'], model_id, model_info))
        
        return sorted(models, key=lambda x: x[0])
    
    def update_training_record(self, model_id, training_record):
        """更新模型训练记录"""
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_info = self.registry['models'][model_id]
        
        # 添加到训练历史
        model_info['training_history'].append(training_record)
        
        # 更新最新训练记录
        model_info['latest_training'] = training_record
        
        # 保存注册表
        self.save_registry()
    
    def train_model(self, model_id):
        """训练指定模型"""
        model_info = self.registry['models'].get(model_id)
        if not model_info:
            print(f"❌ Model {model_id} not found in registry")
            return False
        
        trainer_script = model_info['trainer_script']
        if not os.path.exists(trainer_script):
            print(f"❌ Trainer script {trainer_script} not found")
            return False
        
        print(f"🚀 Training model: {model_info['name']} (Priority {model_info['priority']})")
        print(f"📝 Description: {model_info['description']}")
        print(f"🔧 Trainer: {trainer_script}")
        
        try:
            # 执行训练脚本
            result = subprocess.run([sys.executable, trainer_script], 
                                  capture_output=True, text=True, check=True)
            
            print(f"✅ Training completed successfully for {model_id}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed for {model_id}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def print_status(self):
        """打印当前状态"""
        print(f"\n{'='*70}")
        print("📊 MODEL REGISTRY STATUS")
        print(f"{'='*70}")
        
        total_models = len(self.registry['models'])
        trained_models = 0
        onnx_converted = 0
        
        for model_id, model_info in self.registry['models'].items():
            has_training = len(model_info['training_history']) > 0
            has_onnx = model_info['onnx_status'] != 'not_converted'
            
            if has_training:
                trained_models += 1
            if has_onnx:
                onnx_converted += 1
        
        print(f"Total Models: {total_models}")
        print(f"Trained Models: {trained_models}")
        print(f"ONNX Converted: {onnx_converted}")
        
        print(f"\n📋 Model Details:")
        models = self.get_models_by_priority()
        for priority, model_id, model_info in models:
            has_training = len(model_info['training_history']) > 0
            has_onnx = model_info['onnx_status'] != 'not_converted'
            training_status = "✅" if has_training else "❌"
            onnx_status = "✅" if has_onnx else "❌"
            print(f"  {priority:2d}. {model_info['name']} (Tier {model_info['tier']}) - "
                  f"Training: {training_status}, ONNX: {onnx_status}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Registry Manager')
    parser.add_argument('--action', choices=['status', 'train'], 
                       default='status', help='Action to perform')
    parser.add_argument('--model', help='Specific model ID to process')
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.action == 'status':
        manager.print_status()
    elif args.action == 'train':
        if args.model:
            manager.train_model(args.model)
        else:
            print("❌ Please specify --model for training")

if __name__ == "__main__":
    main()