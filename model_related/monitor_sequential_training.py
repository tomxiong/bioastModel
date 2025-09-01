#!/usr/bin/env python3
"""
顺序训练监控脚本
监控当前训练任务，完成后自动进行测试并启动下一个模型训练
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
import logging

# 设置日志 - 简化版本，避免编码问题
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sequential_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SequentialTrainingMonitor:
    def __init__(self):
        self.base_dir = Path("D:/ws1/bioastModel")
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.reports_dir = self.base_dir / "reports"
        
        # 待训练的模型队列（按优先级排序）
        self.training_queue = [
            "convnext_tiny",  # 当前正在训练
            "vit_tiny",
            "coatnet", 
            "mic_mobilenetv3",
            "micro_vit",
            "airbubble_hybrid_net",
            "enhanced_airbubble_detector"
        ]
        
        self.current_model = None
        self.current_process = None
        
    def check_training_completion(self, model_name):
        """检查模型训练是否完成"""
        # 检查是否有训练结果文件
        result_pattern = f"single_model_result_{model_name}_*.json"
        result_files = list(self.base_dir.glob(result_pattern))
        
        if result_files:
            # 找到最新的结果文件
            latest_result = max(result_files, key=lambda x: x.stat().st_mtime)
            print(f"[SUCCESS] {model_name} training completed, result file: {latest_result.name}")
            logger.info(f"[SUCCESS] {model_name} training completed, result file: {latest_result.name}")
            return True, latest_result
        
        return False, None
    
    def run_test_analysis(self, model_name):
        """运行测试分析"""
        print(f"[TEST] Starting test analysis for {model_name}...")
        logger.info(f"[TEST] Starting test analysis for {model_name}...")
        
        try:
            cmd = [
                sys.executable, "test_result_analyzer.py",
                "--model", model_name,
                "--force"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode == 0:
                print(f"[SUCCESS] {model_name} test analysis completed")
                logger.info(f"[SUCCESS] {model_name} test analysis completed")
                return True
            else:
                print(f"[ERROR] {model_name} test analysis failed: {result.stderr}")
                logger.error(f"[ERROR] {model_name} test analysis failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"[ERROR] {model_name} test analysis timeout")
            logger.error(f"[ERROR] {model_name} test analysis timeout")
            return False
        except Exception as e:
            print(f"[ERROR] {model_name} test analysis exception: {e}")
            logger.error(f"[ERROR] {model_name} test analysis exception: {e}")
            return False
    
    def start_next_training(self, model_name, batch_size=8):
        """启动下一个模型的训练"""
        print(f"[START] Starting training for {model_name}...")
        logger.info(f"[START] Starting training for {model_name}...")
        
        try:
            cmd = [
                sys.executable, "train_single_model.py",
                "--model", model_name,
                "--epochs", "10",
                "--batch_size", str(batch_size)
            ]
            
            # 启动训练进程
            process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print(f"[SUCCESS] {model_name} training started, PID: {process.pid}")
            logger.info(f"[SUCCESS] {model_name} training started, PID: {process.pid}")
            return process
            
        except Exception as e:
            print(f"[ERROR] Failed to start {model_name} training: {e}")
            logger.error(f"[ERROR] Failed to start {model_name} training: {e}")
            return None
    
    def monitor_current_training(self):
        """监控当前训练进程"""
        if not self.current_process:
            return False
            
        # 检查进程是否还在运行
        poll_result = self.current_process.poll()
        
        if poll_result is None:
            # 进程还在运行
            return True
        else:
            # 进程已结束
            if poll_result == 0:
                print(f"[SUCCESS] {self.current_model} training process ended normally")
                logger.info(f"[SUCCESS] {self.current_model} training process ended normally")
            else:
                print(f"[ERROR] {self.current_model} training process ended abnormally, exit code: {poll_result}")
                logger.error(f"[ERROR] {self.current_model} training process ended abnormally, exit code: {poll_result}")
                
            self.current_process = None
            return False
    
    def run_monitoring_cycle(self):
        """运行监控循环"""
        print("[MONITOR] Starting sequential training monitoring...")
        logger.info("[MONITOR] Starting sequential training monitoring...")
        
        # 首先检查ConvNeXt-Tiny是否正在训练
        self.current_model = "convnext_tiny"
        print(f"[MONITOR] Monitoring {self.current_model} training status...")
        logger.info(f"[MONITOR] Monitoring {self.current_model} training status...")
        
        model_index = 0
        
        while model_index < len(self.training_queue):
            current_model = self.training_queue[model_index]
            
            # 检查训练是否完成
            is_completed, result_file = self.check_training_completion(current_model)
            
            if is_completed:
                print(f"[COMPLETE] {current_model} training completed")
                logger.info(f"[COMPLETE] {current_model} training completed")
                
                # 运行测试分析
                test_success = self.run_test_analysis(current_model)
                
                if test_success:
                    print(f"[SUCCESS] {current_model} complete workflow finished")
                    logger.info(f"[SUCCESS] {current_model} complete workflow finished")
                else:
                    print(f"[WARNING] {current_model} test analysis failed, but continuing to next model")
                    logger.warning(f"[WARNING] {current_model} test analysis failed, but continuing to next model")
                
                # 移动到下一个模型
                model_index += 1
                
                if model_index < len(self.training_queue):
                    next_model = self.training_queue[model_index]
                    
                    # 根据模型调整批次大小
                    batch_size = 8
                    if next_model in ["vit_tiny", "coatnet"]:
                        batch_size = 4  # 更小的批次大小
                    
                    # 启动下一个模型训练
                    self.current_process = self.start_next_training(next_model, batch_size)
                    self.current_model = next_model
                    
                    if not self.current_process:
                        print(f"[ERROR] Cannot start {next_model} training, skipping")
                        logger.error(f"[ERROR] Cannot start {next_model} training, skipping")
                        model_index += 1
                        continue
                else:
                    print("[COMPLETE] All model training completed!")
                    logger.info("[COMPLETE] All model training completed!")
                    break
            else:
                # 训练未完成，继续等待
                if self.current_process:
                    # 监控当前进程
                    if not self.monitor_current_training():
                        print(f"[WARNING] {current_model} training process ended but no result file found")
                        logger.warning(f"[WARNING] {current_model} training process ended but no result file found")
                        # 可能需要重新启动训练
                        time.sleep(10)
                        continue
                
                print(f"[WAITING] {current_model} training in progress, waiting 5 minutes before next check...")
                logger.info(f"[WAITING] {current_model} training in progress, waiting 5 minutes before next check...")
                time.sleep(300)  # 等待5分钟
        
        print("[FINISH] Sequential training monitoring ended")
        logger.info("[FINISH] Sequential training monitoring ended")
    
    def generate_final_report(self):
        """生成最终的综合报告"""
        print("[REPORT] Generating final comprehensive report...")
        logger.info("[REPORT] Generating final comprehensive report...")
        
        try:
            # 运行综合模型分析
            subprocess.run([
                sys.executable, "scripts/comprehensive_model_analysis.py"
            ], cwd=self.base_dir, check=True)
            
            # 运行模型比较
            subprocess.run([
                sys.executable, "compare_models.py"
            ], cwd=self.base_dir, check=True)
            
            print("[SUCCESS] Final report generation completed")
            logger.info("[SUCCESS] Final report generation completed")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate final report: {e}")
            logger.error(f"[ERROR] Failed to generate final report: {e}")

def main():
    monitor = SequentialTrainingMonitor()
    
    try:
        monitor.run_monitoring_cycle()
        monitor.generate_final_report()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] User interrupted monitoring")
        logger.info("\n[INTERRUPT] User interrupted monitoring")
    except Exception as e:
        print(f"[ERROR] Error occurred during monitoring: {e}")
        logger.error(f"[ERROR] Error occurred during monitoring: {e}")
        raise

if __name__ == "__main__":
    main()