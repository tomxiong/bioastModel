"""
监控增强型气孔检测器训练进展
"""

import os
import time
import json
from datetime import datetime

def monitor_training_progress():
    """监控训练进展"""
    print("🔍 监控增强型气孔检测器训练进展...")
    print("=" * 60)
    
    # 检查训练目录
    training_dir = "experiments/enhanced_airbubble_detector"
    
    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 检查训练状态...")
        
        # 检查是否创建了训练目录
        if os.path.exists(training_dir):
            print(f"✅ 训练目录已创建: {training_dir}")
            
            # 列出目录内容
            try:
                files = os.listdir(training_dir)
                if files:
                    print(f"📁 训练文件:")
                    for file in sorted(files):
                        file_path = os.path.join(training_dir, file)
                        if os.path.isfile(file_path):
                            size = os.path.getsize(file_path)
                            mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                            print(f"   {file} ({size} bytes, {mtime.strftime('%H:%M:%S')})")
                        else:
                            print(f"   📂 {file}/")
                    
                    # 检查日志文件
                    log_files = [f for f in files if f.endswith('.log')]
                    if log_files:
                        latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(training_dir, x)))
                        log_path = os.path.join(training_dir, latest_log)
                        
                        print(f"\n📋 最新日志内容 ({latest_log}):")
                        try:
                            with open(log_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                # 显示最后10行
                                for line in lines[-10:]:
                                    print(f"   {line.strip()}")
                        except Exception as e:
                            print(f"   ❌ 读取日志失败: {e}")
                    
                    # 检查训练历史
                    history_files = [f for f in files if f.endswith('_history.json')]
                    if history_files:
                        latest_history = max(history_files, key=lambda x: os.path.getmtime(os.path.join(training_dir, x)))
                        history_path = os.path.join(training_dir, latest_history)
                        
                        try:
                            with open(history_path, 'r') as f:
                                history = json.load(f)
                                
                            if history.get('val_acc'):
                                current_acc = history['val_acc'][-1] if history['val_acc'] else 0
                                best_acc = max(history['val_acc']) if history['val_acc'] else 0
                                epochs = len(history['val_acc'])
                                
                                print(f"\n📊 训练进展:")
                                print(f"   当前轮次: {epochs}")
                                print(f"   当前验证准确率: {current_acc:.2f}%")
                                print(f"   最佳验证准确率: {best_acc:.2f}%")
                                print(f"   目标准确率: 92.0%")
                                print(f"   进展: {'✅ 已达标' if best_acc >= 92.0 else f'🎯 还需提升 {92.0 - best_acc:.1f}%'}")
                        except Exception as e:
                            print(f"   ❌ 读取训练历史失败: {e}")
                else:
                    print("📁 训练目录为空，等待文件生成...")
            except Exception as e:
                print(f"❌ 访问训练目录失败: {e}")
        else:
            print("⏳ 训练目录尚未创建，等待训练启动...")
        
        # 检查Python进程
        try:
            import psutil
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
                try:
                    if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        if 'train_enhanced_airbubble_detector' in cmdline:
                            python_processes.append({
                                'pid': proc.info['pid'],
                                'cpu': proc.info['cpu_percent'],
                                'memory': proc.info['memory_info'].rss / 1024 / 1024,  # MB
                                'cmdline': cmdline
                            })
                except Exception:
                    continue
            
            if python_processes:
                print(f"\n🐍 Python训练进程:")
                for proc in python_processes:
                    print(f"   PID: {proc['pid']}, CPU: {proc['cpu']:.1f}%, 内存: {proc['memory']:.1f}MB")
            else:
                print("\n⚠️ 未发现气孔检测器训练进程")
        except ImportError:
            print("\n📝 提示: 安装psutil可获得更详细的进程监控")
        except Exception as e:
            print(f"\n❌ 进程检查失败: {e}")
        
        print("\n" + "=" * 60)
        print("⏰ 等待30秒后继续监控... (Ctrl+C 停止监控)")
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n🛑 监控已停止")
            break

if __name__ == "__main__":
    monitor_training_progress()