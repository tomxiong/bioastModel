"""
简化版气孔检测器训练监控脚本
实时跟踪训练进展和性能指标
"""

import os
import time
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

class SimplifiedDetectorMonitor:
    """简化版气孔检测器训练监控器"""
    
    def __init__(self):
        self.save_dir = "experiments/simplified_airbubble_detector"
        self.log_pattern = r"simplified_training_(\d{8}_\d{6})\.log"
        self.current_log_file = None
        self.last_position = 0
        self.training_data = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': [],
            'train_val_gaps': []
        }
        self.best_val_acc = 0.0
        self.target_accuracy = 92.0
        
    def find_latest_log_file(self) -> Optional[str]:
        """查找最新的训练日志文件"""
        if not os.path.exists(self.save_dir):
            return None
            
        log_files = []
        for file in os.listdir(self.save_dir):
            match = re.match(self.log_pattern, file)
            if match:
                timestamp = match.group(1)
                log_files.append((timestamp, file))
        
        if not log_files:
            return None
            
        # 按时间戳排序，返回最新的
        log_files.sort(reverse=True)
        latest_file = os.path.join(self.save_dir, log_files[0][1])
        return latest_file
    
    def parse_log_line(self, line: str) -> Optional[Dict]:
        """解析日志行，提取训练指标"""
        # 解析训练指标行
        train_pattern = r"Train Loss: ([\d.]+), Train Acc: ([\d.]+)%"
        val_pattern = r"Val Loss: ([\d.]+), Val Acc: ([\d.]+)%, Val F1: ([\d.]+)%"
        lr_pattern = r"Learning Rate: ([\d.e-]+)"
        gap_pattern = r"Train/Val Gap: ([-\d.]+)%"
        epoch_pattern = r"Epoch (\d+)/\d+"
        best_pattern = r"New best validation accuracy: ([\d.]+)%"
        
        result = {}
        
        # 检查是否是新的epoch开始
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            result['type'] = 'epoch_start'
            result['epoch'] = int(epoch_match.group(1))
            return result
        
        # 检查训练指标
        train_match = re.search(train_pattern, line)
        if train_match:
            result['type'] = 'train_metrics'
            result['train_loss'] = float(train_match.group(1))
            result['train_acc'] = float(train_match.group(2))
            return result
        
        # 检查验证指标
        val_match = re.search(val_pattern, line)
        if val_match:
            result['type'] = 'val_metrics'
            result['val_loss'] = float(val_match.group(1))
            result['val_acc'] = float(val_match.group(2))
            result['val_f1'] = float(val_match.group(3))
            return result
        
        # 检查学习率
        lr_match = re.search(lr_pattern, line)
        if lr_match:
            result['type'] = 'learning_rate'
            result['lr'] = float(lr_match.group(1))
            return result
        
        # 检查训练/验证差距
        gap_match = re.search(gap_pattern, line)
        if gap_match:
            result['type'] = 'train_val_gap'
            result['gap'] = float(gap_match.group(1))
            return result
        
        # 检查最佳验证准确率
        best_match = re.search(best_pattern, line)
        if best_match:
            result['type'] = 'best_val_acc'
            result['best_val_acc'] = float(best_match.group(1))
            return result
        
        return None
    
    def update_training_data(self, parsed_data: Dict):
        """更新训练数据"""
        data_type = parsed_data['type']
        
        if data_type == 'epoch_start':
            # 新的epoch开始，准备记录数据
            self.current_epoch = parsed_data['epoch']
            
        elif data_type == 'train_metrics':
            self.current_train_loss = parsed_data['train_loss']
            self.current_train_acc = parsed_data['train_acc']
            
        elif data_type == 'val_metrics':
            self.current_val_loss = parsed_data['val_loss']
            self.current_val_acc = parsed_data['val_acc']
            self.current_val_f1 = parsed_data['val_f1']
            
        elif data_type == 'learning_rate':
            self.current_lr = parsed_data['lr']
            
        elif data_type == 'train_val_gap':
            self.current_gap = parsed_data['gap']
            
            # 当获得gap信息时，说明这个epoch的所有数据都齐全了
            self.training_data['epochs'].append(self.current_epoch)
            self.training_data['train_loss'].append(self.current_train_loss)
            self.training_data['train_acc'].append(self.current_train_acc)
            self.training_data['val_loss'].append(self.current_val_loss)
            self.training_data['val_acc'].append(self.current_val_acc)
            self.training_data['val_f1'].append(self.current_val_f1)
            self.training_data['learning_rates'].append(self.current_lr)
            self.training_data['train_val_gaps'].append(self.current_gap)
            
        elif data_type == 'best_val_acc':
            self.best_val_acc = parsed_data['best_val_acc']
    
    def read_new_log_content(self) -> List[str]:
        """读取日志文件的新内容"""
        if not self.current_log_file or not os.path.exists(self.current_log_file):
            return []
        
        try:
            with open(self.current_log_file, 'r', encoding='utf-8') as f:
                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = f.tell()
                
                if new_content:
                    return new_content.strip().split('\n')
                return []
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
    
    def display_current_status(self):
        """显示当前训练状态"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🔍 简化版气孔检测器训练监控")
        print("=" * 60)
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"⏰ 监控时间: {current_time}")
        
        if self.current_log_file:
            print(f"📁 日志文件: {os.path.basename(self.current_log_file)}")
        else:
            print("📁 日志文件: 未找到")
            return
        
        if not self.training_data['epochs']:
            print("📊 状态: 等待训练数据...")
            return
        
        # 显示最新训练状态
        latest_epoch = self.training_data['epochs'][-1]
        latest_train_acc = self.training_data['train_acc'][-1]
        latest_val_acc = self.training_data['val_acc'][-1]
        latest_val_f1 = self.training_data['val_f1'][-1]
        latest_gap = self.training_data['train_val_gaps'][-1]
        latest_lr = self.training_data['learning_rates'][-1]
        
        print(f"📊 当前轮次: {latest_epoch}")
        print(f"🎯 训练准确率: {latest_train_acc:.2f}%")
        print(f"✅ 验证准确率: {latest_val_acc:.2f}%")
        print(f"📈 验证F1分数: {latest_val_f1:.2f}%")
        print(f"📉 训练/验证差距: {latest_gap:.2f}%")
        print(f"⚡ 学习率: {latest_lr:.6f}")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.2f}%")
        
        # 目标达成进度
        progress = (self.best_val_acc / self.target_accuracy) * 100
        print(f"🎯 目标进度: {progress:.1f}% (目标: {self.target_accuracy}%)")
        
        # 显示进度条
        bar_length = 40
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"📊 [{bar}] {progress:.1f}%")
        
        # 显示改进情况
        if self.best_val_acc > 52.0:  # 原始模型准确率
            improvement = self.best_val_acc - 52.0
            print(f"📈 相比原始模型改进: +{improvement:.2f}%")
        
        # 显示过拟合控制情况
        if abs(latest_gap) < 10:
            print("✅ 过拟合控制: 良好")
        elif abs(latest_gap) < 20:
            print("⚠️ 过拟合控制: 一般")
        else:
            print("❌ 过拟合控制: 需要注意")
        
        # 显示训练趋势
        if len(self.training_data['val_acc']) >= 3:
            recent_trend = np.mean(self.training_data['val_acc'][-3:]) - np.mean(self.training_data['val_acc'][-6:-3]) if len(self.training_data['val_acc']) >= 6 else 0
            if recent_trend > 1:
                print("📈 验证准确率趋势: 上升")
            elif recent_trend < -1:
                print("📉 验证准确率趋势: 下降")
            else:
                print("➡️ 验证准确率趋势: 稳定")
        
        print("=" * 60)
        print("💡 提示: Ctrl+C 停止监控")
    
    def save_monitoring_data(self):
        """保存监控数据"""
        if not self.training_data['epochs']:
            return
        
        # 保存训练数据
        data_file = os.path.join(self.save_dir, "monitoring_data.json")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump({
                'training_data': self.training_data,
                'best_val_acc': self.best_val_acc,
                'target_accuracy': self.target_accuracy,
                'last_update': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        # 生成训练曲线图
        self.generate_training_curves()
    
    def generate_training_curves(self):
        """生成训练曲线图"""
        if len(self.training_data['epochs']) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_data['epochs']
        
        # 准确率曲线
        ax1.plot(epochs, self.training_data['train_acc'], 'b-', label='训练准确率', linewidth=2)
        ax1.plot(epochs, self.training_data['val_acc'], 'r-', label='验证准确率', linewidth=2)
        ax1.axhline(y=self.target_accuracy, color='g', linestyle='--', label=f'目标准确率 ({self.target_accuracy}%)')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_title('训练和验证准确率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        ax2.plot(epochs, self.training_data['train_loss'], 'b-', label='训练损失', linewidth=2)
        ax2.plot(epochs, self.training_data['val_loss'], 'r-', label='验证损失', linewidth=2)
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('损失')
        ax2.set_title('训练和验证损失')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 训练/验证差距
        ax3.plot(epochs, self.training_data['train_val_gaps'], 'purple', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='过拟合警戒线')
        ax3.axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
        ax3.set_xlabel('轮次')
        ax3.set_ylabel('差距 (%)')
        ax3.set_title('训练/验证准确率差距')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax4.plot(epochs, self.training_data['learning_rates'], 'green', linewidth=2)
        ax4.set_xlabel('轮次')
        ax4.set_ylabel('学习率')
        ax4.set_title('学习率调度')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        curves_file = os.path.join(self.save_dir, "monitoring_curves.png")
        plt.savefig(curves_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def monitor(self):
        """开始监控"""
        print("🚀 启动简化版气孔检测器训练监控...")
        
        try:
            while True:
                # 查找最新日志文件
                latest_log = self.find_latest_log_file()
                
                if latest_log != self.current_log_file:
                    self.current_log_file = latest_log
                    self.last_position = 0
                    print(f"📁 发现新的日志文件: {latest_log}")
                
                if self.current_log_file:
                    # 读取新的日志内容
                    new_lines = self.read_new_log_content()
                    
                    # 解析新的日志行
                    for line in new_lines:
                        if line.strip():
                            parsed = self.parse_log_line(line)
                            if parsed:
                                self.update_training_data(parsed)
                
                # 显示当前状态
                self.display_current_status()
                
                # 保存监控数据
                self.save_monitoring_data()
                
                # 等待下次更新
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 监控已停止")
            self.save_monitoring_data()
            print(f"📊 监控数据已保存到: {self.save_dir}")

def main():
    """主函数"""
    monitor = SimplifiedDetectorMonitor()
    monitor.monitor()

if __name__ == "__main__":
    main()