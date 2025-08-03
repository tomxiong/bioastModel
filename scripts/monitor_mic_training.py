"""
MIC MobileNetV3 Training Monitor Script.

This script monitors the training progress and automatically starts
the next model training when current one completes.
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def find_latest_mic_experiment():
    """Find the latest MIC MobileNetV3 experiment directory."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None
    
    # Find all experiment directories with mic_mobilenetv3
    mic_dirs = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name.startswith("experiment_"):
            mic_dir = exp_dir / "mic_mobilenetv3"
            if mic_dir.exists():
                mic_dirs.append(mic_dir)
    
    if not mic_dirs:
        return None
    
    # Sort by creation time and get the latest
    latest_mic = max(mic_dirs, key=lambda x: x.stat().st_ctime)
    return latest_mic

def check_training_completion(mic_dir):
    """Check if MIC MobileNetV3 training is completed."""
    if not mic_dir or not mic_dir.exists():
        return False
    
    # Check for completion indicators
    training_history = mic_dir / "training_history.json"
    final_model = mic_dir / "final_model.pth"
    
    # If training history exists, check if training is complete
    if training_history.exists():
        try:
            with open(training_history, 'r') as f:
                history = json.load(f)
            
            # If we have training history, training is likely complete
            if len(history.get('train_loss', [])) > 0:
                print(f"✅ Training history found with {len(history['train_loss'])} epochs")
                return True
        except Exception as e:
            print(f"⚠️ Error reading training history: {e}")
    
    # Check if final model exists
    if final_model.exists():
        print("✅ Final model file found")
        return True
    
    return False

def get_training_progress(mic_dir):
    """Get current training progress."""
    if not mic_dir or not mic_dir.exists():
        return None
    
    training_history = mic_dir / "training_history.json"
    if training_history.exists():
        try:
            with open(training_history, 'r') as f:
                history = json.load(f)
            
            if history.get('val_acc'):
                latest_epoch = len(history['val_acc'])
                latest_val_acc = history['val_acc'][-1]
                latest_train_acc = history['train_acc'][-1] if history.get('train_acc') else 0
                
                return {
                    'epoch': latest_epoch,
                    'val_acc': latest_val_acc,
                    'train_acc': latest_train_acc
                }
        except Exception:
            pass
    
    return None

def start_next_training(model_name):
    """Start training for the next model."""
    print(f"\n🚀 Starting training for {model_name}")
    print("=" * 50)
    
    script_map = {
        'micro_vit': 'train_micro_vit.py',
        'airbubble_hybrid_net': 'train_airbubble_hybrid.py'
    }
    
    script_name = script_map.get(model_name)
    if not script_name:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    script_path = Path("scripts") / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        # Start training process
        cmd = [sys.executable, str(script_path)]
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"✅ Started {model_name} training (PID: {process.pid})")
        return True
        
    except Exception as e:
        print(f"❌ Error starting {model_name} training: {e}")
        return False

def main():
    """Main monitoring function."""
    print("🔍 MIC MobileNetV3 Training Monitor")
    print("=" * 50)
    
    # Training sequence
    training_queue = ['micro_vit', 'airbubble_hybrid_net']
    current_queue_index = 0
    
    check_interval = 60  # Check every minute
    last_progress = None
    
    print("⏳ Monitoring MIC MobileNetV3 training completion...")
    
    while True:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Find latest MIC experiment
        mic_dir = find_latest_mic_experiment()
        
        if not mic_dir:
            print(f"⏰ {current_time} - No MIC experiment found, waiting...")
            time.sleep(check_interval)
            continue
        
        # Check training progress
        progress = get_training_progress(mic_dir)
        if progress:
            if not last_progress or progress['epoch'] > last_progress.get('epoch', 0):
                print(f"📊 {current_time} - Epoch {progress['epoch']}: "
                      f"Val Acc: {progress['val_acc']:.4f}, "
                      f"Train Acc: {progress['train_acc']:.4f}")
                last_progress = progress
        
        # Check if training is complete
        if check_training_completion(mic_dir):
            print(f"🎉 {current_time} - MIC MobileNetV3 training completed!")
            
            # Get final results
            final_progress = get_training_progress(mic_dir)
            if final_progress:
                print(f"🏆 Final results: Epoch {final_progress['epoch']}, "
                      f"Val Acc: {final_progress['val_acc']:.4f}")
            
            # Start next training if available
            if current_queue_index < len(training_queue):
                next_model = training_queue[current_queue_index]
                success = start_next_training(next_model)
                
                if success:
                    current_queue_index += 1
                    print(f"🔄 Switched to monitoring {next_model} training...")
                    
                    # Reset monitoring for next model
                    last_progress = None
                    time.sleep(30)  # Give new training time to start
                    continue
                else:
                    print(f"❌ Failed to start {next_model} training")
                    break
            else:
                print("🎯 All enhanced models training completed!")
                break
        
        else:
            print(f"⏰ {current_time} - MIC MobileNetV3 still training...")
        
        time.sleep(check_interval)
    
    print("\n🎉 Training sequence monitoring completed!")

if __name__ == "__main__":
    main()