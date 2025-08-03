"""
CoAtNet training monitor script.
Monitors training progress and provides status updates.
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def find_latest_coatnet_experiment():
    """Find the latest CoAtNet experiment directory."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None
    
    # Find all experiment directories with coatnet subdirectory
    coatnet_experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "coatnet").exists():
            coatnet_experiments.append(exp_dir)
    
    if not coatnet_experiments:
        return None
    
    # Sort by creation time and get the latest
    latest_exp = max(coatnet_experiments, key=lambda x: x.stat().st_ctime)
    return latest_exp / "coatnet"

def check_training_files(coatnet_dir):
    """Check for training files and their status."""
    if not coatnet_dir or not coatnet_dir.exists():
        return {"status": "not_found", "files": []}
    
    files_status = {}
    
    # Check for key training files
    key_files = [
        "training_history.json",
        "best_model.pth",
        "config.json",
        "final_model.pth"
    ]
    
    for file_name in key_files:
        file_path = coatnet_dir / file_name
        if file_path.exists():
            stat = file_path.stat()
            files_status[file_name] = {
                "exists": True,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            files_status[file_name] = {"exists": False}
    
    return files_status

def check_python_processes():
    """Check for running Python processes related to training."""
    training_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('train_coatnet' in arg for arg in cmdline):
                    training_processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': ' '.join(cmdline) if cmdline else 'N/A'
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return training_processes

def read_training_history(coatnet_dir):
    """Read training history if available."""
    history_file = coatnet_dir / "training_history.json"
    if not history_file.exists():
        return None
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history
    except Exception as e:
        return {"error": str(e)}

def main():
    """Main monitoring function."""
    print("🔍 CoAtNet Training Monitor")
    print("=" * 50)
    
    # Find latest experiment
    coatnet_dir = find_latest_coatnet_experiment()
    if not coatnet_dir:
        print("❌ No CoAtNet experiment found")
        return
    
    print(f"📁 Monitoring: {coatnet_dir}")
    print()
    
    # Check running processes
    processes = check_python_processes()
    if processes:
        print("🔄 Training processes found:")
        for proc in processes:
            print(f"   PID {proc['pid']}: {proc['cmdline']}")
    else:
        print("⚠️  No training processes detected")
    print()
    
    # Check training files
    files_status = check_training_files(coatnet_dir)
    print("📋 Training files status:")
    for file_name, status in files_status.items():
        if status["exists"]:
            print(f"   ✅ {file_name}: {status['size']} bytes (modified: {status['modified']})")
        else:
            print(f"   ❌ {file_name}: Not found")
    print()
    
    # Read training history if available
    history = read_training_history(coatnet_dir)
    if history:
        if "error" in history:
            print(f"⚠️  Error reading training history: {history['error']}")
        else:
            print("📊 Training Progress:")
            if "train_losses" in history:
                epochs_completed = len(history["train_losses"])
                print(f"   Epochs completed: {epochs_completed}")
                
                if epochs_completed > 0:
                    latest_train_loss = history["train_losses"][-1]
                    latest_val_loss = history["val_losses"][-1] if "val_losses" in history else "N/A"
                    latest_train_acc = history["train_accuracies"][-1] if "train_accuracies" in history else "N/A"
                    latest_val_acc = history["val_accuracies"][-1] if "val_accuracies" in history else "N/A"
                    
                    print(f"   Latest train loss: {latest_train_loss:.4f}")
                    print(f"   Latest val loss: {latest_val_loss}")
                    print(f"   Latest train acc: {latest_train_acc}")
                    print(f"   Latest val acc: {latest_val_acc}")
    else:
        print("📊 Training history not yet available")
    
    print()
    print("💡 Tips:")
    print("   - Training is running on CPU, which may take longer")
    print("   - Check terminal output for real-time progress")
    print("   - Training files will appear as epochs complete")

if __name__ == "__main__":
    main()