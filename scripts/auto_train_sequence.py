"""
Automated Training Sequence Script.

This script monitors CoAtNet training completion and automatically
starts training the enhanced models in sequence.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def check_coatnet_completion():
    """Check if CoAtNet training is completed."""
    coatnet_dir = Path("experiments/experiment_20250803_032628/coatnet")
    
    if not coatnet_dir.exists():
        return False
    
    # Check for completion indicators
    final_model = coatnet_dir / "final_model.pth"
    training_history = coatnet_dir / "training_history.json"
    
    # If both files exist, training is likely complete
    if final_model.exists() and training_history.exists():
        return True
    
    # Alternative: check if training process is still running
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe"],
            capture_output=True, text=True, shell=True
        )
        
        # Count python processes (rough estimation)
        python_processes = result.stdout.count("python.exe")
        
        # If very few python processes, training might be done
        if python_processes <= 2:  # Only IDE and this script
            return True
            
    except Exception:
        pass
    
    return False

def train_enhanced_model(model_name: str):
    """Train a specific enhanced model."""
    print(f"\n🚀 Starting training for {model_name}")
    print("=" * 50)
    
    cmd = [
        "python", "scripts/train_enhanced_models.py",
        "--model", model_name,
        "--batch_size", "32",
        "--epochs", "50"
    ]
    
    try:
        # Run training
        result = subprocess.run(
            cmd, 
            cwd=Path.cwd(),
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {model_name} training completed successfully!")
            return True
        else:
            print(f"❌ {model_name} training failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        return False

def main():
    """Main function."""
    print("🔍 Automated Training Sequence Monitor")
    print("=" * 50)
    print("⏳ Waiting for CoAtNet training to complete...")
    
    # Enhanced models to train in sequence
    enhanced_models = [
        "mic_mobilenetv3",
        "micro_vit", 
        "airbubble_hybrid_net"
    ]
    
    # Monitor CoAtNet completion
    check_interval = 300  # Check every 5 minutes
    
    while True:
        if check_coatnet_completion():
            print("🎉 CoAtNet training appears to be completed!")
            break
        
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"⏰ {current_time} - CoAtNet still training, checking again in {check_interval//60} minutes...")
        time.sleep(check_interval)
    
    print("\n🚀 Starting enhanced models training sequence...")
    
    # Train enhanced models in sequence
    successful_models = []
    failed_models = []
    
    for model_name in enhanced_models:
        print(f"\n📋 Training queue: {model_name}")
        
        success = train_enhanced_model(model_name)
        
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # Brief pause between models
        time.sleep(10)
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 Training Sequence Summary")
    print("=" * 50)
    
    if successful_models:
        print("✅ Successfully trained models:")
        for model in successful_models:
            print(f"   - {model}")
    
    if failed_models:
        print("❌ Failed models:")
        for model in failed_models:
            print(f"   - {model}")
    
    print(f"\n📊 Total: {len(successful_models)}/{len(enhanced_models)} models trained successfully")
    
    if len(successful_models) == len(enhanced_models):
        print("🎉 All enhanced models trained successfully!")
        print("🔄 Ready for comprehensive model comparison!")
    else:
        print("⚠️ Some models failed to train. Check logs for details.")

if __name__ == "__main__":
    main()