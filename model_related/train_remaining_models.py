#!/usr/bin/env python3
"""
Sequential training script for all remaining models
"""
import subprocess
import sys
import os

def run_training(script_name):
    """Run a training script and return success status"""
    print(f"\n🚀 Starting training: {script_name}")
    print("=" * 60)
    
    cmd = [
        sys.executable, 
        f"trainers/{script_name}"
    ]
    
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/aaa/ws/bioastModel'
    
    try:
        result = subprocess.run(cmd, cwd='/home/aaa/ws/bioastModel', env=env, 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ {script_name} failed with exception: {e}")
        return False

def main():
    """Train all remaining models"""
    print("🧬 BioAst Model Training Pipeline")
    print("📊 Using real biomedical data (13,024 images)")
    print("📐 Input size: 70x70 pixels")
    
    models_to_train = [
        "train_resnet_micro.py",
        "train_densenet_compact.py", 
        "train_inception_micro.py",
        "train_efficientnet_b0_micro.py"
    ]
    
    results = {}
    
    for model_script in models_to_train:
        success = run_training(model_script)
        results[model_script] = success
        
        if not success:
            print(f"\n⚠️  Training failed for {model_script}")
            print("Do you want to continue with the next model? (y/n)")
            # For automated execution, continue anyway
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TRAINING SUMMARY")
    print("=" * 60)
    
    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{script:<30} {status}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nOverall: {successful}/{total} models trained successfully")

if __name__ == "__main__":
    main()