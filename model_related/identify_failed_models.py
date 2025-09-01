#!/usr/bin/env python3
"""
Identify Failed Models Script
Check which models need to be fixed and retrained
"""

import os
import glob
import json
from pathlib import Path
from datetime import datetime

def get_all_model_files():
    """Get all model files from models/ directory"""
    model_files = []
    models_dir = Path("models")
    
    for py_file in models_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            model_name = py_file.stem
            model_files.append(model_name)
    
    return sorted(model_files)

def check_training_status(model_name):
    """Check if a model has been successfully trained today"""
    today = datetime.now().strftime("%Y%m%d")
    checkpoint_pattern = f"checkpoints/{model_name}_{today}_*_best.pth"
    checkpoints = glob.glob(checkpoint_pattern)
    
    return {
        'model_name': model_name,
        'has_recent_checkpoint': len(checkpoints) > 0,
        'checkpoint_files': checkpoints,
        'trainer_exists': os.path.exists(f"trainers/train_{model_name}.py")
    }

def identify_failed_models():
    """Identify models that need to be fixed and retrained"""
    print("🔍 Identifying Failed Models")
    print("=" * 50)
    
    all_models = get_all_model_files()
    print(f"📊 Found {len(all_models)} model files")
    
    failed_models = []
    successful_models = []
    
    for model_name in all_models:
        status = check_training_status(model_name)
        
        if status['has_recent_checkpoint']:
            successful_models.append(status)
            print(f"✅ {model_name} - Training completed")
        else:
            failed_models.append(status)
            trainer_status = "✅" if status['trainer_exists'] else "❌"
            print(f"❌ {model_name} - No recent checkpoint (Trainer: {trainer_status})")
    
    print("\n" + "=" * 50)
    print("📋 Summary")
    print("=" * 50)
    print(f"✅ Successfully trained: {len(successful_models)}")
    print(f"❌ Need attention: {len(failed_models)}")
    
    if failed_models:
        print(f"\n🔧 Models that need fixing:")
        for model in failed_models:
            print(f"  - {model['model_name']}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(all_models),
        'successful_models': successful_models,
        'failed_models': failed_models,
        'summary': {
            'successful_count': len(successful_models),
            'failed_count': len(failed_models)
        }
    }
    
    with open('failed_models_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Detailed analysis saved to: failed_models_analysis.json")
    
    return failed_models

if __name__ == "__main__":
    failed_models = identify_failed_models()
    
    if failed_models:
        print(f"\n🚀 Ready to fix {len(failed_models)} models")
    else:
        print(f"\n🎉 All models have been successfully trained!")