#!/usr/bin/env python3
"""
Quick Training Script for All Models
Trains all available models in priority order for comparison
Complete inventory of 22 models optimized for 70x70 biomedical images
"""

import subprocess
import sys
import time
from pathlib import Path

# Complete model training configuration (22 models in 4 tiers)
TRAINING_PLAN = [
    # TIER 1: Immediate Training (Highest Priority) - Best for 70x70 images
    {
        'name': 'simplified_airbubble_detector',
        'batch_size': 128,
        'epochs': 30,
        'lr': 0.001,
        'tier': 1,
        'priority': 1,
        'description': 'Ultra-lightweight specialized detector (~100K params)',
        'expected_time': 5
    },
    {
        'name': 'micro_vit_tiny',
        'batch_size': 64,
        'epochs': 50,
        'lr': 0.0005,
        'tier': 1,
        'priority': 2,
        'description': 'Transformer optimized for small images (~1.8M params)',
        'expected_time': 12
    },
    {
        'name': 'mic_mobilenetv3',
        'batch_size': 64,
        'epochs': 50,
        'lr': 0.001,
        'tier': 1,
        'priority': 3,
        'description': 'Mobile-optimized with medical features (~2.5M params)',
        'expected_time': 10
    },
    {
        'name': 'shufflenet_v2_05x',
        'batch_size': 128,
        'epochs': 40,
        'lr': 0.001,
        'tier': 1,
        'priority': 4,
        'description': 'Very efficient baseline (~1.4M params)',
        'expected_time': 8
    },
    {
        'name': 'enhanced_airbubble_detector',
        'batch_size': 32,
        'epochs': 40,
        'lr': 0.001,
        'tier': 1,
        'priority': 5,
        'description': 'Advanced biomedical features (~2.5M params)',
        'expected_time': 15
    },
    
    # TIER 2: Secondary Training (High Priority) - Good performance-efficiency balance
    {
        'name': 'ghostnet',
        'batch_size': 64,
        'epochs': 40,
        'lr': 0.001,
        'tier': 2,
        'priority': 6,
        'description': 'Efficient ghost convolutions (~5.2M params)',
        'expected_time': 15
    },
    {
        'name': 'regnet_y400mf',
        'batch_size': 64,
        'epochs': 40,
        'lr': 0.001,
        'tier': 2,
        'priority': 7,
        'description': 'Design space optimized (~4.3M params)',
        'expected_time': 15
    },
    {
        'name': 'efficientnet_b0',
        'batch_size': 32,
        'epochs': 40,
        'lr': 0.001,
        'tier': 2,
        'priority': 8,
        'description': 'Proven efficient architecture (~5.3M params)',
        'expected_time': 18
    },
    {
        'name': 'mobilenet_v3_small',
        'batch_size': 64,
        'epochs': 40,
        'lr': 0.001,
        'tier': 2,
        'priority': 9,
        'description': 'Compact mobile architecture (~2.9M params)',
        'expected_time': 12
    },
    {
        'name': 'resnet18_improved',
        'batch_size': 32,
        'epochs': 40,
        'lr': 0.001,
        'tier': 2,
        'priority': 10,
        'description': 'Enhanced residual learning (~11.7M params)',
        'expected_time': 20
    },
    
    # TIER 3: Advanced Training (Medium Priority) - More complex models
    {
        'name': 'mnasnet_10',
        'batch_size': 32,
        'epochs': 50,
        'lr': 0.001,
        'tier': 3,
        'priority': 11,
        'description': 'NAS-optimized architecture (~4.4M params)',
        'expected_time': 15
    },
    {
        'name': 'shufflenet_v2_10x',
        'batch_size': 64,
        'epochs': 40,
        'lr': 0.001,
        'tier': 3,
        'priority': 12,
        'description': 'Balanced shuffle architecture (~2.3M params)',
        'expected_time': 10
    },
    {
        'name': 'regnet_x400mf',
        'batch_size': 32,
        'epochs': 40,
        'lr': 0.001,
        'tier': 3,
        'priority': 13,
        'description': 'Pure convolution design (~5.2M params)',
        'expected_time': 15
    },
    {
        'name': 'airbubble_hybrid_net',
        'batch_size': 16,
        'epochs': 50,
        'lr': 0.0005,
        'tier': 3,
        'priority': 14,
        'description': 'Hybrid CNN-Transformer (~8.5M params)',
        'expected_time': 25
    },
    {
        'name': 'mobilenet_v3_large',
        'batch_size': 32,
        'epochs': 40,
        'lr': 0.001,
        'tier': 3,
        'priority': 15,
        'description': 'Larger mobile architecture (~5.4M params)',
        'expected_time': 15
    },
    
    # TIER 4: Research/Comparison (Lower Priority) - Large models, may overfit
    {
        'name': 'vit_tiny',
        'batch_size': 16,
        'epochs': 50,
        'lr': 0.0005,
        'tier': 4,
        'priority': 16,
        'description': 'Standard transformer (~5.7M params)',
        'expected_time': 20
    },
    {
        'name': 'densenet121',
        'batch_size': 16,
        'epochs': 40,
        'lr': 0.001,
        'tier': 4,
        'priority': 17,
        'description': 'Dense connections (~8.0M params)',
        'expected_time': 25
    },
    {
        'name': 'efficientnet_v2_s',
        'batch_size': 16,
        'epochs': 40,
        'lr': 0.0005,
        'tier': 4,
        'priority': 18,
        'description': 'Advanced efficient architecture (~21.5M params)',
        'expected_time': 30
    },
    {
        'name': 'coatnet_small',
        'batch_size': 8,
        'epochs': 50,
        'lr': 0.0005,
        'tier': 4,
        'priority': 19,
        'description': 'Hybrid attention model (~25.0M params)',
        'expected_time': 40
    },
    {
        'name': 'resnet34_improved',
        'batch_size': 16,
        'epochs': 40,
        'lr': 0.001,
        'tier': 4,
        'priority': 20,
        'description': 'Deeper residual network (~21.8M params)',
        'expected_time': 30
    },
    {
        'name': 'densenet169',
        'batch_size': 8,
        'epochs': 40,
        'lr': 0.0005,
        'tier': 4,
        'priority': 21,
        'description': 'Very deep dense network (~14.1M params)',
        'expected_time': 35
    },
    {
        'name': 'convnext_tiny',
        'batch_size': 8,
        'epochs': 40,
        'lr': 0.0005,
        'tier': 4,
        'priority': 22,
        'description': 'Modern large-kernel CNN (~28.6M params)',
        'expected_time': 45
    }
]

def get_tier_models(tier):
    """Get models from a specific tier"""
    return [model for model in TRAINING_PLAN if model['tier'] == tier]

def run_training(model_config):
    """Run training for a single model"""
    print(f"\n{'='*70}")
    print(f"🚀 Training: {model_config['name']} (Tier {model_config['tier']}, Priority {model_config['priority']})")
    print(f"📝 Description: {model_config['description']}")
    print(f"⚙️ Config: batch_size={model_config['batch_size']}, epochs={model_config['epochs']}, lr={model_config['lr']}")
    print(f"⏱️ Expected time: ~{model_config['expected_time']} min/epoch")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, 'start_training.py',
        '--model', model_config['name'],
        '--batch_size', str(model_config['batch_size']),
        '--epochs', str(model_config['epochs']),
        '--lr', str(model_config['lr']),
        '--data_dir', 'bioast_dataset'
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        training_time = time.time() - start_time
        
        print(f"✅ {model_config['name']} training completed successfully!")
        print(f"⏱️ Actual training time: {training_time/60:.1f} minutes")
        
        # Extract key metrics from output
        output_lines = result.stdout.split('\n')
        for line in output_lines[-10:]:
            if 'Best validation accuracy' in line or 'Final test accuracy' in line:
                print(f"📊 {line.strip()}")
        
        return True, training_time
        
    except subprocess.CalledProcessError as e:
        training_time = time.time() - start_time
        print(f"❌ {model_config['name']} training failed!")
        print(f"Error: {e.stderr}")
        return False, training_time

def print_tier_summary():
    """Print summary of all tiers"""
    print(f"\n{'='*70}")
    print("📋 COMPLETE MODEL INVENTORY (22 Models)")
    print(f"{'='*70}")
    
    for tier in range(1, 5):
        tier_models = get_tier_models(tier)
        tier_names = {
            1: "Immediate Training (Highest Priority)",
            2: "Secondary Training (High Priority)", 
            3: "Advanced Training (Medium Priority)",
            4: "Research/Comparison (Lower Priority)"
        }
        
        print(f"\n🎯 TIER {tier}: {tier_names[tier]} ({len(tier_models)} models)")
        for model in tier_models:
            print(f"  {model['priority']:2d}. {model['name']} - {model['description']}")

def main():
    print("🧬 BioAst Comprehensive Model Training Pipeline")
    print("Training all 22 models in priority order for 70x70 biomedical images")
    
    # Check if dataset exists
    if not Path('bioast_dataset').exists():
        print("❌ Dataset directory 'bioast_dataset' not found!")
        print("Please upload your dataset first.")
        return
    
    # Check if training script exists
    if not Path('start_training.py').exists():
        print("❌ Training script 'start_training.py' not found!")
        return
    
    print_tier_summary()
    
    # Ask user which tiers to train
    print(f"\n🎯 TRAINING OPTIONS:")
    print("1. Train Tier 1 only (5 models, ~1 hour)")
    print("2. Train Tiers 1-2 (10 models, ~2.5 hours)")
    print("3. Train Tiers 1-3 (15 models, ~4 hours)")
    print("4. Train all tiers (22 models, ~6+ hours)")
    print("5. Train specific tier")
    print("6. Train all models (full pipeline)")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            models_to_train = get_tier_models(1)
        elif choice == '2':
            models_to_train = get_tier_models(1) + get_tier_models(2)
        elif choice == '3':
            models_to_train = get_tier_models(1) + get_tier_models(2) + get_tier_models(3)
        elif choice == '4' or choice == '6':
            models_to_train = TRAINING_PLAN
        elif choice == '5':
            tier = int(input("Enter tier number (1-4): "))
            if tier in [1, 2, 3, 4]:
                models_to_train = get_tier_models(tier)
            else:
                print("Invalid tier number!")
                return
        else:
            print("Invalid choice!")
            return
            
    except (ValueError, KeyboardInterrupt):
        print("\nTraining cancelled.")
        return
    
    results = []
    total_start_time = time.time()
    
    print(f"\n🚀 Starting training pipeline with {len(models_to_train)} models...")
    
    # Train each model
    for i, model_config in enumerate(models_to_train, 1):
        print(f"\n🎯 Step {i}/{len(models_to_train)}")
        
        success, training_time = run_training(model_config)
        
        results.append({
            'model': model_config['name'],
            'success': success,
            'training_time': training_time,
            'priority': model_config['priority'],
            'tier': model_config['tier'],
            'description': model_config['description']
        })
        
        if not success:
            print(f"⚠️ Continuing with next model...")
        
        # Short break between models
        time.sleep(2)
    
    # Summary
    total_time = time.time() - total_start_time
    successful_models = [r for r in results if r['success']]
    failed_models = [r for r in results if not r['success']]
    
    print(f"\n{'='*70}")
    print("📊 TRAINING PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"⏱️ Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
    print(f"✅ Successful: {len(successful_models)}/{len(models_to_train)} models")
    print(f"❌ Failed: {len(failed_models)} models")
    
    if successful_models:
        print(f"\n🏆 Successfully trained models:")
        for result in sorted(successful_models, key=lambda x: x['priority']):
            print(f"  ✅ {result['priority']:2d}. {result['model']} (Tier {result['tier']}) - {result['training_time']/60:.1f} min")
    
    if failed_models:
        print(f"\n❌ Failed models:")
        for result in sorted(failed_models, key=lambda x: x['priority']):
            print(f"  ❌ {result['priority']:2d}. {result['model']} (Tier {result['tier']})")
    
    # Tier-based summary
    print(f"\n📈 TIER PERFORMANCE SUMMARY:")
    for tier in range(1, 5):
        tier_results = [r for r in results if r['tier'] == tier]
        if tier_results:
            tier_success = [r for r in tier_results if r['success']]
            print(f"  Tier {tier}: {len(tier_success)}/{len(tier_results)} successful")
    
    print(f"\n💾 Check 'trained_models/' directory for saved models")
    print(f"📈 Check individual log files for detailed training metrics")
    print(f"📋 Review 'comprehensive_model_analysis.md' for detailed model specifications")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if len(successful_models) >= 5:
        print("1. Compare validation accuracies to identify top performers")
        print("2. Consider ensemble of top 3-5 models from different tiers")
        print("3. Run hyperparameter tuning on best Tier 1 model")
        print("4. Analyze feature maps from specialized biomedical models")
    elif len(successful_models) >= 3:
        print("1. Focus on successful models for production deployment")
        print("2. Debug failed models if they were high priority")
        print("3. Consider data augmentation for better generalization")
    elif len(successful_models) >= 1:
        print("1. Investigate why other models failed")
        print("2. Check dataset quality and preprocessing")
        print("3. Verify GPU memory and computational resources")
    else:
        print("1. Check dataset format and structure")
        print("2. Verify CUDA/GPU setup if using GPU")
        print("3. Review error logs for systematic issues")
        print("4. Consider starting with simplest Tier 1 models only")

if __name__ == "__main__":
    main()