#!/usr/bin/env python3
"""
Update all training scripts to use real biomedical data
"""

import os
import glob

def update_trainer_file(filepath):
    """Update a single trainer file to use real data"""
    print(f"Updating {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace imports
    if 'from core.data_loader import create_data_loaders' in content:
        content = content.replace(
            'from core.data_loader import create_data_loaders',
            'from core.real_data_loader import create_real_data_loaders'
        )
    
    if 'from core.data_loader import MICDataLoader' in content:
        content = content.replace(
            'from core.data_loader import MICDataLoader',
            '# MICDataLoader not needed for real data'
        )
    
    # Replace data loading code
    old_data_loading = '''    # Get data loaders
    print("\\nLoading data...")
    data_loader = MICDataLoader()
    train_loader, val_loader, test_loader = create_data_loaders(
        data_loader,
        batch_size=32,
        num_workers=4
    )'''
    
    new_data_loading = '''    # Get data loaders
    print("\\nLoading real biomedical data...")
    train_loader, val_loader, test_loader = create_real_data_loaders(
        batch_size=32,
        num_workers=4
    )'''
    
    if old_data_loading in content:
        content = content.replace(old_data_loading, new_data_loading)
    
    # Alternative pattern
    alt_old_pattern = '''    data_loader = MICDataLoader()
    train_loader, val_loader, test_loader = create_data_loaders(
        data_loader,
        batch_size=32,
        num_workers=4
    )'''
    
    alt_new_pattern = '''    train_loader, val_loader, test_loader = create_real_data_loaders(
        batch_size=32,
        num_workers=4
    )'''
    
    if alt_old_pattern in content:
        content = content.replace(alt_old_pattern, alt_new_pattern)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {filepath}")

def main():
    print("🔄 Updating all training scripts to use real biomedical data...")
    
    # Find all training scripts
    trainer_files = glob.glob('trainers/train_*.py')
    
    for trainer_file in trainer_files:
        if os.path.exists(trainer_file):
            update_trainer_file(trainer_file)
    
    print(f"\n✅ Updated {len(trainer_files)} training scripts")
    print("🎉 All trainers now use real biomedical data!")

if __name__ == "__main__":
    main()