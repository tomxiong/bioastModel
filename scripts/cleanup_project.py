"""
Project cleanup and organization script.

This script helps organize the project by:
1. Identifying and removing duplicate/redundant files
2. Moving files to appropriate directories
3. Creating a clean project structure
4. Backing up important files before cleanup

Usage:
    python scripts/cleanup_project.py --dry-run  # Preview changes
    python scripts/cleanup_project.py --backup   # Create backup before cleanup
    python scripts/cleanup_project.py --force    # Execute cleanup
"""

import sys
import os
import argparse
import shutil
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Clean up and organize project structure')
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before cleanup'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Execute cleanup without confirmation'
    )
    
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary/debug files'
    )
    
    return parser.parse_args()

def analyze_project_files():
    """Analyze current project files and categorize them."""
    project_root = Path('.')
    
    file_categories = {
        'core_models': [],
        'core_training': [],
        'core_evaluation': [],
        'experiments': [],
        'reports': [],
        'scripts_unified': [],
        'scripts_legacy': [],
        'temp_files': [],
        'duplicate_files': [],
        'config_files': [],
        'documentation': []
    }
    
    # Define file patterns
    patterns = {
        'core_models': ['models/*.py'],
        'core_training': ['training/*.py'],
        'core_evaluation': ['training/evaluator.py', 'training/visualizer.py'],
        'experiments': ['experiments/**/*'],
        'reports': ['*.html', 'reports/**/*'],
        'scripts_unified': ['scripts/*.py'],
        'config_files': ['core/config/*.py', '*.json', '*.yaml', '*.yml'],
        'documentation': ['*.md', 'docs/**/*'],
        'temp_files': [
            '*_temp.py', '*_debug.py', '*_test.py', '*_backup.py',
            'fix_*.py', 'regenerate_*.py', 'create_*.py', 'generate_*.py',
            'comprehensive_*.py', 'enhanced_*.py', 'simple_*.py',
            'comparison_*.py', 'save_*.py', 'complete_*.py'
        ]
    }
    
    # Scan files
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            for file_path in project_root.glob(pattern):
                if file_path.is_file():
                    file_categories[category].append(file_path)
    
    # Identify duplicates by functionality
    duplicate_candidates = [
        'comparison_visualizations.py',
        'fixed_comparison_visualizations.py',
        'save_comparison_charts.py',
        'save_charts_direct.py',
        'comprehensive_report_generator.py',
        'comprehensive_report_html.py',
        'enhanced_report_html.py',
        'simple_report_html.py',
        'generate_resnet_report.py',
        'complete_resnet_evaluation.py',
        'complete_resnet_evaluation_unified.py',
        'fixed_resnet_report.py'
    ]
    
    for candidate in duplicate_candidates:
        file_path = project_root / candidate
        if file_path.exists():
            file_categories['duplicate_files'].append(file_path)
    
    return file_categories

def create_backup(file_categories):
    """Create backup of important files before cleanup."""
    backup_dir = Path('backup') / f"cleanup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Creating backup in: {backup_dir}")
    
    # Backup categories that might be modified
    categories_to_backup = ['scripts_legacy', 'duplicate_files', 'temp_files']
    
    backed_up_files = []
    for category in categories_to_backup:
        if file_categories[category]:
            category_backup_dir = backup_dir / category
            category_backup_dir.mkdir(exist_ok=True)
            
            for file_path in file_categories[category]:
                if file_path.exists():
                    backup_file_path = category_backup_dir / file_path.name
                    shutil.copy2(file_path, backup_file_path)
                    backed_up_files.append(str(file_path))
    
    # Create backup manifest
    manifest = {
        'backup_date': datetime.now().isoformat(),
        'backed_up_files': backed_up_files,
        'categories': {k: [str(f) for f in v] for k, v in file_categories.items()}
    }
    
    with open(backup_dir / 'backup_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Backup created with {len(backed_up_files)} files")
    return backup_dir

def create_cleanup_plan(file_categories):
    """Create a cleanup plan based on file analysis."""
    plan = {
        'files_to_remove': [],
        'files_to_move': [],
        'directories_to_create': [],
        'files_to_keep': []
    }
    
    # Directories to create
    plan['directories_to_create'] = [
        'archive/legacy_scripts',
        'archive/temp_files',
        'reports/individual',
        'reports/comparisons',
        'docs/models'
    ]
    
    # Files to remove (duplicates and temp files)
    files_to_remove = [
        'comparison_visualizations.py',  # Keep fixed_comparison_visualizations.py
        'save_comparison_charts.py',    # Keep save_charts_direct.py
        'comprehensive_report_generator.py',  # Redundant
        'comprehensive_report_html.py',       # Redundant
        'enhanced_report_html.py',            # Redundant
        'simple_report_html.py',              # Redundant
        'generate_resnet_report.py',          # Redundant
        'complete_resnet_evaluation.py',      # Keep unified version
        'fixed_resnet_report.py',             # Redundant
        'regenerate_visualizations.py',       # Temp file
        'fix_visualizations.py',              # Temp file
        'sample_analysis.py',                 # Integrated into evaluator
        'create_comprehensive_report.py',     # Redundant
        'train_resnet_improved.py',           # Use unified training
        'main_training.py',                   # Redundant
        'quick_start.py',                     # Redundant
        'run_training.py',                    # Redundant
        'check_status.py'                     # Redundant
    ]
    
    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            plan['files_to_remove'].append(file_path)
    
    # Files to move to archive
    files_to_archive = [
        ('fixed_comparison_visualizations.py', 'archive/legacy_scripts/'),
        ('save_charts_direct.py', 'archive/legacy_scripts/'),
        ('complete_resnet_evaluation_unified.py', 'archive/legacy_scripts/')
    ]
    
    for file_name, target_dir in files_to_archive:
        file_path = Path(file_name)
        if file_path.exists():
            plan['files_to_move'].append((file_path, Path(target_dir) / file_name))
    
    # Files to keep (core functionality)
    core_files = [
        'scripts/train_model.py',
        'scripts/evaluate_model.py',
        'scripts/compare_models.py',
        'scripts/add_new_model.py',
        'core/config/model_configs.py',
        'core/config/training_configs.py',
        'core/config/paths.py',
        'models/efficientnet.py',
        'models/resnet_improved.py',
        'training/dataset.py',
        'training/trainer.py',
        'training/evaluator.py',
        'training/visualizer.py'
    ]
    
    for file_name in core_files:
        file_path = Path(file_name)
        if file_path.exists():
            plan['files_to_keep'].append(file_path)
    
    return plan

def execute_cleanup_plan(plan, dry_run=True):
    """Execute the cleanup plan."""
    if dry_run:
        print("📋 CLEANUP PLAN (DRY RUN)")
        print("=" * 50)
    else:
        print("🧹 EXECUTING CLEANUP")
        print("=" * 50)
    
    # Create directories
    print(f"\n📁 Directories to create: {len(plan['directories_to_create'])}")
    for dir_path in plan['directories_to_create']:
        print(f"  + {dir_path}")
        if not dry_run:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Remove files
    print(f"\n🗑️  Files to remove: {len(plan['files_to_remove'])}")
    for file_path in plan['files_to_remove']:
        print(f"  - {file_path}")
        if not dry_run and file_path.exists():
            file_path.unlink()
    
    # Move files
    print(f"\n📦 Files to move: {len(plan['files_to_move'])}")
    for source, target in plan['files_to_move']:
        print(f"  {source} -> {target}")
        if not dry_run and source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    
    # Keep files
    print(f"\n✅ Files to keep: {len(plan['files_to_keep'])}")
    for file_path in plan['files_to_keep']:
        print(f"  ✓ {file_path}")
    
    if not dry_run:
        print(f"\n✅ Cleanup completed successfully!")

def create_project_summary():
    """Create a summary of the cleaned project structure."""
    summary = f"""# Project Structure Summary

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Core Structure

### Models (`models/`)
- `efficientnet.py` - EfficientNet-B0 implementation
- `resnet_improved.py` - Improved ResNet-18 implementation

### Training System (`training/`)
- `dataset.py` - Data loading and preprocessing
- `trainer.py` - Training loop and optimization
- `evaluator.py` - Model evaluation and metrics
- `visualizer.py` - Visualization and reporting

### Configuration (`core/config/`)
- `model_configs.py` - Model configurations
- `training_configs.py` - Training parameters
- `paths.py` - Path management

### Scripts (`scripts/`)
- `train_model.py` - Unified training script
- `evaluate_model.py` - Unified evaluation script
- `compare_models.py` - Model comparison script
- `add_new_model.py` - New model template generator
- `cleanup_project.py` - Project organization script

### Experiments (`experiments/`)
- Contains all training experiments and results
- Organized by timestamp and model name

### Reports (`reports/`)
- `individual/` - Individual model reports
- `comparisons/` - Model comparison reports

### Archive (`archive/`)
- `legacy_scripts/` - Archived legacy scripts
- `temp_files/` - Temporary files backup

## Usage Workflows

### Training a New Model
1. `python scripts/add_new_model.py --name model_name`
2. Implement model architecture
3. `python scripts/train_model.py --model model_name`

### Evaluating a Model
1. `python scripts/evaluate_model.py --model model_name`

### Comparing Models
1. `python scripts/compare_models.py --models model1 model2`

### Adding New Models
1. Use the standardized template system
2. Follow the established directory structure
3. Integrate with existing evaluation pipeline

## Best Practices
- Use unified scripts for consistency
- Follow naming conventions
- Document new models and changes
- Regular cleanup and organization
"""
    
    with open('PROJECT_STRUCTURE.md', 'w') as f:
        f.write(summary)
    
    print("📖 Created PROJECT_STRUCTURE.md")

def main():
    """Main function."""
    args = parse_arguments()
    
    print("🧹 Project Cleanup and Organization")
    print("=" * 50)
    
    try:
        # Analyze current project files
        print("🔍 Analyzing project files...")
        file_categories = analyze_project_files()
        
        # Show analysis results
        print("\n📊 File Analysis Results:")
        for category, files in file_categories.items():
            if files:
                print(f"  {category}: {len(files)} files")
        
        # Create backup if requested
        backup_dir = None
        if args.backup and not args.dry_run:
            backup_dir = create_backup(file_categories)
        
        # Create cleanup plan
        print("\n📋 Creating cleanup plan...")
        cleanup_plan = create_cleanup_plan(file_categories)
        
        # Execute cleanup plan
        if args.dry_run:
            execute_cleanup_plan(cleanup_plan, dry_run=True)
        elif args.force or input("\n❓ Execute cleanup plan? (y/N): ").lower() == 'y':
            execute_cleanup_plan(cleanup_plan, dry_run=False)
            
            # Create project summary
            create_project_summary()
            
            if backup_dir:
                print(f"💾 Backup available at: {backup_dir}")
        else:
            print("❌ Cleanup cancelled")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()