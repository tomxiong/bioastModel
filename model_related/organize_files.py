#!/usr/bin/env python3
"""
文件清理和组织脚本
根据核心功能将文件分类到不同的目录中：
- model_related/: 模型相关文件
- test_related/: 测试相关文件  
- training_related/: 训练相关文件
- reports/: 报告文件
- config/: 配置文件
- analysis/: 分析文件
- cleanup/: 清理和修复文件
"""

import os
import shutil
from pathlib import Path
import json
from typing import Dict, List, Tuple
import re

class FileOrganizer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.categories = {
            'model_related': {
                'description': '模型定义、配置和转换相关文件',
                'patterns': [
                    r'.*model.*\.py$',
                    r'.*convert.*\.py$',
                    r'.*onnx.*\.py$',
                    r'^model_.*\.py$',
                    r'.*_model\.py$',
                    r'models/.*',
                    r'converters/.*',
                    r'deployment/.*'
                ],
                'files': []
            },
            'test_related': {
                'description': '测试、验证和评估相关文件',
                'patterns': [
                    r'.*test.*\.py$',
                    r'.*validate.*\.py$',
                    r'.*check.*\.py$',
                    r'.*verify.*\.py$',
                    r'.*evaluate.*\.py$',
                    r'.*benchmark.*\.py$',
                    r'test_.*\.py$',
                    r'.*_test\.py$',
                    r'evaluation/.*',
                    r'quick_test_evaluation/.*'
                ],
                'files': []
            },
            'training_related': {
                'description': '训练、监控和优化相关文件',
                'patterns': [
                    r'.*train.*\.py$',
                    r'.*training.*\.py$',
                    r'.*monitor.*\.py$',
                    r'.*optimize.*\.py$',
                    r'.*fix.*\.py$',
                    r'.*continue.*\.py$',
                    r'.*start.*\.py$',
                    r'train_.*\.py$',
                    r'.*_train\.py$',
                    r'training/.*',
                    r'trainers/.*',
                    r'experiments/.*'
                ],
                'files': []
            },
            'analysis': {
                'description': '分析和报告生成相关文件',
                'patterns': [
                    r'.*analyze.*\.py$',
                    r'.*analysis.*\.py$',
                    r'.*report.*\.py$',
                    r'.*generate.*\.py$',
                    r'.*compare.*\.py$',
                    r'.*extract.*\.py$',
                    r'.*summary.*\.py$',
                    r'analyze_.*\.py$',
                    r'.*_analysis\.py$',
                    r'analysis/.*',
                    r'reports/.*',
                    r'dashboard/.*'
                ],
                'files': []
            },
            'config': {
                'description': '配置文件',
                'patterns': [
                    r'.*\.yaml$',
                    r'.*\.yml$',
                    r'.*\.json$',
                    r'.*\.toml$',
                    r'.*config.*\.py$',
                    r'config_.*\.py$',
                    r'.*_config\.py$',
                    r'core/config/.*'
                ],
                'files': []
            },
            'documentation': {
                'description': '文档文件',
                'patterns': [
                    r'.*\.md$',
                    r'.*\.html$',
                    r'.*\.txt$',
                    r'docs/.*',
                    r'documentation/.*'
                ],
                'files': []
            },
            'cleanup': {
                'description': '清理和维护相关文件',
                'patterns': [
                    r'.*cleanup.*\.py$',
                    r'.*fix.*\.py$',
                    r'.*repair.*\.py$',
                    r'.*debug.*\.py$',
                    r'cleanup_.*\.py$',
                    r'.*_cleanup\.py$'
                ],
                'files': []
            },
            'core': {
                'description': '核心功能文件',
                'patterns': [
                    r'^core/.*',
                    r'^utils/.*',
                    r'^workflow/.*',
                    r'^memory-bank/.*',
                    r'^templates/.*'
                ],
                'files': []
            },
            'dataset': {
                'description': '数据集相关文件',
                'patterns': [
                    r'.*dataset.*\.py$',
                    r'dataset_.*\.py$',
                    r'.*_dataset\.py$',
                    r'bioast_dataset/.*',
                    r'dataset/.*'
                ],
                'files': []
            }
        }
        
        # 排除的文件和目录
        self.exclude_patterns = [
            r'^\.venv.*',
            r'^\.git.*',
            r'^__pycache__.*',
            r'^\.pyc.*',
            r'^\.DS_Store.*',
            r'^\.idea.*',
            r'^\.vscode.*',
            r'^node_modules.*',
            r'^build.*',
            r'^dist.*',
            r'^\.coverage.*'
        ]
        
        # 核心文件（保留在根目录）
        self.core_files = [
            'README.md',
            'CLAUDE.md', 
            'requirements.txt',
            '.gitignore',
            'main.py',
            'quick_start.py',
            'dataset_manager.py',
            'model_manager.py',
            'train_single_model.py',
            'compare_models.py',
            'config_template.yaml',
            'training_config.json',
            'model_registry.json'
        ]
    
    def should_exclude(self, file_path: Path) -> bool:
        """检查文件是否应该被排除"""
        for pattern in self.exclude_patterns:
            if re.match(pattern, file_path.name):
                return True
        return False
    
    def categorize_file(self, file_path: Path) -> str:
        """根据文件名和路径分类文件"""
        if self.should_exclude(file_path):
            return 'exclude'
        
        # 检查是否为核心文件
        if file_path.name in self.core_files:
            return 'core'
        
        # 根据模式分类
        for category, info in self.categories.items():
            for pattern in info['patterns']:
                if re.match(pattern, file_path.name, re.IGNORECASE):
                    return category
                if re.match(pattern, str(file_path), re.IGNORECASE):
                    return category
        
        return 'uncategorized'
    
    def scan_files(self) -> Dict[str, List[Path]]:
        """扫描所有文件并分类"""
        categorized_files = {}
        
        for category in self.categories.keys():
            categorized_files[category] = []
        
        categorized_files['core'] = []
        categorized_files['exclude'] = []
        categorized_files['uncategorized'] = []
        
        # 扫描根目录文件
        for file_path in self.root_path.iterdir():
            if file_path.is_file():
                category = self.categorize_file(file_path)
                categorized_files[category].append(file_path)
        
        # 扫描子目录
        for dir_path in self.root_path.iterdir():
            if dir_path.is_dir() and not self.should_exclude(dir_path):
                category = self.categorize_file(dir_path)
                if category != 'exclude':
                    categorized_files[category].append(dir_path)
        
        return categorized_files
    
    def create_directories(self):
        """创建分类目录"""
        for category in self.categories.keys():
            category_dir = self.root_path / category
            category_dir.mkdir(exist_ok=True)
            print(f"创建目录: {category_dir}")
    
    def move_files(self, categorized_files: Dict[str, List[Path]]):
        """移动文件到对应目录"""
        for category, files in categorized_files.items():
            if category == 'exclude' or category == 'core':
                continue
                
            category_dir = self.root_path / category
            
            for file_path in files:
                try:
                    target_path = category_dir / file_path.name
                    
                    # 如果目标文件已存在，添加序号
                    if target_path.exists():
                        counter = 1
                        while target_path.exists():
                            stem = file_path.stem
                            suffix = file_path.suffix
                            target_path = category_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    shutil.move(str(file_path), str(target_path))
                    print(f"移动: {file_path} -> {target_path}")
                    
                except Exception as e:
                    print(f"移动失败 {file_path}: {e}")
    
    def generate_report(self, categorized_files: Dict[str, List[Path]]):
        """生成分类报告"""
        report = {
            'timestamp': str(os.path.getmtime(self.root_path)),
            'total_files': sum(len(files) for files in categorized_files.values()),
            'categories': {}
        }
        
        for category, files in categorized_files.items():
            report['categories'][category] = {
                'count': len(files),
                'description': self.categories.get(category, {}).get('description', 'Unknown'),
                'files': [str(f.name) for f in files]
            }
        
        # 保存报告
        report_path = self.root_path / 'organization_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"生成报告: {report_path}")
        
        # 打印摘要
        print("\n=== 文件分类摘要 ===")
        for category, info in report['categories'].items():
            if info['count'] > 0:
                print(f"{category}: {info['count']} 个文件 - {info['description']}")
    
    def organize(self):
        """执行文件整理"""
        print("开始文件整理...")
        
        # 创建目录
        self.create_directories()
        
        # 扫描文件
        categorized_files = self.scan_files()
        
        # 生成报告
        self.generate_report(categorized_files)
        
        # 显示预览
        print("\n=== 文件移动预览 ===")
        for category, files in categorized_files.items():
            if category in ['exclude', 'core']:
                continue
            if files:
                print(f"\n{category} ({len(files)} 个文件):")
                for file_path in files[:5]:  # 只显示前5个
                    print(f"  - {file_path.name}")
                if len(files) > 5:
                    print(f"  ... 还有 {len(files) - 5} 个文件")
        
        # 自动执行移动
        print("\n开始移动文件...")
        self.move_files(categorized_files)
        print("文件整理完成！")
        
        return categorized_files

def main():
    """主函数"""
    organizer = FileOrganizer('/home/aaa/ws/bioastModel')
    organizer.organize()

if __name__ == "__main__":
    main()