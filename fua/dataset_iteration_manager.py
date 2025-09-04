"""
FUA 数据集迭代管理模块
支持数据集版本控制、增量更新和智能分析
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from PIL import Image
import numpy as np


class DatasetVersionManager:
    """数据集版本管理器"""
    
    def __init__(self, base_path: str = "bioast_dataset"):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / "versions"
        self.metadata_path = self.base_path / "metadata.json"
        self.versions_path.mkdir(exist_ok=True)
        self._load_metadata()
    
    def _load_metadata(self):
        """加载元数据"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "versions": [],
                "current_version": "v1.0",
                "stats": {
                    "total_images": 0,
                    "positive_samples": 0,
                    "negative_samples": 0
                }
            }
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_version(self, version_name: str, description: str = "") -> Dict:
        """创建新版本"""
        version_data = {
            "version": version_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "parent": self.metadata["current_version"],
            "stats": self._calculate_dataset_stats()
        }
        
        # 创建版本目录
        version_dir = self.versions_path / version_name
        version_dir.mkdir(exist_ok=True)
        
        # 保存当前数据集快照（只保存元数据，不复制实际文件）
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(version_data, f, indent=2)
        
        # 更新元数据
        self.metadata["versions"].append(version_data)
        self.metadata["current_version"] = version_name
        self._save_metadata()
        
        return version_data
    
    def get_version_info(self, version: str = None) -> Dict:
        """获取版本信息"""
        if version is None:
            version = self.metadata["current_version"]
        
        version_file = self.versions_path / version / "metadata.json"
        if version_file.exists():
            with open(version_file, 'r') as f:
                return json.load(f)
        return {}
    
    def list_versions(self) -> List[Dict]:
        """列出所有版本"""
        return self.metadata["versions"]
    
    def _calculate_dataset_stats(self) -> Dict:
        """计算数据集统计信息"""
        stats = {
            "train": {"positive": 0, "negative": 0},
            "val": {"positive": 0, "negative": 0},
            "test": {"positive": 0, "negative": 0}
        }
        
        for split in ["train", "val", "test"]:
            for label in ["positive", "negative"]:
                path = self.base_path / split / label
                if path.exists():
                    stats[split][label] = len(list(path.glob("*.jpg")))
        
        return stats


class DatasetIncrementalUpdater:
    """数据集增量更新器"""
    
    def __init__(self, base_path: str = "bioast_dataset"):
        self.base_path = Path(base_path)
        self.version_manager = DatasetVersionManager(base_path)
    
    def add_new_data(self, source_path: str, target_split: str, 
                    label: str, metadata: Dict = None) -> Dict:
        """添加新数据到数据集"""
        source_path = Path(source_path)
        target_path = self.base_path / target_split / label
        target_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            "added": 0,
            "duplicates": 0,
            "errors": 0,
            "files": []
        }
        
        # 支持单个文件或目录
        if source_path.is_file():
            files = [source_path]
        else:
            files = list(source_path.glob("*.jpg"))
        
        for file_path in files:
            try:
                # 检查是否重复
                if self._is_duplicate(file_path, target_path):
                    results["duplicates"] += 1
                    continue
                
                # 复制文件
                file_hash = self._calculate_file_hash(file_path)
                new_filename = f"{file_hash[:8]}_{file_path.name}"
                new_path = target_path / new_filename
                
                shutil.copy2(file_path, new_path)
                
                # 记录文件信息
                file_info = {
                    "original_name": file_path.name,
                    "new_name": new_filename,
                    "hash": file_hash,
                    "size": file_path.stat().st_size,
                    "metadata": metadata or {}
                }
                results["files"].append(file_info)
                results["added"] += 1
                
            except Exception as e:
                results["errors"] += 1
                print(f"Error processing {file_path}: {e}")
        
        # 更新统计信息
        self.version_manager.metadata["stats"] = self.version_manager._calculate_dataset_stats()
        self.version_manager._save_metadata()
        
        return results
    
    def _is_duplicate(self, file_path: Path, target_dir: Path) -> bool:
        """检查文件是否重复"""
        file_hash = self._calculate_file_hash(file_path)
        
        # 检查目标目录中是否有相同hash的文件
        for existing_file in target_dir.glob("*.jpg"):
            if existing_file.name.startswith(file_hash[:8]):
                return True
        
        return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件hash"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()
    
    def analyze_dataset_gaps(self) -> Dict:
        """分析数据集缺口"""
        current_stats = self.version_manager._calculate_dataset_stats()
        
        analysis = {
            "class_imbalance": {},
            "total_samples": 0,
            "recommendations": []
        }
        
        # 计算各类别数量
        for split in current_stats:
            if split != "total_images":
                total = current_stats[split]["positive"] + current_stats[split]["negative"]
                analysis["total_samples"] += total
                
                if total > 0:
                    pos_ratio = current_stats[split]["positive"] / total
                    analysis["class_imbalance"][split] = {
                        "positive_ratio": pos_ratio,
                        "negative_ratio": 1 - pos_ratio,
                        "is_balanced": 0.4 <= pos_ratio <= 0.6
                    }
        
        # 生成建议
        for split, imbalance in analysis["class_imbalance"].items():
            if not imbalance["is_balanced"]:
                if imbalance["positive_ratio"] < 0.4:
                    analysis["recommendations"].append(
                        f"建议在{split}集中添加更多正样本"
                    )
                else:
                    analysis["recommendations"].append(
                        f"建议在{split}集中添加更多负样本"
                    )
        
        return analysis


class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, base_path: str = "bioast_dataset"):
        self.base_path = Path(base_path)
    
    def generate_quality_report(self) -> Dict:
        """生成数据集质量报告"""
        report = {
            "summary": {},
            "quality_issues": [],
            "statistics": {},
            "recommendations": []
        }
        
        # 基础统计
        stats = self._calculate_basic_stats()
        report["summary"] = stats
        
        # 图像质量检查
        quality_issues = self._check_image_quality()
        report["quality_issues"] = quality_issues
        
        # 详细统计
        detailed_stats = self._calculate_detailed_stats()
        report["statistics"] = detailed_stats
        
        # 生成建议
        report["recommendations"] = self._generate_recommendations(stats, quality_issues)
        
        return report
    
    def _calculate_basic_stats(self) -> Dict:
        """计算基础统计信息"""
        stats = {
            "total_images": 0,
            "total_size_mb": 0,
            "formats": {},
            "splits": {}
        }
        
        for split in ["train", "val", "test"]:
            split_path = self.base_path / split
            if split_path.exists():
                split_stats = {"positive": 0, "negative": 0, "total": 0}
                for label in ["positive", "negative"]:
                    label_path = split_path / label
                    if label_path.exists():
                        count = len(list(label_path.glob("*.jpg")))
                        split_stats[label] = count
                        split_stats["total"] += count
                        
                        # 计算文件大小
                        for file in label_path.glob("*.jpg"):
                            stats["total_size_mb"] += file.stat().st_size / (1024 * 1024)
                
                stats["splits"][split] = split_stats
                stats["total_images"] += split_stats["total"]
        
        return stats
    
    def _check_image_quality(self) -> List[Dict]:
        """检查图像质量"""
        issues = []
        
        for split in ["train", "val", "test"]:
            split_path = self.base_path / split
            if split_path.exists():
                for label in ["positive", "negative"]:
                    label_path = split_path / label
                    if label_path.exists():
                        for img_file in label_path.glob("*.jpg"):
                            try:
                                with Image.open(img_file) as img:
                                    # 检查尺寸
                                    if img.size != (70, 70):
                                        issues.append({
                                            "type": "size_mismatch",
                                            "file": str(img_file),
                                            "actual_size": img.size,
                                            "expected_size": (70, 70)
                                        })
                                    
                                    # 检查是否损坏
                                    img.verify()
                                    
                            except Exception as e:
                                issues.append({
                                    "type": "corrupted",
                                    "file": str(img_file),
                                    "error": str(e)
                                })
        
        return issues
    
    def _calculate_detailed_stats(self) -> Dict:
        """计算详细统计信息"""
        stats = {
            "brightness": [],
            "contrast": [],
            "file_sizes": []
        }
        
        # 采样分析（避免处理所有图像）
        sample_files = []
        for split in ["train", "val", "test"]:
            split_path = self.base_path / split
            if split_path.exists():
                for label in ["positive", "negative"]:
                    label_path = split_path / label
                    if label_path.exists():
                        files = list(label_path.glob("*.jpg"))
                        sample_files.extend(files[:min(50, len(files))])
        
        # 分析采样图像
        for img_file in sample_files:
            try:
                with Image.open(img_file) as img:
                    # 转换为灰度计算亮度和对比度
                    gray = np.array(img.convert('L'))
                    brightness = np.mean(gray)
                    contrast = np.std(gray)
                    
                    stats["brightness"].append(brightness)
                    stats["contrast"].append(contrast)
                    stats["file_sizes"].append(img_file.stat().st_size)
                    
            except Exception:
                continue
        
        # 计算统计值
        for key in stats:
            if stats[key]:
                stats[f"{key}_mean"] = np.mean(stats[key])
                stats[f"{key}_std"] = np.std(stats[key])
                stats[f"{key}_min"] = np.min(stats[key])
                stats[f"{key}_max"] = np.max(stats[key])
        
        return stats
    
    def _generate_recommendations(self, stats: Dict, issues: List) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于统计的建议
        total_images = stats["total_images"]
        if total_images < 1000:
            recommendations.append("数据集规模较小，建议收集更多训练数据")
        
        # 基于质量问题的建议
        size_issues = [i for i in issues if i["type"] == "size_mismatch"]
        if size_issues:
            recommendations.append(f"发现{len(size_issues)}张图像尺寸不是70x70，建议统一尺寸")
        
        corrupted_issues = [i for i in issues if i["type"] == "corrupted"]
        if corrupted_issues:
            recommendations.append(f"发现{len(corrupted_issues)}张损坏图像，需要修复或删除")
        
        return recommendations


# 使用示例
if __name__ == "__main__":
    # 创建数据集管理器
    manager = DatasetVersionManager()
    
    # 创建新版本
    version_info = manager.create_version("v1.1", "添加了新的训练数据")
    print(f"创建版本: {version_info}")
    
    # 添加新数据
    updater = DatasetIncrementalUpdater()
    # result = updater.add_new_data("path/to/new/data", "train", "positive")
    # print(f"添加结果: {result}")
    
    # 分析数据集
    analyzer = DatasetAnalyzer()
    report = analyzer.generate_quality_report()
    print(f"质量报告: {json.dumps(report, indent=2, ensure_ascii=False)}")