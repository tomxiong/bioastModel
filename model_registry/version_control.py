"""模型版本控制系统

管理模型版本历史、变更记录和版本比较。
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import semver


class VersionControl:
    """模型版本控制器"""
    
    def __init__(self, versions_file: str = "data/model_versions.json"):
        self.versions_file = Path(versions_file)
        self.versions_file.parent.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, Dict[str, Any]] = {}
        self._load_versions()
    
    def _load_versions(self):
        """加载版本信息"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r', encoding='utf-8') as f:
                    self.versions = json.load(f)
            except Exception as e:
                print(f"加载版本信息失败: {e}")
                self.versions = {}
    
    def _save_versions(self):
        """保存版本信息"""
        try:
            with open(self.versions_file, 'w', encoding='utf-8') as f:
                json.dump(self.versions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存版本信息失败: {e}")
    
    def get_next_version(self, model_name: str, version_type: str = "patch") -> str:
        """获取下一个版本号
        
        Args:
            model_name: 模型名称
            version_type: 版本类型 (major, minor, patch)
            
        Returns:
            下一个版本号
        """
        if model_name not in self.versions:
            return "1.0.0"
        
        versions = list(self.versions[model_name].keys())
        if not versions:
            return "1.0.0"
        
        # 获取最新版本
        latest_version = max(versions, key=lambda v: semver.VersionInfo.parse(v))
        
        # 生成下一个版本
        try:
            version_info = semver.VersionInfo.parse(latest_version)
            if version_type == "major":
                return str(version_info.bump_major())
            elif version_type == "minor":
                return str(version_info.bump_minor())
            else:  # patch
                return str(version_info.bump_patch())
        except Exception:
            # 如果解析失败，使用简单的数字递增
            try:
                parts = latest_version.split('.')
                if len(parts) == 3:
                    major, minor, patch = map(int, parts)
                    if version_type == "major":
                        return f"{major + 1}.0.0"
                    elif version_type == "minor":
                        return f"{major}.{minor + 1}.0"
                    else:
                        return f"{major}.{minor}.{patch + 1}"
            except Exception:
                pass
            
            # 最后的备选方案
            return f"{len(versions) + 1}.0.0"
    
    def add_version(self, 
                   model_name: str, 
                   version: str, 
                   metadata: Dict[str, Any],
                   changelog: str = "",
                   parent_version: Optional[str] = None):
        """添加新版本
        
        Args:
            model_name: 模型名称
            version: 版本号
            metadata: 模型元数据
            changelog: 变更日志
            parent_version: 父版本号
        """
        if model_name not in self.versions:
            self.versions[model_name] = {}
        
        version_info = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata,
            "changelog": changelog,
            "parent_version": parent_version,
            "status": "active",
            "tags": [],
            "performance_delta": {},
            "size_delta": {}
        }
        
        # 如果有父版本，计算性能和大小变化
        if parent_version and parent_version in self.versions[model_name]:
            parent_metadata = self.versions[model_name][parent_version]["metadata"]
            version_info["performance_delta"] = self._calculate_performance_delta(
                parent_metadata.get("performance", {}),
                metadata.get("performance", {})
            )
            version_info["size_delta"] = self._calculate_size_delta(
                parent_metadata.get("architecture", {}),
                metadata.get("architecture", {})
            )
        
        self.versions[model_name][version] = version_info
        self._save_versions()
    
    def _calculate_performance_delta(self, 
                                   parent_perf: Dict[str, float], 
                                   current_perf: Dict[str, float]) -> Dict[str, float]:
        """计算性能变化"""
        delta = {}
        for metric in set(parent_perf.keys()) | set(current_perf.keys()):
            parent_val = parent_perf.get(metric, 0.0)
            current_val = current_perf.get(metric, 0.0)
            if parent_val > 0:
                delta[metric] = (current_val - parent_val) / parent_val
            else:
                delta[metric] = current_val
        return delta
    
    def _calculate_size_delta(self, 
                            parent_arch: Dict[str, Any], 
                            current_arch: Dict[str, Any]) -> Dict[str, float]:
        """计算大小变化"""
        delta = {}
        
        # 参数量变化
        parent_params = parent_arch.get("total_parameters", 0)
        current_params = current_arch.get("total_parameters", 0)
        if parent_params > 0:
            delta["parameters"] = (current_params - parent_params) / parent_params
        
        # 模型大小变化
        parent_size = parent_arch.get("model_size_mb", 0.0)
        current_size = current_arch.get("model_size_mb", 0.0)
        if parent_size > 0:
            delta["size_mb"] = (current_size - parent_size) / parent_size
        
        return delta
    
    def get_versions(self, model_name: str) -> List[str]:
        """获取模型的所有版本"""
        if model_name not in self.versions:
            return []
        return list(self.versions[model_name].keys())
    
    def get_version_info(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """获取特定版本的信息"""
        if model_name in self.versions and version in self.versions[model_name]:
            return self.versions[model_name][version]
        return None
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """获取最新版本"""
        versions = self.get_versions(model_name)
        if not versions:
            return None
        
        try:
            return max(versions, key=lambda v: semver.VersionInfo.parse(v))
        except Exception:
            # 如果无法解析语义化版本，按字符串排序
            return max(versions)
    
    def compare_versions(self, 
                        model_name: str, 
                        version1: str, 
                        version2: str) -> Dict[str, Any]:
        """比较两个版本"""
        v1_info = self.get_version_info(model_name, version1)
        v2_info = self.get_version_info(model_name, version2)
        
        if not v1_info or not v2_info:
            return {"error": "版本不存在"}
        
        comparison = {
            "model_name": model_name,
            "version1": version1,
            "version2": version2,
            "created_at_diff": v2_info["created_at"] > v1_info["created_at"],
            "performance_comparison": {},
            "architecture_comparison": {},
            "changelog_diff": {
                "v1_changelog": v1_info.get("changelog", ""),
                "v2_changelog": v2_info.get("changelog", "")
            }
        }
        
        # 性能比较
        v1_perf = v1_info["metadata"].get("performance", {})
        v2_perf = v2_info["metadata"].get("performance", {})
        
        for metric in set(v1_perf.keys()) | set(v2_perf.keys()):
            v1_val = v1_perf.get(metric, 0.0)
            v2_val = v2_perf.get(metric, 0.0)
            comparison["performance_comparison"][metric] = {
                "v1": v1_val,
                "v2": v2_val,
                "improvement": v2_val - v1_val,
                "improvement_percent": ((v2_val - v1_val) / v1_val * 100) if v1_val > 0 else 0
            }
        
        # 架构比较
        v1_arch = v1_info["metadata"].get("architecture", {})
        v2_arch = v2_info["metadata"].get("architecture", {})
        
        comparison["architecture_comparison"] = {
            "parameters_change": {
                "v1": v1_arch.get("total_parameters", 0),
                "v2": v2_arch.get("total_parameters", 0),
                "change": v2_arch.get("total_parameters", 0) - v1_arch.get("total_parameters", 0)
            },
            "size_change": {
                "v1": v1_arch.get("model_size_mb", 0.0),
                "v2": v2_arch.get("model_size_mb", 0.0),
                "change": v2_arch.get("model_size_mb", 0.0) - v1_arch.get("model_size_mb", 0.0)
            }
        }
        
        return comparison
    
    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """获取版本历史"""
        if model_name not in self.versions:
            return []
        
        history = []
        for version, info in self.versions[model_name].items():
            history.append({
                "version": version,
                "created_at": info["created_at"],
                "changelog": info.get("changelog", ""),
                "status": info.get("status", "active"),
                "performance": info["metadata"].get("performance", {}),
                "performance_delta": info.get("performance_delta", {}),
                "size_delta": info.get("size_delta", {})
            })
        
        # 按版本号排序
        try:
            history.sort(key=lambda x: semver.VersionInfo.parse(x["version"]), reverse=True)
        except Exception:
            history.sort(key=lambda x: x["version"], reverse=True)
        
        return history
    
    def tag_version(self, model_name: str, version: str, tag: str):
        """为版本添加标签"""
        if (model_name in self.versions and 
            version in self.versions[model_name]):
            if "tags" not in self.versions[model_name][version]:
                self.versions[model_name][version]["tags"] = []
            
            if tag not in self.versions[model_name][version]["tags"]:
                self.versions[model_name][version]["tags"].append(tag)
                self._save_versions()
    
    def deprecate_version(self, model_name: str, version: str, reason: str = ""):
        """废弃版本"""
        if (model_name in self.versions and 
            version in self.versions[model_name]):
            self.versions[model_name][version]["status"] = "deprecated"
            self.versions[model_name][version]["deprecation_reason"] = reason
            self.versions[model_name][version]["deprecated_at"] = datetime.now().isoformat()
            self._save_versions()
    
    def get_version_tree(self, model_name: str) -> Dict[str, Any]:
        """获取版本树结构"""
        if model_name not in self.versions:
            return {}
        
        tree = {"name": model_name, "children": []}
        versions = self.versions[model_name]
        
        # 构建父子关系
        version_nodes = {}
        root_versions = []
        
        for version, info in versions.items():
            node = {
                "version": version,
                "created_at": info["created_at"],
                "status": info.get("status", "active"),
                "children": []
            }
            version_nodes[version] = node
            
            parent = info.get("parent_version")
            if not parent or parent not in versions:
                root_versions.append(node)
        
        # 建立父子关系
        for version, info in versions.items():
            parent = info.get("parent_version")
            if parent and parent in version_nodes:
                version_nodes[parent]["children"].append(version_nodes[version])
        
        tree["children"] = root_versions
        return tree
    
    def export_version_history(self, model_name: str, output_file: str):
        """导出版本历史"""
        history = self.get_version_history(model_name)
        
        export_data = {
            "model_name": model_name,
            "exported_at": datetime.now().isoformat(),
            "total_versions": len(history),
            "version_history": history,
            "version_tree": self.get_version_tree(model_name)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取版本控制统计信息"""
        total_models = len(self.versions)
        total_versions = sum(len(versions) for versions in self.versions.values())
        
        # 状态统计
        status_stats = {"active": 0, "deprecated": 0, "experimental": 0}
        for model_versions in self.versions.values():
            for version_info in model_versions.values():
                status = version_info.get("status", "active")
                status_stats[status] = status_stats.get(status, 0) + 1
        
        # 最活跃的模型
        most_active_model = max(
            self.versions.items(), 
            key=lambda x: len(x[1]),
            default=("None", {})
        )
        
        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "average_versions_per_model": total_versions / total_models if total_models > 0 else 0,
            "status_distribution": status_stats,
            "most_active_model": {
                "name": most_active_model[0],
                "version_count": len(most_active_model[1])
            },
            "versions_file": str(self.versions_file)
        }