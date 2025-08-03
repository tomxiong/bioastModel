#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集管理工具
支持误判调整、增量数据添加、去重、重新平衡等功能
支持dataset_builder.py的源数据结构（cfg文件 + 孔位图片）
"""

import os
import shutil
import hashlib
import yaml
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import random
from PIL import Image
import numpy as np
from datetime import datetime

class DatasetManager:
    def __init__(self, dataset_path="bioast_dataset"):
        self.dataset_path = Path(dataset_path)
        self.splits = ['train', 'test', 'val']
        self.classes = ['positive', 'negative']
        
        # 图片哈希缓存
        self.image_hashes = {}
        self.duplicate_groups = []
        
        # 统计信息
        self.stats = self._get_current_stats()
    
    def _get_current_stats(self):
        """获取当前数据集统计信息"""
        stats = defaultdict(dict)
        total_count = 0
        
        for class_name in self.classes:
            class_total = 0
            for split in self.splits:
                split_path = self.dataset_path / class_name / split
                if split_path.exists():
                    count = len(list(split_path.glob("*.png")))
                    stats[class_name][split] = count
                    class_total += count
                    total_count += count
                else:
                    stats[class_name][split] = 0
            stats[class_name]['total'] = class_total
        
        stats['total'] = total_count
        return dict(stats)
    
    def print_stats(self):
        """打印当前统计信息"""
        print("📊 当前数据集统计:")
        print("=" * 50)
        
        for class_name in self.classes:
            print(f"\n{class_name.upper()}:")
            for split in self.splits:
                count = self.stats[class_name][split]
                print(f"  {split}: {count}")
            print(f"  总计: {self.stats[class_name]['total']}")
        
        total = self.stats['total']
        pos_ratio = self.stats['positive']['total'] / total * 100 if total > 0 else 0
        print(f"\n总计: {total}")
        print(f"平衡比例: {pos_ratio:.1f}% positive, {100-pos_ratio:.1f}% negative")
    
    def calculate_image_hash(self, image_path):
        """计算图片的感知哈希"""
        try:
            with Image.open(image_path) as img:
                # 转换为灰度并调整大小
                img = img.convert('L').resize((8, 8))
                # 计算像素平均值
                pixels = np.array(img)
                avg = pixels.mean()
                # 生成哈希
                hash_bits = pixels > avg
                hash_str = ''.join(['1' if b else '0' for b in hash_bits.flatten()])
                return hash_str
        except Exception:
            return None
    
    def find_duplicates(self):
        """查找重复图片"""
        print("🔍 扫描重复图片...")
        self.image_hashes.clear()
        hash_to_files = defaultdict(list)
        
        # 扫描所有图片
        for class_name in self.classes:
            for split in self.splits:
                split_path = self.dataset_path / class_name / split
                if split_path.exists():
                    for img_file in split_path.glob("*.png"):
                        img_hash = self.calculate_image_hash(img_file)
                        if img_hash:
                            self.image_hashes[str(img_file)] = img_hash
                            hash_to_files[img_hash].append(img_file)
        
        # 找出重复组
        self.duplicate_groups = [files for files in hash_to_files.values() if len(files) > 1]
        
        if self.duplicate_groups:
            print(f"❗ 发现 {len(self.duplicate_groups)} 组重复图片:")
            for i, group in enumerate(self.duplicate_groups, 1):
                print(f"  组 {i}: {len(group)} 张图片")
                for file_path in group:
                    print(f"    {file_path}")
        else:
            print("✅ 未发现重复图片")
        
        return self.duplicate_groups
    
    def remove_duplicates(self, keep_strategy='first'):
        """移除重复图片"""
        if not self.duplicate_groups:
            self.find_duplicates()
        
        if not self.duplicate_groups:
            print("✅ 没有重复图片需要处理")
            return
        
        removed_count = 0
        for group in self.duplicate_groups:
            if keep_strategy == 'first':
                keep_file = group[0]
                remove_files = group[1:]
            elif keep_strategy == 'train_priority':
                # 优先保留训练集中的
                train_files = [f for f in group if '/train/' in str(f)]
                if train_files:
                    keep_file = train_files[0]
                    remove_files = [f for f in group if f != keep_file]
                else:
                    keep_file = group[0]
                    remove_files = group[1:]
            else:
                keep_file = group[0]
                remove_files = group[1:]
            
            print(f"保留: {keep_file}")
            for remove_file in remove_files:
                print(f"删除: {remove_file}")
                os.remove(remove_file)
                removed_count += 1
        
        print(f"✅ 删除了 {removed_count} 张重复图片")
        self.stats = self._get_current_stats()
    
    def move_misclassified_samples(self, misclassified_list):
        """移动误判样本
        
        Args:
            misclassified_list: 格式 [(src_path, target_class), ...]
        """
        print("🔄 移动误判样本...")
        moved_count = 0
        
        for src_path, target_class in misclassified_list:
            src_path = Path(src_path)
            if not src_path.exists():
                print(f"❌ 文件不存在: {src_path}")
                continue
            
            # 确定目标路径 - 移动到train目录以增加训练数据
            target_dir = self.dataset_path / target_class / 'train'
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / src_path.name
            
            # 避免覆盖
            counter = 1
            while target_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # 移动文件
            shutil.move(str(src_path), str(target_path))
            print(f"✅ {src_path} → {target_path}")
            moved_count += 1
        
        print(f"✅ 移动了 {moved_count} 个误判样本")
        self.stats = self._get_current_stats()
    
    def add_incremental_data(self, source_dirs, auto_split=True, split_ratio=(0.7, 0.2, 0.1)):
        """添加增量数据
        
        Args:
            source_dirs: 源数据目录列表 [(path, class_name), ...]
            auto_split: 是否自动分割数据到train/test/val
            split_ratio: 分割比例 (train, test, val)
        """
        print("📥 添加增量数据...")
        
        new_files = []
        duplicate_count = 0
        
        for source_dir, class_name in source_dirs:
            source_path = Path(source_dir)
            if not source_path.exists():
                print(f"❌ 源目录不存在: {source_path}")
                continue
            
            print(f"处理 {class_name} 类别，源目录: {source_path}")
            
            for img_file in source_path.glob("*.png"):
                # 计算新图片哈希
                new_hash = self.calculate_image_hash(img_file)
                if not new_hash:
                    continue
                
                # 检查是否已存在
                is_duplicate = False
                for existing_path, existing_hash in self.image_hashes.items():
                    if new_hash == existing_hash:
                        print(f"⚠️  跳过重复图片: {img_file.name} (与 {existing_path} 重复)")
                        duplicate_count += 1
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    new_files.append((img_file, class_name, new_hash))
        
        if not new_files:
            print("❌ 没有新的有效图片可以添加")
            return
        
        print(f"📋 准备添加 {len(new_files)} 张新图片")
        print(f"📋 跳过了 {duplicate_count} 张重复图片")
        
        if auto_split:
            # 自动分割
            random.shuffle(new_files)
            total = len(new_files)
            train_end = int(total * split_ratio[0])
            test_end = train_end + int(total * split_ratio[1])
            
            splits_files = {
                'train': new_files[:train_end],
                'test': new_files[train_end:test_end],
                'val': new_files[test_end:]
            }
        else:
            # 全部加到train（用于误判调整后的增强）
            splits_files = {'train': new_files, 'test': [], 'val': []}
        
        # 复制文件
        added_count = 0
        for split, files in splits_files.items():
            for img_file, class_name, img_hash in files:
                target_dir = self.dataset_path / class_name / split
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # 避免文件名冲突
                target_path = target_dir / img_file.name
                counter = 1
                while target_path.exists():
                    stem = img_file.stem
                    suffix = img_file.suffix
                    target_path = target_dir / f"{stem}_inc{counter}{suffix}"
                    counter += 1
                
                shutil.copy2(img_file, target_path)
                self.image_hashes[str(target_path)] = img_hash
                added_count += 1
                print(f"✅ 添加到 {class_name}/{split}: {target_path.name}")
        
        print(f"✅ 成功添加 {added_count} 张新图片")
        self.stats = self._get_current_stats()
    
    def rebalance_dataset(self, target_ratio=0.5, method='oversample'):
        """重新平衡数据集
        
        Args:
            target_ratio: 目标正样本比例
            method: 'oversample', 'undersample', 'mixed'
        """
        print("⚖️ 重新平衡数据集...")
        
        pos_count = self.stats['positive']['total']
        neg_count = self.stats['negative']['total']
        total = pos_count + neg_count
        current_ratio = pos_count / total if total > 0 else 0
        
        print(f"当前比例: {current_ratio:.3f} ({pos_count}pos / {neg_count}neg)")
        print(f"目标比例: {target_ratio:.3f}")
        
        if abs(current_ratio - target_ratio) < 0.01:
            print("✅ 数据集已经平衡")
            return
        
        if method == 'oversample':
            self._oversample_minority_class(target_ratio)
        elif method == 'undersample':
            self._undersample_majority_class(target_ratio)
        else:
            print("❌ 不支持的平衡方法")
    
    def _oversample_minority_class(self, target_ratio):
        """通过过采样平衡数据集"""
        pos_count = self.stats['positive']['total']
        neg_count = self.stats['negative']['total']
        total = pos_count + neg_count
        
        if pos_count / total < target_ratio:
            # 需要增加正样本
            minority_class = 'positive'
            majority_count = neg_count
        else:
            # 需要增加负样本
            minority_class = 'negative'
            majority_count = pos_count
        
        # 计算需要的样本数
        target_minority_count = int(majority_count * target_ratio / (1 - target_ratio))
        current_minority_count = self.stats[minority_class]['total']
        need_count = target_minority_count - current_minority_count
        
        if need_count <= 0:
            print("✅ 无需过采样")
            return
        
        print(f"需要为 {minority_class} 类别增加 {need_count} 个样本")
        
        # 收集可用于复制的图片（主要从train集）
        source_files = []
        for split in ['train', 'val', 'test']:  # 优先train
            split_path = self.dataset_path / minority_class / split
            if split_path.exists():
                source_files.extend(list(split_path.glob("*.png")))
        
        if not source_files:
            print("❌ 没有可用于过采样的源文件")
            return
        
        # 复制文件到train目录
        target_dir = self.dataset_path / minority_class / 'train'
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(need_count):
            source_file = random.choice(source_files)
            
            # 生成新文件名
            counter = 1
            target_path = target_dir / f"{source_file.stem}_aug{counter}.png"
            while target_path.exists():
                counter += 1
                target_path = target_dir / f"{source_file.stem}_aug{counter}.png"
            
            shutil.copy2(source_file, target_path)
            if i < 5:  # 只显示前5个
                print(f"✅ 复制: {source_file.name} → {target_path.name}")
        
        print(f"✅ 过采样完成，增加了 {need_count} 个样本")
        self.stats = self._get_current_stats()
    
    def save_dataset_info(self):
        """保存数据集信息(JSON格式)"""
        report_path = self.dataset_path / "dataset_stats.json"
        
        # 准备报告数据
        report_data = {
            "metadata": {
                "source_directory": "增量更新",
                "output_directory": str(self.dataset_path),
                "sample_count": "N/A",
                "generation_time": datetime.now().isoformat(),
                "tool_version": "dataset_manager.py"
            },
            "summary": {
                "positive_total": self.stats.get('positive', {}).get('total', 0),
                "negative_total": self.stats.get('negative', {}).get('total', 0),
                "total_samples": self.stats.get('total', 0)
            },
            "dataset_splits": {}
        }
        
        # 添加各数据集统计信息
        for subset in self.splits:
            pos_count = self.stats.get('positive', {}).get(subset, 0)
            neg_count = self.stats.get('negative', {}).get(subset, 0)
            total_count = pos_count + neg_count
            
            if total_count > 0:
                pos_ratio = pos_count / total_count
                neg_ratio = neg_count / total_count
                
                report_data["dataset_splits"][subset] = {
                    "positive": {
                        "count": pos_count,
                        "ratio": round(pos_ratio, 4)
                    },
                    "negative": {
                        "count": neg_count,
                        "ratio": round(neg_ratio, 4)
                    },
                    "total": total_count
                }
        
        # 写入JSON文件
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 数据集统计信息已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='数据集管理工具')
    parser.add_argument('--dataset', type=str, default='bioast_dataset', help='数据集路径')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 统计命令
    subparsers.add_parser('stats', help='显示数据集统计信息')
    
    # 查找重复
    subparsers.add_parser('find-duplicates', help='查找重复图片')
    
    # 删除重复
    dup_parser = subparsers.add_parser('remove-duplicates', help='删除重复图片')
    dup_parser.add_argument('--strategy', choices=['first', 'train_priority'], default='train_priority', 
                           help='保留策略')
    
    # 移动误判样本
    move_parser = subparsers.add_parser('move-misclassified', help='移动误判样本')
    move_parser.add_argument('--file', type=str, required=True, help='误判文件路径')
    move_parser.add_argument('--target-class', choices=['positive', 'negative'], required=True, 
                            help='目标类别')
    
    # 添加增量数据
    add_parser = subparsers.add_parser('add-data', help='添加增量数据')
    add_parser.add_argument('--source', type=str, required=True, help='源数据目录')
    add_parser.add_argument('--class', type=str, choices=['positive', 'negative'], 
                           required=True, dest='class_name', help='数据类别')
    add_parser.add_argument('--no-split', action='store_true', help='不自动分割，全部加到train')
    
    # 重新平衡
    balance_parser = subparsers.add_parser('rebalance', help='重新平衡数据集')
    balance_parser.add_argument('--ratio', type=float, default=0.5, help='目标正样本比例')
    balance_parser.add_argument('--method', choices=['oversample', 'undersample'], 
                               default='oversample', help='平衡方法')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = DatasetManager(args.dataset)
    
    if args.command == 'stats':
        manager.print_stats()
        
    elif args.command == 'find-duplicates':
        manager.find_duplicates()
        
    elif args.command == 'remove-duplicates':
        manager.remove_duplicates(args.strategy)
        
    elif args.command == 'move-misclassified':
        misclassified_list = [(args.file, args.target_class)]
        manager.move_misclassified_samples(misclassified_list)
        manager.save_dataset_info()
        
    elif args.command == 'add-data':
        source_dirs = [(args.source, args.class_name)]
        manager.add_incremental_data(source_dirs, auto_split=not args.no_split)
        manager.save_dataset_info()
        
    elif args.command == 'rebalance':
        manager.rebalance_dataset(args.ratio, args.method)
        manager.save_dataset_info()

if __name__ == "__main__":
    main()