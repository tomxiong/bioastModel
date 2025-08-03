#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版数据集管理工具
支持dataset_builder.py的源数据结构（cfg文件 + 孔位图片）
支持从cfg源数据按比例同步到train/test/val，保持比例平衡和去重
"""

import os
import shutil
import yaml
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import random
from PIL import Image
import numpy as np
from datetime import datetime

class EnhancedDatasetManager:
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
    
    def get_current_split_ratios(self):
        """获取当前数据集的分割比例"""
        total = self.stats['total']
        if total == 0:
            # 默认比例
            return {'train': 0.7, 'test': 0.2, 'val': 0.1}
        
        ratios = {}
        for split in self.splits:
            split_total = sum(self.stats[class_name][split] for class_name in self.classes)
            ratios[split] = split_total / total if total > 0 else 0
        
        return ratios
    
    def get_current_class_ratios(self):
        """获取当前数据集的类别比例"""
        total = self.stats['total']
        if total == 0:
            return {'positive': 0.5, 'negative': 0.5}
        
        pos_total = self.stats['positive']['total']
        neg_total = self.stats['negative']['total']
        
        return {
            'positive': pos_total / total if total > 0 else 0.5,
            'negative': neg_total / total if total > 0 else 0.5
        }
    
    def print_stats(self):
        """打印当前统计信息"""
        print("📊 当前数据集统计:")
        print("=" * 50)
        
        # 打印各类别和分割的统计
        for class_name in self.classes:
            print(f"\n{class_name.upper()}:")
            for split in self.splits:
                count = self.stats[class_name][split]
                print(f"  {split}: {count}")
            print(f"  总计: {self.stats[class_name]['total']}")
        
        # 打印总体统计
        total = self.stats['total']
        if total > 0:
            pos_ratio = self.stats['positive']['total'] / total * 100
            print(f"\n总计: {total}")
            print(f"类别比例: {pos_ratio:.1f}% positive, {100-pos_ratio:.1f}% negative")
            
            # 打印分割比例
            print("分割比例:")
            for split in self.splits:
                split_total = sum(self.stats[class_name][split] for class_name in self.classes)
                split_ratio = split_total / total * 100 if total > 0 else 0
                print(f"  {split}: {split_total} ({split_ratio:.1f}%)")
        else:
            print("\n数据集为空")
    
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
    
    def parse_cfg_file(self, cfg_path):
        """解析cfg文件获取阴阳性标签"""
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                
            # 格式：文件名,标签字符串
            parts = line.split(',')
            if len(parts) != 2:
                print(f"警告: cfg文件格式异常: {cfg_path}")
                return None
                
            filename = parts[0]
            labels_str = parts[1]
            
            # 解析标签字符串，+表示阳性，-表示阴性
            # cfg索引i直接对应hole_{i}.png
            labels = []
            for i, char in enumerate(labels_str):
                hole_num = i  # cfg索引i直接对应hole_{i}.png
                # 只处理存在的图片文件（hole_24到hole_119）
                if 24 <= hole_num <= 119:
                    if char == '+':
                        labels.append((hole_num, 'positive'))
                    elif char == '-':
                        labels.append((hole_num, 'negative'))
                    # 忽略其他字符
                        
            return {
                'filename': filename,
                'labels': labels,
                'sample_name': filename.replace('.bmp', '')
            }
            
        except Exception as e:
            print(f"错误: 无法解析cfg文件 {cfg_path}: {e}")
            return None
    
    def load_existing_hashes(self):
        """加载现有数据集的图片哈希"""
        print("🔍 扫描现有数据集图片...")
        self.image_hashes.clear()
        
        total_images = 0
        for class_name in self.classes:
            for split in self.splits:
                split_path = self.dataset_path / class_name / split
                if split_path.exists():
                    for img_file in split_path.glob("*.png"):
                        img_hash = self.calculate_image_hash(img_file)
                        if img_hash:
                            self.image_hashes[str(img_file)] = img_hash
                            total_images += 1
        
        print(f"✅ 加载了 {total_images} 张现有图片的哈希")
        return total_images
    
    def sync_from_cfg_source(self, source_dir, preserve_class_ratio=True, preserve_split_ratio=True, split_ratio=None):
        """从cfg源数据按比例同步图片到各个分割
        
        Args:
            source_dir: 源数据目录（包含cfg文件和同名子目录）
            preserve_class_ratio: 是否保持现有的阴阳性比例
            preserve_split_ratio: 是否保持现有的train/test/val分割比例
            split_ratio: 自定义分割比例 {'train': 0.7, 'test': 0.2, 'val': 0.1}
        """
        print("🔄 从cfg源数据按比例同步图片...")
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"❌ 源目录不存在: {source_path}")
            return
        
        # 加载现有图片哈希
        self.load_existing_hashes()
        
        # 获取当前比例
        current_stats = self._get_current_stats()
        current_class_ratios = self.get_current_class_ratios()
        
        if preserve_split_ratio:
            current_split_ratios = self.get_current_split_ratios()
        else:
            current_split_ratios = split_ratio or {'train': 0.7, 'test': 0.2, 'val': 0.1}
        
        total_current = current_stats['total']
        print(f"📊 当前数据集: {total_current} 张图片")
        print(f"   类别比例: positive={current_class_ratios['positive']:.3f}, negative={current_class_ratios['negative']:.3f}")
        print(f"   分割比例: train={current_split_ratios['train']:.3f}, test={current_split_ratios['test']:.3f}, val={current_split_ratios['val']:.3f}")
        
        # 扫描cfg文件
        cfg_files = list(source_path.glob("*.cfg"))
        print(f"🔍 发现 {len(cfg_files)} 个cfg文件")
        
        samples_data = []
        for cfg_file in cfg_files:
            cfg_data = self.parse_cfg_file(cfg_file)
            if cfg_data:
                # 检查对应的图片目录是否存在
                sample_dir = source_path / cfg_data['sample_name']
                if sample_dir.exists():
                    cfg_data['sample_dir'] = sample_dir
                    samples_data.append(cfg_data)
                    print(f"✅ 加载样本: {cfg_data['sample_name']}, 孔位数: {len(cfg_data['labels'])}")
                else:
                    print(f"⚠️  未找到对应图片目录: {sample_dir}")
        
        if not samples_data:
            print("❌ 没有有效的样本数据")
            return
        
        # 收集所有图片及其标签
        new_positive_images = []
        new_negative_images = []
        duplicate_count = 0
        
        for sample in samples_data:
            sample_dir = sample['sample_dir']
            
            for hole_num, label in sample['labels']:
                hole_file = sample_dir / f"hole_{hole_num}.png"
                
                if hole_file.exists():
                    # 计算图片哈希检查重复
                    img_hash = self.calculate_image_hash(hole_file)
                    if not img_hash:
                        continue
                
                    # 检查是否已存在
                    is_duplicate = False
                    for existing_path, existing_hash in self.image_hashes.items():
                        if img_hash == existing_hash:
                            duplicate_count += 1
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        image_info = {
                            'path': hole_file,
                            'sample': sample['sample_name'],
                            'hole': hole_num,
                            'label': label,
                            'hash': img_hash
                        }
                        
                        if label == 'positive':
                            new_positive_images.append(image_info)
                        else:
                            new_negative_images.append(image_info)
        
        total_new = len(new_positive_images) + len(new_negative_images)
        print(f"📋 发现 {total_new} 张新图片")
        print(f"   阳性: {len(new_positive_images)} 张")
        print(f"   阴性: {len(new_negative_images)} 张")
        print(f"📋 跳过 {duplicate_count} 张重复图片")
        
        if total_new == 0:
            print("❌ 没有新图片可以添加")
            return
        
        # 按类别比例选择图片
        if preserve_class_ratio and total_current > 0:
            target_pos_count = int(total_new * current_class_ratios['positive'])
            target_neg_count = total_new - target_pos_count
            
            print(f"📊 按类别比例({current_class_ratios['positive']:.3f})选择:")
            print(f"   目标正样本: {target_pos_count}")
            print(f"   目标负样本: {target_neg_count}")
            
            # 随机选择
            random.shuffle(new_positive_images)
            random.shuffle(new_negative_images)
            
            selected_positive = new_positive_images[:target_pos_count] if len(new_positive_images) >= target_pos_count else new_positive_images
            selected_negative = new_negative_images[:target_neg_count] if len(new_negative_images) >= target_neg_count else new_negative_images
        else:
            # 使用所有新图片
            selected_positive = new_positive_images  
            selected_negative = new_negative_images
            print("📊 使用所有新图片")
        
        all_selected = selected_positive + selected_negative
        total_selected = len(all_selected)
        
        print(f"📊 最终选择: {len(selected_positive)} 正样本, {len(selected_negative)} 负样本 (总计{total_selected})")
        
        # 按分割比例分配图片
        random.shuffle(all_selected)
        
        train_end = int(total_selected * current_split_ratios['train'])
        test_end = train_end + int(total_selected * current_split_ratios['test'])
        
        split_assignments = {
            'train': all_selected[:train_end],
            'test': all_selected[train_end:test_end],
            'val': all_selected[test_end:]
        }
        
        print(f"📊 分割分配:")
        for split, images in split_assignments.items():
            pos_count = sum(1 for img in images if img['label'] == 'positive')
            neg_count = len(images) - pos_count
            print(f"   {split}: {len(images)} 张 (正样本:{pos_count}, 负样本:{neg_count})")
        
        # 复制图片到各个分割
        total_added = 0
        split_added = defaultdict(lambda: defaultdict(int))
        
        for split, images in split_assignments.items():
            for img_info in images:
                target_dir = self.dataset_path / img_info['label'] / split
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # 生成唯一文件名：样本名_孔位号.png
                new_filename = f"{img_info['sample']}_hole_{img_info['hole']}.png"
                target_path = target_dir / new_filename
                
                # 避免文件名冲突
                counter = 1
                while target_path.exists():
                    stem = f"{img_info['sample']}_hole_{img_info['hole']}"
                    target_path = target_dir / f"{stem}_sync{counter}.png"
                    counter += 1
                
                try:
                    shutil.copy2(img_info['path'], target_path)
                    self.image_hashes[str(target_path)] = img_info['hash']
                    total_added += 1
                    split_added[split][img_info['label']] += 1
                    
                    if total_added <= 10:  # 只显示前10个
                        print(f"✅ 添加: {img_info['sample']}_hole_{img_info['hole']} → {img_info['label']}/{split}")
                    elif total_added == 11:
                        print("   ... (更多文件)")
                        
                except Exception as e:
                    print(f"❌ 复制失败 {img_info['path']}: {e}")
        
        print(f"\n✅ 同步完成！总计添加 {total_added} 张图片")
        for split in self.splits:
            pos_added = split_added[split]['positive']
            neg_added = split_added[split]['negative']
            split_total = pos_added + neg_added
            if split_total > 0:
                print(f"   {split}: {split_total} 张 (正:{pos_added}, 负:{neg_added})")
        
        # 更新统计信息
        self.stats = self._get_current_stats()
        
        # 显示更新后的统计
        new_total = self.stats['total']
        new_class_ratios = self.get_current_class_ratios()
        new_split_ratios = self.get_current_split_ratios()
        
        print(f"\n📊 更新后数据集: {new_total} 张图片")
        print(f"   类别比例: positive={new_class_ratios['positive']:.3f}, negative={new_class_ratios['negative']:.3f}")
        print(f"   分割比例: train={new_split_ratios['train']:.3f}, test={new_split_ratios['test']:.3f}, val={new_split_ratios['val']:.3f}")
        
        return total_added
    
    def save_dataset_info(self):
        """保存数据集信息(JSON格式)"""
        report_path = self.dataset_path / "dataset_stats.json"
        
        # 准备报告数据
        report_data = {
            "metadata": {
                "source_directory": "cfg源数据按比例同步",
                "output_directory": str(self.dataset_path),
                "sample_count": "N/A",
                "generation_time": datetime.now().isoformat(),
                "tool_version": "enhanced_dataset_manager.py"
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
    parser = argparse.ArgumentParser(description='增强版数据集管理工具')
    parser.add_argument('--dataset', type=str, default='bioast_dataset', help='数据集路径')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 统计命令
    subparsers.add_parser('stats', help='显示数据集统计信息')
    
    # 从cfg源同步数据
    sync_parser = subparsers.add_parser('sync-from-cfg', help='从cfg源数据按比例同步图片到train/test/val')
    sync_parser.add_argument('--source', type=str, required=True, help='cfg源数据目录')
    sync_parser.add_argument('--preserve-class-ratio', action='store_true', default=True, 
                            help='保持现有的阴阳性比例')
    sync_parser.add_argument('--preserve-split-ratio', action='store_true', default=True,
                            help='保持现有的train/test/val分割比例')
    sync_parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    sync_parser.add_argument('--test-ratio', type=float, default=0.2, help='测试集比例')
    sync_parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = EnhancedDatasetManager(args.dataset)
    
    if args.command == 'stats':
        manager.print_stats()
        
    elif args.command == 'sync-from-cfg':
        # 构建自定义分割比例
        custom_split_ratio = None
        if not args.preserve_split_ratio:
            custom_split_ratio = {
                'train': args.train_ratio,
                'test': args.test_ratio,
                'val': args.val_ratio
            }
        
        added_count = manager.sync_from_cfg_source(
            source_dir=args.source,
            preserve_class_ratio=args.preserve_class_ratio,
            preserve_split_ratio=args.preserve_split_ratio,
            split_ratio=custom_split_ratio
        )
        
        if added_count > 0:
            manager.save_dataset_info()
            print(f"\n🎉 同步完成！按比例添加了 {added_count} 张新图片到train/test/val")
        else:
            print("\n⚠️  没有新图片被添加")

if __name__ == "__main__":
    main()