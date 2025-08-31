#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物分析数据集构建工具
从D:\image_analysis目录解析cfg文件并构建阴阳性分类数据集
"""

import os
import shutil
import glob
import random
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

class BioDatasetBuilder:
    def __init__(self, source_dir="D:\\bioast_images", output_dir="bioast_dataset"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.stats = defaultdict(int)
        self.samples_data = []
        
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
            # 根据用户描述：药敏判别从hole_25开始到hole_96
            labels = []
            for i, char in enumerate(labels_str):
                hole_num = i + 1  # cfg索引i对应hole_{i+1}.png
                # 只处理药敏判别相关的图片文件（hole_25到hole_120）
                if 25 <= hole_num <= 120:
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
    
    def scan_source_data(self):
        """扫描源目录获取所有样本数据"""
        print("扫描源数据...")
        
        cfg_files = list(self.source_dir.glob("*.cfg"))
        print(f"发现 {len(cfg_files)} 个cfg文件")
        
        for cfg_file in cfg_files:
            cfg_data = self.parse_cfg_file(cfg_file)
            if cfg_data:
                # 检查对应的图片目录是否存在
                sample_dir = self.source_dir / cfg_data['sample_name']
                if sample_dir.exists():
                    cfg_data['sample_dir'] = sample_dir
                    self.samples_data.append(cfg_data)
                    print(f"处理样本: {cfg_data['sample_name']}, 孔位数: {len(cfg_data['labels'])}")
                else:
                    print(f"警告: 未找到对应图片目录: {sample_dir}")
        
        print(f"成功加载 {len(self.samples_data)} 个样本")
        return len(self.samples_data) > 0
    
    def collect_all_images(self):
        """收集所有图片及其标签"""
        print("收集图片数据...")
        
        positive_images = []
        negative_images = []
        
        for sample in self.samples_data:
            sample_dir = sample['sample_dir']
            
            # 处理每个孔位的图像（从hole_25到hole_120，用于药敏判别）
            for hole_num, label in sample['labels']:
                # 确保只处理hole_25到hole_120的图像
                if 25 <= hole_num <= 120:
                    hole_file = sample_dir / f"hole_{hole_num}.png"
                    
                    if hole_file.exists():
                        image_info = {
                            'path': str(hole_file),
                            'sample': sample['sample_name'],
                            'hole': hole_num,
                            'label': label
                        }
                        
                        if label == 'positive':
                            positive_images.append(image_info)
                        else:
                            negative_images.append(image_info)
                        
                        self.stats[f'{label}_total'] += 1
                    else:
                        print(f"警告: 未找到图片文件: {hole_file}")
        
        print(f"阳性样本: {len(positive_images)}")
        print(f"阴性样本: {len(negative_images)}")
        
        return positive_images, negative_images
    
    def split_dataset(self, images, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
        """按比例划分数据集"""
        assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, "比例之和必须为1"
        
        # 随机打乱
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * train_ratio)
        test_end = train_end + int(total * test_ratio)
        
        train_set = images[:train_end]
        test_set = images[train_end:test_end]
        val_set = images[test_end:]
        
        return train_set, test_set, val_set
    
    def copy_images_to_dataset(self, images, dest_subdir):
        """复制图片到目标目录"""
        dest_path = self.output_dir / dest_subdir
        
        # 确保目标目录存在
        dest_path.mkdir(parents=True, exist_ok=True)
        
        for img_info in images:
            # 生成唯一文件名：样本名_孔位号.png
            new_filename = f"{img_info['sample']}_hole_{img_info['hole']}.png"
            dest_file = dest_path / new_filename
            
            try:
                shutil.copy2(img_info['path'], dest_file)
            except Exception as e:
                print(f"错误: 复制文件失败 {img_info['path']} -> {dest_file}: {e}")
        
        print(f"复制了 {len(images)} 张图片到 {dest_subdir}")
        return len(images)
    
    def build_dataset(self):
        """构建完整数据集"""
        print("开始构建数据集...")
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 扫描源数据
        if not self.scan_source_data():
            print("错误: 未找到有效的源数据")
            return False
        
        # 收集所有图片
        positive_images, negative_images = self.collect_all_images()
        
        if not positive_images and not negative_images:
            print("错误: 未找到任何有效图片")
            return False
        
        # 为每类数据分别划分数据集
        pos_train, pos_test, pos_val = self.split_dataset(positive_images)
        neg_train, neg_test, neg_val = self.split_dataset(negative_images)
        
        # 复制图片到对应目录
        datasets = [
            (pos_train, "positive/train"),
            (pos_test, "positive/test"), 
            (pos_val, "positive/val"),
            (neg_train, "negative/train"),
            (neg_test, "negative/test"),
            (neg_val, "negative/val")
        ]
        
        for images, subdir in datasets:
            count = self.copy_images_to_dataset(images, subdir)
            self.stats[subdir] = count
        
        # 生成统计报告
        self.generate_report()
        
        print("数据集构建完成！")
        return True
    
    def generate_report(self):
        """生成数据集统计报告(JSON格式)"""
        report_path = self.output_dir / "dataset_stats.json"
        
        # 准备报告数据
        report_data = {
            "metadata": {
                "source_directory": str(self.source_dir),
                "output_directory": str(self.output_dir),
                "sample_count": len(self.samples_data),
                "generation_time": datetime.now().isoformat(),
                "tool_version": "1.0"
            },
            "summary": {
                "positive_total": self.stats.get('positive_total', 0),
                "negative_total": self.stats.get('negative_total', 0),
                "total_samples": self.stats.get('positive_total', 0) + self.stats.get('negative_total', 0)
            },
            "dataset_splits": {}
        }
        
        # 添加各数据集统计信息
        for subset in ['train', 'test', 'val']:
            pos_count = self.stats.get(f'positive/{subset}', 0)
            neg_count = self.stats.get(f'negative/{subset}', 0)
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
        
        print(f"统计报告已保存到: {report_path}")

def main():
    """主函数"""
    print("生物分析数据集构建工具")
    print("-" * 30)
    
    # 设置随机种子以便复现
    random.seed(42)
    
    # 创建构建器并执行
    builder = BioDatasetBuilder()
    
    if builder.build_dataset():
        print("\n✅ 数据集构建成功！")
        print(f"📁 输出目录: {builder.output_dir.absolute()}")
        print("📊 请查看 dataset_stats.json 获取详细统计信息")
    else:
        print("\n❌ 数据集构建失败")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())