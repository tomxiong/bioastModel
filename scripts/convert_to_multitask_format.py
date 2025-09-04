#!/usr/bin/env python3
"""
将现有数据格式转换为多任务标注格式
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import logging
from collections import defaultdict


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def analyze_existing_structure(data_root: str) -> Dict:
    """分析现有数据结构"""
    data_root = Path(data_root)
    structure = {
        'splits': ['train', 'val', 'test'],
        'classes': ['negative', 'positive'],
        'samples': defaultdict(lambda: defaultdict(list))
    }
    
    for split in structure['splits']:
        for class_name in structure['classes']:
            class_dir = data_root / split / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*.png'):
                    structure['samples'][split][class_name].append(str(img_file))
    
    return structure


def infer_annotations(file_path: str, class_name: str) -> Dict:
    """根据文件路径和类别推断多任务标注"""
    # 这里使用简单的启发式规则，实际应用中可能需要更复杂的逻辑
    # 或手动标注
    
    annotations = {
        'growth_level': class_name if class_name in ['negative', 'positive'] else 'positive',
        'growth_pattern': 'clean' if class_name == 'negative' else 'clustered',
        'interference_mapping': ['pores'],  # 默认假设有气孔干扰
        'fine_grained': f'{class_name}_clean'
    }
    
    # 根据文件名推断更多信息
    filename = Path(file_path).stem.lower()
    
    if 'weak' in filename or 'small' in filename:
        annotations['growth_level'] = 'weak_growth'
        annotations['growth_pattern'] = 'small_dots'
        annotations['fine_grained'] = 'weak_growth_small_dots_pores'
    
    if 'scatter' in filename:
        annotations['growth_pattern'] = 'scattered'
        annotations['fine_grained'] = f'{class_name}_scattered_pores'
    
    if 'heavy' in filename or 'dense' in filename:
        annotations['growth_pattern'] = 'heavy_growth'
        annotations['fine_grained'] = 'heavy_growth_pores'
    
    return annotations


def convert_dataset(data_root: str, output_dir: str, split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
    """转换数据集格式"""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # 创建输出目录结构
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # 分析现有结构
    structure = analyze_existing_structure(data_root)
    
    # 收集所有样本
    all_samples = []
    for split in structure['splits']:
        for class_name in structure['classes']:
            for file_path in structure['samples'][split][class_name]:
                all_samples.append({
                    'file_path': file_path,
                    'original_split': split,
                    'class': class_name
                })
    
    # 如果需要重新分割
    if split_ratio != (0, 0, 0):
        import random
        random.shuffle(all_samples)
        
        total = len(all_samples)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])
        
        splits = {
            'train': all_samples[:train_end],
            'val': all_samples[train_end:val_end],
            'test': all_samples[val_end:]
        }
    else:
        # 保持原有分割
        splits = {}
        for split in structure['splits']:
            splits[split] = [s for s in all_samples if s['original_split'] == split]
    
    # 转换并复制图像
    image_id_counter = 0
    annotations = []
    
    for split_name, samples in splits.items():
        print(f"处理 {split_name} 分割: {len(samples)} 个样本")
        
        for sample in samples:
            # 生成新的文件路径
            image_id = f"image_{image_id_counter:06d}"
            image_id_counter += 1
            
            # 复制图像文件
            src_path = Path(sample['file_path'])
            dst_filename = f"{image_id}.png"
            dst_path = output_dir / 'images' / dst_filename
            
            shutil.copy2(src_path, dst_path)
            
            # 创建标注
            annotation = {
                'image_id': image_id,
                'file_path': f"images/{dst_filename}",
                'original_file': str(src_path.relative_to(data_root)),
                'split': split_name,
                'annotations': infer_annotations(sample['file_path'], sample['class'])
            }
            
            annotations.append(annotation)
    
    # 保存标注文件
    output_file = output_dir / 'annotations' / 'multitask_annotations.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成!")
    print(f"总样本数: {len(annotations)}")
    print(f"标注文件: {output_file}")
    
    # 生成统计报告
    generate_conversion_report(annotations, output_dir / 'conversion_report.txt')
    
    return output_file


def generate_conversion_report(annotations: List[Dict], output_path: str):
    """生成转换报告"""
    report = []
    report.append("=== 数据集转换报告 ===\n")
    
    # 基本统计
    report.append(f"总样本数: {len(annotations)}")
    
    # 分割统计
    split_counts = defaultdict(int)
    for ann in annotations:
        split_counts[ann['split']] += 1
    
    report.append("\n分割分布:")
    for split, count in split_counts.items():
        percentage = count / len(annotations) * 100
        report.append(f"  {split}: {count} ({percentage:.1f}%)")
    
    # 标注统计
    report.append("\n标注统计:")
    
    # 生长级别
    gl_counts = defaultdict(int)
    # 生长模式
    gp_counts = defaultdict(int)
    # 干扰因素
    interference_counts = defaultdict(int)
    # 多标签统计
    multilabel_count = 0
    
    for ann in annotations:
        ann_data = ann['annotations']
        
        gl_counts[ann_data['growth_level']] += 1
        gp_counts[ann_data['growth_pattern']] += 1
        
        interference_list = ann_data['interference_mapping']
        if len(interference_list) > 1:
            multilabel_count += 1
        
        for interference in interference_list:
            interference_counts[interference] += 1
    
    report.append("\n生长级别分布:")
    for gl, count in sorted(gl_counts.items()):
        percentage = count / len(annotations) * 100
        report.append(f"  {gl}: {count} ({percentage:.1f}%)")
    
    report.append("\n生长模式分布:")
    for gp, count in sorted(gp_counts.items()):
        percentage = count / len(annotations) * 100
        report.append(f"  {gp}: {count} ({percentage:.1f}%)")
    
    report.append("\n干扰因素分布:")
    for interference, count in sorted(interference_counts.items()):
        percentage = count / len(annotations) * 100
        report.append(f"  {interference}: {count} ({percentage:.1f}%)")
    
    report.append(f"\n多标签样本数: {multilabel_count}")
    report.append(f"多标签比例: {multilabel_count / len(annotations) * 100:.1f}%")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"转换报告已保存: {output_path}")


def validate_conversion(annotation_file: str):
    """验证转换结果"""
    print("\n=== 验证转换结果 ===")
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 检查必需字段
    required_fields = ['image_id', 'file_path', 'split', 'annotations']
    missing_fields = []
    
    for i, ann in enumerate(annotations):
        for field in required_fields:
            if field not in ann:
                missing_fields.append(f"样本 {i}: 缺少 {field}")
    
    if missing_fields:
        print("发现以下问题:")
        for issue in missing_fields[:10]:  # 只显示前10个
            print(f"  - {issue}")
        if len(missing_fields) > 10:
            print(f"  ... 还有 {len(missing_fields) - 10} 个问题")
    else:
        print("✓ 所有样本都包含必需字段")
    
    # 检查标注完整性
    required_tasks = ['growth_level', 'growth_pattern', 'interference_mapping']
    task_issues = []
    
    for i, ann in enumerate(annotations):
        ann_data = ann['annotations']
        for task in required_tasks:
            if task not in ann_data:
                task_issues.append(f"样本 {i}: 缺少任务 {task}")
    
    if task_issues:
        print("\n标注任务问题:")
        for issue in task_issues[:10]:
            print(f"  - {issue}")
    else:
        print("✓ 所有样本都包含必需的标注任务")
    
    # 检查图像文件
    missing_images = []
    annotation_dir = Path(annotation_file).parent
    image_root = annotation_dir.parent / 'images'
    
    for ann in annotations[:100]:  # 只检查前100个
        image_path = image_root / ann['file_path'].split('/')[-1]
        if not image_path.exists():
            missing_images.append(ann['image_id'])
    
    if missing_images:
        print(f"\n缺失图像文件: {len(missing_images)} 个")
    else:
        print("✓ 检查的图像文件都存在")


def main():
    parser = argparse.ArgumentParser(description='转换为多任务标注格式')
    parser.add_argument('--data_root', type=str, required=True,
                       help='现有数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                       metavar=('TRAIN', 'VAL', 'TEST'),
                       help='训练/验证/测试分割比例 (默认: 0.7 0.15 0.15)')
    parser.add_argument('--validate', action='store_true',
                       help='验证转换结果')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # 执行转换
    annotation_file = convert_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        split_ratio=tuple(args.split_ratio)
    )
    
    # 验证结果
    if args.validate:
        validate_conversion(annotation_file)


if __name__ == "__main__":
    main()