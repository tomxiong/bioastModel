import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_dataset():
    """分析数据集的基本特征"""
    dataset_path = "./bioast_dataset"
    
    # 统计信息
    stats = {
        'positive': {'train': 0, 'val': 0, 'test': 0},
        'negative': {'train': 0, 'val': 0, 'test': 0}
    }
    
    # 图像特征统计
    image_sizes = []
    pixel_values = []
    
    for class_name in ['positive', 'negative']:
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(dataset_path, class_name, split)
            if os.path.exists(split_path):
                files = [f for f in os.listdir(split_path) if f.endswith('.png')]
                stats[class_name][split] = len(files)
                
                # 分析前几张图像的特征
                for i, filename in enumerate(files[:3]):
                    img_path = os.path.join(split_path, filename)
                    try:
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        
                        image_sizes.append(img_array.shape)
                        pixel_values.extend(img_array.flatten())
                        
                        if i == 0:  # 打印第一张图像的详细信息
                            print(f"\n{class_name}/{split} 样例图像 {filename}:")
                            print(f"  尺寸: {img_array.shape}")
                            print(f"  数据类型: {img_array.dtype}")
                            print(f"  像素值范围: {img_array.min()} - {img_array.max()}")
                            print(f"  均值: {img_array.mean():.2f}")
                            print(f"  标准差: {img_array.std():.2f}")
                    except Exception as e:
                        print(f"读取图像 {img_path} 失败: {e}")
    
    # 打印统计信息
    print("\n数据集统计:")
    print("=" * 50)
    for class_name in ['positive', 'negative']:
        print(f"{class_name.upper()}:")
        for split in ['train', 'val', 'test']:
            count = stats[class_name][split]
            print(f"  {split}: {count}")
    
    # 图像特征汇总
    if image_sizes:
        unique_sizes = list(set(image_sizes))
        print(f"\n图像尺寸: {unique_sizes}")
        
    if pixel_values:
        pixel_values = np.array(pixel_values[:10000])  # 采样分析
        print(f"整体像素值统计:")
        print(f"  范围: {pixel_values.min()} - {pixel_values.max()}")
        print(f"  均值: {pixel_values.mean():.2f}")
        print(f"  标准差: {pixel_values.std():.2f}")

if __name__ == "__main__":
    analyze_dataset()